from typing import Any, Literal

import torch
from pydantic import Field
from torch import Tensor

from ltx_core.model.transformer.modality import Modality
from ltx_core.types import VideoLatentShape
from ltx_trainer import logger
from ltx_trainer.timestep_samplers import TimestepSampler
from ltx_trainer.training_strategies.base_strategy import (
    DEFAULT_FPS,
    VIDEO_SCALE_FACTORS,
    ModelInputs,
    TrainingStrategy,
    TrainingStrategyConfigBase,
)
import random


class TextToVideoEncoreConfig(TrainingStrategyConfigBase):
    """Configuration for the Encore text-to-video training strategy."""

    name: Literal["text_to_video_encore"] = "text_to_video_encore"

    first_frame_conditioning_p: float = Field(
        default=0.1,
        description="Probability of conditioning on the first frame during training",
        ge=0.0,
        le=1.0,
    )

    with_audio: bool = Field(
        default=False,
        description="Whether to include audio in training (joint audio-video generation)",
    )

    audio_latents_dir: str = Field(
        default="audio_latents",
        description="Directory name for audio latents when with_audio is True",
    )

    with_reference_image: bool = Field(
        default=True,
        description="Whether to include reference image in training",
    )

    with_reference_audio: bool = Field(
        default=True,
        description="Whether to include reference audio in training. Only used if with_audio is True.",
    )

    drop_ref_audio_p: float = Field(
        default=0.1,
        description="Probability of dropping (zeroing out) the reference audio during training for classifier-free guidance. Only used if with_reference_audio is True.",
        ge=0.0,
        le=1.0,
    )

    error_injection_p: float = Field(
        default=0.8,
        description="Probability of injecting past prediction errors into the first frame latent.",
    )

    clean_prob: float = Field(
        default=0.2,
        description="Probability of keeping the input 'clean' (no error injection).",
    )

    clean_buffer_update_prob: float = Field(
        default=0.1,
        description="Probability of updating the error buffer, ONLY if the current step is 'clean'.",
    )

    error_buffer_size: int = Field(
        default=25000,
        description="Maximum number of error samples to store in the error buffer.",
    )

    num_conditioning_frames: int = Field(
        default=2,
        description=(
            "Number of LATENT frames to condition on (keep clean, no loss). "
            "When with_reference_image=True, this includes the reference image. "
            "Due to causal VAE: 1st latent = 1 pixel frame, each subsequent = 8 pixel frames. "
            "E.g. 2 = ref_image(1px) + 1 latent(8px) = 9 pixel frames. "
            "4 = ref_image(1px) + 3 latents(8px each) = 25 pixel frames."
        ),
        ge=1,
    )

    ref_audio_num_frames: int = Field(
        default=1,
        description=(
            "Length of reference audio measured in video latent frame durations. "
            "Each latent frame = 8 pixel frames = 8/fps seconds. "
            "E.g. 1 = 8/24 ~= 0.33s at 24fps (default), 3 = 24/24 = 1s."
        ),
        ge=1,
    )

    ref_position_offset: float = Field(
        default=0.0,
        description=(
            "Extra time offset (in seconds) to push reference image and reference audio "
            "positions further into negative time, creating a gap between the reference "
            "and the first video/audio frame. This prevents the model from treating the "
            "reference as temporally adjacent to the generated content. "
            "E.g. 1.0 means ref_image ends at t=-1.0s instead of t=0."
        ),
        ge=0.0,
    )

    replace_ref_with_first_frame_p: float = Field(
        default=0.0,
        description=(
            "Probability of replacing the reference image with the first frame of the video latents. "
            "This acts as a data augmentation that teaches the model to handle cases where "
            "the reference image is identical to the first frame."
        ),
        ge=0.0,
        le=1.0,
    )

    enable_condition_routing: bool = Field(
        default=False,
        description=(
            "Enable unified condition routing table for layer-adaptive conditioning. "
            "Adds a learnable R[L,H,4] routing table that provides per-layer per-head "
            "independent control over Anchor/Continuation/Semantic/Sync condition strengths."
        ),
    )

    cont_position_decay_lambda: float = Field(
        default=0.3,
        description=(
            "Decay rate for continuation condition's spatial influence. "
            "Controls how fast the continuation routing bias decays with distance from "
            "conditioning frames: decay = exp(-distance / (lambda * gen_length)). "
            "Smaller values = faster decay = continuation affects fewer tokens."
        ),
        ge=0.01,
        le=1.0,
    )
    
class TextToVideoEncoreStrategy(TrainingStrategy):
    """Encore text-to-video training strategy.
    This strategy implements regular video generation training where:
    - Target latents and reference image latent are used
    - Standard noise application and loss computation
    - Joint audio-video training
    """

    config: TextToVideoEncoreConfig

    def __init__(self, config: TextToVideoEncoreConfig):
        """Initialize strategy with configuration.
        Args:
            config: Encore text-to-video configuration
        """
        super().__init__(config)

        self.error_buffer: dict[tuple[int, int], list[Tensor]] = {}
        self._debug_step = 0
        self._DEBUG_LOG_STEPS = 3  # Print debug info for the first N steps

    @property
    def requires_audio(self) -> bool:
        """Whether this training strategy requires audio components."""
        return self.config.with_audio

    @property
    def requires_reference_image(self) -> bool:
        """Whether this training strategy requires reference image."""
        return self.config.with_reference_image

    @property
    def requires_reference_audio(self) -> bool:
        """Whether this training strategy requires reference audio."""
        return self.config.with_audio and self.config.with_reference_audio

    def get_data_sources(self) -> list[str] | dict[str, str]:
        """
        Text-to-video training requires latents and text conditions.
        When with_audio is True, also requires audio latents.
        """
        sources = {
            "latents": "latents",
            "conditions": "conditions",
        }

        if self.config.with_audio:
            sources[self.config.audio_latents_dir] = "audio_latents"

        return sources

    def prepare_training_inputs(
        self,
        batch: dict[str, Any],
        timestep_sampler: TimestepSampler,
    ) -> ModelInputs:
        """Prepare inputs for text-to-video training."""
        # Get pre-encoded latents - dataset provides uniform non-patchified format [B, C, F, H, W]
        latents = batch["latents"]
        video_latents = latents["latents"]
        video_latents = video_latents.clone()

        # Get video dimensions (assume same for all batch elements)
        num_frames = latents["num_frames"][0].item()
        height = latents["height"][0].item()
        width = latents["width"][0].item()

        # Patchify latents: [B, C, F, H, W] -> [B, seq_len, C]
        # Save original video frame count before prepending ref image (needed for position encoding)
        original_num_frames = num_frames
        if self.config.with_reference_image:
            reference_image = latents["ref_image_latent"]
            # With a certain probability, replace the reference image with the first frame of the video
            if self.config.replace_ref_with_first_frame_p > 0 and random.random() < self.config.replace_ref_with_first_frame_p:
                reference_image = video_latents[:, :, 0:1, :, :].clone()
            # Since the patchifier's frame is 1, we can directly concatenate the reference image to the video latents
            video_latents = torch.cat([reference_image, video_latents], dim=2)
            num_frames += 1

        # --- Error Injection ---
        is_clean_step = random.random() < self.config.clean_prob
        if not is_clean_step and self.config.error_injection_p > 0 and random.random() < self.config.error_injection_p:
            # If NOT clean and error should be injected
            h, w = video_latents.shape[-2], video_latents.shape[-1]
            bucket_key = (h, w)
            
            if bucket_key in self.error_buffer:
                current_bs = video_latents.shape[0]
                
                # Randomly pick one batch tensor from buffer: [B_stored, C, F, H, W]
                stored_batch = random.choice(self.error_buffer[bucket_key])
                stored_bs, _, stored_f, _, _ = stored_batch.shape
                
                # Randomly sample indices for Batch and Frame
                batch_indices = torch.randint(0, stored_bs, (current_bs,))
                frame_indices = torch.randint(0, stored_f, (current_bs,))
                
                # Extract the frames: [B_curr, C, H, W]
                device = video_latents.device
                dtype = video_latents.dtype                
                selected_error_frames = stored_batch[batch_indices, :, frame_indices, :, :].to(device=device, dtype=dtype)
                
                # Add to the FIRST frame of current video latents
                # video_latents: [B, C, F, H, W]
                video_latents[:, :, 1, :, :] += selected_error_frames
        # -----------------------------

        video_latents = self._video_patchifier.patchify(video_latents)
        
        # Handle FPS with backward compatibility
        fps = latents.get("fps", None)
        if fps is not None and not torch.all(fps == fps[0]):
            logger.warning(
                f"Different FPS values found in the batch. Found: {fps.tolist()}, using the first one: {fps[0].item()}"
            )
        fps = fps[0].item() if fps is not None else DEFAULT_FPS

        # Get text embeddings (already processed by embedding connectors in trainer)
        conditions = batch["conditions"]
        video_prompt_embeds = conditions["video_prompt_embeds"]
        audio_prompt_embeds = conditions["audio_prompt_embeds"]
        prompt_attention_mask = conditions["prompt_attention_mask"]

        batch_size = video_latents.shape[0]
        video_seq_len = video_latents.shape[1]
        device = video_latents.device
        dtype = video_latents.dtype

        # Determine conditioning once — shared by both video and audio
        apply_conditioning = self.config.first_frame_conditioning_p > 0 and random.random() < self.config.first_frame_conditioning_p

        # Create conditioning mask for context frames
        video_conditioning_mask = self._create_conditioning_mask(
            batch_size=batch_size,
            sequence_length=video_seq_len,
            height=height,
            width=width,
            num_frames=self.config.num_conditioning_frames,
            apply_conditioning=apply_conditioning,
            device=device,
        )

        # Sample noise and sigmas
        sigmas = timestep_sampler.sample_for(video_latents)
        video_noise = torch.randn_like(video_latents)

        # Apply noise: noisy = (1 - sigma) * clean + sigma * noise
        sigmas_expanded = sigmas.view(-1, 1, 1)
        noisy_video = (1 - sigmas_expanded) * video_latents + sigmas_expanded * video_noise

        # For conditioning tokens, use clean latents
        conditioning_mask_expanded = video_conditioning_mask.unsqueeze(-1)
        noisy_video = torch.where(conditioning_mask_expanded, video_latents, noisy_video)

        # Compute video targets (velocity prediction)
        video_targets = video_noise - video_latents

        # Create per-token timesteps
        video_timesteps = self._create_per_token_timesteps(video_conditioning_mask, sigmas.squeeze())

        # Generate video positions
        if self.config.with_reference_image:
            # ref_image and video_latents are encoded SEPARATELY, each with its own
            # causal VAE. So both have a causal first frame (1 pixel). We must generate
            # positions independently and concatenate, not treat them as one sequence.
            #
            # ref_image: 1 frame (causal first = 1px) → pixel [0, 1]
            # video:     N frames (own causal: frame0=1px, frame1+=8px)
            #            → pixel [0,1], [1,9], [9,17], ...
            # Offset video by ref's pixel extent (1 pixel) then concatenate:
            # Combined:  ref=[0,1], vid[0]=[1,2], vid[1]=[2,10], vid[2]=[10,18], ...
            # Shift all by -1/fps so ref is at negative time:
            #            ref=[-1/fps, 0], vid[0]=[0, 1/fps], vid[1]=[1/fps, 9/fps], ...

            ref_positions = self._get_video_positions(
                num_frames=1,
                height=height,
                width=width,
                batch_size=batch_size,
                fps=fps,
                device=device,
                dtype=dtype,
            )

            vid_positions = self._get_video_positions(
                num_frames=original_num_frames,
                height=height,
                width=width,
                batch_size=batch_size,
                fps=fps,
                device=device,
                dtype=dtype,
            )

            # Offset video temporal positions by ref image's pixel extent (1 pixel = 1/fps sec)
            vid_positions[:, 0, :, :] += (1.0 / fps)

            # Shift ref_positions further into negative time to isolate ref from video.
            # This creates a temporal gap of ref_position_offset seconds between ref and vid[0].
            if self.config.ref_position_offset > 0:
                ref_positions[:, 0, :, :] -= self.config.ref_position_offset

            # Concatenate along seq_len dimension (dim=2)
            video_positions = torch.cat([ref_positions, vid_positions], dim=2)

            # Shift entire sequence so ref image is at negative time
            video_positions[:, 0, :, :] -= (1.0 / fps)
        else:
            raise Exception("with_reference_image must be True")

        # Build attention_mask and routing metadata when condition routing is enabled
        routing_metadata = None
        video_attention_mask = None
        if self.config.enable_condition_routing:
            # Compact all-ones attention mask [B, 1, T] → converted to additive
            # log-space [B, 1, 1, T] by _prepare_self_attention_mask.
            # Row-uniform (same for every query position) to avoid allocating
            # the full T×T matrix.
            video_attention_mask = torch.ones(
                batch_size, 1, video_seq_len, device=device, dtype=dtype,
            )

            # Compute token counts for A/B routing
            tokens_per_frame = height * width
            if self.config.with_reference_image and apply_conditioning:
                ref_token_count = tokens_per_frame  # 1 ref frame
                cond_token_count = tokens_per_frame * (self.config.num_conditioning_frames - 1)
            else:
                ref_token_count = 0
                cond_token_count = 0

            # Build log-space position decay for continuation routing
            cont_pos_decay = None
            if cond_token_count > 0:
                gen_start = ref_token_count + cond_token_count
                gen_len = video_seq_len - gen_start
                if gen_len > 0:
                    lam = self.config.cont_position_decay_lambda
                    # Normalized distance: u[i] = i / gen_len, i = 0..gen_len-1
                    u = torch.arange(gen_len, device=device, dtype=dtype) / gen_len
                    # Log-space decay: log(exp(-u / λ)) = -u / λ
                    cont_pos_decay = -u / lam

            routing_metadata = {
                "ref_token_count": ref_token_count,
                "cond_token_count": cond_token_count,
                "cont_pos_decay": cont_pos_decay,
            }

        # Create video Modality
        video_modality = Modality(
            enabled=True,
            sigma=sigmas,
            latent=noisy_video,
            timesteps=video_timesteps,
            positions=video_positions,
            context=video_prompt_embeds,
            context_mask=prompt_attention_mask,
            attention_mask=video_attention_mask,
        )

        # Video loss mask: True for tokens we want to compute loss on (non-conditioning tokens)
        video_loss_mask = ~video_conditioning_mask

        # Handle audio if enabled
        audio_modality = None
        audio_targets = None
        audio_loss_mask = None

        if self.config.with_audio:
            audio_modality, audio_targets, audio_loss_mask = self._prepare_audio_inputs(
                batch=batch,
                sigmas=sigmas,
                audio_prompt_embeds=audio_prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
                fps=fps,
                apply_conditioning=apply_conditioning,
            )

        inputs = ModelInputs(
            video=video_modality,
            audio=audio_modality,
            video_targets=video_targets,
            audio_targets=audio_targets,
            video_loss_mask=video_loss_mask,
            audio_loss_mask=audio_loss_mask,
            routing_metadata=routing_metadata,
        )

        # Save metadata for compute_loss
        inputs.is_clean_step = is_clean_step
        inputs.orig_height = height
        inputs.orig_width = width
        inputs.orig_num_frames = num_frames

        # --- Debug logging (first N steps only) ---
        if self._debug_step < self._DEBUG_LOG_STEPS:
            self._debug_step += 1
            self._log_alignment_debug(
                step=self._debug_step,
                fps=fps,
                num_frames=num_frames,
                height=height,
                width=width,
                apply_conditioning=apply_conditioning,
                video_positions=video_positions,
                video_conditioning_mask=video_conditioning_mask,
                video_timesteps=video_timesteps,
                audio_modality=audio_modality,
                audio_loss_mask=audio_loss_mask,
            )

        return inputs

    def _prepare_audio_inputs(
        self,
        batch: dict[str, Any],
        sigmas: Tensor,
        audio_prompt_embeds: Tensor,
        prompt_attention_mask: Tensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        fps: float = DEFAULT_FPS,
        apply_conditioning: bool = False,
    ) -> tuple[Modality, Tensor, Tensor]:
        """Prepare audio inputs for joint audio-video training.
        Args:
            batch: Raw batch data containing audio_latents
            sigmas: Sampled sigma values (same as video)
            audio_prompt_embeds: Audio context embeddings
            prompt_attention_mask: Attention mask for context
            batch_size: Batch size
            device: Target device
            dtype: Target dtype
            fps: Video frames per second
            apply_conditioning: Whether conditioning is active (shared with video)
        Returns:
            Tuple of (audio_modality, audio_targets, audio_loss_mask)
        """
        # Get audio latents - dataset provides uniform non-patchified format [B, C, T, F]
        audio_data = batch["audio_latents"]
        audio_latents = audio_data["latents"]

        # Patchify audio latents: [B, C, T, F] -> [B, T, C*F]
        audio_latents = self._audio_patchifier.patchify(audio_latents)

        # Original Audio Sequence Length (before adding reference)
        orig_audio_seq_len = audio_latents.shape[1]

        # Calculate audio tokens per video frame from fundamental parameters:
        # video_frame_duration = VIDEO_SCALE_FACTORS.time / fps  (seconds per video latent frame)
        # audio_token_duration = downsample_factor * hop_length / sample_rate  (seconds per audio token)
        ap = self._audio_patchifier
        video_frame_duration = VIDEO_SCALE_FACTORS.time / fps
        audio_token_duration = (ap.audio_latent_downsample_factor * ap.hop_length) / ap.sample_rate
        tokens_per_frame = max(1, round(video_frame_duration / audio_token_duration))

        ref_audio_len = tokens_per_frame * self.config.ref_audio_num_frames

        if self.config.with_reference_audio:
            # Use precomputed reference audio latents from data (non-overlapping with main audio)
            ref_audio_latent = audio_data["ref_audio_latent"]  # [B, C, T, F]
            reference_audio = self._audio_patchifier.patchify(ref_audio_latent)  # [B, T, C*F]
            ref_t = reference_audio.shape[1]

            if ref_t >= ref_audio_len:
                reference_audio = reference_audio[:, :ref_audio_len, :]
            else:
                # Pad with zeros if precomputed ref audio is shorter than expected
                padding = torch.zeros(
                    (batch_size, ref_audio_len - ref_t, reference_audio.shape[2]),
                    device=device, dtype=dtype,
                )
                reference_audio = torch.cat([reference_audio, padding], dim=1)

            # Prepend reference audio to audio latents
            audio_latents = torch.cat([reference_audio, audio_latents], dim=1)

            # Drop reference audio with probability drop_ref_audio_p (replace with zeros for CFG)
            if self.config.drop_ref_audio_p > 0 and random.random() < self.config.drop_ref_audio_p:
                audio_latents[:, :ref_audio_len, :] = 0.0

        audio_seq_len = audio_latents.shape[1]

        # --- Audio Masking Logic ---
        # We want to mask (condition on):
        # 1. Reference Audio (if present)
        # 2. First Frame Audio (corresponding to Video First Frame)

        audio_conditioning_mask = torch.zeros(batch_size, audio_seq_len, dtype=torch.bool, device=device)

        current_idx = 0
        if self.config.with_reference_audio:
            # Mask the reference part
            audio_conditioning_mask[:, :ref_audio_len] = True
            current_idx += ref_audio_len

        # Mask the audio context frames (corresponding to video conditioning frames)
        if self.config.with_reference_image:
            # ref_image occupies latent frame 0 (1px, no corresponding audio).
            # The remaining conditioning video latent frames have their OWN causal structure
            # (separately encoded from ref image): 1st video latent = 1px, each subsequent = 8px.
            num_audio_context_latent_frames = self.config.num_conditioning_frames - 1
        else:
            raise ValueError("with_reference_image must be True")

        # TODO: Currently hardcoded for num_conditioning_frames=2 (1 video context latent frame).
        # Audio VAE is causal: token0 covers ~0.01s, token1+ covers ~0.04s each.
        # 1 video latent = 1 pixel frame = 1/fps ≈ 0.042s → need 2 audio tokens (covers 0.05s).
        # When num_conditioning_frames > 2, use _get_audio_latent_time_in_sec to compute exactly.
        assert num_audio_context_latent_frames <= 1, (
            f"audio_context_len is hardcoded for num_conditioning_frames=2, "
            f"got num_conditioning_frames={self.config.num_conditioning_frames}"
        )
        audio_context_len = 2 if num_audio_context_latent_frames == 1 else 0
        # Sanity check: verify hardcoded value matches actual patchifier params.
        # 2 tokens must cover >= 1 pixel frame / fps, and 1 token must not.
        if audio_context_len == 2:
            t2 = ap._get_audio_latent_time_in_sec(0, 3, dtype=torch.float32, device=device)[-1].item()
            t1 = ap._get_audio_latent_time_in_sec(0, 2, dtype=torch.float32, device=device)[-1].item()
            target = 1.0 / fps
            assert t1 < target <= t2, (
                f"Hardcoded audio_context_len=2 is wrong for current AudioPatchifier params: "
                f"1 token covers {t1:.4f}s, 2 tokens cover {t2:.4f}s, need >= {target:.4f}s"
            )

        if apply_conditioning:
            if current_idx + audio_context_len <= audio_seq_len:
                audio_conditioning_mask[:, :current_idx + audio_context_len] = True

        # Sample audio noise
        audio_noise = torch.randn_like(audio_latents)

        # Apply noise to audio
        sigmas_expanded = sigmas.view(-1, 1, 1)
        noisy_audio = (1 - sigmas_expanded) * audio_latents + sigmas_expanded * audio_noise

        # Apply conditioning (overwrite noisy with clean for masked tokens)
        mask_expanded = audio_conditioning_mask.unsqueeze(-1)
        noisy_audio = torch.where(mask_expanded, audio_latents, noisy_audio)

        # Compute audio targets
        audio_targets = audio_noise - audio_latents

        # Conditioning tokens get sigma=0, others get sampled sigma
        audio_timesteps = torch.where(audio_conditioning_mask, torch.zeros_like(sigmas_expanded.squeeze(-1)), sigmas_expanded.squeeze(-1))

        # Generate audio positions
        audio_positions = self._get_audio_positions(
            num_time_steps=audio_seq_len,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

        if self.config.with_reference_audio:
            # Shift ENTIRE audio sequence back so that:
            # - Ref Audio (indices 0..ref_audio_len-1): t < 0
            # - Main Audio (index ref_audio_len): t = 0
            # Convert ref_audio_len (token count) to seconds to match position units
            ref_audio_time_in_sec = self._audio_patchifier._get_audio_latent_time_in_sec(
                ref_audio_len, ref_audio_len + 1, dtype=audio_positions.dtype, device=audio_positions.device
            ).item()
            audio_positions[:, 0, :, :] -= ref_audio_time_in_sec
            # Additionally shift only ref audio tokens further into negative time
            if self.config.ref_position_offset > 0:
                audio_positions[:, 0, :ref_audio_len, :] -= self.config.ref_position_offset

        # Create audio Modality
        audio_modality = Modality(
            enabled=True,
            sigma=sigmas,
            latent=noisy_audio,
            timesteps=audio_timesteps,
            positions=audio_positions,
            context=audio_prompt_embeds,
            context_mask=prompt_attention_mask,
        )

        # Audio loss mask: Inverse of conditioning mask
        # We don't compute loss on conditioned tokens (Reference & First Frame)
        audio_loss_mask = ~audio_conditioning_mask

        return audio_modality, audio_targets, audio_loss_mask

    def _log_alignment_debug(
        self,
        step: int,
        fps: float,
        num_frames: int,
        height: int,
        width: int,
        apply_conditioning: bool,
        video_positions: Tensor,
        video_conditioning_mask: Tensor,
        video_timesteps: Tensor,
        audio_modality: Modality | None,
        audio_loss_mask: Tensor | None,
    ) -> None:
        """Log debug info for audio-video alignment. Only called for first N steps.

        All values are read directly from tensors — no re-derivation of masks or positions.
        """
        hw = height * width
        video_seq_len = video_positions.shape[2]
        total_latent_frames = video_seq_len // hw
        has_ref = self.config.with_reference_image

        # --- Header ---
        lines = [
            f"\n{'='*80}",
            f"[DEBUG] AV Alignment - Step {step}",
            f"{'='*80}",
            f"  fps={fps}, total_latent_frames={total_latent_frames} "
            f"(ref_image={has_ref}, video_latent_frames={total_latent_frames - (1 if has_ref else 0)})",
            f"  latent H={height}, W={width}, tokens_per_latent_frame={hw}",
            f"  conditioning: {'ON' if apply_conditioning else 'OFF'} "
            f"(p={self.config.first_frame_conditioning_p}, "
            f"num_conditioning_frames={self.config.num_conditioning_frames})",
        ]

        # --- Video conditioning mask summary ---
        vid_cond_count = video_conditioning_mask[0].sum().item()
        vid_cond_frames = vid_cond_count // hw if hw > 0 else 0
        lines.append(
            f"  video conditioned: {vid_cond_count}/{video_seq_len} tokens "
            f"({vid_cond_frames} latent frames)"
        )

        # --- Per-latent-frame video info (read from tensor) ---
        lines.append(f"  --- Video per-frame detail (batch 0) ---")
        vp_time = video_positions[0, 0, :, :]  # [seq_len, 2] (time start, end)
        show_frames = list(range(min(total_latent_frames, 6)))
        if total_latent_frames > 6:
            show_frames.append(total_latent_frames - 1)

        for f_idx in show_frames:
            tok_start = f_idx * hw
            t_start = vp_time[tok_start, 0].item()
            t_end = vp_time[tok_start, 1].item()
            sigma = video_timesteps[0, tok_start].item()
            is_cond = video_conditioning_mask[0, tok_start].item()

            # Determine label based on structure:
            # frame 0 with ref_image → ref_image (1px, separately encoded)
            # frame 1 with ref_image → video latent[0] (1px, causal first of separate encode)
            # frame 2+ with ref_image → video latent[N] (8px each)
            if has_ref and f_idx == 0:
                label = "ref_image, 1px"
            elif has_ref:
                vid_idx = f_idx - 1
                px = 1 if vid_idx == 0 else 8
                label = f"vid_latent[{vid_idx}], {px}px"
            else:
                px = 1 if f_idx == 0 else 8
                label = f"vid_latent[{f_idx}], {px}px"

            cond_tag = " [COND sigma=0]" if is_cond else ""
            ellipsis = " ..." if f_idx == show_frames[-1] and total_latent_frames > 6 else ""
            lines.append(
                f"    frame[{f_idx}] ({label}): "
                f"t=[{t_start:.6f}, {t_end:.6f})s, sigma={sigma:.4f}{cond_tag}{ellipsis}"
            )

        # --- Video gen start (first non-conditioned token from mask) ---
        vid_nocond = (~video_conditioning_mask[0]).nonzero(as_tuple=True)[0]
        if len(vid_nocond) > 0:
            vid_gen_tok = vid_nocond[0].item()
            vid_gen_frame = vid_gen_tok // hw
            vid_gen_t = vp_time[vid_gen_tok, 0].item()
        else:
            vid_gen_tok = video_seq_len
            vid_gen_frame = total_latent_frames
            vid_gen_t = vp_time[-1, 1].item()
        lines.append(
            f"  video gen start: frame[{vid_gen_frame}] token[{vid_gen_tok}], t={vid_gen_t:.6f}s"
        )

        # --- Sanity checks ---
        checks = []
        if apply_conditioning:
            # Conditioned tokens must have sigma=0
            cond_sigmas = video_timesteps[0][video_conditioning_mask[0]]
            if cond_sigmas.numel() > 0 and not torch.all(cond_sigmas == 0):
                checks.append(f"[FAIL] conditioned video tokens have non-zero sigma: "
                              f"max={cond_sigmas.max().item():.4f}")
            else:
                checks.append(f"[OK] conditioned video tokens all have sigma=0")

        if has_ref:
            ref_t_end = vp_time[hw - 1, 1].item()
            vid0_t_start = vp_time[hw, 0].item()
            # ref must end at t<=0, vid[0] must start at t>=0
            # Use 1e-4 tolerance to account for bf16 precision (1 ULP ~ 3.9e-5 at ~0.04)
            _tol = 1e-4
            if ref_t_end > _tol:
                checks.append(f"[FAIL] ref_image end time {ref_t_end:.6f}s > 0 (should be <=0)")
            else:
                checks.append(f"[OK] ref_image time range ends at {ref_t_end:.6f}s (<=0)")
            if abs(vid0_t_start) > _tol:
                checks.append(f"[FAIL] vid[0] start time {vid0_t_start:.6f}s != 0")
            else:
                checks.append(f"[OK] vid[0] starts at t={vid0_t_start:.6f}s (~0)")

        # --- Audio info ---
        if audio_modality is not None and audio_loss_mask is not None:
            audio_pos = audio_modality.positions  # [B, 1, T, 2]
            audio_ts = audio_modality.timesteps   # [B, T]
            audio_seq = audio_pos.shape[2]
            audio_cond_mask = ~audio_loss_mask[0]  # True = conditioned
            audio_cond_count = audio_cond_mask.sum().item()
            a_time = audio_pos[0, 0, :, :]  # [T, 2]

            lines.append(f"  --- Audio ---")
            lines.append(
                f"    seq_len={audio_seq}, "
                f"conditioned: {audio_cond_count}/{audio_seq} tokens"
            )
            lines.append(
                f"    positions range: [{a_time[0, 0].item():.6f}, {a_time[-1, 1].item():.6f})s"
            )

            # Show boundary between conditioned and generated audio
            audio_nocond = (~audio_cond_mask).nonzero(as_tuple=True)[0]
            if len(audio_nocond) > 0:
                audio_gen_tok = audio_nocond[0].item()
                audio_gen_t = a_time[audio_gen_tok, 0].item()
            else:
                audio_gen_tok = audio_seq
                audio_gen_t = a_time[-1, 1].item()
            lines.append(
                f"  audio gen start: token[{audio_gen_tok}], t={audio_gen_t:.6f}s"
            )

            # Show first few conditioned and first gen token details
            if audio_cond_count > 0:
                # Last conditioned token
                last_cond_idx = audio_cond_mask.nonzero(as_tuple=True)[0][-1].item()
                lines.append(
                    f"    last cond token[{last_cond_idx}]: "
                    f"t=[{a_time[last_cond_idx, 0].item():.6f}, {a_time[last_cond_idx, 1].item():.6f})s, "
                    f"sigma={audio_ts[0, last_cond_idx].item():.4f}"
                )
            if audio_gen_tok < audio_seq:
                lines.append(
                    f"    first gen token[{audio_gen_tok}]: "
                    f"t=[{a_time[audio_gen_tok, 0].item():.6f}, {a_time[audio_gen_tok, 1].item():.6f})s, "
                    f"sigma={audio_ts[0, audio_gen_tok].item():.4f}"
                )

            # Audio conditioned tokens must have sigma=0
            if audio_cond_count > 0:
                audio_cond_sigmas = audio_ts[0][audio_cond_mask]
                if not torch.all(audio_cond_sigmas == 0):
                    checks.append(f"[FAIL] conditioned audio tokens have non-zero sigma: "
                                  f"max={audio_cond_sigmas.max().item():.4f}")
                else:
                    checks.append(f"[OK] conditioned audio tokens all have sigma=0")

            # Alignment: video gen start vs audio gen start
            delta = vid_gen_t - audio_gen_t
            checks.append(
                f"{'[OK]' if abs(delta) <= 0.1 else '[WARN]'} "
                f"AV gen start alignment: video={vid_gen_t:.6f}s, audio={audio_gen_t:.6f}s, "
                f"delta={delta:.6f}s"
            )
        else:
            lines.append(f"  --- Audio: DISABLED ---")

        # --- Print checks ---
        if checks:
            lines.append(f"  --- Sanity Checks ---")
            for c in checks:
                lines.append(f"    {c}")

        lines.append(f"{'='*80}")
        logger.info("\n".join(lines))

    def compute_loss(
        self,
        video_pred: Tensor,
        audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> Tensor:
        """Compute masked MSE loss for video and optionally audio."""

        # --- Buffer Update Logic ---
        # Retrieve the clean status
        is_clean_step = getattr(inputs, "is_clean_step", False)

        # Update buffer ONLY IF:
        #    a. The current input was clean and The update probability (clean_buffer_update_prob) is met
        #    b. The current input was dirty (is_clean_step is False)
        if is_clean_step and random.random() < self.config.clean_buffer_update_prob or not is_clean_step:
            # 1. Prepare Shape Info
            num_video_frames = getattr(inputs, "orig_num_frames")
            height = getattr(inputs, "orig_height")
            width = getattr(inputs, "orig_width")

            unpatchify_shape = VideoLatentShape(
                batch=video_pred.shape[0],
                channels=video_pred.shape[-1],
                frames=num_video_frames,
                height=height,
                width=width
            )

            # Unpatchify EVERYTHING to 5D [B, C, F, H, W]
            pred_5d = self._video_patchifier.unpatchify(video_pred, unpatchify_shape)
            target_5d = self._video_patchifier.unpatchify(inputs.video_targets, unpatchify_shape)
            noisy_5d = self._video_patchifier.unpatchify(inputs.video.latent, unpatchify_shape)

            # Strip conditioning frames (ref_image + context video frames)
            strip_frames = self.config.num_conditioning_frames
            if strip_frames > 0:
                pred_5d = pred_5d[:, :, strip_frames:, :, :]
                target_5d = target_5d[:, :, strip_frames:, :, :]
                noisy_5d = noisy_5d[:, :, strip_frames:, :, :]

            # Get Correct Sigma
            # inputs.video.timesteps is [B, SeqLen].
            # Reference tokens have sigma=0, Target tokens have sigma=t.
            # We just need 't'. Since t >= 0, max() gets us t safely.
            t_sigma = inputs.video.timesteps.max(dim=1).values # [B]
            t_sigma = t_sigma.view(-1, 1, 1, 1, 1) # [B, 1, 1, 1, 1] for broadcasting

            # Calculate Clean Latents (x0 = xt - sigma * v)
            # Both pred and target are purely the generated video now, so the math holds.
            pred_clean_5d = noisy_5d - t_sigma * pred_5d
            true_clean_5d = noisy_5d - t_sigma * target_5d

            # Calculate Error
            error_5d = pred_clean_5d - true_clean_5d

            # Store in Buffer
            detached_error = error_5d.detach().cpu()
            h, w = detached_error.shape[-2], detached_error.shape[-1]
            bucket_key = (h, w)

            if bucket_key not in self.error_buffer:
                self.error_buffer[bucket_key] = []

            if len(self.error_buffer[bucket_key]) >= self.config.error_buffer_size:
                self.error_buffer[bucket_key].pop(0)

            self.error_buffer[bucket_key].append(detached_error)
        # -----------------------------

        # Video loss
        video_loss = (video_pred - inputs.video_targets).pow(2)
        video_loss_mask = inputs.video_loss_mask.unsqueeze(-1).float()
        video_loss = video_loss.mul(video_loss_mask).div(video_loss_mask.mean())
        video_loss = video_loss.mean()

        # If no audio, return video loss only
        if not self.config.with_audio or audio_pred is None or inputs.audio_targets is None:
            return video_loss

        # Audio loss
        audio_loss = (audio_pred - inputs.audio_targets).pow(2)
        if inputs.audio_loss_mask is not None:
            # audio_loss_mask is [B, T], targets/pred are [B, T, C] or similar.
            # We need to broadcast. Audio outputs are patchified [B, T, Channels].
            audio_loss_mask = inputs.audio_loss_mask.unsqueeze(-1).float()
            audio_loss = audio_loss.mul(audio_loss_mask).div(audio_loss_mask.mean() + 1e-8)

        audio_loss = audio_loss.mean()

        # Combined loss
        return video_loss + audio_loss, video_loss, audio_loss

    @staticmethod
    def _create_conditioning_mask(
        batch_size: int,
        sequence_length: int,
        height: int,
        width: int,
        num_frames: int,
        apply_conditioning: bool,
        device: torch.device,
    ) -> Tensor:
        """Create conditioning mask for the first N frames.

        Args:
            batch_size: Batch size
            sequence_length: Total sequence length
            height: Latent height
            width: Latent width
            num_frames: Number of frames to condition on (e.g. 2 = ref_image + 1 video frame)
            apply_conditioning: Whether to apply conditioning (shared decision for video+audio)
            device: Target device

        Returns:
            Boolean mask where True indicates conditioned tokens
        """
        conditioning_mask = torch.zeros(batch_size, sequence_length, dtype=torch.bool, device=device)

        if apply_conditioning:
            end_idx = height * width * num_frames
            if end_idx < sequence_length:
                conditioning_mask[:, :end_idx] = True

        return conditioning_mask