"""Single-segment Encore audio-video generation.

Stage 1 denoises at half resolution with CFG/STG guidance, stage 2 upsamples by 2x
and refines with the distilled LoRA using the res_2s sampler.

On top of the upstream two-stage HQ pipeline this module adds the Encore token layout
(reference image + reference audio anchors, continuation frames) and the learned
condition routing table (see ``--routing-scale``).

For multi-segment long-form generation see
:mod:`ltx_pipelines.ti2vid_two_stages_hq_encore_long`.
"""

import logging
from collections.abc import Iterator
from dataclasses import replace

import torch

from ltx_core.components.diffusion_steps import Res2sDiffusionStep
from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.protocols import DiffusionStepProtocol
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.conditioning import AudioConditionByLatentIndex
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
from ltx_core.model.audio_vae.ops import AudioProcessor
from ltx_core.model.upsampler import upsample_video
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.model.video_vae import decode_video as vae_decode_video
from ltx_core.quantization import QuantizationPolicy
from ltx_core.tools import VideoLatentShape
from ltx_core.types import Audio, LatentState, VideoPixelShape
from ltx_pipelines.utils import (
    ModelLedger,
    assert_resolution,
    cleanup_memory,
    combined_image_conditionings,
    denoise_audio_video_encore,
    encode_prompts,
    get_device,
    multi_modal_guider_denoising_func,
    res2s_audio_video_denoising_loop,
    simple_denoising_func,
)
from ltx_pipelines.utils.args import ImageConditioningInput, hq_2_stage_arg_parser
from ltx_pipelines.utils.constants import LTX_2_3_HQ_PARAMS, STAGE_2_DISTILLED_SIGMA_VALUES
from ltx_pipelines.utils.media_io import encode_video, load_audio_conditioning
from ltx_pipelines.utils.types import PipelineComponents

device = get_device()

class TI2VidTwoStagesHQEncorePipeline:
    """
    Two-stage text/image-to-video generation pipeline using the res_2s sampler.
    Same structure as :class:`TI2VidTwoStagesPipeline`: stage 1 generates video at
    half of the target resolution with CFG guidance (assuming  full model is used),
    then Stage 2 upsamples by 2x and refines using a distilled LoRA for higher
    quality output.
    Uses the res_2s second-order sampler instead of Euler, allowing fewer
    steps for comparable quality. Supports optional image conditioning via
    the images parameter.
    """

    def __init__(
        self,
        checkpoint_path: str,
        distilled_lora: list[LoraPathStrengthAndSDOps],
        distilled_lora_strength_stage_1: float,
        distilled_lora_strength_stage_2: float,
        spatial_upsampler_path: str,
        gemma_root: str,
        loras: tuple[LoraPathStrengthAndSDOps, ...],
        routing_scale: float = 1.0,
        device: str = device,
        quantization: QuantizationPolicy | None = None,
    ):
        self.device = device
        self.dtype = torch.bfloat16
        distilled_lora_stage_1 = LoraPathStrengthAndSDOps(
            path=distilled_lora[0].path,
            strength=distilled_lora_strength_stage_1,
            sd_ops=distilled_lora[0].sd_ops,
        )
        distilled_lora_stage_2 = LoraPathStrengthAndSDOps(
            path=distilled_lora[0].path,
            strength=distilled_lora_strength_stage_2,
            sd_ops=distilled_lora[0].sd_ops,
        )
        self.stage_1_model_ledger = ModelLedger(
            dtype=self.dtype,
            device=device,
            checkpoint_path=checkpoint_path,
            gemma_root_path=gemma_root,
            spatial_upsampler_path=spatial_upsampler_path,
            loras=(distilled_lora_stage_1, *loras),
            quantization=quantization,
            routing_scale=routing_scale,
        )

        self.stage_2_model_ledger = ModelLedger(
            dtype=self.dtype,
            device=device,
            checkpoint_path=checkpoint_path,
            gemma_root_path=gemma_root,
            spatial_upsampler_path=spatial_upsampler_path,
            loras=(distilled_lora_stage_2, *loras),
            quantization=quantization,
            routing_scale=routing_scale,
        )

        self.pipeline_components = PipelineComponents(
            dtype=self.dtype,
            device=device,
        )

        audio_encoder = self.stage_1_model_ledger.audio_encoder()
        self.audio_encoder = audio_encoder
        self.audio_processor = AudioProcessor(
            target_sample_rate=audio_encoder.sample_rate,
            mel_bins=audio_encoder.mel_bins,
            mel_hop_length=audio_encoder.mel_hop_length,
            n_fft=audio_encoder.n_fft,
        )

    @torch.inference_mode()
    def __call__(  # noqa: PLR0913
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        num_inference_steps: int,
        video_guider_params: MultiModalGuiderParams,
        audio_guider_params: MultiModalGuiderParams,
        images: list[ImageConditioningInput],
        tiling_config: TilingConfig | None = None,
        enhance_prompt: bool = False,
        audios: list[tuple[str, int, float]] = [],
        prev_decoded_audio: torch.Tensor | None = None,
        ref_position_offset: float = 0.0,
        ref_audio_latent_override: torch.Tensor | None = None,
    ) -> tuple[Iterator[torch.Tensor], Audio, torch.Tensor]:
        assert_resolution(height=height, width=width, is_two_stage=True)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        dtype = torch.bfloat16

        ctx_p, ctx_n = encode_prompts(
            [prompt, negative_prompt],
            self.stage_1_model_ledger,
            enhance_first_prompt=enhance_prompt,
            enhance_prompt_image=images[0][1] if len(images) > 0 else None,
            enhance_prompt_seed=seed,
        )
        v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
        v_context_n, a_context_n = ctx_n.video_encoding, ctx_n.audio_encoding

        # Stage 1: encode image conditionings with the VAE encoder, then free it
        # before loading the transformer to reduce peak VRAM.
        stage_1_output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames + 8, # the reference image placeholder
            width=width // 2,
            height=height // 2,
            fps=frame_rate,
        )
        video_encoder = self.stage_1_model_ledger.video_encoder()
        stage_1_conditionings = combined_image_conditionings(
            images=images,
            height=stage_1_output_shape.height,
            width=stage_1_output_shape.width,
            video_encoder=video_encoder,
            dtype=dtype,
            device=self.device,
        )
        torch.cuda.synchronize()
        del video_encoder
        cleanup_memory()

        # Encode audio from input wav: extract ref_audio (global) and First Input Audio
        audio_encoder = self.stage_1_model_ledger.audio_encoder()
        # ref_audio is a GLOBAL reference (speaker timbre/style).
        # tokens_per_frame=8, ref_audio_num_frames=8 → 64 tokens (~2.56s)
        ref_audio_len = 8 * 8

        if audios and len(audios) != 0:
            audio_path = audios[0][0]
            spectrogram = load_audio_conditioning(
                audio_path=audio_path,
                audio_processor=self.audio_processor,
                dtype=dtype,
                device=self.device,
            )
            encoded_audio = audio_encoder(spectrogram)
            ref_audio_latent = torch.zeros_like(encoded_audio)
            min_len = min(ref_audio_len, encoded_audio.shape[2])
            ref_audio_latent[:, :, :min_len, :] = encoded_audio[:, :, :min_len, :]
        elif ref_audio_latent_override is not None:
            # Use cached ref_audio from first inference
            ref_audio_latent = ref_audio_latent_override
        else:
            # No audio input yet — construct zero ref_audio_latent
            # Audio latent shape: [B, z_channels=8, T, mel_bins=16]
            ref_audio_latent = torch.zeros(
                1, 8, ref_audio_len, 16, dtype=dtype, device=self.device,
            )

        # First Input Audio: 2 tokens at target latent_idx=0, corresponding to the
        # conditioning video frame's audio.
        audio_context_len = 2
        if prev_decoded_audio is not None:
            vocoder_sr = self.stage_2_model_ledger.vocoder().output_sampling_rate
            # Take enough tail waveform to produce audio_context_len tokens after encoding.
            # audio_context_len=2 tokens need 4*2+1=9 mel frames = 9*hop_length samples
            tail_samples = 9 * self.audio_processor.mel_transform.hop_length
            tail_waveform = prev_decoded_audio[:, -tail_samples:]  # [channels, samples]
            # Resample first, then pad if needed so STFT has enough samples
            tail_audio = Audio(waveform=tail_waveform.unsqueeze(0).cpu(), sampling_rate=vocoder_sr)
            tail_audio = self.audio_processor.resample_audio(tail_audio)
            min_samples = self.audio_processor.mel_transform.n_fft + 1
            if tail_audio.waveform.shape[-1] < min_samples:
                pad_size = min_samples - tail_audio.waveform.shape[-1]
                padded = torch.nn.functional.pad(tail_audio.waveform, (pad_size, 0))
                tail_audio = Audio(waveform=padded, sampling_rate=tail_audio.sampling_rate)
            tail_spectrogram = self.audio_processor.waveform_to_mel(tail_audio).to(device=self.device, dtype=dtype)

            tail_encoded = audio_encoder(tail_spectrogram)  # [1, C, T, F]
            first_input_audio = tail_encoded[:, :, -audio_context_len:, :]
        else:
            first_input_audio = torch.zeros(
                1, 8, audio_context_len, 16, dtype=dtype, device=self.device,
            )

        stage_1_audio_conditionings = [
            AudioConditionByLatentIndex(
                latent=first_input_audio,
                strength=1.,
                latent_idx=0,
            )
        ]

        transformer = self.stage_1_model_ledger.transformer()

        empty_latent = torch.empty(VideoLatentShape.from_pixel_shape(stage_1_output_shape).to_torch_shape())
        stepper = Res2sDiffusionStep()
        sigmas = (
            LTX2Scheduler()
            .execute(latent=empty_latent, steps=num_inference_steps)
            .to(dtype=torch.float32, device=self.device)
        )

        def first_stage_denoising_loop(
            sigmas: torch.Tensor, video_state: LatentState, audio_state: LatentState, stepper: DiffusionStepProtocol
        ) -> tuple[LatentState, LatentState]:
            return res2s_audio_video_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=multi_modal_guider_denoising_func(
                    video_guider=MultiModalGuider(
                        params=video_guider_params,
                        negative_context=v_context_n,
                    ),
                    audio_guider=MultiModalGuider(
                        params=audio_guider_params,
                        negative_context=a_context_n,
                    ),
                    v_context=v_context_p,
                    a_context=a_context_p,
                    transformer=transformer,  # noqa: F821
                ),
            )

        video_state, audio_state = denoise_audio_video_encore(
            output_shape=stage_1_output_shape,
            conditionings=stage_1_conditionings,
            noiser=noiser,
            sigmas=sigmas,
            stepper=stepper,
            denoising_loop_fn=first_stage_denoising_loop,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            audio_conditionings=stage_1_audio_conditionings,
            ref_audio_latent=ref_audio_latent,
            ref_position_offset=ref_position_offset,
        )

        torch.cuda.synchronize()
        del transformer
        cleanup_memory()

        # Stage 2: Upsample and refine the video at higher resolution with distilled LORA.
        video_encoder = self.stage_1_model_ledger.video_encoder()
        upscaled_video_latent = upsample_video(
            latent=video_state.latent[:1],
            video_encoder=video_encoder,
            upsampler=self.stage_2_model_ledger.spatial_upsampler(),
        )

        stage_2_output_shape = VideoPixelShape(batch=1, frames=num_frames+8, width=width, height=height, fps=frame_rate)
        stage_2_conditionings = combined_image_conditionings(
            images=images,
            height=stage_2_output_shape.height,
            width=stage_2_output_shape.width,
            video_encoder=video_encoder,
            dtype=dtype,
            device=self.device,
        )
        torch.cuda.synchronize()
        del video_encoder
        cleanup_memory()

        transformer = self.stage_2_model_ledger.transformer()
        distilled_sigmas = torch.tensor(STAGE_2_DISTILLED_SIGMA_VALUES, device=self.device)

        def second_stage_denoising_loop(
            sigmas: torch.Tensor, video_state: LatentState, audio_state: LatentState, stepper: DiffusionStepProtocol
        ) -> tuple[LatentState, LatentState]:
            return res2s_audio_video_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=simple_denoising_func(
                    video_context=v_context_p,
                    audio_context=a_context_p,
                    transformer=transformer,  # noqa: F821
                ),
            )

        video_state, audio_state = denoise_audio_video_encore(
            output_shape=stage_2_output_shape,
            conditionings=stage_2_conditionings,
            noiser=noiser,
            sigmas=distilled_sigmas,
            stepper=stepper,
            denoising_loop_fn=second_stage_denoising_loop,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            noise_scale=distilled_sigmas[0],
            initial_video_latent=upscaled_video_latent,
            initial_audio_latent=audio_state.latent,
            audio_conditionings=stage_1_audio_conditionings,
            ref_audio_latent=ref_audio_latent,
            ref_position_offset=ref_position_offset,
        )

        torch.cuda.synchronize()
        del transformer
        cleanup_memory()

        # Strip ref_image from video (first latent frame).
        # Audio ref was already stripped inside denoise_audio_video_encore.
        video_state = replace(video_state, latent=video_state.latent[:, :, 1:, :, :])

        # Save raw audio latent before decoding (for ref_audio extraction)
        raw_audio_latent = audio_state.latent.clone()

        decoded_video = vae_decode_video(
            video_state.latent, self.stage_2_model_ledger.video_decoder(), tiling_config, generator
        )
        decoded_audio = vae_decode_audio(
            audio_state.latent, self.stage_2_model_ledger.audio_decoder(), self.stage_2_model_ledger.vocoder()
        )
        return decoded_video, decoded_audio, raw_audio_latent


@torch.inference_mode()
def main() -> None:
    """Generate a single audio-video segment from one image + one prompt."""
    logging.getLogger().setLevel(logging.INFO)
    parser = hq_2_stage_arg_parser(params=LTX_2_3_HQ_PARAMS)
    args = parser.parse_args()

    if not args.images:
        raise ValueError("--image PATH FRAME_IDX STRENGTH is required for Encore inference.")
    init_image: ImageConditioningInput = args.images[0]

    pipeline = TI2VidTwoStagesHQEncorePipeline(
        checkpoint_path=args.checkpoint_path,
        distilled_lora=args.distilled_lora,
        distilled_lora_strength_stage_1=args.distilled_lora_strength_stage_1,
        distilled_lora_strength_stage_2=args.distilled_lora_strength_stage_2,
        spatial_upsampler_path=args.spatial_upsampler_path,
        gemma_root=args.gemma_root,
        loras=tuple(args.lora) if args.lora else (),
        routing_scale=args.routing_scale,
        quantization=args.quantization,
    )
    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(args.num_frames, tiling_config)

    # Encore token layout: frame_idx=0 is the anchor (reference) frame, frame_idx=1 is
    # the first generated frame, conditioned with a lossy (crf=27) copy of the anchor.
    images = [
        ImageConditioningInput(path=init_image.path, frame_idx=0, strength=1.0, crf=0),
        ImageConditioningInput(path=init_image.path, frame_idx=1, strength=1.0, crf=27),
    ]

    video, audio, _raw_audio_latent = pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        num_inference_steps=args.num_inference_steps,
        video_guider_params=MultiModalGuiderParams(
            cfg_scale=args.video_cfg_guidance_scale,
            stg_scale=args.video_stg_guidance_scale,
            rescale_scale=args.video_rescale_scale,
            modality_scale=args.a2v_guidance_scale,
            skip_step=args.video_skip_step,
            stg_blocks=args.video_stg_blocks,
        ),
        audio_guider_params=MultiModalGuiderParams(
            cfg_scale=args.audio_cfg_guidance_scale,
            stg_scale=args.audio_stg_guidance_scale,
            rescale_scale=args.audio_rescale_scale,
            modality_scale=args.v2a_guidance_scale,
            skip_step=args.audio_skip_step,
            stg_blocks=args.audio_stg_blocks,
        ),
        images=images,
        tiling_config=tiling_config,
        enhance_prompt=args.enhance_prompt,
        audios=args.audios,
        ref_position_offset=args.ref_position_offset,
    )

    encode_video(
        video=video,
        fps=args.frame_rate,
        audio=audio,
        output_path=args.output_path,
        video_chunks_number=video_chunks_number,
    )


if __name__ == "__main__":
    main()
