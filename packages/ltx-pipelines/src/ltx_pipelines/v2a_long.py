from __future__ import annotations

import logging
import math
import wave
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import av
import numpy as np
import torch

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.protocols import DiffusionStepProtocol
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.conditioning import AudioConditionByLatentIndex, ConditioningItem
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
from ltx_core.model.audio_vae.ops import AudioProcessor
from ltx_core.quantization import QuantizationPolicy
from ltx_core.tools import LatentTools
from ltx_core.types import Audio, AudioLatentShape, LatentState, VideoPixelShape
from ltx_pipelines.retake import TemporalRegionMask
from ltx_pipelines.utils import ModelLedger, cleanup_memory, encode_prompts, get_device, multi_modal_guider_denoising_func
from ltx_pipelines.utils.args import default_1_stage_arg_parser
from ltx_pipelines.utils.helpers import (
    assert_resolution,
    noise_audio_state,
    noise_video_state,
    simple_denoising_func,
)
from ltx_pipelines.utils.media_io import (
    decode_video_from_file,
    encode_video,
    get_videostream_metadata,
    load_audio_conditioning,
    normalize_latent,
    resize_and_center_crop,
)
from ltx_pipelines.utils.samplers import euler_denoising_loop
from ltx_pipelines.utils.types import PipelineComponents

logger = logging.getLogger(__name__)
device = get_device()
TARGET_FPS = 24.0
# Single-stage pipeline runs at half of the 2-stage base resolution (1920x1088),
# matching the distribution used by stage-1 of the two-stage training recipe.
LANDSCAPE_SIZE = (544, 960)
PORTRAIT_SIZE = (960, 544)


def _get_target_dimensions(width: int, height: int) -> tuple[int, int]:
    if height > width:
        return PORTRAIT_SIZE
    return LANDSCAPE_SIZE


def _load_standardized_video(video_path: str) -> tuple[torch.Tensor, float, int, int]:
    fps, num_frames, width, height = get_videostream_metadata(video_path)
    target_height, target_width = _get_target_dimensions(width, height)
    duration = num_frames / fps

    container = av.open(video_path)
    try:
        video_stream = next(stream for stream in container.streams if stream.type == "video")
        decoded_frames: list[torch.Tensor] = []
        frame_times: list[float] = []
        for frame_idx, frame in enumerate(container.decode(video_stream)):
            decoded_frames.append(torch.from_numpy(frame.to_rgb().to_ndarray()).to(torch.uint8))
            frame_time = frame.time
            if frame_time is None:
                frame_time = frame_idx / fps
            frame_times.append(float(frame_time))
    finally:
        container.close()

    if not decoded_frames:
        raise ValueError(f"No frames decoded from {video_path}")

    target_num_frames = max(1, round(duration * TARGET_FPS))
    target_times = np.arange(target_num_frames, dtype=np.float32) / TARGET_FPS
    frame_times_np = np.asarray(frame_times, dtype=np.float32)
    right_indices = np.searchsorted(frame_times_np, target_times, side="left")
    right_indices = np.clip(right_indices, 0, len(decoded_frames) - 1)
    left_indices = np.clip(right_indices - 1, 0, len(decoded_frames) - 1)

    use_right = np.abs(frame_times_np[right_indices] - target_times) <= np.abs(frame_times_np[left_indices] - target_times)
    selected_indices = np.where(use_right, right_indices, left_indices)
    selected_frames = torch.stack([decoded_frames[idx] for idx in selected_indices.tolist()], dim=0)

    resized = resize_and_center_crop(selected_frames.to(torch.float32), target_height, target_width)
    standardized_frames = resized[0].permute(1, 2, 3, 0).round().clamp(0, 255).to(torch.uint8).cpu()
    return standardized_frames, TARGET_FPS, target_width, target_height


def _encode_video_clip(
    video_encoder: torch.nn.Module,
    clip_frames: torch.Tensor,
    output_shape: VideoPixelShape,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    normalized = normalize_latent(
        resize_and_center_crop(clip_frames.to(device=device, dtype=torch.float32), output_shape.height, output_shape.width),
        device,
        dtype,
    )
    return video_encoder(normalized)


def _prepare_audio_waveform(audio: Audio) -> torch.Tensor:
    waveform = audio.waveform
    if waveform.dim() == 3 and waveform.shape[0] == 1:
        waveform = waveform.squeeze(0)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    return waveform


def _trim_audio_to_duration(audio: Audio, target_duration: float) -> Audio:
    waveform = _prepare_audio_waveform(audio)
    target_samples = round(target_duration * audio.sampling_rate)
    trimmed = waveform[..., :target_samples]
    return Audio(waveform=trimmed, sampling_rate=audio.sampling_rate)


def _extract_ref_audio_latent(encoded_audio_latent: torch.Tensor, ref_audio_len: int) -> torch.Tensor:
    ref_audio_latent = torch.zeros(
        encoded_audio_latent.shape[0],
        encoded_audio_latent.shape[1],
        ref_audio_len,
        encoded_audio_latent.shape[3],
        dtype=encoded_audio_latent.dtype,
        device=encoded_audio_latent.device,
    )
    min_len = min(ref_audio_len, encoded_audio_latent.shape[2])
    ref_audio_latent[:, :, :min_len, :] = encoded_audio_latent[:, :, :min_len, :]
    return ref_audio_latent


def _write_audio_wav(audio: Audio, output_path: Path) -> None:
    waveform = _prepare_audio_waveform(audio).cpu().numpy()
    if waveform.shape[0] == 1:
        interleaved = waveform[0]
        channels = 1
    else:
        interleaved = waveform.T.reshape(-1)
        channels = waveform.shape[0]
    samples = np.clip(interleaved, -1.0, 1.0)
    pcm16 = (samples * 32767.0).astype(np.int16)
    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(audio.sampling_rate)
        wav_file.writeframes(pcm16.tobytes())


def _merge_audio_segments(existing: Audio | None, new_segment: Audio, overlap_duration: float) -> Audio:
    if existing is None:
        return new_segment

    if existing.sampling_rate != new_segment.sampling_rate:
        raise ValueError("Sampling rates of audio segments do not match")

    existing_waveform = _prepare_audio_waveform(existing)
    new_waveform = _prepare_audio_waveform(new_segment)
    overlap_samples = min(
        round(overlap_duration * existing.sampling_rate),
        existing_waveform.shape[-1],
        new_waveform.shape[-1],
    )
    if overlap_samples <= 0:
        merged = torch.cat([existing_waveform, new_waveform], dim=-1)
        return Audio(waveform=merged, sampling_rate=existing.sampling_rate)

    fade = torch.linspace(0.0, 1.0, overlap_samples, dtype=existing_waveform.dtype)
    fade = fade.to(existing_waveform.device).view(1, -1)
    mixed = existing_waveform[..., -overlap_samples:] * (1.0 - fade) + new_waveform[..., :overlap_samples] * fade
    merged = torch.cat(
        [existing_waveform[..., :-overlap_samples], mixed, new_waveform[..., overlap_samples:]],
        dim=-1,
    )
    return Audio(waveform=merged, sampling_rate=existing.sampling_rate)


def _decode_video_segment(
    standardized_video: torch.Tensor,
    start_frame: int,
    frame_count: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    clip = standardized_video[start_frame : start_frame + frame_count]
    if clip.numel() == 0:
        raise ValueError(f"Could not decode frames starting at frame {start_frame}")

    actual_frames = clip.shape[0]
    if actual_frames < frame_count:
        pad_frame = clip[-1:].clone().expand(frame_count - actual_frames, -1, -1, -1)
        padded_clip = torch.cat([clip, pad_frame], dim=0)
    else:
        padded_clip = clip

    return padded_clip, clip, actual_frames


class V2ALongPipeline:
    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str,
        loras: tuple[LoraPathStrengthAndSDOps, ...],
        device: torch.device = device,
        quantization: QuantizationPolicy | None = None,
    ):
        self.device = device
        self.dtype = torch.bfloat16
        self.model_ledger = ModelLedger(
            dtype=self.dtype,
            device=device,
            checkpoint_path=checkpoint_path,
            gemma_root_path=gemma_root,
            loras=loras,
            quantization=quantization,
        )
        self.pipeline_components = PipelineComponents(dtype=self.dtype, device=device)
        audio_encoder = self.model_ledger.audio_encoder()
        self.audio_encoder = audio_encoder
        self.audio_processor = AudioProcessor(
            target_sample_rate=audio_encoder.sample_rate,
            mel_bins=audio_encoder.mel_bins,
            mel_hop_length=audio_encoder.mel_hop_length,
            n_fft=audio_encoder.n_fft,
        )

    def _build_first_input_audio(self, prev_generated_audio: Audio | None, audio_context_len: int = 2) -> torch.Tensor:
        if prev_generated_audio is None:
            return torch.zeros(1, 8, audio_context_len, 16, dtype=self.dtype, device=self.device)

        waveform = _prepare_audio_waveform(prev_generated_audio).unsqueeze(0)
        tail_samples = 9 * self.audio_processor.mel_transform.hop_length
        tail_waveform = waveform[:, :, -tail_samples:].cpu()
        tail_audio = Audio(waveform=tail_waveform, sampling_rate=prev_generated_audio.sampling_rate)
        tail_audio = self.audio_processor.resample_audio(tail_audio)

        min_samples = self.audio_processor.mel_transform.n_fft + 1
        if tail_audio.waveform.shape[-1] < min_samples:
            pad_size = min_samples - tail_audio.waveform.shape[-1]
            padded = torch.nn.functional.pad(tail_audio.waveform, (pad_size, 0))
            tail_audio = Audio(waveform=padded, sampling_rate=tail_audio.sampling_rate)

        tail_spectrogram = self.audio_processor.waveform_to_mel(tail_audio).to(device=self.device, dtype=self.dtype)
        tail_encoded = self.audio_encoder(tail_spectrogram)
        return tail_encoded[:, :, -audio_context_len:, :]

    def _denoise_audio_only_with_ref(  # noqa: PLR0913
        self,
        output_shape: VideoPixelShape,
        noiser: GaussianNoiser,
        sigmas: torch.Tensor,
        stepper: DiffusionStepProtocol,
        denoising_loop_fn,
        initial_video_latent: torch.Tensor,
        audio_conditionings: list[ConditioningItem],
        ref_audio_latent: torch.Tensor | None = None,
        initial_audio_latent: torch.Tensor | None = None,
    ) -> tuple[LatentState, LatentState]:
        video_state, video_tools = noise_video_state(
            output_shape=output_shape,
            noiser=noiser,
            conditionings=[TemporalRegionMask(start_time=0.0, end_time=0.0, fps=output_shape.fps)],
            components=self.pipeline_components,
            dtype=self.dtype,
            device=self.device,
            initial_latent=initial_video_latent,
        )
        audio_state, audio_tools = noise_audio_state(
            output_shape=output_shape,
            noiser=noiser,
            conditionings=audio_conditionings,
            components=self.pipeline_components,
            dtype=self.dtype,
            device=self.device,
            initial_latent=initial_audio_latent,
        )

        ref_audio_tokens = 0
        if ref_audio_latent is not None:
            ref_audio_patchified = self.pipeline_components.audio_patchifier.patchify(ref_audio_latent)
            ref_audio_tokens = ref_audio_patchified.shape[1]
            ref_audio_mask = torch.zeros(
                ref_audio_patchified.shape[0],
                ref_audio_tokens,
                1,
                device=self.device,
                dtype=torch.float32,
            )
            ref_audio_time_in_sec = self.pipeline_components.audio_patchifier._get_audio_latent_time_in_sec(
                ref_audio_tokens,
                ref_audio_tokens + 1,
                dtype=audio_state.positions.dtype,
                device=self.device,
            ).item()
            target_audio_tokens = audio_state.positions.shape[2]
            combined_audio_len = ref_audio_tokens + target_audio_tokens
            combined_audio_shape = AudioLatentShape(
                batch=ref_audio_latent.shape[0],
                channels=ref_audio_latent.shape[1],
                frames=combined_audio_len,
                mel_bins=ref_audio_latent.shape[3],
            )
            combined_audio_positions = self.pipeline_components.audio_patchifier.get_patch_grid_bounds(
                output_shape=combined_audio_shape,
                device=self.device,
            ).to(audio_state.positions.dtype)
            combined_audio_positions[:, 0, :, :] -= ref_audio_time_in_sec
            audio_state = replace(
                audio_state,
                latent=torch.cat([ref_audio_patchified, audio_state.latent], dim=1),
                denoise_mask=torch.cat([ref_audio_mask, audio_state.denoise_mask], dim=1),
                positions=combined_audio_positions,
                clean_latent=torch.cat([ref_audio_patchified, audio_state.clean_latent], dim=1),
            )

        video_state, audio_state = denoising_loop_fn(sigmas, video_state, audio_state, stepper)

        if ref_audio_tokens > 0:
            audio_state = replace(
                audio_state,
                latent=audio_state.latent[:, ref_audio_tokens:],
                denoise_mask=audio_state.denoise_mask[:, ref_audio_tokens:],
                positions=audio_state.positions[:, :, ref_audio_tokens:],
                clean_latent=audio_state.clean_latent[:, ref_audio_tokens:],
            )

        video_state = video_tools.clear_conditioning(video_state)
        video_state = video_tools.unpatchify(video_state)
        audio_state = audio_tools.clear_conditioning(audio_state)
        audio_state = audio_tools.unpatchify(audio_state)
        return video_state, audio_state

    @torch.inference_mode()
    def generate_segment(
        self,
        clip_frames: torch.Tensor,
        prompt: str,
        negative_prompt: str,
        seed: int,
        frame_rate: float,
        num_inference_steps: int,
        video_guider_params: MultiModalGuiderParams,
        audio_guider_params: MultiModalGuiderParams,
        prev_generated_audio: Audio | None = None,
        ref_audio_latent: torch.Tensor | None = None,
        enhance_prompt: bool = False,
    ) -> tuple[Audio, torch.Tensor]:
        height, width = clip_frames.shape[1], clip_frames.shape[2]
        output_shape = VideoPixelShape(
            batch=1,
            frames=clip_frames.shape[0],
            width=width,
            height=height,
            fps=frame_rate,
        )
        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()

        contexts = encode_prompts(
            [prompt, negative_prompt],
            self.model_ledger,
            enhance_first_prompt=enhance_prompt,
            enhance_prompt_seed=seed,
        )
        v_context_p, a_context_p = contexts[0].video_encoding, contexts[0].audio_encoding
        v_context_n, a_context_n = contexts[1].video_encoding, contexts[1].audio_encoding

        video_encoder = self.model_ledger.video_encoder()
        initial_video_latent = _encode_video_clip(
            video_encoder=video_encoder,
            clip_frames=clip_frames,
            output_shape=output_shape,
            dtype=self.dtype,
            device=self.device,
        )
        del video_encoder
        cleanup_memory()

        first_input_audio = self._build_first_input_audio(prev_generated_audio)
        audio_conditionings = [
            AudioConditionByLatentIndex(latent=first_input_audio, strength=1.0, latent_idx=0)
        ]

        transformer = self.model_ledger.transformer()
        sigmas = LTX2Scheduler().execute(steps=num_inference_steps).to(dtype=torch.float32, device=self.device)
        video_guider = MultiModalGuider(params=video_guider_params, negative_context=v_context_n)
        audio_guider = MultiModalGuider(params=audio_guider_params, negative_context=a_context_n)
        denoise_fn = multi_modal_guider_denoising_func(
            video_guider,
            audio_guider,
            v_context=v_context_p,
            a_context=a_context_p,
            transformer=transformer,
        )

        def denoising_loop(
            sigmas: torch.Tensor,
            video_state: LatentState,
            audio_state: LatentState,
            stepper: DiffusionStepProtocol,
        ) -> tuple[LatentState, LatentState]:
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=denoise_fn,
            )

        _, audio_state = self._denoise_audio_only_with_ref(
            output_shape=output_shape,
            noiser=noiser,
            sigmas=sigmas,
            stepper=stepper,
            denoising_loop_fn=denoising_loop,
            initial_video_latent=initial_video_latent,
            audio_conditionings=audio_conditionings,
            ref_audio_latent=ref_audio_latent,
        )

        torch.cuda.synchronize()
        del transformer
        cleanup_memory()

        raw_audio_latent = audio_state.latent.clone()
        decoded_audio = vae_decode_audio(
            audio_state.latent,
            self.model_ledger.audio_decoder(),
            self.model_ledger.vocoder(),
        )
        return decoded_audio, raw_audio_latent

    @torch.inference_mode()
    def __call__(  # noqa: PLR0913
        self,
        video_path: str,
        output_dir: Path,
        prompt: str,
        negative_prompt: str,
        seed: int,
        num_frames: int,
        num_inference_steps: int,
        video_guider_params: MultiModalGuiderParams,
        audio_guider_params: MultiModalGuiderParams,
        overlap_frames: int = 8,
        enhance_prompt: bool = False,
        audios: list[tuple[str, int, float]] = [],
    ) -> None:
        standardized_video, fps, width, height = _load_standardized_video(video_path)
        total_frames = standardized_video.shape[0]
        assert_resolution(height=height, width=width, is_two_stage=False)
        if (num_frames - 1) % 8 != 0:
            raise ValueError(f"num_frames must satisfy 8k+1, got {num_frames}")
        if overlap_frames < 0 or overlap_frames >= num_frames:
            raise ValueError("overlap_frames must be in [0, num_frames)")

        stride_frames = num_frames - overlap_frames
        num_segments = max(1, math.ceil(max(total_frames - num_frames, 0) / stride_frames) + 1)
        output_dir.mkdir(parents=True, exist_ok=True)

        ref_audio_latent: torch.Tensor | None = None
        if audios:
            spectrogram = load_audio_conditioning(
                audio_path=audios[0][0],
                audio_processor=self.audio_processor,
                dtype=self.dtype,
                device=self.device,
            )
            encoded_audio = self.audio_encoder(spectrogram)
            ref_audio_latent = _extract_ref_audio_latent(encoded_audio, ref_audio_len=8 * 8)

        merged_audio: Audio | None = None
        prev_generated_audio: Audio | None = None
        for segment_index in range(num_segments):
            start_frame = segment_index * stride_frames
            padded_clip, actual_clip, actual_frames = _decode_video_segment(standardized_video, start_frame, num_frames)
            decoded_audio, raw_audio_latent = self.generate_segment(
                clip_frames=padded_clip,
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed + segment_index,
                frame_rate=fps,
                num_inference_steps=num_inference_steps,
                video_guider_params=video_guider_params,
                audio_guider_params=audio_guider_params,
                prev_generated_audio=prev_generated_audio,
                ref_audio_latent=ref_audio_latent,
                enhance_prompt=enhance_prompt,
            )

            actual_duration = actual_frames / fps
            segment_audio = _trim_audio_to_duration(decoded_audio, actual_duration)
            if not audios and ref_audio_latent is None:
                ref_audio_latent = _extract_ref_audio_latent(raw_audio_latent, ref_audio_len=8 * 8)

            handoff_duration = min(actual_duration, stride_frames / fps)
            prev_generated_audio = _trim_audio_to_duration(segment_audio, handoff_duration)
            merged_audio = _merge_audio_segments(
                existing=merged_audio,
                new_segment=segment_audio,
                overlap_duration=min(overlap_frames / fps, actual_duration),
            )

            segment_output_path = output_dir / f"{segment_index:03d}.mp4"
            encode_video(
                video=actual_clip,
                fps=int(round(fps)),
                audio=segment_audio,
                output_path=str(segment_output_path),
                video_chunks_number=1,
            )

        if merged_audio is None:
            raise ValueError("No audio segments were generated")

        total_duration = total_frames / fps
        merged_audio = _trim_audio_to_duration(merged_audio, total_duration)
        _write_audio_wav(merged_audio, output_dir / "generated_audio.wav")
        encode_video(
            video=standardized_video,
            fps=int(round(fps)),
            audio=merged_audio,
            output_path=str(output_dir / "generated_video.mp4"),
            video_chunks_number=1,
        )


@torch.inference_mode()
def main() -> None:
    logging.getLogger().setLevel(logging.INFO)
    parser = default_1_stage_arg_parser()
    parser.add_argument("--video-path", type=str, required=True, help="Path to the source video.")
    parser.add_argument(
        "--overlap-frames",
        type=int,
        default=8,
        help="Number of overlapping frames between consecutive V2A segments.",
    )
    args = parser.parse_args()

    pipeline = V2ALongPipeline(
        checkpoint_path=args.checkpoint_path,
        gemma_root=args.gemma_root,
        loras=tuple(args.lora) if args.lora else (),
        quantization=args.quantization,
    )
    pipeline(
        video_path=args.video_path,
        output_dir=Path(args.output_path),
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        num_frames=args.num_frames,
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
        overlap_frames=args.overlap_frames,
        enhance_prompt=args.enhance_prompt,
        audios=args.audios,
    )


if __name__ == "__main__":
    main()
