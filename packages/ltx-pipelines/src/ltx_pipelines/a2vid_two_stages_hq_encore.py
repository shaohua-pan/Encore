import logging
import math
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import av
import torch
from PIL import Image

from ltx_core.components.diffusion_steps import Res2sDiffusionStep
from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.protocols import DiffusionStepProtocol
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.model.audio_vae import encode_audio as vae_encode_audio
from ltx_core.model.upsampler import upsample_video
from ltx_core.model.video_vae import TilingConfig
from ltx_core.model.video_vae import decode_video as vae_decode_video
from ltx_core.model.audio_vae.ops import AudioProcessor
from ltx_core.quantization import QuantizationPolicy
from ltx_core.tools import VideoLatentShape
from ltx_core.types import Audio, AudioLatentShape, LatentState, VideoPixelShape
from ltx_core.conditioning import AudioConditionByLatentIndex
from ltx_pipelines.utils import (
    ModelLedger,
    assert_resolution,
    cleanup_memory,
    combined_image_conditionings,
    denoise_video_only_encore,
    encode_prompts,
    get_device,
    multi_modal_guider_denoising_func,
    res2s_audio_video_denoising_loop,
    simple_denoising_func,
)
from ltx_pipelines.utils.args import ImageConditioningInput, hq_2_stage_arg_parser
from ltx_pipelines.utils.constants import LTX_2_3_HQ_PARAMS, STAGE_2_DISTILLED_SIGMA_VALUES
from ltx_pipelines.utils.media_io import decode_audio_from_file, encode_video, load_audio_conditioning
from ltx_pipelines.utils.types import PipelineComponents

device = get_device()


class A2VidTwoStagesHQEncorePipeline:
    """Two-stage Encore A2V pipeline with frozen target audio and long-video continuation."""

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
        if len(loras) == 2:
            all_loras_stage_1 = (loras[0], distilled_lora_stage_1, loras[1])
        else:
            all_loras_stage_1 = (distilled_lora_stage_1, loras[0])
        self.stage_1_model_ledger = ModelLedger(
            dtype=self.dtype,
            device=device,
            checkpoint_path=checkpoint_path,
            gemma_root_path=gemma_root,
            spatial_upsampler_path=spatial_upsampler_path,
            loras=all_loras_stage_1,
            quantization=quantization,
            routing_scale=routing_scale,
        )

        if len(loras) == 2:
            all_loras_stage_2 = (loras[0], distilled_lora_stage_2, loras[1])
        else:
            all_loras_stage_2 = (distilled_lora_stage_2, loras[0])
        self.stage_2_model_ledger = ModelLedger(
            dtype=self.dtype,
            device=device,
            checkpoint_path=checkpoint_path,
            gemma_root_path=gemma_root,
            spatial_upsampler_path=spatial_upsampler_path,
            loras=all_loras_stage_2,
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

    def _encode_target_audio(self, audio: Audio, num_frames: int, frame_rate: float) -> torch.Tensor:
        waveform = audio.waveform.to(device=self.device, dtype=torch.float32)
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)
        encoded_audio_latent = vae_encode_audio(
            Audio(waveform=waveform, sampling_rate=audio.sampling_rate),
            self.stage_1_model_ledger.audio_encoder(),
            None,
        )
        audio_shape = AudioLatentShape.from_duration(
            batch=1,
            duration=num_frames / frame_rate,
            channels=8,
            mel_bins=16,
        )
        expected_frames = audio_shape.frames
        actual_frames = encoded_audio_latent.shape[2]
        if actual_frames > expected_frames:
            encoded_audio_latent = encoded_audio_latent[:, :, :expected_frames, :]
        elif actual_frames < expected_frames:
            pad = torch.zeros(
                encoded_audio_latent.shape[0],
                encoded_audio_latent.shape[1],
                expected_frames - actual_frames,
                encoded_audio_latent.shape[3],
                dtype=encoded_audio_latent.dtype,
                device=encoded_audio_latent.device,
            )
            encoded_audio_latent = torch.cat([encoded_audio_latent, pad], dim=2)
        return encoded_audio_latent

    def _extract_ref_audio_latent(self, encoded_audio_latent: torch.Tensor, ref_audio_len: int) -> torch.Tensor:
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

    def _build_first_input_audio(self, prev_input_audio: Audio | None, audio_context_len: int = 2) -> torch.Tensor:
        if prev_input_audio is None:
            return torch.zeros(1, 8, audio_context_len, 16, dtype=self.dtype, device=self.device)

        waveform = prev_input_audio.waveform
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)

        tail_samples = 9 * self.audio_processor.mel_transform.hop_length
        tail_waveform = waveform[:, :, -tail_samples:].cpu()
        tail_audio = Audio(waveform=tail_waveform, sampling_rate=prev_input_audio.sampling_rate)
        tail_audio = self.audio_processor.resample_audio(tail_audio)

        min_samples = self.audio_processor.mel_transform.n_fft + 1
        if tail_audio.waveform.shape[-1] < min_samples:
            pad_size = min_samples - tail_audio.waveform.shape[-1]
            padded = torch.nn.functional.pad(tail_audio.waveform, (pad_size, 0))
            tail_audio = Audio(waveform=padded, sampling_rate=tail_audio.sampling_rate)

        tail_spectrogram = self.audio_processor.waveform_to_mel(tail_audio).to(device=self.device, dtype=self.dtype)
        tail_encoded = self.audio_encoder(tail_spectrogram)
        return tail_encoded[:, :, -audio_context_len:, :]

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
        images: list[ImageConditioningInput],
        driving_audio: Audio,
        tiling_config: TilingConfig | None = None,
        enhance_prompt: bool = False,
        audios: list[tuple[str, int, float]] = [],
        prev_input_audio: Audio | None = None,
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
            enhance_prompt_image=images[0].path if len(images) > 0 else None,
            enhance_prompt_seed=seed,
        )
        v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
        v_context_n, _ = ctx_n.video_encoding, ctx_n.audio_encoding

        encoded_audio_latent = self._encode_target_audio(driving_audio, num_frames=num_frames, frame_rate=frame_rate)

        stage_1_output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames + 8,
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

        ref_audio_len = 8 * 8
        if audios:
            audio_path = audios[0][0]
            spectrogram = load_audio_conditioning(
                audio_path=audio_path,
                audio_processor=self.audio_processor,
                dtype=dtype,
                device=self.device,
            )
            ref_audio_latent = self._extract_ref_audio_latent(self.audio_encoder(spectrogram), ref_audio_len)
        elif ref_audio_latent_override is not None:
            ref_audio_latent = ref_audio_latent_override
        else:
            ref_audio_latent = self._extract_ref_audio_latent(encoded_audio_latent, ref_audio_len)

        stage_1_audio_conditionings = [
            AudioConditionByLatentIndex(
                latent=self._build_first_input_audio(prev_input_audio),
                strength=1.0,
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
            sigmas: torch.Tensor,
            video_state: LatentState,
            audio_state: LatentState,
            stepper: DiffusionStepProtocol,
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
                        params=MultiModalGuiderParams(),
                    ),
                    v_context=v_context_p,
                    a_context=a_context_p,
                    transformer=transformer,
                ),
            )

        video_state, audio_state = denoise_video_only_encore(
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
            initial_audio_latent=encoded_audio_latent,
            ref_audio_latent=ref_audio_latent,
            ref_position_offset=ref_position_offset,
        )

        torch.cuda.synchronize()
        del transformer
        cleanup_memory()

        video_encoder = self.stage_1_model_ledger.video_encoder()
        upscaled_video_latent = upsample_video(
            latent=video_state.latent[:1],
            video_encoder=video_encoder,
            upsampler=self.stage_2_model_ledger.spatial_upsampler(),
        )

        stage_2_output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames + 8,
            width=width,
            height=height,
            fps=frame_rate,
        )
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
            sigmas: torch.Tensor,
            video_state: LatentState,
            audio_state: LatentState,
            stepper: DiffusionStepProtocol,
        ) -> tuple[LatentState, LatentState]:
            return res2s_audio_video_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=simple_denoising_func(
                    video_context=v_context_p,
                    audio_context=a_context_p,
                    transformer=transformer,
                ),
            )

        video_state, audio_state = denoise_video_only_encore(
            output_shape=stage_2_output_shape,
            conditionings=stage_2_conditionings,
            noiser=noiser,
            sigmas=distilled_sigmas,
            stepper=stepper,
            denoising_loop_fn=second_stage_denoising_loop,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            audio_conditionings=stage_1_audio_conditionings,
            noise_scale=distilled_sigmas[0],
            initial_video_latent=upscaled_video_latent,
            initial_audio_latent=audio_state.latent,
            ref_audio_latent=ref_audio_latent,
            ref_position_offset=ref_position_offset,
        )

        torch.cuda.synchronize()
        del transformer
        cleanup_memory()

        video_state = replace(video_state, latent=video_state.latent[:, :, 1:, :, :])
        raw_audio_latent = audio_state.latent.clone()

        decoded_video = vae_decode_video(
            video_state.latent,
            self.stage_2_model_ledger.video_decoder(),
            tiling_config,
            generator,
        )
        original_audio = _squeeze_audio_batch(driving_audio)
        return decoded_video, original_audio, raw_audio_latent


def _get_audio_duration(audio_path: str) -> float:
    container = av.open(audio_path)
    try:
        try:
            audio_stream = next(stream for stream in container.streams if stream.type == "audio")
        except StopIteration as exc:
            raise ValueError(f"No audio stream found in {audio_path}") from exc

        if audio_stream.duration is not None and audio_stream.time_base is not None:
            return float(audio_stream.duration * audio_stream.time_base)
        if container.duration is not None:
            return float(container.duration / av.time_base)

        total_samples = 0
        sample_rate = audio_stream.rate
        for frame in container.decode(audio=audio_stream.index):
            total_samples += frame.samples
        return total_samples / sample_rate
    finally:
        container.close()


def _squeeze_audio_batch(audio: Audio) -> Audio:
    waveform = audio.waveform
    if waveform.dim() == 3 and waveform.shape[0] == 1:
        waveform = waveform.squeeze(0)
    return Audio(waveform=waveform, sampling_rate=audio.sampling_rate)


def _pad_audio_to_duration(audio: Audio, target_duration: float) -> Audio:
    waveform = audio.waveform
    if waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)
    target_samples = round(target_duration * audio.sampling_rate)
    current_samples = waveform.shape[-1]
    if current_samples >= target_samples:
        return Audio(waveform=waveform[..., :target_samples], sampling_rate=audio.sampling_rate)

    padded = torch.nn.functional.pad(waveform, (0, target_samples - current_samples))
    return Audio(waveform=padded, sampling_rate=audio.sampling_rate)


def _trim_video_chunks(video_chunks: list[torch.Tensor], target_frames: int) -> list[torch.Tensor]:
    remaining_frames = target_frames
    trimmed_chunks: list[torch.Tensor] = []
    for chunk in video_chunks:
        if remaining_frames <= 0:
            break
        if chunk.shape[0] <= remaining_frames:
            trimmed_chunks.append(chunk)
            remaining_frames -= chunk.shape[0]
        else:
            trimmed_chunks.append(chunk[:remaining_frames])
            remaining_frames = 0
    return trimmed_chunks


def _trim_audio_to_duration(audio: Audio, target_duration: float) -> Audio:
    waveform = audio.waveform
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)
    target_samples = round(target_duration * audio.sampling_rate)
    trimmed = waveform[..., :target_samples]
    return Audio(waveform=trimmed, sampling_rate=audio.sampling_rate)


def _select_handoff_frame(video_chunks: list[torch.Tensor], overlap_frames: int = 8) -> torch.Tensor:
    frames_from_end = overlap_frames + 1
    remaining = frames_from_end
    for chunk in reversed(video_chunks):
        if chunk.shape[0] >= remaining:
            return chunk[-remaining]
        remaining -= chunk.shape[0]
    return video_chunks[0][0]


@torch.inference_mode()
def main() -> None:
    logging.getLogger().setLevel(logging.INFO)
    parser = hq_2_stage_arg_parser(params=LTX_2_3_HQ_PARAMS)
    parser.add_argument(
        "--a2v-audio",
        type=str,
        required=True,
        help="Path to the long driving audio used for A2V generation.",
    )
    args = parser.parse_args()

    if not args.images:
        raise ValueError("Encore A2V requires at least one --image conditioning input.")

    pipeline = A2VidTwoStagesHQEncorePipeline(
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
    clip_duration = args.num_frames / args.frame_rate
    handoff_frames = 0
    handoff_duration = (args.num_frames - handoff_frames) / args.frame_rate
    total_duration = _get_audio_duration(args.a2v_audio)
    num_segments = max(1, math.ceil(max(total_duration - clip_duration, 0.0) / handoff_duration) + 1)

    images = list(args.images)
    if len(images) == 1:
        images.append(
            ImageConditioningInput(
                path=images[0].path,
                frame_idx=1,
                strength=images[0].strength,
                crf=0,
            )
        )

    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    prev_input_audio: Audio | None = None
    cached_ref_audio_latent: torch.Tensor | None = None
    ref_audio_len = 8 * 8
    for index in range(num_segments):
        clip_start = index * handoff_duration
        remaining_duration = max(total_duration - clip_start, 0.0)
        actual_duration = min(clip_duration, remaining_duration)
        if actual_duration <= 0:
            break

        decoded_audio = decode_audio_from_file(
            args.a2v_audio,
            pipeline.device,
            start_time=clip_start,
            max_duration=actual_duration,
        )
        if decoded_audio is None:
            raise ValueError(f"Failed to decode driving audio from {args.a2v_audio}")

        padded_audio = _pad_audio_to_duration(decoded_audio, clip_duration)
        video, _, raw_audio_latent = pipeline(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            seed=args.seed + index,
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
            images=images,
            driving_audio=padded_audio,
            tiling_config=tiling_config,
            enhance_prompt=args.enhance_prompt,
            audios=args.audios,
            prev_input_audio=prev_input_audio,
            ref_position_offset=args.ref_position_offset,
            ref_audio_latent_override=cached_ref_audio_latent,
        )

        if not args.audios and cached_ref_audio_latent is None:
            min_len = min(ref_audio_len, raw_audio_latent.shape[2])
            cached_ref_audio_latent = torch.zeros(
                1,
                raw_audio_latent.shape[1],
                ref_audio_len,
                raw_audio_latent.shape[3],
                dtype=raw_audio_latent.dtype,
                device=raw_audio_latent.device,
            )
            cached_ref_audio_latent[:, :, :min_len, :] = raw_audio_latent[:, :, :min_len, :]

        output_audio = _squeeze_audio_batch(decoded_audio)
        prev_input_audio = _squeeze_audio_batch(_trim_audio_to_duration(output_audio, min(actual_duration, handoff_duration)))

        all_video_chunks = list(video)
        target_frames = min(args.num_frames, max(1, round(actual_duration * args.frame_rate)))
        if target_frames < args.num_frames:
            all_video_chunks = _trim_video_chunks(all_video_chunks, target_frames)

        final_frame = _select_handoff_frame(all_video_chunks, overlap_frames=handoff_frames).cpu().numpy()
        image_path = output_dir / f"{index:03d}.png"
        Image.fromarray(final_frame).save(image_path)
        images[1] = ImageConditioningInput(
            path=str(image_path),
            frame_idx=1,
            strength=1.0,
            crf=0,
        )

        encode_video(
            video=iter(all_video_chunks),
            fps=args.frame_rate,
            audio=output_audio,
            output_path=str(output_dir / f"{index:03d}.mp4"),
            video_chunks_number=len(all_video_chunks),
        )


if __name__ == "__main__":
    main()
