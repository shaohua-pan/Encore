import logging
import math
from pathlib import Path

import torch
from PIL import Image

from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_core.model.video_vae import TilingConfig
from ltx_core.types import Audio
from ltx_pipelines.a2vid_two_stages_hq_encore import (
    A2VidTwoStagesHQEncorePipeline,
    _get_audio_duration,
    _pad_audio_to_duration,
    _squeeze_audio_batch,
    _trim_audio_to_duration,
    _trim_video_chunks,
)
from ltx_pipelines.utils.args import ImageConditioningInput, hq_2_stage_arg_parser
from ltx_pipelines.utils.constants import LTX_2_3_HQ_PARAMS
from ltx_pipelines.utils.media_io import decode_audio_from_file, encode_video


def _drop_leading_video_frames(video_chunks: list[torch.Tensor], drop_frames: int) -> list[torch.Tensor]:
    if drop_frames <= 0:
        return video_chunks

    remaining = drop_frames
    kept_chunks: list[torch.Tensor] = []
    for chunk in video_chunks:
        if remaining >= chunk.shape[0]:
            remaining -= chunk.shape[0]
            continue
        if remaining > 0:
            kept_chunks.append(chunk[remaining:])
            remaining = 0
        else:
            kept_chunks.append(chunk)
    return kept_chunks


def _drop_audio_start(audio: Audio, start_duration: float) -> Audio:
    if start_duration <= 0:
        return audio

    waveform = audio.waveform
    start_samples = min(round(start_duration * audio.sampling_rate), waveform.shape[-1])
    return Audio(waveform=waveform[..., start_samples:], sampling_rate=audio.sampling_rate)


def _tail_video_frames(video_chunks: list[torch.Tensor], frame_count: int) -> list[torch.Tensor]:
    if frame_count <= 0:
        return []

    remaining = frame_count
    tail_chunks: list[torch.Tensor] = []
    for chunk in reversed(video_chunks):
        if remaining <= 0:
            break
        take_frames = min(remaining, chunk.shape[0])
        tail_chunks.insert(0, chunk[-take_frames:])
        remaining -= take_frames

    if not tail_chunks:
        return []
    return list(torch.cat(tail_chunks, dim=0))


def _save_handoff_conditionings(
    video_chunks: list[torch.Tensor],
    output_dir: Path,
    segment_index: int,
    frame_count: int,
    strength: float,
) -> list[ImageConditioningInput]:
    handoff_dir = output_dir / "handoff_frames" / f"{segment_index:03d}"
    handoff_dir.mkdir(parents=True, exist_ok=True)

    conditionings: list[ImageConditioningInput] = []
    for frame_idx, frame in enumerate(_tail_video_frames(video_chunks, frame_count)):
        image_path = handoff_dir / f"{frame_idx:02d}.png"
        Image.fromarray(frame.cpu().numpy()).save(image_path)
        conditionings.append(
            ImageConditioningInput(
                path=str(image_path),
                frame_idx=frame_idx,
                strength=strength,
                crf=0,
            )
        )
    return conditionings


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
    parser.add_argument(
        "--smooth-overlap-frames",
        type=int,
        default=12,
        help="Number of overlapped frames reused as multi-frame visual handoff for the next segment.",
    )
    parser.add_argument(
        "--smooth-conditioning-strength",
        type=float,
        default=1.0,
        help="Strength for multi-frame handoff image conditionings.",
    )
    parser.add_argument(
        "--save-overlap-debug-video",
        action="store_true",
        help="Also save each raw segment before dropping duplicated overlap frames.",
    )
    args = parser.parse_args()

    if not args.images:
        raise ValueError("Encore A2V smooth requires at least one --image conditioning input.")
    if args.smooth_overlap_frames < 1:
        raise ValueError("--smooth-overlap-frames must be greater than 0 for smooth handoff.")
    if args.smooth_overlap_frames >= args.num_frames:
        raise ValueError("--smooth-overlap-frames must be smaller than --num-frames.")

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
    handoff_frames = args.smooth_overlap_frames
    handoff_duration = (args.num_frames - handoff_frames) / args.frame_rate
    overlap_duration = handoff_frames / args.frame_rate
    total_duration = _get_audio_duration(args.a2v_audio)
    num_segments = max(1, math.ceil(max(total_duration - clip_duration, 0.0) / handoff_duration) + 1)

    images = list(args.images)
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

        if index + 1 < num_segments:
            images = _save_handoff_conditionings(
                all_video_chunks,
                output_dir,
                index,
                min(handoff_frames, target_frames),
                args.smooth_conditioning_strength,
            )

        if args.save_overlap_debug_video:
            encode_video(
                video=iter(all_video_chunks),
                fps=args.frame_rate,
                audio=output_audio,
                output_path=str(output_dir / f"{index:03d}_overlap_debug.mp4"),
                video_chunks_number=len(all_video_chunks),
            )

        drop_frames = handoff_frames if index > 0 else 0
        segment_video_chunks = _drop_leading_video_frames(all_video_chunks, drop_frames)
        segment_audio = _drop_audio_start(output_audio, overlap_duration if index > 0 else 0.0)
        if not segment_video_chunks:
            continue

        encode_video(
            video=iter(segment_video_chunks),
            fps=args.frame_rate,
            audio=segment_audio,
            output_path=str(output_dir / f"{index:03d}.mp4"),
            video_chunks_number=len(segment_video_chunks),
        )


if __name__ == "__main__":
    main()
