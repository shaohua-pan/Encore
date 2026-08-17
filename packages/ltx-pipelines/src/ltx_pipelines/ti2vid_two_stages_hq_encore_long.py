"""Long-form Encore inference driven by a single user image + reference audio.

Repeatedly calls :class:`TI2VidTwoStagesHQEncorePipeline` for multiple contiguous
segments. Between segments we carry:

* last decoded frame   -> next segment's conditioning image (frame_idx=1)
* decoded audio tail   -> next segment's first-input-audio (lip-sync continuity)
* cached ref_audio     -> global speaker timbre/style across all segments

Per-segment ``seg_{j}.mp4`` files are written and finally concatenated with
ffmpeg into ``full.mp4``.

Per-segment prompts come from the JSON array passed to ``--prompts``. The array
length determines the number of segments. Without ``--prompts``, ``--prompt`` is
repeated ``--num-segments`` times.
"""

import argparse
import json
import logging
import os
import subprocess
from pathlib import Path

import torch
from PIL import Image

from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_pipelines.ti2vid_two_stages_hq_encore import TI2VidTwoStagesHQEncorePipeline
from ltx_pipelines.utils.args import ImageConditioningInput, hq_2_stage_arg_parser
from ltx_pipelines.utils.constants import LTX_2_3_HQ_PARAMS
from ltx_pipelines.utils.media_io import encode_video


def _extend_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--num-segments",
        type=int,
        default=6,
        help="Number of segments when --prompts is not provided (default: 6).",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default=None,
        help=(
            "Optional JSON array of per-segment prompt strings. "
            "Its length determines the number of segments and overrides --prompt and --num-segments."
        ),
    )
    parser.add_argument(
        "--concat-output",
        type=str,
        default="full.mp4",
        help="Name of the concatenated output file written inside --output-path (default: full.mp4).",
    )


def _load_segment_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompts is None:
        return [args.prompt] * args.num_segments

    try:
        prompts = json.loads(args.prompts)
    except json.JSONDecodeError as error:
        raise ValueError("--prompts must be a valid JSON array of strings.") from error

    if not isinstance(prompts, list) or not prompts:
        raise ValueError("--prompts must be a non-empty JSON array of strings.")
    if not all(isinstance(prompt, str) and prompt.strip() for prompt in prompts):
        raise ValueError("Every item in --prompts must be a non-empty string.")
    return prompts


def _ffmpeg_concat(segment_paths: list[str], output_path: str) -> None:
    list_file = str(Path(output_path).with_suffix(".concat.txt"))
    with open(list_file, "w", encoding="utf-8") as f:
        for p in segment_paths:
            f.write(f"file '{os.path.abspath(p)}'\n")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", list_file,
        "-c", "copy", output_path,
    ]
    subprocess.run(cmd, check=True)
    os.remove(list_file)


@torch.inference_mode()
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = hq_2_stage_arg_parser(params=LTX_2_3_HQ_PARAMS)
    _extend_parser(parser)
    args = parser.parse_args()

    if not args.images:
        raise ValueError("--image PATH FRAME_IDX STRENGTH is required.")
    init_image: ImageConditioningInput = args.images[0]

    segment_prompts = _load_segment_prompts(args)
    args.num_segments = len(segment_prompts)

    os.makedirs(args.output_path, exist_ok=True)

    target_w, target_h = args.width, args.height

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

    # First segment: condition only on the user image. For subsequent segments we
    # replace the second conditioning (frame_idx=1) with the last frame of the
    # previous segment for temporal continuity.
    images = [
        ImageConditioningInput(path=init_image.path, frame_idx=0, strength=1.0, crf=0),
        ImageConditioningInput(path=init_image.path, frame_idx=1, strength=1.0, crf=27),
    ]

    prev_decoded_audio: torch.Tensor | None = None
    cached_ref_audio_latent: torch.Tensor | None = None
    ref_audio_len = 8 * 8

    segment_paths: list[str] = []

    for j in range(args.num_segments):
        logging.info("[long] segment %d/%d starting", j + 1, args.num_segments)
        video, audio, raw_audio_latent = pipeline(
            prompt=segment_prompts[j],
            negative_prompt=args.negative_prompt,
            seed=args.seed + j,
            height=target_h,
            width=target_w,
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
            prev_decoded_audio=prev_decoded_audio,
            ref_position_offset=args.ref_position_offset,
            ref_audio_latent_override=cached_ref_audio_latent,
        )

        # Cache ref_audio from the first segment when user did NOT pass --audio.
        # When --audio is given, pipeline derives ref_audio from it every call and
        # this cache is ignored (see TI2VidTwoStagesHQPipeline.__call__).
        if not args.audios and cached_ref_audio_latent is None:
            min_len = min(ref_audio_len, raw_audio_latent.shape[2])
            cached_ref_audio_latent = torch.zeros(
                1, raw_audio_latent.shape[1], ref_audio_len, raw_audio_latent.shape[3],
                dtype=raw_audio_latent.dtype, device=raw_audio_latent.device,
            )
            cached_ref_audio_latent[:, :, :min_len, :] = raw_audio_latent[:, :, :min_len, :]

        prev_decoded_audio = audio.waveform

        # Save final frame -> next segment's frame_idx=1 conditioning image.
        all_video_chunks = list(video)
        final_frame = all_video_chunks[-1][-1].cpu().numpy()
        image_path = os.path.join(args.output_path, f"seg_{j}_last.png")
        Image.fromarray(final_frame).save(image_path)
        images[1] = ImageConditioningInput(path=image_path, frame_idx=1, strength=1.0, crf=0)

        seg_mp4 = os.path.join(args.output_path, f"seg_{j}.mp4")
        encode_video(
            video=iter(all_video_chunks),
            fps=args.frame_rate,
            audio=audio,
            output_path=seg_mp4,
            video_chunks_number=video_chunks_number,
        )
        segment_paths.append(seg_mp4)
        logging.info("[long] segment %d/%d -> %s", j + 1, args.num_segments, seg_mp4)

    concat_path = os.path.join(args.output_path, args.concat_output)
    _ffmpeg_concat(segment_paths, concat_path)
    logging.info("[long] concatenated %d segments -> %s", len(segment_paths), concat_path)


if __name__ == "__main__":
    main()
