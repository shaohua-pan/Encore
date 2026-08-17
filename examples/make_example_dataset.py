#!/usr/bin/env python3
# ruff: noqa: T201  - this is a CLI helper; printing progress is the point
"""Generate a tiny synthetic dataset that exercises the whole training path.

Our training data cannot be released, so this script fabricates a handful of
clips with the exact same schema. It is meant for smoke-testing preprocessing
and training (does the config load? do latents get written? does a step run?),
not for producing a useful model.

Each generated clip is a short synthetic video with an audio track, written
alongside a JSONL metadata file in the layout the preprocessing script expects.

Requires ffmpeg on PATH.

Usage:
    python examples/make_example_dataset.py
    python examples/make_example_dataset.py --num-clips 8 --out-dir /tmp/toy
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

# The resolution buckets used for training are "WxHxF" triples; the clips below
# match the 960x544x121 bucket at 24 fps.
WIDTH = 960
HEIGHT = 544
FPS = 24
NUM_FRAMES = 121

SCENES = [
    (
        "A woman in a light shirt stands in a bright minimalist room and speaks to the camera. "
        "She keeps gesturing with both hands, opening her palms and pointing forward, and her head "
        "turns slowly left and right before returning to the front. The camera stays still.",
        "a woman in her late twenties, medium pace, clear articulation, warm and enthusiastic, "
        "clean recording with no background music",
        "This is an example clip used to smoke test the training pipeline.",
    ),
    (
        "A man in a dark jacket sits at a wooden desk in front of a bookshelf and talks to the camera. "
        "He nods slightly, leans forward and straightens up again, and moves his hands above the desk "
        "while he speaks. The camera stays still.",
        "an adult male, medium-slow pace, calm and even tone, slight reverberation from a small room",
        "Every field in this record mirrors the schema of the real training data.",
    ),
    (
        "A person in a bright hoodie walks slowly along a corridor towards the camera while speaking, "
        "the background sliding past behind them. Their shoulders and arms move continuously with each step.",
        "a young adult voice, medium-fast pace, energetic, faint ambient noise in the background",
        "Replace these synthetic clips with your own videos before training for real.",
    ),
]


def _relative_if_possible(path: Path) -> str:
    """Prefer a CWD-relative path so the metadata file stays portable."""
    try:
        return str(path.resolve().relative_to(Path.cwd()))
    except ValueError:
        return str(path.resolve())


def _require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found on PATH; install it to generate the example clips.")


def _render_clip(path: Path, index: int) -> None:
    """Render one synthetic clip with a video and an audio track."""
    duration = (NUM_FRAMES + FPS) / FPS  # a little longer than the frame window
    # A moving test pattern stands in for the video, a sine tone for the audio.
    # Both vary per clip so the latents are not identical across samples.
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size={WIDTH}x{HEIGHT}:rate={FPS}:duration={duration}",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency={220 + 55 * index}:sample_rate=48000:duration={duration}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        str(path),
    ]
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).parent / "data",
        help="Directory to write clips/ and example_dataset.jsonl into.",
    )
    parser.add_argument("--num-clips", type=int, default=3, help="How many clips to generate.")
    args = parser.parse_args()

    _require_ffmpeg()

    clips_dir = args.out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.out_dir / "example_dataset.jsonl"

    records = []
    for i in range(args.num_clips):
        scene, voice, speech = SCENES[i % len(SCENES)]
        clip_path = clips_dir / f"example_{i:04d}.mp4"
        _render_clip(clip_path, i)
        clip_ref = _relative_if_possible(clip_path)

        caption_audio = (
            f'Speech in the audio: "{speech}"\n'
            f"Description of the voice and background: {voice}."
        )
        records.append(
            {
                # Path to the clip. Relative paths are resolved against the CWD
                # of the preprocessing script; --video-prefix-path is stripped
                # from this path to derive where the latents are written.
                "video_path": clip_ref,
                # The audio is read from the same file here, which is the usual
                # case for talking-head data.
                "audio_path": clip_ref,
                # Visual-only caption.
                "caption_video": scene,
                # Audio caption: transcript plus a description of the voice.
                "caption_audio": caption_audio,
                # Transcript only.
                "caption_audio_speech": speech,
                # Voice / background description only.
                "caption_audio_bg": voice,
                # The caption actually fed to the text encoder
                # (--caption-column caption_all). Joint audio-video training
                # needs the visual description, the transcript and the voice
                # description in a single string.
                "caption_all": f"{scene}\n{caption_audio}",
                # Which frames of the source clip form the training window.
                # Length must equal the frame count of the resolution bucket
                # (121 here). The first index also defines the audio offset:
                # audio_start = frame_idx[0] / fps.
                "frame_idx": list(range(NUM_FRAMES)),
                "fps": float(FPS),
            }
        )

    with jsonl_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} clips to {clips_dir}")
    print(f"Wrote metadata to {jsonl_path}")
    print()
    print("Next step — precompute latents and text embeddings:")
    print(
        f"  VIDEO_PREFIX_PATH={clips_dir} NUM_GPUS=1 "
        f"bash scripts/preprocess/process_dataset.sh {jsonl_path}"
    )


if __name__ == "__main__":
    main()
