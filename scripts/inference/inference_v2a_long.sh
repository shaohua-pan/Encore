#!/usr/bin/env bash
# Video-to-audio: generate a soundtrack for an existing (silent) video.
#
# The input video is normalised inside the pipeline to 1920x1088 (landscape) or
# 1088x1920 (portrait) at 24 fps, then processed in overlapping windows so the
# audio stays coherent across the whole clip.
#
# Usage:
#   VIDEO=my_clip.mp4 PROMPT='...' bash scripts/inference/inference_v2a_long.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/env.sh"

VIDEO="${VIDEO:?set VIDEO to the input video file}"
PROMPT="${PROMPT:?set PROMPT to a description of the audio to generate}"
OUTPUT_PATH="${OUTPUT_PATH:-${OUTPUT_ROOT}/v2a_long}"

NUM_FRAMES="${NUM_FRAMES:-121}"
OVERLAP_FRAMES="${OVERLAP_FRAMES:-8}"

python -m ltx_pipelines.v2a_long \
    --checkpoint-path "${CHECKPOINT_PATH}" \
    --gemma-root "${GEMMA_ROOT}" \
    --lora "${ENCORE_LORA}" 1 \
    --video-path "${VIDEO}" \
    --prompt "${PROMPT}" \
    --num-frames "${NUM_FRAMES}" \
    --overlap-frames "${OVERLAP_FRAMES}" \
    --output-path "${OUTPUT_PATH}"

# Optional: --audio path/to/reference.wav 0 1.0 to pin timbre/style globally.
