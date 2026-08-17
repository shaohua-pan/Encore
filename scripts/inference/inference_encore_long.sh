#!/usr/bin/env bash
# Long-form generation: N contiguous segments from one image, concatenated.
#
# Between segments the pipeline carries over
#   * the last decoded frame  -> next segment's continuation conditioning
#   * the decoded audio tail  -> next segment's audio continuation (lip sync)
#   * a cached reference audio -> a globally consistent speaker timbre
#
# Per-segment mp4s plus the concatenated full.mp4 are written to OUTPUT_PATH.
#
# Usage:
#   REF_IMAGE=my.png REF_AUDIO=voice.wav \
#   PROMPTS='["prompt for segment 1", "prompt for segment 2"]' \
#       bash scripts/inference/inference_encore_long.sh
#
# REF_AUDIO is optional; omit it or set it to "none" to use generated audio.
# PROMPTS must be a JSON array of non-empty strings. Its length determines the
# number of segments. Without PROMPTS, PROMPT is reused NUM_SEGMENTS times.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/env.sh"

REF_IMAGE="${REF_IMAGE:?set REF_IMAGE to your own first-frame image (png/jpg)}"
REF_AUDIO="${REF_AUDIO:-}"
PROMPT="${PROMPT:-A woman looks into the camera and speaks naturally, her head turning slightly, her expression focused. The camera stays still.}"

NUM_SEGMENTS="${NUM_SEGMENTS:-6}"
NUM_FRAMES="${NUM_FRAMES:-121}"
HEIGHT="${HEIGHT:-1088}"
WIDTH="${WIDTH:-1920}"
ROUTING_SCALE="${ROUTING_SCALE:-0.5}"
OUTPUT_PATH="${OUTPUT_PATH:-${OUTPUT_ROOT}/encore_long_$(basename "${REF_IMAGE%.*}")_rscale_${ROUTING_SCALE}}"

if [ ! -f "${REF_IMAGE}" ]; then
    echo "ERROR: file not found: ${REF_IMAGE}" >&2
    exit 1
fi

AUDIO_ARGS=()
if [ -n "${REF_AUDIO}" ] && [ "${REF_AUDIO}" != "none" ]; then
    if [ ! -f "${REF_AUDIO}" ]; then
        echo "ERROR: file not found: ${REF_AUDIO}" >&2
        exit 1
    fi
    AUDIO_ARGS=(--audio "${REF_AUDIO}" 0 1.0)
fi

PROMPT_ARGS=(--num-segments "${NUM_SEGMENTS}")
if [ -n "${PROMPTS:-}" ]; then
    PROMPT_ARGS=(--prompts "${PROMPTS}")
fi

python -m ltx_pipelines.ti2vid_two_stages_hq_encore_long \
    --checkpoint-path "${CHECKPOINT_PATH}" \
    --distilled-lora "${DISTILLED_LORA}" \
    --spatial-upsampler-path "${SPATIAL_UPSAMPLER_PATH}" \
    --gemma-root "${GEMMA_ROOT}" \
    --lora "${ENCORE_LORA}" 1 \
    --prompt "${PROMPT}" \
    --image "${REF_IMAGE}" 0 1.0 \
    "${AUDIO_ARGS[@]}" \
    --num-frames "${NUM_FRAMES}" \
    --height "${HEIGHT}" \
    --width "${WIDTH}" \
    --output-path "${OUTPUT_PATH}" \
    --routing-scale "${ROUTING_SCALE}" \
    "${PROMPT_ARGS[@]}"
