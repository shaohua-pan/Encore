#!/usr/bin/env bash
# Audio-to-video: drive a portrait with a long audio track (Encore + routing).
#
# The audio is split across segments, each segment is generated with the audio
# latents frozen, and the segments are written to OUTPUT_PATH.
#
# Usage:
#   bash scripts/inference/inference_encore_a2v.sh
#   IMAGE=my.png A2V_AUDIO=song.mp3 bash scripts/inference/inference_encore_a2v.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/env.sh"

IMAGE="${IMAGE:?set IMAGE to your own first-frame image (png/jpg)}"
A2V_AUDIO="${A2V_AUDIO:?set A2V_AUDIO to the driving audio file (wav/mp3)}"
PROMPT="${PROMPT:-A woman is singing, her hands and head swaying naturally. The camera stays still.}"
OUTPUT_PATH="${OUTPUT_PATH:-${OUTPUT_ROOT}/encore_a2v}"

HEIGHT="${HEIGHT:-1088}"
WIDTH="${WIDTH:-1920}"
NUM_FRAMES="${NUM_FRAMES:-121}"
ROUTING_SCALE="${ROUTING_SCALE:-0.5}"

python -m ltx_pipelines.a2vid_two_stages_hq_encore \
    --checkpoint-path "${CHECKPOINT_PATH}" \
    --distilled-lora "${DISTILLED_LORA}" \
    --spatial-upsampler-path "${SPATIAL_UPSAMPLER_PATH}" \
    --gemma-root "${GEMMA_ROOT}" \
    --lora "${ENCORE_LORA}" 1 \
    --prompt "${PROMPT}" \
    --image "${IMAGE}" 0 1.0 \
    --a2v-audio "${A2V_AUDIO}" \
    --num-frames "${NUM_FRAMES}" \
    --height "${HEIGHT}" \
    --width "${WIDTH}" \
    --output-path "${OUTPUT_PATH}" \
    --routing-scale "${ROUTING_SCALE}"

# Optional: --audio path/to/reference.wav 0 1.0 to also pin the global timbre.
