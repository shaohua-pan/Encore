#!/usr/bin/env bash
# Single-segment image-to-video with synchronized audio (Encore + condition routing).
#
# Takes one image and one prompt and writes one ~5 s mp4 with audio.
#
# Prompt format that the model was trained on — describe the scene, the speech
# and the voice:
#   "<scene and motion description>
#    Speech: "<what is said>"
#    Voice: <age / pace / accent / tone>"
#
# Usage:
#   bash scripts/inference/inference_encore_i2v.sh
#   IMAGE=my.png PROMPT='...' OUTPUT=out.mp4 \
#       bash scripts/inference/inference_encore_i2v.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/env.sh"

IMAGE="${IMAGE:?set IMAGE to your own first-frame image (png/jpg)}"
PROMPT="${PROMPT:-A woman looks into the camera and speaks naturally, her head turning slightly, her expression focused. The camera stays still.}"
OUTPUT="${OUTPUT:-${OUTPUT_ROOT}/encore_i2v.mp4}"

# Height/width must be divisible by 32; num-frames must satisfy frames % 8 == 1.
HEIGHT="${HEIGHT:-1088}"
WIDTH="${WIDTH:-1920}"
NUM_FRAMES="${NUM_FRAMES:-121}"

# Strength of the learned condition routing table at inference time. The
# effective strength is lora_strength * routing_scale. 0.5 is what we report;
# 0.0 disables routing.
ROUTING_SCALE="${ROUTING_SCALE:-0.5}"

mkdir -p "$(dirname "${OUTPUT}")"

python -m ltx_pipelines.ti2vid_two_stages_hq_encore \
    --checkpoint-path "${CHECKPOINT_PATH}" \
    --distilled-lora "${DISTILLED_LORA}" \
    --spatial-upsampler-path "${SPATIAL_UPSAMPLER_PATH}" \
    --gemma-root "${GEMMA_ROOT}" \
    --lora "${ENCORE_LORA}" 1 \
    --prompt "${PROMPT}" \
    --image "${IMAGE}" 0 1.0 \
    --output-path "${OUTPUT}" \
    --height "${HEIGHT}" \
    --width "${WIDTH}" \
    --num-frames "${NUM_FRAMES}" \
    --routing-scale "${ROUTING_SCALE}"

# Optional extras:
#   --audio path/to/reference.wav 0 1.0   fix the speaker timbre from a reference clip
#   --seed 42                             reproducible sampling
#   --negative-prompt "..."               steer away from artifacts
