#!/usr/bin/env bash
# Single-node Encore LoRA training (8 GPUs, FSDP).
#
# Edit packages/ltx-trainer/configs/ltx2_av_lora_encore.yaml first: it holds the
# model paths, the preprocessed data root and the output directory.
#
# Usage:
#   bash scripts/train/train_lora_encore.sh [CONFIG]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/env.sh"

CONFIG="${1:-packages/ltx-trainer/configs/ltx2_av_lora_encore.yaml}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

accelerate launch \
    --config_file packages/ltx-trainer/configs/accelerate/fsdp.yaml \
    packages/ltx-trainer/scripts/train.py \
    "${CONFIG}"
