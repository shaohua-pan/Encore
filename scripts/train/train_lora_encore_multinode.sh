#!/usr/bin/env bash
# Multi-node Encore LoRA training (8 GPUs per node, FSDP).
#
# Run this on every node with the same MASTER_ADDR / MASTER_PORT / NUM_NODES and
# a distinct MACHINE_RANK (0 on the main node).
#
# Usage:
#   MACHINE_RANK=0 MASTER_ADDR=<ip-of-rank-0> NUM_NODES=3 \
#       bash scripts/train/train_lora_encore_multinode.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/env.sh"

CONFIG="${1:-packages/ltx-trainer/configs/ltx2_av_lora_encore.yaml}"

MACHINE_RANK="${MACHINE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:?set MASTER_ADDR to the IP of the rank-0 node}"
MASTER_PORT="${MASTER_PORT:-29509}"
NUM_NODES="${NUM_NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
NUM_PROCESSES=$((NUM_NODES * GPUS_PER_NODE))

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

accelerate launch \
    --config_file packages/ltx-trainer/configs/accelerate/fsdp_multinode.yaml \
    --num_machines "${NUM_NODES}" \
    --num_processes "${NUM_PROCESSES}" \
    --machine_rank "${MACHINE_RANK}" \
    --main_process_ip "${MASTER_ADDR}" \
    --main_process_port "${MASTER_PORT}" \
    packages/ltx-trainer/scripts/train.py \
    "${CONFIG}"
