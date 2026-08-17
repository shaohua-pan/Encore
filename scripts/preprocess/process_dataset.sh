#!/usr/bin/env bash
# Precompute video latents, audio latents and text embeddings for training.
#
# Input: a JSONL file, one clip per line. See examples/data/example_dataset.jsonl
# for the exact schema, and examples/README.md for a runnable toy dataset.
#
# Output layout (under ${DATA_ROOT}/precomputed):
#   latents/        video latents        (mirrors the clip directory tree)
#   audio_latents/  audio latents
#   conditions/     text embeddings
#
# The JSONL is split into ${NUM_GPUS} shards that are processed in parallel, one
# process per GPU, all writing into the same output directory. Output file paths
# are derived from each clip's path, so shards never collide.
#
# Usage:
#   bash scripts/preprocess/process_dataset.sh [DATASET_JSONL]
#   NUM_GPUS=4 bash scripts/preprocess/process_dataset.sh my_dataset.jsonl
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/env.sh"

DATASET_JSONL="${1:-examples/data/example_dataset.jsonl}"
if [ ! -f "${DATASET_JSONL}" ]; then
    echo "ERROR: dataset file not found: ${DATASET_JSONL}" >&2
    exit 1
fi

# Prefix stripped from each clip path when deriving the output path of its
# latents, so the output mirrors the input directory tree. Set it to the common
# root of the video paths in your JSONL.
VIDEO_PREFIX_PATH="${VIDEO_PREFIX_PATH:-${DATA_ROOT}/clips}"

OUT_DIR="${DATA_ROOT}/precomputed"
NUM_GPUS="${NUM_GPUS:-8}"
RESOLUTION_BUCKETS="${RESOLUTION_BUCKETS:-960x544x121;544x960x121}"

SHARD_DIR="$(mktemp -d)"
trap 'rm -rf "${SHARD_DIR}"' EXIT

# Round-robin the lines so every shard sees a similar mix of clips.
awk -v n="${NUM_GPUS}" -v dir="${SHARD_DIR}" \
    '{ print > sprintf("%s/shard_%02d.jsonl", dir, NR % n) }' "${DATASET_JSONL}"

mkdir -p "${OUT_DIR}"

declare -a PIDS=()
for shard in "${SHARD_DIR}"/shard_*.jsonl; do
    gpu="$(basename "${shard}" .jsonl)"
    gpu="${gpu#shard_}"
    gpu="$((10#${gpu}))"
    echo "[launch] gpu=${gpu} shard=${shard} ($(wc -l <"${shard}") clips)"

    CUDA_VISIBLE_DEVICES="${gpu}" python packages/ltx-trainer/scripts/process_dataset.py \
        "${shard}" \
        --resolution-buckets "${RESOLUTION_BUCKETS}" \
        --model-path "${CHECKPOINT_PATH}" \
        --text-encoder-path "${GEMMA_ROOT}" \
        --output-dir "${OUT_DIR}" \
        --caption-column caption_all \
        --video-column video_path \
        --audio-indices-column frame_idx \
        --video-prefix-path "${VIDEO_PREFIX_PATH}" \
        --with-audio \
        --with-reference-image \
        --with-reference-audio \
        >"${OUT_DIR}/preprocess_gpu${gpu}.log" 2>&1 &
    PIDS+=("$!")
done

echo "[info] launched ${#PIDS[@]} workers; logs in ${OUT_DIR}/preprocess_gpu*.log"
FAIL=0
for pid in "${PIDS[@]}"; do
    wait "${pid}" || FAIL=$((FAIL + 1))
done

if [ "${FAIL}" -ne 0 ]; then
    echo "[summary] ${FAIL} worker(s) failed, see ${OUT_DIR}/preprocess_gpu*.log" >&2
    exit 1
fi
echo "[summary] preprocessing complete -> ${OUT_DIR}"

# Add --decode to a worker command above to also decode the latents back to
# mp4/wav under ${OUT_DIR}/decoded_videos and ${OUT_DIR}/decoded_audio, which is
# the quickest way to sanity-check the preprocessing.
