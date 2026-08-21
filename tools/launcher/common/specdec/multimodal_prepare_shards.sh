#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Download a multimodal source dataset and write prompt shards for synthetic
# generation. No GPU is used: this only fetches data and reshapes it into the
# train-%05d-%05d.jsonl layout the generation workers consume.
#
# One invocation prepares one DATASET. Run it once per source, then merge.
#
# Usage from YAML:
#   script: common/specdec/multimodal_prepare_shards.sh
#   args:
#     - --dataset pai_understanding
#     - --shard-path /scratchspace/data/pai_shards
#   environment:
#     - DATA_ROOT: /scratchspace/data
#
# Env:
#   DATA_ROOT        — download/extract root (default: /scratchspace/data)
#   NUM_SAMPLES      — cap records before sharding (default: dataset-specific)
#   SHUFFLE_SEED     — deterministic shuffle before slicing (default: 42)
#   LINES_PER_SHARD  — records per generated shard (default: 128; text: 1024)
#   PAI_REPO_ID      — override the PAI-Bench-U Hugging Face repo
#   FORCE_DOWNLOAD   — re-download PAI even if it is already materialized
#   PAI_REVISION / TEXT_PROMPT_REVISION — pin a Hugging Face dataset revision

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source ${SCRIPT_DIR}/../service_utils.sh

trap 'error_handler $0 $LINENO' ERR
trap 'exit_handler' EXIT

set -euo pipefail

DATASET=""
SHARD_PATH=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASET="$2"; shift 2 ;;
        --shard-path) SHARD_PATH="$2"; shift 2 ;;
        *) echo "ERROR: unknown argument: $1" >&2; exit 1 ;;
    esac
done
[ -n "$DATASET" ] && [ -n "$SHARD_PATH" ] \
    || { echo "ERROR: --dataset and --shard-path are required." >&2; exit 1; }

DATA_ROOT=${DATA_ROOT:-/scratchspace/data}
SHUFFLE_SEED=${SHUFFLE_SEED:-42}
PREPARE=modules/Model-Optimizer/examples/speculative_decoding/recipes/prepare_multimodal_synthetic_shards.py
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$DATA_ROOT/.hf_datasets_cache}

pip install "huggingface-hub>=1.2.1" pillow
mkdir -p "$DATA_ROOT"

# Direct `hf download` calls below go through `python3 -m huggingface_hub.cli.hf`
# rather than a bare `hf`, which may be a stale executable on PATH. If a download
# stalls, retry the task with HF_HUB_DISABLE_XET=1.
case "$DATASET" in
pai_understanding)
    PAI_ROOT=${PAI_ROOT:-$DATA_ROOT/pai_understanding}
    LINES_PER_SHARD=${LINES_PER_SHARD:-128}
    # PAI is sampled by shard count so the generation step gets a whole number
    # of shards per node; see the YAML's PAI_NUM_GENERATION_SHARDS comment.
    NUM_SAMPLES=${NUM_SAMPLES:-$(( ${NUM_GENERATION_SHARDS:-5} * LINES_PER_SHARD ))}
    # prepare_multimodal_synthetic_shards.py downloads PAI itself via --download,
    # so let it. Its snapshot_download has no revision argument, so pin a
    # revision here instead when PAI_REVISION is set.
    PAI_DOWNLOAD_ARGS=(--download)
    if [ -n "${PAI_REVISION:-}" ]; then
        python3 -m huggingface_hub.cli.hf download "${PAI_REPO_ID:-shi-labs/physical-ai-bench-understanding}" \
            --repo-type dataset --local-dir "$PAI_ROOT" --max-workers 8 \
            --revision "$PAI_REVISION"
        PAI_DOWNLOAD_ARGS=()
    fi
    python3 "$PREPARE" \
        --dataset pai_understanding \
        --dataset_dir "$PAI_ROOT" \
        --media_root "$PAI_ROOT" \
        --output_dir "$SHARD_PATH" \
        --max_lines_per_shard "$LINES_PER_SHARD" \
        --num_samples "$NUM_SAMPLES" \
        --shuffle_seed "$SHUFFLE_SEED" \
        "${PAI_DOWNLOAD_ARGS[@]}" \
        ${PAI_REPO_ID:+--repo_id "$PAI_REPO_ID"} \
        ${FORCE_DOWNLOAD:+--force_download} \
        --overwrite
    ;;
vqa_v2)
    VQA_ROOT=${VQA_ROOT:-$DATA_ROOT/vqa_v2}
    IMAGE_ROOT="$VQA_ROOT/images"
    LINES_PER_SHARD=${LINES_PER_SHARD:-128}
    NUM_SAMPLES=${NUM_SAMPLES:-20000}
    VQA_QUESTIONS_URL=${VQA_QUESTIONS_URL:-https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip}
    VQA_ANNOTATIONS_URL=${VQA_ANNOTATIONS_URL:-https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip}
    COCO_TRAIN_URL=${COCO_TRAIN_URL:-https://images.cocodataset.org/zips/train2014.zip}

    extract_zip() {
        local archive=$1 destination=$2 marker=$3
        if [ -f "$marker" ]; then
            echo "Already extracted: $archive"
            return
        fi
        if command -v unzip >/dev/null 2>&1; then
            unzip -n "$archive" -d "$destination"
        else
            # Minimal container images may omit `unzip`.
            python3 -m zipfile -e "$archive" "$destination"
        fi
        touch "$marker"
    }

    mkdir -p "$VQA_ROOT" "$IMAGE_ROOT"
    curl --proto '=https' -L --fail --retry 5 -C - -o "$VQA_ROOT/questions.zip" "$VQA_QUESTIONS_URL"
    curl --proto '=https' -L --fail --retry 5 -C - -o "$VQA_ROOT/annotations.zip" "$VQA_ANNOTATIONS_URL"
    curl --proto '=https' -L --fail --retry 5 -C - -o "$IMAGE_ROOT/train2014.zip" "$COCO_TRAIN_URL"
    extract_zip "$VQA_ROOT/questions.zip" "$VQA_ROOT" "$VQA_ROOT/.questions.extracted"
    extract_zip "$VQA_ROOT/annotations.zip" "$VQA_ROOT" "$VQA_ROOT/.annotations.extracted"
    extract_zip "$IMAGE_ROOT/train2014.zip" "$IMAGE_ROOT" "$IMAGE_ROOT/.train2014.extracted"

    python3 "$PREPARE" \
        --dataset vqa_v2 \
        --vqa_root "$VQA_ROOT" \
        --image_root "$IMAGE_ROOT" \
        --vqa_splits "${VQA_SPLITS:-train}" \
        --num_samples "$NUM_SAMPLES" \
        --shuffle_seed "$SHUFFLE_SEED" \
        --output_dir "$SHARD_PATH" \
        --overwrite
    ;;
specdec_multilingual_prompt)
    TEXT_ROOT=${TEXT_ROOT:-$DATA_ROOT/specdec_multilingual_prompt}
    # Only default.jsonl: the repo's sample-*.jsonl subsets are slices of it, so
    # downloading everything would duplicate prompts.
    python3 -m huggingface_hub.cli.hf download nvidia/Speculative-Decoding-Multilingual-Prompt-v2 \
        default.jsonl --repo-type dataset --local-dir "$TEXT_ROOT" --max-workers 8 \
        ${TEXT_PROMPT_REVISION:+--revision "$TEXT_PROMPT_REVISION"}
    python3 "$PREPARE" \
        --dataset specdec_multilingual_prompt \
        --text_data "$TEXT_ROOT" \
        --max_lines_per_shard "${LINES_PER_SHARD:-1024}" \
        --output_dir "$SHARD_PATH" \
        --overwrite
    ;;
*)
    echo "ERROR: unsupported --dataset: $DATASET" >&2
    exit 1
    ;;
esac

echo "Prepared $(find "$SHARD_PATH" -maxdepth 1 -name 'train-*.jsonl' | wc -l) shard(s) in $SHARD_PATH"
