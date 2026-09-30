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

set -euo pipefail

if [ $# -lt 4 ] || [ $# -gt 5 ]; then
    echo "Usage: MODEL_PATH=/path/to/model $0 <pai_understanding|vqa_v2|specdec_multilingual_prompt> [job_id] <start_shard> <jobs_per_node> <comma_separated_node_names>" >&2
    echo "PAI env: DATASET_DIR=/dataset/root [MEDIA_ROOT=/video/root] [PREPARE_DOWNLOAD=1] [PAI_CATEGORY=...]" >&2
    echo "VQA env: VQA_ROOT=/questions-and-annotations IMAGE_ROOT=/coco-images [VQA_SPLITS=train,val]" >&2
    echo "Text env: TEXT_DATA=/path/to/sample-1K.jsonl [BACKEND=vllm|sglang]" >&2
    exit 1
fi

DATASET=$1
shift
case "$DATASET" in pai_understanding|vqa_v2|specdec_multilingual_prompt) ;; *) echo "ERROR: unsupported dataset: $DATASET" >&2; exit 1;; esac
if [ $# -eq 3 ]; then
    JOB_ID=${SLURM_JOB_ID:-}
    START_SHARD=$1
    JOBS_PER_NODE=$2
    NODE_NAMES=$3
else
    JOB_ID=$1
    START_SHARD=$2
    JOBS_PER_NODE=$3
    NODE_NAMES=$4
fi
[ -n "${JOB_ID:-}" ] || { echo "ERROR: SLURM_JOB_ID is not set; pass job_id explicitly." >&2; exit 1; }
[ -n "${MODEL_PATH:-}" ] || { echo "ERROR: MODEL_PATH must name the local VLM checkpoint." >&2; exit 1; }

SPEC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS_PATH=${SCRIPTS_PATH:-$SPEC_ROOT}
MEDIA_MODE=${MEDIA_MODE:-native}
SHARD_PATH=${SHARD_PATH:-$SPEC_ROOT/CR3_data/${DATASET}_${MEDIA_MODE}_shards}
OUTPUT_PATH=${OUTPUT_PATH:-$SPEC_ROOT/${DATASET}_synthetic_outputs}
MAX_LINES_PER_SHARD=${MAX_LINES_PER_SHARD:-128}
PROMPT_STYLE=${PROMPT_STYLE:-thinking}
NUM_FRAMES=${NUM_FRAMES:-16}

export API_MODE=${API_MODE:-native}
export OVERWRITE_OUTPUT=${OVERWRITE_OUTPUT:-1}
export SGLANG_TP_SIZE=${SGLANG_TP_SIZE:-1}
export NUM_TEMPERATURES=${NUM_TEMPERATURES:-8}
export NUM_THREADS=${NUM_THREADS:-8}

prepare_args=(--dataset "$DATASET" --output_dir "$SHARD_PATH" --media_mode "$MEDIA_MODE" --prompt_style "$PROMPT_STYLE" --num_frames "$NUM_FRAMES" --max_lines_per_shard "$MAX_LINES_PER_SHARD" --missing_media "${MISSING_MEDIA:-error}")
if [ "$DATASET" = "pai_understanding" ]; then
    [ -n "${DATASET_DIR:-}" ] || { echo "ERROR: DATASET_DIR is required for pai_understanding." >&2; exit 1; }
    MEDIA_ROOT=${MEDIA_ROOT:-$DATASET_DIR}
    prepare_args+=(--dataset_dir "$DATASET_DIR" --media_root "$MEDIA_ROOT")
    [ "${PREPARE_DOWNLOAD:-0}" = "1" ] && prepare_args+=(--download)
    [ "${FORCE_DOWNLOAD:-0}" = "1" ] && prepare_args+=(--force_download)
    [ -n "${PAI_REPO_ID:-}" ] && prepare_args+=(--repo_id "$PAI_REPO_ID")
    [ -n "${PAI_CATEGORY:-}" ] && prepare_args+=(--category "$PAI_CATEGORY")
    if [ "$MEDIA_MODE" = "image_mosaic" ]; then GENERATION_MEDIA_ROOT=$SHARD_PATH; else GENERATION_MEDIA_ROOT=$MEDIA_ROOT; fi
elif [ "$DATASET" = "vqa_v2" ]; then
    [ -n "${VQA_ROOT:-}" ] || { echo "ERROR: VQA_ROOT is required for vqa_v2." >&2; exit 1; }
    [ -n "${IMAGE_ROOT:-}" ] || { echo "ERROR: IMAGE_ROOT is required for vqa_v2." >&2; exit 1; }
    prepare_args+=(--vqa_root "$VQA_ROOT" --image_root "$IMAGE_ROOT" --vqa_splits "${VQA_SPLITS:-train}")
    [ "${VQA_INCLUDE_UNANNOTATED:-0}" = "1" ] && prepare_args+=(--include_unannotated_vqa)
    [ "${VQA_INCLUDE_ALL_ANSWERS:-0}" = "1" ] && prepare_args+=(--include_all_answers)
    GENERATION_MEDIA_ROOT=$IMAGE_ROOT
else
    [ -n "${TEXT_DATA:-}" ] || { echo "ERROR: TEXT_DATA is required for specdec_multilingual_prompt." >&2; exit 1; }
    prepare_args+=(--text_data "$TEXT_DATA")
fi
[ -n "${NUM_SAMPLES:-}" ] && prepare_args+=(--num_samples "$NUM_SAMPLES")
[ -n "${START_INDEX:-}" ] && prepare_args+=(--start_index "$START_INDEX")
[ "${PREPARE_OVERWRITE:-0}" = "1" ] && prepare_args+=(--overwrite)

if [ "${PREPARE_SHARDS:-auto}" = "1" ] || ! compgen -G "$SHARD_PATH/train-*.jsonl" >/dev/null; then
    python3 "$SCRIPTS_PATH/recipes/prepare_multimodal_synthetic_shards.py" "${prepare_args[@]}"
fi

if [ "$DATASET" = "specdec_multilingual_prompt" ]; then
    bash "$SCRIPTS_PATH/distributed_generate/launch.sh" "$JOB_ID" "${BACKEND:-vllm}" "$MODEL_PATH" "$SHARD_PATH" "$OUTPUT_PATH" "$SCRIPTS_PATH" "$START_SHARD" "$JOBS_PER_NODE" "$NODE_NAMES" "${SYSTEM_PROMPT:-}"
else
    bash "$SCRIPTS_PATH/distributed_generate/launch_multimodal.sh" "$JOB_ID" sglang "$MODEL_PATH" "$SHARD_PATH" "$OUTPUT_PATH" "$SCRIPTS_PATH" "$START_SHARD" "$JOBS_PER_NODE" "$GENERATION_MEDIA_ROOT" "$NUM_FRAMES" "$NODE_NAMES"
fi
