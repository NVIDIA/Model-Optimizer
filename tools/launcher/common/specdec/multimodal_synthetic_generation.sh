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

# Generate target-model completions for prepared prompt shards.
#
# DFlash learns from the *target model's* own completions, not from human-written
# answers, so every prompt shard is replayed through the target and the responses
# become the training labels.
#
# Shard-to-node assignment is derived from the launcher's own allocation
# (SLURM_JOB_ID + SLURM_JOB_NODELIST); no job id or node list is passed in.
# Each node serves the target locally and processes its own slice of shards.
#
# Usage from YAML:
#   script: common/specdec/multimodal_synthetic_generation.sh
#   args:
#     - --dataset vqa_v2
#     - --shard-path /scratchspace/data/vqa_shards
#     - --output-path /scratchspace/data/vqa_outputs
#     - --media-root /scratchspace/data/vqa_v2/images
#   environment:
#     - MODEL_PATH: /hf-local/nvidia/Cosmos3-Nano
#
# Env:
#   MODEL_PATH        — target checkpoint (required)
#   NUM_SHARDS        — shards to process; default: all remaining from START_SHARD
#   START_SHARD       — first shard index (default: 0; set to resume a partial run)
#   SGLANG_TP_SIZE    — TP per server (default 1 => one server per GPU)
#   NUM_TEMPERATURES  — temperature sweep width (default 8)
#   BACKEND           — vllm or sglang (default: sglang for media, vllm for text)

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source ${SCRIPT_DIR}/../service_utils.sh

trap 'error_handler $0 $LINENO' ERR
trap 'exit_handler' EXIT

set -euo pipefail

DATASET=""
SHARD_PATH=""
OUTPUT_PATH=""
MEDIA_ROOT=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASET="$2"; shift 2 ;;
        --shard-path) SHARD_PATH="$2"; shift 2 ;;
        --output-path) OUTPUT_PATH="$2"; shift 2 ;;
        --media-root) MEDIA_ROOT="$2"; shift 2 ;;
        *) echo "ERROR: unknown argument: $1" >&2; exit 1 ;;
    esac
done

[ -n "$DATASET" ] || { echo "ERROR: --dataset is required." >&2; exit 1; }
[ -n "$SHARD_PATH" ] || { echo "ERROR: --shard-path is required." >&2; exit 1; }
[ -n "$OUTPUT_PATH" ] || { echo "ERROR: --output-path is required." >&2; exit 1; }
[ -n "${MODEL_PATH:-}" ] || { echo "ERROR: MODEL_PATH must name the target checkpoint." >&2; exit 1; }
[ -d "$SHARD_PATH" ] || { echo "ERROR: missing shard path: $SHARD_PATH" >&2; exit 1; }

SPEC_ROOT=modules/Model-Optimizer/examples/speculative_decoding

# The launcher allocates the nodes, so read them back instead of taking a list.
if [ -n "${SLURM_JOB_NODELIST:-}" ]; then
    NODE_NAMES="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | paste -sd, -)"
    NUM_NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
else
    NODE_NAMES="$(hostname)"
    NUM_NODES=1
fi
[ "$NUM_NODES" -gt 0 ] || { echo "ERROR: no allocated nodes found." >&2; exit 1; }

AVAILABLE_SHARDS=$(find "$SHARD_PATH" -maxdepth 1 -type f -name 'train-*.jsonl' | wc -l)
[ "$AVAILABLE_SHARDS" -gt 0 ] || { echo "ERROR: no train-*.jsonl shards in $SHARD_PATH" >&2; exit 1; }

START_SHARD=${START_SHARD:-0}
[[ "$START_SHARD" =~ ^[0-9]+$ ]] || { echo "ERROR: START_SHARD must be a non-negative integer." >&2; exit 1; }
(( START_SHARD < AVAILABLE_SHARDS )) || { echo "ERROR: START_SHARD=$START_SHARD is beyond $AVAILABLE_SHARDS shards." >&2; exit 1; }

REMAINING=$(( AVAILABLE_SHARDS - START_SHARD ))
NUM_SHARDS=${NUM_SHARDS:-$REMAINING}
(( NUM_SHARDS > 0 && START_SHARD + NUM_SHARDS <= AVAILABLE_SHARDS )) \
    || { echo "ERROR: shard range [$START_SHARD, $((START_SHARD + NUM_SHARDS))) exceeds $AVAILABLE_SHARDS shards." >&2; exit 1; }

# Round up so the tail shards are still assigned; workers skip a missing shard.
JOBS_PER_NODE=${JOBS_PER_NODE:-$(( (NUM_SHARDS + NUM_NODES - 1) / NUM_NODES ))}

echo "Generating $DATASET: shards [$START_SHARD, $((START_SHARD + NUM_SHARDS))) over $NUM_NODES node(s), $JOBS_PER_NODE per node"

export MODEL_PATH
export SHARD_PATH
export OUTPUT_PATH
export PREPARE_SHARDS=0
export SGLANG_TP_SIZE=${SGLANG_TP_SIZE:-1}
export NUM_TEMPERATURES=${NUM_TEMPERATURES:-8}

mkdir -p "$OUTPUT_PATH"

if [ "$DATASET" = "specdec_multilingual_prompt" ]; then
    export BACKEND=${BACKEND:-vllm}
    export TEXT_DATA=${TEXT_DATA:-$SHARD_PATH}
else
    # Media generation goes through SGLang's native image/video client.
    [ -n "$MEDIA_ROOT" ] || { echo "ERROR: --media-root is required for $DATASET." >&2; exit 1; }
    export BACKEND=sglang
    export MEDIA_ROOT
    export DATASET_DIR=${DATASET_DIR:-$MEDIA_ROOT}
    export IMAGE_ROOT=${IMAGE_ROOT:-$MEDIA_ROOT}
    export VQA_ROOT=${VQA_ROOT:-$MEDIA_ROOT}
fi

bash "$SPEC_ROOT/recipes/run_multimodal_synthetic_generation.sh" \
    "$DATASET" "${SLURM_JOB_ID:-0}" "$START_SHARD" "$JOBS_PER_NODE" "$NODE_NAMES"

echo "Wrote $(find "$OUTPUT_PATH" -maxdepth 1 -name '*.jsonl' | wc -l) output file(s) to $OUTPUT_PATH"
