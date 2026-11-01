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

if [ $# -lt 10 ]; then
    echo "Usage: $0 <job_id> <backend> <model_path> <shard_path> <output_path> <scripts_path> <start_shard> <jobs_per_node> <media_path> [num_frames] <comma_separated_node_names> [system_prompt]"
    echo "Also accepted: $0 <job_id> <backend> <model_path> <shard_path> <output_path> <scripts_path> <start_shard> <jobs_per_node> <comma_separated_node_names> <media_path> [num_frames] [system_prompt]"
    echo "Example: $0 245387 sglang /model/ /shards/ /output/ /scripts/ 0 10 /media/ 16 cluster-01"
    echo "Optional env: SGLANG_TP_SIZE=8 NUM_TEMPERATURES=8 NUM_THREADS=8 SGLANG_EXTRA_ARGS='--mem-fraction-static 0.75'"
    exit 1
fi

JOB_ID=$1
BACKEND=$2
MODEL_PATH=$3
DATA_PATH=$4
OUTPUT_PATH=$5
SCRIPTS_PATH=$6
START_SHARD=$7
JOBS_PER_NODE=$8
ARG9=$9
ARG10=${10:-}
ARG11=${11:-}
ARG12=${12:-}

if [[ "$ARG9" == */* || "$ARG9" == .* ]]; then
    MEDIA_PATH=$ARG9
    NUM_FRAMES="${ARG10:-32}"
    NODE_NAME=$ARG11
    SYSTEM_PROMPT="$ARG12"
else
    NODE_NAME=$ARG9
    MEDIA_PATH=$ARG10
    NUM_FRAMES="${ARG11:-32}"
    SYSTEM_PROMPT="$ARG12"
fi

if [ -z "${NODE_NAME:-}" ] || [ -z "${MEDIA_PATH:-}" ]; then
    echo "ERROR: both media_path and node_name are required." >&2
    exit 1
fi

IFS=',' read -r -a NODE_LIST <<< "$NODE_NAME"

if [ "$BACKEND" != "sglang" ]; then
    echo "Multimodal generation currently supports backend=sglang."
    exit 1
fi

mkdir -p "$OUTPUT_PATH"

# Set CONTAINER_IMAGE to a local .sqsh image to avoid pulling from the registry.
DEFAULT_CONTAINER_IMAGE="lmsysorg/sglang:v0.5.3-cu129"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-$DEFAULT_CONTAINER_IMAGE}"

counter=$START_SHARD
worker_pids=()
for node in "${NODE_LIST[@]}"; do
    echo "Processing node: $node"
    srun --output=srun_vlm_worker_${node}.log --jobid=$JOB_ID -N 1 --ntasks=1 --ntasks-per-node=1 -w "$node" \
        --mpi pmix --overlap --container-image="$CONTAINER_IMAGE" \
        --container-mounts="$MODEL_PATH":/model/,"$DATA_PATH":/input_data/,"$OUTPUT_PATH":/output_data/,"$SCRIPTS_PATH":/scripts/,"$MEDIA_PATH":/media_data/ \
        bash /scripts/distributed_generate/worker_multimodal.sh "$counter" "$BACKEND" "$JOBS_PER_NODE" "$NUM_FRAMES" "$SYSTEM_PROMPT" &

    echo "srun multimodal command for node $node started with PID $!" >> srun_launch_multimodal.log
    worker_pids+=("$!")
    counter=$((counter + JOBS_PER_NODE))
done

echo "Started multimodal workers, each processing $JOBS_PER_NODE shards of data. Will process shards $START_SHARD through $((counter - 1))."

worker_status=0
for worker_pid in "${worker_pids[@]}"; do
    if ! wait "$worker_pid"; then
        worker_status=1
    fi
done

if [ "$worker_status" -ne 0 ]; then
    echo "ERROR: one or more multimodal workers failed." >&2
fi
exit "$worker_status"
