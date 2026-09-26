#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

if [ $# -lt 9 ]; then
    echo "Usage: $0 <job_id> <backend> <model_path> <data_path> <output_path> <scripts_path> <start_shard> <jobs_per_node> <comma_separated_node_names> <system_prompt>"
    echo "Example: $0 245387 vllm /model/ /input_data/ /output_data/ /scripts/ 0 20 cluster-01,cluster-02 "\"You are a helpful assistant.\"""
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
NODE_NAME=$9
SYSTEM_PROMPT="${10:-}"
IFS=',' read -r -a NODE_LIST <<< "$NODE_NAME"

# backend needs to be either vllm or sglang
if [ "$BACKEND" != "vllm" ] && [ "$BACKEND" != "sglang" ]; then
    echo "Invalid backend: $BACKEND"
    exit 1
fi

if [ "$BACKEND" == "vllm" ]; then
    CONTAINER_IMAGE="vllm/vllm-openai:v0.8.5"
else
    CONTAINER_IMAGE="lmsysorg/sglang:v0.4.6.post2-cu124"
fi
counter=$START_SHARD
for node in "${NODE_LIST[@]}"; do
    echo "Processing node: $node"
    srun --output=srun_worker_${node}.log --jobid=$JOB_ID -N 1 --ntasks=1 --ntasks-per-node=1 -w $node \
        --mpi pmix --overlap --container-image=$CONTAINER_IMAGE \
        --container-mounts=$MODEL_PATH:/model/,$DATA_PATH:/input_data/,$OUTPUT_PATH:/output_data/,$SCRIPTS_PATH:/scripts/ \
        bash /scripts/distributed_generate/worker.sh $counter $BACKEND $JOBS_PER_NODE "$SYSTEM_PROMPT" &

    echo "srun command for node $node started with PID $!" >> srun_launch.log

    # increment counter by JOBS_PER_NODE
    counter=$((counter + JOBS_PER_NODE))
done

echo "Started workers, each processing $JOBS_PER_NODE shards of data. Will process shards $START_SHARD through $((counter - 1))."
