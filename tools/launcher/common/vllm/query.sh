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

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

source ${SCRIPT_DIR}/../service_utils.sh

###################################################################################################
# Usage:
#   query.sh --model MODEL [SERVE_ARGS...] -- [QUERY_ARGS...]
#
# Launches vllm serve with the given model, waits for it to be ready,
# then runs common/query.py against the server.
#
# --model MODEL is required and is consumed by this script. It is used as the
# positional model argument for both vllm serve and common/query.py.
#
# Remaining arguments are split on "--":
#   - Args BEFORE "--" are appended to the vllm serve command (SERVE_ARGS).
#   - Args AFTER  "--" are passed to common/query.py (QUERY_ARGS).
#   - If "--" is absent, all remaining args go to common/query.py.
#
# Environment variables (optional, set by Slurm):
#   SLURM_ARRAY_TASK_ID     Used to shard query.py work across array jobs.
#   SLURM_ARRAY_TASK_COUNT  Total number of array tasks for sharding.
#
# vLLM notes:
#   - vLLM manages GPU distribution internally; run with ntasks_per_node: 1
#     in slurm_config and pass --tensor-parallel-size to match gpus_per_node.
#   - NVFP4 models require vllm/vllm-openai:v0.15.0+ on Blackwell GPUs.
#   - Use --trust-remote-code for models with custom architectures (e.g. Kimi).
#
# In a pipeline YAML task config:
#   args:
#     - --model /hf-local/Qwen/Qwen3-8B  # required
#     - --tensor-parallel-size 4           # vllm serve args (before --)
#     - --max-num-seqs 32
#     - --trust-remote-code
#     - --                                 # separator
#     - --data /hf-local/dataset           # query.py args (after --)
#     - --save /scratchspace/data
#   slurm_config:
#     ntasks_per_node: 1                   # vLLM is single-process
#     gpus_per_node: 4
###################################################################################################

export OPENAI_API_KEY="token-abc123"

if [ -z ${SLURM_ARRAY_TASK_ID} ]; then
    TASK_ID=0
else
    echo "SLURM_ARRAY_TASK_ID ${SLURM_ARRAY_TASK_ID}"
    TASK_ID=${SLURM_ARRAY_TASK_ID}
fi

if [ -z ${SLURM_ARRAY_TASK_COUNT} ]; then
    TASK_COUNT=1
else
    echo "SLURM_ARRAY_TASK_COUNT ${SLURM_ARRAY_TASK_COUNT}"
    TASK_COUNT=${SLURM_ARRAY_TASK_COUNT}
fi

# Parse --model and split remaining args on "--".
# --model is consumed here; args before "--" go to vllm serve, args after go to query.py.
MODEL=""
SERVE_EXTRA_ARGS=()
QUERY_ARGS=(--shard-id-begin ${TASK_ID} --shard-id-step ${TASK_COUNT})
past_separator=false
skip_next=false

for arg in "$@"; do
    if $skip_next; then
        MODEL="$arg"
        skip_next=false
    elif [ "$arg" = "--model" ]; then
        skip_next=true
    elif [ "$arg" = "--" ]; then
        past_separator=true
    elif [ "$past_separator" = false ]; then
        SERVE_EXTRA_ARGS+=("$arg")
    else
        QUERY_ARGS+=("$arg")
    fi
done

# vLLM is single-process: GPU parallelism is handled internally via --tensor-parallel-size.
# No MPI multi-rank logic needed; this script always runs as a single task.
vllm serve \
    ${MODEL} \
    "${SERVE_EXTRA_ARGS[@]}" \
    &
SERVER_PID=$!


# Wait for server to start up by polling the health endpoint
echo "Waiting for server to start..."
while true; do
    response=$(curl -s -o /dev/null -w "%{http_code}" "http://$(hostname -f):8000/health" || true)
    if [ "$response" -eq 200 ]; then
        echo "Server is up!"
        break
    fi
    echo "Server not ready yet, retrying in 10 seconds..."
    sleep 10
done

pip3 install -q datasets openai 2>/dev/null || true
echo "Running: python3 common/query.py http://localhost:8000/v1 ${MODEL} ${QUERY_ARGS[*]}"
python3 common/query.py http://localhost:8000/v1 "${MODEL}" "${QUERY_ARGS[@]}"
echo "Main process exit"

kill $SERVER_PID
wait $SERVER_PID 2>/dev/null || true

exit 0
