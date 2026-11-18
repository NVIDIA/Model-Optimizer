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
# Launches trtllm-serve with the given model, waits for it to be ready,
# then runs common/query.py against the server.
#
# --model MODEL is required and is consumed by this script. It is used as the
# positional model argument for both trtllm-serve and common/query.py.
#
# Remaining arguments are split on "--":
#   - Args BEFORE "--" are appended to the trtllm-serve command (SERVE_ARGS).
#   - Args AFTER  "--" are passed to common/query.py (QUERY_ARGS).
#   - If "--" is absent, all remaining args go to common/query.py.
#
# Environment variables (optional, set by Slurm):
#   SLURM_ARRAY_TASK_ID     Used to shard query.py work across array jobs.
#   SLURM_ARRAY_TASK_COUNT  Total number of array tasks for sharding.
#
# In a pipeline YAML task config:
#   args:
#     - --model /hf-local/Qwen/Qwen3-8B  # required
#     - --tp_size 4                        # trtllm-serve args (before --)
#     - --ep_size 4
#     - --max_num_tokens 32000
#     - --port 8000
#     - --host 0.0.0.0
#     - --trust_remote_code
#     - --                                 # separator
#     - --data /hf-local/dataset           # query.py args (after --)
#     - --save /scratchspace/data
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
# --model is consumed here; args before "--" go to trtllm-serve, args after go to query.py.
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

trtllm-llmapi-launch trtllm-serve \
    ${MODEL} \
    "${SERVE_EXTRA_ARGS[@]}" \
    &


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

if [[ "$mpi_rank" -eq 0 ]]; then
    cmd="python common/query.py http://localhost:8000/v1 ${MODEL} ${QUERY_ARGS[*]}"
    echo "Running command: $cmd"
    eval $cmd
    echo "Main process exit"
else
    while true; do
        response=$(curl -s -o /dev/null -w "%{http_code}" "http://$(hostname -f):8000/health" || true)
        if [[ "$response" -ne 200 ]]; then
            break
        fi
        #echo "Server is up!"
        sleep 60
    done
fi

pkill trtllm-serve

exit 0
