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

_CURRENT_COUNTER="$1"

BACKEND="$2"
JOBS_PER_NODE="$3"
SYSTEM_PROMPT="$4"

if [ "$BACKEND" == "vllm" ]; then
    vllm serve /model/  --tensor-parallel-size 8 --served-model-name model --port 8000 --host 0.0.0.0 --trust-remote-code &
else
    python3 -m sglang.launch_server --model-path /model --served-model-name model --tp 8 --port 8000 --host 0.0.0.0 --trust-remote-code &
fi
# Wait for server to start up by polling the health endpoint
echo "Waiting for server to start..."
while true; do
    response=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:8000/health" || true)
    if [ "$response" -eq 200 ]; then
        echo "Server is up!"
        break
    fi
    echo "Server not ready yet, retrying in 10 seconds..."
    sleep 10
done


native_mpi_rank=$OMPI_COMM_WORLD_RANK
# Works with Slurm launching with `--mpi=pmix`
mpi_rank=${PMIX_RANK:-$native_mpi_rank}
echo "Rank: $mpi_rank"
echo "Counter: $_CURRENT_COUNTER"


if [ "$mpi_rank" -eq 0 ]; then
    start_shard=$((_CURRENT_COUNTER + 0))
    end_shard=$((_CURRENT_COUNTER + JOBS_PER_NODE - 1))

    for i in $(seq $start_shard $end_shard); do
        echo "Processing shard: $i"
        shard=$(printf "/input_data/train-%05d-%05d.jsonl" $i $i)
        OUTPUT=$(printf "/output_data/output-%05d-%05d.jsonl" $i $i)

        cmd="python3 /scripts/server_generate.py --data_path $shard --output_path $OUTPUT --num_threads 320 --max_tokens 4096 --temperature 0.0 --chat --log_empty_conversations"

        if [ -n "$SYSTEM_PROMPT" ]; then
            cmd+=" --system_prompt $SYSTEM_PROMPT"
        fi
        echo "Running command: $cmd"
        eval $cmd
    done
fi
