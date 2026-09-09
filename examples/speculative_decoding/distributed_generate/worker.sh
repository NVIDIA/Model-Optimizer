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

set -euo pipefail

_CURRENT_COUNTER="$1"
BACKEND="$2"
JOBS_PER_NODE="$3"
SYSTEM_PROMPT="$4"

BASE_PORT=${BASE_PORT:-8000}
SGLANG_TP_SIZE=${SGLANG_TP_SIZE:-1}
NUM_TEMPERATURES=${NUM_TEMPERATURES:-8}
MAX_TOKENS=${MAX_TOKENS:-4096}
MODEL_NAME=${MODEL_NAME:-model}
STARTUP_TIMEOUT_SECONDS=${STARTUP_TIMEOUT_SECONDS:-600}
GPU_COUNT=${GPU_COUNT:-$(nvidia-smi -L 2>/dev/null | wc -l)}

if [ "$BACKEND" != "vllm" ] && [ "$BACKEND" != "sglang" ]; then
    echo "ERROR: backend must be vllm or sglang, got: $BACKEND" >&2
    exit 1
fi
if [ "$GPU_COUNT" -le 0 ]; then
    echo "ERROR: no GPUs are visible inside the container." >&2
    exit 1
fi
if [ "$SGLANG_TP_SIZE" -le 0 ] || [ "$SGLANG_TP_SIZE" -gt "$GPU_COUNT" ]; then
    echo "ERROR: SGLANG_TP_SIZE=$SGLANG_TP_SIZE is invalid for GPU_COUNT=$GPU_COUNT." >&2
    exit 1
fi
if [ "$NUM_TEMPERATURES" -le 0 ]; then
    echo "ERROR: NUM_TEMPERATURES must be positive." >&2
    exit 1
fi
if [ "$SGLANG_TP_SIZE" -eq 1 ] && [ "$NUM_TEMPERATURES" -gt "$GPU_COUNT" ]; then
    echo "ERROR: NUM_TEMPERATURES=$NUM_TEMPERATURES exceeds GPU_COUNT=$GPU_COUNT for TP=1 serving." >&2
    exit 1
fi
if ! [[ "$STARTUP_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: STARTUP_TIMEOUT_SECONDS must be a positive integer." >&2
    exit 1
fi

if [ "$SGLANG_TP_SIZE" -eq 1 ]; then
    TEXT_NUM_THREADS=${TEXT_NUM_THREADS:-64}
else
    TEXT_NUM_THREADS=${TEXT_NUM_THREADS:-320}
fi

SERVER_PIDS=()
GENERATION_PIDS=()
SERVER_PORTS=()

cleanup() {
    local attempt pid still_running
    for pid in "${SERVER_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    for attempt in $(seq 1 10); do
        still_running=0
        for pid in "${SERVER_PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                still_running=1
            fi
        done
        [ "$still_running" -eq 0 ] && break
        sleep 1
    done
    for pid in "${SERVER_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done
    for pid in "${SERVER_PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [ "$SGLANG_TP_SIZE" -eq 1 ]; then
    for gpu in $(seq 0 $((NUM_TEMPERATURES - 1))); do
        port=$((BASE_PORT + gpu))
        SERVER_PORTS+=("$port")
        if [ "$BACKEND" = "vllm" ]; then
            CUDA_VISIBLE_DEVICES=$gpu vllm serve /model/ \
                --tensor-parallel-size 1 --served-model-name "$MODEL_NAME" \
                --port "$port" --host 0.0.0.0 --trust-remote-code &
        else
            CUDA_VISIBLE_DEVICES=$gpu python3 -m sglang.launch_server \
                --model-path /model --served-model-name "$MODEL_NAME" --tp 1 \
                --port "$port" --host 0.0.0.0 ${SGLANG_EXTRA_ARGS:-} --trust-remote-code &
        fi
        SERVER_PIDS+=("$!")
    done
else
    GPU_LIST=$(seq -s, 0 $((SGLANG_TP_SIZE - 1)))
    SERVER_PORTS=("$BASE_PORT")
    if [ "$BACKEND" = "vllm" ]; then
        CUDA_VISIBLE_DEVICES=$GPU_LIST vllm serve /model/ \
            --tensor-parallel-size "$SGLANG_TP_SIZE" --served-model-name "$MODEL_NAME" \
            --port "$BASE_PORT" --host 0.0.0.0 --trust-remote-code &
    else
        CUDA_VISIBLE_DEVICES=$GPU_LIST python3 -m sglang.launch_server \
            --model-path /model --served-model-name "$MODEL_NAME" --tp "$SGLANG_TP_SIZE" \
            --port "$BASE_PORT" --host 0.0.0.0 ${SGLANG_EXTRA_ARGS:-} --trust-remote-code &
    fi
    SERVER_PIDS+=("$!")
fi

wait_for_servers() {
    local deadline=$((SECONDS + STARTUP_TIMEOUT_SECONDS))
    local index pid port ready response

    echo "Waiting for ${#SERVER_PORTS[@]} server(s) to start..."
    while (( SECONDS < deadline )); do
        for index in "${!SERVER_PIDS[@]}"; do
            pid=${SERVER_PIDS[$index]}
            if ! kill -0 "$pid" 2>/dev/null; then
                echo "ERROR: server process $pid exited before becoming ready." >&2
                return 1
            fi
        done

        ready=0
        for port in "${SERVER_PORTS[@]}"; do
            response=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${port}/health" || true)
            [ "$response" = "200" ] && ready=$((ready + 1))
        done
        if [ "$ready" -eq "${#SERVER_PORTS[@]}" ]; then
            echo "All servers are up!"
            return 0
        fi
        echo "$ready/${#SERVER_PORTS[@]} servers ready; retrying in 5 seconds..."
        sleep 5
    done

    echo "ERROR: servers did not become ready within ${STARTUP_TIMEOUT_SECONDS} seconds." >&2
    return 1
}

wait_for_servers

native_mpi_rank=${OMPI_COMM_WORLD_RANK:-0}
mpi_rank=${PMIX_RANK:-$native_mpi_rank}
echo "Rank: $mpi_rank"
echo "Counter: $_CURRENT_COUNTER"

if [ "$mpi_rank" -eq 0 ]; then
    start_shard=$_CURRENT_COUNTER
    end_shard=$((_CURRENT_COUNTER + JOBS_PER_NODE - 1))

    run_temperature() {
        temp_id=$1
        port=$2
        temperature=$(printf "0.%d" "$temp_id")
        for i in $(seq "$start_shard" "$end_shard"); do
            echo "Temperature $temperature processing shard: $i on port $port"
            shard=$(printf "/input_data/train-%05d-%05d.jsonl" "$i" "$i")
            if [ ! -s "$shard" ]; then
                echo "Skipping missing shard: $shard"
                continue
            fi
            output=$(printf "/output_data/output-%05d-%05d-temp-%s.jsonl" "$i" "$i" "$temperature")
            cmd=(
                python3 /scripts/scripts/server_generate.py
                --data_path "$shard"
                --output_path "$output"
                --num_threads "$TEXT_NUM_THREADS"
                --max_tokens "$MAX_TOKENS"
                --temperature "$temperature"
                --url "http://localhost:$port/v1"
                --log_empty_conversations
            )
            if [ -n "$SYSTEM_PROMPT" ]; then
                cmd+=(--system_prompt "$SYSTEM_PROMPT")
            fi
            echo "Running: ${cmd[*]}"
            "${cmd[@]}"
        done
    }

    if [ "$SGLANG_TP_SIZE" -eq 1 ]; then
        for temp_id in $(seq 0 $((NUM_TEMPERATURES - 1))); do
            run_temperature "$temp_id" "$((BASE_PORT + temp_id))" &
            GENERATION_PIDS+=("$!")
        done
        for generation_pid in "${GENERATION_PIDS[@]}"; do
            wait "$generation_pid"
        done
    else
        for temp_id in $(seq 0 $((NUM_TEMPERATURES - 1))); do
            run_temperature "$temp_id" "$BASE_PORT"
        done
    fi
fi
