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

_CURRENT_COUNTER="$1"
BACKEND="$2"
JOBS_PER_NODE="$3"
NUM_FRAMES="${4:-32}"
SYSTEM_PROMPT="${5:-}"

if [ "$BACKEND" != "sglang" ]; then
    echo "Multimodal generation currently uses the SGLang native video client; backend must be sglang."
    exit 1
fi

if [ "${INSTALL_OPENCV_HEADLESS:-0}" = "1" ]; then
    python3 -c "import cv2" || python3 -m pip install --user opencv-python-headless
fi

BASE_PORT=${BASE_PORT:-8000}
SGLANG_TP_SIZE=${SGLANG_TP_SIZE:-8}
NUM_TEMPERATURES=${NUM_TEMPERATURES:-8}
API_MODE=${API_MODE:-openai}
MEDIA_HTTP_PORT=${MEDIA_HTTP_PORT:-18080}
MEDIA_URL_BASE=${MEDIA_URL_BASE:-http://127.0.0.1:${MEDIA_HTTP_PORT}}
STARTUP_TIMEOUT_SECONDS=${STARTUP_TIMEOUT_SECONDS:-600}
GPU_COUNT=${GPU_COUNT:-$(nvidia-smi -L 2>/dev/null | wc -l || true)}

if [ "$GPU_COUNT" -le 0 ]; then
    echo "ERROR: no GPUs are visible inside the container." >&2
    exit 1
fi
if [ "$SGLANG_TP_SIZE" -gt "$GPU_COUNT" ]; then
    echo "ERROR: SGLANG_TP_SIZE=$SGLANG_TP_SIZE exceeds GPU_COUNT=$GPU_COUNT." >&2
    exit 1
fi
if [ "$SGLANG_TP_SIZE" -eq 1 ] && [ "$NUM_TEMPERATURES" -gt "$GPU_COUNT" ]; then
    echo "ERROR: NUM_TEMPERATURES=$NUM_TEMPERATURES exceeds GPU_COUNT=$GPU_COUNT for single-GPU serving." >&2
    exit 1
fi
if ! [[ "$STARTUP_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: STARTUP_TIMEOUT_SECONDS must be a positive integer." >&2
    exit 1
fi

SERVER_PIDS=()
GENERATION_PIDS=()
MEDIA_HTTP_PID=""

cleanup() {
    local attempt pid still_running
    for pid in "${SERVER_PIDS[@]}" "${MEDIA_HTTP_PID:-}"; do
        [ -n "${pid:-}" ] && kill "$pid" 2>/dev/null || true
    done
    for attempt in $(seq 1 10); do
        still_running=0
        for pid in "${SERVER_PIDS[@]}" "${MEDIA_HTTP_PID:-}"; do
            if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
                still_running=1
            fi
        done
        [ "$still_running" -eq 0 ] && break
        sleep 1
    done
    for pid in "${SERVER_PIDS[@]}" "${MEDIA_HTTP_PID:-}"; do
        if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done
    for pid in "${SERVER_PIDS[@]}" "${MEDIA_HTTP_PID:-}"; do
        [ -n "${pid:-}" ] && wait "$pid" 2>/dev/null || true
    done
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [ "$API_MODE" = "openai" ]; then
    python3 -m http.server "$MEDIA_HTTP_PORT" --bind 127.0.0.1 --directory /media_data \
        >/tmp/multimodal_media_http_${MEDIA_HTTP_PORT}.log 2>&1 &
    MEDIA_HTTP_PID=$!

    echo "Waiting for media HTTP server at $MEDIA_URL_BASE..."
    for _ in $(seq 1 30); do
        if curl -fsS "$MEDIA_URL_BASE/" >/dev/null 2>&1; then
            echo "Media HTTP server is up."
            break
        fi
        sleep 1
    done
    if ! curl -fsS "$MEDIA_URL_BASE/" >/dev/null 2>&1; then
        echo "ERROR: media HTTP server did not start. See /tmp/multimodal_media_http_${MEDIA_HTTP_PORT}.log" >&2
        exit 1
    fi
fi

SERVER_PORTS=()
if [ "$SGLANG_TP_SIZE" -eq 1 ]; then
    for gpu in $(seq 0 $((NUM_TEMPERATURES - 1))); do
        port=$((BASE_PORT + gpu))
        SERVER_PORTS+=("$port")
        CUDA_VISIBLE_DEVICES=$gpu \
            python3 -m sglang.launch_server \
            --model-path /model \
            --served-model-name model \
            --tp 1 \
            --port "$port" \
            --host 0.0.0.0 \
            ${SGLANG_EXTRA_ARGS:-} \
            --trust-remote-code &
        SERVER_PIDS+=("$!")
    done
else
    gpu_list=$(seq -s, 0 $((SGLANG_TP_SIZE - 1)))
    SERVER_PORTS=("$BASE_PORT")
    CUDA_VISIBLE_DEVICES=$gpu_list \
        python3 -m sglang.launch_server \
        --model-path /model \
        --served-model-name model \
        --tp "$SGLANG_TP_SIZE" \
        --port "$BASE_PORT" \
        --host 0.0.0.0 \
        ${SGLANG_EXTRA_ARGS:-} \
        --trust-remote-code &
    SERVER_PIDS+=("$!")
fi

wait_for_servers() {
    local deadline=$((SECONDS + STARTUP_TIMEOUT_SECONDS))
    local index pid port ready response

    echo "Waiting for server to start..."
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
            if [ "$response" -eq 200 ]; then
                ready=$((ready + 1))
            fi
        done
        if [ "$ready" -eq "${#SERVER_PORTS[@]}" ]; then
            echo "All servers are up!"
            return 0
        fi
        echo "$ready/${#SERVER_PORTS[@]} servers ready, retrying in 5 seconds..."
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

if [ "$API_MODE" = "openai" ]; then
    NUM_THREADS=${NUM_THREADS:-4}
elif [ "$SGLANG_TP_SIZE" -eq 1 ]; then
    NUM_THREADS=${NUM_THREADS:-64}
else
    NUM_THREADS=${NUM_THREADS:-8}
fi
MAX_TOKENS=${MAX_TOKENS:-6144}
VISION_TOKEN_FORMAT=${VISION_TOKEN_FORMAT:-qwen_vl}
MODEL_NAME=${MODEL_NAME:-model}

if [ "$mpi_rank" -eq 0 ]; then
    start_shard=$((_CURRENT_COUNTER + 0))
    end_shard=$((_CURRENT_COUNTER + JOBS_PER_NODE - 1))

    run_temperature() {
        server_id=$1
        port=$2
        temperature=$(printf "0.%d" "$server_id")

        echo "Starting multimodal generation on port $port, temperature=$temperature"

        for i in $(seq $start_shard $end_shard); do
            echo "Temperature $temperature processing shard $i"

            shard=$(printf "/input_data/train-%05d-%05d.jsonl" "$i" "$i")
            if [ ! -s "$shard" ]; then
                echo "Skipping missing shard: $shard"
                continue
            fi

            output=$(printf \
                "/output_data/output-%05d-%05d-temp-%s.jsonl" \
                "$i" "$server_id" "$temperature")

            cmd=(
                python3 /scripts/distributed_generate/server_generate_vlm_sglang.py
                --data_path "$shard"
                --output_path "$output"
                --input_root /input_data
                --media_root /media_data
                --num_threads "$NUM_THREADS"
                --max_tokens "$MAX_TOKENS"
                --num_frames "$NUM_FRAMES"
                --temperature "$temperature"
                --api_mode "$API_MODE"
                --model_name "$MODEL_NAME"
                --media_url_base "$MEDIA_URL_BASE"
                --vision_token_format "$VISION_TOKEN_FORMAT"
                --log_empty_conversations
                --url "http://localhost:$port"
            )

            if [ "${OVERWRITE_OUTPUT:-0}" = "1" ]; then
                cmd+=(--overwrite)
            fi

            if [ -n "$SYSTEM_PROMPT" ]; then
                echo "WARNING: SYSTEM_PROMPT is ignored for multimodal generation; include it in the shard prompts instead."
            fi

            printf 'Running:'
            printf ' %q' "${cmd[@]}"
            printf '\n'
            "${cmd[@]}"
        done
    }

    if [ "$SGLANG_TP_SIZE" -eq 1 ]; then
        for server_id in $(seq 0 $((NUM_TEMPERATURES - 1))); do
            (
                port=$((BASE_PORT + server_id))
                run_temperature "$server_id" "$port"
            ) &
            GENERATION_PIDS+=("$!")
        done
        for generation_pid in "${GENERATION_PIDS[@]}"; do
            wait "$generation_pid"
        done
    else
        # A tensor-parallel VLM uses all GPUs for one server, so run the
        # temperature sweep sequentially against that server.
        for server_id in $(seq 0 $((NUM_TEMPERATURES - 1))); do
            run_temperature "$server_id" "$BASE_PORT"
        done
    fi
fi
