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

MODEL=${1:?Usage: fakequant_nemo_gym_eval.sh MODEL [QUANT_CFG] [BENCHMARK] [LIMIT]}
QUANT_CFG=${2:-E5M2_DEFAULT_CFG}
BENCHMARK=${3:-gpqa}
LIMIT=${4:-5}
PORT=${PORT:-8000}
NEMO_GYM_REV=${NEMO_GYM_REV:-a85670eb167ba9b48cc53a36a070eed815e6c40d}

REPO=./modules/Model-Optimizer
RESULTS_DIR="$PWD/nemo_gym_${BENCHMARK}"
SERVER_LOG="$PWD/vllm_fakequant_server.log"
GYM_LOG="$PWD/nemo_gym_servers.log"

if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
else
    echo "Neither python3 nor python is available in the workload container." >&2
    exit 127
fi

test -f "$MODEL/config.json"
test -f "$REPO/examples/vllm_serve/vllm_serve_fakequant.py"

"$PYTHON_BIN" -c \
    'import sys; assert sys.version_info >= (3, 12), "NeMo Gym requires Python 3.12 or newer"'
"$PYTHON_BIN" -m pip install --editable "$REPO"
"$PYTHON_BIN" -m pip install "uv>=0.9.30"

GYM_DIR=$(mktemp -d)
git -C "$GYM_DIR" init
git -C "$GYM_DIR" remote add origin https://github.com/NVIDIA-NeMo/Gym.git
git -C "$GYM_DIR" fetch --depth 1 origin "$NEMO_GYM_REV"
git -C "$GYM_DIR" checkout --detach FETCH_HEAD
uv sync --directory "$GYM_DIR" --frozen --no-dev

mkdir -p "$RESULTS_DIR"
(
    cd "$GYM_DIR"
    uv run --frozen gym eval prepare \
        --benchmark "$BENCHMARK" \
        '+hf_token=${oc.env:HF_TOKEN,null}'
)

export QUANT_CFG
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export OPENAI_API_KEY=${OPENAI_API_KEY:-token-abc123}
export PYTHONUNBUFFERED=1

"$PYTHON_BIN" "$REPO/examples/vllm_serve/vllm_serve_fakequant.py" \
    "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --enforce-eager \
    --gpu-memory-utilization 0.85 \
    --max-model-len 4096 \
    >"$SERVER_LOG" 2>&1 &
server_pid=$!
gym_pid=

cleanup() {
    if [[ -n "$gym_pid" ]]; then
        kill "$gym_pid" 2>/dev/null || true
        wait "$gym_pid" 2>/dev/null || true
    fi
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 120); do
    if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null; then
        break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
        tail -200 "$SERVER_LOG"
        exit 1
    fi
    sleep 5
done
curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null

(
    cd "$GYM_DIR"
    uv run --frozen gym env start \
        --benchmark "$BENCHMARK" \
        --model-type vllm_model \
        --model "$MODEL" \
        --model-url "http://127.0.0.1:$PORT/v1" \
        --model-api-key "$OPENAI_API_KEY"
) >"$GYM_LOG" 2>&1 &
gym_pid=$!

for _ in $(seq 1 120); do
    if grep -qE 'All [0-9]+ / [0-9]+ servers ready!' "$GYM_LOG"; then
        break
    fi
    if ! kill -0 "$gym_pid" 2>/dev/null; then
        tail -200 "$GYM_LOG"
        exit 1
    fi
    sleep 5
done
grep -qE 'All [0-9]+ / [0-9]+ servers ready!' "$GYM_LOG"

(
    cd "$GYM_DIR"
    uv run --frozen gym eval run --no-serve \
        --agent "${BENCHMARK}_mcqa_simple_agent" \
        --input "benchmarks/$BENCHMARK/data/${BENCHMARK}_diamond_benchmark.jsonl" \
        --output "$RESULTS_DIR/rollouts.jsonl" \
        --prompt-config benchmarks/prompts/eval/aai/mcq-4choices.yaml \
        --limit "$LIMIT" \
        --num-repeats 1 \
        --concurrency 1
)
