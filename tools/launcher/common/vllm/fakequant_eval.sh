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

MODEL=${1:?Usage: fakequant_eval.sh MODEL [QUANT_CFG] [TASKS] [LIMIT]}
QUANT_CFG=${2:-E5M2_DEFAULT_CFG}
TASKS=${3:-gsm8k}
LIMIT=${4:-20}
PORT=${PORT:-8000}

REPO=./modules/Model-Optimizer
SERVER_LOG="$PWD/vllm_fakequant_server.log"

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

"$PYTHON_BIN" -m pip install --editable "$REPO"
"$PYTHON_BIN" -m pip install "lm_eval>=0.4.8"

export QUANT_CFG
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export OPENAI_API_KEY=${OPENAI_API_KEY:-token-abc123}

"$PYTHON_BIN" "$REPO/examples/vllm_serve/vllm_serve_fakequant.py" \
    "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --enforce-eager \
    --gpu-memory-utilization 0.85 \
    --max-model-len 4096 \
    >"$SERVER_LOG" 2>&1 &
server_pid=$!

cleanup() {
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

lm_eval \
    --model local-completions \
    --tasks "$TASKS" \
    --limit "$LIMIT" \
    --batch_size 1 \
    --output_path "$PWD/lm_eval_fakequant" \
    --model_args \
    "model=$MODEL,base_url=http://127.0.0.1:$PORT/v1/completions,num_concurrent=1,max_retries=3,tokenized_requests=False,tokenizer_backend=None"
