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

# Launch vLLM server with DFlash speculative decoding, run benchmark, then shut down.
#
# Required env vars:
#   HF_MODEL_CKPT  — target model path
#   DRAFT_MODEL    — DFlash draft model path
#
# Optional env vars:
#   NUM_SPEC_TOKENS — number of speculative tokens (default: 15)
#   VLLM_PORT       — server port (default: 8000)
#   MAX_BATCHED_TOKENS — max batched tokens (default: 32768)
#   BENCHMARK_PROMPTS  — path to benchmark prompts file

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source ${SCRIPT_DIR}/../service_utils.sh 2>/dev/null || true

trap 'kill $SERVER_PID 2>/dev/null; exit' EXIT ERR

MODEL=${HF_MODEL_CKPT}
DRAFT=${DRAFT_MODEL}
NUM_SPEC=${NUM_SPEC_TOKENS:-15}
PORT=${VLLM_PORT:-8000}
MAX_TOKENS=${MAX_BATCHED_TOKENS:-32768}

echo "=== vLLM DFlash Speculative Decoding ==="
echo "Target: ${MODEL}"
echo "Draft:  ${DRAFT}"
echo "Spec tokens: ${NUM_SPEC}"

# Start vLLM server in background
vllm serve ${MODEL} \
    --speculative-config "{\"method\": \"dflash\", \"model\": \"${DRAFT}\", \"num_speculative_tokens\": ${NUM_SPEC}}" \
    --attention-backend flash_attn \
    --max-num-batched-tokens ${MAX_TOKENS} \
    --port ${PORT} \
    &
SERVER_PID=$!

# Wait for server to be ready
echo "Waiting for vLLM server to start..."
for i in $(seq 1 120); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
        echo "Server ready after ${i}s"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "ERROR: Server process died"
        wait $SERVER_PID
        exit 1
    fi
    sleep 1
done

if ! curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
    echo "ERROR: Server failed to start within 120s"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

# Run a quick test
echo ""
echo "=== Quick generation test ==="
curl -s http://localhost:${PORT}/v1/completions \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"${MODEL}\",
        \"prompt\": \"What is 2+3?\",
        \"max_tokens\": 64,
        \"temperature\": 0
    }" | python3 -c "import json,sys; r=json.load(sys.stdin); print(r.get('choices',[{}])[0].get('text','ERROR')[:200]); print(f'Usage: {r.get(\"usage\",{})}')"

# Run benchmark if prompts file provided
if [ -n "${BENCHMARK_PROMPTS}" ] && [ -f "${BENCHMARK_PROMPTS}" ]; then
    echo ""
    echo "=== MT-Bench Benchmark ==="
    python3 -c "
import json, time, requests

with open('${BENCHMARK_PROMPTS}') as f:
    prompts = [json.loads(line) for line in f][:20]

url = 'http://localhost:${PORT}/v1/completions'
times = []
tokens = []
for i, p in enumerate(prompts):
    q = p.get('turns', [p.get('question', 'Hello')])[0] if isinstance(p, dict) else str(p)
    start = time.time()
    r = requests.post(url, json={
        'model': '${MODEL}',
        'prompt': q,
        'max_tokens': 512,
        'temperature': 0,
    }).json()
    elapsed = time.time() - start
    n = r.get('usage', {}).get('completion_tokens', 0)
    times.append(elapsed)
    tokens.append(n)
    tps = n / elapsed if elapsed > 0 else 0
    print(f'  [{i+1}/{len(prompts)}] {n} tokens in {elapsed:.1f}s = {tps:.1f} tok/s')

total_tokens = sum(tokens)
total_time = sum(times)
print(f'\nTotal: {total_tokens} tokens in {total_time:.1f}s = {total_tokens/total_time:.1f} tok/s')
"
fi

# Shut down server
echo ""
echo "Shutting down server..."
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null || true

echo "Done"
