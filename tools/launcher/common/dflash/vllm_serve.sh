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

cleanup() { kill $SERVER_PID 2>/dev/null; sleep 2; kill -9 $SERVER_PID 2>/dev/null; }
trap cleanup EXIT

MODEL=${HF_MODEL_CKPT}
DRAFT=${DRAFT_MODEL}
NUM_SPEC=${NUM_SPEC_TOKENS:-15}
PORT=${VLLM_PORT:-8000}
MAX_TOKENS=${MAX_BATCHED_TOKENS:-32768}
TP=${TP_SIZE:-1}

echo "=== vLLM DFlash Speculative Decoding ==="
echo "Target: ${MODEL}"
echo "Draft:  ${DRAFT}"
echo "Spec tokens: ${NUM_SPEC}, TP: ${TP}"

# Start vLLM server in background
vllm serve ${MODEL} \
    --speculative-config "{\"method\": \"dflash\", \"model\": \"${DRAFT}\", \"num_speculative_tokens\": ${NUM_SPEC}}" \
    --max-num-batched-tokens ${MAX_TOKENS} \
    --tensor-parallel-size ${TP} \
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
from collections import defaultdict

with open('${BENCHMARK_PROMPTS}') as f:
    prompts = [json.loads(line) for line in f][:80]

url = 'http://localhost:${PORT}/v1/completions'
cat_results = defaultdict(lambda: {'tokens': [], 'times': []})

for i, p in enumerate(prompts):
    q = p.get('prompt', p.get('turns', [p.get('question', 'Hello')]))[0] if isinstance(p, dict) else str(p)
    cat = p.get('category', 'unknown') if isinstance(p, dict) else 'unknown'
    start = time.time()
    r = requests.post(url, json={
        'model': '${MODEL}',
        'prompt': q,
        'max_tokens': 1024,
        'temperature': 0,
    }).json()
    elapsed = time.time() - start
    n = r.get('usage', {}).get('completion_tokens', 0)
    cat_results[cat]['tokens'].append(n)
    cat_results[cat]['times'].append(elapsed)
    tps = n / elapsed if elapsed > 0 else 0
    print(f'  [{i+1}/{len(prompts)}] [{cat}] {n} tokens in {elapsed:.1f}s = {tps:.1f} tok/s')

print(f'\n=== Per-Category Results ===')
print(f'{\"Category\":>12} | {\"Prompts\":>7} | {\"Tokens\":>8} | {\"Time(s)\":>8} | {\"TPS\":>8}')
print('-' * 55)
all_tokens = 0
all_time = 0
for cat in sorted(cat_results):
    t = sum(cat_results[cat]['tokens'])
    s = sum(cat_results[cat]['times'])
    n = len(cat_results[cat]['tokens'])
    tps = t / s if s > 0 else 0
    all_tokens += t
    all_time += s
    print(f'{cat:>12} | {n:>7} | {t:>8} | {s:>8.1f} | {tps:>8.1f}')
print('-' * 55)
print(f'{\"ALL\":>12} | {sum(len(v[\"tokens\"]) for v in cat_results.values()):>7} | {all_tokens:>8} | {all_time:>8.1f} | {all_tokens/all_time:>8.1f}')
"
fi

# Shut down server
echo ""
echo "Shutting down server..."
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null || true

echo "Done"
