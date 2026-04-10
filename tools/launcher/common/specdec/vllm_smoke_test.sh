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

# Quick vLLM smoke test for speculative decoding (EAGLE3, DFlash, etc.).
# Launches server, sends a few test prompts, verifies responses, and shuts down.
#
# Required env vars:
#   HF_MODEL_CKPT  — target model path
#   DRAFT_MODEL    — draft model path
#
# Optional env vars:
#   SPEC_METHOD     — speculative method: "eagle", "dflash", etc. (default: "eagle")
#   NUM_SPEC_TOKENS — number of speculative tokens (default: 15)
#   TP_SIZE         — tensor parallel size (default: 1)
#   VLLM_PORT       — server port (default: 8000)

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source ${SCRIPT_DIR}/../service_utils.sh 2>/dev/null || true

cleanup() { kill $SERVER_PID 2>/dev/null; sleep 2; kill -9 $SERVER_PID 2>/dev/null; }
trap cleanup EXIT

MODEL=${HF_MODEL_CKPT}
DRAFT=${DRAFT_MODEL}
METHOD=${SPEC_METHOD:-eagle}
NUM_SPEC=${NUM_SPEC_TOKENS:-15}
PORT=${VLLM_PORT:-8000}
TP=${TP_SIZE:-1}

echo "=== vLLM Speculative Decoding Smoke Test ==="
echo "Method: ${METHOD}"
echo "Target: ${MODEL}"
echo "Draft:  ${DRAFT}"
echo "Spec tokens: ${NUM_SPEC}, TP: ${TP}"

# Build speculative config
SPEC_CONFIG="{\"method\": \"${METHOD}\", \"model\": \"${DRAFT}\", \"num_speculative_tokens\": ${NUM_SPEC}}"

# Start vLLM server
vllm serve ${MODEL} \
    --speculative-config "${SPEC_CONFIG}" \
    --max-num-batched-tokens 32768 \
    --tensor-parallel-size ${TP} \
    --port ${PORT} \
    &
SERVER_PID=$!

# Wait for server
echo "Waiting for vLLM server..."
for i in $(seq 1 180); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
        echo "Server ready after ${i}s"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "ERROR: Server died"; wait $SERVER_PID; exit 1
    fi
    sleep 1
done

if ! curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
    echo "ERROR: Server timeout"; exit 1
fi

# Run quick test prompts
echo ""
echo "=== Test Prompts ==="
PASS=0
FAIL=0
for PROMPT in \
    "What is 2+3? Answer with just the number." \
    "Write a haiku about mountains." \
    "Explain what a CPU is in one sentence."; do
    RESPONSE=$(curl -s http://localhost:${PORT}/v1/completions \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"${MODEL}\", \"prompt\": \"${PROMPT}\", \"max_tokens\": 64, \"temperature\": 0}" \
        | python3 -c "import json,sys; r=json.load(sys.stdin); t=r.get('choices',[{}])[0].get('text',''); u=r.get('usage',{}); print(f'{t.strip()[:100]}|||{u.get(\"completion_tokens\",0)}')" 2>/dev/null)
    TEXT=$(echo "$RESPONSE" | cut -d'|||' -f1)
    TOKENS=$(echo "$RESPONSE" | cut -d'|||' -f2)
    if [ -n "$TEXT" ] && [ "$TOKENS" -gt 0 ] 2>/dev/null; then
        echo "  PASS: \"${PROMPT}\" → ${TOKENS} tokens"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: \"${PROMPT}\" → empty or error"
        FAIL=$((FAIL + 1))
    fi
done

echo ""
echo "Results: ${PASS} passed, ${FAIL} failed"

if [ $FAIL -gt 0 ]; then
    echo "ERROR: Some prompts failed"
    exit 1
fi

echo "Done"
