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

# Quick vLLM smoke test for speculative decoding (EAGLE3, DFlash, MTP, etc.).
# Launches server, sends a few test prompts, verifies responses, and shuts down.
#
# Required env vars:
#   HF_MODEL_CKPT  — target model path
#
# Optional env vars:
#   DRAFT_MODEL     — draft model path (not needed for MTP)
#   SPEC_METHOD     — speculative method: "eagle", "dflash", "mtp", etc. (default: "eagle")
#   NUM_SPEC_TOKENS — number of speculative tokens (default: 15)
#   TP_SIZE         — tensor parallel size (default: 1)
#   VLLM_PORT       — server port (default: 8000)
#   REASONING_PARSER — reasoning parser (e.g., "qwen3" for Qwen3.5)
#   DISABLE_PREFIX_CACHING — set to "1" to disable prefix caching

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source ${SCRIPT_DIR}/../service_utils.sh 2>/dev/null || true

# Ensure pandas is available (missing in some vLLM nightly builds)
pip install pandas 2>/dev/null || true

cleanup() { kill $SERVER_PID 2>/dev/null; sleep 2; kill -9 $SERVER_PID 2>/dev/null; rm -f "${VLLM_LOG:-}" 2>/dev/null; }
trap cleanup EXIT

MODEL=${HF_MODEL_CKPT}
DRAFT=${DRAFT_MODEL:-}
# Auto-detect exported checkpoint from training output dir
if [ -z "$DRAFT" ] && [ -n "${DRAFT_CKPT_DIR:-}" ]; then
    DRAFT=$(ls -d ${DRAFT_CKPT_DIR}/exported-checkpoint-* 2>/dev/null | sort -t- -k3 -n | tail -1)
    if [ -n "$DRAFT" ]; then
        echo "Auto-detected draft model: ${DRAFT}"
    fi
fi
METHOD=${SPEC_METHOD:-eagle}
NUM_SPEC=${NUM_SPEC_TOKENS:-15}
PORT=${VLLM_PORT:-8000}
TP=${TP_SIZE:-1}

echo "=== vLLM Speculative Decoding Smoke Test ==="
echo "Method: ${METHOD}"
echo "Target: ${MODEL}"
echo "Draft:  ${DRAFT:-none (self-draft)}"
echo "Spec tokens: ${NUM_SPEC}, TP: ${TP}"

# Build speculative config: include "model" only if DRAFT_MODEL is set
if [ -n "$DRAFT" ] && [ "$DRAFT" != "none" ]; then
    SPEC_CONFIG="{\"method\": \"${METHOD}\", \"model\": \"${DRAFT}\", \"num_speculative_tokens\": ${NUM_SPEC}}"
else
    # Self-draft methods (MTP, Medusa) — no separate draft model
    SPEC_CONFIG="{\"method\": \"${METHOD}\", \"num_speculative_tokens\": ${NUM_SPEC}}"
fi

# Build optional args
OPTIONAL_ARGS=""
if [ -n "${REASONING_PARSER:-}" ]; then
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --reasoning-parser ${REASONING_PARSER}"
fi
if [ "${DISABLE_PREFIX_CACHING:-}" = "1" ]; then
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --no-enable-prefix-caching"
fi

# Start vLLM server (capture output for regression check parsing)
VLLM_LOG=$(mktemp /tmp/vllm_server_XXXXXX.log)
if [ -n "$SPEC_CONFIG" ]; then
    vllm serve ${MODEL} \
        --speculative-config "${SPEC_CONFIG}" \
        --max-num-batched-tokens 32768 \
        --tensor-parallel-size ${TP} \
        --port ${PORT} \
        ${OPTIONAL_ARGS} \
        > >(tee -a "$VLLM_LOG") 2>&1 &
else
    vllm serve ${MODEL} \
        --max-num-batched-tokens 32768 \
        --tensor-parallel-size ${TP} \
        --port ${PORT} \
        ${OPTIONAL_ARGS} \
        > >(tee -a "$VLLM_LOG") 2>&1 &
fi
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

# Run quick test prompts using chat completions API
MAX_TOKENS=${MAX_OUTPUT_TOKENS:-1024}
echo ""
echo "=== Test Prompts (max_tokens=${MAX_TOKENS}) ==="
PASS=0
FAIL=0
TOTAL_TOKENS=0
TOTAL_TIME=0
# 8 prompts mimicking MT-Bench categories: writing, roleplay, reasoning,
# math, coding, extraction, stem, humanities
for PROMPT in \
    "Write a persuasive email to your manager requesting a four-day work week. Include at least three supporting arguments." \
    "You are a medieval blacksmith. A traveler asks you to forge a sword. Describe your process and the qualities of your finest work." \
    "A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left? Explain your reasoning carefully." \
    "Solve the equation 3x + 7 = 22. Show each step of your solution." \
    "Write a Python function that takes a list of integers and returns the second largest unique value. Include error handling." \
    "Extract all the dates, names, and locations from: On March 15 2024 Dr. Alice Chen presented her findings at the Berlin Conference on Climate Science." \
    "Explain the process of photosynthesis. What role does chlorophyll play and why are plants green?" \
    "Discuss the main themes in George Orwells 1984. How do they relate to modern society?"; do
    START=$(date +%s%N)
    RESULT=$(curl -s http://localhost:${PORT}/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"${MODEL}\", \"messages\": [{\"role\": \"user\", \"content\": \"${PROMPT}\"}], \"max_tokens\": ${MAX_TOKENS}, \"temperature\": 0}" \
        2>/dev/null)
    END=$(date +%s%N)
    ELAPSED=$(echo "scale=2; ($END - $START) / 1000000000" | bc 2>/dev/null || echo "0")
    TOKENS=$(echo "$RESULT" | python3 -c "import json,sys; r=json.load(sys.stdin); print(r.get('usage',{}).get('completion_tokens',0))" 2>/dev/null)
    if [ -n "$TOKENS" ] && [ "$TOKENS" -gt 0 ] 2>/dev/null; then
        TPS=$(echo "scale=1; $TOKENS / $ELAPSED" | bc 2>/dev/null || echo "?")
        echo "  PASS: ${TOKENS} tokens in ${ELAPSED}s (${TPS} tok/s) — \"${PROMPT:0:50}...\""
        PASS=$((PASS + 1))
        TOTAL_TOKENS=$((TOTAL_TOKENS + TOKENS))
        TOTAL_TIME=$(echo "$TOTAL_TIME + $ELAPSED" | bc 2>/dev/null || echo "0")
    else
        echo "  FAIL: \"${PROMPT}\""
        echo "  Response: $(echo "$RESULT" | head -c 200)"
        FAIL=$((FAIL + 1))
    fi
done

echo ""
echo "Results: ${PASS} passed, ${FAIL} failed"
if [ "$TOTAL_TOKENS" -gt 0 ] 2>/dev/null; then
    AVG_TPS=$(echo "scale=1; $TOTAL_TOKENS / $TOTAL_TIME" | bc 2>/dev/null || echo "?")
    echo "Total: ${TOTAL_TOKENS} tokens in ${TOTAL_TIME}s (${AVG_TPS} tok/s avg)"
fi

# Fetch speculative decoding metrics if available
echo ""
METRICS=$(curl -s http://localhost:${PORT}/metrics 2>/dev/null | grep -i "spec\|accept\|draft" | head -10)
if [ -n "$METRICS" ]; then
    echo "=== Speculative Decoding Metrics ==="
    echo "$METRICS"
fi

if [ $FAIL -gt 0 ]; then
    echo "ERROR: Some prompts failed"
    exit 1
fi

# Regression check: minimum acceptance length for speculative decoding
if [ -n "${MIN_ACCEPTANCE_LENGTH:-}" ]; then
    # Parse mean acceptance length from vLLM's SpecDecoding metrics log.
    # vLLM logs: "SpecDecoding metrics: Mean acceptance length: X.XX, ..."
    # Take the last reported value (most accurate, covers all prompts).
    AVG_ACCEPT=$(grep -oP 'Mean acceptance length: \K[0-9.]+' "$VLLM_LOG" 2>/dev/null | tail -1 || true)
    if [ -n "$AVG_ACCEPT" ]; then
        echo ""
        echo "=== Acceptance Length Regression Check ==="
        echo "  Mean acceptance length: ${AVG_ACCEPT}"
        echo "  Threshold: ${MIN_ACCEPTANCE_LENGTH}"
        PASS_CHECK=$(python3 -c "print('yes' if float('${AVG_ACCEPT}') >= float('${MIN_ACCEPTANCE_LENGTH}') else 'no')")
        if [ "$PASS_CHECK" = "yes" ]; then
            echo "  PASS: ${AVG_ACCEPT} >= ${MIN_ACCEPTANCE_LENGTH}"
        else
            echo "  REGRESSION: ${AVG_ACCEPT} < ${MIN_ACCEPTANCE_LENGTH}"
            exit 1
        fi
    else
        echo "WARNING: Could not parse acceptance length from vLLM log, skipping regression check"
    fi
fi

echo "Done"
