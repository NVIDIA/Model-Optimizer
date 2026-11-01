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
#   SERVER_TIMEOUT  — seconds to wait for the server to become healthy (default: 180). Large or
#                    hybrid targets need longer for compile, kernel JIT and CUDA graph capture
#   MIN_ACCEPTANCE_LENGTH — fail unless the mean acceptance length (from vLLM's /metrics
#                    counters) reaches this; also fails if no speculative drafts were recorded
#   SMOKE_JSON      — also write the result to this file as JSON, on every exit; "stage"
#                    records how far the run got (invalid_config, starting, draft_missing,
#                    server_died, server_timeout, serving, completed)

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

# Ensure pandas is available (missing in some vLLM nightly builds)
pip install pandas 2>/dev/null || true

MODEL=${HF_MODEL_CKPT}
DRAFT=${DRAFT_MODEL:-}
METHOD=${SPEC_METHOD:-eagle}
NUM_SPEC=${NUM_SPEC_TOKENS:-15}
SERVER_TIMEOUT=${SERVER_TIMEOUT:-180}
STAGE=starting
VERDICT=FAIL
REASON=""
PROMPTS_OK=0
PROMPTS_FAILED=0
TOTAL_TOKENS=0
AVG_TPS="?"
AVG_ACCEPT=""

write_smoke_json() {
    [ -n "${SMOKE_JSON:-}" ] || return 0
    python3 - "$SMOKE_JSON" "$STAGE" "$VERDICT" "$REASON" "$METHOD" "$MODEL" "${DRAFT:-}" "$NUM_SPEC" \
        "$PROMPTS_OK" "$PROMPTS_FAILED" "$TOTAL_TOKENS" "$AVG_TPS" "${AVG_ACCEPT:-}" "${MIN_ACCEPTANCE_LENGTH:-}" <<'EOF'
import json
import math
import os
import sys

(path, stage, verdict, reason, method, target, draft, num_spec, n_pass, n_fail, tokens, tps, al,
 min_al) = sys.argv[1:]


def num(value):
    try:
        value = float(value)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def count(value):
    return int(value) if value.isdigit() else None


result = {
    "stage": stage,
    "pass": verdict == "PASS",
    "reason": reason or None,
    "method": method,
    "target": target,
    "draft": draft or None,
    "num_speculative_tokens": count(num_spec),
    "prompts_passed": int(n_pass),
    "prompts_failed": int(n_fail),
    "total_tokens": int(tokens),
    "avg_tok_s": num(tps),
    "mean_acceptance_length": num(al),
    "min_acceptance_length": num(min_al),
}
# A smoke-only experiment has nothing else that creates the directory.
os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
with open(path, "w") as f:
    json.dump(result, f, indent=2)
EOF
}

# A result left by an earlier run must never be read as this run's.
[ -n "${SMOKE_JSON:-}" ] && rm -f "$SMOKE_JSON"
cleanup() {
    write_smoke_json
    if [ -n "${SERVER_PID:-}" ]; then kill $SERVER_PID 2>/dev/null; sleep 2; kill -9 $SERVER_PID 2>/dev/null; fi
    rm -f "${VLLM_LOG:-}" 2>/dev/null
}
trap cleanup EXIT

# Checked after the trap, so a bad value still writes this run's SMOKE_JSON.
if [ -n "${MIN_ACCEPTANCE_LENGTH:-}" ] && ! python3 -c 'import math, sys; sys.exit(not math.isfinite(float(sys.argv[1])))' "$MIN_ACCEPTANCE_LENGTH" 2>/dev/null; then
    REASON="MIN_ACCEPTANCE_LENGTH='${MIN_ACCEPTANCE_LENGTH}' is not a finite number"
# Canonical form only: NUM_SPEC goes into SPEC_CONFIG as a bare JSON number, so "08" breaks it.
elif ! [[ "$NUM_SPEC" =~ ^[1-9][0-9]*$ ]]; then
    REASON="NUM_SPEC_TOKENS='${NUM_SPEC}' is not a positive integer"
elif ! [[ "$SERVER_TIMEOUT" =~ ^[1-9][0-9]*$ ]]; then
    REASON="SERVER_TIMEOUT='${SERVER_TIMEOUT}' is not a positive integer"
fi
if [ -n "$REASON" ]; then
    STAGE=invalid_config; echo "ERROR: $REASON"; exit 1
fi

# source returns the status of service_utils.sh's last command, an unrelated rank-0 append.
[ -f "${SCRIPT_DIR}/../service_utils.sh" ] || { REASON="cannot find service_utils.sh"; echo "ERROR: $REASON"; exit 1; }
source "${SCRIPT_DIR}/../service_utils.sh"

# Auto-detect exported checkpoint from training output dir
if [ -z "$DRAFT" ] && [ -n "${DRAFT_CKPT_DIR:-}" ]; then
    DRAFT=$(latest_checkpoint "${DRAFT_CKPT_DIR}" exported-checkpoint-)
    if [ -n "$DRAFT" ]; then
        echo "Auto-detected draft model: ${DRAFT}"
    else
        # Fail fast: caller asked for a draft from training output, but training
        # produced nothing. Without this, we'd fall through to the self-draft
        # branch and vLLM would reject the config with a confusing
        # "num_speculative_tokens was provided but without speculative model" error.
        echo "ERROR: DRAFT_CKPT_DIR=${DRAFT_CKPT_DIR} contains no exported-checkpoint-* directory."
        echo "       The upstream training task likely failed to produce a draft model."
        STAGE=draft_missing
        REASON="DRAFT_CKPT_DIR=${DRAFT_CKPT_DIR} contains no exported-checkpoint-* directory"
        exit 1
    fi
fi
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
for i in $(seq 1 "$SERVER_TIMEOUT"); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
        echo "Server ready after ${i}s"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "ERROR: Server died"; wait $SERVER_PID
        STAGE=server_died; REASON="vLLM server exited before becoming healthy"; exit 1
    fi
    sleep 1
done

if ! curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
    echo "ERROR: Server timeout"
    STAGE=server_timeout; REASON="vLLM server not healthy after ${SERVER_TIMEOUT}s"; exit 1
fi
STAGE=serving
REASON="exited while sending test prompts"

# Run quick test prompts using chat completions API
MAX_TOKENS=${MAX_OUTPUT_TOKENS:-1024}
echo ""
echo "=== Test Prompts (max_tokens=${MAX_TOKENS}) ==="
TOTAL_NS=0
ratio() { python3 -c 'import sys; a, b = float(sys.argv[1]), float(sys.argv[2]); print(f"{a / b:.{sys.argv[3]}f}" if b else "?")' "$@"; }
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
    ELAPSED_NS=$((END - START))
    ELAPSED=$(ratio "$ELAPSED_NS" 1e9 2)
    TOKENS=$(echo "$RESULT" | python3 -c "import json,sys; r=json.load(sys.stdin); print(r.get('usage',{}).get('completion_tokens',0))" 2>/dev/null)
    if [ -n "$TOKENS" ] && [ "$TOKENS" -gt 0 ] 2>/dev/null; then
        TPS=$(ratio "${TOKENS}e9" "$ELAPSED_NS" 1)
        echo "  PASS: ${TOKENS} tokens in ${ELAPSED}s (${TPS} tok/s) — \"${PROMPT:0:50}...\""
        PROMPTS_OK=$((PROMPTS_OK + 1))
        TOTAL_TOKENS=$((TOTAL_TOKENS + TOKENS))
        TOTAL_NS=$((TOTAL_NS + ELAPSED_NS))
    else
        echo "  FAIL: \"${PROMPT}\""
        echo "  Response: $(echo "$RESULT" | head -c 200)"
        PROMPTS_FAILED=$((PROMPTS_FAILED + 1))
    fi
done

echo ""
echo "Results: ${PROMPTS_OK} passed, ${PROMPTS_FAILED} failed"
if [ "$TOTAL_TOKENS" -gt 0 ] 2>/dev/null; then
    AVG_TPS=$(ratio "${TOTAL_TOKENS}e9" "$TOTAL_NS" 1)
    echo "Total: ${TOTAL_TOKENS} tokens in $(ratio "$TOTAL_NS" 1e9 2)s (${AVG_TPS} tok/s avg)"
fi

# Fetch speculative decoding metrics if available
echo ""
ALL_METRICS=$(curl -s http://localhost:${PORT}/metrics 2>/dev/null || true)
METRICS=$(echo "$ALL_METRICS" | grep -i "spec\|accept\|draft" | head -10)
if [ -n "$METRICS" ]; then
    echo "=== Speculative Decoding Metrics ==="
    echo "$METRICS"
fi

# Mean acceptance length over all prompts, from vLLM's cumulative Prometheus counters:
# 1 + accepted draft tokens / drafts (the bonus token counts, as in vLLM's own log). The
# "SpecDecoding metrics: Mean acceptance length" log line covers only the last logging
# interval (vLLM resets its stats after each log), so it is not used. Printed twice: full
# precision for the floor check and SMOKE_JSON, rounded for the log.
AL_OUT=$(echo "$ALL_METRICS" | python3 -c '
import sys

drafts = accepted = 0.0
for line in sys.stdin:
    name = line.split("{")[0].split(" ")[0]
    if name == "vllm:spec_decode_num_drafts_total":
        drafts += float(line.split()[-1])
    elif name == "vllm:spec_decode_num_accepted_tokens_total":
        accepted += float(line.split()[-1])
if drafts > 0:
    al = 1 + accepted / drafts
    print(repr(al), f"{al:.4f}")
' 2>/dev/null || true)
read -r AVG_ACCEPT AL_SHOWN <<< "$AL_OUT"

STAGE=completed
VERDICT=PASS
REASON=""
if [ $PROMPTS_FAILED -gt 0 ]; then
    VERDICT=FAIL
    REASON="${PROMPTS_FAILED} of $((PROMPTS_OK + PROMPTS_FAILED)) prompts failed"
elif [ -n "${MIN_ACCEPTANCE_LENGTH:-}" ]; then
    echo ""
    echo "=== Acceptance Length Regression Check ==="
    echo "  Mean acceptance length: ${AL_SHOWN:-not reported}"
    echo "  Threshold: ${MIN_ACCEPTANCE_LENGTH}"
    if [ -z "$AVG_ACCEPT" ]; then
        # A threshold that was never evaluated must not pass.
        VERDICT=FAIL
        REASON="REGRESSION: MIN_ACCEPTANCE_LENGTH=${MIN_ACCEPTANCE_LENGTH} was set but vLLM /metrics reports no speculative drafts"
    elif python3 -c 'import sys; sys.exit(float(sys.argv[1]) < float(sys.argv[2]))' "$AVG_ACCEPT" "$MIN_ACCEPTANCE_LENGTH"; then
        echo "  PASS: ${AL_SHOWN} >= ${MIN_ACCEPTANCE_LENGTH}"
    else
        VERDICT=FAIL
        REASON="REGRESSION: ${AL_SHOWN} < ${MIN_ACCEPTANCE_LENGTH}"
    fi
fi

if [ "$VERDICT" = FAIL ]; then
    echo "  $REASON"
    exit 1
fi

echo "Done"
