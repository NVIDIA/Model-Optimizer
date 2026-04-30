#!/bin/bash
# SGLang Speculative Decoding Smoke Test
#
# Starts python -m sglang.launch_server with MTP enabled (EAGLE algorithm +
# SGLANG_ENABLE_SPEC_V2=1), sends 8 test prompts via the OpenAI-compatible
# API, and validates that every prompt returns a non-empty response.
#
# Environment variables (all optional with defaults):
#   HF_MODEL_CKPT         — model path (default: /hf-local/deepseek-ai/DeepSeek-V4-Flash)
#   NUM_SPEC_TOKENS        — speculative draft tokens (default: 1)
#   DATA_PARALLEL_SIZE     — DP size (default: 8)
#   TP_SIZE                — TP size (default: 1)
#   KV_CACHE_DTYPE         — e.g. "fp8_e5m2" or "fp8" (default: unset = auto)
#   TRUST_REMOTE_CODE      — "1" to pass --trust-remote-code
#   COPY_MODEL_TO_TMPFS    — "1" to rsync model to /dev/shm before loading
#   EXPERT_PARALLEL_SIZE   — expert parallelism degree (default: unset = no EP)
#   ATTENTION_BACKEND      — e.g. "trtllm_mha" for Blackwell (default: unset = auto)
#   MOE_BACKEND            — e.g. "flashinfer_trtllm" for Blackwell (default: unset = auto)
#   SGLANG_PORT            — server port (default: 8000)
#   SERVER_TIMEOUT         — seconds to wait for server ready (default: 900)
#   MAX_OUTPUT_TOKENS      — max tokens per query (default: 1024)
#   MIN_ACCEPTANCE_LENGTH  — optional regression threshold for mean acceptance length
#   SGLANG_EXTRA_ARGS      — any extra flags appended verbatim to launch_server

set -euo pipefail

MODEL=${HF_MODEL_CKPT:-/hf-local/deepseek-ai/DeepSeek-V4-Flash}
NUM_SPEC=${NUM_SPEC_TOKENS:-1}
PORT=${SGLANG_PORT:-8000}
DP=${DATA_PARALLEL_SIZE:-8}
TP=${TP_SIZE:-1}

# ── tmpfs copy ────────────────────────────────────────────────────────────────
TMPFS_MODEL=""
cleanup() {
    kill "$SERVER_PID" 2>/dev/null || true
    sleep 2
    kill -9 "$SERVER_PID" 2>/dev/null || true
    if [ -n "$TMPFS_MODEL" ] && [ -d "$TMPFS_MODEL" ]; then
        echo "Removing tmpfs model copy: $TMPFS_MODEL"
        rm -rf "$TMPFS_MODEL"
    fi
}
trap cleanup EXIT

if [ "${COPY_MODEL_TO_TMPFS:-0}" = "1" ]; then
    MODEL_NAME=$(basename "$MODEL")
    TMPFS_MODEL="/dev/shm/${MODEL_NAME}"
    if [ -d "$TMPFS_MODEL" ] && [ -f "$TMPFS_MODEL/config.json" ]; then
        echo "Using existing tmpfs model copy: $TMPFS_MODEL"
    else
        MODEL_SIZE=$(du -sh "$MODEL" 2>/dev/null | cut -f1 || echo "?")
        AVAIL_SHM=$(df -h /dev/shm 2>/dev/null | tail -1 | awk '{print $4}' || echo "?")
        echo "Copying model to /dev/shm (${MODEL_SIZE}, available: ${AVAIL_SHM})..."
        cp -r "$MODEL" "$TMPFS_MODEL"
        echo "Model copy done: $TMPFS_MODEL"
    fi
    MODEL="$TMPFS_MODEL"
    echo "Loading from tmpfs: $MODEL"
fi

# ── container patches ─────────────────────────────────────────────────────────
# Upgrade transformers so newly-registered model types (e.g. deepseek_v4) are
# available without requiring trust_remote_code in the AutoConfig pre-check path.
echo "Upgrading transformers (--pre for deepseek_v4 support)..."
pip install --upgrade --pre transformers -q || echo "WARNING: transformers upgrade failed, continuing"

# Register deepseek_v4 in HF Transformers via a site-packages .pth startup file.
# deepseek_v4 is not in the stable transformers release; the stub class preserves
# all config.json fields (including `architectures`) so SGLang's model registry works.
# The .pth propagates to every spawned worker process automatically.
python3 << 'PYEOF'
import os, site

STUB = r'''
try:
    from transformers import AutoConfig, PretrainedConfig
    class DeepseekV4Config(PretrainedConfig):
        model_type = "deepseek_v4"
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                object.__setattr__(self, k, v)
            super().__init__(**kwargs)
    AutoConfig.register("deepseek_v4", DeepseekV4Config, exist_ok=True)
    print("[patch] deepseek_v4 registered in AutoConfig")
except Exception as e:
    print(f"[patch] deepseek_v4 registration failed: {e}")
'''

for sp in site.getsitepackages() + [site.getusersitepackages()]:
    if not os.path.isdir(sp):
        continue
    try:
        with open(os.path.join(sp, '_deepseek_v4_patch.py'), 'w') as f:
            f.write(STUB)
        with open(os.path.join(sp, 'deepseek_v4.pth'), 'w') as f:
            f.write('import _deepseek_v4_patch\n')
        print(f"[patch] Wrote deepseek_v4.pth to {sp}")
        break
    except Exception as e:
        print(f"[patch] Could not write to {sp}: {e}")

exec(STUB)
PYEOF

GPU_CC=$(python3 -c "import torch; cc=torch.cuda.get_device_capability(); print(f'{cc[0]}.{cc[1]}')" 2>/dev/null || echo "unknown")
echo "GPU compute capability: ${GPU_CC}"

# ── build args ────────────────────────────────────────────────────────────────
EXTRA_ARGS=""
[ -n "${KV_CACHE_DTYPE:-}" ]      && EXTRA_ARGS="$EXTRA_ARGS --kv-cache-dtype ${KV_CACHE_DTYPE}"
[ "${TRUST_REMOTE_CODE:-}" = "1" ] && EXTRA_ARGS="$EXTRA_ARGS --trust-remote-code"
[ -n "${EXPERT_PARALLEL_SIZE:-}" ]  && EXTRA_ARGS="$EXTRA_ARGS --expert-parallel-size ${EXPERT_PARALLEL_SIZE}"
[ -n "${ATTENTION_BACKEND:-}" ]    && EXTRA_ARGS="$EXTRA_ARGS --attention-backend ${ATTENTION_BACKEND}"
[ -n "${MOE_BACKEND:-}" ]          && EXTRA_ARGS="$EXTRA_ARGS --moe-runner-backend ${MOE_BACKEND}"
[ -n "${SGLANG_EXTRA_ARGS:-}" ]    && EXTRA_ARGS="$EXTRA_ARGS ${SGLANG_EXTRA_ARGS}"

# ── start server ──────────────────────────────────────────────────────────────
echo "=== SGLang Speculative Decoding Smoke Test ==="
echo "Model:       ${MODEL}"
echo "DP: ${DP}, TP: ${TP}, Spec tokens: ${NUM_SPEC}"

# Speculative decoding (EAGLE MTP) — skip when NUM_SPEC_TOKENS=0
SPEC_ARGS=""
if [ "${NUM_SPEC}" -gt 0 ]; then
    export SGLANG_ENABLE_SPEC_V2=1
    SPEC_ARGS="--speculative-num-draft-tokens ${NUM_SPEC}"
fi

# shellcheck disable=SC2086
python -m sglang.launch_server \
    --model-path "${MODEL}" \
    --tp "${TP}" \
    --dp "${DP}" \
    --enable-dp-attention \
    --host 0.0.0.0 \
    --port "${PORT}" \
    ${SPEC_ARGS} \
    ${EXTRA_ARGS} \
    &
SERVER_PID=$!

# ── wait for ready ────────────────────────────────────────────────────────────
SERVER_TIMEOUT=${SERVER_TIMEOUT:-900}
echo "Waiting for SGLang server (timeout: ${SERVER_TIMEOUT}s)..."
for i in $(seq 1 "${SERVER_TIMEOUT}"); do
    if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "Server ready after ${i}s"
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: Server died"
        wait "$SERVER_PID" || true
        exit 1
    fi
    sleep 1
done

if ! curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
    echo "ERROR: Server did not become ready within ${SERVER_TIMEOUT}s"
    exit 1
fi

# ── test prompts ──────────────────────────────────────────────────────────────
MAX_TOKENS=${MAX_OUTPUT_TOKENS:-1024}
echo ""
echo "=== Test Prompts (max_tokens=${MAX_TOKENS}) ==="
PASS=0
FAIL=0
TOTAL_TOKENS=0
TOTAL_TIME=0

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
    RESULT=$(curl -s "http://localhost:${PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"${MODEL}\", \"messages\": [{\"role\": \"user\", \"content\": \"${PROMPT}\"}], \"max_tokens\": ${MAX_TOKENS}, \"temperature\": 0}" \
        2>/dev/null)
    END=$(date +%s%N)
    ELAPSED=$(echo "scale=2; ($END - $START) / 1000000000" | bc 2>/dev/null || echo "0")
    TOKENS=$(echo "$RESULT" | python3 -c "import json,sys; r=json.load(sys.stdin); print(r.get('usage',{}).get('completion_tokens',0))" 2>/dev/null || echo "0")
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

# ── speculative metrics ───────────────────────────────────────────────────────
echo ""
METRICS=$(curl -s "http://localhost:${PORT}/metrics" 2>/dev/null | grep -i "spec\|accept\|draft\|mtp" | head -10 || true)
if [ -n "$METRICS" ]; then
    echo "=== Speculative Decoding Metrics ==="
    echo "$METRICS"
fi

if [ "$FAIL" -gt 0 ]; then
    echo "ERROR: ${FAIL} prompt(s) failed"
    exit 1
fi

# ── optional acceptance-length regression check ───────────────────────────────
if [ -n "${MIN_ACCEPTANCE_LENGTH:-}" ]; then
    AVG_ACCEPT=$(curl -s "http://localhost:${PORT}/metrics" 2>/dev/null \
        | grep -oP 'sglang.*acceptance.*\K[0-9.]+' | tail -1 || true)
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
        echo "WARNING: Could not parse acceptance length from SGLang metrics, skipping regression check"
    fi
fi

echo "=== PASS ==="
