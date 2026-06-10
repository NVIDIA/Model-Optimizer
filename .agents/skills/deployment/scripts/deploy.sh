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

# ModelOpt Deployment Script
# Deploy quantized or unquantized models via vLLM, SGLang, or TRT-LLM
# Supports ModelOpt FP8/FP4 checkpoints with automatic quantization flag detection

set -eo pipefail

# Default configuration
MODEL=""
PORT=8000
HOST="0.0.0.0"
FRAMEWORK="vllm"
TP_SIZE=1
VRAM=0.9
MAX_WAIT=300  # 5 min for large models
QUANTIZATION=""  # auto-detected from checkpoint

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LOG_DIR:-/tmp/modelopt-deploy}"
LOG_FILE="$LOG_DIR/server.log"
PID_FILE="$LOG_DIR/server.pid"
META_FILE="$LOG_DIR/server.meta"  # persists model/framework/port for status

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()    { printf "${BLUE}[INFO]${NC} %s\n" "$1" >&2; }
log_success() { printf "${GREEN}[OK]${NC} %s\n" "$1" >&2; }
log_warn()    { printf "${YELLOW}[WARN]${NC} %s\n" "$1" >&2; }
log_error()   { printf "${RED}[ERROR]${NC} %s\n" "$1" >&2; }

usage() {
    cat <<EOF
Usage: $0 <command> [OPTIONS]

Commands:
  start    - Start the inference server
  stop     - Stop the inference server
  test     - Test the API endpoint
  status   - Show server status
  restart  - Restart the server
  detect   - Detect checkpoint format (without starting)

Options:
  --model PATH              Model path or HF model ID (required for start)
  --framework FRAMEWORK     vllm, sglang, or trtllm (default: vllm)
  --port PORT               Server port (default: 8000)
  --tp SIZE                 Tensor parallel size (default: 1)
  --quantization QUANT      Force quantization flag (modelopt, modelopt_fp4, or none)
  --gpu-memory-utilization  GPU memory utilization 0.0-1.0 (default: 0.9)
  --log-dir DIR             Log directory (default: /tmp/modelopt-deploy)

Examples:
  $0 start --model ./qwen3-0.6b-fp8
  $0 start --model ./llama-70b-nvfp4 --framework sglang --tp 4
  $0 start --model nvidia/Llama-3.1-8B-Instruct-FP8 --framework vllm
  $0 test --port 8000
  $0 stop
EOF
    exit 1
}

# ─── Checkpoint Detection ───────────────────────────────────────────

detect_quantization() {
    local model_path="$1"

    # Skip detection for HF model IDs (no local path)
    if [[ ! -d "$model_path" ]]; then
        log_info "Model is a HF ID — using name-based heuristic for quantization flag"
        # Best-effort: infer from model name. This is a fallback; local checkpoints
        # use hf_quant_config.json which is reliable.
        if echo "$model_path" | grep -qi "fp8"; then
            log_info "HF model name contains 'fp8' — assuming modelopt quantization"
            echo "modelopt"
        elif echo "$model_path" | grep -qi "fp4\|nvfp4"; then
            log_info "HF model name contains 'fp4/nvfp4' — assuming modelopt_fp4 quantization"
            echo "modelopt_fp4"
        else
            log_info "No quantization format detected in model name — treating as unquantized"
            echo "none"
        fi
        return
    fi

    # Require python3 for JSON parsing
    if ! command -v python3 &>/dev/null; then
        log_error "python3 is required to detect quantization format but is not installed"
        return 1
    fi

    # Local checkpoint: check hf_quant_config.json
    local quant_config="$model_path/hf_quant_config.json"
    if [[ -f "$quant_config" ]]; then
        log_info "Found hf_quant_config.json"

        local quant_algo
        quant_algo=$(python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    cfg = json.load(f)
quant_algo = cfg.get('quantization', {}).get('quant_algo', '')
print(quant_algo)
" "$quant_config" 2>&1) || {
            log_error "Failed to parse hf_quant_config.json: $quant_algo"
            return 1
        }

        if echo "$quant_algo" | grep -qi "fp4\|nvfp4"; then
            echo "modelopt_fp4"
        elif echo "$quant_algo" | grep -qi "fp8"; then
            echo "modelopt"
        elif echo "$quant_algo" | grep -qi "int4_awq\|w4a8_awq\|w4a16_awq\|w8a"; then
            log_error "Quantization format '$quant_algo' is only supported by TRT-LLM, not vLLM/SGLang"
            log_error "Use --framework trtllm or deploy with TRT-LLM directly"
            return 1
        elif [[ -z "$quant_algo" ]]; then
            echo "none"
        else
            log_warn "Unknown quant_algo '$quant_algo' — trying --quantization modelopt"
            echo "modelopt"
        fi
    elif [[ -f "$model_path/config.json" ]]; then
        # Fallback: check config.json for quantization_config with quant_method=modelopt
        local quant_method
        quant_method=$(python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    cfg = json.load(f)
qc = cfg.get('quantization_config', {})
if qc.get('quant_method') == 'modelopt':
    print(qc.get('quant_algo', 'fp8'))
" "$model_path/config.json" 2>&1) || {
            log_error "Failed to parse config.json: $quant_method"
            return 1
        }
        if [[ -n "$quant_method" ]]; then
            log_info "Found quantization_config in config.json (quant_method=modelopt)"
            if echo "$quant_method" | grep -qi "fp4"; then
                echo "modelopt_fp4"
            else
                echo "modelopt"
            fi
        else
            log_info "No quantization config found — treating as unquantized"
            echo "none"
        fi
    else
        log_info "No hf_quant_config.json or config.json found — treating as unquantized"
        echo "none"
    fi
}

detect_gpu() {
    if command -v nvidia-smi &>/dev/null; then
        local gpu_count
        gpu_count=$(nvidia-smi -L 2>/dev/null | wc -l)
        local gpu_name
        gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
        log_info "GPUs: ${gpu_count}x ${gpu_name}"
        echo "$gpu_count"
    else
        log_error "No NVIDIA GPU detected (nvidia-smi not found)"
        return 1
    fi
}

# ─── Server Management ──────────────────────────────────────────────

is_server_running() {
    if [[ -f "$PID_FILE" ]]; then
        local pid
        pid=$(cat "$PID_FILE" 2>/dev/null)
        if [[ ! "$pid" =~ ^[0-9]+$ ]]; then
            rm -f "$PID_FILE"
            return 1
        fi
        # Verify the PID is actually a Python/vLLM/SGLang process (not PID reuse)
        local cmdline
        cmdline=$(ps -p "$pid" -o args= 2>/dev/null) || { rm -f "$PID_FILE"; return 1; }
        if echo "$cmdline" | grep -q "vllm\|sglang\|python"; then
            return 0
        fi
        # PID exists but is not our server — stale PID file
        rm -f "$PID_FILE"
    fi
    return 1
}

start_server() {
    # Validate GPU availability and TP size
    local gpu_count
    gpu_count=$(detect_gpu) || exit 1
    if [[ "$TP_SIZE" -gt "$gpu_count" ]]; then
        log_error "Requested TP size ($TP_SIZE) exceeds available GPUs ($gpu_count)"
        exit 1
    fi

    if [[ -z "$MODEL" ]]; then
        log_error "--model is required"
        usage
    fi

    if is_server_running; then
        log_warn "Server already running (PID: $(cat "$PID_FILE"))"
        return 0
    fi

    # Check if port is already in use
    if ss -tlnp 2>/dev/null | grep -q ":${PORT} " || \
       lsof -i ":${PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
        log_error "Port $PORT is already in use — stop the existing service or use --port <other_port>"
        exit 1
    fi

    mkdir -p "$LOG_DIR"

    # Auto-detect quantization if not forced
    if [[ -z "$QUANTIZATION" ]]; then
        if ! QUANTIZATION=$(detect_quantization "$MODEL"); then
            log_error "Failed to detect quantization — fix the checkpoint or use --quantization to override"
            exit 1
        fi
    fi
    log_info "Quantization: $QUANTIZATION"

    # Save metadata for status command (values single-quoted for safe reading)
    cat >"$META_FILE" <<METAEOF
FRAMEWORK='${FRAMEWORK//\'/}'
MODEL='${MODEL//\'/}'
PORT='${PORT//\'/}'
QUANTIZATION='${QUANTIZATION//\'/}'
TP_SIZE='${TP_SIZE//\'/}'
METAEOF

    # Build and run the command
    case "$FRAMEWORK" in
        vllm)
            start_vllm
            ;;
        sglang)
            start_sglang
            ;;
        trtllm)
            start_trtllm
            ;;
        *)
            log_error "Unknown framework: $FRAMEWORK (use vllm, sglang, or trtllm)"
            exit 1
            ;;
    esac

    # Wait for server readiness
    wait_for_server
}

start_vllm() {
    log_info "Starting vLLM server..."

    local -a cmd=(python3 -m vllm.entrypoints.openai.api_server
        --model "$MODEL"
        --host "$HOST" --port "$PORT"
        --tensor-parallel-size "$TP_SIZE"
        --gpu-memory-utilization "$VRAM")

    if [[ "$QUANTIZATION" != "none" ]]; then
        cmd+=(--quantization "$QUANTIZATION")
    fi

    log_info "Command: ${cmd[*]}"
    nohup "${cmd[@]}" >"$LOG_FILE" 2>&1 &
    echo $! >"$PID_FILE"

    # Check for immediate crash (missing module, port conflict, CUDA error)
    sleep 2
    if ! ps -p "$(cat "$PID_FILE")" >/dev/null 2>&1; then
        log_error "Server process exited immediately. Last log lines:"
        tail -20 "$LOG_FILE" 2>/dev/null
        rm -f "$PID_FILE"
        exit 1
    fi
    log_success "vLLM started (PID: $(cat "$PID_FILE"))"
}

start_sglang() {
    log_info "Starting SGLang server..."

    local -a cmd=(python3 -m sglang.launch_server
        --model-path "$MODEL"
        --host "$HOST" --port "$PORT"
        --tp "$TP_SIZE")

    if [[ "$QUANTIZATION" != "none" ]]; then
        cmd+=(--quantization "$QUANTIZATION")
    fi

    log_info "Command: ${cmd[*]}"
    nohup "${cmd[@]}" >"$LOG_FILE" 2>&1 &
    echo $! >"$PID_FILE"

    # Check for immediate crash
    sleep 2
    if ! ps -p "$(cat "$PID_FILE")" >/dev/null 2>&1; then
        log_error "Server process exited immediately. Last log lines:"
        tail -20 "$LOG_FILE" 2>/dev/null
        rm -f "$PID_FILE"
        exit 1
    fi
    log_success "SGLang started (PID: $(cat "$PID_FILE"))"
}

start_trtllm() {
    log_info "Starting TRT-LLM server..."
    log_info "TRT-LLM serving is not automated by this script."
    log_info "Options for TRT-LLM deployment:"

    cat <<TRTEOF

# Option 1: AutoDeploy (recommended)
./examples/llm_autodeploy/scripts/run_auto_quant_and_deploy.sh \\
    --hf_ckpt "$MODEL" \\
    --save_quantized_ckpt <output_path> \\
    --quant fp8,nvfp4 \\
    --effective_bits 4.5

# Option 2: Python API
python3 -c "
from tensorrt_llm import LLM, SamplingParams
llm = LLM(model='$MODEL')
print(llm.generate(['Hello, my name is'], SamplingParams(temperature=0.8)))
"
TRTEOF

    log_warn "TRT-LLM server mode not yet automated in this script."
    log_warn "Use vLLM or SGLang for OpenAI-compatible serving of ModelOpt checkpoints."
    return 1
}

wait_for_server() {
    log_info "Waiting for server at http://localhost:$PORT ..."
    local elapsed=0
    while [[ $elapsed -lt $MAX_WAIT ]]; do
        if curl -s "http://localhost:$PORT/health" >/dev/null 2>&1; then
            log_success "Server is ready! (${elapsed}s)"
            return 0
        fi

        # Check if process died
        if ! is_server_running; then
            log_error "Server process died. Check logs: $LOG_FILE"
            tail -20 "$LOG_FILE" 2>/dev/null
            exit 1
        fi

        sleep 5
        elapsed=$((elapsed + 5))
        printf "."
    done

    echo ""
    log_error "Server not ready after ${MAX_WAIT}s. Check logs: $LOG_FILE"
    tail -20 "$LOG_FILE" 2>/dev/null
    exit 1
}

stop_server() {
    if ! is_server_running; then
        log_warn "Server is not running"
        return 0
    fi

    local pid
    pid=$(cat "$PID_FILE")
    log_info "Stopping server (PID: $pid)..."

    # Kill the process group to catch child processes (vLLM/SGLang may fork)
    kill -- -"$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true

    # Wait for graceful shutdown
    for i in {1..15}; do
        if ! ps -p "$pid" >/dev/null 2>&1; then
            rm -f "$PID_FILE" "$META_FILE"
            log_success "Server stopped"
            return 0
        fi
        sleep 1
    done

    # Force kill
    log_warn "Force killing..."
    kill -9 -- -"$pid" 2>/dev/null || kill -9 "$pid" 2>/dev/null || true
    sleep 1
    if ps -p "$pid" >/dev/null 2>&1; then
        log_error "Failed to kill server process $pid — manual intervention required"
    fi
    rm -f "$PID_FILE" "$META_FILE"

    # Check for orphaned GPU worker processes
    local orphans
    orphans=$(pgrep -f "vllm\|sglang" 2>/dev/null | wc -l)
    if [[ "$orphans" -gt 0 ]]; then
        log_warn "Found $orphans potential orphaned server processes — run: pkill -f 'vllm|sglang'"
    fi
    log_success "Server stopped (forced)"
}

test_api() {
    log_info "Testing API at http://localhost:$PORT ..."

    # Health check
    if ! curl -s "http://localhost:$PORT/health" >/dev/null 2>&1; then
        log_error "Server not responding at port $PORT"
        exit 1
    fi
    log_success "Health check passed"

    # List models
    log_info "Available models:"
    curl -s "http://localhost:$PORT/v1/models" | python3 -m json.tool 2>/dev/null || true

    # Test completion
    log_info "Sending test request..."
    local model_id
    model_id=$(curl -s "http://localhost:$PORT/v1/models" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(data['data'][0]['id'])
" 2>/dev/null)

    if [[ -z "$model_id" ]]; then
        log_error "Could not determine model ID from /v1/models endpoint"
        exit 1
    fi

    local payload
    payload=$(python3 -c "
import json, sys
print(json.dumps({'model': sys.argv[1], 'prompt': 'The capital of France is', 'max_tokens': 32, 'temperature': 0.7}))
" "$model_id")

    local response
    response=$(curl -s "http://localhost:$PORT/v1/completions" \
        -H "Content-Type: application/json" \
        -d "$payload")

    echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"

    local text
    text=$(echo "$response" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(data['choices'][0]['text'])
" 2>/dev/null)

    if [[ -n "$text" ]]; then
        log_success "API test passed!"
        printf "${GREEN}Response:${NC} %s\n" "$text"
    else
        log_error "No valid response from API"
        exit 1
    fi
}

show_status() {
    echo "=== ModelOpt Deployment Status ==="
    echo ""
    if is_server_running; then
        local pid
        pid=$(cat "$PID_FILE")
        log_success "Server running (PID: $pid)"

        # Read saved metadata safely (no source — avoids shell injection)
        if [[ -f "$META_FILE" ]]; then
            FRAMEWORK=$(grep '^FRAMEWORK=' "$META_FILE" | cut -d= -f2- | tr -d "'")
            MODEL=$(grep '^MODEL=' "$META_FILE" | cut -d= -f2- | tr -d "'")
            PORT=$(grep '^PORT=' "$META_FILE" | cut -d= -f2- | tr -d "'")
            QUANTIZATION=$(grep '^QUANTIZATION=' "$META_FILE" | cut -d= -f2- | tr -d "'")
            TP_SIZE=$(grep '^TP_SIZE=' "$META_FILE" | cut -d= -f2- | tr -d "'")
        fi

        echo "  Framework:    ${FRAMEWORK:-unknown}"
        echo "  Model:        ${MODEL:-unknown}"
        echo "  Endpoint:     http://localhost:${PORT:-8000}"
        echo "  Logs:         $LOG_FILE"
        echo ""
        if [[ -f "$LOG_FILE" ]]; then
            echo "Recent logs:"
            tail -5 "$LOG_FILE"
        fi
    else
        log_warn "Server is not running"
        echo "  Start with: $0 start --model <path>"
    fi
}

# ─── Argument Parsing ────────────────────────────────────────────────

COMMAND=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --model|--framework|--port|--tp|--quantization|--gpu-memory-utilization|--log-dir)
            if [[ -z "${2:-}" || "$2" == -* ]]; then
                log_error "Option $1 requires a value"
                usage
            fi
            ;;&
        --model)               MODEL="$2"; _CLI_MODEL=1; shift 2 ;;
        --framework)           FRAMEWORK="$2"; _CLI_FRAMEWORK=1; shift 2 ;;
        --port)                PORT="$2"; _CLI_PORT=1; shift 2 ;;
        --tp)                  TP_SIZE="$2"; _CLI_TP=1; shift 2 ;;
        --quantization)        QUANTIZATION="$2"; _CLI_QUANT=1; shift 2 ;;
        --gpu-memory-utilization) VRAM="$2"; shift 2 ;;
        --log-dir)             LOG_DIR="$2"; LOG_FILE="$LOG_DIR/server.log"; PID_FILE="$LOG_DIR/server.pid"; META_FILE="$LOG_DIR/server.meta"; shift 2 ;;
        start|stop|test|status|restart|detect)
            COMMAND="$1"; shift ;;
        *)
            log_error "Unknown option: $1"
            usage ;;
    esac
done

if [[ -z "$COMMAND" ]]; then
    usage
fi

# Validate numeric arguments
if [[ -n "$PORT" && ! "$PORT" =~ ^[0-9]+$ ]]; then
    log_error "--port must be a number, got: $PORT"
    exit 1
fi
if [[ -n "$TP_SIZE" && ! "$TP_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    log_error "--tp must be a positive integer, got: $TP_SIZE"
    exit 1
fi

# Execute
case "$COMMAND" in
    start)   start_server ;;
    stop)    stop_server ;;
    test)    test_api ;;
    status)  show_status ;;
    restart)
        # Load ALL fields from metadata, then let CLI args override
        if [[ -f "$META_FILE" ]]; then
            _saved_model=$(grep '^MODEL=' "$META_FILE" | cut -d= -f2- | tr -d "'")
            _saved_framework=$(grep '^FRAMEWORK=' "$META_FILE" | cut -d= -f2- | tr -d "'")
            _saved_port=$(grep '^PORT=' "$META_FILE" | cut -d= -f2- | tr -d "'")
            _saved_quant=$(grep '^QUANTIZATION=' "$META_FILE" | cut -d= -f2- | tr -d "'")
            _saved_tp=$(grep '^TP_SIZE=' "$META_FILE" | cut -d= -f2- | tr -d "'")
            # Apply saved values as defaults; CLI args (tracked via _CLI_*) win
            [[ -z "${_CLI_MODEL:-}" ]]     && MODEL="${_saved_model:-$MODEL}"
            [[ -z "${_CLI_FRAMEWORK:-}" ]] && FRAMEWORK="${_saved_framework:-$FRAMEWORK}"
            [[ -z "${_CLI_PORT:-}" ]]      && PORT="${_saved_port:-$PORT}"
            [[ -z "${_CLI_QUANT:-}" ]]     && QUANTIZATION="${_saved_quant:-$QUANTIZATION}"
            [[ -z "${_CLI_TP:-}" ]]        && TP_SIZE="${_saved_tp:-$TP_SIZE}"
        fi
        stop_server; sleep 2; start_server ;;
    detect)
        if [[ -z "$MODEL" ]]; then
            log_error "--model is required for detect"
            exit 1
        fi
        if ! quant=$(detect_quantization "$MODEL"); then
            exit 1
        fi
        echo "Detected quantization: $quant"
        ;;
    *)       usage ;;
esac
