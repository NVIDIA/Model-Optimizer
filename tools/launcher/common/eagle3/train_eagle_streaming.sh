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

# EAGLE3 streaming training: co-locates `vllm serve` (KV-transfer producer of
# hidden states) with the trainer on the same node, and routes hidden states
# over HTTP rather than dumping to disk. Sibling of train_eagle.sh.
#
# Env vars (required):
#   HF_MODEL_CKPT       Target model path. Used by both vllm serve (as the
#                       model arg, becomes the served-model-name) and the
#                       trainer (data.streaming_model_name).
#   EAGLE_CAPTURE_IDS   JSON list of 1-based layer ids vllm should capture.
#                       Must equal default_eagle_aux_layer_ids(L) shifted by +1,
#                       plus the final layer L. For Qwen3-8B (L=36):
#                       default = [1,17,32] -> capture = [2,18,33,36].
#
# Env vars (optional):
#   SERVE_HOST          default 127.0.0.1
#   SERVE_PORT          default 8765
#   SERVE_GPU           CUDA_VISIBLE_DEVICES for vllm serve. default "0"
#   SERVE_TP            tensor-parallel-size for vllm serve. default 1
#   SERVE_GPU_MEM_UTIL  default 0.4
#   TRAIN_GPUS          CUDA_VISIBLE_DEVICES for the trainer. default = all
#                       local GPUs except SERVE_GPU.
#
# All script args after the env-var setup are forwarded to launch_train.sh
# (typically: --config <yaml> plus OmegaConf dotlist overrides).

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "${SCRIPT_DIR}/../service_utils.sh"

###################################################################################################
# Container provisioning
#
# vllm/vllm-openai:* has vllm and torch but not modelopt or the speculative
# trainer's deps. modelopt is bind-mounted at
# /usr/local/lib/python3.12/dist-packages/modelopt, but it has no .dist-info
# (so `importlib.metadata.version('nvidia-modelopt')` would fail). nemo_run
# only ships modelopt subdirs, not the real pyproject.toml, so we synthesize
# a minimal one with a correctly-scoped setuptools.packages.find include —
# without `include = ["modelopt*"]`, setuptools sees both `modelopt/` and
# `modelopt_recipes/` at the top level and refuses with a "flat-layout"
# error. We then `pip install -e .` to register the dist-info.

TOML=modules/Model-Optimizer/pyproject.toml
if [ ! -f "$TOML" ]; then
    cat > "$TOML" <<'EOF'
[build-system]
requires = ["setuptools>=80"]
build-backend = "setuptools.build_meta"

[project]
name = "nvidia-modelopt"
version = "0.0.0"
dependencies = [
    "omegaconf>=2.3.0",
    "PyYAML>=6.0",
    "pulp<4.0",
    "pydantic>=2.0",
    "regex",
    "rich",
    "safetensors",
    "scipy",
    "nvidia-ml-py>=12",
    "packaging",
    "tqdm",
]

[tool.setuptools.packages.find]
include = ["modelopt*", "modelopt_recipes*"]
EOF
fi
pip install --no-cache-dir -e modules/Model-Optimizer/
pip install --no-cache-dir -r modules/Model-Optimizer/examples/speculative_decoding/requirements.txt
pip install --no-cache-dir 'datasets' 'huggingface-hub>=1.2.1'
export PATH=$PATH:/workspace/.local/bin

###################################################################################################

trap 'error_handler $0 $LINENO' ERR # ERROR HANDLER

if [ -z "$HF_MODEL_CKPT" ]; then
    echo "ERROR: HF_MODEL_CKPT must be set." >&2; exit 1
fi
if [ -z "$EAGLE_CAPTURE_IDS" ]; then
    echo "ERROR: EAGLE_CAPTURE_IDS must be set (e.g. '[2, 18, 33, 36]' for Qwen3-8B)." >&2; exit 1
fi

SERVE_HOST="${SERVE_HOST:-127.0.0.1}"
SERVE_PORT="${SERVE_PORT:-8765}"
SERVE_GPU="${SERVE_GPU:-0}"
SERVE_TP="${SERVE_TP:-1}"
SERVE_GPU_MEM_UTIL="${SERVE_GPU_MEM_UTIL:-0.4}"

if [ -z "$TRAIN_GPUS" ]; then
    TRAIN_GPUS=$(python3 - <<PY
total = int("$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -n1)")
exclude = {int(x) for x in "$SERVE_GPU".split(",") if x != ""}
print(",".join(str(i) for i in range(total) if i not in exclude))
PY
)
fi
if [ -z "$TRAIN_GPUS" ]; then
    echo "ERROR: no GPUs left for the trainer (SERVE_GPU=$SERVE_GPU consumed them all)." >&2; exit 1
fi

SERVE_SCRATCH="/scratchspace/streaming_serve_scratch"
SERVE_LOG="/scratchspace/vllm_serve.log"
mkdir -p "$SERVE_SCRATCH"

echo "Launching vllm serve on $SERVE_HOST:$SERVE_PORT (GPUs=$SERVE_GPU, TP=$SERVE_TP, log: $SERVE_LOG)..."
CUDA_VISIBLE_DEVICES="$SERVE_GPU" vllm serve "$HF_MODEL_CKPT" \
    --host "$SERVE_HOST" \
    --port "$SERVE_PORT" \
    --tensor-parallel-size "$SERVE_TP" \
    --enforce-eager \
    --gpu-memory-utilization "$SERVE_GPU_MEM_UTIL" \
    --speculative-config "{
        \"method\":\"extract_hidden_states\",
        \"num_speculative_tokens\":1,
        \"draft_model_config\":{
            \"hf_config\":{
                \"eagle_aux_hidden_state_layer_ids\":$EAGLE_CAPTURE_IDS
            }
        }
    }" \
    --kv-transfer-config "{
        \"kv_connector\":\"ExampleHiddenStatesConnector\",
        \"kv_role\":\"kv_producer\",
        \"kv_connector_extra_config\":{\"shared_storage_path\":\"$SERVE_SCRATCH\"}
    }" \
    > "$SERVE_LOG" 2>&1 &
SERVE_PID=$!

cleanup() {
    echo "Cleaning up vllm serve (PID=$SERVE_PID)..."
    kill $SERVE_PID 2>/dev/null || true
    wait $SERVE_PID 2>/dev/null || true
}
trap cleanup INT TERM EXIT

echo "Waiting for vllm serve to become ready (up to 15 min)..."
READY=0
for i in $(seq 1 180); do
    if curl -fsS "http://$SERVE_HOST:$SERVE_PORT/v1/models" > /dev/null 2>&1; then
        READY=1; break
    fi
    if ! kill -0 $SERVE_PID 2>/dev/null; then
        echo "vllm serve died early. Tail of $SERVE_LOG:"; tail -100 "$SERVE_LOG"; exit 1
    fi
    sleep 5
done
[ "$READY" = "1" ] || { echo "Server not ready in 900s. Tail:"; tail -100 "$SERVE_LOG"; exit 1; }
echo "vllm serve ready."

# Train. dataloader_num_workers must be 0 (the streaming dataset owns one
# asyncio loop per process; multi-worker would duplicate it). Server URL,
# served-model-name, and ``shared_storage_path`` (must match the connector's,
# used as the trainer-side allowlist) are injected here.
echo "Launching trainer on GPUs=$TRAIN_GPUS..."
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" bash modules/Model-Optimizer/examples/speculative_decoding/launch_train.sh \
    "${@}" \
    data.streaming_server_url="http://$SERVE_HOST:$SERVE_PORT" \
    data.streaming_model_name="$HF_MODEL_CKPT" \
    data.streaming_shared_storage_path="$SERVE_SCRATCH" \
    training.dataloader_num_workers=0

python3 modules/Model-Optimizer/examples/speculative_decoding/scripts/export_hf_checkpoint.py \
    --model_path /scratchspace/eagle3 \
    --export_path /scratchspace/export

###################################################################################################

#exit_handler $0
