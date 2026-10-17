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

# Megatron-Bridge PTQ generation: load a quantized MCore checkpoint and run text generation.
# Wraps /opt/Megatron-Bridge/examples/quantization/ptq_generate.py.
# Assumes nvcr.io/nvidia/nemo:26.04+ container (megatron-bridge preinstalled at /opt/Megatron-Bridge).
#
# Required env:
#   HF_MODEL_ID           HF model id used for tokenizer and architecture template.
#   MEGATRON_LOAD_PATH    Quantized MCore ckpt dir produced by quantize.sh.
# Optional env:
#   TP, PP, EP, ETP       Parallelism degrees (defaults: 1, 1, 1, 1).
#   NPROC_PER_NODE        GPUs per node for torchrun (default: nvidia-smi GPU count).
#   PROMPTS               |-separated input prompts.
#   OSL                   Output sequence length (default: 32).
#   TRUST_REMOTE_CODE     "true" to pass --trust-remote-code.
#
# Extra positional args ("$@") are forwarded to ptq_generate.py.

set -e

if [[ -z "${HF_MODEL_ID}" ]]; then
    echo "[ERROR] HF_MODEL_ID is required" >&2
    exit 1
fi
if [[ -z "${MEGATRON_LOAD_PATH}" ]]; then
    echo "[ERROR] MEGATRON_LOAD_PATH is required" >&2
    exit 1
fi

TP="${TP:-1}"
PP="${PP:-1}"
EP="${EP:-1}"
ETP="${ETP:-1}"
OSL="${OSL:-32}"

if [[ -z "${NPROC_PER_NODE}" ]]; then
    NPROC_PER_NODE=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -n 1)
fi

# Multi-node torchrun: derive rendezvous from Slurm env. Falls back to standalone.
NNODES="${SLURM_NNODES:-${NNODES:-1}}"
NODE_RANK="${SLURM_NODEID:-${NODE_RANK:-0}}"
if [[ "${NNODES}" -gt 1 ]]; then
    if [[ -z "${MASTER_ADDR}" && -n "${SLURM_NODELIST}" ]]; then
        MASTER_ADDR=$(scontrol show hostname "${SLURM_NODELIST}" 2>/dev/null | head -n 1)
    fi
    MASTER_PORT="${MASTER_PORT:-29500}"
    RDZV_ARGS=("--nnodes=${NNODES}" "--node-rank=${NODE_RANK}" \
               "--master-addr=${MASTER_ADDR}" "--master-port=${MASTER_PORT}")
else
    RDZV_ARGS=("--standalone" "--nnodes=1")
fi

EXTRA_FLAGS=()
[[ "${TRUST_REMOTE_CODE:-false}" == "true" ]] && EXTRA_FLAGS+=("--trust-remote-code")
[[ -n "${PROMPTS}" ]] && EXTRA_FLAGS+=("--prompts" "${PROMPTS}")

# ptq_generate.py imports `quantize` as a sibling module — run from its directory.
cd /opt/Megatron-Bridge/examples/quantization

echo "=== Generating with ${HF_MODEL_ID} (TP=${TP} PP=${PP} EP=${EP} ETP=${ETP}, ${NPROC_PER_NODE} GPUs) ==="
echo "    load <- ${MEGATRON_LOAD_PATH}"

exec python -m torch.distributed.run --nproc_per_node "${NPROC_PER_NODE}" "${RDZV_ARGS[@]}" ptq_generate.py \
    --hf-model-id "${HF_MODEL_ID}" \
    --megatron-load-path "${MEGATRON_LOAD_PATH}" \
    --tp "${TP}" \
    --pp "${PP}" \
    --ep "${EP}" \
    --etp "${ETP}" \
    --osl "${OSL}" \
    "${EXTRA_FLAGS[@]}" \
    "$@"
