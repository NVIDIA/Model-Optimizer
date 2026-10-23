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

# Megatron-Bridge export: convert a quantized MCore checkpoint to HuggingFace format.
# Wraps /opt/Megatron-Bridge/examples/quantization/export.py.
# Assumes nvcr.io/nvidia/nemo:26.04+ container (megatron-bridge preinstalled at /opt/Megatron-Bridge).
#
# Required env:
#   HF_MODEL_ID           HF model id used for architecture template + tokenizer.
#   MEGATRON_LOAD_PATH    Quantized MCore ckpt dir produced by quantize.sh.
# Optional env:
#   OUTPUT_DIR            Parent dir for export (default: cwd).
#   EXPORT_DIR            HF output dir
#                         (default: ${OUTPUT_DIR}/<basename(HF_MODEL_ID)>_hf_export).
#   TP, PP, EP, ETP       Parallelism degrees (defaults: 1, 1, 1, 1).
#                         NOTE: HF exporter does not gather TP-sharded weights —
#                         use PP > 1 to shard large models across GPUs.
#   NPROC_PER_NODE        GPUs per node for torchrun (default: nvidia-smi GPU count).
#   DTYPE                 Export dtype (default: bfloat16). One of bfloat16, float16, float32.
#   EXPORT_EXTRA_MODULES  "true" to include Medusa / EAGLE / MTP heads.
#   TRUST_REMOTE_CODE     "true" to pass --trust-remote-code.
#
# Extra positional args ("$@") are forwarded to export.py.

set -e

if [[ -z "${HF_MODEL_ID}" ]]; then
    echo "[ERROR] HF_MODEL_ID is required" >&2
    exit 1
fi
if [[ -z "${MEGATRON_LOAD_PATH}" ]]; then
    echo "[ERROR] MEGATRON_LOAD_PATH is required" >&2
    exit 1
fi

OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)}"
MODEL_NAME="$(basename "${HF_MODEL_ID}")"
EXPORT_DIR="${EXPORT_DIR:-${OUTPUT_DIR}/${MODEL_NAME}_hf_export}"

TP="${TP:-1}"
PP="${PP:-1}"
EP="${EP:-1}"
ETP="${ETP:-1}"
DTYPE="${DTYPE:-bfloat16}"

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

mkdir -p "${OUTPUT_DIR}"

EXTRA_FLAGS=()
[[ "${EXPORT_EXTRA_MODULES:-false}" == "true" ]] && EXTRA_FLAGS+=("--export-extra-modules")
[[ "${TRUST_REMOTE_CODE:-false}" == "true" ]] && EXTRA_FLAGS+=("--trust-remote-code")

cd /opt/Megatron-Bridge/examples/quantization

echo "=== Exporting ${HF_MODEL_ID} (TP=${TP} PP=${PP} EP=${EP} ETP=${ETP}, ${NPROC_PER_NODE} GPUs, dtype=${DTYPE}) ==="
echo "    load <- ${MEGATRON_LOAD_PATH}"
echo "    save -> ${EXPORT_DIR}"

python -m torch.distributed.run --nproc_per_node "${NPROC_PER_NODE}" "${RDZV_ARGS[@]}" export.py \
    --hf-model-id "${HF_MODEL_ID}" \
    --megatron-load-path "${MEGATRON_LOAD_PATH}" \
    --export-dir "${EXPORT_DIR}" \
    --tp "${TP}" \
    --pp "${PP}" \
    --ep "${EP}" \
    --etp "${ETP}" \
    --dtype "${DTYPE}" \
    "${EXTRA_FLAGS[@]}" \
    "$@"

ls "${EXPORT_DIR}"
if [[ -f "${EXPORT_DIR}/hf_quant_config.json" ]]; then
    cat "${EXPORT_DIR}/hf_quant_config.json"
fi
