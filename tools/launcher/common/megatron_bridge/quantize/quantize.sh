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

# Megatron-Bridge PTQ quantization: HuggingFace -> quantized MCore checkpoint.
# Wraps /opt/Megatron-Bridge/examples/quantization/quantize.py.
# Assumes nvcr.io/nvidia/nemo:26.04+ container (megatron-bridge preinstalled at /opt/Megatron-Bridge).
#
# Required env: HF_MODEL_ID  (e.g. meta-llama/Llama-3.2-1B)
# Optional env:
#   OUTPUT_DIR             Parent dir for outputs (default: cwd).
#   EXPORT_QUANT_CFG       ModelOpt quant config (default: fp8). Supported:
#                          int8_sq, fp8, fp8_blockwise, int4_awq, w4a8_awq,
#                          nvfp4, mamba_moe_fp8_aggressive, mamba_moe_fp8_conservative,
#                          mamba_moe_nvfp4_aggressive, mamba_moe_nvfp4_conservative.
#   MEGATRON_SAVE_PATH     Output MCore ckpt dir
#                          (default: ${OUTPUT_DIR}/<basename(HF_MODEL_ID)>_quantized_${EXPORT_QUANT_CFG}).
#   TP, PP, EP, ETP        Parallelism degrees (defaults: 1, 1, 1, 1).
#   NPROC_PER_NODE         GPUs per node for torchrun (default: nvidia-smi GPU count).
#   CALIB_SIZE             Calibration sample count (default: 512).
#   COMPRESS               "true" to apply mtq.compress() for real low-bit weights.
#   WEIGHT_ONLY            "true" to disable input quantization.
#   EXPORT_KV_CACHE_QUANT  "true" to enable FP8 KV-cache quantization.
#   TRUST_REMOTE_CODE      "true" to pass --trust-remote-code.
#   PROMPTS                |-separated test prompts.
#   DISABLE_HF_DATASETS_FILE_LOCK  "true" for read-only HF cache dirs.
#
# Extra positional args ("$@") are forwarded to quantize.py.

set -e

if [[ -z "${HF_MODEL_ID}" ]]; then
    echo "[ERROR] HF_MODEL_ID is required" >&2
    exit 1
fi

OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)}"
EXPORT_QUANT_CFG="${EXPORT_QUANT_CFG:-fp8}"
MODEL_NAME="$(basename "${HF_MODEL_ID}")"
MEGATRON_SAVE_PATH="${MEGATRON_SAVE_PATH:-${OUTPUT_DIR}/${MODEL_NAME}_quantized_${EXPORT_QUANT_CFG}}"

TP="${TP:-1}"
PP="${PP:-1}"
EP="${EP:-1}"
ETP="${ETP:-1}"
CALIB_SIZE="${CALIB_SIZE:-512}"

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
[[ "${COMPRESS:-false}" == "true" ]] && EXTRA_FLAGS+=("--compress")
[[ "${WEIGHT_ONLY:-false}" == "true" ]] && EXTRA_FLAGS+=("--weight-only")
[[ "${EXPORT_KV_CACHE_QUANT:-false}" == "true" ]] && EXTRA_FLAGS+=("--export-kv-cache-quant")
[[ "${TRUST_REMOTE_CODE:-false}" == "true" ]] && EXTRA_FLAGS+=("--trust-remote-code")
[[ "${DISABLE_HF_DATASETS_FILE_LOCK:-false}" == "true" ]] && EXTRA_FLAGS+=("--disable-hf-datasets-file-lock")
[[ -n "${PROMPTS}" ]] && EXTRA_FLAGS+=("--prompts" "${PROMPTS}")

# Workaround for upstream Megatron-Bridge using the deprecated dataset id
# `cnn_dailymail` (no namespace). Newer huggingface_hub requires `namespace/name`
# and rejects the bare form with HfUriError. Rewrite to `abisee/cnn_dailymail`,
# which is the canonical id and is cached under /hf-local/abisee/cnn_dailymail.
_UPSTREAM_QUANT=/opt/Megatron-Bridge/examples/quantization/quantize.py
if [[ -w "${_UPSTREAM_QUANT}" ]] && grep -q 'load_dataset("cnn_dailymail"' "${_UPSTREAM_QUANT}"; then
    sed -i 's|load_dataset("cnn_dailymail"|load_dataset("abisee/cnn_dailymail"|g' "${_UPSTREAM_QUANT}"
fi

# quantize.py imports `quantize_utils` as a sibling module — run from its directory.
cd /opt/Megatron-Bridge/examples/quantization

echo "=== Quantizing ${HF_MODEL_ID} with ${EXPORT_QUANT_CFG} (TP=${TP} PP=${PP} EP=${EP} ETP=${ETP}, ${NPROC_PER_NODE} GPUs) ==="
echo "    save -> ${MEGATRON_SAVE_PATH}"

exec python -m torch.distributed.run --nproc_per_node "${NPROC_PER_NODE}" "${RDZV_ARGS[@]}" quantize.py \
    --hf-model-id "${HF_MODEL_ID}" \
    --export-quant-cfg "${EXPORT_QUANT_CFG}" \
    --megatron-save-path "${MEGATRON_SAVE_PATH}" \
    --tp "${TP}" \
    --pp "${PP}" \
    --ep "${EP}" \
    --etp "${ETP}" \
    --calib-size "${CALIB_SIZE}" \
    "${EXTRA_FLAGS[@]}" \
    "$@"
