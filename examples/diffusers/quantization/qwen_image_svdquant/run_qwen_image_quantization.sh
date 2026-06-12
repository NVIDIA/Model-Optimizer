#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Reproducible Qwen-Image quantization (FP8 / NVFP4 / NVFP4-SVDQuant) using the
# diffusers quantization example. This script is meant to run INSIDE a container
# that already has NVIDIA Model Optimizer installed from source and a
# Qwen-capable diffusers (see README.md for building the container and the
# Slurm/srun wrapper).
#
# It downloads Qwen/Qwen-Image (idempotently), then for each requested format
# runs `quantize.py` to calibrate the transformer (only `transformer_blocks`,
# excluding the first 2 / last 2 blocks), generate a quantized-inference sanity
# image, and export a HuggingFace checkpoint.
#
# All paths are parameterized via environment variables; the defaults match the
# kernel-dev experiment layout described in README.md.
set -euo pipefail

# --- Configuration (override via environment) --------------------------------
KERNEL_DEV_ROOT="${KERNEL_DEV_ROOT:-/lustre/fsw/coreai_dlalgo_modelopt/users/jingyux/kernel-dev}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen-Image}"
MODEL_DIR="${MODEL_DIR:-${KERNEL_DEV_ROOT}/models/Qwen-Image}"
OUTPUT_DIR="${OUTPUT_DIR:-${KERNEL_DEV_ROOT}/qwen_image_ckpts}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-${KERNEL_DEV_ROOT}/HF_TOKEN.txt}"
# Path to the diffusers quantization example (this script lives one level below it).
QUANT_DIR="${QUANT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# Formats to run: any of {fp8, nvfp4, svdquant}.
FORMATS="${FORMATS:-fp8 nvfp4 svdquant}"

# Calibration knobs (small defaults for a quick run; raise CALIB_SIZE for quality).
CALIB_SIZE="${CALIB_SIZE:-64}"
BATCH_SIZE="${BATCH_SIZE:-2}"
N_STEPS="${N_STEPS:-20}"
LOWRANK="${LOWRANK:-32}"
MODEL_DTYPE="${MODEL_DTYPE:-BFloat16}"

# Set DRY_RUN=1 to print the planned commands without executing them.
DRY_RUN="${DRY_RUN:-0}"

log() { echo "[qwen-image-quant] $*"; }
run() {
    log "+ $*"
    if [[ "${DRY_RUN}" != "1" ]]; then
        "$@"
    fi
}

# --- Hugging Face token ------------------------------------------------------
if [[ ! -r "${HF_TOKEN_FILE}" ]]; then
    echo "ERROR: HF token file not found or not readable: ${HF_TOKEN_FILE}" >&2
    echo "       Set HF_TOKEN_FILE to a readable file containing your Hugging Face token." >&2
    exit 1
fi
HF_TOKEN="$(tr -d '[:space:]' < "${HF_TOKEN_FILE}")"
if [[ -z "${HF_TOKEN}" ]]; then
    echo "ERROR: HF token file is empty: ${HF_TOKEN_FILE}" >&2
    exit 1
fi
export HF_TOKEN
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"

# --- Download the model (idempotent) ----------------------------------------
log "Downloading ${MODEL_ID} -> ${MODEL_DIR} (skipped if already present)"
run mkdir -p "${MODEL_DIR}"
run huggingface-cli download "${MODEL_ID}" --local-dir "${MODEL_DIR}" --exclude "*.onnx"

# --- Quantize + export for each format --------------------------------------
mkdir -p "${OUTPUT_DIR}"
for fmt in ${FORMATS}; do
    case "${fmt}" in
        fp8)      quant_args=(--format fp8 --quant-algo max) ;;
        nvfp4)    quant_args=(--format fp4 --quant-algo max) ;;
        svdquant) quant_args=(--format fp4 --quant-algo svdquant --lowrank "${LOWRANK}") ;;
        *) echo "ERROR: unknown format '${fmt}' (expected fp8|nvfp4|svdquant)" >&2; exit 1 ;;
    esac

    out="${OUTPUT_DIR}/qwen-image-${fmt}"
    log "=== Quantizing Qwen-Image (${fmt}) -> ${out} ==="
    run python "${QUANT_DIR}/quantize.py" \
        --model qwen-image \
        --override-model-path "${MODEL_DIR}" \
        --model-dtype "${MODEL_DTYPE}" \
        "${quant_args[@]}" \
        --calib-size "${CALIB_SIZE}" \
        --batch-size "${BATCH_SIZE}" \
        --n-steps "${N_STEPS}" \
        --hf-ckpt-dir "${out}" \
        --sanity-image-path "${out}/sanity.png"
    log "Done: ${fmt}. Checkpoint at ${out}, sanity image at ${out}/sanity.png"
done

log "All requested formats complete. Checkpoints under ${OUTPUT_DIR}"
