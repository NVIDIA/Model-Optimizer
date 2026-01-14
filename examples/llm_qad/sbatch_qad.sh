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

# QAD SLURM Batch Submission Script
# Usage: sbatch sbatch_qad.sh --config configs/your-config.conf
# Override: sbatch --nodes=4 --account=<account> sbatch_qad.sh --config ...

#SBATCH -p batch
#SBATCH --account=<your-account>
#SBATCH --nodes=4
#SBATCH -t 4:00:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=qad-training

set -x -e

# === Parse Arguments ===
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
CONFIG_FILE=""
HF_TOKEN_ARG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --config|-c) CONFIG_FILE="$2"; shift 2;;
        --hf-token) HF_TOKEN_ARG="$2"; shift 2;;
        *) break;;
    esac
done

[[ -n "$HF_TOKEN_ARG" ]] && export HF_TOKEN="$HF_TOKEN_ARG"

# === Load Config ===
if [[ -n "$CONFIG_FILE" ]]; then
    [[ "$CONFIG_FILE" = /* ]] || CONFIG_FILE="${SCRIPT_DIR}/${CONFIG_FILE}"
    if [[ -f "$CONFIG_FILE" ]]; then
        echo "Loading config: ${CONFIG_FILE}"
        source "$CONFIG_FILE"
    else
        echo "ERROR: Config not found: ${CONFIG_FILE}"
        ls -1 "${SCRIPT_DIR}/configs/"*.conf 2>/dev/null || echo "(no configs found)"
        exit 1
    fi
fi

# === Default Paths (override in config) ===
MLM_DIR="${MLM_DIR:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_modelopt/users/weimingc/workspace/Megatron-LM}"
MODELOPT_DIR="${MODELOPT_DIR:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_modelopt/users/weimingc/workspace/TensorRT-Model-Optimizer}"
MODELS_ROOT="${MODELS_ROOT:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_modelopt/users/weimingc/models}"
QAD_CHECKPOINT_ROOT="${QAD_CHECKPOINT_ROOT:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_modelopt/users/weimingc/checkpoints}"
DATACACHE_DIR="${DATACACHE_DIR:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_modelopt/users/weimingc/data_cache}"
LOG_DIR="${LOG_DIR:-${QAD_CHECKPOINT_ROOT}/logs_slurm}"

# Container settings
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_modelopt/users/weimingc/containers/pytorch_25.06-py3.sqsh}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-/lustre/fs1:/lustre/fs1}"
CONTAINER_WORKDIR="${CONTAINER_WORKDIR:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_modelopt/users/weimingc/workspace/TensorRT-Model-Optimizer/examples/llm_qad}"

# Parallelism (required from config)
TP_SIZE="${TP_SIZE:?ERROR: TP_SIZE must be set in config}"
MBS="${MBS:?ERROR: MBS must be set in config}"
PP_SIZE="${PP_SIZE:-1}"
EP_SIZE="${EP_SIZE:-1}"
NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29500}"

# Multi-node from SLURM
NNODES="${SLURM_NNODES:-4}"
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

mkdir -p "${LOG_DIR}"
DATETIME=$(date +'date_%y-%m-%d_time_%H-%M-%S')

# === Display Configuration ===
echo "========================================"
echo "QAD Training Configuration"
echo "========================================"
[[ -n "$CONFIG_FILE" ]] && echo "Config: ${CONFIG_FILE}"
echo "Model: ${STUDENT_MODEL:-unknown} -> Teacher: ${TEACHER_MODEL:-unknown}"
echo "LR: ${LR:-?} | Dataset: ${DATASET_NAME:-?}"
echo "Parallelism: TP=${TP_SIZE} PP=${PP_SIZE} EP=${EP_SIZE} MBS=${MBS}"
echo "Nodes: ${NNODES} x ${NUM_GPUS} GPUs = $((NNODES * NUM_GPUS)) total"
echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo ""
echo "Paths:"
echo "  MLM_DIR: ${MLM_DIR}"
echo "  MODELOPT_DIR: ${MODELOPT_DIR}"
echo "  Checkpoints: ${QAD_CHECKPOINT_ROOT}"
echo ""
echo "Container: ${CONTAINER_IMAGE}"
echo ""
echo "Checkpoints:"
echo "  Student: ${STUDENT_CKPT:-NOT SET}"
echo "  Teacher: ${TEACHER_CKPT:-NOT SET}"
[[ -n "${BLEND_PATH:-}" ]] && echo "  Blend: ${BLEND_PATH}"
echo "========================================"

# Validate required
[[ -z "${STUDENT_CKPT:-}" ]] && echo "ERROR: STUDENT_CKPT required" && exit 1
[[ -z "${TEACHER_CKPT:-}" ]] && echo "ERROR: TEACHER_CKPT required" && exit 1

# === Build Container Exports ===
# Use local /tmp for Triton cache to avoid race conditions
EXPORTS="export TRITON_CACHE_DIR=/tmp/triton_cache_\${SLURM_JOB_ID}_\${SLURM_PROCID}"
EXPORTS="${EXPORTS} && export NODE_RANK=\${SLURM_PROCID}"
EXPORTS="${EXPORTS} && export NNODES=${NNODES} NUM_GPUS=${NUM_GPUS}"
EXPORTS="${EXPORTS} && export TP_SIZE=${TP_SIZE} PP_SIZE=${PP_SIZE} EP_SIZE=${EP_SIZE} MBS=${MBS}"
EXPORTS="${EXPORTS} && export IS_MOE=${IS_MOE:-false}"
EXPORTS="${EXPORTS} && export MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
EXPORTS="${EXPORTS} && export MLM_DIR=${MLM_DIR} MODELOPT_DIR=${MODELOPT_DIR}"
EXPORTS="${EXPORTS} && export QAD_CHECKPOINT_ROOT=${QAD_CHECKPOINT_ROOT} DATACACHE_DIR=${DATACACHE_DIR}"
EXPORTS="${EXPORTS} && export STUDENT_CKPT=${STUDENT_CKPT} TEACHER_CKPT=${TEACHER_CKPT}"

# Training hyperparameters
for v in LR GBS MIN_LR LR_DECAY_STYLE SAVE_INTERVAL LOG_INTERVAL STUDENT_MODEL TEACHER_MODEL DATASET_NAME; do
    [[ -n "${!v:-}" ]] && EXPORTS="${EXPORTS} && export ${v}=${!v}"
done

# Model config
[[ -n "${STUDENT_CONFIG_FILE:-}" ]] && EXPORTS="${EXPORTS} && export STUDENT_CONFIG_FILE=${STUDENT_CONFIG_FILE}"
[[ -n "${TOKENIZER_MODEL:-}" ]] && EXPORTS="${EXPORTS} && export TOKENIZER_MODEL=${TOKENIZER_MODEL}"
[[ -n "${TEACHER_MODEL_CONFIG:-}" ]] && EXPORTS="${EXPORTS} && export TEACHER_MODEL_CONFIG=${TEACHER_MODEL_CONFIG}"

# Dataset
[[ -n "${BLEND_PATH:-}" ]] && EXPORTS="${EXPORTS} && export BLEND_PATH=${BLEND_PATH}"
[[ -n "${TRAIN_SAMPLES:-}" ]] && EXPORTS="${EXPORTS} && export TRAIN_SAMPLES=${TRAIN_SAMPLES}"

# Optional
[[ -n "${HF_TOKEN:-}" ]] && EXPORTS="${EXPORTS} && export HF_TOKEN=${HF_TOKEN} HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}"
[[ -n "${ITERATIONS_TO_SKIP:-}" ]] && EXPORTS="${EXPORTS} && export ITERATIONS_TO_SKIP=${ITERATIONS_TO_SKIP}"
[[ -n "${DISTILL_CONFIG_PATH:-}" ]] && EXPORTS="${EXPORTS} && export DISTILL_CONFIG_PATH=${DISTILL_CONFIG_PATH}"

# === Launch ===
CONFIG_ARGS=""
[[ -n "${CONFIG_FILE}" ]] && CONFIG_ARGS="--config ${CONFIG_FILE}"
[[ -n "${HF_TOKEN:-}" ]] && CONFIG_ARGS="${CONFIG_ARGS} --hf-token ${HF_TOKEN}"

run_cmd="pip install transformers==4.54 && ${EXPORTS} && cd ${CONTAINER_WORKDIR} && bash qad.sh ${CONFIG_ARGS}"

echo "Running: ${run_cmd}"

srun -l \
    --output=${LOG_DIR}/%x_%j_${DATETIME}.log \
    --error=${LOG_DIR}/err_%x_%j_${DATETIME}.log \
    --container-image ${CONTAINER_IMAGE} \
    --container-mounts ${CONTAINER_MOUNTS} \
    --container-workdir ${CONTAINER_WORKDIR} \
    sh -c "${run_cmd}"

echo "========================================"
echo "QAD Training completed at $(date)"
echo "Logs: ${LOG_DIR}/"
echo "========================================"
