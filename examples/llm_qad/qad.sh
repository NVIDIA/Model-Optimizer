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

# =============================================================================
# QAD (Quantization-Aware Distillation) Training Script
# =============================================================================
#
# This script trains quantized language models using knowledge distillation
# from a teacher model. Supports both dense and MoE (Mixture of Experts) models.
#
# USAGE:
#   bash qad.sh --config configs/qwen3-8b.conf
#   bash qad.sh --config configs/qwen3-30b-a3b-instruct-2507-moe.conf
#   bash qad.sh --hf-token hf_xxx --config configs/qwen3-8b.conf
#
# REQUIRED CONFIG VARIABLES:
#   Model:      STUDENT_MODEL, TEACHER_MODEL, IS_MOE, TOKENIZER_MODEL
#   Training:   LR, GBS, MIN_LR, LR_DECAY_STYLE, SAVE_INTERVAL, LOG_INTERVAL
#   Data:       DATASET_NAME, BLEND_PATH, TRAIN_SAMPLES
#   Parallel:   TP_SIZE, MBS
#   Paths:      STUDENT_CKPT, TEACHER_CKPT, TEACHER_MODEL_CONFIG,
#               STUDENT_CONFIG_FILE, MLM_DIR, MODELOPT_DIR,
#               QAD_CHECKPOINT_ROOT, DATACACHE_DIR
#
# =============================================================================

set -euo pipefail

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

log_info()  { echo "[INFO] $*"; }
log_warn()  { echo "[WARN] $*"; }
log_error() { echo "[ERROR] $*" >&2; }

die() {
    log_error "$@"
    exit 1
}

require_var() {
    local var_name="$1"
    local var_value="${!var_name:-}"
    if [[ -z "$var_value" ]]; then
        die "$var_name must be set in config"
    fi
}

require_file() {
    local path="$1"
    local desc="${2:-File}"
    [[ -f "$path" ]] || die "$desc not found: $path"
}

require_dir() {
    local path="$1"
    local desc="${2:-Directory}"
    [[ -d "$path" ]] || die "$desc not found: $path"
}

sanitize_for_path() {
    echo "$1" | sed -e 's/[\/ :]/_/g' -e 's/[=]/_/g'
}

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

# NCCL and distributed training settings
export NCCL_IB_SL=1
export NCCL_IB_TIMEOUT=19
export NCCL_P2P_NET_CHUNKSIZE=2097152
export NCCL_DEBUG=WARN
export NCCL_SHM_DISABLE=1
export NCCL_NVLS_ENABLE=0

# CUDA settings
export CUDA_DEVICE_MAX_CONNECTIONS=1
export UB_TIMEOUT=720

# Transformer Engine margins
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16

# PyTorch settings (disable features that cause issues during training)
export TORCHINDUCTOR_COMPILE_THREADS=1
export TORCH_COMPILE_DISABLE=1
export PYTORCH_NO_CUDA_MEMORY_CACHING=0
export TORCH_DISTRIBUTED_DEBUG=OFF
export PYTORCH_JIT=0
export TORCH_USE_CUDA_DSA=0

# Network interface
export GLOO_SOCKET_IFNAME=ibp26s0

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE=""
HF_TOKEN_ARG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --config|-c)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --hf-token)
            HF_TOKEN_ARG="$2"
            shift 2
            ;;
        *)
            die "Unknown argument: $1"
            ;;
    esac
done

# HuggingFace token (from arg takes precedence)
if [[ -n "$HF_TOKEN_ARG" ]]; then
    export HF_TOKEN="$HF_TOKEN_ARG"
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
    log_info "HuggingFace token configured"
fi

# =============================================================================
# CONFIG LOADING
# =============================================================================

if [[ -z "$CONFIG_FILE" ]]; then
    log_error "Config file is required. Use --config <path>"
    echo "Available configs:"
    ls -1 "${SCRIPT_DIR}/configs/"*.conf 2>/dev/null || echo "  (none found)"
    exit 1
fi

# Handle relative paths
[[ "$CONFIG_FILE" = /* ]] || CONFIG_FILE="${SCRIPT_DIR}/${CONFIG_FILE}"

require_file "$CONFIG_FILE" "Config file"
log_info "Loading config from: ${CONFIG_FILE}"
source "$CONFIG_FILE"

# =============================================================================
# CONFIG VALIDATION
# =============================================================================

# Required: Training hyperparameters
require_var LR
require_var GBS
require_var MIN_LR
require_var LR_DECAY_STYLE
require_var SAVE_INTERVAL
require_var LOG_INTERVAL

# Required: Model and data
require_var STUDENT_MODEL
require_var TEACHER_MODEL
require_var DATASET_NAME
require_var BLEND_PATH
require_var TRAIN_SAMPLES
require_var IS_MOE
require_var TOKENIZER_MODEL

# Required: Parallelism
require_var TP_SIZE
require_var MBS

# Required: Checkpoints
require_var STUDENT_CKPT
require_var TEACHER_CKPT
require_var TEACHER_MODEL_CONFIG

# Required: Paths
require_var STUDENT_CONFIG_FILE
require_var MLM_DIR
require_var MODELOPT_DIR
require_var QAD_CHECKPOINT_ROOT
require_var DATACACHE_DIR

# =============================================================================
# OPTIONAL CONFIG WITH DEFAULTS
# =============================================================================

# Parallelism defaults
EP_SIZE="${EP_SIZE:-1}"
PP_SIZE="${PP_SIZE:-1}"
NUM_GPUS="${NUM_GPUS:-8}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"

# Training schedule (derived from TRAIN_SAMPLES if not set)
LR_DECAY_SAMPLES="${LR_DECAY_SAMPLES:-$(( TRAIN_SAMPLES * 99 / 100 ))}"
LR_WARMUP_SAMPLES="${LR_WARMUP_SAMPLES:-$(( TRAIN_SAMPLES / 100 ))}"

# Checkpoint intervals
SAVE_RETAIN_INTERVAL="${SAVE_RETAIN_INTERVAL:-$SAVE_INTERVAL}"
EVAL_INTERVAL="${EVAL_INTERVAL:-$SAVE_INTERVAL}"
EVAL_ITERS="${EVAL_ITERS:-20}"

# Optional overrides
MAX_SEQ="${MAX_SEQ:-}"
RUN_TAG="${RUN_TAG:-}"
KD_CFG_PATH="${KD_CFG_PATH:-}"
ITERATIONS_TO_SKIP="${ITERATIONS_TO_SKIP:-}"

# MoE performance flags
ENABLE_MOE_PERF="${ENABLE_MOE_PERF:-1}"
ENABLE_MOE_EXPERIMENTAL="${ENABLE_MOE_EXPERIMENTAL:-0}"

# Logging
LOG_PARAMS_NORM="${LOG_PARAMS_NORM:-}"

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

require_file "$STUDENT_CONFIG_FILE" "Student model config"
log_info "Loading student model config from: ${STUDENT_CONFIG_FILE}"

# Temporarily disable strict mode for external config (may use unset vars)
set +u
source "$STUDENT_CONFIG_FILE"
set -u

STUDENT_MODEL_ARGS="${MODEL_ARGS}"

# Log params norm setting
if [[ "${LOG_PARAMS_NORM}" == "1" ]]; then
    LOG_PARAMS_NORM_ARG="--log-params-norm"
elif [[ "$IS_MOE" == "true" ]]; then
    LOG_PARAMS_NORM_ARG=""
    log_warn "log-params-norm disabled for MoE model to save memory"
else
    LOG_PARAMS_NORM_ARG="--log-params-norm"
fi

log_info "Model: ${STUDENT_MODEL}"
log_info "Parallelism: TP=${TP_SIZE}, PP=${PP_SIZE}, EP=${EP_SIZE}, MBS=${MBS}, MoE=${IS_MOE}"

# =============================================================================
# CHECKPOINT VALIDATION
# =============================================================================

require_dir "$STUDENT_CKPT" "Student checkpoint"
require_dir "$TEACHER_CKPT" "Teacher checkpoint"
require_file "$TEACHER_MODEL_CONFIG" "Teacher model config"

log_info "Student checkpoint: ${STUDENT_CKPT}"
log_info "Teacher checkpoint: ${TEACHER_CKPT}"

# =============================================================================
# OUTPUT PATH SETUP
# =============================================================================

DATETIME=$(date +'date_%y-%m-%d_time_%H-%M-%S')
STUDENT_CKPT_NAME=$(basename "${STUDENT_CKPT}")
TEACHER_CKPT_NAME=$(basename "${TEACHER_CKPT}")

# Build descriptive run name from hyperparameters
TAG_PARTS="lr$(sanitize_for_path "$LR")"
TAG_PARTS="${TAG_PARTS}-minlr$(sanitize_for_path "$MIN_LR")"
TAG_PARTS="${TAG_PARTS}-decay$(sanitize_for_path "$LR_DECAY_STYLE")"
[[ -n "$MAX_SEQ" ]] && TAG_PARTS="${TAG_PARTS}-seq${MAX_SEQ}"
[[ -n "$RUN_TAG" ]] && TAG_PARTS="${TAG_PARTS}-tag$(sanitize_for_path "$RUN_TAG")"

OUTPUT_ROOT="${QAD_CHECKPOINT_ROOT}/${STUDENT_CKPT_NAME}-Teacher-${TEACHER_CKPT_NAME}-Data-${DATASET_NAME}-${TAG_PARTS}"
NAME="${STUDENT_CKPT_NAME}"

RUN_DIR="${OUTPUT_ROOT}"
LOGS_DIR="${RUN_DIR}/logs"
CHECKPOINT_DIR="${RUN_DIR}/checkpoints/${NAME}"
TENSORBOARD_DIR="${RUN_DIR}/tensorboard/${NAME}"

# Create directories
mkdir -p "${LOGS_DIR}" "${CHECKPOINT_DIR}" "${DATACACHE_DIR}" "${TENSORBOARD_DIR}"

# =============================================================================
# RESUME LOGIC
# =============================================================================

if [[ -f "${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt" ]]; then
    log_info "Resuming from existing checkpoint: ${CHECKPOINT_DIR}"
    LOAD_CHECKPOINT_DIR="${CHECKPOINT_DIR}"
    FINETUNE_FLAG=""
    LOAD_OPTIM_ARGS=""
    CKPT_PARALLEL_LOAD_ARG="--ckpt-fully-parallel-load"
else
    log_info "Starting fresh from base student checkpoint"
    LOAD_CHECKPOINT_DIR="${STUDENT_CKPT}"
    FINETUNE_FLAG="--finetune"
    LOAD_OPTIM_ARGS="--no-load-optim --no-load-rng"
    CKPT_PARALLEL_LOAD_ARG=""
fi

# =============================================================================
# TRAINING CONFIGURATION LOGGING
# =============================================================================

ENV_LOG="${LOGS_DIR}/${NAME}_${DATETIME}.env.log"

{
    echo "========================================"
    echo "QAD Training: ${STUDENT_MODEL}"
    echo "Time: ${DATETIME}"
    echo "========================================"
    echo ""
    echo "MODEL CONFIG"
    echo "  Student: ${STUDENT_MODEL}"
    echo "  Teacher: ${TEACHER_MODEL}"
    echo "  Config: ${STUDENT_CONFIG_FILE}"
    echo "  MoE: ${IS_MOE}"
    echo ""
    echo "TRAINING HYPERPARAMETERS"
    echo "  LR: ${LR}, Min LR: ${MIN_LR}"
    echo "  LR Decay: ${LR_DECAY_STYLE}"
    echo "  GBS: ${GBS}, MBS: ${MBS}"
    echo "  Train Samples: ${TRAIN_SAMPLES}"
    echo "  Save Interval: ${SAVE_INTERVAL}, Log Interval: ${LOG_INTERVAL}"
    echo ""
    echo "PARALLELISM"
    echo "  TP: ${TP_SIZE}, PP: ${PP_SIZE}, EP: ${EP_SIZE}"
    echo "  Nodes: ${NNODES}, GPUs/node: ${NUM_GPUS}"
    echo "  Total GPUs: $((NNODES * NUM_GPUS))"
    echo ""
    echo "PATHS"
    echo "  MLM_DIR: ${MLM_DIR}"
    echo "  Checkpoint: ${CHECKPOINT_DIR}"
    echo "  TensorBoard: ${TENSORBOARD_DIR}"
    echo ""
    echo "ENVIRONMENT"
    env
    echo "========================================"
} | tee "$ENV_LOG"

# =============================================================================
# BUILD TRAINING ARGUMENTS
# =============================================================================

# Checkpoint loading
CHECKPOINT_ARGS=" \
    --auto-detect-ckpt-format \
    --export-te-mcore-model \
    --dist-ckpt-strictness log_unexpected \
    ${FINETUNE_FLAG} \
    ${LOAD_OPTIM_ARGS} \
    --load ${LOAD_CHECKPOINT_DIR} \
    --export-kd-teacher-load ${TEACHER_CKPT} \
    --teacher-model-config ${TEACHER_MODEL_CONFIG}"

# KD config (optional)
if [[ -n "$KD_CFG_PATH" && -f "$KD_CFG_PATH" ]]; then
    CHECKPOINT_ARGS="${CHECKPOINT_ARGS} --export-kd-cfg ${KD_CFG_PATH}"
    log_info "Using KD config: ${KD_CFG_PATH}"
fi

# Tokenizer
TOKENIZER_ARGS=" \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL}"

# Data
DATA_ARGS=" \
    --per-split-data-args-path ${BLEND_PATH} \
    --data-cache-path ${DATACACHE_DIR} \
    --no-mmap-bin-files \
    --num-dataset-builder-threads 16 \
    --no-create-attention-mask-in-dataloader"

# Sequence length override
SEQ_ARGS=""
if [[ -n "$MAX_SEQ" ]]; then
    SEQ_ARGS="--seq-length ${MAX_SEQ} --max-position-embeddings ${MAX_SEQ}"
    log_info "Sequence length override: ${MAX_SEQ}"
fi

# Training
TRAINING_ARGS=" \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --train-samples ${TRAIN_SAMPLES} \
    --lr-decay-samples ${LR_DECAY_SAMPLES} \
    --lr-warmup-samples ${LR_WARMUP_SAMPLES} \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --bf16 \
    ${SEQ_ARGS}"

# Optimizer
OPTIMIZER_ARGS=" \
    --lr ${LR} \
    --min-lr ${MIN_LR} \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --lr-decay-style ${LR_DECAY_STYLE} \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --overlap-param-gather"

# Parallelism
PARALLEL_ARGS=" \
    --tensor-model-parallel-size ${TP_SIZE} \
    --pipeline-model-parallel-size ${PP_SIZE} \
    --distributed-timeout-minutes 360 \
    --disable-gloo-process-groups \
    --ddp-num-buckets 7"

# Expert parallelism for MoE
if [[ "$IS_MOE" == "true" && "$EP_SIZE" -gt 1 ]]; then
    PARALLEL_ARGS="${PARALLEL_ARGS} --expert-model-parallel-size ${EP_SIZE}"
    log_info "MoE Expert Parallelism: EP=${EP_SIZE}"
fi

# Sequence parallel (add if not in model config)
if ! echo "$STUDENT_MODEL_ARGS" | grep -q "sequence-parallel"; then
    PARALLEL_ARGS="${PARALLEL_ARGS} --sequence-parallel"
fi

# MoE performance optimizations
MOE_PERF_ARGS=""
if [[ "$IS_MOE" == "true" && "$ENABLE_MOE_PERF" == "1" ]]; then
    log_info "MoE Performance Optimizations: ENABLED"
    MOE_PERF_ARGS=" \
        --moe-token-dispatcher-type alltoall \
        --moe-shared-expert-overlap \
        --moe-permute-fusion \
        --moe-grouped-gemm \
        --cross-entropy-loss-fusion \
        --cross-entropy-fusion-impl native"
    
    if [[ "$ENABLE_MOE_EXPERIMENTAL" == "1" ]]; then
        MOE_PERF_ARGS="${MOE_PERF_ARGS} --enable-experimental"
        log_warn "Experimental MoE features enabled"
    fi
elif [[ "$IS_MOE" == "true" ]]; then
    log_warn "MoE Performance Optimizations: DISABLED"
fi

# Memory optimization
MEMORY_ARGS=" \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1 \
    --no-gradient-accumulation-fusion"

# Checkpoint saving
SAVE_ARGS=" \
    --save ${CHECKPOINT_DIR} \
    --save-interval ${SAVE_INTERVAL} \
    --save-retain-interval ${SAVE_RETAIN_INTERVAL} \
    --ckpt-format torch_dist \
    --ckpt-fully-parallel-save \
    --ckpt-assume-constant-structure \
    ${CKPT_PARALLEL_LOAD_ARG}"

# Logging
LOGGING_ARGS=" \
    --log-interval ${LOG_INTERVAL} \
    --eval-iters ${EVAL_ITERS} \
    --eval-interval ${EVAL_INTERVAL} \
    --log-progress \
    --timing-log-option minmax \
    ${LOG_PARAMS_NORM_ARG:-} \
    --log-num-zeros-in-grad \
    --log-throughput \
    --log-straggler \
    --disable-straggler-on-startup \
    --straggler-minmax-count 16 \
    --tensorboard-dir ${TENSORBOARD_DIR}"

# Runtime
RUNTIME_ARGS=" \
    --exit-duration-in-mins 1200 \
    --num-workers 8 \
    --no-check-for-nan-in-loss-and-grad"

# Combine all arguments
ALL_ARGS=" \
    ${CHECKPOINT_ARGS} \
    ${STUDENT_MODEL_ARGS} \
    ${TOKENIZER_ARGS} \
    ${DATA_ARGS} \
    ${TRAINING_ARGS} \
    ${OPTIMIZER_ARGS} \
    ${PARALLEL_ARGS} \
    ${MOE_PERF_ARGS} \
    ${MEMORY_ARGS} \
    ${SAVE_ARGS} \
    ${LOGGING_ARGS} \
    ${RUNTIME_ARGS}"

# Optional: iterations to skip
[[ -n "$ITERATIONS_TO_SKIP" ]] && ALL_ARGS="${ALL_ARGS} --iterations-to-skip ${ITERATIONS_TO_SKIP}"

# =============================================================================
# LAUNCH TRAINING
# =============================================================================

export PYTHONPATH="${MODELOPT_DIR}:${MLM_DIR}:${PYTHONPATH:-}"

LOG_FILE="${LOGS_DIR}/${NAME}_qad_${DATETIME}.log"

log_info "Starting training..."
log_info "Log file: ${LOG_FILE}"
log_info "Distributed: ${NNODES} nodes × ${NUM_GPUS} GPUs = $((NNODES * NUM_GPUS)) total"

torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${MLM_DIR}/pretrain_gpt.py" ${ALL_ARGS} 2>&1 | tee "${LOG_FILE}"

log_info "Training completed. Logs: ${LOG_FILE}"
