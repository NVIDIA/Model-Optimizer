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

# QAD (Quantization-Aware Distillation) Training Script
# Usage: bash qad.sh --config configs/your-config.conf

set -euo pipefail

# === Helpers ===
die() { echo "[ERROR] $*" >&2; exit 1; }
log_info() { echo "[INFO] $*"; }
log_warn() { echo "[WARN] $*"; }
require_var() { [[ -n "${!1:-}" ]] || die "$1 must be set in config"; }
require_file() { [[ -f "$1" ]] || die "${2:-File} not found: $1"; }
require_dir() { [[ -d "$1" ]] || die "${2:-Directory} not found: $1"; }
sanitize() { echo "$1" | sed -e 's/[\/ :]/_/g' -e 's/[=]/_/g'; }

# === Environment ===
export NCCL_IB_SL=1
export NCCL_IB_TIMEOUT=19
export NCCL_P2P_NET_CHUNKSIZE=2097152
export NCCL_DEBUG=WARN
export NCCL_SHM_DISABLE=1
export NCCL_NVLS_ENABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export UB_TIMEOUT=720
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export TORCHINDUCTOR_COMPILE_THREADS=1
export TORCH_COMPILE_DISABLE=1
export PYTORCH_NO_CUDA_MEMORY_CACHING=0
export TORCH_DISTRIBUTED_DEBUG=OFF
export PYTORCH_JIT=0
export TORCH_USE_CUDA_DSA=0
export GLOO_SOCKET_IFNAME=ibp26s0

# === Argument Parsing ===
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE=""
HF_TOKEN_ARG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --config|-c) CONFIG_FILE="$2"; shift 2;;
        --hf-token) HF_TOKEN_ARG="$2"; shift 2;;
        *) die "Unknown argument: $1";;
    esac
done

# HuggingFace token
[[ -n "$HF_TOKEN_ARG" ]] && export HF_TOKEN="$HF_TOKEN_ARG"
[[ -n "${HF_TOKEN:-}" ]] && export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" && log_info "HuggingFace token configured"

# === Load Config ===
if [[ -z "$CONFIG_FILE" ]]; then
    die "Config file required. Use --config <path>\nAvailable: $(ls -1 "${SCRIPT_DIR}/configs/"*.conf 2>/dev/null | tr '\n' ' ')"
fi
[[ "$CONFIG_FILE" = /* ]] || CONFIG_FILE="${SCRIPT_DIR}/${CONFIG_FILE}"
require_file "$CONFIG_FILE" "Config file"
log_info "Loading config: ${CONFIG_FILE}"
source "$CONFIG_FILE"

# === Validate Required Config ===
for v in LR GBS MIN_LR LR_DECAY_STYLE SAVE_INTERVAL LOG_INTERVAL \
         STUDENT_MODEL TEACHER_MODEL DATASET_NAME BLEND_PATH TRAIN_SAMPLES IS_MOE TOKENIZER_MODEL \
         TP_SIZE MBS STUDENT_CKPT TEACHER_CKPT TEACHER_MODEL_CONFIG \
         STUDENT_CONFIG_FILE MLM_DIR MODELOPT_DIR QAD_CHECKPOINT_ROOT DATACACHE_DIR; do
    require_var "$v"
done

# === Defaults for Optional Config ===
EP_SIZE="${EP_SIZE:-1}"
PP_SIZE="${PP_SIZE:-1}"
NUM_GPUS="${NUM_GPUS:-8}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"
LR_DECAY_SAMPLES="${LR_DECAY_SAMPLES:-$(( TRAIN_SAMPLES * 99 / 100 ))}"
LR_WARMUP_SAMPLES="${LR_WARMUP_SAMPLES:-$(( TRAIN_SAMPLES / 100 ))}"
SAVE_RETAIN_INTERVAL="${SAVE_RETAIN_INTERVAL:-$SAVE_INTERVAL}"
EVAL_INTERVAL="${EVAL_INTERVAL:-$SAVE_INTERVAL}"
EVAL_ITERS="${EVAL_ITERS:-20}"
MAX_SEQ="${MAX_SEQ:-}"
RUN_TAG="${RUN_TAG:-}"
KD_CFG_PATH="${KD_CFG_PATH:-}"
ITERATIONS_TO_SKIP="${ITERATIONS_TO_SKIP:-}"
ENABLE_MOE_PERF="${ENABLE_MOE_PERF:-1}"
ENABLE_MOE_EXPERIMENTAL="${ENABLE_MOE_EXPERIMENTAL:-0}"
LOG_PARAMS_NORM="${LOG_PARAMS_NORM:-}"

# === Load Student Model Config ===
require_file "$STUDENT_CONFIG_FILE" "Student model config"
log_info "Loading student model config: ${STUDENT_CONFIG_FILE}"
set +u; source "$STUDENT_CONFIG_FILE"; set -u
STUDENT_MODEL_ARGS="${MODEL_ARGS}"

# Log params norm (disabled for MoE to save memory)
if [[ "${LOG_PARAMS_NORM}" == "1" ]]; then
    LOG_PARAMS_NORM_ARG="--log-params-norm"
elif [[ "$IS_MOE" == "true" ]]; then
    LOG_PARAMS_NORM_ARG=""
    log_warn "log-params-norm disabled for MoE model"
else
    LOG_PARAMS_NORM_ARG="--log-params-norm"
fi

log_info "Model: ${STUDENT_MODEL} | TP=${TP_SIZE} PP=${PP_SIZE} EP=${EP_SIZE} MBS=${MBS} MoE=${IS_MOE}"

# === Validate Checkpoints ===
require_dir "$STUDENT_CKPT" "Student checkpoint"
require_dir "$TEACHER_CKPT" "Teacher checkpoint"
require_file "$TEACHER_MODEL_CONFIG" "Teacher model config"
log_info "Student: ${STUDENT_CKPT}"
log_info "Teacher: ${TEACHER_CKPT}"

# === Output Paths ===
DATETIME=$(date +'date_%y-%m-%d_time_%H-%M-%S')
STUDENT_CKPT_NAME=$(basename "${STUDENT_CKPT}")
TEACHER_CKPT_NAME=$(basename "${TEACHER_CKPT}")

TAG_PARTS="lr$(sanitize "$LR")-minlr$(sanitize "$MIN_LR")-decay$(sanitize "$LR_DECAY_STYLE")"
[[ -n "$MAX_SEQ" ]] && TAG_PARTS="${TAG_PARTS}-seq${MAX_SEQ}"
[[ -n "$RUN_TAG" ]] && TAG_PARTS="${TAG_PARTS}-tag$(sanitize "$RUN_TAG")"

OUTPUT_ROOT="${QAD_CHECKPOINT_ROOT}/${STUDENT_CKPT_NAME}-Teacher-${TEACHER_CKPT_NAME}-Data-${DATASET_NAME}-${TAG_PARTS}"
CHECKPOINT_DIR="${OUTPUT_ROOT}/checkpoints/${STUDENT_CKPT_NAME}"
TENSORBOARD_DIR="${OUTPUT_ROOT}/tensorboard/${STUDENT_CKPT_NAME}"
LOGS_DIR="${OUTPUT_ROOT}/logs"
mkdir -p "${LOGS_DIR}" "${CHECKPOINT_DIR}" "${DATACACHE_DIR}" "${TENSORBOARD_DIR}"

# === Resume Logic ===
if [[ -f "${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt" ]]; then
    log_info "Resuming from: ${CHECKPOINT_DIR}"
    LOAD_CHECKPOINT_DIR="${CHECKPOINT_DIR}"
    FINETUNE_FLAG=""
    LOAD_OPTIM_ARGS=""
    CKPT_PARALLEL_LOAD_ARG="--ckpt-fully-parallel-load"
else
    log_info "Starting fresh from base checkpoint"
    LOAD_CHECKPOINT_DIR="${STUDENT_CKPT}"
    FINETUNE_FLAG="--finetune"
    LOAD_OPTIM_ARGS="--no-load-optim --no-load-rng"
    CKPT_PARALLEL_LOAD_ARG=""
fi

# === Log Configuration ===
ENV_LOG="${LOGS_DIR}/${STUDENT_CKPT_NAME}_${DATETIME}.env.log"
{
    echo "=== QAD Training: ${STUDENT_MODEL} ==="
    echo "Time: ${DATETIME}"
    echo "LR=${LR} MinLR=${MIN_LR} Decay=${LR_DECAY_STYLE} GBS=${GBS} MBS=${MBS}"
    echo "TrainSamples=${TRAIN_SAMPLES} SaveInterval=${SAVE_INTERVAL} LogInterval=${LOG_INTERVAL}"
    echo "TP=${TP_SIZE} PP=${PP_SIZE} EP=${EP_SIZE} Nodes=${NNODES} GPUs/node=${NUM_GPUS}"
    echo "Checkpoint: ${CHECKPOINT_DIR}"
    echo "TensorBoard: ${TENSORBOARD_DIR}"
    env
} > "$ENV_LOG"

# === Build Training Arguments ===

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

# === Launch Training ===
export PYTHONPATH="${MODELOPT_DIR}:${MLM_DIR}:${PYTHONPATH:-}"
LOG_FILE="${LOGS_DIR}/${STUDENT_CKPT_NAME}_qad_${DATETIME}.log"

log_info "Starting training..."
log_info "Log file: ${LOG_FILE}"
log_info "Distributed: ${NNODES} nodes x ${NUM_GPUS} GPUs = $((NNODES * NUM_GPUS)) total"

torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${MLM_DIR}/pretrain_gpt.py" ${ALL_ARGS} 2>&1 | tee "${LOG_FILE}"

log_info "Training completed. Logs: ${LOG_FILE}"
