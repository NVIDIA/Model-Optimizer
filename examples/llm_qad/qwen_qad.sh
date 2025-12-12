#!/bin/bash
# Generic QAD training script for Qwen models - Docker/Interactive Version
# Supports: Qwen3-8B, Qwen3-30B-A3B (MoE), and other Qwen variants
#
# Usage:
#   # With config file (recommended)
#   bash qwen_qad.sh --config configs/qwen3-8b-default.conf
#   bash qwen_qad.sh --config configs/qwen3-8b-nemotron.conf
#
#   # With HuggingFace token (secure, not logged)
#   bash qwen_qad.sh --hf-token hf_xxx --config configs/qwen3-8b-default.conf
#
#   # With command line args
#   bash qwen_qad.sh [LR] [TEACHER_MODEL] [DATASET_NAME] [STUDENT_MODEL] [KD_CFG_PATH]
#   bash qwen_qad.sh 1e-6 Qwen3-8B nemotron Qwen3-8B
#   bash qwen_qad.sh 1e-6 Qwen3-8B nemotron Qwen3-8B /path/to/kd_config.yaml
#
#   # Config + override
#   LR=1e-5 bash qwen_qad.sh --config configs/qwen3-8b-default.conf
#
# Get interactive node:
#   srun -A coreai_dlalgo_modelopt --nodes=1 -p batch --mpi=pmix \
#     -J qwen-qad:dev \
#     --container-image=/lustre/.../pytorch_25.06-py3.sqsh \
#     --container-mounts="/lustre/fsw:/lustre/fsw" \
#     --container-workdir="/lustre/.../workspace" \
#     -t 4:0:0 --pty bash

set -e  # Exit on error

export NCCL_IB_SL=1
export NCCL_IB_TIMEOUT=19
export UB_TIMEOUT=720
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export NCCL_P2P_NET_CHUNKSIZE=2097152
export NCCL_DEBUG=WARN
export NCCL_SHM_DISABLE=1
export NCCL_NVLS_ENABLE=0
export GLOO_SOCKET_IFNAME=ibp26s0
# Disable torch inductor subprocess compilation to avoid CUDA fork issues
export TORCHINDUCTOR_COMPILE_THREADS=1
# Disable PyTorch compilation to avoid Triton/cubin errors during training
export TORCH_COMPILE_DISABLE=1
# Workaround for B300 autograd issues with quantization
export PYTORCH_NO_CUDA_MEMORY_CACHING=0
export TORCH_DISTRIBUTED_DEBUG=OFF
# Force fallback for missing autograd kernels
export PYTORCH_JIT=0
export TORCH_USE_CUDA_DSA=0

# HuggingFace token for accessing gated models (avoids rate limiting)
# Set via: export HF_TOKEN=hf_xxx or in config file
if [ -n "${HF_TOKEN:-}" ]; then
    export HF_TOKEN="${HF_TOKEN}"
    export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"  # Legacy variable name
    echo "🔑 HuggingFace token configured"
fi

########################################################
#### CONFIG FILE LOADING ####
########################################################

SCRIPT_PATH=$(realpath "$0")
SCRIPT_DIR=$(dirname ${SCRIPT_PATH})
CONFIG_FILE=""
HF_TOKEN_ARG=""

# Parse arguments
POSITIONAL_ARGS=()
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
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# Set HF_TOKEN from arg (takes precedence, doesn't appear in logs)
if [ -n "$HF_TOKEN_ARG" ]; then
    export HF_TOKEN="$HF_TOKEN_ARG"
fi
# Restore positional args (handle empty array for set -u)
if [ ${#POSITIONAL_ARGS[@]} -gt 0 ]; then
    set -- "${POSITIONAL_ARGS[@]}"
else
    set --
fi

# Load config file if specified
if [ -n "$CONFIG_FILE" ]; then
    # Handle relative paths
    if [[ ! "$CONFIG_FILE" = /* ]]; then
        CONFIG_FILE="${SCRIPT_DIR}/${CONFIG_FILE}"
    fi
    
    if [ -f "$CONFIG_FILE" ]; then
        echo "📄 Loading config from: ${CONFIG_FILE}"
        source "$CONFIG_FILE"
    else
        echo "❌ ERROR: Config file not found: ${CONFIG_FILE}"
        echo "Available configs:"
        ls -1 "${SCRIPT_DIR}/configs/"*.conf 2>/dev/null || echo "  (none found)"
        exit 1
    fi
fi

########################################################
#### CONFIGURATION PARAMETERS ####
########################################################

CURRENT_DIR=$(pwd)

# Command line args override config/env
# Order: LR, TEACHER_MODEL, DATASET_NAME, STUDENT_MODEL, KD_CFG_PATH
LR="${1:-${LR:-1e-6}}"
TEACHER_MODEL="${2:-${TEACHER_MODEL:-Qwen3-8B}}"
DATASET_NAME="${3:-${DATASET_NAME:-openscience}}"
STUDENT_MODEL="${4:-${STUDENT_MODEL:-Qwen3-8B}}"
KD_CFG_PATH="${5:-${KD_CFG_PATH:-}}"

# Allow environment variable override (takes precedence)
STUDENT_MODEL="${STUDENT_MODEL_ENV:-$STUDENT_MODEL}"
TEACHER_MODEL="${TEACHER_MODEL_ENV:-$TEACHER_MODEL}"

########################################################
#### PATH CONFIGURATION ####
########################################################

MLM_DIR="${MLM_DIR:-/lustre/fsw/coreai_dlalgo_modelopt/weimingc/workspace/Megatron-LM}"
MODELOPT_DIR="${MODELOPT_DIR:-/lustre/fsw/coreai_dlalgo_modelopt/weimingc/workspace/TensorRT-Model-Optimizer}"
MODEL_CONF_DIR="${MLM_DIR}/examples/post_training/modelopt/conf/Qwen"
MODELS_ROOT="${MODELS_ROOT:-/lustre/fsw/coreai_dlalgo_modelopt/weimingc/models}"
QAD_CHECKPOINT_ROOT="${QAD_CHECKPOINT_ROOT:-/lustre/fsw/coreai_dlalgo_modelopt/weimingc/checkpoints}"
DATACACHE_DIR="${DATACACHE_DIR:-/lustre/fsw/coreai_dlalgo_modelopt/weimingc/data_cache}"

########################################################
#### MODEL CONFIGURATION - Source from conf file ####
########################################################

# Load student model architecture config (MODEL_ARGS)
# Config file path can be set explicitly or auto-detected
STUDENT_CONFIG_FILE="${STUDENT_CONFIG_FILE:-${MODEL_CONF_DIR}/${STUDENT_MODEL}.sh}"

if [ ! -f "${STUDENT_CONFIG_FILE}" ]; then
    echo "❌ ERROR: Student model config not found: ${STUDENT_CONFIG_FILE}"
    echo "Available model configs:"
    ls -1 "${MODEL_CONF_DIR}/"*.sh 2>/dev/null | xargs -n1 basename | sed 's/.sh$//' || echo "  (none)"
    exit 1
fi

echo "📄 Loading student model config from: ${STUDENT_CONFIG_FILE}"
source "${STUDENT_CONFIG_FILE}"
STUDENT_MODEL_ARGS="${MODEL_ARGS}"

# Parallelism settings (from config file, required)
TP_SIZE="${TP_SIZE:?ERROR: TP_SIZE must be set in config}"
EP_SIZE="${EP_SIZE:-1}"
MBS="${MBS:?ERROR: MBS must be set in config}"

# Detect MoE from EP_SIZE
if [ "${EP_SIZE}" -gt 1 ]; then
    IS_MOE=true
else
    IS_MOE=false
fi

# Disable log-params-norm for MoE models by default (causes OOM due to FP32 conversion)
# Can be overridden with LOG_PARAMS_NORM=1 in config
if [ "${LOG_PARAMS_NORM:-}" = "1" ]; then
    LOG_PARAMS_NORM_ARG="--log-params-norm"
elif [ "$IS_MOE" = "true" ]; then
    LOG_PARAMS_NORM_ARG=""  # Disabled for MoE to save memory
    echo "⚠️  log-params-norm disabled for MoE model (saves ~2GB memory)"
else
    LOG_PARAMS_NORM_ARG="--log-params-norm"
fi

echo "🔧 Model: ${STUDENT_MODEL}"
echo "   TP=${TP_SIZE}, EP=${EP_SIZE}, MBS=${MBS}, MoE=${IS_MOE}"

########################################################
#### CHECKPOINT PATHS (REQUIRED) ####
########################################################

# STUDENT_CKPT: Path to student checkpoint (REQUIRED)
# TEACHER_CKPT: Path to teacher checkpoint (REQUIRED)
# These must be set in config file or environment

if [ -z "${STUDENT_CKPT:-}" ]; then
    echo "❌ ERROR: STUDENT_CKPT is required. Set it in config or environment."
    exit 1
fi
if [ -z "${TEACHER_CKPT:-}" ]; then
    echo "❌ ERROR: TEACHER_CKPT is required. Set it in config or environment."
    exit 1
fi

BASE_STUDENT_CKPT="${STUDENT_CKPT}"
TEACHER_CKPT_DIR="${TEACHER_CKPT}"

# TEACHER_MODEL_CONFIG is required
if [ -z "${TEACHER_MODEL_CONFIG:-}" ]; then
    echo "❌ ERROR: TEACHER_MODEL_CONFIG is required. Set it in config or environment."
    exit 1
fi

if [ ! -f "${TEACHER_MODEL_CONFIG}" ]; then
    echo "❌ ERROR: Teacher model config file not found: ${TEACHER_MODEL_CONFIG}"
    exit 1
fi

echo "📚 Student checkpoint: ${STUDENT_CKPT}"
echo "🎓 Teacher checkpoint: ${TEACHER_CKPT}"

########################################################
#### OUTPUT PATHS ####
########################################################

DATETIME=$(date +'date_%y-%m-%d_time_%H-%M-%S')

# Use checkpoint directory name for output path
STUDENT_CKPT_NAME=$(basename "${STUDENT_CKPT}")
TEACHER_CKPT_NAME=$(basename "${TEACHER_CKPT}")
OUTPUT_ROOT="${QAD_CHECKPOINT_ROOT}/${STUDENT_CKPT_NAME}-Teacher-${TEACHER_CKPT_NAME}-Data-${DATASET_NAME}-lr${LR}"
NAME="${STUDENT_CKPT_NAME}"

RUN_DIR="${OUTPUT_ROOT}"
LOGS_DIR="${RUN_DIR}/logs"
CHECKPOINT_DIR="${RUN_DIR}/checkpoints/${NAME}"
TENSORBOARD_DIR="${RUN_DIR}/tensorboard/${NAME}"
ENV_LOG_FILENAME=${NAME}_${DATETIME}.env.log

########################################################
#### KD CONFIG ####
########################################################

# KD_CFG_PATH: Path to custom KD config YAML (optional)
# If set, uses custom distillation configuration
if [ -n "${KD_CFG_PATH}" ]; then
    if [ -f "${KD_CFG_PATH}" ]; then
        KD_CFG_ARGS="--export-kd-cfg ${KD_CFG_PATH}"
        echo "🎓 Using KD config: ${KD_CFG_PATH}"
    else
        echo "⚠️  Warning: KD config not found: ${KD_CFG_PATH}, using default KD settings"
        KD_CFG_ARGS=""
    fi
else
    KD_CFG_ARGS=""
fi

########################################################
#### DATASET SELECTION ####
########################################################

# Select Datablend based on argument
# Naming convention:
#   - Plain text: datablend_<dataset>.json
#   - With COT (chain-of-thought): datablend_<dataset>_cot.json
#   - With chat template: datablend_<dataset>_chat.json
#   - With both COT and chat: datablend_<dataset>_cot_chat.json
case "$DATASET_NAME" in
    # ====================
    # Nemotron-v1 options (plain text)
    # ====================
    nemotron_10pct|nemotron_all_10pct)
        BLEND_PATH="${BLEND_PATH:-/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/datablend_nemotron_all_10pct.json}"
        DEFAULT_TRAIN_SAMPLES=2500000
        echo "📊 Using Nemotron-v1 ALL Subjects @ 10% (~2.5M samples)"
        ;;
    nemotron|nemotron_30pct|nemotron_all_30pct)
        BLEND_PATH="${BLEND_PATH:-/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/datablend_nemotron_all_30pct.json}"
        DEFAULT_TRAIN_SAMPLES=7500000
        echo "📊 Using Nemotron-v1 ALL Subjects @ 30% (~7.5M samples)"
        ;;
    nemotron_50pct|nemotron_all_50pct)
        BLEND_PATH="${BLEND_PATH:-/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/datablend_nemotron_all_50pct.json}"
        DEFAULT_TRAIN_SAMPLES=12500000
        echo "📊 Using Nemotron-v1 ALL Subjects @ 50% (~12.5M samples)"
        ;;
    nemotron_100pct|nemotron_all_100pct|nemotron_full)
        BLEND_PATH="${BLEND_PATH:-/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/datablend_nemotron_all_100pct.json}"
        DEFAULT_TRAIN_SAMPLES=25000000
        echo "📊 Using Nemotron-v1 ALL Subjects @ 100% (~25M samples)"
        ;;
    nemotron_stem)
        BLEND_PATH="${BLEND_PATH:-/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/datablend_nemotron_stem.json}"
        DEFAULT_TRAIN_SAMPLES=5000000
        echo "📊 Using Nemotron-v1 STEM Dataset (Best for MMLU)"
        ;;
    nemotron_math)
        BLEND_PATH="${BLEND_PATH:-/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/datablend_nemotron_math.json}"
        DEFAULT_TRAIN_SAMPLES=2000000
        echo "📊 Using Nemotron-v1 Math Dataset"
        ;;

    # ====================
    # Nemotron-v1 with COT (chain-of-thought reasoning)
    # ====================
    nemotron_10pct_cot|nemotron_all_10pct_cot)
        BLEND_PATH="${BLEND_PATH:-/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/datablend_nemotron_all_10pct_cot.json}"
        DEFAULT_TRAIN_SAMPLES=2500000
        echo "📊 Using Nemotron-v1 ALL @ 10% + COT (~2.5M samples)"
        ;;
    nemotron_30pct_cot|nemotron_all_30pct_cot)
        BLEND_PATH="${BLEND_PATH:-/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/datablend_nemotron_all_30pct_cot.json}"
        DEFAULT_TRAIN_SAMPLES=7500000
        echo "📊 Using Nemotron-v1 ALL @ 30% + COT (~7.5M samples)"
        ;;
    nemotron_50pct_cot|nemotron_all_50pct_cot)
        BLEND_PATH="${BLEND_PATH:-/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/datablend_nemotron_all_50pct_cot.json}"
        DEFAULT_TRAIN_SAMPLES=12500000
        echo "📊 Using Nemotron-v1 ALL @ 50% + COT (~12.5M samples)"
        ;;
    nemotron_stem_cot)
        BLEND_PATH="${BLEND_PATH:-/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/datablend_nemotron_stem_cot.json}"
        DEFAULT_TRAIN_SAMPLES=5000000
        echo "📊 Using Nemotron-v1 STEM + COT"
        ;;

    # ====================
    # Nemotron-v1 with chat template (no COT)
    # ====================
    nemotron_10pct_chat|nemotron_all_10pct_chat)
        BLEND_PATH="${BLEND_PATH:-/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/datablend_nemotron_all_10pct_chat.json}"
        DEFAULT_TRAIN_SAMPLES=2500000
        echo "📊 Using Nemotron-v1 ALL @ 10% + Chat Template (~2.5M samples)"
        ;;
    nemotron_chat|nemotron_30pct_chat|nemotron_all_30pct_chat)
        BLEND_PATH="${BLEND_PATH:-/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/datablend_nemotron_all_30pct_chat.json}"
        DEFAULT_TRAIN_SAMPLES=7500000
        echo "📊 Using Nemotron-v1 ALL @ 30% + Chat Template (~7.5M samples)"
        ;;
    nemotron_50pct_chat|nemotron_all_50pct_chat)
        BLEND_PATH="${BLEND_PATH:-/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/datablend_nemotron_all_50pct_chat.json}"
        DEFAULT_TRAIN_SAMPLES=12500000
        echo "📊 Using Nemotron-v1 ALL @ 50% + Chat Template (~12.5M samples)"
        ;;
    nemotron_stem_chat)
        BLEND_PATH="${BLEND_PATH:-/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/datablend_nemotron_stem_chat.json}"
        DEFAULT_TRAIN_SAMPLES=5000000
        echo "📊 Using Nemotron-v1 STEM + Chat Template"
        ;;

    # ====================
    # Nemotron-v1 with COT + chat template
    # ====================
    nemotron_10pct_cot_chat|nemotron_all_10pct_cot_chat)
        BLEND_PATH="${BLEND_PATH:-/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/datablend_nemotron_all_10pct_cot_chat.json}"
        DEFAULT_TRAIN_SAMPLES=2500000
        echo "📊 Using Nemotron-v1 ALL @ 10% + COT + Chat Template (~2.5M samples)"
        ;;
    nemotron_cot_chat|nemotron_30pct_cot_chat|nemotron_all_30pct_cot_chat)
        BLEND_PATH="${BLEND_PATH:-/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/datablend_nemotron_all_30pct_cot_chat.json}"
        DEFAULT_TRAIN_SAMPLES=7500000
        echo "📊 Using Nemotron-v1 ALL @ 30% + COT + Chat Template (~7.5M samples)"
        ;;
    nemotron_50pct_cot_chat|nemotron_all_50pct_cot_chat)
        BLEND_PATH="${BLEND_PATH:-/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/datablend_nemotron_all_50pct_cot_chat.json}"
        DEFAULT_TRAIN_SAMPLES=12500000
        echo "📊 Using Nemotron-v1 ALL @ 50% + COT + Chat Template (~12.5M samples)"
        ;;
    nemotron_stem_cot_chat)
        BLEND_PATH="${BLEND_PATH:-/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/datablend_nemotron_stem_cot_chat.json}"
        DEFAULT_TRAIN_SAMPLES=5000000
        echo "📊 Using Nemotron-v1 STEM + COT + Chat Template"
        ;;

    # ====================
    # Nemotron-v1 individual splits (fine-grained control)
    # Format: nemotron_v1_<split>_<pct>pct_cot_chat
    # ====================
    nemotron_v1_stem|nemotron_v1_stem_30pct_cot_chat)
        BLEND_PATH="${BLEND_PATH:-/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/datablend_nemotron_v1_stem_30pct_cot_chat.json}"
        DEFAULT_TRAIN_SAMPLES=5886717  # ~6M (30% of 20.6M * 0.95 train split)
        echo "📊 Using Nemotron-v1 STEM @ 30% + COT + Chat (~5.9M samples)"
        ;;
    nemotron_v1_math|nemotron_v1_math_30pct_cot_chat)
        BLEND_PATH="${BLEND_PATH:-/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/datablend_nemotron_v1_math_30pct_cot_chat.json}"
        DEFAULT_TRAIN_SAMPLES=582654   # ~583K (30% of 2M * 0.95)
        echo "📊 Using Nemotron-v1 Math @ 30% + COT + Chat (~583K samples)"
        ;;
    nemotron_v1_code|nemotron_v1_code_30pct_cot_chat)
        BLEND_PATH="${BLEND_PATH:-/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/datablend_nemotron_v1_code_30pct_cot_chat.json}"
        DEFAULT_TRAIN_SAMPLES=540472   # ~540K (30% of 1.9M * 0.95)
        echo "📊 Using Nemotron-v1 Code @ 30% + COT + Chat (~540K samples)"
        ;;
    nemotron_v1_chat|nemotron_v1_chat_30pct_cot_chat)
        BLEND_PATH="${BLEND_PATH:-/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/datablend_nemotron_v1_chat_30pct_cot_chat.json}"
        DEFAULT_TRAIN_SAMPLES=212786   # ~213K (30% of 746K * 0.95)
        echo "📊 Using Nemotron-v1 Chat @ 30% + COT + Chat (~213K samples)"
        ;;
    nemotron_v1_all|nemotron_v1_all_en_30pct_cot_chat)
        BLEND_PATH="${BLEND_PATH:-/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/datablend_nemotron_v1_all_en_30pct_cot_chat.json}"
        DEFAULT_TRAIN_SAMPLES=7222629  # Sum of all splits
        echo "📊 Using Nemotron-v1 ALL splits @ 30% + COT + Chat (~7.2M samples)"
        ;;

    # ====================
    # Nemotron-v2 combined options (plain text)
    # Total @ 30%: stem(101K) + math(68K) + code(50K) + chat(179K) = ~398K
    # ====================
    nemotron_v2|nemotron_v2_30pct|nemotron_v2_all_en_30pct)
        BLEND_PATH="${BLEND_PATH:-/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/datablend_nemotron_v2_all_en_30pct.json}"
        DEFAULT_TRAIN_SAMPLES=398198
        echo "📊 Using Nemotron-v2 English @ 30% (~398K samples)"
        ;;
    nemotron_v2_50pct|nemotron_v2_all_en_50pct)
        BLEND_PATH="${BLEND_PATH:-/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/datablend_nemotron_v2_all_en_50pct.json}"
        DEFAULT_TRAIN_SAMPLES=663663
        echo "📊 Using Nemotron-v2 English @ 50% (~664K samples)"
        ;;
    nemotron_v2_multilingual|nemotron_v2_all_multilingual_30pct)
        BLEND_PATH="${BLEND_PATH:-/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/datablend_nemotron_v2_all_multilingual_30pct.json}"
        DEFAULT_TRAIN_SAMPLES=600000
        echo "📊 Using Nemotron-v2 ALL Languages @ 30% (~600K samples)"
        ;;

    # ====================
    # Nemotron-v2 combined with chat template
    # ====================
    nemotron_v2_chat_tmpl|nemotron_v2_30pct_chat|nemotron_v2_all_en_30pct_chat)
        BLEND_PATH="${BLEND_PATH:-/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/datablend_nemotron_v2_all_en_30pct_chat.json}"
        DEFAULT_TRAIN_SAMPLES=398198
        echo "📊 Using Nemotron-v2 English @ 30% + Chat Template (~398K samples)"
        ;;
    nemotron_v2_50pct_chat|nemotron_v2_all_en_50pct_chat)
        BLEND_PATH="${BLEND_PATH:-/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/datablend_nemotron_v2_all_en_50pct_chat.json}"
        DEFAULT_TRAIN_SAMPLES=663663
        echo "📊 Using Nemotron-v2 English @ 50% + Chat Template (~664K samples)"
        ;;

    # ====================
    # Nemotron-v2 combined with COT (chain-of-thought reasoning)
    # ====================
    nemotron_v2_cot|nemotron_v2_30pct_cot|nemotron_v2_all_en_30pct_cot)
        BLEND_PATH="${BLEND_PATH:-/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/datablend_nemotron_v2_all_en_30pct_cot.json}"
        DEFAULT_TRAIN_SAMPLES=398198
        echo "📊 Using Nemotron-v2 English @ 30% + COT (~398K samples)"
        ;;
    nemotron_v2_50pct_cot|nemotron_v2_all_en_50pct_cot)
        BLEND_PATH="${BLEND_PATH:-/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/datablend_nemotron_v2_all_en_50pct_cot.json}"
        DEFAULT_TRAIN_SAMPLES=663663
        echo "📊 Using Nemotron-v2 English @ 50% + COT (~664K samples)"
        ;;

    # ====================
    # Nemotron-v2 combined with COT + chat template
    # ====================
    nemotron_v2_cot_chat|nemotron_v2_30pct_cot_chat|nemotron_v2_all_en_30pct_cot_chat)
        BLEND_PATH="${BLEND_PATH:-/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/datablend_nemotron_v2_all_en_30pct_cot_chat.json}"
        DEFAULT_TRAIN_SAMPLES=398198
        echo "📊 Using Nemotron-v2 English @ 30% + COT + Chat Template (~398K samples)"
        ;;
    nemotron_v2_50pct_cot_chat|nemotron_v2_all_en_50pct_cot_chat)
        BLEND_PATH="${BLEND_PATH:-/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/datablend_nemotron_v2_all_en_50pct_cot_chat.json}"
        DEFAULT_TRAIN_SAMPLES=663663
        echo "📊 Using Nemotron-v2 English @ 50% + COT + Chat Template (~664K samples)"
        ;;

    # ====================
    # Nemotron-v2 individual splits (plain text)
    # ====================
    nemotron_v2_stem|nemotron_v2_stem_30pct)
        BLEND_PATH="${BLEND_PATH:-/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/datablend_nemotron_v2_stem_30pct.json}"
        DEFAULT_TRAIN_SAMPLES=101175
        echo "📊 Using Nemotron-v2 STEM split @ 30% (~101K samples)"
        ;;
    nemotron_v2_math|nemotron_v2_math_30pct)
        BLEND_PATH="${BLEND_PATH:-/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/datablend_nemotron_v2_math_30pct.json}"
        DEFAULT_TRAIN_SAMPLES=68248
        echo "📊 Using Nemotron-v2 Math split @ 30% (~68K samples)"
        ;;
    nemotron_v2_code|nemotron_v2_code_30pct)
        BLEND_PATH="${BLEND_PATH:-/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/datablend_nemotron_v2_code_30pct.json}"
        DEFAULT_TRAIN_SAMPLES=99750
        echo "📊 Using Nemotron-v2 Code split @ 30% (~50K x2 epochs)"
        ;;
    nemotron_v2_chat|nemotron_v2_chat_30pct)
        BLEND_PATH="${BLEND_PATH:-/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/datablend_nemotron_v2_chat_30pct.json}"
        DEFAULT_TRAIN_SAMPLES=178900
        echo "📊 Using Nemotron-v2 Chat split @ 30% (~179K samples)"
        ;;

    # ====================
    # Nemotron-v2 individual splits with chat template
    # ====================
    nemotron_v2_stem_chat|nemotron_v2_stem_30pct_chat)
        BLEND_PATH="${BLEND_PATH:-/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/datablend_nemotron_v2_stem_30pct_chat.json}"
        DEFAULT_TRAIN_SAMPLES=101175
        echo "📊 Using Nemotron-v2 STEM @ 30% + Chat Template (~101K samples)"
        ;;
    nemotron_v2_math_chat|nemotron_v2_math_30pct_chat)
        BLEND_PATH="${BLEND_PATH:-/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/datablend_nemotron_v2_math_30pct_chat.json}"
        DEFAULT_TRAIN_SAMPLES=68248
        echo "📊 Using Nemotron-v2 Math @ 30% + Chat Template (~68K samples)"
        ;;
    nemotron_v2_code_chat|nemotron_v2_code_30pct_chat)
        BLEND_PATH="${BLEND_PATH:-/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/datablend_nemotron_v2_code_30pct_chat.json}"
        DEFAULT_TRAIN_SAMPLES=99750
        echo "📊 Using Nemotron-v2 Code @ 30% + Chat Template (~50K x2 epochs)"
        ;;
    nemotron_v2_chat_chat|nemotron_v2_chat_30pct_chat)
        BLEND_PATH="${BLEND_PATH:-/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/datablend_nemotron_v2_chat_30pct_chat.json}"
        DEFAULT_TRAIN_SAMPLES=178900
        echo "📊 Using Nemotron-v2 Chat @ 30% + Chat Template (~179K samples)"
        ;;

    # ====================
    # Nemotron-v2 individual splits with COT (chain-of-thought reasoning)
    # ====================
    nemotron_v2_stem_cot|nemotron_v2_stem_30pct_cot)
        BLEND_PATH="${BLEND_PATH:-/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/datablend_nemotron_v2_stem_30pct_cot.json}"
        DEFAULT_TRAIN_SAMPLES=101175
        echo "📊 Using Nemotron-v2 STEM split @ 30% + COT (~101K samples)"
        ;;
    nemotron_v2_math_cot|nemotron_v2_math_30pct_cot)
        BLEND_PATH="${BLEND_PATH:-/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/datablend_nemotron_v2_math_30pct_cot.json}"
        DEFAULT_TRAIN_SAMPLES=68248
        echo "📊 Using Nemotron-v2 Math split @ 30% + COT (~68K samples)"
        ;;
    nemotron_v2_code_cot|nemotron_v2_code_30pct_cot)
        BLEND_PATH="${BLEND_PATH:-/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/datablend_nemotron_v2_code_30pct_cot.json}"
        DEFAULT_TRAIN_SAMPLES=49875
        echo "📊 Using Nemotron-v2 Code split @ 30% + COT (~50K samples)"
        ;;
    nemotron_v2_chat_cot|nemotron_v2_chat_30pct_cot)
        BLEND_PATH="${BLEND_PATH:-/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/datablend_nemotron_v2_chat_30pct_cot.json}"
        DEFAULT_TRAIN_SAMPLES=178900
        echo "📊 Using Nemotron-v2 Chat split @ 30% + COT (~179K samples)"
        ;;

    # ====================
    # Nemotron-v2 individual splits with COT + chat template
    # ====================
    nemotron_v2_stem_cot_chat|nemotron_v2_stem_30pct_cot_chat)
        BLEND_PATH="${BLEND_PATH:-/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/datablend_nemotron_v2_stem_30pct_cot_chat.json}"
        DEFAULT_TRAIN_SAMPLES=101175
        echo "📊 Using Nemotron-v2 STEM @ 30% + COT + Chat Template (~101K samples)"
        ;;
    nemotron_v2_math_cot_chat|nemotron_v2_math_30pct_cot_chat)
        BLEND_PATH="${BLEND_PATH:-/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/datablend_nemotron_v2_math_30pct_cot_chat.json}"
        DEFAULT_TRAIN_SAMPLES=68248
        echo "📊 Using Nemotron-v2 Math @ 30% + COT + Chat Template (~68K samples)"
        ;;
    nemotron_v2_code_cot_chat|nemotron_v2_code_30pct_cot_chat)
        BLEND_PATH="${BLEND_PATH:-/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/datablend_nemotron_v2_code_30pct_cot_chat.json}"
        DEFAULT_TRAIN_SAMPLES=49875
        echo "📊 Using Nemotron-v2 Code @ 30% + COT + Chat Template (~50K samples)"
        ;;
    nemotron_v2_chat_cot_chat|nemotron_v2_chat_30pct_cot_chat)
        BLEND_PATH="${BLEND_PATH:-/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/datablend_nemotron_v2_chat_30pct_cot_chat.json}"
        DEFAULT_TRAIN_SAMPLES=178900
        echo "📊 Using Nemotron-v2 Chat @ 30% + COT + Chat Template (~179K samples)"
        ;;

    # ====================
    # OpenScience datasets
    # ====================
    openscience)
        BLEND_PATH="${BLEND_PATH:-/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/datablend_openscience.json}"
        DEFAULT_TRAIN_SAMPLES=299800
        echo "📊 Using OpenScience Dataset (plain text)"
        ;;
    openscience_chat)
        BLEND_PATH="${BLEND_PATH:-/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/datablend_openscience_chat.json}"
        DEFAULT_TRAIN_SAMPLES=299800
        echo "📊 Using OpenScience Dataset + Chat Template"
        ;;

    # ====================
    # Combined datasets
    # ====================
    combined|combined_v1_v2_openscience)
        BLEND_PATH="${BLEND_PATH:-/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/datablend_combined_v1_v2_openscience.json}"
        DEFAULT_TRAIN_SAMPLES=10000000
        echo "📊 Using Combined Dataset: 50% Nemotron-v1 + 30% Nemotron-v2 + 20% OpenScience (~10M samples)"
        ;;
    combined_chat|combined_v1_v2_openscience_chat)
        BLEND_PATH="${BLEND_PATH:-/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/datablend_combined_v1_v2_openscience_chat.json}"
        DEFAULT_TRAIN_SAMPLES=10000000
        echo "📊 Using Combined Dataset + Chat Template (~10M samples)"
        ;;
    combined_cot_chat|combined_all_cot_chat)
        # Combined: 20% OpenScience + 50% Nemotron-v1 + 30% Nemotron-v2 (all splits)
        # All with COT reasoning + Qwen3 chat template
        # Nemotron-v2 breakdown: 7.5% stem + 7.5% math + 5% code + 10% chat = 30%
        BLEND_PATH="${BLEND_PATH:-/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/datablend_combined_cot_chat.json}"
        # Total samples: ~300K OpenScience + ~7.5M Nemotron-v1 + ~398K Nemotron-v2 ≈ 8.2M
        DEFAULT_TRAIN_SAMPLES=8200000
        echo "📊 Using Combined Dataset + COT + Chat Template (~8.2M samples)"
        echo "   - 20% OpenScience (chat)"
        echo "   - 50% Nemotron-v1 @ 30% (cot+chat)"
        echo "   - 30% Nemotron-v2 @ 30% (stem+math+code+chat, cot+chat)"
        ;;
    combined_v2|combined_v2_cot_chat)
        # Combined V2: Code & Math focused
        # All with COT reasoning + Qwen3 chat template
        BLEND_PATH="${BLEND_PATH:-/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/datablend_combined_v2_cot_chat.json}"
        DEFAULT_TRAIN_SAMPLES=1024000  # manually set to 1M samples
        echo "📊 Using Combined V2 (Code & Math focused) + COT + Chat (~8.2M samples)"
        echo "   - 20% OpenScience"
        echo "   - 40% Nemotron-v1 (10% stem, 10% math, 15% code, 5% chat)"
        echo "   - 40% Nemotron-v2 (5% stem, 10% math, 15% code, 10% chat)"
        ;;

    # ====================
    # Other datasets
    # ====================
    slimorca)
        BLEND_PATH="${BLEND_PATH:-/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/datablend_slimorca.json}"
        DEFAULT_TRAIN_SAMPLES=500000
        echo "📊 Using SlimOrca Dataset"
        ;;
    slimorca_chat)
        BLEND_PATH="${BLEND_PATH:-/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/datablend_slimorca_chat.json}"
        DEFAULT_TRAIN_SAMPLES=500000
        echo "📊 Using SlimOrca Dataset + Chat Template"
        ;;

    # ====================
    # Default fallback
    # ====================
    *)
        BLEND_PATH="${BLEND_PATH:-/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/datablend_openscience.json}"
        DEFAULT_TRAIN_SAMPLES=299800
        echo "📊 Using OpenScience Dataset (Default)"
        ;;
esac

# Allow override via environment variable
TRAIN_SAMPLES=${TRAIN_SAMPLES:-$DEFAULT_TRAIN_SAMPLES}
LR_DECAY_SAMPLES=$(python3 -c "print(int(${TRAIN_SAMPLES} * 0.99))")
LR_WARMUP_SAMPLES=$(python3 -c "print(int(${TRAIN_SAMPLES} * 0.01))")

echo "📈 Training samples configuration:"
echo "    Train samples: ${TRAIN_SAMPLES}"
echo "    LR decay samples: ${LR_DECAY_SAMPLES}"
echo "    LR warmup samples: ${LR_WARMUP_SAMPLES}"

########################################################
#### RESUME LOGIC ####
########################################################

if [ -f "${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt" ]; then
    echo "🔄 Found existing checkpoint at ${CHECKPOINT_DIR}"
    echo "   Resuming training from there..."
    LOAD_CHECKPOINT_DIR="${CHECKPOINT_DIR}"
    FINETUNE_FLAG=""
    LOAD_OPTIM_ARGS=""
    CKPT_PARALLEL_LOAD_ARG="--ckpt-fully-parallel-load"
else
    echo "🆕 No existing checkpoint found. Starting fresh from base student."
    LOAD_CHECKPOINT_DIR="${BASE_STUDENT_CKPT}"
    FINETUNE_FLAG="--finetune"
    LOAD_OPTIM_ARGS="--no-load-optim --no-load-rng"
    CKPT_PARALLEL_LOAD_ARG=""
fi

########################################################
#### CREATE DIRECTORIES ####
########################################################

mkdir -p ${LOGS_DIR}
mkdir -p ${CHECKPOINT_DIR}
mkdir -p ${DATACACHE_DIR}
mkdir -p ${TENSORBOARD_DIR}

########################################################
#### LOG ENVIRONMENT ####
########################################################

echo "========================================" | tee ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "Starting ${STUDENT_MODEL} NVFP4 QAD Training" | tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "Time: ${DATETIME}" | tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "========================================" | tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "" | tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}

echo "<< MODEL CONFIG >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "STUDENT_MODEL=${STUDENT_MODEL}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "TEACHER_MODEL=${TEACHER_MODEL}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "CONFIG_FILE=${STUDENT_CONFIG_FILE}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "IS_MOE=${IS_MOE}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "<< END MODEL CONFIG >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo -e "\n" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}

echo "<< START PATHS >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "SCRIPT_DIR=${SCRIPT_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "MLM_DIR=${MLM_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "RUN_DIR=${RUN_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "LOGS_DIR=${LOGS_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "DATACACHE_DIR=${DATACACHE_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "TENSORBOARD_DIR=${TENSORBOARD_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "LOAD_CHECKPOINT_DIR=${LOAD_CHECKPOINT_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "<< END PATHS >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo -e "\n\n" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}

echo "<< START GIT >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "GIT LOG" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
git -C ${MLM_DIR} log --oneline -1 |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo -e "\n\n" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "GIT STATUS" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
git -C ${MLM_DIR} status --porcelain --branch |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo -e "\n\n" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "GIT DIFF" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
git -C ${MLM_DIR} diff |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "<< END GIT >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo -e "\n\n" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}

echo "<< START ENV >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
env |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "<< END ENV >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}

########################################################
#### TRAINING ARGUMENTS ####
########################################################

# Iterations to skip (if any)
ITERATIONS_TO_SKIP="${ITERATIONS_TO_SKIP:-}"

# Number of GPUs to use
NUM_GPUS=${NUM_GPUS:-8}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}

# Checkpoint and Model Loading
CHECKPOINT_ARGS=" \
    --auto-detect-ckpt-format \
    --export-te-mcore-model \
    --dist-ckpt-strictness log_unexpected \
    ${FINETUNE_FLAG} \
    ${LOAD_OPTIM_ARGS} \
    --load ${LOAD_CHECKPOINT_DIR}"

# Add KD teacher args (always enabled - TEACHER_CKPT and TEACHER_MODEL_CONFIG are required)
CHECKPOINT_ARGS="${CHECKPOINT_ARGS} \
    --export-quant-cfg nvfp4 \
    --export-kd-teacher-load ${TEACHER_CKPT_DIR} \
    --teacher-model-config ${TEACHER_MODEL_CONFIG} \
    ${KD_CFG_ARGS}"

# Tokenizer Settings (from sourced config or default)
TOKENIZER_MODEL="${TOKENIZER_MODEL:-${HF_MODEL_CKPT:-Qwen/${STUDENT_MODEL}}}"
TOKENIZER_ARGS=" \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL}"

# Data Settings
DATA_ARGS=" \
    --per-split-data-args-path ${BLEND_PATH} \
    --data-cache-path ${DATACACHE_DIR} \
    --no-mmap-bin-files \
    --num-dataset-builder-threads 16 \
    --no-create-attention-mask-in-dataloader"

# Training Hyperparameters
TRAINING_ARGS=" \
    --micro-batch-size ${MBS} \
    --global-batch-size 256 \
    --train-samples ${TRAIN_SAMPLES} \
    --lr-decay-samples ${LR_DECAY_SAMPLES} \
    --lr-warmup-samples ${LR_WARMUP_SAMPLES} \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --bf16 \
    --no-masked-softmax-fusion"

# Optimizer Settings
OPTIMIZER_ARGS=" \
    --lr ${LR} \
    --min-lr 0.0 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --lr-decay-style cosine \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --overlap-param-gather"

# Parallelism Settings
# Build parallel args based on model type
PARALLEL_ARGS=" \
    --tensor-model-parallel-size ${TP_SIZE} \
    --pipeline-model-parallel-size ${PP_SIZE:-1} \
    --distributed-timeout-minutes 360 \
    --disable-gloo-process-groups \
    --ddp-num-buckets 7"

# Add expert parallelism for MoE models
if [ "$IS_MOE" = "true" ] && [ "$EP_SIZE" -gt 1 ]; then
    PARALLEL_ARGS="${PARALLEL_ARGS} \
    --expert-model-parallel-size ${EP_SIZE}"
    echo "🔧 MoE Expert Parallelism: EP=${EP_SIZE}"
fi

# Add sequence parallel if supported (check if it's in MODEL_ARGS)
if echo "$STUDENT_MODEL_ARGS" | grep -q "sequence-parallel"; then
    echo "🔧 Sequence Parallel: enabled (from model config)"
else
    PARALLEL_ARGS="${PARALLEL_ARGS} --sequence-parallel"
fi

# Memory Optimization
MEMORY_ARGS=" \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1 \
    --no-gradient-accumulation-fusion"

# Checkpoint Saving
SAVE_ARGS=" \
    --save ${CHECKPOINT_DIR} \
    --save-interval 200 \
    --save-retain-interval 200 \
    --ckpt-format torch_dist \
    --ckpt-fully-parallel-save \
    --ckpt-assume-constant-structure \
    --exit-duration-in-mins 230 \
    ${CKPT_PARALLEL_LOAD_ARG}"

# Logging and Monitoring
LOGGING_ARGS=" \
    --log-interval 10 \
    --eval-iters 20 \
    --eval-interval 200 \
    --log-progress \
    --timing-log-option minmax \
    ${LOG_PARAMS_NORM_ARG:-} \
    --log-num-zeros-in-grad \
    --log-throughput \
    --log-straggler \
    --disable-straggler-on-startup \
    --straggler-minmax-count 16 \
    --tensorboard-dir ${TENSORBOARD_DIR}"

# Runtime Settings
RUNTIME_ARGS=" \
    --exit-duration-in-mins 1200 \
    --num-workers 8 \
    --no-check-for-nan-in-loss-and-grad"

# Combine all arguments
# NOTE: Argument order matters! Later args override earlier ones (argparse behavior)
# 
# Order explanation:
#   1. CHECKPOINT_ARGS   - Loading/saving config
#   2. STUDENT_MODEL_ARGS - From conf file (may contain --micro-batch-size, --bf16, --save-interval, etc.)
#   3. TOKENIZER_ARGS    - Overrides --tokenizer-type from conf file
#   4. DATA_ARGS         - Dataset configuration
#   5. TRAINING_ARGS     - Overrides --micro-batch-size, --bf16 from conf file
#   6. OPTIMIZER_ARGS    - Learning rate, optimizer settings
#   7. PARALLEL_ARGS     - TP/PP/EP settings
#   8. MEMORY_ARGS       - Recompute settings
#   9. SAVE_ARGS         - Overrides --save-interval from conf file
#   10. LOGGING_ARGS     - Logging configuration
#   11. RUNTIME_ARGS     - Runtime settings
#
# This allows conf files to set defaults that QAD script can override
ALL_ARGS=" \
    ${CHECKPOINT_ARGS} \
    ${STUDENT_MODEL_ARGS} \
    ${TOKENIZER_ARGS} \
    ${DATA_ARGS} \
    ${TRAINING_ARGS} \
    ${OPTIMIZER_ARGS} \
    ${PARALLEL_ARGS} \
    ${MEMORY_ARGS} \
    ${SAVE_ARGS} \
    ${LOGGING_ARGS} \
    ${RUNTIME_ARGS}"

if [ -n "${ITERATIONS_TO_SKIP}" ]; then
    ALL_ARGS="${ALL_ARGS} --iterations-to-skip ${ITERATIONS_TO_SKIP}"
fi

# Update PYTHONPATH
export PYTHONPATH="${MODELOPT_DIR}:${MLM_DIR}:${PYTHONPATH:-}"

########################################################
#### LAUNCH TRAINING ####
########################################################

echo "========================================" | tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "Running training command..." | tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "========================================" | tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}

LOG_FILE="${LOGS_DIR}/${MODEL_SHORT_NAME}_qad_${DATETIME}.log"

echo "Output will be written to: ${LOG_FILE}" | tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "" | tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}

# Multi-node configuration
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}

echo "<< DISTRIBUTED CONFIG >>" | tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "  NNODES: ${NNODES}" | tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "  NODE_RANK: ${NODE_RANK}" | tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "  NUM_GPUS per node: ${NUM_GPUS}" | tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "  MASTER_ADDR: ${MASTER_ADDR}" | tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "  MASTER_PORT: ${MASTER_PORT}" | tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "  TP: ${TP_SIZE}, PP: ${PP_SIZE:-1}, EP: ${EP_SIZE}" | tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "  MBS: ${MBS}" | tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "  Total GPUs: $((NNODES * NUM_GPUS))" | tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "<< END DISTRIBUTED CONFIG >>" | tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}

# Launch training
torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    ${MLM_DIR}/pretrain_gpt.py ${ALL_ARGS} 2>&1 | tee ${LOG_FILE}

echo "" | tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "========================================" | tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "Training completed or exited" | tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "Check logs at: ${LOG_FILE}" | tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "========================================" | tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
