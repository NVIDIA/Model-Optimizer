#!/bin/bash
# =============================================================================
# QAD SLURM Batch Submission Script
# =============================================================================
#
# Override these SLURM settings via command line:
#   sbatch --nodes=4 --account=<your-account> sbatch_qad.sh --config ...
#
# Or set defaults below for your cluster:

#SBATCH -p batch
#SBATCH --account=<your-account>
#SBATCH --nodes=4
#SBATCH -t 4:00:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=qad-training

# Usage:
#   sbatch sbatch_qad.sh --config configs/qwen3-8b.conf
#   sbatch sbatch_qad.sh --config configs/qwen3-30b-a3b-instruct-2507-moe.conf
#
# With HuggingFace token:
#   sbatch sbatch_qad.sh --hf-token hf_xxx --config configs/qwen3-8b.conf
#

set -x -e

########################################################
# Parse Arguments
########################################################

# Use SLURM_SUBMIT_DIR if available (SLURM copies script to temp location)
# Otherwise use the script's directory
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
CONFIG_FILE=""
HF_TOKEN_ARG=""

# Parse arguments
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
            break
            ;;
    esac
done

# Set HF_TOKEN from arg (takes precedence, doesn't appear in logs)
if [ -n "$HF_TOKEN_ARG" ]; then
    export HF_TOKEN="$HF_TOKEN_ARG"
fi

########################################################
# Load Config File
########################################################

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
# Default Values (only if not set by config/env)
########################################################

# Training args (command line can override)
# Order: LR, TEACHER_MODEL, DATASET_NAME, STUDENT_MODEL, KD_CFG_PATH
LR="${1:-${LR:-1e-6}}"
TEACHER_MODEL="${2:-${TEACHER_MODEL:-Qwen3-8B}}"
DATASET_NAME="${3:-${DATASET_NAME:-openscience}}"
STUDENT_MODEL="${4:-${STUDENT_MODEL:-Qwen3-8B}}"
KD_CFG_PATH="${5:-${KD_CFG_PATH:-}}"

# Paths
MLM_DIR="${MLM_DIR:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_modelopt/users/weimingc/workspace/Megatron-LM}"
MODELOPT_DIR="${MODELOPT_DIR:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_modelopt/users/weimingc/workspace/TensorRT-Model-Optimizer}"
MODELS_ROOT="${MODELS_ROOT:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_modelopt/users/weimingc/models}"
QAD_CHECKPOINT_ROOT="${QAD_CHECKPOINT_ROOT:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_modelopt/users/weimingc/checkpoints}"
DATACACHE_DIR="${DATACACHE_DIR:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_modelopt/users/weimingc/data_cache}"
LOG_DIR="${LOG_DIR:-${QAD_CHECKPOINT_ROOT}/logs_slurm}"

# Container
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_modelopt/users/weimingc/containers/pytorch_25.06-py3.sqsh}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-/lustre/fs1:/lustre/fs1}"
CONTAINER_WORKDIR="${CONTAINER_WORKDIR:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_modelopt/users/weimingc/workspace/TensorRT-Model-Optimizer/examples/llm_qad}"

# Parallelism settings (from config, required)
TP_SIZE="${TP_SIZE:?ERROR: TP_SIZE must be set in config}"
PP_SIZE="${PP_SIZE:-1}"
EP_SIZE="${EP_SIZE:-1}"
MBS="${MBS:?ERROR: MBS must be set in config}"

# Other settings
NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29500}"

# Multi-node config from SLURM (passed via sbatch --nodes=N)
NNODES="${SLURM_NNODES:-4}"
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

# Create directories
mkdir -p ${LOG_DIR}

DATETIME=$(date +'date_%y-%m-%d_time_%H-%M-%S')

########################################################
# Display Configuration
########################################################

echo "========================================"
echo "QAD Training Configuration"
echo "========================================"
if [ -n "$CONFIG_FILE" ]; then
    echo "CONFIG_FILE: ${CONFIG_FILE}"
fi
echo ""
echo "Model:"
echo "  STUDENT_MODEL: ${STUDENT_MODEL}"
echo "  TEACHER_MODEL: ${TEACHER_MODEL}"
echo ""
echo "Training:"
echo "  LR: ${LR}"
echo "  DATASET: ${DATASET_NAME}"
echo "  KD_CFG_PATH: ${KD_CFG_PATH:-none}"
echo ""
echo "Parallelism:"
echo "  TP: ${TP_SIZE}, PP: ${PP_SIZE}, EP: ${EP_SIZE}"
echo "  MBS: ${MBS}"
echo "  NNODES: ${NNODES}"
echo "  NUM_GPUS/node: ${NUM_GPUS}"
echo "  Total GPUs: $((NNODES * NUM_GPUS))"
echo ""
echo "Distributed:"
echo "  MASTER_ADDR: ${MASTER_ADDR}"
echo "  MASTER_PORT: ${MASTER_PORT}"
echo "  SLURM_NODELIST: ${SLURM_JOB_NODELIST}"
echo ""
echo "Paths:"
echo "  MLM_DIR: ${MLM_DIR}"
echo "  MODELOPT_DIR: ${MODELOPT_DIR}"
echo "  MODELS_ROOT: ${MODELS_ROOT}"
echo "  QAD_CHECKPOINT_ROOT: ${QAD_CHECKPOINT_ROOT}"
echo "  DATACACHE_DIR: ${DATACACHE_DIR}"
echo "  LOG_DIR: ${LOG_DIR}"
echo ""
echo "Container:"
echo "  IMAGE: ${CONTAINER_IMAGE}"
echo "  WORKDIR: ${CONTAINER_WORKDIR}"

# Show checkpoint paths
echo ""
echo "Checkpoints:"
echo "  STUDENT_CKPT: ${STUDENT_CKPT:-NOT SET}"
echo "  TEACHER_CKPT: ${TEACHER_CKPT:-NOT SET}"
if [ -n "${TEACHER_MODEL_CONFIG:-}" ]; then
    echo "  TEACHER_MODEL_CONFIG: ${TEACHER_MODEL_CONFIG}"
fi
if [ -n "${BLEND_PATH:-}" ]; then
    echo "  BLEND_PATH: ${BLEND_PATH}"
fi
echo "========================================"

# Validate required checkpoints
if [ -z "${STUDENT_CKPT:-}" ]; then
    echo "❌ ERROR: STUDENT_CKPT is required. Set it in config file."
    exit 1
fi
if [ -z "${TEACHER_CKPT:-}" ]; then
    echo "❌ ERROR: TEACHER_CKPT is required. Set it in config file."
    exit 1
fi

########################################################
# Build Container Environment Exports
########################################################

# Core exports (environment variables that qwen_qad.sh will read)
# Use local /tmp for Triton cache to avoid race conditions on shared filesystem
EXPORTS="export TRITON_CACHE_DIR=/tmp/triton_cache_\${SLURM_JOB_ID}_\${SLURM_PROCID} && \
export NODE_RANK=\${SLURM_PROCID} && \
export NNODES=${NNODES} && \
export NUM_GPUS=${NUM_GPUS} && \
export TP_SIZE=${TP_SIZE} && \
export PP_SIZE=${PP_SIZE} && \
export EP_SIZE=${EP_SIZE} && \
export MBS=${MBS} && \
export IS_MOE=${IS_MOE:-false} && \
export MASTER_ADDR=${MASTER_ADDR} && \
export MASTER_PORT=${MASTER_PORT} && \
export MLM_DIR=${MLM_DIR} && \
export MODELOPT_DIR=${MODELOPT_DIR} && \
export QAD_CHECKPOINT_ROOT=${QAD_CHECKPOINT_ROOT} && \
export DATACACHE_DIR=${DATACACHE_DIR}"

# Training hyperparameters (required by qwen_qad.sh)
EXPORTS="${EXPORTS} && export LR=${LR:-}"
EXPORTS="${EXPORTS} && export GBS=${GBS:-}"
EXPORTS="${EXPORTS} && export MIN_LR=${MIN_LR:-}"
EXPORTS="${EXPORTS} && export LR_DECAY_STYLE=${LR_DECAY_STYLE:-}"
EXPORTS="${EXPORTS} && export SAVE_INTERVAL=${SAVE_INTERVAL:-}"
EXPORTS="${EXPORTS} && export LOG_INTERVAL=${LOG_INTERVAL:-}"
EXPORTS="${EXPORTS} && export STUDENT_MODEL=${STUDENT_MODEL:-}"
EXPORTS="${EXPORTS} && export TEACHER_MODEL=${TEACHER_MODEL:-}"
EXPORTS="${EXPORTS} && export DATASET_NAME=${DATASET_NAME:-}"

# Student config file (required)
if [ -n "${STUDENT_CONFIG_FILE:-}" ]; then
    EXPORTS="${EXPORTS} && export STUDENT_CONFIG_FILE=${STUDENT_CONFIG_FILE}"
fi

# Tokenizer model (optional - defaults to Qwen/${STUDENT_MODEL} in qad.sh)
if [ -n "${TOKENIZER_MODEL:-}" ]; then
    EXPORTS="${EXPORTS} && export TOKENIZER_MODEL=${TOKENIZER_MODEL}"
fi

# Checkpoint exports (required)
EXPORTS="${EXPORTS} && export STUDENT_CKPT=${STUDENT_CKPT}"
EXPORTS="${EXPORTS} && export TEACHER_CKPT=${TEACHER_CKPT}"
if [ -n "${TEACHER_MODEL_CONFIG:-}" ]; then
    EXPORTS="${EXPORTS} && export TEACHER_MODEL_CONFIG=${TEACHER_MODEL_CONFIG}"
fi

# Optional dataset exports
if [ -n "${BLEND_PATH:-}" ]; then
    EXPORTS="${EXPORTS} && export BLEND_PATH=${BLEND_PATH}"
fi
if [ -n "${TRAIN_SAMPLES:-}" ]; then
    EXPORTS="${EXPORTS} && export TRAIN_SAMPLES=${TRAIN_SAMPLES}"
fi

# HuggingFace token (avoid rate limiting)
if [ -n "${HF_TOKEN:-}" ]; then
    EXPORTS="${EXPORTS} && export HF_TOKEN=${HF_TOKEN}"
    EXPORTS="${EXPORTS} && export HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}"
fi
if [ -n "${ITERATIONS_TO_SKIP:-}" ]; then
    EXPORTS="${EXPORTS} && export ITERATIONS_TO_SKIP=${ITERATIONS_TO_SKIP}"
fi

# Optional KD config
if [ -n "${DISTILL_CONFIG_PATH:-}" ]; then
    EXPORTS="${EXPORTS} && export DISTILL_CONFIG_PATH=${DISTILL_CONFIG_PATH}"
fi

########################################################
# Launch Training
########################################################

SCRIPT_NAME="qad.sh"

# Build config args for qwen_qad.sh
CONFIG_ARGS=""
if [ -n "${CONFIG_FILE}" ]; then
    CONFIG_ARGS="--config ${CONFIG_FILE}"
fi
if [ -n "${HF_TOKEN:-}" ]; then
    CONFIG_ARGS="${CONFIG_ARGS} --hf-token ${HF_TOKEN}"
fi

run_cmd="pip install transformers==4.54 && \
${EXPORTS} && \
cd ${CONTAINER_WORKDIR} && \
bash ${SCRIPT_NAME} ${CONFIG_ARGS}"

echo ""
echo "Running command:"
echo "${run_cmd}"
echo ""

srun -l \
    --output=${LOG_DIR}/%x_%j_${DATETIME}.log \
    --error=${LOG_DIR}/err_%x_%j_${DATETIME}.log \
    --container-image ${CONTAINER_IMAGE} \
    --container-mounts ${CONTAINER_MOUNTS} \
    --container-workdir ${CONTAINER_WORKDIR} \
    sh -c "${run_cmd}"

echo ""
echo "========================================"
echo "QAD Training completed at $(date)"
echo "Logs: ${LOG_DIR}/"
echo "========================================"
