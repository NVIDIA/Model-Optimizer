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

# Example SLURM launch script for QAD (Quantization-Aware Distillation) training.
#
# Supports both LTX-2 and Wan models. The script handles container setup,
# multi-node distributed training via accelerate, and SLURM configuration.
#
# Before running, fill in the required parameters in the "Default Config" section below:
#   - PARTITION / ACCOUNT: Your SLURM cluster partition and account
#   - ROOT_PATH:           Working directory containing Model-Optimizer and models
#   - CONTAINER:           Path to your container image (.sqsh / .sif)
#   - MOUNTS:              Filesystem mount spec for the container
#
# Usage:
#   ./slurm_example.sh --model ltx2 --model-path /path/to/ltx-2-19b-dev.safetensors \
#       --text-encoder-path /path/to/gemma-3-12b-it [options] [config_overrides...]
#   ./slurm_example.sh --model wan --model-path /path/to/Wan2.2-TI2V-5B [options] [config_overrides...]
#
# Examples:
#   # Wan multi-node
#   ./slurm_example.sh --model wan --model-path /path/to/Wan2.2-TI2V-5B --nodes 2
#
#   # With config overrides
#   ./slurm_example.sh --model ltx2 --model-path /path/to/checkpoint \
#       optimization.steps=5000 optimization.learning_rate=1e-5
#
#   # Dry run (print sbatch script without submitting)
#   ./slurm_example.sh --model ltx2 --model-path /path/to/checkpoint --dry-run

set -eo pipefail

######################
### Default Config ###
######################

# --- SLURM settings (modify for your cluster) ---
JOB_NAME=""
PARTITION="your-partition"
ACCOUNT="your-account"
TIME_LIMIT="04:00:00"

# --- Cluster resources ---
NUM_NODES=1
GPUS_PER_NODE=8

# --- Paths (modify for your environment) ---
# Root working directory
ROOT_PATH="/path/to/workdir"
# Model-Optimizer source tree (mounted into the container)
MODELOPT_PATH="$ROOT_PATH/Model-Optimizer"
DISTILL_PATH="$MODELOPT_PATH/examples/diffusers/train"
# Container image
CONTAINER="$ROOT_PATH/container.sqsh"
# Container mount spec (adjust for your filesystem)
MOUNTS="$MODELOPT_PATH:/opt/modelopt"

# --- Model selection ---
MODEL=""              # "ltx2" or "wan"  (required)
MODEL_PATH=""         # Path to model checkpoint
TEXT_ENCODER_PATH=""  # Path to text encoder (Gemma for LTX-2; not needed for Wan)

# --- Output ---
OUTPUT_DIR=""

# --- Training config & overrides ---
CONFIG=""
OVERRIDES=""

# --- Flags ---
DRY_RUN=false

######################
### Parse Args     ###
######################

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)            MODEL="$2";              shift 2 ;;
        --model-path)       MODEL_PATH="$2";         shift 2 ;;
        --text-encoder-path) TEXT_ENCODER_PATH="$2"; shift 2 ;;
        --job-name)         JOB_NAME="$2";           shift 2 ;;
        --partition)        PARTITION="$2";           shift 2 ;;
        --account)          ACCOUNT="$2";             shift 2 ;;
        --time)             TIME_LIMIT="$2";         shift 2 ;;
        --nodes)            NUM_NODES="$2";          shift 2 ;;
        --gpus)             GPUS_PER_NODE="$2";      shift 2 ;;
        --config)           CONFIG="$2";             shift 2 ;;
        --container)        CONTAINER="$2";          shift 2 ;;
        --output-dir)       OUTPUT_DIR="$2";         shift 2 ;;
        --dry-run)          DRY_RUN=true;            shift ;;
        --help|-h)
            echo "Usage: $0 --model <ltx2|wan> [options] [config_overrides...]"
            echo ""
            echo "Required:"
            echo "  --model MODEL           Model backend: 'ltx2' or 'wan'"
            echo "  --model-path PATH       Path to model checkpoint"
            echo ""
            echo "Options:"
            echo "  --text-encoder-path PATH  Text encoder path (required for LTX-2)"
            echo "  --job-name NAME         SLURM job name"
            echo "  --partition PART        SLURM partition"
            echo "  --account ACCT          SLURM account"
            echo "  --time TIME             Time limit HH:MM:SS (default: $TIME_LIMIT)"
            echo "  --nodes N               Number of nodes (default: $NUM_NODES)"
            echo "  --gpus N                GPUs per node (default: $GPUS_PER_NODE)"
            echo "  --config FILE           Training config YAML (auto-detected from model)"
            echo "  --container FILE        Container image"
            echo "  --output-dir DIR        Output directory"
            echo "  --dry-run               Print sbatch script without submitting"
            echo ""
            echo "Config overrides (passed as OmegaConf dotlist):"
            echo "  optimization.steps=5000"
            echo "  optimization.learning_rate=1e-5"
            echo "  distillation.distillation_alpha=0.5"
            exit 0
            ;;
        *)
            # Collect remaining args as config overrides
            OVERRIDES="$OVERRIDES $1"
            shift
            ;;
    esac
done

######################
### Validate       ###
######################

if [[ -z "$MODEL" ]]; then
    echo "Error: --model is required (ltx2 or wan)"
    echo "Run '$0 --help' for usage."
    exit 1
fi

if [[ -z "$MODEL_PATH" ]]; then
    echo "Error: --model-path is required"
    exit 1
fi

# Set model-specific defaults
case "$MODEL" in
    ltx2)
        CONFIG="${CONFIG:-$DISTILL_PATH/configs/ltx2_distillation.yaml}"
        ACCELERATE_CONFIG="$DISTILL_PATH/configs/accelerate/fsdp_ltx2.yaml"
        JOB_NAME="${JOB_NAME:-ltx2-qad}"
        INSTALL_CMD=""
        # LTX-2 needs text encoder path
        if [[ -n "$TEXT_ENCODER_PATH" ]]; then
            MODEL_OVERRIDES="model.model_path=$MODEL_PATH model.text_encoder_path=$TEXT_ENCODER_PATH"
        else
            MODEL_OVERRIDES="model.model_path=$MODEL_PATH"
        fi
        ;;
    wan)
        CONFIG="${CONFIG:-$DISTILL_PATH/configs/wan_distillation.yaml}"
        ACCELERATE_CONFIG="$DISTILL_PATH/configs/accelerate/fsdp_wan.yaml"
        JOB_NAME="${JOB_NAME:-wan-qad}"
        # Wan requires extra system packages and pip dependencies
        INSTALL_CMD="echo 'Installing Wan dependencies...'
apt-get update -qq && apt-get install -y -qq libgl1 > /dev/null 2>&1
pip install -e '${DISTILL_PATH}[wan]' --quiet 2>&1 | tail -3
echo 'Install done.'"
        MODEL_OVERRIDES="model.model_path=$MODEL_PATH"
        ;;
    *)
        echo "Error: Unknown model '$MODEL'. Use 'ltx2' or 'wan'."
        exit 1
        ;;
esac

if [[ ! -f "$CONFIG" ]]; then
    echo "Error: Config file not found: $CONFIG"
    exit 1
fi

if [[ ! -f "$CONTAINER" ]] && [[ "$DRY_RUN" == "false" ]]; then
    echo "Warning: Container not found: $CONTAINER"
fi

######################
### Setup          ###
######################

OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_PATH/results/$JOB_NAME}"
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))

TRAINER_SCRIPT="$DISTILL_PATH/train.py"
TRAINER_CWD="$DISTILL_PATH"

######################
### Build Command  ###
######################

# Part 1: Environment setup (evaluated at runtime inside the container)
read -r -d '' CONTAINER_CMD <<'CMDEOF' || true
set -eo pipefail

export TOKENIZERS_PARALLELISM=false

echo "=== Node $(hostname) | SLURM_NODEID=$SLURM_NODEID ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv

MACHINE_RANK=${SLURM_NODEID:-0}
CMDEOF

# Part 2: Optional install step + accelerate launch (variables expanded now)
CONTAINER_CMD="$CONTAINER_CMD
$INSTALL_CMD
cd $TRAINER_CWD

accelerate launch \\
    --config_file $ACCELERATE_CONFIG \\
    --num_processes $TOTAL_GPUS \\
    --num_machines $NUM_NODES \\
    --machine_rank \$MACHINE_RANK \\
    --rdzv_backend c10d \\
    --main_process_ip MASTER_IP_PLACEHOLDER \\
    --main_process_port 29500 \\
    $TRAINER_SCRIPT \\
    --config $CONFIG \\
    $MODEL_OVERRIDES \\
    output_dir=$OUTPUT_DIR \\
    $OVERRIDES
"

######################
### Print Info     ###
######################

echo "========================================"
echo "QAD Training ($MODEL) - SLURM Launch"
echo "========================================"
echo "Job Name:        $JOB_NAME"
echo "Partition:       $PARTITION"
echo "Account:         $ACCOUNT"
echo "Time Limit:      $TIME_LIMIT"
echo "Nodes:           $NUM_NODES"
echo "GPUs/Node:       $GPUS_PER_NODE"
echo "Total GPUs:      $TOTAL_GPUS"
echo "Model:           $MODEL"
echo "Model Path:      $MODEL_PATH"
echo "Config:          $CONFIG"
echo "Output Dir:      $OUTPUT_DIR"
echo "Overrides:       $OVERRIDES"
echo "========================================"
echo ""

######################
### Build Script   ###
######################

read -r -d '' SBATCH_SCRIPT <<EOF || true
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --partition=$PARTITION
#SBATCH --account=$ACCOUNT
#SBATCH --time=$TIME_LIMIT
#SBATCH --nodes=$NUM_NODES
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:$GPUS_PER_NODE
#SBATCH --exclusive
#SBATCH --dependency=singleton

set -eo pipefail

# Get master node IP for multi-node rendezvous
head_node=\$(scontrol show hostnames \$SLURM_JOB_NODELIST | head -n 1)
head_node_ip=\$(srun --nodes=1 --ntasks=1 -w "\$head_node" hostname --ip-address | head -n 1)
echo "Master node: \$head_node (\$head_node_ip)"
echo "SLURM_JOB_NODELIST: \$SLURM_JOB_NODELIST"

# Embed the container command and replace the master IP placeholder at runtime
read -r -d '' CONTAINER_CMD <<'INNEREOF' || true
$CONTAINER_CMD
INNEREOF
CONTAINER_CMD=\${CONTAINER_CMD//MASTER_IP_PLACEHOLDER/\$head_node_ip}

# Launch on all allocated nodes
srun --nodes=\$SLURM_NNODES --ntasks-per-node=1 \\
    --output="$LOG_DIR/slurm-%j-%t.out" \\
    --error="$LOG_DIR/slurm-%j-%t.err" \\
    --no-container-remap-root --no-container-mount-home \\
    --container-image="$CONTAINER" --container-mounts="$MOUNTS" \\
    bash -c "\$CONTAINER_CMD"
EOF

######################
### Submit Job     ###
######################

if [[ "$DRY_RUN" == "true" ]]; then
    echo "=== DRY RUN - Would submit: ==="
    echo ""
    echo "$SBATCH_SCRIPT"
    echo ""
else
    echo "Submitting job..."
    echo "$SBATCH_SCRIPT" | sbatch
    echo ""
    echo "Job submitted! Logs: $LOG_DIR/"
fi
