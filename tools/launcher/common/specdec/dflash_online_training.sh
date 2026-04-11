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

# DFlash online training script for the ModelOpt Launcher.
# Trains a DFlash draft model using accelerate launch + main.py --config.
#
# All training config comes from the YAML recipe (--config) and OmegaConf overrides.
# All args are passed directly to main.py (--config + key=value overrides).
#
# Multi-node env vars (set by Slurm or user):
#   NUM_NODES       — number of nodes (default: 1)
#   HEAD_NODE_IP    — head node IP (auto-detected if not set)
#
# Usage from YAML:
#   script: common/dflash/online_training.sh
#   args:
#     - --config modules/Model-Optimizer/modelopt_recipes/general/speculative_decoding/dflash.yaml
#     - model.model_name_or_path=/hf-local/Qwen/Qwen3-8B
#     - data.data_path=/path/to/data.jsonl
#     - training.output_dir=/scratchspace/dflash
#   environment:
#     - NUM_NODES: "8"

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source ${SCRIPT_DIR}/../service_utils.sh

pip install -r modules/Model-Optimizer/examples/speculative_decoding/requirements.txt
pip install huggingface-hub>=1.2.1
export PATH=$PATH:/workspace/.local/bin

###################################################################################################

trap 'error_handler $0 $LINENO' ERR

# Auto-detect head node IP for multi-node training
NUM_NODES=${NUM_NODES:-1}
if [ -z "$HEAD_NODE_IP" ] && [[ "$NUM_NODES" != "1" ]]; then
    HEAD_NODE_IP=$(scontrol show hostnames "$SLURM_JOB_NODELIST" 2>/dev/null | head -1)
    HEAD_NODE_IP=${HEAD_NODE_IP:-$SLURM_LAUNCH_NODE_IPADDR}
    if [ -z "$HEAD_NODE_IP" ] && [ -n "$SLURM_JOB_NODELIST" ]; then
        HEAD_NODE_IP=$(python3 -c "
import socket, re, os
nl = os.environ.get('SLURM_JOB_NODELIST', '')
m = re.match(r'([a-zA-Z0-9-]+?)(?:\[(\d+))?', nl)
if m:
    host = m.group(1) + (m.group(2) or '')
    try:
        print(socket.gethostbyname(host))
    except:
        print(host)
" 2>/dev/null)
    fi
    if [ -z "$HEAD_NODE_IP" ] && [ "${SLURM_PROCID:-0}" = "0" ]; then
        HEAD_NODE_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
    fi
    export HEAD_NODE_IP
    echo "Auto-detected HEAD_NODE_IP: ${HEAD_NODE_IP}"
fi

# Build accelerate launch command
MAIN_PY=modules/Model-Optimizer/examples/speculative_decoding/main.py

if [[ "$NUM_NODES" != "1" ]]; then
    if [ -z "$HEAD_NODE_IP" ]; then
        echo "ERROR: HEAD_NODE_IP is empty. Cannot launch multi-node training."
        exit 1
    fi
    GPU_PER_NODE=${GPU_PER_NODE:-$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)}
    TOTAL_GPU=$((NUM_NODES * GPU_PER_NODE))
    echo "Total GPUs: $TOTAL_GPU (NUM_NODES: $NUM_NODES, GPU_PER_NODE: $GPU_PER_NODE)"
    MULTI_NODE_ARGS="--num_processes $TOTAL_GPU \
                     --num_machines $NUM_NODES \
                     --machine_rank $SLURM_PROCID \
                     --rdzv_backend c10d \
                     --main_process_ip $HEAD_NODE_IP \
                     --main_process_port 29500"
else
    TOTAL_GPU=$(python3 -c "import torch; print(torch.cuda.device_count())")
    echo "Total GPUs: $TOTAL_GPU (single node)"
    MULTI_NODE_ARGS=""
fi

export TOKENIZERS_PARALLELISM=False

set -x
start_time=$(date +%s)
accelerate launch --mixed_precision bf16 $MULTI_NODE_ARGS $MAIN_PY "$@"
echo "Training time: $(( $(date +%s) - start_time )) seconds"
set +x

# Export last checkpoint to deployment format (rank 0 only, single GPU)
if [ "${SLURM_PROCID:-0}" = "0" ]; then
    OUTPUT_DIR=$(python3 -c "
import sys
for arg in sys.argv[1:]:
    if arg.startswith('training.output_dir='):
        print(arg.split('=', 1)[1])
        break
" "$@")

    if [ -n "$OUTPUT_DIR" ]; then
        # Find the best checkpoint: last checkpoint-* subdir, or output_dir itself
        LAST_CKPT=$(ls -d ${OUTPUT_DIR}/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
        if [ -n "$LAST_CKPT" ]; then
            STEP=$(basename "$LAST_CKPT" | sed 's/checkpoint-//')
        elif [ -f "${OUTPUT_DIR}/model.safetensors" ] || [ -f "${OUTPUT_DIR}/modelopt_state.pth" ]; then
            LAST_CKPT="${OUTPUT_DIR}"
            STEP="final"
        fi

        if [ -n "$LAST_CKPT" ]; then
            EXPORT_DIR="${OUTPUT_DIR}/exported-checkpoint-${STEP}"
            echo "=== Exporting: ${LAST_CKPT} → ${EXPORT_DIR} ==="
            CUDA_VISIBLE_DEVICES=0 python3 modules/Model-Optimizer/examples/speculative_decoding/scripts/export_hf_checkpoint.py \
                --model_path "${LAST_CKPT}" \
                --export_path "${EXPORT_DIR}" \
                --trust_remote_code
            echo "Export contents:"
            ls -lh "${EXPORT_DIR}/"
        else
            echo "No checkpoints found in ${OUTPUT_DIR}, skipping export"
        fi
    fi
fi
