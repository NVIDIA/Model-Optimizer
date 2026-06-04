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

# Usage:
#   Single GPU:   ./launch_train.sh --config ../../modelopt_recipes/general/speculative_decoding/eagle3.yaml model.model_name_or_path=xxx
#   Multi-node:   ./launch_train.sh --config ../../modelopt_recipes/general/speculative_decoding/eagle3.yaml --num_nodes 2 --head_node_ip <IP>
#   With overrides: ./launch_train.sh --config my.yaml model.model_name_or_path=xxx training.output_dir=yyy
#
# Extra key=value args are forwarded as OmegaConf dotlist overrides to main.py.
# All training config (model, data, hyperparams, eagle, fsdp) lives in the YAML file.
# Only multi-node routing args are passed here; mixed_precision is fixed to bf16.

set -eo pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

CONFIG_FILE=""
NUM_NODES=1
HEAD_NODE_IP=""
MACHINE_RANK=""
EXTRA_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --config*)     if [[ "$1" != *=* ]]; then shift; fi; CONFIG_FILE="${1#*=}" ;;
    --num_nodes*)  if [[ "$1" != *=* ]]; then shift; fi; NUM_NODES="${1#*=}" ;;
    --head_node_ip*) if [[ "$1" != *=* ]]; then shift; fi; HEAD_NODE_IP="${1#*=}" ;;
    --machine_rank*) if [[ "$1" != *=* ]]; then shift; fi; MACHINE_RANK="${1#*=}" ;;
    *) EXTRA_ARGS+=("$1") ;;
  esac
  shift
done

if [ -z "$CONFIG_FILE" ]; then
  >&2 echo "Usage: ./launch_train.sh --config <yaml_file> [--num_nodes N] [--head_node_ip IP] [key=value ...]"
  exit 1
fi

# GPU count detection
if [[ "$NUM_NODES" != "1" ]]; then
  GPU_PER_NODE=${GPU_PER_NODE:-$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)}
  TOTAL_GPU=$((NUM_NODES * GPU_PER_NODE))
  echo "Total GPUs: $TOTAL_GPU (NUM_NODES: $NUM_NODES, GPU_PER_NODE: $GPU_PER_NODE)"
else
  TOTAL_GPU=$(python3 -c "import torch; print(torch.cuda.device_count())")
  echo "Total GPUs: $TOTAL_GPU (single node)"
fi

# Multi-node routing args (accelerate only; training config comes from the YAML)
MULTI_NODE_ARGS=()
if [[ "$NUM_NODES" != "1" ]]; then
  # machine_rank defaults to $SLURM_PROCID; pass --machine_rank explicitly when the
  # allocation reserves node 0 for something else (e.g. a streaming vllm serve).
  # --multi_gpu is required even at 1 GPU/node -- without it accelerate treats a lone
  # local process as non-distributed and never forms the process group (each node
  # would train its own world=1). Use static rendezvous via main_process_ip/port; NOT
  # --rdzv_backend c10d, which switches to the elastic launcher and ignores it.
  MULTI_NODE_ARGS=(
    --multi_gpu
    --num_processes "$TOTAL_GPU"
    --num_machines "$NUM_NODES"
    --machine_rank "${MACHINE_RANK:-$SLURM_PROCID}"
    --main_process_ip "$HEAD_NODE_IP"
    --main_process_port 29500
  )
fi

export TOKENIZERS_PARALLELISM=False

# Run as an argv array (not `sh -c "..."`, which would word-split overrides
# containing spaces and execute command substitutions embedded in their values).
CMD=(accelerate launch --mixed_precision bf16
     "${MULTI_NODE_ARGS[@]}"
     "${SCRIPT_DIR}/main.py" --config "$CONFIG_FILE" "${EXTRA_ARGS[@]}")

set -x
start_time=$(date +%s)
"${CMD[@]}"
echo "Total time: $(( $(date +%s) - $start_time )) seconds"
