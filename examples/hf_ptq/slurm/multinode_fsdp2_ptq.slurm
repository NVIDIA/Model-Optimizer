#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#
# Multi-node post-training quantization with FSDP2, ready to run under Slurm.
#
# Slurm allocates the nodes and launches one task per node; each task starts a `torchrun` that
# spawns one process per GPU. `hf_ptq.py --use_fsdp2` then shards the model across all ranks
# (world_size = num_nodes * gpus_per_node) for distributed loading, calibration, and export.
#
# The defaults below quantize Nemotron-3-Super to NVFP4 with an FP8 KV cache. Edit the CONFIG
# block for your cluster/model, then submit with e.g.:
#
#   sbatch --nodes=2 multinode_fsdp2_ptq.slurm
#
# For a single-node run, `--nodes=1` works unchanged.

#SBATCH --job-name=fsdp2-ptq
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1          # one torchrun launcher per node; it fans out to the GPUs
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.log

set -euo pipefail

# ---------------------------------------------------------------------------
# CONFIG — edit these for your cluster and model. They are exported so `srun`
# propagates them into the container task below.
# ---------------------------------------------------------------------------
export CONTAINER_IMAGE={container_image}                # e.g. nvcr.io#nvidia/pytorch:25.10-py3
export MODELOPT_PATH={path_to_modelopt_repo}            # host clone of TensorRT-Model-Optimizer
export MODEL_PATH={path_to_nemotron3_super}             # local HF checkpoint dir
export EXPORT_PATH={path_to_export_dir}                 # where the quantized checkpoint is written
export RECIPE=general/ptq/nvfp4_default-kv_fp8_cast     # built-in recipe name or /path/to/recipe.yaml
export CALIB_SIZE=512
export BATCH_SIZE=4

# Rendezvous: node 0 is the master; all ranks meet at MASTER_ADDR:MASTER_PORT.
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=29531

# Mount the repo (and the model/export dirs) into the container; run from the llm_ptq example dir.
srun --container-image="${CONTAINER_IMAGE}" \
     --container-mounts="${MODELOPT_PATH}:/modelopt,${MODEL_PATH}:${MODEL_PATH},${EXPORT_PATH}:${EXPORT_PATH}" \
     --container-workdir=/modelopt/examples/llm_ptq \
  bash -c '
    set -euo pipefail
    export PYTHONPATH="/modelopt:${PYTHONPATH:-}"
    export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    pip install -q -r requirements.txt

    torchrun \
      --nnodes="${SLURM_NNODES}" \
      --node_rank="${SLURM_NODEID}" \
      --nproc_per_node="$(nvidia-smi -L | wc -l)" \
      --rdzv_backend=c10d \
      --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
      hf_ptq.py \
        --pyt_ckpt_path "${MODEL_PATH}" \
        --recipe "${RECIPE}" \
        --calib_size "${CALIB_SIZE}" \
        --batch_size "${BATCH_SIZE}" \
        --export_path "${EXPORT_PATH}" \
        --use_fsdp2
  '
