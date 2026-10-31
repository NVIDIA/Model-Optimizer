#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# Multi-node expert-parallel post-training quantization of an MoE model, ready to run under Slurm.
#
# Slurm allocates the nodes and launches one task per node; each task starts a `torchrun` that
# spawns one process per GPU. Without `--use_fsdp2`, `hf_ptq.py` shards the MoE's routed experts
# across all ranks (world_size = num_nodes * gpus_per_node), calibrates each rank on its own share
# of the data, and has every rank write its share of the checkpoint. Expert parallelism needs
# transformers 5.7 or newer; see the README's expert-parallel section for the supported models.
#
# The defaults below quantize Qwen/Qwen3-235B-A22B to NVFP4 with an FP8 KV cache. Edit the CONFIG
# block for your cluster/model, then submit with e.g.:
#
#   sbatch --nodes=2 multinode_ep_ptq.slurm
#
# For a single-node run, `--nodes=1` works unchanged.

#SBATCH --job-name=ep-ptq
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
export MODEL_PATH=Qwen/Qwen3-235B-A22B                  # HF repo id (auto-downloaded) or a local dir
export EXPORT_PATH={path_to_export_dir}                 # where the quantized checkpoint is written
export HF_HOME={path_to_hf_cache}                       # HF cache; persist so a repo id isn't re-downloaded each run
# export HF_TOKEN={hf_token}                            # required for gated repos (or `huggingface-cli login` on host)
export RECIPE=general/ptq/nvfp4_default-kv_fp8          # built-in recipe name or /path/to/recipe.yaml
export CALIB_SIZE=512                                   # total over all ranks
export BATCH_SIZE=4
# export MODELOPT_EP_DEGREE=16                          # expert-parallel degree (default: auto)
# export MODELOPT_FSDP_SIZE=1                           # 1 replicates the dense part, world shards it (default: auto)

# Rendezvous: node 0 is the master; all ranks meet at MASTER_ADDR:MASTER_PORT.
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=29531

# Mount the repo, export dir, and HF cache; add the model dir only when MODEL_PATH is a local path
# (a repo id is downloaded into HF_HOME instead). Run from the hf_ptq example dir.
CONTAINER_MOUNTS="${MODELOPT_PATH}:/modelopt,${EXPORT_PATH}:${EXPORT_PATH},${HF_HOME}:${HF_HOME}"
if [ -d "${MODEL_PATH}" ]; then
    CONTAINER_MOUNTS="${CONTAINER_MOUNTS},${MODEL_PATH}:${MODEL_PATH}"
fi
srun --container-image="${CONTAINER_IMAGE}" \
     --container-mounts="${CONTAINER_MOUNTS}" \
     --container-workdir=/modelopt/examples/hf_ptq \
  bash -c '
    set -euo pipefail
    export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    # ModelOpt + its HF deps (transformers/accelerate/datasets/...) from the mounted source, then example extras.
    # Expert parallelism needs transformers 5.7 or newer, which the image may not ship.
    pip install -q -e "/modelopt[hf]" "transformers>=5.7" --no-build-isolation
    pip install -q -r requirements.txt
    python -c "import transformers as t; print(t.__name__, t.__version__)"

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
        --export_path "${EXPORT_PATH}"
  '
