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
# Multi-node AutoQuantize with format shards, ready to run under Slurm.
#
# AutoQuantize scores every candidate format of every layer group against the same reference, so
# the work splits into independent processes, format shards, whose checkpoints merge into the
# unsharded search. Step 1 runs every shard at once: one per GPU, or one per GPUS_PER_SHARD GPUs
# when the model needs several to hold its weights. Shards beyond the number of searched formats
# split a format's layer groups, so adding nodes keeps cutting the scoring time, until a shard's
# calibration dominates. Step 2 merges the shards, solves the recipe's budget and exports on one
# node. Rerunning it with a recipe that changes only the budget solves that budget too; the shipped
# Qwen3.6 budgets also differ in KL scoring context, so each needs its own shards.
#
# The defaults search Qwen3.6-35B-A3B at 3 bits, whose recipe deals six recipes to the shards (five
# formats and BF16, which its KL scoring context measures too); a multiple of six balances best.
# Edit the CONFIG block for your cluster/model, then submit with e.g.:
#
#   sbatch --nodes=3 multinode_autoquant_ptq.slurm
#
# Resubmitting resumes: finished shards restore their scores, unfinished ones their calibration.

#SBATCH --job-name=autoquant-shards
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1          # one launcher per node; it starts that node's shards
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.log

set -euo pipefail

# ---------------------------------------------------------------------------
# CONFIG — edit these for your cluster and model. They are exported so `srun`
# propagates them into the container tasks below. MODEL_PATH, WORK_DIR, EXPORT_PATH and HF_HOME
# must be on storage every node can see; every shard loads the model, so download it first.
# ---------------------------------------------------------------------------
export CONTAINER_IMAGE={container_image}                # e.g. nvcr.io#nvidia/pytorch:25.10-py3
export MODELOPT_PATH={path_to_modelopt_repo}            # host clone of TensorRT-Model-Optimizer
export MODEL_PATH={path_to_model}                       # local dir, e.g. of Qwen/Qwen3.6-35B-A3B
export WORK_DIR={path_to_work_dir}                      # shard checkpoints, merged scores and logs
export EXPORT_PATH={path_to_export_dir}                 # where the quantized checkpoint is written
export HF_HOME={path_to_hf_cache}                       # HF cache for the calibration datasets
export RECIPE=models/Qwen/Qwen3.6-35B-A3B/auto_quantize/iq2_gptq_nvfp4_fp8_experts_at_3p0bits
export CALIB_SIZE=512
export BATCH_SIZE=4                                     # per shard; the same in every shard, for the merge
export GPUS_PER_SHARD=1                                 # GPUs each shard spreads its model over

CONTAINER_MOUNTS="${MODELOPT_PATH}:/modelopt,${MODEL_PATH}:${MODEL_PATH},${WORK_DIR}:${WORK_DIR}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS},${EXPORT_PATH}:${EXPORT_PATH},${HF_HOME}:${HF_HOME}"
mkdir -p "${WORK_DIR}/shards" "${WORK_DIR}/logs"

# Runs a shell command in each task's container, from the hf_ptq example dir, with ModelOpt
# installed and as an independent process: an image may derive torch.distributed variables from
# Slurm, which would make hf_ptq join a process group.
run() {
  srun "$@" --container-image="${CONTAINER_IMAGE}" --container-mounts="${CONTAINER_MOUNTS}" \
       --container-workdir=/modelopt/examples/hf_ptq bash -c '
    set -euo pipefail
    export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
    # The editable install writes into the one clone every node mounts, so nodes take turns.
    mkdir -p /modelopt/build
    (flock 9 || true; pip install -q -e "/modelopt[hf]" --no-build-isolation) 9>/modelopt/build/.pip-install.lock
    pip install -q -r requirements.txt
    '"${COMMAND}"
}
SEARCH_ARGS='--pyt_ckpt_path "${MODEL_PATH}" --recipe "${RECIPE}" --calib_size "${CALIB_SIZE}"'

# Step 1: each node starts one shard per GPUS_PER_SHARD of its GPUs; together they number
# COUNT = nodes * shards per node, the index running over nodes, then GPUs.
COMMAND='
    per_node=$(( $(nvidia-smi -L | wc -l) / GPUS_PER_SHARD ))
    pids=()
    for slot in $(seq 0 $((per_node - 1))); do
      index=$((SLURM_NODEID * per_node + slot))
      gpus=$(seq $((slot * GPUS_PER_SHARD)) $(((slot + 1) * GPUS_PER_SHARD - 1)) | paste -sd, -)
      CUDA_VISIBLE_DEVICES=${gpus} python hf_ptq.py '"${SEARCH_ARGS}"' \
        --auto_quantize_checkpoint "${WORK_DIR}/shards/" \
        --auto_quantize_format_shard "${index}/$((SLURM_NNODES * per_node))" \
        --batch_size "${BATCH_SIZE}" \
        > "${WORK_DIR}/logs/shard${index}.log" 2>&1 &
      pids+=($!)
    done
    # Let every shard finish and save before a failed one fails the step.
    rc=0; for pid in "${pids[@]}"; do wait "${pid}" || rc=1; done; exit "${rc}"
'
run --ntasks-per-node=1

# Step 2: merge the shards' checkpoints, solve the recipe's budget and export, on one node.
COMMAND='
    python hf_ptq.py '"${SEARCH_ARGS}"' \
      --auto_quantize_merge_shards "${WORK_DIR}/shards" \
      --auto_quantize_checkpoint "${WORK_DIR}/merged.pth" \
      --export_path "${EXPORT_PATH}"
'
run --nodes=1 --ntasks=1
