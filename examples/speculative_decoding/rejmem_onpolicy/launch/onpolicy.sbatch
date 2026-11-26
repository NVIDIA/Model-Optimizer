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

# On-policy rejection-memory training on a Slurm + pyxis cluster with 4-GPU nodes: every node runs
# launch/node.sh in the vLLM image. Host paths are mounted at the container paths the config uses.
#
#   IMG=<vllm image .sqsh> CODE=<this repo> TARGET=<target model> DRAFT0=<initial drafter export> \
#   DATA=<prompt corpus> OUT=<output dir> [MOUNTS=<extra mounts>] [CONFIG=<container path>] \
#   [EXTRA="key=value ..."] [SERVE_PYTHONPATH=...] [TRAIN_PYTHONPATH=...] \
#     sbatch -N <nodes> -A <account> -p <partition> -t <time> launch/onpolicy.sbatch
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH -J rejmem-onpolicy
set -euo pipefail
: "${IMG:?}" "${CODE:?}" "${TARGET:?}" "${DRAFT0:?}" "${DATA:?}" "${OUT:?}"
mkdir -p "$OUT"
HEAD=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_ADDR=$(getent hosts "$HEAD" | awk '{print $1; exit}')
export NNODES=$SLURM_NNODES OUT_HOST=$OUT OUT=/out
export CONFIG=${CONFIG:-/code/examples/speculative_decoding/rejmem_onpolicy/configs/qwen3.5-9b.yaml}
MNT="$CODE:/code:ro,$TARGET:/model:ro,$DRAFT0:/draft0:ro,$DATA:/data:ro,$OUT_HOST:/out${MOUNTS:+,$MOUNTS}"
srun --ntasks-per-node=1 --container-image="$IMG" --no-container-mount-home --container-mounts="$MNT" \
  --export=ALL bash -c 'NODE_RANK=$SLURM_NODEID exec bash /code/examples/speculative_decoding/rejmem_onpolicy/launch/node.sh' \
  < /dev/null
