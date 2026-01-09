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

# CHANGE THE FOLLOWING TO YOUR ACCOUNT AND CHANGE THE JOB NAME TO COMPLY WITH THE
# USAGE. SWITCH TO `-p luna -t 04:00:00` IF YOU HAVE BEEN GRANTED CAPACITY FROM
# THE BIWEEKLY CAPACITY MEETING. IF YOU DON'T KNOW WHO IS THE PIC OF YOUR CSRG PPP
# MANAGEMET, GO WITH `-p backfill -t 00:25:00`.

#SBATCH -A <account_name>
#SBATCH --job-name=<job_name>
#SBATCH --nodes=1 --ntasks-per-node=4 --gpus-per-node=4
#SBATCH -p batch
#SBATCH -t 04:00:00

echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "SLURM_ARRAY_TASK_COUNT: $SLURM_ARRAY_TASK_COUNT"

CONTAINER="nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc4"

INPUT_DIR="<Can be directory containing the .jsonl files, or path to single .jsonl file>"
DUMP_DIR="<Directory for output hidden states>"
MODELOPT_DIR="<Path to Modelopt repo>"
TEACHER_MODEL="<Path to teacher model>"

if [ ! -d "$DUMP_DIR" ]; then
    mkdir -p "$DUMP_DIR"
fi

MOUNTS=$INPUT_DIR:/input,$DUMP_DIR:/output,$MODELOPT_DIR:/modelopt,$TEACHER_MODEL:/model

#By default: TP inside node, and DP across slurm array
#EP optionally available by setting --moe-ep-size and --moe-tp-size. See compute_hidden_states_trtllm.py.
PARALLEL_ARGS="--tp 4 --dp-rank $SLURM_ARRAY_TASK_ID --dp-world-size $SLURM_ARRAY_TASK_COUNT"

RUN_DUMPER="export TLLM_LOG_LEVEL="error";
trtllm-llmapi-launch python3 /modelopt/examples/speculative_decoding/collect_hidden_states/compute_hidden_states_trtllm.py \
  --model /model \
  --input-data /input/ \
  --output-dir /output \
  $PARALLEL_ARGS \
  "

timeout 235m srun -l \
    --mpi=pmix --overlap \
    --output=%x_%j_$DATETIME.log \
    --container-image ${CONTAINER} \
    --container-mounts ${MOUNTS} \
    bash -c "$RUN_DUMPER"
