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

#
# Knowledge Distillation Training Script for Megatron-Bridge
#
# This script launches knowledge distillation training using torchrun.
# It uses distill.py which loads models from Megatron-Bridge checkpoints.
#
# Usage:
#  bash distill.sh model.tensor_model_parallel_size=8 model.teacher.tensor_model_parallel_size=8 train.global_batch_size=4 train.micro_batch_size=1 dataset.sequence_length=8192 train.train_iters=5000 logger.log_interval=1
#

# Default paths
STUDENT_CKPT="/workspace/mbridge_models/meta-llama/Llama-3.2-3B-Instruct/iter_0000000"
TEACHER_CKPT="/workspace/mbridge_models/meta-llama/Llama-3.2-3B-Instruct/iter_0000000"
DATA_PATH="/workspace/hf_datasets/wikitext-103-v1/wikitext-train_text_document"
OUTPUT_DIR="./distilled_output"

# Training parameters
TP=8
CP=1
PP=1
DP=1
NUM_NODES=1
DEVICES_PER_NODE=8

# Parse command line arguments (pass through to distill.py)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DISTILL_SCRIPT="${SCRIPT_DIR}/distill.py"

# Set PYTHONPATH to include Megatron-Bridge source and Model-Optimizer
# Model-Optimizer is needed for DistillationConfig class
export PYTHONPATH="/workspace/Megatron-Bridge/src:/workspace/Model-Optimizer:${PYTHONPATH}"

# Launch command
launch_cmd="torchrun --nproc_per_node=$(($TP * $CP * $PP * $DP))"

echo "Starting knowledge distillation..."
echo "Student checkpoint: ${STUDENT_CKPT}"
echo "Teacher checkpoint: ${TEACHER_CKPT}"
echo "Data path: ${DATA_PATH}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Parallelism: TP=${TP}, CP=${CP}, PP=${PP}, DP=${DP}"
echo ""

${launch_cmd} "${DISTILL_SCRIPT}" \
    --student-ckpt "${STUDENT_CKPT}" \
    --teacher-ckpt "${TEACHER_CKPT}" \
    --data-path "${DATA_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    "$@"

