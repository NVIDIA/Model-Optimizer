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

set -ex

MODEL_NAME_OR_PATH=bert-large-uncased-whole-word-masking-finetuned-squad
FLOPS_PERCENT=50
BASE_DIR=results/bert_large_pruned_${FLOPS_PERCENT}_percent
PRUNED_MODEL_PATH=${BASE_DIR}/pruned/
FINETUNED_MODEL_PATH=${BASE_DIR}/pruned_finetuned/

modelopt_args=""
if [ ! -d ${PRUNED_MODEL_PATH} ]; then
    modelopt_args="--do_modelopt_prune \
        --modelopt_prune_flops_percent ${FLOPS_PERCENT} \
        --pruned_model_path ${PRUNED_MODEL_PATH}"
else
    MODEL_NAME_OR_PATH=${PRUNED_MODEL_PATH}
fi

# Run pruning followed by distributed fine-tuning on the pruned model
accelerate launch --multi_gpu --mixed_precision bf16 bert_prune_distill_quantize.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --finetuned_model_path ${FINETUNED_MODEL_PATH} \
    ${modelopt_args} \
    --do_train \
    --do_modelopt_distill \
    --lr_scheduler_type cosine \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 16 \
    --num_train_epochs 15 \
    --with_tracking \
    --resume_from_last_ckpt
