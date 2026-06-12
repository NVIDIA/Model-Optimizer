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

FLOPS_PERCENT=50
BASE_DIR=results/bert_large_pruned_${FLOPS_PERCENT}_percent
PRUNED_MODEL_PATH=${BASE_DIR}/pruned_finetuned/
PTQ_MODEL_PATH=${BASE_DIR}/int8_ptq/
QAT_MODEL_PATH=${BASE_DIR}/int8_qat/

modelopt_args=""
if [ ! -d ${PTQ_MODEL_PATH} ]; then
    modelopt_args="--modelopt_quantize_cfg INT8_DEFAULT_CFG --ptq_model_path ${PTQ_MODEL_PATH}"
    MODEL_NAME_OR_PATH=${PRUNED_MODEL_PATH}
else
    MODEL_NAME_OR_PATH=${PTQ_MODEL_PATH}
fi

# Run distributed QAT on the pruned model with 0.1x LR and less epochs
accelerate launch --multi_gpu --mixed_precision bf16 bert_prune_distill_quantize.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --finetuned_model_path ${QAT_MODEL_PATH} \
    ${modelopt_args} \
    --do_train \
    --do_modelopt_distill \
    --lr_scheduler_type cosine \
    --learning_rate 5e-6 \
    --per_device_train_batch_size 16 \
    --num_train_epochs 2 \
    --with_tracking \
    --resume_from_last_ckpt
