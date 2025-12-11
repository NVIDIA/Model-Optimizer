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

while [ $# -gt 0 ]; do
  case "$1" in
    --model*)
      if [[ "$1" != *=* ]]; then shift; fi
      MODEL="${1#*=}"
      ;;
    --max_length*)
      if [[ "$1" != *=* ]]; then shift; fi
      MODEL_MAX_LENGTH="${1#*=}"
      ;;
    --output_dir*)
      if [[ "$1" != *=* ]]; then shift; fi
      OUTPUT_DIR="${1#*=}"
      ;;
    --num_epochs*)
      if [[ "$1" != *=* ]]; then shift; fi
      NUM_EPOCHS="${1#*=}"
      ;;
    --train_bs*)
      if [[ "$1" != *=* ]]; then shift; fi
      TRAIN_BS="${1#*=}"
      ;;
    --eval_bs*)
      if [[ "$1" != *=* ]]; then shift; fi
      EVAL_BS="${1#*=}"
      ;;
    --restore_path*)
      if [[ "$1" != *=* ]]; then shift; fi
      MODELOPT_RESTORE_PATH="${1#*=}"
      ;;
    --data_path*)
      if [[ "$1" != *=* ]]; then shift; fi
      DATA_PATH="${1#*=}"
      ;;
    *)
      >&2 printf "Error: Invalid argument ${1#*=}\n"
      exit 1
      ;;
  esac
  shift
done

set -x

MODEL=${MODEL:-"meta-llama/Llama-2-7b-hf"}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-2048}
OUTPUT_DIR=${OUTPUT_DIR:-"saved_models_Llama-2-7b-hf_sparsegpt_tp1_pp1/finetuned"}
NUM_EPOCHS=${NUM_EPOCHS:-1}
TRAIN_BS=${TRAIN_BS:-1}
EVAL_BS=${EVAL_BS:-4}
MODELOPT_RESTORE_PATH=${MODELOPT_RESTORE_PATH:-"saved_models_Llama-2-7b-hf_sparsegpt_tp1_pp1/pts_modelopt_state.pth"}
DATA_PATH=${DATA_PATH:-"data"}

CMD="accelerate launch --multi_gpu --mixed_precision bf16 finetune.py \
    --model_name_or_path $MODEL \
    --model_max_length $MODEL_MAX_LENGTH \
    --train_datapath $DATA_PATH/cnn_train.json \
    --val_datapath $DATA_PATH/cnn_eval.json \
    --dataloader_drop_last True \
    --bf16 True \
    --do_train \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $TRAIN_BS \
    --per_device_eval_batch_size $EVAL_BS \
    --gradient_accumulation_steps 1 \
    --save_strategy steps \
    --save_steps 2000 \
    --eval_strategy steps \
    --eval_steps 2000 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --weight_decay 0.1 \
    --warmup_ratio 0.0 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --fsdp 'full_shard auto_wrap' \
    --fsdp_transformer_layer_cls_to_wrap LlamaDecoderLayer \
    --tf32 True \
    --modelopt_restore_path $MODELOPT_RESTORE_PATH \
    --report_to tensorboard \
    --gradient_checkpointing True
"

sh -c "$CMD"
