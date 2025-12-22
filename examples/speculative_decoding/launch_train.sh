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

set -eo pipefail

while [ $# -gt 0 ]; do
  case "$1" in
    --training_seq_len*)
      if [[ "$1" != *=* ]]; then shift; fi
      TRAINING_SEQ_LEN="${1#*=}"
      ;;
    --model*)
      if [[ "$1" != *=* ]]; then shift; fi
      MODEL="${1#*=}"
      ;;
    --data*)
      if [[ "$1" != *=* ]]; then shift; fi
      DATA="${1#*=}"
      ;;
    --offline-data*)
      if [[ "$1" != *=* ]]; then shift; fi
      OFFLINE_DATA_PATH="${1#*=}"
      ;;
    --mode*)
      if [[ "$1" != *=* ]]; then shift; fi
      MODE="${1#*=}"
      ;;
    --eagle_decoder_type*)
      if [[ "$1" != *=* ]]; then shift; fi
      EAGLE_DECODER_TYPE="${1#*=}"
      ;;
    --output_dir*)
      if [[ "$1" != *=* ]]; then shift; fi
      OUTPUT_DIR="${1#*=}"
      ;;
    --num_epochs*)
      if [[ "$1" != *=* ]]; then shift; fi
      NUM_EPOCHS="${1#*=}"
      ;;
    --save_steps*)
      if [[ "$1" != *=* ]]; then shift; fi
      SAVE_STEPS="${1#*=}"
      ;;
    --lr*)
      if [[ "$1" != *=* ]]; then shift; fi
      LR="${1#*=}"
      ;;
    --train_bs*)
      if [[ "$1" != *=* ]]; then shift; fi
      TRAIN_BS="${1#*=}"
      ;;
    --medusa_num_heads*)
      if [[ "$1" != *=* ]]; then shift; fi
      MEDUSA_NUM_HEADS="${1#*=}"
      ;;
    --medusa_num_layers*)
      if [[ "$1" != *=* ]]; then shift; fi
      MEDUSA_NUM_LAYERS="${1#*=}"
      ;;
    --eagle_config*)
      if [[ "$1" != *=* ]]; then shift; fi
      EAGLE_CONFIG="${1#*=}"
      ;;
    --fsdp_transformer_layer_cls_to_wrap*)
      if [[ "$1" != *=* ]]; then shift; fi
      FSDP_TRANSFORMER_LAYER_CLS_TO_WRAP="${1#*=}"
      ;;
    --num_gpu*)
      if [[ "$1" != *=* ]]; then shift; fi
      NUM_GPU="${1#*=}"
      ;;
    --disable_tqdm*)
      if [[ "$1" != *=* ]]; then shift; fi
      DISABLE_TQDM="${1#*=}"
      ;;
    --vlm_processor*)
      if [[ "$1" != *=* ]]; then shift; fi
      VLM_PROCESSOR="${1#*=}"
      ;;
    --vlm_img_dir*)
      if [[ "$1" != *=* ]]; then shift; fi
      VLM_IMG_DIR="${1#*=}"
      ;;
    --estimate_ar*)
      if [[ "$1" != *=* ]]; then shift; fi
      ESTIMATE_AR="${1#*=}"
      ;;
    --ar_validate_steps*)
      if [[ "$1" != *=* ]]; then shift; fi
      AR_VALIDATE_STEPS="${1#*=}"
      ;;
    *)
      >&2 printf "Error: Invalid argument ${1#*=}\n"
      exit 1
      ;;
  esac
  shift
done

set -x

# Get the default value for save_steps based on the available number of GPUs
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
# Calculate save_steps
DEFAULT_SAVE_STEPS=$((8192 / GPU_COUNT))

MODEL=${MODEL:-"TinyLlama/TinyLlama-1.1B-Chat-v1.0"}
MODE=${MODE:-"eagle3"}
EAGLE_DECODER_TYPE=${EAGLE_DECODER_TYPE:-"llama"}
# Set default OUTPUT_DIR to ckpts/{modelname}, where {modelname} is the last part of the model path
MODEL_BASENAME=$(basename "$MODEL")
OUTPUT_DIR=${OUTPUT_DIR:-"ckpts/${MODEL_BASENAME}-$(date +%Y%m%d_%H%M)"}
NUM_EPOCHS=${NUM_EPOCHS:-1}
SAVE_STEPS=${SAVE_STEPS:-$DEFAULT_SAVE_STEPS}
LR=${LR:-"1e-4"}
TRAIN_BS=${TRAIN_BS:-4}
MEDUSA_NUM_HEADS=${MEDUSA_NUM_HEADS:-1}
MEDUSA_NUM_LAYERS=${MEDUSA_NUM_LAYERS:-1}
FSDP_TRANSFORMER_LAYER_CLS_TO_WRAP=${FSDP_TRANSFORMER_LAYER_CLS_TO_WRAP:-"LlamaDecoderLayer"}
NUM_GPU=${NUM_GPU:-1}
TRAINING_SEQ_LEN=${TRAINING_SEQ_LEN:-2048}
OFFLINE_DATA_PATH=${OFFLINE_DATA_PATH:-""}
DISABLE_TQDM=${DISABLE_TQDM:-False}
VLM_PROCESSOR=${VLM_PROCESSOR:-}
VLM_IMG_DIR=${VLM_IMG_DIR:-}
AR_VALIDATE_STEPS=${AR_VALIDATE_STEPS:-1000}
ESTIMATE_AR=${ESTIMATE_AR:-False}

if [[ "$MODE" == "medusa" ]]; then
  SPECULATIVE_ARGS="--medusa_num_heads $MEDUSA_NUM_HEADS --medusa_num_layers $MEDUSA_NUM_LAYERS"
elif [[ "$MODE" == "eagle1" || "$MODE" == "eagle3" ]]; then
  if [[ -n "$EAGLE_CONFIG" ]]; then
    SPECULATIVE_ARGS="--eagle_config $EAGLE_CONFIG"
  else
    SPECULATIVE_ARGS=""
  fi
else
  echo "Only medusa, eagle1, eagle3 supported for now!"
  exit 1
fi

if [[ "$OFFLINE_DATA_PATH" != "" ]]; then
  if [[ ! -d "$OFFLINE_DATA_PATH" ]]; then
    echo "Offline data path $OFFLINE_DATA_PATH does not exist or is not a directory."
    exit 1
  else
    OFFLINE_TRAINING_ARGS="--offline-data-path $OFFLINE_DATA_PATH --ar_validate_steps -1"
  fi
else
  OFFLINE_TRAINING_ARGS=""
fi

if [[ "$NUM_GPU" == 1 ]]; then
  MULTI_GPU=""
else
  MULTI_GPU="--multi_gpu"
fi

if [[ "$VLM_PROCESSOR" != "" ]]; then
  VLM_ARGS="--vlm_processor $VLM_PROCESSOR --vlm_img_dir $VLM_IMG_DIR"
else
  VLM_ARGS=""
fi

# Disable tokenizers parallelism to avoid warning
export TOKENIZERS_PARALLELISM=False
CMD="accelerate launch $MULTI_GPU --mixed_precision bf16 main.py \
    --mode $MODE \
    --eagle_decoder_type $EAGLE_DECODER_TYPE \
    --model_name_or_path $MODEL \
    --training_seq_len $TRAINING_SEQ_LEN \
    --dataloader_drop_last True \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $TRAIN_BS \
    --per_device_eval_batch_size $TRAIN_BS \
    --gradient_accumulation_steps 1 \
    --do_eval False \
    --eval_accumulation_steps 1 \
    --save_strategy steps \
    --save_steps $SAVE_STEPS \
    --learning_rate $LR \
    --weight_decay 0.0 \
    --warmup_steps 100 \
    --lr_scheduler_type linear \
    --logging_steps 100 \
    --tf32 True \
    --data_path $DATA \
    --disable_tqdm $DISABLE_TQDM \
    --estimate_ar $ESTIMATE_AR \
    --ar_validate_steps $AR_VALIDATE_STEPS \
    $VLM_ARGS \
    $OFFLINE_TRAINING_ARGS \
    $SPECULATIVE_ARGS \
"

start_time=$(date +%s)
sh -c "$CMD"
echo "Total time taken: $(( $(date +%s) - $start_time )) seconds"
