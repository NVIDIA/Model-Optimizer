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

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Helper function to parse a single argument value
parse_value() {
    if [[ "$1" != *=* ]]; then shift; fi
    echo "${1#*=}"
}

while [ $# -gt 0 ]; do
  case "$1" in
    --model*)                                   MODEL=$(parse_value "$@"); [[ "$1" != *=* ]] && shift ;;
    --output_dir*)                              OUTPUT_DIR=$(parse_value "$@"); [[ "$1" != *=* ]] && shift ;;
    --dataset*)                                 DATASET=$(parse_value "$@"); [[ "$1" != *=* ]] && shift ;;
    --train_size*)                              TRAIN_SIZE=$(parse_value "$@"); [[ "$1" != *=* ]] && shift ;;
    --eval_size*)                               EVAL_SIZE=$(parse_value "$@"); [[ "$1" != *=* ]] && shift ;;
    --num_epochs*)                              NUM_EPOCHS=$(parse_value "$@"); [[ "$1" != *=* ]] && shift ;;
    --max_steps*)                               MAX_STEPS=$(parse_value "$@"); [[ "$1" != *=* ]] && shift ;;
    --save_steps*)                              SAVE_STEPS=$(parse_value "$@"); [[ "$1" != *=* ]] && shift ;;
    --accum_steps*)                             ACCUM_STEPS=$(parse_value "$@"); [[ "$1" != *=* ]] && shift ;;
    --lr*)                                      LR=$(parse_value "$@"); [[ "$1" != *=* ]] && shift ;;
    --quant_cfg*)                               QUANT_CFG=$(parse_value "$@"); [[ "$1" != *=* ]] && shift ;;
    --compress*)                                COMPRESS=$(parse_value "$@"); [[ "$1" != *=* ]] && shift ;;
    --calib_size*)                              CALIB_SIZE=$(parse_value "$@"); [[ "$1" != *=* ]] && shift ;;
    --train_bs*)                                TRAIN_BS=$(parse_value "$@"); [[ "$1" != *=* ]] && shift ;;
    --eval_bs*)                                 EVAL_BS=$(parse_value "$@"); [[ "$1" != *=* ]] && shift ;;
    --do_train*)                                DO_TRAIN=$(parse_value "$@"); [[ "$1" != *=* ]] && shift ;;
    --lora*)                                    LORA=$(parse_value "$@"); [[ "$1" != *=* ]] && shift ;;
    --teacher_model*)                           TEACHER_MODEL=$(parse_value "$@"); [[ "$1" != *=* ]] && shift ;;
    --distill*)                                 DISTILL=$(parse_value "$@"); [[ "$1" != *=* ]] && shift ;;
    --fsdp_transformer_layer_cls_to_wrap*)      FSDP_TRANSFORMER_LAYER_CLS_TO_WRAP=$(parse_value "$@"); [[ "$1" != *=* ]] && shift ;;
    --max_seq_length*)                          MAX_SEQ_LENGTH=$(parse_value "$@"); [[ "$1" != *=* ]] && shift ;;
    --backend*)                                 BACKEND=$(parse_value "$@"); [[ "$1" != *=* ]] && shift ;;
    --use_fsdp2*)                               USE_FSDP2=$(parse_value "$@"); [[ "$1" != *=* ]] && shift ;;
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
DEFAULT_SAVE_STEPS=$((192 / GPU_COUNT))

MODEL=${MODEL:-"meta-llama/Llama-2-7b-hf"}
OUTPUT_DIR=${OUTPUT_DIR:-"llama2-finetune"}
DATASET=${DATASET:-"Daring-Anteater"}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-4096}
TRAIN_SIZE=${TRAIN_SIZE:-0}
EVAL_SIZE=${EVAL_SIZE:-0}
NUM_EPOCHS=${NUM_EPOCHS:-1}
SAVE_STEPS=${SAVE_STEPS:-$DEFAULT_SAVE_STEPS}
ACCUM_STEPS=${ACCUM_STEPS:-1}
LR=${LR:-"1e-4"}
CALIB_SIZE=${CALIB_SIZE:-512}
TRAIN_BS=${TRAIN_BS:-4}
EVAL_BS=${EVAL_BS:-4}
DO_TRAIN=${DO_TRAIN:-True}
LORA=${LORA:-"False"}
COMPRESS=${COMPRESS:-"False"}
DISTILL=${DISTILL:-"False"}
TEACHER_MODEL=${TEACHER_MODEL:-$MODEL}
FSDP_TRANSFORMER_LAYER_CLS_TO_WRAP=${FSDP_TRANSFORMER_LAYER_CLS_TO_WRAP:-"LlamaDecoderLayer"}
BACKEND=${BACKEND:-"fsdp1"}

if [ -z $QUANT_CFG ]; then
  QUANT_ARGS=""
else
  QUANT_ARGS="--quant_cfg $QUANT_CFG --calib_size $CALIB_SIZE"
fi

OPTIONAL_ARGS=""
if [ ! -z $MAX_STEPS ]; then
  OPTIONAL_ARGS="$OPTIONAL_ARGS --max_steps $MAX_STEPS"
fi

# Set backend based on --backend parameter, with backward compatibility for --use_fsdp2
if [[ "${USE_FSDP2,,}" == "true" ]]; then
  echo "Warning: --use_fsdp2 is deprecated. Use --backend=fsdp2 instead."
  BACKEND="fsdp2"
fi

# if compress is true, set backend to ddp
if [[ "${COMPRESS,,}" == "true" ]]; then
  BACKEND="ddp"
fi

# Configure backend-specific settings
GRADIENT_CHECKPOINTING_ARGS=""
case "${BACKEND,,}" in
  "fsdp1"|"fsdp")
    CONFIG_FILE="fsdp1.yaml"
    FSDP_ARGS="--fsdp_transformer_layer_cls_to_wrap $FSDP_TRANSFORMER_LAYER_CLS_TO_WRAP"
    ;;
  "fsdp2")
    echo "Using FSDP2 instead of FSDP1. FSDP2 is not mature yet! Please use it with latest torch and transformers."
    CONFIG_FILE="fsdp2.yaml"
    FSDP_ARGS="--fsdp_transformer_layer_cls_to_wrap $FSDP_TRANSFORMER_LAYER_CLS_TO_WRAP"
    ;;
  "ddp")
    CONFIG_FILE="ddp.yaml"
    FSDP_ARGS=""
    GRADIENT_CHECKPOINTING_ARGS="--gradient_checkpointing True"
    ;;
  "deepspeed")
    CONFIG_FILE="deepspeed.yaml"
    FSDP_ARGS=""
    GRADIENT_CHECKPOINTING_ARGS="--gradient_checkpointing True"
    ;;
  *)
    echo "Error: Invalid backend '$BACKEND'. Supported backends: fsdp1, fsdp2, ddp, deepspeed"
    exit 1
    ;;
esac

# TODO: Remove this after simple distillation is supported
DISTILLATION_ARGS=""
if [[ "${DISTILL,,}" == "true" ]]; then
  DISTILLATION_ARGS="--distill $DISTILL --teacher_model $TEACHER_MODEL"
  # Distillation does not work with memory efficient loading for FSDP
  if [[ "${BACKEND,,}" == "fsdp1" || "${BACKEND,,}" == "fsdp2" ]]; then
    FSDP_ARGS="$FSDP_ARGS --fsdp_cpu_ram_efficient_loading False"
  fi
fi

CMD="accelerate launch --config-file accelerate_config/$CONFIG_FILE $FSDP_ARGS \
    main.py \
    --model_name_or_path $MODEL \
    --model_max_length $MAX_SEQ_LENGTH \
    --dataloader_drop_last True \
    --do_train $DO_TRAIN \
    --do_eval True \
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --train_size $TRAIN_SIZE \
    --eval_size $EVAL_SIZE \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $TRAIN_BS \
    --per_device_eval_batch_size $EVAL_BS \
    --gradient_accumulation_steps $ACCUM_STEPS \
    --eval_accumulation_steps 1 \
    --save_strategy steps \
    --save_steps $SAVE_STEPS \
    --eval_strategy steps \
    --eval_steps $SAVE_STEPS \
    --load_best_model_at_end True \
    --save_total_limit 2 \
    --learning_rate $LR \
    --weight_decay 0.0 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type linear \
    --logging_steps 1 \
    --report_to tensorboard \
    --lora $LORA \
    --compress $COMPRESS \
      $GRADIENT_CHECKPOINTING_ARGS $QUANT_ARGS $OPTIONAL_ARGS $DISTILLATION_ARGS
"

start_time=$(date +%s)
sh -c "$CMD"
echo "Total time taken: $(( $(date +%s) - $start_time )) seconds"