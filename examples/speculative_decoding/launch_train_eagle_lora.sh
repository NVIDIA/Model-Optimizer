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

# Launch script for co-training Eagle3 + LoRA.
# Usage:
#   ./launch_train_eagle_lora.sh --model Qwen/Qwen3-VL-2B-Instruct --data path/to/data.jsonl ...

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
    --eagle_config*)
      if [[ "$1" != *=* ]]; then shift; fi
      EAGLE_CONFIG="${1#*=}"
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
    --cp_size*)
      if [[ "$1" != *=* ]]; then shift; fi
      CP_SIZE="${1#*=}"
      ;;
    --dp_size*)
      if [[ "$1" != *=* ]]; then shift; fi
      DP_SHARD_SIZE="${1#*=}"
      ;;
    --log_steps*)
      if [[ "$1" != *=* ]]; then shift; fi
      LOG_STEPS="${1#*=}"
      ;;
    --draft_vocab_cache*)
      if [[ "$1" != *=* ]]; then shift; fi
      DRAFT_VOCAB_CACHE="${1#*=}"
      ;;
    --mix_hidden_states*)
      if [[ "$1" != *=* ]]; then shift; fi
      MIX_HIDDEN_STATES="${1#*=}"
      ;;
    --num_nodes*)
      if [[ "$1" != *=* ]]; then shift; fi
      NUM_NODES="${1#*=}"
      ;;
    --head_node_ip*)
      if [[ "$1" != *=* ]]; then shift; fi
      HEAD_NODE_IP="${1#*=}"
      ;;
    # LoRA-specific arguments
    --lora_r*)
      if [[ "$1" != *=* ]]; then shift; fi
      LORA_R="${1#*=}"
      ;;
    --lora_alpha*)
      if [[ "$1" != *=* ]]; then shift; fi
      LORA_ALPHA="${1#*=}"
      ;;
    --lora_dropout*)
      if [[ "$1" != *=* ]]; then shift; fi
      LORA_DROPOUT="${1#*=}"
      ;;
    --lora_target_modules*)
      if [[ "$1" != *=* ]]; then shift; fi
      LORA_TARGET_MODULES="${1#*=}"
      ;;
    --lora_adapter_path*)
      if [[ "$1" != *=* ]]; then shift; fi
      LORA_ADAPTER_PATH="${1#*=}"
      ;;
    *)
      >&2 printf "Error: Invalid argument ${1#*=}\n"
      exit 1
      ;;
  esac
  shift
done

set -x

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
NUM_NODES=${NUM_NODES:-1}
GPU_PER_NODE=${GPU_PER_NODE:-$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)}
TOTAL_GPU=$((NUM_NODES * GPU_PER_NODE))
echo "Total GPUs: $TOTAL_GPU (NUM_NODES: $NUM_NODES, GPU_PER_NODE: $GPU_PER_NODE)"
DEFAULT_SAVE_STEPS=$((8192 / TOTAL_GPU))

MODEL=${MODEL:-"TinyLlama/TinyLlama-1.1B-Chat-v1.0"}
EAGLE_DECODER_TYPE=${EAGLE_DECODER_TYPE:-"llama"}
MODEL_BASENAME=$(basename "$MODEL")
OUTPUT_DIR=${OUTPUT_DIR:-"ckpts/${MODEL_BASENAME}-eagle-lora-$(date +%Y%m%d_%H%M)"}
NUM_EPOCHS=${NUM_EPOCHS:-1}
SAVE_STEPS=${SAVE_STEPS:-$DEFAULT_SAVE_STEPS}
LR=${LR:-"1e-4"}
TRAIN_BS=${TRAIN_BS:-1}
TRAINING_SEQ_LEN=${TRAINING_SEQ_LEN:-2048}
VLM_PROCESSOR=${VLM_PROCESSOR:-}
VLM_IMG_DIR=${VLM_IMG_DIR:-}
AR_VALIDATE_STEPS=${AR_VALIDATE_STEPS:-1000}
ESTIMATE_AR=${ESTIMATE_AR:-False}
CP_SIZE=${CP_SIZE:-1}
DP_SHARD_SIZE=${DP_SHARD_SIZE:-$((TOTAL_GPU/CP_SIZE))}
LOG_STEPS=${LOG_STEPS:-100}
DRAFT_VOCAB_CACHE=${DRAFT_VOCAB_CACHE:-""}
MIX_HIDDEN_STATES=${MIX_HIDDEN_STATES:-"False"}

# LoRA defaults
LORA_R=${LORA_R:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-"q_proj,k_proj,v_proj,o_proj"}
LORA_ADAPTER_PATH=${LORA_ADAPTER_PATH:-""}

if [[ -n "$EAGLE_CONFIG" ]]; then
  SPECULATIVE_ARGS="--eagle_config $EAGLE_CONFIG"
else
  SPECULATIVE_ARGS=""
fi

if [[ "$VLM_PROCESSOR" != "" ]]; then
  VLM_ARGS="--vlm_processor $VLM_PROCESSOR --vlm_img_dir $VLM_IMG_DIR"
else
  VLM_ARGS=""
fi

if [[ "$TOTAL_GPU" -gt 1 ]]; then
  FSDP_ARGS="--fsdp 'full_shard' --fsdp_config ${SCRIPT_DIR}/fsdp_config.json"
else
  FSDP_ARGS=""
fi

if [[ "$DRAFT_VOCAB_CACHE" != "" ]]; then
  DRAFT_VOCAB_CACHE_ARGS="--draft_vocab_cache $DRAFT_VOCAB_CACHE"
else
  DRAFT_VOCAB_CACHE_ARGS=""
fi

if [[ "$LORA_ADAPTER_PATH" != "" ]]; then
  LORA_ADAPTER_ARGS="--lora_adapter_path $LORA_ADAPTER_PATH"
else
  LORA_ADAPTER_ARGS=""
fi

if [[ "$NUM_NODES" != 1 ]]; then
  MULTI_NODE_ARGS="--num_processes $TOTAL_GPU \
                   --num_machines $NUM_NODES \
                   --machine_rank $SLURM_PROCID \
                   --rdzv_backend c10d \
                   --main_process_ip $HEAD_NODE_IP \
                   --main_process_port 29500"
else
  MULTI_NODE_ARGS=""
fi

export TOKENIZERS_PARALLELISM=False
CMD="accelerate launch $MULTI_NODE_ARGS --mixed_precision bf16 ${SCRIPT_DIR}/main_eagle_lora.py \
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
    --logging_steps $LOG_STEPS \
    --tf32 True \
    --data_path $DATA \
    --estimate_ar $ESTIMATE_AR \
    --ar_validate_steps $AR_VALIDATE_STEPS \
    --mix_hidden_states $MIX_HIDDEN_STATES \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --lora_target_modules $LORA_TARGET_MODULES \
    $LORA_ADAPTER_ARGS \
    $DRAFT_VOCAB_CACHE_ARGS \
    $VLM_ARGS \
    $SPECULATIVE_ARGS \
    $FSDP_ARGS \
    --cp_size $CP_SIZE \
    --dp_shard_size $DP_SHARD_SIZE \
"

start_time=$(date +%s)
sh -c "$CMD"
echo "Total time taken: $(( $(date +%s) - start_time )) seconds"
