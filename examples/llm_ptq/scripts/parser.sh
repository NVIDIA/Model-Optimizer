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

# Define a function to parse command-line options
parse_options() {
  # Default values
    MODEL_PATH=""
    QFORMAT=""
    KV_CACHE_QUANT=""
    TP=1
    PP=1
    SPARSITY_FMT="dense"
    LM_EVAL_TASKS="mmlu,gsm8k"
    LM_EVAL_LIMIT=
    SIMPLE_EVAL_TASKS="mmlu"

    TASKS="quant"

    TRUST_REMOTE_CODE=false
    KV_CACHE_FREE_GPU_MEMORY_FRACTION=0.8
    VERBOSE=true
    USE_SEQ_DEVICE_MAP=false

  # Parse command-line options
  ARGS=$(getopt -o "" -l "model:,quant:,kv_cache_quant:,tp:,pp:,sparsity:,awq_block_size:,calib:,calib_batch_size:,auto_quantize_bits:,output:,batch:,tasks:,lm_eval_tasks:,lm_eval_limit:,simple_eval_tasks:,trust_remote_code,use_seq_device_map,gpu_max_mem_percentage:,kv_cache_free_gpu_memory_fraction:,low_memory_mode,no-verbose,calib_dataset:,calib_seq:,auto_quantize_method:,auto_quantize_score_size:,auto_quantize_checkpoint:" -n "$0" -- "$@")

  eval set -- "$ARGS"
  while true; do
    case "$1" in
      --model ) MODEL_PATH="$2"; shift 2;;
      --quant ) QFORMAT="$2"; shift 2;;
      --kv_cache_quant ) KV_CACHE_QUANT="$2"; shift 2;;
      --tp ) TP="$2"; shift 2;;
      --pp ) PP="$2"; shift 2;;
      --sparsity ) SPARSITY_FMT="$2"; shift 2;;
      --awq_block_size ) AWQ_BLOCK_SIZE="$2"; shift 2;;
      --calib ) CALIB_SIZE="$2"; shift 2;;
      --calib_batch_size ) CALIB_BATCH_SIZE="$2"; shift 2;;
      --auto_quantize_bits ) AUTO_QUANTIZE_BITS="$2"; shift 2;;
      --output ) BUILD_MAX_OUTPUT_LEN="$2"; shift 2;;
      --batch ) BUILD_MAX_BATCH_SIZE="$2"; shift 2;;
      --tasks ) TASKS="$2"; shift 2;;
      --lm_eval_tasks ) LM_EVAL_TASKS="$2"; shift 2;;
      --lm_eval_limit ) LM_EVAL_LIMIT="$2"; shift 2;;
      --simple_eval_tasks ) SIMPLE_EVAL_TASKS="$2"; shift 2;;
      --trust_remote_code ) TRUST_REMOTE_CODE=true; shift;;
      --use_seq_device_map ) USE_SEQ_DEVICE_MAP=true; shift;;
      --gpu_max_mem_percentage ) GPU_MAX_MEM_PERCENTAGE="$2"; shift 2;;
      --kv_cache_free_gpu_memory_fraction ) KV_CACHE_FREE_GPU_MEMORY_FRACTION="$2"; shift 2;;
      --no-verbose ) VERBOSE=false; shift;;
      --low_memory_mode ) LOW_MEMORY_MODE=true; shift;;
      --calib_dataset ) CALIB_DATASET="$2"; shift 2;;
      --calib_seq ) CALIB_SEQ="$2"; shift 2;;
      --auto_quantize_method ) AUTO_QUANTIZE_METHOD="$2"; shift 2;;
      --auto_quantize_score_size ) AUTO_QUANTIZE_SCORE_SIZE="$2"; shift 2;;
      --auto_quantize_checkpoint ) AUTO_QUANTIZE_CHECKPOINT="$2"; shift 2;;
      -- ) shift; break ;;
      * ) break ;;
    esac
  done

  DEFAULT_CALIB_SIZE=512
  DEFAULT_CALIB_SEQ=512
  DEFAULT_CALIB_BATCH_SIZE=0
  DEFAULT_BUILD_MAX_OUTPUT_LEN=1024
  DEFAULT_BUILD_MAX_BATCH_SIZE=2

  if [ -z "$CALIB_SIZE" ]; then
    CALIB_SIZE=$DEFAULT_CALIB_SIZE
  fi
  if [ -z "$CALIB_SEQ" ]; then
    CALIB_SEQ=$DEFAULT_CALIB_SEQ
  fi
  if [ -z "$CALIB_BATCH_SIZE" ]; then
    CALIB_BATCH_SIZE=$DEFAULT_CALIB_BATCH_SIZE
  fi
  if [ -z "$BUILD_MAX_OUTPUT_LEN" ]; then
    BUILD_MAX_OUTPUT_LEN=$DEFAULT_BUILD_MAX_OUTPUT_LEN
  fi
  if [ -z "$BUILD_MAX_BATCH_SIZE" ]; then
    BUILD_MAX_BATCH_SIZE=$DEFAULT_BUILD_MAX_BATCH_SIZE
  fi

  # Verify required options are provided
  if [ -z "$MODEL_PATH" ] || [ -z "$QFORMAT" ] || [ -z "$TASKS" ]; then
    echo "Usage: $0 --model=<MODEL_PATH> --quant=<QFORMAT> --tasks=<TASK,...>"
    echo "Optional args: --sparsity=<SPARSITY_FMT> --awq_block_size=<AWQ_BLOCK_SIZE> --calib=<CALIB_SIZE>"
    exit 1
  fi

  VALID_TASKS=("quant" "mmlu" "mtbench" "lm_eval" "livecodebench" "simple_eval")

  for task in $(echo "$TASKS" | tr ',' ' '); do
    is_valid_task=false
    for valid_task in "${VALID_TASKS[@]}"; do
      if [[ "$valid_task" == "$task" ]]; then
        is_valid_task=true
        break
      fi
    done
    if [ "$is_valid_task" = false ]; then
      echo "task $task is not valid"
      VALID_TASKS_STRING=$(IFS=','; echo "${VALID_TASKS[*]}")
      echo "Allowed tasks are: $VALID_TASKS_STRING"
      exit 1
    fi
  done

  # Make sparsity and int4 quantization mutually exclusive as it does not brings speedup
  if [[ "$SPARSITY_FMT" = "sparsegpt" || "$SPARSITY_FMT" = "sparse_magnitude" ]]; then
    if [[ "$QFORMAT" == *"awq"* ]]; then
        echo "Sparsity is not compatible with 'awq' quantization for TRT-LLM deployment."
        exit 1  # Exit script with an error
    fi
  fi

  # Now you can use the variables $GPU, $MODEL, and $TASKS in your script
  echo "================="
  echo "model: $MODEL_PATH"
  echo "quant: $QFORMAT"
  echo "tp (TensorRT-LLM Checkpoint only): $TP"
  echo "pp (TensorRT-LLM Checkpoint only): $PP"
  echo "sparsity: $SPARSITY_FMT"
  echo "awq_block_size: $AWQ_BLOCK_SIZE"
  echo "calib: $CALIB_SIZE"
  echo "calib_batch_size: $CALIB_BATCH_SIZE"
  echo "auto_quantize_bits: $AUTO_QUANTIZE_BITS"
  echo "input: $BUILD_MAX_INPUT_LEN"
  echo "output: $BUILD_MAX_OUTPUT_LEN"
  echo "batch: $BUILD_MAX_BATCH_SIZE"
  echo "tasks: $TASKS"
  echo "lm_eval_tasks: $LM_EVAL_TASKS"
  echo "lm_eval_limit: $LM_EVAL_LIMIT"
  echo "simple_eval_tasks: $SIMPLE_EVAL_TASKS"
  echo "num_sample: $NUM_SAMPLES"
  echo "use_seq_device_map: $USE_SEQ_DEVICE_MAP"
  echo "gpu_max_mem_percentage: $GPU_MAX_MEM_PERCENTAGE"
  echo "kv_cache_free_gpu_memory_fraction: $KV_CACHE_FREE_GPU_MEMORY_FRACTION"
  echo "low_memory_mode: $LOW_MEMORY_MODE"
  echo "calib_dataset: $CALIB_DATASET"
  echo "calib_seq: $CALIB_SEQ"
  echo "auto_quantize_method: $AUTO_QUANTIZE_METHOD"
  echo "auto_quantize_score_size: $AUTO_QUANTIZE_SCORE_SIZE"
  echo "auto_quantize_checkpoint: $AUTO_QUANTIZE_CHECKPOINT"
  echo "================="
}
