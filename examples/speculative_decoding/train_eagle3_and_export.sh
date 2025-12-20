#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Set default values for BASE_MODEL, NUM_GPU, and DATA
BASE_MODEL=meta-llama/Llama-3.2-1B-Instruct
NUM_GPU=1
DATA=input_conversations/train.jsonl

# Parse input arguments --base_model, --num_gpu, and --data
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --base_model)
      BASE_MODEL="$2"
      shift; shift
      ;;
    --num_gpu)
      NUM_GPU="$2"
      shift; shift
      ;;
    --data)
      DATA="$2"
      shift; shift
      ;;
    --offline_data)
      OFFLINE_DATA_PATH="$2"
      shift; shift
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done


if [[ "$NUM_GPU" == 1 ]]; then
  export CUDA_VISIBLE_DEVICES=0
else
  # Export as 0,1,...,N-1 for NUM_GPU GPUs
  devs="$(seq -s, 0 $((NUM_GPU-1)))"
  export CUDA_VISIBLE_DEVICES="$devs"
fi

if [[ "$OFFLINE_DATA_PATH" != "" ]]; then
  OFFLINE_DATA_ARGS="--offline-data $OFFLINE_DATA_PATH"
else
  OFFLINE_DATA_ARGS=""
fi

MODEL_BASENAME=$(basename "$BASE_MODEL")

echo "==== [1/3] Training draft model ===="
OUTPUT_DIR=ckpts/${MODEL_BASENAME}-$(date +%Y%m%d_%H%M)
mkdir -p "$(dirname "$OUTPUT_DIR")"
./launch_train.sh --model $BASE_MODEL \
            --output_dir $OUTPUT_DIR \
            $OFFLINE_DATA_ARGS \
            --data $DATA \
            --num_gpu $NUM_GPU \
            --num_epochs 2 \
            --eagle_config eagle_config.json

echo "==== [2/3] Evaluating ModelOpt checkpoint on MT-Bench ===="
python scripts/ar_validate.py --model_path $OUTPUT_DIR

echo "==== [3/3] Exporting checkpoint to deployment format ===="
EXPORT_PATH=export/${MODEL_BASENAME}-$(date +%Y%m%d_%H%M)
mkdir -p "$(dirname "$EXPORT_PATH")"
python scripts/export_hf_checkpoint.py --model_path $OUTPUT_DIR --export_path $EXPORT_PATH
