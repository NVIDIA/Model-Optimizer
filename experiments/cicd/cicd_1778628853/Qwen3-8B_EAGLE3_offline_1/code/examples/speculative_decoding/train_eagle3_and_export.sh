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

BASE_MODEL=meta-llama/Llama-3.2-1B-Instruct
DATA=input_conversations/train.jsonl

while [[ $# -gt 0 ]]; do
  case $1 in
    --base_model) BASE_MODEL="$2"; shift; shift ;;
    --data)       DATA="$2";       shift; shift ;;
    --offline_data) OFFLINE_DATA_PATH="$2"; shift; shift ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

MODEL_BASENAME=$(basename "$BASE_MODEL")
OUTPUT_DIR=ckpts/${MODEL_BASENAME}-$(date +%Y%m%d_%H%M)
mkdir -p "$OUTPUT_DIR"

BASE_CFG="$(dirname "$(readlink -f "$0")")/../../modelopt_recipes/general/speculative_decoding/eagle3.yaml"

# Build dotlist overrides
OVERRIDES=(
  model.model_name_or_path="$BASE_MODEL"
  training.output_dir="$OUTPUT_DIR"
)
if [[ -n "$OFFLINE_DATA_PATH" ]]; then
  OVERRIDES+=( data.offline_data_path="$OFFLINE_DATA_PATH" )
else
  OVERRIDES+=( data.data_path="$DATA" )
fi

echo "==== [1/3] Training draft model ===="
./launch_train.sh --config "$BASE_CFG" "${OVERRIDES[@]}"

echo "==== [2/3] Evaluating ModelOpt checkpoint on MT-Bench ===="
python scripts/ar_validate.py --model_path $OUTPUT_DIR

echo "==== [3/3] Exporting checkpoint to deployment format ===="
EXPORT_PATH=export/${MODEL_BASENAME}-$(date +%Y%m%d_%H%M)
mkdir -p "$(dirname "$EXPORT_PATH")"
python scripts/export_hf_checkpoint.py --model_path $OUTPUT_DIR --export_path $EXPORT_PATH
