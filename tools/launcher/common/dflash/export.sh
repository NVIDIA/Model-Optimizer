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

# Export speculative decoding checkpoint to deployment format.
# Auto-detects latest checkpoint and exports via export_hf_checkpoint.py.
#
# Args:
#   --model_path   Training output dir (auto-detects latest checkpoint)
#   --export_path  Destination directory

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source ${SCRIPT_DIR}/../service_utils.sh

pip install -r modules/Model-Optimizer/examples/speculative_decoding/requirements.txt 2>&1 | tail -3

trap 'error_handler $0 $LINENO' ERR

# Auto-detect latest checkpoint
MODEL_PATH=""
EXPORT_PATH=""
while [ $# -gt 0 ]; do
  case "$1" in
    --model_path)  shift; MODEL_PATH="$1" ;;
    --export_path) shift; EXPORT_PATH="$1" ;;
    *) ;;
  esac
  shift
done

# Find latest checkpoint if model_path is a training dir
if [ ! -f "${MODEL_PATH}/model.safetensors" ]; then
    LAST_CKPT=$(ls -d ${MODEL_PATH}/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
    if [ -n "$LAST_CKPT" ]; then
        echo "Using latest checkpoint: $LAST_CKPT"
        MODEL_PATH="$LAST_CKPT"
    fi
fi

echo "=== Export ==="
echo "Model: ${MODEL_PATH}"
echo "Export: ${EXPORT_PATH}"

CUDA_VISIBLE_DEVICES=0 python3 modules/Model-Optimizer/examples/speculative_decoding/scripts/export_hf_checkpoint.py \
    --model_path "${MODEL_PATH}" \
    --export_path "${EXPORT_PATH}"

echo "Export contents:"
ls -lh ${EXPORT_PATH}/

report_result "PASS: Export"
