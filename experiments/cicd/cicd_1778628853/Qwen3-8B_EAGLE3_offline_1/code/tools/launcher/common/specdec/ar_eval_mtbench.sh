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

# MT-Bench AR evaluation using scripts/ar_validate.py.
# Finds the latest checkpoint and runs per-category AR validation.
#
# Args are passed directly to ar_validate.py (--model_path, --osl, --steps, etc.)
# If --model_path is not provided, auto-detects from --ckpt_dir.

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source ${SCRIPT_DIR}/../service_utils.sh

pip install -r modules/Model-Optimizer/examples/speculative_decoding/requirements.txt 2>&1 | tail -3

trap 'error_handler $0 $LINENO' ERR

# Parse --ckpt_dir to find latest checkpoint (ar_validate.py expects --model_path)
ARGS=()
CKPT_DIR=""
while [ $# -gt 0 ]; do
  case "$1" in
    --ckpt_dir) shift; CKPT_DIR="$1" ;;
    *) ARGS+=("$1") ;;
  esac
  shift
done

# Auto-detect model_path from ckpt_dir if not explicitly provided
MODEL_PATH=""
if [ -n "$CKPT_DIR" ]; then
    # Find latest checkpoint subdir
    LAST_CKPT=$(ls -d ${CKPT_DIR}/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
    if [ -f "${CKPT_DIR}/model.safetensors" ]; then
        MODEL_PATH="${CKPT_DIR}"
    elif [ -n "$LAST_CKPT" ]; then
        MODEL_PATH="${LAST_CKPT}"
    fi
    echo "Auto-detected model_path: ${MODEL_PATH}"
fi

if [ -z "$MODEL_PATH" ]; then
    echo "ERROR: No checkpoint found. Provide --ckpt_dir or --model_path."
    exit 1
fi

CUDA_VISIBLE_DEVICES=0 python3 modules/Model-Optimizer/examples/speculative_decoding/scripts/ar_validate.py \
    --model_path "${MODEL_PATH}" \
    --per_category \
    "${ARGS[@]}"

report_result "PASS: MT-Bench AR evaluation"
