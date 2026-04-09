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

# PTQ + export for speculative decoding checkpoints (EAGLE3, DFlash).
# Uses hf_ptq.py to quantize and export in one step.
#
# Args are passed directly to hf_ptq.py.

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source ${SCRIPT_DIR}/../service_utils.sh

pip install -r modules/Model-Optimizer/examples/speculative_decoding/requirements.txt 2>&1 | tail -3

trap 'error_handler $0 $LINENO' ERR

# Find latest checkpoint if model_dir points to a training output dir
MODEL_DIR=""
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --model_dir)
      shift
      MODEL_DIR="$1"
      # Auto-detect latest checkpoint
      LAST_CKPT=$(ls -d ${MODEL_DIR}/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
      if [ -f "${MODEL_DIR}/model.safetensors" ]; then
        ARGS+=("--model_dir" "$MODEL_DIR")
      elif [ -n "$LAST_CKPT" ]; then
        echo "Using latest checkpoint: $LAST_CKPT"
        ARGS+=("--model_dir" "$LAST_CKPT")
      else
        ARGS+=("--model_dir" "$MODEL_DIR")
      fi
      ;;
    *) ARGS+=("$1") ;;
  esac
  shift
done

echo "=== PTQ + Export ==="
echo "Args: ${ARGS[*]}"

CUDA_VISIBLE_DEVICES=0 python3 modules/Model-Optimizer/examples/llm_ptq/hf_ptq.py \
    "${ARGS[@]}"

report_result "PASS: PTQ + Export"
