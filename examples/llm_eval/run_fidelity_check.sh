#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# ---
# Quantization fidelity health check: compares a base LLM and its quantized
# variant under teacher-forcing (KL, EAR, top-1 mismatch, ΔNLL).
#
# Prerequisite: two vLLM servers running, one for the base and one for the
# quant.  Both must be started with --max-logprobs >= TOP_K (default 16).
#
#   # Terminal A (base, on GPU 0):
#   CUDA_VISIBLE_DEVICES=0 vllm serve meta-llama/Llama-3-8B \
#       --port 8000 --max-logprobs 20
#
#   # Terminal B (quant, on GPU 1):
#   CUDA_VISIBLE_DEVICES=1 vllm serve /ckpts/llama-3-8b-nvfp4 \
#       --port 8001 --max-logprobs 20 --quantization modelopt
#
#   # Terminal C:
#   bash run_fidelity_check.sh meta-llama/Llama-3-8B 8000 /ckpts/llama-3-8b-nvfp4 8001
# ---

set -euo pipefail

if [ "$#" -lt 4 ]; then
    echo "Usage: $0 <base_model> <base_port> <quant_model> <quant_port> [dataset] [num_prompts] [max_new_tokens]"
    exit 1
fi

BASE_MODEL=$1
BASE_PORT=$2
QUANT_MODEL=$3
QUANT_PORT=$4
DATASET=${5:-cnn_dailymail}
NUM_PROMPTS=${6:-128}
MAX_NEW_TOKENS=${7:-128}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

OUT="fidelity_$(basename "${BASE_MODEL}")_vs_$(basename "${QUANT_MODEL}").json"

cd "${REPO_ROOT}"
# Prepend the repo to PYTHONPATH so the in-tree modelopt resolves even if a
# different editable install is registered in the active environment.
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
python examples/llm_eval/fidelity_check.py \
    --base-url  "http://localhost:${BASE_PORT}/v1"  --base-model  "${BASE_MODEL}" \
    --quant-url "http://localhost:${QUANT_PORT}/v1" --quant-model "${QUANT_MODEL}" \
    --dataset "${DATASET}" \
    --num-prompts "${NUM_PROMPTS}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --output "${OUT}"

echo "Report: ${OUT}"
