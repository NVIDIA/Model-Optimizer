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

# ---
# Script to run lm-evaluation-harness against a running vLLM OpenAI-compatible server.
#
# Usage:
#   bash run_lm_eval_vllm.sh <model_name> [port] [task]
#
# Arguments:
#   <model_name>: The name of the model being served (e.g., Qwen/Qwen3-30B-A3B). Used for the 'model' argument in lm_eval.
#   [port]:       The port the vLLM server is listening on (default: 8000).
#   [task]:       The lm_eval task(s) to run (default: mmlu).
#
# Example:
#   # Start vLLM server first (in another terminal):
#   # vllm serve Qwen/Qwen3-30B-A3B --port 8000 --tensor-parallel-size 4
#
#   # Then run this script for MMLU (default task):
#   bash run_lm_eval_vllm.sh Qwen/Qwen3-30B-A3B 8000
#
#   # Run for a different task, e.g., hellaswag:
#   bash run_lm_eval_vllm.sh Qwen/Qwen3-30B-A3B 8000 hellaswag
# ---

set -e
set -x

# --- Argument Parsing ---
if [ -z "$1" ]; then
  echo "Usage: $0 <model_name> [port] [task]"
  exit 1
fi
MODEL_NAME=$1
PORT=${2:-8000}       # Default port is 8000 if not provided
TASK=${3:-mmlu}       # Default task is mmlu if not provided

# --- Environment Setup ---
export OPENAI_API_KEY="local" # Not strictly required for local, but good practice
BASE_URL="http://localhost:${PORT}/v1"
COMPLETIONS_URL="${BASE_URL}/completions"

# --- Evaluation ---
echo "Starting ${TASK} evaluation for model: ${MODEL_NAME} via ${BASE_URL}"
start_ts=$(date +%s)

lm_eval \
  --model local-completions \
  --model_args \
model=${MODEL_NAME},\
base_url=${COMPLETIONS_URL},\
tokenized_requests=False \
  --batch_size auto \
  --tasks ${TASK}

# --- Timing ---
end_ts=$(date +%s)
elapsed=$((end_ts - start_ts))
echo "🏁 ${TASK} eval for ${MODEL_NAME} finished in ${elapsed}s"
