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

# Evaluate a trained DMS model using lm-eval-harness.
#
# Prerequisites:
#   pip install -e .  (installs dms + lm-eval)
#
# Usage:
#   bash scripts/evaluate.sh /path/to/student_model
#
# The saved model imports from the dms package, so it must be installed
# in the environment where evaluation runs.

set -x

MODEL_PATH=$1
test -z "$MODEL_PATH" && echo "Usage: bash scripts/evaluate.sh MODEL_PATH" && exit 1

accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=${MODEL_PATH},dtype="bfloat16",trust_remote_code=true,dms_chunked_prefill=4096 \
    --tasks niah_single_2 \
    --output_path "${MODEL_PATH}/eval_results" \
    --log_samples \
    --device cuda \
    --batch_size 2 \
    --metadata '{"max_seq_lengths":[32768]}'
