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

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

source ${SCRIPT_DIR}/../service_utils.sh

###################################################################################################
# HF-based hidden state dumping for models not supported by TRT-LLM.
# Uses compute_hidden_states_hf.py with device_map="auto" (no TP/EP flags needed).
# Suitable for: VLMs, models with custom code, architectures not yet in TRT-LLM.
#
# Required environment:
#   HF_MODEL_CKPT   Path to the HF model checkpoint
#
# Args passed through to compute_hidden_states_hf.py:
#   --input-data, --output-dir, --max-seq-len, etc.
###################################################################################################

pip install datasets 2>/dev/null || true

if [ -z ${SLURM_ARRAY_TASK_ID} ]; then
    TASK_ID=0
else
    echo "SLURM_ARRAY_TASK_ID ${SLURM_ARRAY_TASK_ID}"
    TASK_ID=${SLURM_ARRAY_TASK_ID}
fi

if [ -z ${SLURM_ARRAY_TASK_COUNT} ]; then
    TASK_COUNT=1
else
    echo "SLURM_ARRAY_TASK_COUNT ${SLURM_ARRAY_TASK_COUNT}"
    TASK_COUNT=${SLURM_ARRAY_TASK_COUNT}
fi

python3 modules/Model-Optimizer/examples/speculative_decoding/collect_hidden_states/compute_hidden_states_hf.py \
    --model ${HF_MODEL_CKPT} \
    --dp-rank ${TASK_ID} \
    --dp-world-size ${TASK_COUNT} \
    --trust_remote_code \
    ${@}
