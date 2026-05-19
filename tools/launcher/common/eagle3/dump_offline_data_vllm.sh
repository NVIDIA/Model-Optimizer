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
# vLLM-based hidden state dumping using the speculators library.
# Uses compute_hidden_states_vllm.py with VllmHiddenStatesGenerator.
# Suitable for: any model supported by vLLM (broader coverage than TRT-LLM or HF device_map).
#
# Required environment:
#   HF_MODEL_CKPT   Path to the HF model checkpoint
#
# Args passed through to compute_hidden_states_vllm.py:
#   --input-data, --output-dir, --max-seq-len, etc.
###################################################################################################

pip install "speculators<0.5.0" --no-deps 2>/dev/null || true
pip install datasets 2>/dev/null || true

# vLLM API compatibility: speculators 0.4.0.1 uses Request(eos_token_id=...) which
# was removed in newer vLLM. Patch to remove the unsupported kwarg.
python3 -c "
import site, os
for d in site.getsitepackages():
    path = os.path.join(d, 'speculators', 'data_generation', 'vllm_hidden_states_generator.py')
    if not os.path.exists(path):
        continue
    with open(path) as f:
        c = f.read()
    old = '                eos_token_id=self.tokenizer.eos_token_id,\n'
    if old in c:
        with open(path, 'w') as f:
            f.write(c.replace(old, ''))
        print('Patched vllm_hidden_states_generator.py: removed eos_token_id from Request()')
    else:
        print('vllm_hidden_states_generator.py: eos_token_id already removed or not found')
    break
" 2>/dev/null || true

# Pydantic 2.13 compatibility: speculators.ReloadableBaseModel.reload_schema() calls
# model_rebuild(force=True) without a types_namespace. In pydantic 2.13+, inherited
# torch.dtype annotations from transformers.PretrainedConfig cannot be resolved in
# subclass modules that don't import torch. Fix by injecting torch into the namespace.
python3 -c "
import site, os
for d in site.getsitepackages():
    path = os.path.join(d, 'speculators', 'utils', 'pydantic_utils.py')
    if not os.path.exists(path):
        continue
    with open(path) as f:
        c = f.read()
    old = 'cls.model_rebuild(force=True)'
    new = 'import torch as _torch; cls.model_rebuild(force=True, _types_namespace={\"torch\": _torch})'
    if old in c and new not in c:
        with open(path, 'w') as f:
            f.write(c.replace(old, new))
        print('Patched pydantic_utils.py: model_rebuild now passes torch namespace')
    else:
        print('pydantic_utils.py already patched or pattern not found')
    break
" 2>/dev/null || true

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

python3 modules/Model-Optimizer/examples/speculative_decoding/collect_hidden_states/compute_hidden_states_vllm.py \
    --model ${HF_MODEL_CKPT} \
    --dp-rank ${TASK_ID} \
    --dp-world-size ${TASK_COUNT} \
    --trust_remote_code \
    ${@}
