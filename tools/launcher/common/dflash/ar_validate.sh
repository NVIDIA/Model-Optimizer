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

# DFlash AR (Acceptance Rate) validation script.
# Loads a trained DFlash checkpoint and evaluates speculative decoding AR on MT-Bench.
#
# Required env vars:
#   HF_MODEL_CKPT       — path to the target HuggingFace model
#   DFLASH_CKPT          — path to the trained DFlash checkpoint
#   DFLASH_BLOCK_SIZE    — block size (default: 16)
#   DFLASH_NUM_LAYERS    — number of draft layers (default: 5)
#   DFLASH_MASK_TOKEN_ID — mask token ID (default: auto-detect)
#   NUM_SAMPLES          — number of MT-Bench samples to evaluate (default: 20)

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source ${SCRIPT_DIR}/../service_utils.sh
trap 'error_handler $0 $LINENO' ERR

pip install --upgrade "transformers>=4.57" 2>&1 | tail -3

DFLASH_BLOCK_SIZE=${DFLASH_BLOCK_SIZE:-16}
DFLASH_NUM_LAYERS=${DFLASH_NUM_LAYERS:-5}
NUM_SAMPLES=${NUM_SAMPLES:-20}

# Build mask_token_id arg
if [ -n "$DFLASH_MASK_TOKEN_ID" ]; then
    MASK_ARG="'mask_token_id': ${DFLASH_MASK_TOKEN_ID},"
else
    MASK_ARG=""
fi

echo "=== DFlash AR Validation ==="
echo "Target model: ${HF_MODEL_CKPT}"
echo "DFlash checkpoint: ${DFLASH_CKPT}"
echo "Block size: ${DFLASH_BLOCK_SIZE}"
echo "Draft layers: ${DFLASH_NUM_LAYERS}"
echo "Samples: ${NUM_SAMPLES}"

CUDA_VISIBLE_DEVICES=0 python3 -c "
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from modelopt.torch.speculative.plugins.transformers import HFARValidation
import modelopt.torch.opt as mto
import modelopt.torch.speculative as mtsp

mto.enable_huggingface_checkpointing()

model = AutoModelForCausalLM.from_pretrained(
    '${HF_MODEL_CKPT}', torch_dtype=torch.bfloat16, device_map={'': 0}, trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained('${HF_MODEL_CKPT}', trust_remote_code=True)

config = {
    'dflash_block_size': ${DFLASH_BLOCK_SIZE},
    'dflash_architecture_config': {
        'num_hidden_layers': ${DFLASH_NUM_LAYERS},
        ${MASK_ARG}
    },
    'dflash_use_torch_compile': False,
}
mtsp.convert(model, [('dflash', config)])

# Load trained DFlash weights
import glob
from safetensors.torch import load_file
ckpt_files = sorted(glob.glob('${DFLASH_CKPT}/model*.safetensors'))
if ckpt_files:
    state = {}
    for f in ckpt_files:
        state.update(load_file(f))
    # Try with dflash_module prefix first (ModelOpt format)
    dflash_keys = {k: v for k, v in state.items() if 'dflash_module' in k}
    if dflash_keys:
        model.load_state_dict(dflash_keys, strict=False)
        print(f'Loaded {len(dflash_keys)} DFlash weights (with prefix)')
    else:
        # No prefix — SpecForge format, load directly into dflash_module
        result = model.dflash_module.load_state_dict(state, strict=False)
        loaded = len(state) - len(result.unexpected_keys)
        print(f'Loaded {loaded} DFlash weights (no prefix), missing={len(result.missing_keys)}')
else:
    print('WARNING: No checkpoint files found, using random weights')

model.eval()
validator = HFARValidation(model, tokenizer)

ds = load_dataset('/hf-local/HuggingFaceH4/mt_bench_prompts')['train']
num_samples = min(${NUM_SAMPLES}, len(ds))

ars = []
for i in range(num_samples):
    prompt = ds[i]['prompt'][0]
    chat = [{'role': 'user', 'content': prompt}]
    text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(text, return_tensors='pt').input_ids.cuda()
    try:
        _, ar = validator.validate(osl=32, input_ids=input_ids, steps=3)
        ars.append(ar)
        print(f'  AR={ar:.2f} | {prompt[:60]}')
    except Exception as e:
        print(f'  ERROR | {prompt[:60]}... | {e}')

if ars:
    avg_ar = sum(ars) / len(ars)
    print(f'\n==== DFlash AR Results ====')
    print(f'Samples: {len(ars)}')
    print(f'Average AR: {avg_ar:.4f}')
    print(f'Min AR: {min(ars):.4f}')
    print(f'Max AR: {max(ars):.4f}')
else:
    print('No AR results collected')
"
