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

# DFlash online training + AR validation script for the ModelOpt Launcher.
# Trains a DFlash draft model alongside the frozen target model,
# then evaluates acceptance rate on MT-Bench.
#
# Required env vars:
#   HF_MODEL_CKPT  — path to the target HuggingFace model
#
# Optional env vars:
#   NUM_AR_SAMPLES — number of MT-Bench samples for AR validation (default: 20, 0 to skip)
#
# All other args are passed through to launch_train.sh.

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source ${SCRIPT_DIR}/../service_utils.sh

pip install -r modules/Model-Optimizer/examples/speculative_decoding/requirements.txt
pip install huggingface-hub>=1.2.1
export PATH=$PATH:/workspace/.local/bin

###################################################################################################

trap 'error_handler $0 $LINENO' ERR

# Auto-detect head node IP for multi-node training
if [ -n "$SLURM_JOB_NODELIST" ] && [ -z "$HEAD_NODE_IP" ]; then
    # Try scontrol first (works outside container), then parse SLURM env directly
    HEAD_NODE_IP=$(scontrol show hostnames "$SLURM_JOB_NODELIST" 2>/dev/null | head -1)
    if [ -z "$HEAD_NODE_IP" ]; then
        # Parse nodelist directly: "node[001-002]" → "node001"
        HEAD_NODE_IP=$(echo "$SLURM_JOB_NODELIST" | sed 's/\[.*//; s/,.*//; s/ .*//')
        # Resolve to IP
        HEAD_NODE_IP=$(getent hosts "$HEAD_NODE_IP" 2>/dev/null | awk '{print $1}' || echo "$HEAD_NODE_IP")
    fi
    export HEAD_NODE_IP
    echo "Auto-detected HEAD_NODE_IP: ${HEAD_NODE_IP}"
fi

# Parse DFlash-specific args from the command line for AR validation
DFLASH_BLOCK_SIZE=16
DFLASH_NUM_LAYERS=5
DFLASH_MASK_TOKEN_ID=""
OUTPUT_DIR=""
for arg in "$@"; do
    case "$arg" in
        --dflash_block_size) next_is_block_size=1 ;;
        --dflash_num_layers) next_is_num_layers=1 ;;
        --dflash_mask_token_id) next_is_mask_id=1 ;;
        --output_dir) next_is_output=1 ;;
        *)
            if [ "$next_is_block_size" = "1" ]; then DFLASH_BLOCK_SIZE="$arg"; next_is_block_size=0; fi
            if [ "$next_is_num_layers" = "1" ]; then DFLASH_NUM_LAYERS="$arg"; next_is_num_layers=0; fi
            if [ "$next_is_mask_id" = "1" ]; then DFLASH_MASK_TOKEN_ID="$arg"; next_is_mask_id=0; fi
            if [ "$next_is_output" = "1" ]; then OUTPUT_DIR="$arg"; next_is_output=0; fi
            ;;
    esac
done

# Step 1: Training
bash modules/Model-Optimizer/examples/speculative_decoding/launch_train.sh \
    --model ${HF_MODEL_CKPT} \
    --mode dflash \
    ${@}

# Step 2: AR Validation
NUM_AR_SAMPLES=${NUM_AR_SAMPLES:-20}
if [ "${NUM_AR_SAMPLES}" = "0" ]; then
    echo "Skipping AR validation (NUM_AR_SAMPLES=0)"
    exit 0
fi

if [ -z "$OUTPUT_DIR" ]; then
    echo "WARNING: --output_dir not found in args, skipping export and AR validation"
    exit 0
fi

# Step 2: Export checkpoint to z-lab HF format
EXPORT_DIR=${OUTPUT_DIR}/export
echo ""
echo "=== Exporting DFlash checkpoint ==="
echo "Source: ${OUTPUT_DIR}"
echo "Export: ${EXPORT_DIR}"

CUDA_VISIBLE_DEVICES=0 python3 -c "
import torch
import modelopt.torch.opt as mto
from modelopt.torch.export import export_speculative_decoding
from modelopt.torch.speculative.utils import load_vlm_or_llm, patch_transformers5_params_loading

mto.enable_huggingface_checkpointing()
try:
    with patch_transformers5_params_loading():
        model = load_vlm_or_llm(
            '${OUTPUT_DIR}',
            torch_dtype=torch.bfloat16,
            device_map={'': 'cpu'},
            trust_remote_code=True,
        )
    model.eval()
    with torch.inference_mode():
        export_speculative_decoding(model, export_dir='${EXPORT_DIR}')
    print('Export complete')
except Exception as e:
    print(f'Export failed: {e}')
" || echo "WARNING: Export script failed, continuing with AR validation"

echo ""
echo "Export contents:"
ls -la ${EXPORT_DIR}/ 2>/dev/null || echo "No export dir"

# Step 3: AR Validation
# Build mask_token_id config
if [ -n "$DFLASH_MASK_TOKEN_ID" ]; then
    MASK_ARG="'mask_token_id': ${DFLASH_MASK_TOKEN_ID},"
else
    MASK_ARG=""
fi

echo ""
echo "=== DFlash AR Validation ==="
echo "Target model: ${HF_MODEL_CKPT}"
# Prefer exported checkpoint (no prefix), fall back to training output (with prefix)
if [ -f "${EXPORT_DIR}/model.safetensors" ]; then
    AR_CKPT=${EXPORT_DIR}
    echo "Using exported checkpoint: ${AR_CKPT}"
else
    AR_CKPT=${OUTPUT_DIR}
    echo "Using training checkpoint: ${AR_CKPT}"
fi
echo "Block size: ${DFLASH_BLOCK_SIZE}"
echo "Draft layers: ${DFLASH_NUM_LAYERS}"
echo "Samples: ${NUM_AR_SAMPLES}"

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
ckpt_files = sorted(glob.glob('${AR_CKPT}/model*.safetensors'))
if ckpt_files:
    state = {}
    for f in ckpt_files:
        state.update(load_file(f))
    dflash_keys = {k: v for k, v in state.items() if 'dflash_module' in k}
    if dflash_keys:
        model.load_state_dict(dflash_keys, strict=False)
        print(f'Loaded {len(dflash_keys)} DFlash weights (with prefix)')
    else:
        result = model.dflash_module.load_state_dict(state, strict=False)
        loaded = len(state) - len(result.unexpected_keys)
        print(f'Loaded {loaded} DFlash weights (no prefix), missing={len(result.missing_keys)}')
else:
    print('WARNING: No checkpoint files found, using random weights')

model.eval()
validator = HFARValidation(model, tokenizer)

ds = load_dataset('/hf-local/HuggingFaceH4/mt_bench_prompts')['train']
num_samples = min(${NUM_AR_SAMPLES}, len(ds))

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

###################################################################################################
