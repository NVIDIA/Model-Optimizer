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

# MT-Bench per-category AR evaluation for DFlash checkpoints.
# Evaluates the latest checkpoint using ModelOpt's pseudo_speculative_generate
# with online (context-dependent) ground truth validation.
#
# Required env vars:
#   HF_MODEL_CKPT — path to the target HuggingFace model
#
# Args:
#   --ckpt_dir      Path to directory containing checkpoint-* subdirs
#   --block_size    Block size for DFlash (default: 16)
#   --num_layers    Number of draft decoder layers (default: 5)
#   --mask_token_id Mask token ID (default: auto-detect from checkpoint)
#   --osl           Output sequence length (default: 512)
#   --steps         Draft steps per block (default: block_size-1)
#   --online        Use online validation (default: true)

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source ${SCRIPT_DIR}/../service_utils.sh

pip install -r modules/Model-Optimizer/examples/speculative_decoding/requirements.txt 2>&1 | tail -3

# Overlay DFlash code
for INSTALL_PATH in $(python3 -c "
import modelopt, os, site
paths = set()
paths.add(os.path.dirname(modelopt.__file__))
for sp in site.getsitepackages():
    p = os.path.join(sp, 'modelopt')
    if os.path.isdir(p): paths.add(p)
for p in paths: print(p)
"); do
    cp -rf modules/Model-Optimizer/modelopt/torch/speculative/dflash ${INSTALL_PATH}/torch/speculative/ 2>/dev/null || true
    cp -f modules/Model-Optimizer/modelopt/torch/speculative/plugins/hf_dflash.py ${INSTALL_PATH}/torch/speculative/plugins/ 2>/dev/null || true
    cp -f modules/Model-Optimizer/modelopt/torch/speculative/plugins/__init__.py ${INSTALL_PATH}/torch/speculative/plugins/ 2>/dev/null || true
    cp -f modules/Model-Optimizer/modelopt/torch/speculative/config.py ${INSTALL_PATH}/torch/speculative/ 2>/dev/null || true
    cp -f modules/Model-Optimizer/modelopt/torch/speculative/mode.py ${INSTALL_PATH}/torch/speculative/ 2>/dev/null || true
    cp -f modules/Model-Optimizer/modelopt/torch/speculative/utils.py ${INSTALL_PATH}/torch/speculative/ 2>/dev/null || true
done

# Parse args
CKPT_DIR=""
BLOCK_SIZE=16
NUM_LAYERS=5
MASK_TOKEN_ID=""
OSL=512
STEPS=""
ONLINE=true

while [ $# -gt 0 ]; do
  case "$1" in
    --ckpt_dir)       shift; CKPT_DIR="$1" ;;
    --block_size)     shift; BLOCK_SIZE="$1" ;;
    --num_layers)     shift; NUM_LAYERS="$1" ;;
    --mask_token_id)  shift; MASK_TOKEN_ID="$1" ;;
    --osl)            shift; OSL="$1" ;;
    --steps)          shift; STEPS="$1" ;;
    --online)         shift; ONLINE="$1" ;;
    *) ;;
  esac
  shift
done

if [ -z "$STEPS" ]; then
    STEPS=$((BLOCK_SIZE - 1))
fi

MODEL=${HF_MODEL_CKPT}

echo "=== DFlash MT-Bench AR Evaluation ==="
echo "Checkpoint dir: ${CKPT_DIR}"
echo "Model: ${MODEL}"
echo "Block size: ${BLOCK_SIZE}, Layers: ${NUM_LAYERS}"
echo "OSL: ${OSL}, Steps: ${STEPS}, Online: ${ONLINE}"

# Find latest checkpoint
LAST_CKPT=$(ls -d ${CKPT_DIR}/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
if [ -z "$LAST_CKPT" ]; then
    # Check for top-level model
    if [ -f "${CKPT_DIR}/model.safetensors" ]; then
        LAST_CKPT=${CKPT_DIR}
    else
        echo "ERROR: No checkpoints found in ${CKPT_DIR}"
        exit 1
    fi
fi
echo "Evaluating: ${LAST_CKPT}"

CUDA_VISIBLE_DEVICES=0 python3 -c "
import torch, glob, os, json
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file
from datasets import load_dataset
from collections import defaultdict
import modelopt.torch.opt as mto
import modelopt.torch.speculative as mtsp
from modelopt.torch.speculative.plugins.transformers import HFARValidation

mto.enable_huggingface_checkpointing()

MODEL = '${MODEL}'
CKPT_PATH = '${LAST_CKPT}'
BLOCK_SIZE = ${BLOCK_SIZE}
NUM_LAYERS = ${NUM_LAYERS}
MASK_TOKEN_ID_STR = '${MASK_TOKEN_ID}'
OSL = ${OSL}
STEPS = ${STEPS}
ONLINE = '${ONLINE}' == 'true'

# Auto-detect mask_token_id from checkpoint config
MASK_TOKEN_ID = int(MASK_TOKEN_ID_STR) if MASK_TOKEN_ID_STR else None
if MASK_TOKEN_ID is None:
    cfg_path = os.path.join(CKPT_PATH, 'config.json')
    if os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            ckpt_cfg = json.load(f)
            dflash_cfg = ckpt_cfg.get('dflash_config', {})
            MASK_TOKEN_ID = dflash_cfg.get('mask_token_id')
    if MASK_TOKEN_ID is None:
        MASK_TOKEN_ID = 151669  # default for Qwen3
        print(f'WARNING: Could not auto-detect mask_token_id, using default {MASK_TOKEN_ID}')
print(f'Using mask_token_id={MASK_TOKEN_ID}')

# Use flash_attention_2 if available
try:
    import flash_attn
    ATTN_IMPL = 'flash_attention_2'
except ImportError:
    ATTN_IMPL = 'sdpa'
print(f'Using attn_implementation={ATTN_IMPL}')

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

# Load MT-Bench by category
ds = load_dataset('/hf-local/HuggingFaceH4/mt_bench_prompts')['train']
cat_samples = defaultdict(list)
for i in range(len(ds)):
    cat = ds[i].get('category', 'unknown')
    cat_samples[cat].append(ds[i]['prompt'][0])
categories = sorted(cat_samples.keys())
print(f'Categories: {categories}')
for c in categories:
    print(f'  {c}: {len(cat_samples[c])} samples')

# Load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map={'': 0}, trust_remote_code=True,
    attn_implementation=ATTN_IMPL,
)
config = {
    'dflash_block_size': BLOCK_SIZE,
    'dflash_architecture_config': {
        'num_hidden_layers': NUM_LAYERS,
        'mask_token_id': MASK_TOKEN_ID,
        '_attn_implementation': ATTN_IMPL,
    },
    'dflash_use_torch_compile': False,
}
mtsp.convert(model, [('dflash', config)])

# Load weights
sf_files = sorted(glob.glob(os.path.join(CKPT_PATH, 'model*.safetensors')))
if sf_files:
    state = {}
    for f in sf_files:
        state.update(load_file(f))
    dflash_keys = {k: v for k, v in state.items() if 'dflash_module' in k}
    if dflash_keys:
        model.load_state_dict(dflash_keys, strict=False)
        print(f'Loaded {len(dflash_keys)} DFlash weights (with prefix)')
    else:
        model.dflash_module.load_state_dict(state, strict=False)
        print(f'Loaded {len(state)} DFlash weights (no prefix)')
else:
    print('ERROR: No safetensors found')
    exit(1)

model.eval()
validator = HFARValidation(model, tokenizer)

# Evaluate per category
cat_ars = {}
all_ars = []
for cat in categories:
    ars = []
    for prompt in cat_samples[cat]:
        chat = [{'role': 'user', 'content': prompt}]
        text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(text, return_tensors='pt').input_ids.cuda()
        try:
            if ONLINE:
                _, ar = validator.validate_online(osl=OSL, input_ids=input_ids, steps=STEPS)
            else:
                _, ar = validator.validate(osl=OSL, input_ids=input_ids, steps=STEPS)
            ars.append(ar)
            all_ars.append(ar)
        except Exception as e:
            print(f'  ERROR [{cat}]: {e}')
    cat_ars[cat] = sum(ars) / len(ars) if ars else 0.0

avg_all = sum(all_ars) / len(all_ars) if all_ars else 0.0
mode_str = 'online' if ONLINE else 'fixed GT'

print(f'\n=== Results (OSL={OSL}, steps={STEPS}, {mode_str}) ===')
for c in categories:
    print(f'  {c:>12}: {cat_ars[c]:.4f}')
print(f'{\"ALL\":>14}: {avg_all:.4f}')
"

report_result "PASS: MT-Bench AR evaluation"
