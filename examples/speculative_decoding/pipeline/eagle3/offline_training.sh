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

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source ${SCRIPT_DIR}/../../service_utils.sh 2>/dev/null || true

pip install -r modules/Model-Optimizer/examples/speculative_decoding/requirements.txt
pip install huggingface-hub>=1.2.1
export PATH=$PATH:/workspace/.local/bin

# Patch load_vlm_or_llm in installed modelopt to handle VLMs with text_config
# (e.g. mistral3 / Ministral-3-8B) that are not caught by the "vl" name check.
# For offline training, routes them through FakeBaseModel which handles VLM weight layouts.
python3 << 'PYEOF' || true
import site, os
for d in site.getsitepackages():
    path = os.path.join(d, 'modelopt', 'torch', 'speculative', 'utils.py')
    if not os.path.exists(path):
        continue
    with open(path) as f:
        c = f.read()
    old = (
        '    if "vl" in model_config.model_type.lower():\n'
        '        model_cls = transformers.AutoModelForVision2Seq\n'
        '    else:\n'
        '        model_cls = transformers.AutoModelForCausalLM\n'
    )
    if old not in c:
        print('load_vlm_or_llm: VLM patch already applied or pattern not found')
        break
    new = (
        '    # Detect VLMs: "vl" in model_type OR has text_config/llm_config (e.g. mistral3)\n'
        '    _is_vlm = "vl" in model_config.model_type.lower() or any(\n'
        '        getattr(model_config, _a, None) is not None for _a in ["text_config", "llm_config"]\n'
        '    )\n'
        '    if _is_vlm and use_offline_training:\n'
        '        from modelopt.torch.speculative.plugins.modeling_fakebase import FakeBaseModel\n'
        '        return FakeBaseModel.from_source(model_name_or_path, trust_remote_code=trust_remote_code)\n'
        '    if _is_vlm:\n'
        '        model_cls = transformers.AutoModelForVision2Seq\n'
        '    else:\n'
        '        model_cls = transformers.AutoModelForCausalLM\n'
    )
    with open(path, 'w') as f:
        f.write(c.replace(old, new))
    print('Patched load_vlm_or_llm: added VLM text_config detection for offline training')
    break
PYEOF

# Patch FakeBaseModel._load_weights in installed modelopt to fall back to
# consolidated.safetensors when an HF shard file is missing.
# Handles Ministral-3-8B-Instruct-2512-BF16 which is missing shard 1 but has
# a complete consolidated.safetensors with Mistral native key names.
python3 << 'PYEOF' || true
import site, os
for d in site.getsitepackages():
    path = os.path.join(d, 'modelopt', 'torch', 'speculative', 'plugins', 'modeling_fakebase.py')
    if not os.path.exists(path):
        continue
    with open(path) as f:
        c = f.read()
    old = (
        '        lm_head_state = safetensors_load_file(lm_head_path, device="cpu")\n'
        '        embed_tokens_state = safetensors_load_file(embed_tokens_path, device="cpu")\n'
        '\n'
        '        return lm_head_state[lm_head_key], embed_tokens_state[embed_tokens_key]\n'
    )
    if old not in c:
        print('modeling_fakebase.py: consolidated fallback already applied or pattern not found')
        break
    new = (
        '        def _load_with_consolidated_fallback(shard_path, key, role):\n'
        '            try:\n'
        '                return safetensors_load_file(shard_path, device="cpu")[key]\n'
        '            except FileNotFoundError:\n'
        '                _aliases = {"embed_tokens": ["tok_embeddings.weight"], "lm_head": ["output.weight"]}\n'
        '                _consolidated = os.path.join(os.path.dirname(shard_path), "consolidated.safetensors")\n'
        '                if os.path.isfile(_consolidated):\n'
        '                    _state = safetensors_load_file(_consolidated, device="cpu")\n'
        '                    for _alias in _aliases.get(role, []):\n'
        '                        if _alias in _state:\n'
        '                            return _state[_alias]\n'
        '                raise\n'
        '\n'
        '        lm_head_w = _load_with_consolidated_fallback(lm_head_path, lm_head_key, "lm_head")\n'
        '        embed_tokens_w = _load_with_consolidated_fallback(embed_tokens_path, embed_tokens_key, "embed_tokens")\n'
        '\n'
        '        return lm_head_w, embed_tokens_w\n'
    )
    with open(path, 'w') as f:
        f.write(c.replace(old, new))
    print('Patched FakeBaseModel._load_weights: added consolidated.safetensors fallback')
    break
PYEOF

###################################################################################################

set -eo pipefail

# Parse old-style CLI args; translate to OmegaConf key=value for launch_train.sh.
# This allows existing yamls (which use --offline-data, --lr, etc.) to keep working
# as launch_train.sh migrated from per-flag CLI to --config yaml + dotlist overrides.
OFFLINE_DATA=""
DATA_PATH="None"
MODE="eagle3"
NUM_EPOCHS=1
LR=""
SAVE_STEPS=""
OUTPUT_DIR="/scratchspace/eagle3"
TRAIN_BS=""
TRAINING_SEQ_LEN=""
DISABLE_TQDM=""
AR_VALIDATE_STEPS=""
TRUST_REMOTE_CODE=false
EXTRA_ARGS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --offline-data)       shift; OFFLINE_DATA="$1" ;;
    --data_path)          shift; DATA_PATH="$1" ;;
    --mode)               shift; MODE="$1" ;;
    --num_epochs)         shift; NUM_EPOCHS="$1" ;;
    --lr)                 shift; LR="$1" ;;
    --save_steps)         shift; SAVE_STEPS="$1" ;;
    --output_dir)         shift; OUTPUT_DIR="$1" ;;
    --train_bs)           shift; TRAIN_BS="$1" ;;
    --training_seq_len)   shift; TRAINING_SEQ_LEN="$1" ;;
    --eagle_config)       shift; ;;  # deprecated — ignore
    --disable_tqdm)       shift; DISABLE_TQDM="$1" ;;
    --ar_validate_steps)  shift; AR_VALIDATE_STEPS="$1" ;;
    --trust_remote_code)  TRUST_REMOTE_CODE=true ;;
    *) EXTRA_ARGS+=("$1") ;;
  esac
  shift
done

OVERRIDES=(
    "model.model_name_or_path=${HF_MODEL_CKPT}"
    "model.trust_remote_code=${TRUST_REMOTE_CODE}"
    "training.output_dir=${OUTPUT_DIR}"
    "training.mode=${MODE}"
    "training.num_train_epochs=${NUM_EPOCHS}"
)
[ -n "$OFFLINE_DATA" ]      && OVERRIDES+=("data.offline_data_path=${OFFLINE_DATA}")
[ -n "$LR" ]                && OVERRIDES+=("training.learning_rate=${LR}")
[ -n "$SAVE_STEPS" ]        && OVERRIDES+=("training.save_steps=${SAVE_STEPS}")
[ -n "$TRAIN_BS" ]          && OVERRIDES+=("training.per_device_train_batch_size=${TRAIN_BS}")
[ -n "$TRAINING_SEQ_LEN" ]  && OVERRIDES+=("training.training_seq_len=${TRAINING_SEQ_LEN}")
[ -n "$DISABLE_TQDM" ]      && OVERRIDES+=("training.disable_tqdm=${DISABLE_TQDM}")
[ -n "$AR_VALIDATE_STEPS" ] && OVERRIDES+=("training.ar_validate_steps=${AR_VALIDATE_STEPS}")
[ "$DATA_PATH" != "None" ] && [ -n "$DATA_PATH" ] && OVERRIDES+=("data.data_path=${DATA_PATH}")

bash modules/Model-Optimizer/examples/speculative_decoding/launch_train.sh \
    --config modules/Model-Optimizer/modelopt_recipes/general/speculative_decoding/eagle3.yaml \
    "${OVERRIDES[@]}" \
    "${EXTRA_ARGS[@]}"

python modules/Model-Optimizer/examples/speculative_decoding/scripts/export_hf_checkpoint.py \
    --model_path "${OUTPUT_DIR}" \
    --export_path /scratchspace/export \
    --trust_remote_code

# Fix use_cache: null → true in exported config (newer huggingface_hub rejects None for bool fields)
python3 -c "
import json, pathlib
p = pathlib.Path('/scratchspace/export/config.json')
if p.exists():
    c = json.loads(p.read_text())
    if c.get('use_cache') is None:
        c['use_cache'] = True
        p.write_text(json.dumps(c, indent=4))
        print('Fixed use_cache=null -> true in exported config.json')
" || true

###################################################################################################
