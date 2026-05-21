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

# HF-based hidden-states dump for offline speculative-decoding training.
#
# Wraps examples/speculative_decoding/collect_hidden_states/compute_hidden_states_hf.py
# with one quality-of-life addition: if --input-data lacks a ``conversation_id``
# field (compute_hidden_states_hf.py asserts on it), a tagged copy is
# materialized under /scratchspace and substituted in-place. This lets the
# packaged ``examples/dataset/synthetic_conversations_1k.jsonl`` flow straight
# into the dump task without an out-of-band preprocessing step.
#
# All args are passed through to compute_hidden_states_hf.py except
# --input-data, which may be rewritten if tagging is needed.
#
# Usage from YAML:
#   script: common/specdec/dump_hidden_states_hf.sh
#   args:
#     - --model <hf_model_path>
#     - --input-data <jsonl>
#     - --output-dir /scratchspace/hidden_states
#     - --aux-layers dflash
#     - --answer-only-loss
#     - --chat-template <jinja_path>
#     - --debug-max-num-conversations 200

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source ${SCRIPT_DIR}/../service_utils.sh

pip install -r modules/Model-Optimizer/examples/speculative_decoding/requirements.txt
pip install huggingface-hub>=1.2.1
export PATH=$PATH:/workspace/.local/bin

###################################################################################################

trap 'error_handler $0 $LINENO' ERR
trap 'exit_handler' EXIT

# Extract --input-data so we can tag it if needed. Other args pass through unchanged.
ARGS=("$@")
INPUT_DATA=""
for ((i=0; i<${#ARGS[@]}; i++)); do
    if [[ "${ARGS[i]}" == "--input-data" ]]; then
        INPUT_DATA="${ARGS[i+1]}"
        INPUT_IDX=$i
        break
    fi
done

if [ -z "$INPUT_DATA" ]; then
    echo "ERROR: --input-data is required" >&2
    exit 1
fi

# Probe first non-empty line for conversation_id / uuid. If absent, materialize
# a tagged copy under /scratchspace and rewrite the arg.
NEEDS_TAGGING=$(python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        print('false' if (entry.get('conversation_id') or entry.get('uuid')) else 'true')
        break
" "$INPUT_DATA")

if [[ "$NEEDS_TAGGING" == "true" ]]; then
    mkdir -p /scratchspace
    TAGGED_DATA="/scratchspace/$(basename ${INPUT_DATA%.jsonl})_tagged.jsonl"
    python3 -c "
import json, sys
with open(sys.argv[1]) as src, open(sys.argv[2], 'w') as dst:
    for i, line in enumerate(src):
        if not line.strip():
            continue
        entry = json.loads(line)
        entry.setdefault('conversation_id', f'{i:08d}')
        dst.write(json.dumps(entry) + '\n')
" "$INPUT_DATA" "$TAGGED_DATA"
    echo "Tagged ${INPUT_DATA} -> ${TAGGED_DATA}"
    ARGS[$((INPUT_IDX + 1))]="$TAGGED_DATA"
fi

set -x
python3 modules/Model-Optimizer/examples/speculative_decoding/collect_hidden_states/compute_hidden_states_hf.py "${ARGS[@]}"
set +x
