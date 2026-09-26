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

# Download and preprocess OpenScience + Nemotron-v2 datasets for QAD training.
# Usage: bash generate_dataset.sh --output-dir <path> --mlm-path <path> --tokenizer <model>

set -e

# Defaults
OUTPUT_DIR="" MLM_DIR="" TOKENIZER="" SAMPLE_PERCENT=30 INCLUDE_REASONING=false WORKERS=32

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir) OUTPUT_DIR="$2"; shift 2;;
        --mlm-path) MLM_DIR="$2"; shift 2;;
        --tokenizer) TOKENIZER="$2"; shift 2;;
        --sample-percent) SAMPLE_PERCENT="$2"; shift 2;;
        --include-reasoning) INCLUDE_REASONING=true; shift;;
        --workers) WORKERS="$2"; shift 2;;
        *) echo "Unknown: $1"; exit 1;;
    esac
done

# Validate
if [ -z "$OUTPUT_DIR" ] || [ -z "$MLM_DIR" ] || [ -z "$TOKENIZER" ]; then
    echo "Usage: bash generate_dataset.sh --output-dir <path> --mlm-path <path> --tokenizer <model>"
    echo "Optional: --sample-percent N --include-reasoning --workers N"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUFFIX="${SAMPLE_PERCENT}pct$( [ "$INCLUDE_REASONING" = true ] && echo "_cot" )_chat"
REASONING_FLAG=$( [ "$INCLUDE_REASONING" = true ] && echo "--include-reasoning" )

echo "=== QAD Dataset Generation ==="
echo "Output: $OUTPUT_DIR | Tokenizer: $TOKENIZER | Sample: ${SAMPLE_PERCENT}%"

# Helper: preprocess JSONL to Megatron format
preprocess() {
    [ -f "$1" ] && python "$MLM_DIR/tools/preprocess_data.py" \
        --input "$1" --output-prefix "$2" \
        --tokenizer-type HuggingFaceTokenizer --tokenizer-model "$TOKENIZER" \
        --append-eod --workers "$WORKERS" --json-keys text
}

# Step 1: Download
echo -e "\n=== Downloading ==="
python "$SCRIPT_DIR/download_dataset.py" --dataset openscience --output-dir "$OUTPUT_DIR" --tokenizer "$TOKENIZER"
python "$SCRIPT_DIR/download_dataset.py" --dataset nemotron-v2 --output-dir "$OUTPUT_DIR" \
    --sample-percent "$SAMPLE_PERCENT" $REASONING_FLAG --tokenizer "$TOKENIZER"

# Step 2: Preprocess
echo -e "\n=== Preprocessing ==="
OS_IN="$OUTPUT_DIR/openscience_splits" OS_OUT="$OUTPUT_DIR/openscience_splits_preprocessed"
NV_IN="$OUTPUT_DIR/nemotron_v2" NV_OUT="$OUTPUT_DIR/nemotron_v2_preprocessed"
mkdir -p "$OS_OUT"

for s in train validation test; do preprocess "$OS_IN/openscience_chat_$s.jsonl" "$OS_OUT/openscience_chat_$s" || true; done

for split in code math stem chat; do
    mkdir -p "$NV_OUT/$split"
    for s in train validation test; do
        preprocess "$NV_IN/$split/${split}_${SUFFIX}_$s.jsonl" "$NV_OUT/$split/${split}_${SUFFIX}_$s" || true
    done
done

# Step 3: Create combined datablend
BLEND="$OUTPUT_DIR/datablend_combined.json"
cat > "$BLEND" << EOF
{
    "train": [
        0.3, "$NV_OUT/code/code_${SUFFIX}_train_text_document",
        0.2, "$NV_OUT/math/math_${SUFFIX}_train_text_document",
        0.2, "$NV_OUT/stem/stem_${SUFFIX}_train_text_document",
        0.1, "$NV_OUT/chat/chat_${SUFFIX}_train_text_document",
        0.2, "$OS_OUT/openscience_chat_train_text_document"
    ],
    "valid": [
        0.5, "$NV_OUT/stem/stem_${SUFFIX}_validation_text_document",
        0.5, "$OS_OUT/openscience_chat_validation_text_document"
    ],
    "test": [
        0.5, "$NV_OUT/stem/stem_${SUFFIX}_test_text_document",
        0.5, "$OS_OUT/openscience_chat_test_text_document"
    ]
}
EOF

echo -e "\n=== Done! ==="
echo "Datablend: $BLEND"
echo "Set BLEND_PATH in your config and run: sbatch sbatch_qad.sh --config <config>"
