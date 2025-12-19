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

# =============================================================================
# One-Button Dataset Generation for QAD Training
# =============================================================================
#
# Downloads and preprocesses OpenScience + Nemotron-v2 datasets for QAD.
# Creates a combined datablend JSON ready for training.
#
# USAGE:
#   bash generate_dataset.sh --output-dir /path/to/datasets \
#                            --mlm-path /path/to/Megatron-LM \
#                            --tokenizer Qwen/Qwen3-30B-A3B-Instruct-2507
#
# REQUIRED ARGUMENTS:
#   --output-dir   Base output directory for datasets
#   --mlm-path     Path to Megatron-LM directory
#   --tokenizer    HuggingFace tokenizer (e.g., Qwen/Qwen3-30B-A3B-Instruct-2507)
#
# OPTIONAL ARGUMENTS:
#   --sample-percent    Percentage of Nemotron-v2 to use (default: 30)
#   --include-reasoning Include chain-of-thought for Thinking models
#   --workers           Parallel workers for preprocessing (default: 32)
#
# REQUIREMENTS:
#   - HuggingFace access to nvidia/Nemotron-Post-Training-Dataset-v2
#   - Run: huggingface-cli login
#
# =============================================================================

set -e

# =============================================================================
# Argument Parsing
# =============================================================================
OUTPUT_DIR=""
MLM_DIR=""
TOKENIZER=""
SAMPLE_PERCENT=30
INCLUDE_REASONING=false
WORKERS=32

while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --mlm-path)
            MLM_DIR="$2"
            shift 2
            ;;
        --tokenizer)
            TOKENIZER="$2"
            shift 2
            ;;
        --sample-percent)
            SAMPLE_PERCENT="$2"
            shift 2
            ;;
        --include-reasoning)
            INCLUDE_REASONING=true
            shift
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: bash generate_dataset.sh --output-dir <path> --mlm-path <path> --tokenizer <model>"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$OUTPUT_DIR" ] || [ -z "$MLM_DIR" ] || [ -z "$TOKENIZER" ]; then
    echo "Error: Missing required arguments"
    echo ""
    echo "Usage: bash generate_dataset.sh --output-dir <path> --mlm-path <path> --tokenizer <model>"
    echo ""
    echo "Required:"
    echo "  --output-dir   Base output directory for datasets"
    echo "  --mlm-path     Path to Megatron-LM directory"
    echo "  --tokenizer    HuggingFace tokenizer (e.g., Qwen/Qwen3-30B-A3B-Instruct-2507)"
    echo ""
    echo "Optional:"
    echo "  --sample-percent    Percentage of data to use (default: 30)"
    echo "  --include-reasoning Include chain-of-thought (for Thinking models)"
    echo "  --workers           Parallel workers (default: 32)"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Build suffix based on options
if [ "$INCLUDE_REASONING" = true ]; then
    SUFFIX="${SAMPLE_PERCENT}pct_cot_chat"
    REASONING_FLAG="--include-reasoning"
else
    SUFFIX="${SAMPLE_PERCENT}pct_chat"
    REASONING_FLAG=""
fi

# Tokenizer settings for Megatron preprocessing
TOKENIZER_TYPE="HuggingFaceTokenizer"

echo "=============================================="
echo "QAD Dataset Generation"
echo "=============================================="
echo "Output:       ${OUTPUT_DIR}"
echo "MLM path:     ${MLM_DIR}"
echo "Tokenizer:    ${TOKENIZER}"
echo "Sample:       ${SAMPLE_PERCENT}%"
echo "Reasoning:    ${INCLUDE_REASONING}"
echo "Workers:      ${WORKERS}"
echo "=============================================="

# Create directories
mkdir -p "${OUTPUT_DIR}"

# =============================================================================
# Helper Function: Preprocess JSONL to Megatron format
# =============================================================================
preprocess_to_megatron() {
    local input_file="$1"
    local output_prefix="$2"
    
    if [ -f "${input_file}" ]; then
        echo "  Preprocessing: $(basename ${input_file})"
        python "${MLM_DIR}/tools/preprocess_data.py" \
            --input "${input_file}" \
            --output-prefix "${output_prefix}" \
            --tokenizer-type "${TOKENIZER_TYPE}" \
            --tokenizer-model "${TOKENIZER}" \
            --append-eod \
            --workers "${WORKERS}" \
            --json-keys text
        return 0
    else
        echo "  Warning: File not found: ${input_file}"
        return 1
    fi
}

# =============================================================================
# Step 1: Download OpenScience
# =============================================================================
echo ""
echo "=== Step 1: Downloading Datasets ==="

echo "[1/2] Downloading OpenScience..."
python "${SCRIPT_DIR}/download_dataset.py" \
    --dataset openscience \
    --output-dir "${OUTPUT_DIR}" \
    --datablend-dir "${OUTPUT_DIR}" \
    --tokenizer "${TOKENIZER}"

echo "[2/2] Downloading Nemotron-v2 @ ${SAMPLE_PERCENT}%..."
python "${SCRIPT_DIR}/download_dataset.py" \
    --dataset nemotron-v2 \
    --output-dir "${OUTPUT_DIR}" \
    --datablend-dir "${OUTPUT_DIR}" \
    --sample-percent "${SAMPLE_PERCENT}" \
    ${REASONING_FLAG} \
    --tokenizer "${TOKENIZER}"

# =============================================================================
# Step 2: Preprocess OpenScience
# =============================================================================
echo ""
echo "=== Step 2: Preprocessing to Megatron Format ==="

OS_INPUT="${OUTPUT_DIR}/openscience_splits"
OS_OUTPUT="${OUTPUT_DIR}/openscience_splits_preprocessed"
mkdir -p "${OS_OUTPUT}"

echo "[1/5] Processing OpenScience..."
preprocess_to_megatron "${OS_INPUT}/openscience_chat_train.jsonl" "${OS_OUTPUT}/openscience_chat_train"
preprocess_to_megatron "${OS_INPUT}/openscience_chat_validation.jsonl" "${OS_OUTPUT}/openscience_chat_validation" || true
preprocess_to_megatron "${OS_INPUT}/openscience_chat_test.jsonl" "${OS_OUTPUT}/openscience_chat_test" || true

# =============================================================================
# Step 3: Preprocess Nemotron-v2 (all splits)
# =============================================================================
NV2_INPUT="${OUTPUT_DIR}/nemotron_v2"
NV2_OUTPUT="${OUTPUT_DIR}/nemotron_v2_preprocessed"

STEP=2
for split in code math stem chat; do
    echo "[${STEP}/5] Processing Nemotron-v2 ${split}..."
    SPLIT_DIR="${NV2_OUTPUT}/${split}"
    mkdir -p "${SPLIT_DIR}"
    
    preprocess_to_megatron \
        "${NV2_INPUT}/${split}/${split}_${SUFFIX}_train.jsonl" \
        "${SPLIT_DIR}/${split}_${SUFFIX}_train"
    
    preprocess_to_megatron \
        "${NV2_INPUT}/${split}/${split}_${SUFFIX}_validation.jsonl" \
        "${SPLIT_DIR}/${split}_${SUFFIX}_validation" || true
    
    preprocess_to_megatron \
        "${NV2_INPUT}/${split}/${split}_${SUFFIX}_test.jsonl" \
        "${SPLIT_DIR}/${split}_${SUFFIX}_test" || true
    
    STEP=$((STEP + 1))
done

# =============================================================================
# Step 4: Create Combined Datablend JSON
# =============================================================================
echo ""
echo "=== Step 3: Creating Combined Datablend ==="

DATABLEND_FILE="${OUTPUT_DIR}/datablend_combined.json"

cat > "${DATABLEND_FILE}" << EOF
{
    "train": [
        0.3, "${NV2_OUTPUT}/code/code_${SUFFIX}_train_text_document",
        0.2, "${NV2_OUTPUT}/math/math_${SUFFIX}_train_text_document",
        0.2, "${NV2_OUTPUT}/stem/stem_${SUFFIX}_train_text_document",
        0.1, "${NV2_OUTPUT}/chat/chat_${SUFFIX}_train_text_document",
        0.2, "${OS_OUTPUT}/openscience_chat_train_text_document"
    ],
    "valid": [
        0.5, "${NV2_OUTPUT}/stem/stem_${SUFFIX}_validation_text_document",
        0.5, "${OS_OUTPUT}/openscience_chat_validation_text_document"
    ],
    "test": [
        0.5, "${NV2_OUTPUT}/stem/stem_${SUFFIX}_test_text_document",
        0.5, "${OS_OUTPUT}/openscience_chat_test_text_document"
    ]
}
EOF

echo "Created: ${DATABLEND_FILE}"

# =============================================================================
# Done
# =============================================================================
echo ""
echo "=============================================="
echo "Dataset generation complete!"
echo "=============================================="
echo ""
echo "Output structure:"
echo "  ${OUTPUT_DIR}/"
echo "  ├── openscience_splits/              # Raw JSONL"
echo "  ├── openscience_splits_preprocessed/ # Megatron format"
echo "  ├── nemotron_v2/                     # Raw JSONL"
echo "  ├── nemotron_v2_preprocessed/        # Megatron format"
echo "  └── datablend_combined.json          # Combined config"
echo ""
echo "Dataset weights (train):"
echo "  - 30% Nemotron-v2 code"
echo "  - 20% Nemotron-v2 math"
echo "  - 20% Nemotron-v2 stem"
echo "  - 10% Nemotron-v2 chat"
echo "  - 20% OpenScience"
echo ""
echo "Next steps:"
echo "  1. Set in your config:"
echo "     export BLEND_PATH=\"${DATABLEND_FILE}\""
echo ""
echo "  2. Run training:"
echo "     sbatch sbatch_qad.sh --config configs/your-config.conf"
echo "=============================================="
