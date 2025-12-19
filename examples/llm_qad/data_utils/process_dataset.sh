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
# Unified Dataset Preprocessor for QAD Training
# =============================================================================
#
# Converts JSONL datasets to Megatron format (.bin/.idx files).
# Supports: OpenScience, Nemotron-v2
#
# USAGE:
#   # OpenScience (flat structure)
#   bash process_dataset.sh --dataset openscience \
#       --output-dir /path/to/preprocessed \
#       --input-dir /path/to/openscience_splits \
#       --mlm-path /path/to/Megatron-LM \
#       --tokenizer Qwen/Qwen3-8B
#
#   # Nemotron-v2 (with splits)
#   bash process_dataset.sh --dataset nemotron-v2 \
#       --output-dir /path/to/preprocessed \
#       --input-dir /path/to/nemotron_v2 \
#       --mlm-path /path/to/Megatron-LM \
#       --tokenizer Qwen/Qwen3-8B \
#       --split stem \
#       --suffix 30pct_chat
#
# REQUIRED ARGUMENTS:
#   --dataset       Dataset type: openscience, nemotron-v2
#   --output-dir    Output directory for preprocessed files
#   --input-dir     Input directory with JSONL files
#   --mlm-path      Path to Megatron-LM directory
#   --tokenizer     HuggingFace tokenizer model
#
# OPTIONAL ARGUMENTS:
#   --split         Split name for nemotron-v2: stem, math, code, chat (default: stem)
#   --suffix        Suffix for file naming (default: chat)
#   --workers       Number of parallel workers (default: 32)
#   --datablend-dir Directory for datablend configs (default: parent of output-dir)
#
# =============================================================================

set -e

# -----------------------------------------------------------------------------
# Argument Parsing
# -----------------------------------------------------------------------------
DATASET=""
OUTPUT_DIR=""
INPUT_DIR=""
MLM_DIR=""
TOKENIZER_MODEL=""
SPLIT="stem"
SUFFIX="chat"
WORKERS=32
DATABLEND_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --input-dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        --mlm-path)
            MLM_DIR="$2"
            shift 2
            ;;
        --tokenizer)
            TOKENIZER_MODEL="$2"
            shift 2
            ;;
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --suffix)
            SUFFIX="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --datablend-dir)
            DATABLEND_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
if [ -z "$DATASET" ]; then
    echo "Error: --dataset is required (openscience, nemotron-v2)"
    exit 1
fi

if [ -z "$OUTPUT_DIR" ]; then
    echo "Error: --output-dir is required"
    exit 1
fi

if [ -z "$INPUT_DIR" ]; then
    echo "Error: --input-dir is required"
    exit 1
fi

if [ -z "$MLM_DIR" ]; then
    echo "Error: --mlm-path is required"
    exit 1
fi

if [ -z "$TOKENIZER_MODEL" ]; then
    echo "Error: --tokenizer is required"
    exit 1
fi

# Set default datablend dir
if [ -z "$DATABLEND_DIR" ]; then
    DATABLEND_DIR="$(dirname "$OUTPUT_DIR")"
fi

# Tokenizer settings
TOKENIZER_TYPE="HuggingFaceTokenizer"

# -----------------------------------------------------------------------------
# Dataset-specific configuration
# -----------------------------------------------------------------------------
case "$DATASET" in
    openscience)
        # OpenScience: flat structure, files named openscience_<suffix>_<split>.jsonl
        if [ -n "$SUFFIX" ]; then
            FILE_PREFIX="openscience_${SUFFIX}"
        else
            FILE_PREFIX="openscience"
        fi
        WORK_DIR="${OUTPUT_DIR}"
        INPUT_SUBDIR="${INPUT_DIR}"
        BLEND_NAME="datablend_openscience_${SUFFIX}"
        ;;
    
    nemotron-v2)
        # Nemotron-v2: organized by split, files named <split>_<suffix>_<split>.jsonl
        # Normalize suffix (handle both 30pctcot and 30pct_cot)
        SUFFIX=$(echo "$SUFFIX" | sed 's/pctcot/pct_cot/g')
        FILE_PREFIX="${SPLIT}_${SUFFIX}"
        WORK_DIR="${OUTPUT_DIR}/${SPLIT}"
        INPUT_SUBDIR="${INPUT_DIR}/${SPLIT}"
        BLEND_NAME="datablend_nemotron_v2_${SPLIT}_${SUFFIX}"
        ;;
    
    *)
        echo "Error: Unknown dataset '$DATASET'. Supported: openscience, nemotron-v2"
        exit 1
        ;;
esac

mkdir -p "${WORK_DIR}"
mkdir -p "${DATABLEND_DIR}"

# -----------------------------------------------------------------------------
# Processing
# -----------------------------------------------------------------------------
echo "=========================================="
echo "Preprocessing Dataset: ${DATASET}"
echo "=========================================="
echo "File prefix: ${FILE_PREFIX}"
echo "Tokenizer:   ${TOKENIZER_MODEL}"
echo "MLM Path:    ${MLM_DIR}"
echo "Input:       ${INPUT_SUBDIR}"
echo "Output:      ${WORK_DIR}"
echo "Workers:     ${WORKERS}"
echo "=========================================="

# Function to preprocess a single split
preprocess_split() {
    local split_type=$1  # train, validation, test
    local input_file="${INPUT_SUBDIR}/${FILE_PREFIX}_${split_type}.jsonl"
    local output_prefix="${WORK_DIR}/${FILE_PREFIX}_${split_type}"
    
    if [ -f "${input_file}" ]; then
        echo "Processing ${split_type}: ${input_file}"
        python "${MLM_DIR}/tools/preprocess_data.py" \
            --input "${input_file}" \
            --output-prefix "${output_prefix}" \
            --tokenizer-type "${TOKENIZER_TYPE}" \
            --tokenizer-model "${TOKENIZER_MODEL}" \
            --append-eod \
            --workers "${WORKERS}" \
            --json-keys text
        return 0
    else
        echo "Warning: ${split_type} file not found: ${input_file}"
        return 1
    fi
}

# Process all splits
TRAIN_OK=0
preprocess_split "train" && TRAIN_OK=1

if [ "$TRAIN_OK" -eq 0 ]; then
    echo "Error: Training file not found. Check if download was successful."
    echo "Expected: ${INPUT_SUBDIR}/${FILE_PREFIX}_train.jsonl"
    ls -la "${INPUT_SUBDIR}/" 2>/dev/null || echo "Directory doesn't exist: ${INPUT_SUBDIR}/"
    exit 1
fi

preprocess_split "validation" || true
preprocess_split "test" || true

# -----------------------------------------------------------------------------
# Create Datablend Config
# -----------------------------------------------------------------------------
BLEND_FILE="${DATABLEND_DIR}/${BLEND_NAME}.json"
echo ""
echo "Creating datablend config: ${BLEND_FILE}"

cat > "${BLEND_FILE}" << EOF
{
    "train": [1.0, "${WORK_DIR}/${FILE_PREFIX}_train_text_document"],
    "valid": [1.0, "${WORK_DIR}/${FILE_PREFIX}_validation_text_document"],
    "test": [1.0, "${WORK_DIR}/${FILE_PREFIX}_test_text_document"]
}
EOF

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Preprocessing complete!"
echo "=========================================="
echo "Dataset:     ${DATASET}"
echo "Output:      ${WORK_DIR}"
echo "Datablend:   ${BLEND_FILE}"
echo ""
echo "Generated files:"
ls -lh "${WORK_DIR}/${FILE_PREFIX}"*.bin 2>/dev/null || echo "  (no .bin files)"
ls -lh "${WORK_DIR}/${FILE_PREFIX}"*.idx 2>/dev/null || echo "  (no .idx files)"
echo "=========================================="

