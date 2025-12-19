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
# REQUIREMENTS:
#   - HuggingFace access to nvidia/Nemotron-Post-Training-Dataset-v2
#   - Run: huggingface-cli login
#
# =============================================================================

set -e

# Parse arguments
OUTPUT_DIR=""
MLM_DIR=""
TOKENIZER=""
SAMPLE_PERCENT=30

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
    echo "  --sample-percent  Percentage of data to use (default: 30)"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUFFIX="${SAMPLE_PERCENT}pct_chat"

echo "=============================================="
echo "QAD Dataset Generation"
echo "=============================================="
echo "Output:       ${OUTPUT_DIR}"
echo "MLM path:     ${MLM_DIR}"
echo "Tokenizer:    ${TOKENIZER}"
echo "Sample:       ${SAMPLE_PERCENT}%"
echo "=============================================="

# Create directories
mkdir -p "${OUTPUT_DIR}"

# =============================================================================
# Step 1: Download datasets (using unified download script)
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
    --tokenizer "${TOKENIZER}"

# =============================================================================
# Step 2: Preprocess datasets to Megatron format
# =============================================================================
echo ""
echo "=== Step 2: Preprocessing to Megatron Format ==="

echo "[1/5] Processing OpenScience..."
bash "${SCRIPT_DIR}/process_openscience.sh" \
    --output-dir "${OUTPUT_DIR}/openscience_splits_preprocessed" \
    --input-dir "${OUTPUT_DIR}/openscience_splits" \
    --mlm-path "${MLM_DIR}" \
    --tokenizer "${TOKENIZER}" \
    --suffix chat \
    --datablend-dir "${OUTPUT_DIR}"

echo "[2/5] Processing Nemotron-v2 code..."
bash "${SCRIPT_DIR}/process_nemotron_v2.sh" \
    --output-dir "${OUTPUT_DIR}/nemotron_v2_preprocessed" \
    --input-dir "${OUTPUT_DIR}/nemotron_v2" \
    --mlm-path "${MLM_DIR}" \
    --tokenizer "${TOKENIZER}" \
    --split code \
    --suffix "${SUFFIX}" \
    --datablend-dir "${OUTPUT_DIR}"

echo "[3/5] Processing Nemotron-v2 math..."
bash "${SCRIPT_DIR}/process_nemotron_v2.sh" \
    --output-dir "${OUTPUT_DIR}/nemotron_v2_preprocessed" \
    --input-dir "${OUTPUT_DIR}/nemotron_v2" \
    --mlm-path "${MLM_DIR}" \
    --tokenizer "${TOKENIZER}" \
    --split math \
    --suffix "${SUFFIX}" \
    --datablend-dir "${OUTPUT_DIR}"

echo "[4/5] Processing Nemotron-v2 stem..."
bash "${SCRIPT_DIR}/process_nemotron_v2.sh" \
    --output-dir "${OUTPUT_DIR}/nemotron_v2_preprocessed" \
    --input-dir "${OUTPUT_DIR}/nemotron_v2" \
    --mlm-path "${MLM_DIR}" \
    --tokenizer "${TOKENIZER}" \
    --split stem \
    --suffix "${SUFFIX}" \
    --datablend-dir "${OUTPUT_DIR}"

echo "[5/5] Processing Nemotron-v2 chat..."
bash "${SCRIPT_DIR}/process_nemotron_v2.sh" \
    --output-dir "${OUTPUT_DIR}/nemotron_v2_preprocessed" \
    --input-dir "${OUTPUT_DIR}/nemotron_v2" \
    --mlm-path "${MLM_DIR}" \
    --tokenizer "${TOKENIZER}" \
    --split chat \
    --suffix "${SUFFIX}" \
    --datablend-dir "${OUTPUT_DIR}"

# =============================================================================
# Step 3: Create combined datablend JSON
# =============================================================================
echo ""
echo "=== Step 3: Creating Combined Datablend ==="

DATABLEND_FILE="${OUTPUT_DIR}/datablend_combined.json"

cat > "${DATABLEND_FILE}" << EOF
{
    "train": [
        0.3, "${OUTPUT_DIR}/nemotron_v2_preprocessed/code/code_${SUFFIX}_train_text_document",
        0.2, "${OUTPUT_DIR}/nemotron_v2_preprocessed/math/math_${SUFFIX}_train_text_document",
        0.2, "${OUTPUT_DIR}/nemotron_v2_preprocessed/stem/stem_${SUFFIX}_train_text_document",
        0.1, "${OUTPUT_DIR}/nemotron_v2_preprocessed/chat/chat_${SUFFIX}_train_text_document",
        0.2, "${OUTPUT_DIR}/openscience_splits_preprocessed/openscience_chat_train_text_document"
    ],
    "valid": [
        0.5, "${OUTPUT_DIR}/nemotron_v2_preprocessed/stem/stem_${SUFFIX}_validation_text_document",
        0.5, "${OUTPUT_DIR}/openscience_splits_preprocessed/openscience_chat_validation_text_document"
    ],
    "test": [
        0.5, "${OUTPUT_DIR}/nemotron_v2_preprocessed/stem/stem_${SUFFIX}_test_text_document",
        0.5, "${OUTPUT_DIR}/openscience_splits_preprocessed/openscience_chat_test_text_document"
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
echo "  ├── openscience_splits/           # Raw JSONL"
echo "  ├── openscience_splits_preprocessed/  # Megatron format"
echo "  ├── nemotron_v2/                  # Raw JSONL"
echo "  ├── nemotron_v2_preprocessed/     # Megatron format"
echo "  └── datablend_combined.json       # Combined dataset config"
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

