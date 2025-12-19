#!/bin/bash
# Preprocess Nemotron-v1 dataset for Qwen3-8B QAD training

set -e

# Default to ALL splits at 30% for best general improvement
# Options: all_30pct (default), all_10pct, all_50pct, all_100pct, stem, math, etc.
# Add _chat suffix for chat template formatted data
# Examples:
#   bash process_nemotron_qwen3-8B.sh all_30pct        # 30% of all splits (simple format)
#   bash process_nemotron_qwen3-8B.sh all_30pct_chat   # 30% of all splits (chat template)
#   bash process_nemotron_qwen3-8B.sh all_10pct        # 10% of all splits (~2.5M samples)
#   bash process_nemotron_qwen3-8B.sh all_50pct        # 50% of all splits (~12.5M samples)
SPLIT_NAME="${1:-all_30pct}"

# Paths
MLM_DIR="/lustre/fsw/coreai_dlalgo_modelopt/weimingc/workspace/Megatron-LM"
INPUT_DIR="/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/nemotron_v1"
OUTPUT_DIR="/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/nemotron_v1_preprocessed"

mkdir -p ${OUTPUT_DIR}

# Install required dependencies
echo "Installing dependencies..."
pip install -q transformers tokenizers || true

# Tokenizer settings for Qwen3-8B
TOKENIZER_TYPE="HuggingFaceTokenizer"
TOKENIZER_MODEL="Qwen/Qwen3-8B"

# Number of workers for parallel processing
WORKERS=32

echo "=========================================="
echo "Preprocessing Nemotron-v1 Dataset (${SPLIT_NAME}) for Qwen3-8B"
echo "=========================================="

# Process training split
TRAIN_FILE="${INPUT_DIR}/nemotron_${SPLIT_NAME}_train.jsonl"
if [ -f "${TRAIN_FILE}" ]; then
    echo "Processing training split: ${TRAIN_FILE}"
    python ${MLM_DIR}/tools/preprocess_data.py \
        --input ${TRAIN_FILE} \
        --output-prefix ${OUTPUT_DIR}/nemotron_${SPLIT_NAME}_train \
        --tokenizer-type ${TOKENIZER_TYPE} \
        --tokenizer-model ${TOKENIZER_MODEL} \
        --append-eod \
        --workers ${WORKERS} \
        --json-keys text
else
    echo "Warning: Training file not found: ${TRAIN_FILE}"
fi

# Process validation split
VALID_FILE="${INPUT_DIR}/nemotron_${SPLIT_NAME}_validation.jsonl"
if [ -f "${VALID_FILE}" ]; then
    echo "Processing validation split: ${VALID_FILE}"
    python ${MLM_DIR}/tools/preprocess_data.py \
        --input ${VALID_FILE} \
        --output-prefix ${OUTPUT_DIR}/nemotron_${SPLIT_NAME}_validation \
        --tokenizer-type ${TOKENIZER_TYPE} \
        --tokenizer-model ${TOKENIZER_MODEL} \
        --append-eod \
        --workers ${WORKERS} \
        --json-keys text
else
    echo "Warning: Validation file not found: ${VALID_FILE}"
fi

# Process test split
TEST_FILE="${INPUT_DIR}/nemotron_${SPLIT_NAME}_test.jsonl"
if [ -f "${TEST_FILE}" ]; then
    echo "Processing test split: ${TEST_FILE}"
    python ${MLM_DIR}/tools/preprocess_data.py \
        --input ${TEST_FILE} \
        --output-prefix ${OUTPUT_DIR}/nemotron_${SPLIT_NAME}_test \
        --tokenizer-type ${TOKENIZER_TYPE} \
        --tokenizer-model ${TOKENIZER_MODEL} \
        --append-eod \
        --workers ${WORKERS} \
        --json-keys text
else
    echo "Warning: Test file not found: ${TEST_FILE}"
fi

# Create datablend config
BLEND_FILE="/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/datablend_nemotron_${SPLIT_NAME}.json"
echo "Creating datablend config: ${BLEND_FILE}"
cat > ${BLEND_FILE} << EOF
{
    "train": [1.0, "${OUTPUT_DIR}/nemotron_${SPLIT_NAME}_train_text_document"],
    "valid": [1.0, "${OUTPUT_DIR}/nemotron_${SPLIT_NAME}_validation_text_document"],
    "test": [1.0, "${OUTPUT_DIR}/nemotron_${SPLIT_NAME}_test_text_document"]
}
EOF

echo "=========================================="
echo "✓ Nemotron-v1 (${SPLIT_NAME}) preprocessing complete!"
echo "Output directory: ${OUTPUT_DIR}"
echo "Datablend config: ${BLEND_FILE}"
echo ""
echo "To run QAD training:"
echo "  bash qwen_qad.sh 1e-5 Qwen3-8B False nemotron_${SPLIT_NAME}"
echo "=========================================="

