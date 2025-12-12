#!/bin/bash
# Preprocess Nemotron-v1 dataset (split mode) for Qwen3 QAD training
#
# New folder structure from download_nemotron_v1.py:
#   nemotron_v1/
#   ├── stem/
#   │   ├── stem_30pct_cot_chat_train.jsonl
#   │   └── ...
#   ├── math/
#   │   └── ...
#
# Usage:
#   bash process_nemotron_v1_qwen3-8B.sh <split> <suffix> [tokenizer]
#
# Examples:
#   bash process_nemotron_v1_qwen3-8B.sh stem 30pct_cot_chat                              # Default: Qwen3-8B
#   bash process_nemotron_v1_qwen3-8B.sh stem 30pct_cot_chat Qwen/Qwen3-30B-A3B-Thinking-2507  # Thinking model

set -e

# Ensure transformers is installed for tokenizer
pip install -q transformers tokenizers

# Arguments
SPLIT="${1:-stem}"           # stem, math, code, chat
SUFFIX="${2:-30pct_cot_chat}"
TOKENIZER_MODEL="${3:-Qwen/Qwen3-8B}"  # Can override with any HuggingFace tokenizer

# Paths
MLM_DIR="/lustre/fsw/coreai_dlalgo_modelopt/weimingc/workspace/Megatron-LM"
INPUT_DIR="/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/nemotron_v1"
OUTPUT_DIR="/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/nemotron_v1_preprocessed"

mkdir -p ${OUTPUT_DIR}/${SPLIT}

# Tokenizer settings
TOKENIZER_TYPE="HuggingFaceTokenizer"

# Number of workers for parallel processing
WORKERS=32

# Full name for output files
FULL_NAME="${SPLIT}_${SUFFIX}"

echo "=========================================="
echo "Preprocessing Nemotron-v1 Dataset"
echo "=========================================="
echo "Split: ${SPLIT}"
echo "Suffix: ${SUFFIX}"
echo "Tokenizer: ${TOKENIZER_MODEL}"
echo "Input dir: ${INPUT_DIR}/${SPLIT}/"
echo "Output dir: ${OUTPUT_DIR}/${SPLIT}/"
echo "=========================================="

# Process training split
TRAIN_FILE="${INPUT_DIR}/${SPLIT}/${FULL_NAME}_train.jsonl"
if [ -f "${TRAIN_FILE}" ]; then
    echo "Processing training split: ${TRAIN_FILE}"
    python ${MLM_DIR}/tools/preprocess_data.py \
        --input ${TRAIN_FILE} \
        --output-prefix ${OUTPUT_DIR}/${SPLIT}/${FULL_NAME}_train \
        --tokenizer-type ${TOKENIZER_TYPE} \
        --tokenizer-model ${TOKENIZER_MODEL} \
        --append-eod \
        --workers ${WORKERS} \
        --json-keys text
else
    echo "❌ Error: Training file not found: ${TRAIN_FILE}"
    echo "   Check if download was successful."
    echo "   Expected file pattern: ${INPUT_DIR}/${SPLIT}/${FULL_NAME}_train.jsonl"
    ls -la ${INPUT_DIR}/${SPLIT}/ 2>/dev/null || echo "   Directory doesn't exist: ${INPUT_DIR}/${SPLIT}/"
    exit 1
fi

# Process validation split
VALID_FILE="${INPUT_DIR}/${SPLIT}/${FULL_NAME}_validation.jsonl"
if [ -f "${VALID_FILE}" ]; then
    echo "Processing validation split: ${VALID_FILE}"
    python ${MLM_DIR}/tools/preprocess_data.py \
        --input ${VALID_FILE} \
        --output-prefix ${OUTPUT_DIR}/${SPLIT}/${FULL_NAME}_validation \
        --tokenizer-type ${TOKENIZER_TYPE} \
        --tokenizer-model ${TOKENIZER_MODEL} \
        --append-eod \
        --workers ${WORKERS} \
        --json-keys text
else
    echo "Warning: Validation file not found: ${VALID_FILE}"
fi

# Process test split
TEST_FILE="${INPUT_DIR}/${SPLIT}/${FULL_NAME}_test.jsonl"
if [ -f "${TEST_FILE}" ]; then
    echo "Processing test split: ${TEST_FILE}"
    python ${MLM_DIR}/tools/preprocess_data.py \
        --input ${TEST_FILE} \
        --output-prefix ${OUTPUT_DIR}/${SPLIT}/${FULL_NAME}_test \
        --tokenizer-type ${TOKENIZER_TYPE} \
        --tokenizer-model ${TOKENIZER_MODEL} \
        --append-eod \
        --workers ${WORKERS} \
        --json-keys text
else
    echo "Warning: Test file not found: ${TEST_FILE}"
fi

# Create datablend config
BLEND_FILE="/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/datablend_nemotron_v1_${FULL_NAME}.json"
echo "Creating datablend config: ${BLEND_FILE}"
cat > ${BLEND_FILE} << EOF
{
    "train": [1.0, "${OUTPUT_DIR}/${SPLIT}/${FULL_NAME}_train_text_document"],
    "valid": [1.0, "${OUTPUT_DIR}/${SPLIT}/${FULL_NAME}_validation_text_document"],
    "test": [1.0, "${OUTPUT_DIR}/${SPLIT}/${FULL_NAME}_test_text_document"]
}
EOF

echo "=========================================="
echo "✓ Nemotron-v1 (${FULL_NAME}) preprocessing complete!"
echo "=========================================="
echo "Output directory: ${OUTPUT_DIR}/${SPLIT}/"
echo "Datablend config: ${BLEND_FILE}"
echo ""
echo "To run QAD training:"
echo "  DATASET_NAME=nemotron_v1_${SPLIT}_${SUFFIX} bash qwen_qad.sh --config configs/your-config.conf"
echo "=========================================="

