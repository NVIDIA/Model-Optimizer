#!/bin/bash
# Preprocess OpenScience dataset for Qwen3 QAD training
#
# Usage:
#   bash process_openscience_qwen3-8B.sh [suffix] [tokenizer]
#
# Examples:
#   bash process_openscience_qwen3-8B.sh                                          # Simple format, Qwen3-8B
#   bash process_openscience_qwen3-8B.sh chat                                     # Chat template, Qwen3-8B
#   bash process_openscience_qwen3-8B.sh chat Qwen/Qwen3-30B-A3B-Thinking-2507    # Chat template, Thinking model

set -e

# Arguments
SUFFIX="${1:-}"  # empty for simple format, "chat" for chat template
TOKENIZER_MODEL="${2:-Qwen/Qwen3-8B}"  # Can override with any HuggingFace tokenizer

# Normalize suffix
if [ -n "$SUFFIX" ]; then
    FILE_SUFFIX="_${SUFFIX}"
else
    FILE_SUFFIX=""
fi

# Paths
MLM_DIR="/lustre/fsw/coreai_dlalgo_modelopt/weimingc/workspace/Megatron-LM"
INPUT_DIR="/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/openscience_splits"
OUTPUT_DIR="/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/openscience_splits_preprocessed"

mkdir -p ${OUTPUT_DIR}

# Tokenizer settings
TOKENIZER_TYPE="HuggingFaceTokenizer"

# Number of workers for parallel processing
WORKERS=32

echo "=========================================="
echo "Preprocessing OpenScience Dataset"
echo "Format suffix: ${FILE_SUFFIX:-none (simple format)}"
echo "Tokenizer: ${TOKENIZER_MODEL}"
echo "=========================================="

# Process training split
TRAIN_FILE="${INPUT_DIR}/openscience${FILE_SUFFIX}_train.jsonl"
if [ -f "${TRAIN_FILE}" ]; then
    echo "Processing training split: ${TRAIN_FILE}"
    python ${MLM_DIR}/tools/preprocess_data.py \
        --input ${TRAIN_FILE} \
        --output-prefix ${OUTPUT_DIR}/openscience${FILE_SUFFIX}_train \
        --tokenizer-type ${TOKENIZER_TYPE} \
        --tokenizer-model ${TOKENIZER_MODEL} \
        --append-eod \
        --workers ${WORKERS} \
        --json-keys text
else
    echo "❌ Training file not found: ${TRAIN_FILE}"
    exit 1
fi

# Process validation split
VALID_FILE="${INPUT_DIR}/openscience${FILE_SUFFIX}_validation.jsonl"
if [ -f "${VALID_FILE}" ]; then
    echo "Processing validation split: ${VALID_FILE}"
    python ${MLM_DIR}/tools/preprocess_data.py \
        --input ${VALID_FILE} \
        --output-prefix ${OUTPUT_DIR}/openscience${FILE_SUFFIX}_validation \
        --tokenizer-type ${TOKENIZER_TYPE} \
        --tokenizer-model ${TOKENIZER_MODEL} \
        --append-eod \
        --workers ${WORKERS} \
        --json-keys text
else
    echo "Warning: Validation file not found: ${VALID_FILE}"
fi

# Process test split (if exists)
TEST_FILE="${INPUT_DIR}/openscience${FILE_SUFFIX}_test.jsonl"
if [ -f "${TEST_FILE}" ]; then
    echo "Processing test split: ${TEST_FILE}"
    python ${MLM_DIR}/tools/preprocess_data.py \
        --input ${TEST_FILE} \
        --output-prefix ${OUTPUT_DIR}/openscience${FILE_SUFFIX}_test \
        --tokenizer-type ${TOKENIZER_TYPE} \
        --tokenizer-model ${TOKENIZER_MODEL} \
        --append-eod \
        --workers ${WORKERS} \
        --json-keys text
else
    echo "Warning: Test file not found: ${TEST_FILE}"
fi

# Create datablend config
BLEND_FILE="/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/datablend_openscience${FILE_SUFFIX}.json"
echo "Creating datablend config: ${BLEND_FILE}"
cat > ${BLEND_FILE} << EOF
{
    "train": [1.0, "${OUTPUT_DIR}/openscience${FILE_SUFFIX}_train_text_document"],
    "valid": [1.0, "${OUTPUT_DIR}/openscience${FILE_SUFFIX}_validation_text_document"],
    "test": [1.0, "${OUTPUT_DIR}/openscience${FILE_SUFFIX}_test_text_document"]
}
EOF

echo "=========================================="
echo "✓ Preprocessing complete!"
echo "Output files are in: ${OUTPUT_DIR}"
echo "Datablend config: ${BLEND_FILE}"
echo "=========================================="

# List generated files
echo "Generated files:"
ls -lh ${OUTPUT_DIR}/openscience${FILE_SUFFIX}*.bin 2>/dev/null || echo "No .bin files found"
ls -lh ${OUTPUT_DIR}/openscience${FILE_SUFFIX}*.idx 2>/dev/null || echo "No .idx files found"

echo ""
echo "To use in QAD training:"
echo "  DATASET_NAME=openscience${FILE_SUFFIX} bash qwen_qad.sh --config configs/your-config.conf"
