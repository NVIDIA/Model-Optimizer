#!/bin/bash
# Preprocess Nemotron-v2 dataset for QAD training (general, model-agnostic)
#
# Usage:
#   bash process_nemotron_v2.sh --output-dir <path> --mlm-path <path> --tokenizer <model> [options]
#
# Required arguments:
#   --output-dir    Output directory for preprocessed files
#   --mlm-path      Path to Megatron-LM directory
#   --tokenizer     HuggingFace tokenizer model (e.g., Qwen/Qwen3-8B)
#
# Optional arguments:
#   --input-dir     Input directory (default: derived from output-dir)
#   --split         Split name: stem, math, code, chat (default: stem)
#   --suffix        Suffix for file naming (default: 30pct_cot_chat)
#   --workers       Number of parallel workers (default: 32)
#   --datablend-dir Directory for datablend configs (default: parent of output-dir)

set -e

# Parse arguments
OUTPUT_DIR=""
MLM_DIR=""
TOKENIZER_MODEL=""
INPUT_DIR=""
SPLIT="stem"
SUFFIX="30pct_cot_chat"
WORKERS=32
DATABLEND_DIR=""

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
            TOKENIZER_MODEL="$2"
            shift 2
            ;;
        --input-dir)
            INPUT_DIR="$2"
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

# Validate required arguments
if [ -z "$OUTPUT_DIR" ]; then
    echo "Error: --output-dir is required"
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

# Set defaults for optional arguments
if [ -z "$INPUT_DIR" ]; then
    INPUT_DIR="${OUTPUT_DIR//_preprocessed/}"
fi

if [ -z "$DATABLEND_DIR" ]; then
    DATABLEND_DIR="$(dirname "$OUTPUT_DIR")"
fi

# Normalize suffix (handle both 30pctcot and 30pct_cot)
SUFFIX=$(echo "$SUFFIX" | sed 's/pctcot/pct_cot/g')

mkdir -p ${OUTPUT_DIR}/${SPLIT}
mkdir -p ${DATABLEND_DIR}

# Ensure transformers is installed for tokenizer
pip install -q transformers tokenizers

# Tokenizer settings
TOKENIZER_TYPE="HuggingFaceTokenizer"

# Full name for output files
FULL_NAME="${SPLIT}_${SUFFIX}"

echo "=========================================="
echo "Preprocessing Nemotron-v2 Dataset"
echo "=========================================="
echo "Split: ${SPLIT}"
echo "Suffix: ${SUFFIX}"
echo "Tokenizer: ${TOKENIZER_MODEL}"
echo "MLM Path: ${MLM_DIR}"
echo "Input dir: ${INPUT_DIR}/${SPLIT}/"
echo "Output dir: ${OUTPUT_DIR}/${SPLIT}/"
echo "Datablend dir: ${DATABLEND_DIR}"
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
BLEND_FILE="${DATABLEND_DIR}/datablend_nemotron_v2_${FULL_NAME}.json"
echo "Creating datablend config: ${BLEND_FILE}"
cat > ${BLEND_FILE} << EOF
{
    "train": [1.0, "${OUTPUT_DIR}/${SPLIT}/${FULL_NAME}_train_text_document"],
    "valid": [1.0, "${OUTPUT_DIR}/${SPLIT}/${FULL_NAME}_validation_text_document"],
    "test": [1.0, "${OUTPUT_DIR}/${SPLIT}/${FULL_NAME}_test_text_document"]
}
EOF

echo "=========================================="
echo "✓ Nemotron-v2 (${FULL_NAME}) preprocessing complete!"
echo "=========================================="
echo "Output directory: ${OUTPUT_DIR}/${SPLIT}/"
echo "Datablend config: ${BLEND_FILE}"
echo ""
echo "To run QAD training:"
echo "  DATASET_NAME=nemotron_v2_${SPLIT}_${SUFFIX} bash qwen_qad.sh --config configs/your-config.conf"
echo "=========================================="

