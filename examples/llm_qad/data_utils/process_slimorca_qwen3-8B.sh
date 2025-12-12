#!/bin/bash
# Preprocess SlimOrca dataset for Qwen3-8B QAD training

set -e

# Paths
MLM_DIR="/lustre/fsw/coreai_dlalgo_modelopt/weimingc/workspace/Megatron-LM"
INPUT_DIR="/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/slimorca"
OUTPUT_DIR="/lustre/fsw/coreai_dlalgo_modelopt/weimingc/datasets/slimorca_preprocessed"

mkdir -p ${OUTPUT_DIR}

# Tokenizer settings for Qwen3-8B
TOKENIZER_TYPE="HuggingFaceTokenizer"
TOKENIZER_MODEL="Qwen/Qwen3-8B"

# Number of workers for parallel processing
WORKERS=32

echo "=========================================="
echo "Preprocessing SlimOrca Dataset for Qwen3-8B"
echo "=========================================="

# Process training split
if [ -f "${INPUT_DIR}/slimorca_train.jsonl" ]; then
    echo "Processing training split..."
    python ${MLM_DIR}/tools/preprocess_data.py \
        --input ${INPUT_DIR}/slimorca_train.jsonl \
        --output-prefix ${OUTPUT_DIR}/slimorca_train \
        --tokenizer-type ${TOKENIZER_TYPE} \
        --tokenizer-model ${TOKENIZER_MODEL} \
        --append-eod \
        --workers ${WORKERS} \
        --json-keys text
fi

# Process validation split
if [ -f "${INPUT_DIR}/slimorca_validation.jsonl" ]; then
    echo "Processing validation split..."
    python ${MLM_DIR}/tools/preprocess_data.py \
        --input ${INPUT_DIR}/slimorca_validation.jsonl \
        --output-prefix ${OUTPUT_DIR}/slimorca_validation \
        --tokenizer-type ${TOKENIZER_TYPE} \
        --tokenizer-model ${TOKENIZER_MODEL} \
        --append-eod \
        --workers ${WORKERS} \
        --json-keys text
fi

# Process test split
if [ -f "${INPUT_DIR}/slimorca_test.jsonl" ]; then
    echo "Processing test split..."
    python ${MLM_DIR}/tools/preprocess_data.py \
        --input ${INPUT_DIR}/slimorca_test.jsonl \
        --output-prefix ${OUTPUT_DIR}/slimorca_test \
        --tokenizer-type ${TOKENIZER_TYPE} \
        --tokenizer-model ${TOKENIZER_MODEL} \
        --append-eod \
        --workers ${WORKERS} \
        --json-keys text
fi

echo "=========================================="
echo "✓ SlimOrca preprocessing complete!"
echo "Output directory: ${OUTPUT_DIR}"
echo "=========================================="

