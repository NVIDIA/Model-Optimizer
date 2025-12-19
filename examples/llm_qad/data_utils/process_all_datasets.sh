#!/bin/bash
# Download and process all datasets (general, model-agnostic)
#
# Usage:
#   bash process_all_datasets.sh --output-dir <path> --mlm-path <path> --tokenizer <model> [options]
#
# Required arguments:
#   --output-dir      Base output directory for datasets
#   --mlm-path        Path to Megatron-LM directory
#   --tokenizer       HuggingFace tokenizer model (e.g., Qwen/Qwen3-8B)
#
# Optional arguments:
#   --datablend-dir     Directory for datablend configs (default: output-dir)
#   --suffix            Suffix for file naming (default: 30pct_chat)
#   --sample-percent    Percentage of data to use (default: 30)
#   --include-reasoning Include chain-of-thought reasoning (for Thinking models)
#                       Default: OFF (suitable for Instruct models)
#
# Examples:
#   # For Instruct models (no COT):
#   bash process_all_datasets.sh --output-dir /data --mlm-path /mlm --tokenizer Qwen/Qwen3-30B-A3B-Instruct-2507
#
#   # For Thinking models (with COT):
#   bash process_all_datasets.sh --output-dir /data --mlm-path /mlm --tokenizer Qwen/Qwen3-30B-A3B-Thinking-2507 --include-reasoning

set -e

# Parse arguments
OUTPUT_DIR=""
MLM_DIR=""
TOKENIZER=""
DATABLEND_DIR=""
SUFFIX=""  # Will be set based on --include-reasoning
SAMPLE_PERCENT=30
INCLUDE_REASONING=false

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
        --datablend-dir)
            DATABLEND_DIR="$2"
            shift 2
            ;;
        --suffix)
            SUFFIX="$2"
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
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set default suffix based on reasoning flag
if [ -z "$SUFFIX" ]; then
    if [ "$INCLUDE_REASONING" = true ]; then
        SUFFIX="${SAMPLE_PERCENT}pct_cot_chat"
    else
        SUFFIX="${SAMPLE_PERCENT}pct_chat"
    fi
fi

# Validate required arguments
if [ -z "$OUTPUT_DIR" ]; then
    echo "Error: --output-dir is required"
    echo "Usage: bash process_all_datasets.sh --output-dir <path> --mlm-path <path> --tokenizer <model>"
    exit 1
fi

if [ -z "$MLM_DIR" ]; then
    echo "Error: --mlm-path is required"
    exit 1
fi

if [ -z "$TOKENIZER" ]; then
    echo "Error: --tokenizer is required"
    exit 1
fi

# Set defaults
if [ -z "$DATABLEND_DIR" ]; then
    DATABLEND_DIR="${OUTPUT_DIR}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Downloading and Processing All Datasets"
echo "=========================================="
echo "Output dir: ${OUTPUT_DIR}"
echo "Datablend dir: ${DATABLEND_DIR}"
echo "MLM path: ${MLM_DIR}"
echo "Tokenizer: ${TOKENIZER}"
echo "Suffix: ${SUFFIX}"
echo "Sample percent: ${SAMPLE_PERCENT}%"
echo "Include reasoning (COT): ${INCLUDE_REASONING}"
echo "=========================================="

# Create directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${DATABLEND_DIR}"

# 1. Download datasets
echo ""
echo "=== Step 1: Downloading Datasets ==="

echo ">>> Downloading OpenScience..."
python "${SCRIPT_DIR}/download_openscience.py" \
    --output-dir "${OUTPUT_DIR}/openscience_splits" \
    --datablend-dir "${DATABLEND_DIR}" \
    --tokenizer "${TOKENIZER}"

# Build reasoning flag for download commands
REASONING_FLAG=""
if [ "$INCLUDE_REASONING" = true ]; then
    REASONING_FLAG="--include-reasoning"
fi

echo ">>> Downloading Nemotron-v1 @ ${SAMPLE_PERCENT}%..."
python "${SCRIPT_DIR}/download_nemotron_v1.py" \
    --output-dir "${OUTPUT_DIR}/nemotron_v1" \
    --datablend-dir "${DATABLEND_DIR}" \
    --sample-percent "${SAMPLE_PERCENT}" \
    ${REASONING_FLAG} \
    --tokenizer "${TOKENIZER}"

echo ">>> Downloading Nemotron-v2 @ ${SAMPLE_PERCENT}%..."
python "${SCRIPT_DIR}/download_nemotron_v2.py" \
    --output-dir "${OUTPUT_DIR}/nemotron_v2" \
    --datablend-dir "${DATABLEND_DIR}" \
    --sample-percent "${SAMPLE_PERCENT}" \
    ${REASONING_FLAG} \
    --tokenizer "${TOKENIZER}"

# 2. Process datasets
echo ""
echo "=== Step 2: Processing Datasets ==="

echo ">>> Processing OpenScience..."
bash "${SCRIPT_DIR}/process_openscience.sh" \
    --output-dir "${OUTPUT_DIR}/openscience_splits_preprocessed" \
    --input-dir "${OUTPUT_DIR}/openscience_splits" \
    --mlm-path "${MLM_DIR}" \
    --tokenizer "${TOKENIZER}" \
    --suffix chat \
    --datablend-dir "${DATABLEND_DIR}"

echo ">>> Processing Nemotron-v1 splits..."
for split in stem math code chat; do
    echo "    Processing nemotron_v1/${split}..."
    bash "${SCRIPT_DIR}/process_nemotron_v1.sh" \
        --output-dir "${OUTPUT_DIR}/nemotron_v1_preprocessed" \
        --input-dir "${OUTPUT_DIR}/nemotron_v1" \
        --mlm-path "${MLM_DIR}" \
        --tokenizer "${TOKENIZER}" \
        --split "${split}" \
        --suffix "${SUFFIX}" \
        --datablend-dir "${DATABLEND_DIR}"
done

echo ">>> Processing Nemotron-v2 splits..."
for split in stem math code chat; do
    echo "    Processing nemotron_v2/${split}..."
    bash "${SCRIPT_DIR}/process_nemotron_v2.sh" \
        --output-dir "${OUTPUT_DIR}/nemotron_v2_preprocessed" \
        --input-dir "${OUTPUT_DIR}/nemotron_v2" \
        --mlm-path "${MLM_DIR}" \
        --tokenizer "${TOKENIZER}" \
        --split "${split}" \
        --suffix "${SUFFIX}" \
        --datablend-dir "${DATABLEND_DIR}"
done

echo ""
echo "=========================================="
echo "✓ All datasets downloaded and processed!"
echo "=========================================="
echo ""
echo "Available datasets:"
echo "  - openscience_chat"
echo "  - nemotron_v1_stem_${SUFFIX}"
echo "  - nemotron_v1_math_${SUFFIX}"
echo "  - nemotron_v1_code_${SUFFIX}"
echo "  - nemotron_v1_chat_${SUFFIX}"
echo "  - nemotron_v2_stem_${SUFFIX}"
echo "  - nemotron_v2_math_${SUFFIX}"
echo "  - nemotron_v2_code_${SUFFIX}"
echo "  - nemotron_v2_chat_${SUFFIX}"
echo ""
echo "Datablend configs are in: ${DATABLEND_DIR}"
echo ""
echo "Usage:"
echo "  DATASET_NAME=combined_cot_chat sbatch sbatch_qwen_qad.sh --config configs/your-config.conf"
echo "  DATASET_NAME=nemotron_v1_stem_${SUFFIX} sbatch sbatch_qwen_qad.sh --config ..."
echo "=========================================="
