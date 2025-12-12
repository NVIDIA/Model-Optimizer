#!/bin/bash
# Download and process all datasets with Qwen3-30B-A3B-Thinking-2507 chat template
# All datasets are split into individual folders for fine-grained control

set -e

cd /lustre/fsw/coreai_dlalgo_modelopt/weimingc/workspace/Megatron-LM/examples/post_training/modelopt

TOKENIZER="Qwen/Qwen3-30B-A3B-Thinking-2507"
SUFFIX="30pct_cot_chat"

echo "=========================================="
echo "Downloading and Processing All Datasets"
echo "Tokenizer: ${TOKENIZER}"
echo "Suffix: ${SUFFIX}"
echo "=========================================="

# 1. Download datasets (all in split mode for fine-grained control)
echo ""
echo "=== Step 1: Downloading Datasets ==="

# echo ">>> Downloading OpenScience..."
# python download_openscience.py --tokenizer $TOKENIZER

# echo ">>> Downloading Nemotron-v1 @ 30% (split mode)..."
# python download_nemotron_v1.py --sample-percent 30 --include-reasoning --tokenizer $TOKENIZER

# echo ">>> Downloading Nemotron-v2 @ 30%..."
# python download_nemotron_v2.py --sample-percent 30 --tokenizer $TOKENIZER

# 2. Process datasets
echo ""
echo "=== Step 2: Processing Datasets ==="

# echo ">>> Processing OpenScience..."
# bash process_openscience_qwen3-8B.sh chat ${TOKENIZER}

echo ">>> Processing Nemotron-v1 splits..."
for split in stem math code chat; do
    echo "    Processing nemotron_v1/${split}..."
    bash process_nemotron_v1_qwen3-8B.sh $split ${SUFFIX} ${TOKENIZER}
done

# echo ">>> Processing Nemotron-v2 splits..."
# for split in stem math code chat; do
#     echo "    Processing nemotron_v2/${split}..."
#     bash process_nemotron_v2_qwen3-8B.sh $split ${SUFFIX} ${TOKENIZER}
# done

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
echo "  - combined_cot_chat (uses all above with weights)"
echo ""
echo "Usage:"
echo "  DATASET_NAME=combined_cot_chat sbatch sbatch_qwen_qad.sh --config configs/your-config.conf"
echo "  DATASET_NAME=nemotron_v1_stem_${SUFFIX} sbatch sbatch_qwen_qad.sh --config ..."
echo "=========================================="
