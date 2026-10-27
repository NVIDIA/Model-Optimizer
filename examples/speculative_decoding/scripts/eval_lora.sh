#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Export EAGLE3 LoRA checkpoint, merge LoRA into base, then evaluate.
#
# Environment variables:
#   HF_MODEL_CKPT:  Path to the original base model (e.g. /hf-local/Qwen/Qwen3-8B)
#   EAGLE_CKPT:     Path to the EAGLE3 training checkpoint directory
#   OUTPUT_DIR:     Output directory (default: /scratchspace)
#   RUN_BASELINE:   If "true", also evaluate the original base model
#   TRUST_REMOTE_CODE: If "true", opt in to remote-code execution during eval (default: false)
#
# Outputs to $OUTPUT_DIR/:
#   export/          - Exported eagle draft model + LoRA adapter
#   draft/           - Clean EAGLE3 head only (for vLLM, no LoRA adapter files)
#   merged_base/     - Base model with LoRA merged in
#   lm_eval/         - lm_eval results

set -euo pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
OUTPUT_DIR="${OUTPUT_DIR:-/scratchspace}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-False}"

: "${HF_MODEL_CKPT:?Set HF_MODEL_CKPT to the base model checkpoint path}"
: "${EAGLE_CKPT:?Set EAGLE_CKPT to the EAGLE training checkpoint path}"

pip install --no-deps lm_eval==0.4.11
pip install langdetect pytablewriter word2number sacrebleu rouge_score immutabledict

###################################################################################################

# If EAGLE_CKPT points to a training output dir (contains checkpoint-*), resolve the latest one.
if [ -d "${EAGLE_CKPT}" ] && ls "${EAGLE_CKPT}"/checkpoint-* &>/dev/null; then
    LATEST=$(ls -d "${EAGLE_CKPT}"/checkpoint-* | sort -V | tail -1)
    echo "Resolved latest checkpoint: ${LATEST}"
    EAGLE_CKPT="${LATEST}"
fi

echo "=== Step 1: Export EAGLE3 checkpoint ==="
python "${SCRIPT_DIR}/export_hf_checkpoint.py" \
    --model_path "${EAGLE_CKPT}" \
    --export_path "${OUTPUT_DIR}/export"

echo "=== Step 1b: Create clean draft dir (EAGLE3 head only, no LoRA adapter) ==="
mkdir -p "${OUTPUT_DIR}/draft"
cp "${OUTPUT_DIR}/export/config.json" "${OUTPUT_DIR}/draft/"
cp "${OUTPUT_DIR}/export/model.safetensors" "${OUTPUT_DIR}/draft/"
# Copy any additional non-adapter files (e.g., generation_config)
for f in "${OUTPUT_DIR}/export"/*.json; do
    base=$(basename "$f")
    if [[ "$base" != "adapter_config.json" ]]; then
        cp "$f" "${OUTPUT_DIR}/draft/"
    fi
done

echo "=== Step 2: Merge LoRA into base model ==="
MERGE_ARGS=(
    --base_model_path "${HF_MODEL_CKPT}"
    --exported_lora_dir "${OUTPUT_DIR}/export"
    --output_path "${OUTPUT_DIR}/merged_base"
)
if [[ "${TRUST_REMOTE_CODE,,}" == "true" ]]; then
    MERGE_ARGS+=(--trust_remote_code)
fi
python "${SCRIPT_DIR}/merge_lora.py" "${MERGE_ARGS[@]}"

echo "=== Step 3: Run lm_eval ==="
mkdir -p "${OUTPUT_DIR}/lm_eval"

python -m lm_eval --model hf \
    --model_args "pretrained=${OUTPUT_DIR}/merged_base,trust_remote_code=${TRUST_REMOTE_CODE}" \
    --tasks ifeval,arc_challenge,winogrande \
    --batch_size auto \
    --output_path "${OUTPUT_DIR}/lm_eval" \
    --log_samples

if [[ "${RUN_BASELINE:-false}" == "true" ]]; then
    echo "=== Step 3b: Run lm_eval on original base model (baseline) ==="
    mkdir -p "${OUTPUT_DIR}/lm_eval_baseline"

    python -m lm_eval --model hf \
        --model_args "pretrained=${HF_MODEL_CKPT},trust_remote_code=${TRUST_REMOTE_CODE}" \
        --tasks ifeval,arc_challenge,winogrande \
        --batch_size auto \
        --output_path "${OUTPUT_DIR}/lm_eval_baseline" \
        --log_samples
fi

echo "PASS: eval_lora export + merge + lm_eval"
