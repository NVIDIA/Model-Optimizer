#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
TRAINING_DIR=${DRAFT_TRAINING_DIR:?DRAFT_TRAINING_DIR must be set}
EXPORT_DIR=${DRAFT_MODEL:?DRAFT_MODEL must be set}

LATEST_CHECKPOINT=""
while IFS= read -r checkpoint; do
    if [ -s "$checkpoint/config.json" ] && [ -s "$checkpoint/model.safetensors" ] && \
       [ -s "$checkpoint/modelopt_state.pth" ] && [ -s "$checkpoint/trainer_state.json" ]; then
        LATEST_CHECKPOINT=$checkpoint
        break
    fi
done < <(find "$TRAINING_DIR" -maxdepth 1 -type d -name 'checkpoint-*' -print | sort -Vr)

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "ERROR: No complete checkpoint found in $TRAINING_DIR" >&2
    exit 1
fi

echo "Exporting latest checkpoint: $LATEST_CHECKPOINT -> $EXPORT_DIR"
EXPORT_ARGS=(--model_path "$LATEST_CHECKPOINT" --export_path "$EXPORT_DIR")
[ "${EXPORT_TRUST_REMOTE_CODE:-0}" = "1" ] && EXPORT_ARGS+=(--trust_remote_code)
python3 -m pip install --no-cache-dir 'omegaconf>=2.3.0' 'pulp<4.0' scipy
python3 -m pip install --no-cache-dir --no-deps -e modules/Model-Optimizer
python3 \
    modules/Model-Optimizer/examples/speculative_decoding/scripts/export_hf_checkpoint.py \
    "${EXPORT_ARGS[@]}"

SMOKE_PROFILE=greedy SMOKE_SAMPLING_FIELDS='"temperature": 0' \
    bash "$SCRIPT_DIR/vllm_smoke_test.sh"
SMOKE_PROFILE=sampled SMOKE_SAMPLING_FIELDS='"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5' \
    bash "$SCRIPT_DIR/vllm_smoke_test.sh"
