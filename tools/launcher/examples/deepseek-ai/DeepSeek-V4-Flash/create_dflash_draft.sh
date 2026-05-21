#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Create a randomly-initialized DFlash draft checkpoint for smoke testing.
#
# Copies the bundled config.json + dflash.py into /scratchspace/dflash_draft,
# then uses AutoModel.from_config to initialize random weights and saves the
# checkpoint in HuggingFace format.
#
# Usage: sourced as task_0 in vllm_dflash_smoke_test_cw_dfw.yaml
#   /scratchspace/dflash_draft is the output path consumed by task_1.

set -euo pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
export OUT="/scratchspace/dflash_draft"

echo "=== Creating DFlash draft checkpoint at ${OUT} ==="

mkdir -p "${OUT}"
cp "${SCRIPT_DIR}/dflash_draft/config.json" "${OUT}/config.json"
cp "${SCRIPT_DIR}/dflash_draft/dflash.py" "${OUT}/dflash.py"

python3 - <<'EOF'
import os, sys, torch
sys.path.insert(0, os.environ.get("OUT", "/scratchspace/dflash_draft"))
from transformers import AutoConfig, AutoModel

out = os.environ.get("OUT", "/scratchspace/dflash_draft")
print(f"Initializing random DFlash draft model from config: {out}/config.json")
config = AutoConfig.from_pretrained(out, trust_remote_code=True)
model = AutoModel.from_config(config, trust_remote_code=True).to(torch.bfloat16)
param_count = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {param_count / 1e6:.1f}M")
model.save_pretrained(out)
print(f"  Saved to: {out}")
EOF

echo "=== DFlash draft checkpoint ready at ${OUT} ==="
