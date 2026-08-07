#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source ${SCRIPT_DIR}/../../service_utils.sh

util_install_extra_dep

trap 'error_handler $0 $LINENO' ERR # ERROR HANDLER
###################################################################################################

# Quantization-Aware Distillation / SFT training on a quantized MCore checkpoint
# (the output of common/megatron_lm/quantize/quantize.sh). Wraps the Megatron-LM
# ModelOpt post-training train.sh example (QAT: --modelopt-enabled).
#
# Extra flags (data/train/optim/eval, e.g. --sft, --seq-length, --lr,
# --dist-ckpt-strictness) are forwarded as CLI args and assembled into
# MLM_EXTRA_ARGS here, so callers pass them via a launcher `args:` list rather
# than space-containing env vars (which don't survive intact to the sbatch).
#
# Required env: MLM_MODEL_CFG.
# Optional env:
#   MLM_MODEL_CKPT  Quantized MCore ckpt to load (default: /cicd/megatron-lm/${MLM_MODEL_CFG})
#   MLM_MODEL_SAVE  Where to save the trained ckpt (default: MLM_MODEL_CKPT)
#   HF_MODEL_CKPT   HF source ckpt for tokenizer/config (default: /hf-local/${MLM_MODEL_CFG})
#   DP CP TP PP EP ETP  Parallelism (defaults from train.sh)

if [[ -z ${MLM_MODEL_CFG} ]]; then
    echo "[ERROR] MLM_MODEL_CFG is required." >&2
    exit 1
fi
if [[ -z ${MLM_MODEL_CKPT} ]]; then
    export MLM_MODEL_CKPT="/cicd/megatron-lm/${MLM_MODEL_CFG}"
fi
if [[ -z ${HF_MODEL_CKPT} ]]; then
    export HF_MODEL_CKPT="/hf-local/${MLM_MODEL_CFG}"
fi
export MLM_MODEL_SAVE=${MLM_MODEL_SAVE:-${MLM_MODEL_CKPT}}
export MLM_SKIP_INSTALL=1

TRAIN_EXE="bash modules/Megatron-LM/examples/post_training/modelopt/train.sh"

export MLM_EXTRA_ARGS=${@}
echo "=== QAD/SFT training ${MLM_MODEL_CFG} (load ${MLM_MODEL_CKPT}) ==="
${TRAIN_EXE} ${MLM_MODEL_CFG}

###################################################################################################

# Pass/fail is driven by the wrapped train.sh's own reporting, the ERR trap
# (error_handler), and exit_handler — matching quantize.sh/export.sh. No
# unconditional trailing PASS report (it would misreport a failed train run).
exit_handler $0
