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

# Export a quantized MCore checkpoint (saved by quantize.sh) to HF format.
#
# Required env: MLM_MODEL_CFG, QUANT_CFG.
# Optional env:
#   MLM_MODEL_CKPT  Saved PTQ MCore ckpt path (default: /scratchspace/megatron-lm/${MLM_MODEL_CFG})
#   EXPORT_DIR      HF output dir            (default: /scratchspace/export/${MLM_MODEL_CFG}_${QUANT_CFG})
#   TP, PP, EP, ETP Parallelism (defaults: 1, 1, 1, 1)

if [[ -z ${MLM_MODEL_CKPT} ]]; then
    export MLM_MODEL_CKPT="/cicd/megatron-lm/${MLM_MODEL_CFG}"
fi
if [[ -z ${EXPORT_DIR} ]]; then
    # Take basename of QUANT_CFG (strip dirs + .yaml/.yml) so recipe paths
    # collapse to a flat tag in EXPORT_DIR.
    _QUANT_CFG_TAG="$(basename "${QUANT_CFG}")"
    _QUANT_CFG_TAG="${_QUANT_CFG_TAG%.yaml}"
    _QUANT_CFG_TAG="${_QUANT_CFG_TAG%.yml}"
    export EXPORT_DIR="/cicd/export/${MLM_MODEL_CFG}_${_QUANT_CFG_TAG}"
fi
if [[ -z ${HF_MODEL_CKPT} ]]; then
    export HF_MODEL_CKPT="/hf-local/${MLM_MODEL_CFG}"
fi
export MLM_SKIP_INSTALL=1

EXPORT_EXE="bash modules/Megatron-LM/examples/post_training/modelopt/export.sh"

export MLM_EXTRA_ARGS=${@}
echo "=== Exporting ${MLM_MODEL_CFG} ${QUANT_CFG} (TP=${TP:-1} PP=${PP:-1} EP=${EP:-1} ETP=${ETP:-1}) ==="
TP=${TP:-1} PP=${PP:-1} EP=${EP:-1} ETP=${ETP:-1} ${EXPORT_EXE} ${MLM_MODEL_CFG}
ls ${EXPORT_DIR}
cat ${EXPORT_DIR}/hf_quant_config.json

###################################################################################################

exit_handler $0
