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

if [[ -z ${HF_MODEL_CKPT} ]]; then
    export HF_MODEL_CKPT="/hf-local/${MLM_MODEL_CFG}"
fi
export MLM_MODEL_SAVE="/scratchspace/megatron-lm/${MLM_MODEL_CFG}"
export EXPORT_DIR="/scratchspace/export/${MLM_MODEL_CFG}_${QUANT_CFG}"
export MLM_SKIP_INSTALL=1

QUANTIZE_EXE="bash modules/Megatron-LM/examples/post_training/modelopt/quantize.sh"
MMLU_EXE="bash modules/Megatron-LM/examples/post_training/modelopt/mmlu.sh"
CONVERT_EXE="bash modules/Megatron-LM/examples/post_training/modelopt/convert.sh"
EXPORT_EXE="bash modules/Megatron-LM/examples/post_training/modelopt/export.sh"

export MLM_EXTRA_ARGS=${@}
TP=${TP:-1} PP=${PP:-1} EP=${EP:-1} ETP=${ETP:-1} ${QUANTIZE_EXE} ${MLM_MODEL_CFG} ${QUANT_CFG}

export MLM_EXTRA_ARGS="--mmlu-dataset ${MMLU_DATASET:-/hf-local/cais/mmlu} --fraction 0.01 --lower-bound ${MMLU_LOWER_BOUND:-0.38} --disable-tqdm"
TP=${TP:-1} PP=${PP:-1} EP=${EP:-1} ETP=${ETP:-1} MLM_MODEL_CKPT=${MLM_MODEL_SAVE} ${MMLU_EXE} ${MLM_MODEL_CFG}

# Export quantized checkpoint to HF format (PP=all GPUs)
TOTAL_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo ${NUM_GPUS:-1})
echo "=== Exporting ${MLM_MODEL_CFG} ${QUANT_CFG} (PP=${TOTAL_GPUS}) ==="
export MLM_EXTRA_ARGS=
TP=1 PP=${TOTAL_GPUS} EP=1 ETP=1 MLM_MODEL_CKPT=${MLM_MODEL_SAVE} ${EXPORT_EXE} ${MLM_MODEL_CFG}
ls ${EXPORT_DIR}
cat ${EXPORT_DIR}/hf_quant_config.json

###################################################################################################

# This function handles the exit status (fails the CI).
exit_handler $0
