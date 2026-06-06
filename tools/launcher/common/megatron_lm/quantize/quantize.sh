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

# Runs Megatron-LM PTQ quantization. Also runs MMLU + HF export inline unless
# RUN_MMLU / RUN_EXPORT are set to "false". Larger models that need different
# parallelism for MMLU/export should set RUN_MMLU=false RUN_EXPORT=false and
# chain the standalone mmlu/mmlu.sh and export/export.sh scripts as separate
# pipeline tasks.

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source ${SCRIPT_DIR}/../../service_utils.sh

util_install_extra_dep

trap 'error_handler $0 $LINENO' ERR # ERROR HANDLER
###################################################################################################

if [[ -z ${HF_MODEL_CKPT} ]]; then
    export HF_MODEL_CKPT="/hf-local/${MLM_MODEL_CFG}"
fi
# Persist PTQ ckpt + HF export under /cicd ($SLURM_JOB_DIR/cicd) so later
# experiments can re-use them. 
export MLM_MODEL_SAVE="/cicd/megatron-lm/${MLM_MODEL_CFG}"
# If QUANT_CFG is a recipe path, collapse to a flat tag (strip dirs + .yaml/.yml).
_QUANT_CFG_TAG="$(basename "${QUANT_CFG}")"
_QUANT_CFG_TAG="${_QUANT_CFG_TAG%.yaml}"
_QUANT_CFG_TAG="${_QUANT_CFG_TAG%.yml}"
export EXPORT_DIR="/cicd/export/${MLM_MODEL_CFG}_${_QUANT_CFG_TAG}"
export MLM_SKIP_INSTALL=1

QUANTIZE_EXE="bash modules/Megatron-LM/examples/post_training/modelopt/quantize.sh"
MMLU_EXE="bash modules/Megatron-LM/examples/post_training/modelopt/mmlu.sh"
EXPORT_EXE="bash modules/Megatron-LM/examples/post_training/modelopt/export.sh"

# Step 1: quantize
export MLM_EXTRA_ARGS=${@}
TP=${TP:-1} PP=${PP:-1} EP=${EP:-1} ETP=${ETP:-1} ${QUANTIZE_EXE} ${MLM_MODEL_CFG} ${QUANT_CFG}

# Step 2 (optional): MMLU on the saved PTQ ckpt
if [[ "${RUN_MMLU:-true}" == "true" ]]; then
    export MLM_EXTRA_ARGS="--mmlu-dataset ${MMLU_DATASET:-/hf-local/cais/mmlu} --fraction 0.01 --lower-bound ${MMLU_LOWER_BOUND:-0.38} --disable-tqdm"
    TP=${TP:-1} PP=${PP:-1} EP=${EP:-1} ETP=${ETP:-1} MLM_MODEL_CKPT=${MLM_MODEL_SAVE} ${MMLU_EXE} ${MLM_MODEL_CFG}
fi

# Step 3 (optional): export PTQ ckpt to HF format
# Use largest PP <= total GPUs that divides the model's num_hidden_layers.
if [[ "${RUN_EXPORT:-true}" == "true" ]]; then
    TOTAL_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo ${NUM_GPUS:-1})
    EXPORT_PP=$(python3 -c "
import json, os
cfg = os.path.join('${HF_MODEL_CKPT}', 'config.json')
n_layers = json.load(open(cfg)).get('num_hidden_layers', 1) if os.path.exists(cfg) else 1
gpus = ${TOTAL_GPUS}
pp = gpus
while pp > 1 and n_layers % pp != 0:
    pp -= 1
print(pp)
" 2>/dev/null || echo ${TOTAL_GPUS})
    echo "=== Exporting ${MLM_MODEL_CFG} ${QUANT_CFG} (PP=${EXPORT_PP}, ${TOTAL_GPUS} GPUs) ==="
    export MLM_EXTRA_ARGS=
    TP=1 PP=${EXPORT_PP} EP=1 ETP=1 MLM_MODEL_CKPT=${MLM_MODEL_SAVE} ${EXPORT_EXE} ${MLM_MODEL_CFG}
    ls ${EXPORT_DIR}
    cat ${EXPORT_DIR}/hf_quant_config.json
fi

###################################################################################################

# This function handles the exit status (fails the CI).
exit_handler $0
