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
source "${SCRIPT_DIR}/../service_utils.sh"

trap 'error_handler $0 $LINENO' ERR # ERROR HANDLER
trap 'exit_handler' EXIT
###################################################################################################

HF_PTQ_DIR=modules/Model-Optimizer/examples/llm_ptq

HF_MODEL=${HF_MODEL:-"Qwen/Qwen3-8B"}
QFORMAT=${QFORMAT:-"fp8"}
CALIB_SIZE=${CALIB_SIZE:-"512"}
EXPORT_PATH=${EXPORT_PATH:-"/scratchspace/exported_model"}

PYTHONPATH="${HF_PTQ_DIR}:${PYTHONPATH}" python ${HF_PTQ_DIR}/hf_ptq.py \
    --pyt_ckpt_path ${HF_MODEL} \
    --qformat ${QFORMAT} \
    --calib_size ${CALIB_SIZE} \
    --export_path ${EXPORT_PATH} \
    "$@"

report_result "PASS: hf_ptq ${HF_MODEL} ${QFORMAT}"
