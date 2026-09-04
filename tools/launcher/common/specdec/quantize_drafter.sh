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

# Calibration-free PTQ for an exported speculative-decoding drafter.
#
# Required env vars:
#   DRAFTER_CKPT — exported drafter path, or a training output_dir to auto-detect under

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source ${SCRIPT_DIR}/../service_utils.sh

trap 'error_handler $0 $LINENO' ERR

###################################################################################################

DRAFTER="${DRAFTER_CKPT}"
# Training writes exported-checkpoint-<step>/ under output_dir; take the newest. Only for a
# local directory -- anything else (an HF repo id) is passed through for the script to
# resolve. -V sorts numerically, so checkpoint-1000 beats checkpoint-900.
if [ -d "${DRAFTER}" ] && [ ! -f "${DRAFTER}/config.json" ]; then
    latest=$(find "${DRAFTER}" -maxdepth 1 -mindepth 1 -type d \
        -name 'exported-checkpoint-*' -printf '%f\n' 2>/dev/null | sort -V | tail -1)
    if [ -z "${latest}" ]; then
        echo "ERROR: ${DRAFTER} is not a checkpoint and holds no exported-checkpoint-* directory."
        exit 1
    fi
    DRAFTER="${DRAFTER}/${latest}"
    echo "Auto-detected drafter: ${DRAFTER}"
fi

python modules/Model-Optimizer/examples/speculative_decoding/scripts/quantize_drafter.py \
    --drafter_path "${DRAFTER}" \
    "$@"
