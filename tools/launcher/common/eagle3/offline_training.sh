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
source ${SCRIPT_DIR}/../service_utils.sh

pip install -r modules/Model-Optimizer/examples/speculative_decoding/requirements.txt
pip install huggingface-hub>=1.2.1
export PATH=$PATH:/workspace/.local/bin

###################################################################################################

trap 'error_handler $0 $LINENO' ERR # ERROR HANDLER

bash modules/Model-Optimizer/examples/speculative_decoding/launch_train.sh \
    --model ${HF_MODEL_CKPT} \
    ${@}

python modules/Model-Optimizer/examples/speculative_decoding/scripts/export_hf_checkpoint.py \
    --model_path /scratchspace/eagle3 \
    --export_path /scratchspace/export

###################################################################################################

# This function handles the exit status (fails the CI).
#exit_handler $0
