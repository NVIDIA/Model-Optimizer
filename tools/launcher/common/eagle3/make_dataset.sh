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

###################################################################################################

trap 'error_handler $0 $LINENO' ERR # ERROR HANDLER

python modules/Model-Optimizer/examples/speculative_decoding/prepare_input_conversations/make_dataset.py \
    ${@}

mkdir -p /scratchspace/data
mv input_conversations/train.jsonl /scratchspace/data/train.jsonl


