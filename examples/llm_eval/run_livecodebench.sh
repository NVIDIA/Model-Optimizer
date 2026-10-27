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

set -e
set -x

MODEL_NAME=$1
BUILD_MAX_BATCH_SIZE=${2:-1}
BUILD_MAX_OUTPUT_LEN=${3:-8192}
PORT=${4:-8000}

if [ ! -d "LiveCodeBench" ]; then
    git clone https://github.com/LiveCodeBench/LiveCodeBench.git
fi
pushd LiveCodeBench

# delete conflictng dependencies
sed -i '/torch/d;/vllm/d' pyproject.toml

pip install -e .

export OPENAI_API_KEY="local"
export OPENAI_BASE_URL="http://localhost:$PORT/v1"
python ../livecodebench.py --model $MODEL_NAME --scenario codegeneration --evaluate --n 1 --openai_timeout 600 --multiprocess $BUILD_MAX_BATCH_SIZE --max_tokens $BUILD_MAX_OUTPUT_LEN
popd
