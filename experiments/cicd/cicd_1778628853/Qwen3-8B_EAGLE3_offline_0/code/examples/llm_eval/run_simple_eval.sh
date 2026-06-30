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
EVALS=$2
BUILD_MAX_OUTPUT_LEN=${3:-2048}
PORT=${4:-8000}

if [ ! -d "human-eval" ]; then
    git clone https://github.com/openai/human-eval.git
fi

if [ ! -d "simple-evals" ]; then
    git clone https://github.com/openai/simple-evals.git
fi

pip install -e human-eval
pip install openai

pushd simple-evals
git checkout 6e84f4e2aed6b60f6a0c7b8f06bbbf4bfde72e58
cp ../simple_evals.py simple_evals.py
popd

export OPENAI_API_KEY="local"
export OPENAI_BASE_URL="http://localhost:$PORT/v1"

python -m simple-evals.simple_evals --model $MODEL_NAME --evals $EVALS --max_tokens $BUILD_MAX_OUTPUT_LEN
