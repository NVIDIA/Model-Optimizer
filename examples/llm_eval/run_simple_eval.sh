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
NUM_EXAMPLES=${5:-}  # optional: limit examples per eval (default: full eval)

if [ ! -d "human-eval" ]; then
    git clone https://github.com/openai/human-eval.git
    # Pin to a known commit for reproducibility (and so the entry-point patch below matches).
    git -C human-eval checkout -q 6d43fb980f9fee3c892a914eda09951f772ad10d
fi

# human-eval's console_scripts entry point lacks the ":callable" suffix, which newer pip/setuptools
# reject ("A callable suffix is required"). The target module defines main(), so point at it.
sed -i 's|human_eval\.evaluate_functional_correctness"|human_eval.evaluate_functional_correctness:main"|' human-eval/setup.py

if [ ! -d "simple-evals" ]; then
    git clone https://github.com/openai/simple-evals.git
fi

# --no-build-isolation: human-eval's legacy setup.py imports pkg_resources at build time,
# which pip's isolated build env does not provide with newer setuptools. Build against the
# base environment (which has setuptools/pkg_resources) instead.
pip install -e human-eval --no-build-isolation
pip install openai

pushd simple-evals
git checkout 6e84f4e2aed6b60f6a0c7b8f06bbbf4bfde72e58
cp ../simple_evals.py simple_evals.py
popd

export OPENAI_API_KEY="local"
export OPENAI_BASE_URL="http://localhost:$PORT/v1"

examples_flag=""
if [ -n "$NUM_EXAMPLES" ]; then
    examples_flag="--examples $NUM_EXAMPLES"
fi

python -m simple-evals.simple_evals --model $MODEL_NAME --evals $EVALS --max_tokens $BUILD_MAX_OUTPUT_LEN $examples_flag
