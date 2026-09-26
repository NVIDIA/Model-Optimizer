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

# HuggingFace PTQ wrapper: downloads the model if needed, then runs huggingface_example.sh.
#
# Usage:
#   ptq.sh --repo <org/model> --local-dir <path> -- [huggingface_example.sh args...]
#
# Everything before "--" is handled by this wrapper (download logic).
# Everything after "--" is passed directly to huggingface_example.sh.
# The --model arg is automatically set to <local-dir> for huggingface_example.sh.

set -e

# Install extra pip dependencies if specified (e.g., mamba-ssm for hybrid Mamba models).
if [ -n "$EXTRA_PIP_DEPS" ]; then
    echo "Installing extra dependencies: $EXTRA_PIP_DEPS"
    unset PIP_CONSTRAINT
    read -r -a _deps <<< "$EXTRA_PIP_DEPS"
    pip install "${_deps[@]}"
fi

REPO=""
LOCAL_DIR=""
PTQ_ARGS=()

# Parse wrapper args up to "--", collect the rest for huggingface_example.sh
while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo) REPO="$2"; shift 2 ;;
        --local-dir) LOCAL_DIR="$2"; shift 2 ;;
        --) shift; PTQ_ARGS=("$@"); break ;;
        *) echo "Unknown argument: $1 (use -- to separate PTQ args)" >&2; exit 1 ;;
    esac
done

if [ -z "$REPO" ] || [ -z "$LOCAL_DIR" ]; then
    echo "Usage: ptq.sh --repo <org/model> --local-dir <path> -- [huggingface_example.sh args...]" >&2
    exit 1
fi

# --- Step 1: Download model if not already present ---
if [ -f "$LOCAL_DIR/config.json" ]; then
    echo "Model already exists at $LOCAL_DIR, skipping download."
else
    echo "Downloading $REPO to $LOCAL_DIR ..."
    pip install -q huggingface_hub 2>/dev/null || true
    huggingface-cli download "$REPO" --local-dir "$LOCAL_DIR"
    echo "Download complete: $LOCAL_DIR"
fi

# --- Step 2: Run huggingface_example.sh ---
script_dir="$(dirname "$(readlink -f "$0")")"
HF_EXAMPLE="${script_dir}/../../modules/Model-Optimizer/examples/llm_ptq/scripts/huggingface_example.sh"

echo "Running huggingface_example.sh --model $LOCAL_DIR --trust_remote_code ${PTQ_ARGS[*]}"
exec bash "$HF_EXAMPLE" --model "$LOCAL_DIR" --trust_remote_code "${PTQ_ARGS[@]}"
