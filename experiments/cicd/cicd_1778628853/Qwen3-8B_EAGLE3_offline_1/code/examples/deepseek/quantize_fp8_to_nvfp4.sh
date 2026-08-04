#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set -e  # Exit immediately if any command fails

usage() {
    echo "Usage: $0 --amax_path <path> --fp4_output_path <path> --fp8_hf_path <path> [--world_size <n>]"
    exit 1
}

# Initialize variables with defaults
AMAX_PATH=""
FP4_PATH=""
FP8_HF_PATH=""
WORLD_SIZE=8

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --amax_path)
            AMAX_PATH="$2"
            shift 2
            ;;
        --fp4_output_path)
            FP4_PATH="$2"
            shift 2
            ;;
        --fp8_hf_path)
            FP8_HF_PATH="$2"
            shift 2
            ;;
        --world_size)
            WORLD_SIZE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            ;;
    esac
done

# Validate required arguments
if [[ -z "$AMAX_PATH" ]]; then
    echo "Error: --amax_path is required."
    usage
fi

if [[ -z "$FP4_PATH" ]]; then
    echo "Error: --fp4_output_path is required."
    usage
fi

if [[ -z "$FP8_HF_PATH" ]]; then
    echo "Error: --fp8_hf_path is required."
    usage
fi

# for KIMI-K2, copy tiktoken.model to tokenizer to the quantized checkpoint
if [[ -f "$FP8_HF_PATH/tiktoken.model" ]]; then
    echo "tiktoken.model found in $FP8_HF_PATH"
    cp $FP8_HF_PATH/tiktoken.model $FP4_PATH/
fi

# Copy miscellaneous files to the quantized checkpoint
mkdir -p $FP4_PATH
cp $FP8_HF_PATH/*.json $FP4_PATH/
cp $FP8_HF_PATH/*.py $FP4_PATH/ || true
cp -r $FP8_HF_PATH/assets $FP4_PATH/ || true

# Run the quantization command
echo "Running quantization..."
python quantize_to_nvfp4.py \
    --amax_path "$AMAX_PATH" \
    --fp4_path "$FP4_PATH" \
    --fp8_hf_path "$FP8_HF_PATH" \
    --world_size "$WORLD_SIZE"

echo "Quantization command completed successfully."
