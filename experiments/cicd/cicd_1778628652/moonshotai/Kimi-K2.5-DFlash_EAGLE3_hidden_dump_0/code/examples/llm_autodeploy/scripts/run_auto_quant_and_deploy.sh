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

# Exit immediately if a command exits with a non-zero status
set -e
set -x

# Function to print error message and exit
error_exit() {
    echo "Error: $1" >&2
    exit 1
}

# Check if required arguments are provided
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 --hf_ckpt <hf_ckpt_path> --save_quantized_ckpt <path_to_save_quantized_ckpt> [--quant <quantization>] [--effective_bits <bits>] [--host <host>] [--port <port>]"
    exit 1
fi

# Default values
HOST="127.0.0.1"
PORT="8000"
WORLD_SIZE="1"
CALIB_BATCH_SIZE=8

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --hf_ckpt)
            HF_CKPT="$2"
            shift 2
            ;;
        --save_quantized_ckpt)
            save_quantized_ckpt="$2"
            shift 2
            ;;
        --quant)
            QUANT="$2"
            shift 2
            ;;
        --effective_bits)
            EFFECTIVE_BITS="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --world_size)
            WORLD_SIZE="$2"
            shift 2
            ;;
        --calib_batch_size)
            CALIB_BATCH_SIZE="$2"
            shift 2
            ;;
        *)
            error_exit "Unknown argument: $1"
            ;;
    esac
done

# Ensure QUANTIZER_CKPT is provided
if [[ -z "$save_quantized_ckpt" ]]; then
    error_exit "--save_quantized_ckpt is required to specify the path to save quantized checkpoint."
fi

if [[ ! -d "$HF_CKPT" ]]; then
    error_exit "Checkpoint path '$HF_CKPT' does not exist!"
fi

# Step 1: Quantize the model only if QUANTIZER_CKPT does not exist
if [[ -d "$save_quantized_ckpt" ]]; then
    echo "Quantized model already exists at '$save_quantized_ckpt'. Skipping quantization step."
else
    if [[ -z "$HF_CKPT" || -z "$QUANT" || -z "$EFFECTIVE_BITS" ]]; then
        error_exit "--hf_ckpt, --quant, --effective_bits must be specified to generate quantized checkpoint."
    fi

    echo "Running Model Quantization..."
    QUANTIZE_CMD="python run_auto_quantize.py --hf_ckpt $HF_CKPT --output_dir $save_quantized_ckpt --quant $QUANT --effective_bits $EFFECTIVE_BITS --calib_batch_size $CALIB_BATCH_SIZE"
    eval "$QUANTIZE_CMD"
    echo "Model Quantization completed successfully!"
fi

# Step 2: Launch the inference server
echo "Starting Inference Server..."
SERVER_CMD="python api_server.py --ckpt_path $save_quantized_ckpt --host $HOST --port $PORT --world_size $WORLD_SIZE"
eval "$SERVER_CMD"
echo "Inference Server is now running!"
