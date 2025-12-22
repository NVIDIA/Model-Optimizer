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

# This script can be used to generate responses for MT-Bench using FastChat.
# Ensure that you are logged in to huggingface-cli using: huggingface-cli login.
# If you are using NIM, ensure that you export the NIM API key using:
# export OPENAI_API_KEY=<NIM_API_KEY>
#
# Usage: bash run_fastchat.sh -h <HF model folder or model card> -e <checkpoint_dir> -n <NIM model model card>
# model_name: The HuggingFace handle or folder of the model to evaluate.
# checkpoint_dir: The directory where the checkpoint is stored.
# nim_model_name: The handle of the NIM model to be used for evaluation.
#
# Example commands:
#
# Evaluate "meta-llama/Meta-Llama-3-8B-Instruct" HF model:
# bash run_fastchat.sh -h meta-llama/Meta-Llama-3-8B-Instruct
#
# Evaluate "meta-llama/Meta-Llama-3-8B-Instruct" HF model with TRT-LLM:
# bash run_fastchat.sh -h meta-llama/Meta-Llama-3-8B-Instruct -e /path/to/checkpoint_dir
#
# Evaluate "meta-llama/Meta-Llama-3-8B-Instruct" HF model with NIM:
# bash run_fastchat.sh -h meta-llama/Meta-Llama-3-8B-Instruct -n meta-llama/Meta-Llama-3-8B-Instruct


set -e
set -x

hf_model_name=""
checkpoint_dir=""
nim_model_name=""
answer_file=""
quant_cfg=""
auto_quantize_bits=""
calib_size=""
calib_batch_size=""

# Parse command line arguments
while [[ "$1" != "" ]]; do
    case $1 in
        -h | --hf_model_name )
            shift
            hf_model_name=$1
            ;;
        -e | --checkpoint_dir )
            shift
            checkpoint_dir=$1
            ;;
        -n | --nim_model_name )
            shift
            nim_model_name=$1
            ;;
        --answer_file )
            shift
            answer_file=$1
            ;;
        --quant_cfg )
            shift
            quant_cfg=$1
            ;;
        --auto_quantize_bits )
            shift
            auto_quantize_bits=$1
            ;;
        --calib_size )
            shift
            calib_size=$1
            ;;
        --calib_batch_size )
            shift
            calib_batch_size=$1
            ;;
        * )
            usage
            exit 1
    esac
    shift
done

if [ "$hf_model_name" == "" ]; then
    echo "Please provide the HF model name and NIM model name."
    exit 1
fi

if [ "$checkpoint_dir" != "" ]; then
    checkpoint_dir=" --checkpoint-dir $checkpoint_dir "
fi

if [ "$nim_model_name" != "" ]; then
    nim_model_name=" --nim-model $nim_model_name "
fi

if [ "$answer_file" != "" ]; then
    answer_file=" --answer-file $answer_file "
fi

if [ "$quant_cfg" != "" ]; then
    quant_args="--quant_cfg $quant_cfg"
    if [ "$auto_quantize_bits" != "" ]; then
        quant_args+=" --auto_quantize_bits $auto_quantize_bits"
    fi
    if [ "$calib_size" != "" ]; then
        quant_args+=" --calib_size $calib_size"
    fi
    if [ "$calib_batch_size" != "" ]; then
        quant_args+=" --calib_batch_size $calib_batch_size"
    fi
else
    quant_args=""
fi

# Clone FastChat repository
if [ ! -d FastChat ]; then
    echo "FastChat repository not found. Cloning FastChat..."
    git clone https://github.com/lm-sys/FastChat.git
    mkdir -p ./data/mt_bench
    cp FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl ./data/mt_bench/question.jsonl
else
    echo "FastChat repository found. Skipping cloning."
fi

# Install FastChat with variants
pushd FastChat
pip install -e ".[model_worker,llm_judge]" -U
popd

PYTHONPATH=FastChat:$PYTHONPATH python gen_model_answer.py \
    --model-path $hf_model_name \
    --model-id $hf_model_name \
    --temperature 0.0001 \
    --top-p 0.0001 \
    $checkpoint_dir \
    $nim_model_name \
    $answer_file \
    $quant_args
