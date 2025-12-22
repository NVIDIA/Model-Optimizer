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

script_dir="$(dirname "$(readlink -f "$0")")"

source $script_dir/../../llm_ptq/scripts/parser.sh
parse_options "$@"

set -x

# This will prevent the script from hanging on Selene/EOS due to the MPI support.
echo "********** unset all SLURM_, PMI_, PMIX_ Variables **********"
for i in $(env | grep ^SLURM_ | cut -d"=" -f 1); do unset -v $i; done
for i in $(env | grep ^PMI_ | cut -d"=" -f 1); do unset -v $i; done
for i in $(env | grep ^PMIX_ | cut -d"=" -f 1); do unset -v $i; done

if [ -z "$MODEL_PATH" ]; then
    echo "Unsupported model argument: Expected a huggingface model path or model name or a nemo path" >&2
    exit 1
fi

case $QFORMAT in
    fp8|int8_sq|int4_awq|w4a8_awq|nvfp4)
        ;;
    *)
        echo "Unknown quant argument: Expected one of: [fp8, int8_sq, int4_awq, w4a8_awq, nvfp4]" >&2
        exit 1
esac

script_dir="$(dirname "$(readlink -f "$0")")"

pushd $script_dir/..

if [ -z "$ROOT_SAVE_PATH" ]; then
    ROOT_SAVE_PATH=$(pwd)
fi

MODEL_NAME=$(basename $MODEL_PATH | sed 's/[^0-9a-zA-Z\-]/_/g')_${QFORMAT}${KV_CACHE_QUANT:+_kv_${KV_CACHE_QUANT}}
SAVE_PATH=${ROOT_SAVE_PATH}/saved_models_${MODEL_NAME}

MODEL_CONFIG=${SAVE_PATH}/config.json

if [ "${REMOVE_EXISTING_MODEL_CONFIG,,}" = "true" ]; then
    rm -f $MODEL_CONFIG
fi

PTQ_ARGS=""

if [ -n "$AUTO_QUANTIZE_BITS" ]; then
    PTQ_ARGS+=" --auto_quantize_bits $AUTO_QUANTIZE_BITS "
fi

if $TRUST_REMOTE_CODE; then
    PTQ_ARGS+=" --trust_remote_code "
fi

if [ -n "$KV_CACHE_QUANT" ]; then
    PTQ_ARGS+=" --kv_cache_qformat=$KV_CACHE_QUANT "
fi

if [[ "${MODEL_NAME,,}" == *"vila"* ]]; then
    # Check transformers version - must be <= 4.50.0
    CURRENT_TRANSFORMERS_VERSION=$(pip show transformers | grep Version | cut -d' ' -f2)
    if [ "$(printf '%s\n' "4.50.0" "$CURRENT_TRANSFORMERS_VERSION" | sort -V | head -n1)" = "4.50.0" ] && [ "$CURRENT_TRANSFORMERS_VERSION" != "4.50.0" ]; then
        echo "ERROR: transformers version $CURRENT_TRANSFORMERS_VERSION is not supported." >&2
        echo "VILA requires transformers<=4.50.0" >&2
        echo "Please refer to examples/vlm_ptq/requirements-vila.txt for the supported versions." >&2
        echo "You also need to download VILA repository from https://github.com/Efficient-Large-Model/VILA.git and checkout ec7fb2c264920bf004fd9fa37f1ec36ea0942db5" >&2
        exit 1
    fi

    pip install -r ../vlm_ptq/requirements-vila.txt
    # Clone original VILA repo
    if [ ! -d "$(dirname "$MODEL_PATH")/VILA" ]; then
        echo "VILA repository is needed until it is added to HF model zoo. Cloning the repository parallel to $MODEL_PATH..."
        git clone https://github.com/Efficient-Large-Model/VILA.git "$(dirname "$MODEL_PATH")/VILA" && \
	cd "$(dirname "$MODEL_PATH")/VILA" && \
	git checkout ec7fb2c264920bf004fd9fa37f1ec36ea0942db5 && \
	cd "$script_dir/.."
    fi
fi

if [[ $TASKS =~ "quant" ]] || [[ ! -d "$SAVE_PATH" ]] || [[ ! $(ls -A $SAVE_PATH) ]]; then
    if ! [ -f $MODEL_CONFIG ]; then
        echo "Quantizing original model..."
        python ../llm_ptq/hf_ptq.py \
            --pyt_ckpt_path=$MODEL_PATH \
            --export_path=$SAVE_PATH \
            --qformat=$QFORMAT \
            --calib_size=$CALIB_SIZE \
            --batch_size=$CALIB_BATCH_SIZE \
            --inference_tensor_parallel=$TP \
            --inference_pipeline_parallel=$PP \
            $PTQ_ARGS
    else
        echo "Quantized model config $MODEL_CONFIG exists, skipping the quantization stage"
    fi
fi

if [[ "$QFORMAT" != "fp8" ]]; then
    echo "For quant format $QFORMAT, please refer to the TensorRT-LLM documentation for deployment. Checkpoint saved to $SAVE_PATH."
    exit 0
fi

if [[ "$QFORMAT" == *"nvfp4"* ]] || [[ "$KV_CACHE_QUANT" == *"nvfp4"* ]]; then
    cuda_major=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0 | cut -d. -f1)

    if [ "$cuda_major" -lt 10 ]; then
        echo "Please deploy the NVFP4 checkpoint on a Blackwell GPU. Checkpoint export_path: $SAVE_PATH"
        exit 0
    fi
fi

# Prepare datasets for TRT-LLM benchmark
if [ -z "$TRT_LLM_CODE_PATH" ]; then
    TRT_LLM_CODE_PATH=/app/tensorrt_llm # default path for the TRT-LLM release docker image
    echo "Setting default TRT_LLM_CODE_PATH to $TRT_LLM_CODE_PATH."
fi

QUICK_START_MULTIMODAL=$TRT_LLM_CODE_PATH/examples/llm-api/quickstart_multimodal.py

if [ -f "$QUICK_START_MULTIMODAL" ]; then
    python3 $QUICK_START_MULTIMODAL --model_dir $SAVE_PATH --modality image
else
    echo "Warning: $QUICK_START_MULTIMODAL cannot be found. Please set TRT_LLM_CODE_PATH to the TRT-LLM code path or test the quantized checkpoint $SAVE_PATH with the TRT-LLM repo directly."
fi

popd
