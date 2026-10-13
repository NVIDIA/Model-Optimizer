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

# This script tests the end to end quantization process of ONNX models using the ONNX PTQ toolkit.
# It prepares calibration data, quantizes the models at different precisions,
# and evaluates all models in a specified folder on the ImageNet validation dataset.
# It is recommended to execute this script inside the Model Optimization Toolkit TensorRT Docker container.

# Usage: ./test_onnx_ptq.sh [--no-clean] [--eval] [/path/to/models] [/path/to/timing_cache]

set -exo pipefail

start_time=$(date +%s)
script_dir="$(dirname "$(readlink -f "$0")")"
public_example_dir=$script_dir/../../examples/onnx_ptq
test_utils_dir=$script_dir/../_test_utils/examples/onnx_ptq/
nvidia_gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -n 1)
cuda_capability=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d .)


pushd $public_example_dir

# Parse arguments
clean_mode=true
eval_mode=false
models_folder=""
timing_cache_path=""

for arg in "$@"; do
    case $arg in
        --no-clean)
            clean_mode=false
            shift
            ;;
        --eval)
            eval_mode=true
            shift
            ;;
        *)
            if [ -z "$models_folder" ]; then
                models_folder="$arg"
            elif [ -z "$timing_cache_path" ]; then
                timing_cache_path="$arg"
            fi
            shift
            ;;
    esac
done

export TQDM_DISABLE=1


# Setting model paths (contains 8 models)
models_folder=${models_folder:-/models/onnx}
timing_cache_path=${timing_cache_path:-/models/onnx/build/timing.cache}
calib_size=1
eval_size=100
batch_size=1


# Define quantization modes and base modes
base_modes=("fp16")
if [ $cuda_capability -ge 89 ]; then
    quant_modes=("fp8" "int8" "int8_iq")
else
    echo "CUDA capability is less than 89, skipping fp8 mode!"
    quant_modes=("int8" "int8_iq")
fi
all_modes=("${base_modes[@]}" "${quant_modes[@]}")


# Populate model_paths array with paths of all .onnx files
declare -a model_paths
for model in "$models_folder"/*.onnx; do
    model_paths+=("$model")
done

declare -A timm_model_name=(
    ["convnext_tiny_timm_opset13_simplified"]="convnext_tiny.in12k_ft_in1k"
    ["convnext_tiny_timm_opset17_simplified"]="convnext_tiny.in12k_ft_in1k"
    ["efficientformer_l3_opset13_simplified"]="efficientformer_l3.snap_dist_in1k"
    ["efficientformer_l3_opset17_simplified"]="efficientformer_l3.snap_dist_in1k"
    ["efficientnet_b0"]="efficientnet_b0.ra_in1k"
    ["efficientnet_b3"]="efficientnet_b3.ra2_in1k"
    ["efficientnet-lite4-11"]=""
    ["faster_vit_timm_opset13_simplified"]="fastvit_t8.apple_in1k"
    ["faster_vit_timm_opset17_simplified"]="fastvit_t8.apple_in1k"
    ["inception-v1-12"]="inception_v1"
    ["inception-v2-9"]="inception_v2"
    ["mobilenet_v1"]="mobilenetv1_100.ra4_e3600_r224_in1k"
    ["mobilenet_v3_opset13_simplified"]="mobilenetv3_large_100.ra_in1k"
    ["mobilenet_v3_opset17_simplified"]="mobilenetv3_large_100.ra_in1k"
    ["mobilenetv2-7"]="mobilenetv2_100.ra_in1k"
    ["resnet50-v1-12"]="resnet50.a1_in1k"
    ["resnet50-v2-7"]="resnetv2_50.a1h_in1k"
    ["swin_tiny_timm_opset13_simplified"]="swin_tiny_patch4_window7_224.ms_in22k"
    ["swin_tiny_timm_opset17_simplified"]="swin_tiny_patch4_window7_224.ms_in22k"
    ["vit_base_opset13_simplified"]="vit_base_patch16_224.augreg2_in21k_ft_in1k"
    ["vit_base_opset17_simplified"]="vit_base_patch16_224.augreg2_in21k_ft_in1k"
)

latency_models=("efficientnet_b0" "efficientnet_b3" "efficientnet-lite4-11" "faster_vit_timm_opset13_simplified" "faster_vit_timm_opset17_simplified" "inception-v1-12" "inception-v2-9")

# Create build directory to store all the results
rm -rf build
mkdir -p build

# Iterate over each model path to create directories for all modes for each model
for model_path in "${model_paths[@]}"; do
    model_name=$(basename "$model_path" .onnx)
    model_dir=build/$model_name

    for mode in "${all_modes[@]}"; do
        mkdir -p $model_dir/$mode
    done

    cp $model_path $model_dir/fp16/model.onnx
done


# Prepare calibration data for quantization at different precisions
calib_data_path="build/calib.$calib_size.npy"
python image_prep.py --output_path=$calib_data_path --calibration_data_size=$calib_size


# Perform model quantization for each model in each mode
for model_path in "${model_paths[@]}"; do
    model_name=$(basename "$model_path" .onnx)
    model_dir=build/$model_name

    echo "Quantizing model $model_name for all quantization modes"
    for quant_mode in "${quant_modes[@]}"; do
        if [ "$quant_mode" == "int8_iq" ]; then
            continue
        fi

        echo "Starting quantization of $model_name for mode: $quant_mode"
        python -m modelopt.onnx.quantization \
            --onnx_path=$model_dir/fp16/model.onnx \
            --quantize_mode=$quant_mode \
            --calibration_data=$calib_data_path \
            --output_path=$model_dir/$quant_mode/model.quant.onnx \
            --calibration_eps=cuda
    done

    echo "Completed quantization of all modes for model: $model_name"
done


# Evaluate the quantized models for each mode
if [ "$eval_mode" = true ]; then
    for model_path in "${model_paths[@]}"; do
        model_name=$(basename "$model_path" .onnx)
        model_dir=build/$model_name

        echo "Evaluating model $model_name for all quantization modes"
        for quant_mode in "${all_modes[@]}"; do
            if [ "$quant_mode" == "fp16" ]; then
                eval_model_path=$model_dir/fp16/model.onnx
                engine_path=$model_dir/fp16/model.engine
                precision="fp16"
            elif [ "$quant_mode" == "int8_iq" ]; then
                eval_model_path=$model_dir/fp16/model.onnx
                engine_path=$model_dir/int8_iq/model.engine
                precision="best"
            else
                eval_model_path=$model_dir/$quant_mode/model.quant.onnx
                engine_path=$model_dir/$quant_mode/model.quant.engine
                precision="stronglyTyped"
            fi

            echo "Starting evaluation of $model_name for mode: $quant_mode"
            if [[ " ${latency_models[@]} " =~ " $model_name " ]]; then
                python evaluate.py \
                    --onnx_path=$eval_model_path \
                    --engine_path=$engine_path \
                    --model_name="${timm_model_name[$model_name]}" \
                    --engine_precision=$precision \
                    --results_path=$model_dir/$quant_mode/${model_name}_${quant_mode}.csv \
                    --timing_cache_path=$timing_cache_path
            else
                python evaluate.py \
                    --onnx_path=$eval_model_path \
                    --engine_path=$engine_path \
                    --eval_data_size=$eval_size \
                    --batch_size $batch_size \
                    --model_name="${timm_model_name[$model_name]}" \
                    --engine_precision=$precision \
                    --results_path=$model_dir/$quant_mode/${model_name}_${quant_mode}.csv \
                    --timing_cache_path=$timing_cache_path
            fi
        done

        echo "Completed evaluation of all modes for model: $model_name"
    done

    python $test_utils_dir/aggregate_results.py --results_dir=build
fi

if [ "$clean_mode" = true ]; then
    echo "Cleaning build artifacts..."
    rm -rf build/
    echo "Build artifacts cleaned successfully."
fi

popd


total_seconds=$(($(date +%s) - start_time))
printf "Total wall time: %02d:%02d:%02d\n" $((total_seconds/3600)) $(((total_seconds%3600)/60)) $((total_seconds%60))
