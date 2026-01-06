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

source $script_dir/parser.sh
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

# Check if ENABLE_SPARSITY environment variable is set to "true"
if [ "$SPARSITY_FMT" = "dense" ]; then
    ENABLE_SPARSITY=false
else
    ENABLE_SPARSITY=true
fi

case $SPARSITY_FMT in
dense | sparsegpt) ;;
*)
    echo "Unknown sparsity argument: Expected one of: [dense, sparsegpt]" >&2
    exit 1
    ;;
esac

#Iterate over list of qformats provided and check if they are valid
IFS=","
for qformat in $QFORMAT; do
    case $qformat in
    fp8 | fp8_pc_pt | fp8_pb_wo | int8_wo | int8_sq | int4_awq | w4a8_awq | fp16 | bf16 | nvfp4 | nvfp4_awq | w4a8_nvfp4_fp8 | w4a8_mxfp4_fp8 | nvfp4_mlp_only | nvfp4_svdquant | mxfp8) ;;
    *)
        echo "Unknown quant argument: Expected one of: [fp8, fp8_pc_pt, fp8_pb_wo, int8_wo, int8_sq, int4_awq, w4a8_awq, fp16, bf16, nvfp4, nvfp4_awq, w4a8_nvfp4_fp8, w4a8_mxfp4_fp8, nvfp4_mlp_only, nvfp4_svdquant, mxfp8]" >&2
        exit 1
        ;;
    esac
done
IFS=" "

script_dir="$(dirname "$(readlink -f "$0")")"

pushd $script_dir/..

if [ -z "$ROOT_SAVE_PATH" ]; then
    ROOT_SAVE_PATH=$(pwd)
fi

QFORMAT_MODIFIED="${QFORMAT//,/_}"

MODEL_NAME=$(basename $MODEL_PATH | sed 's/[^0-9a-zA-Z\-]/_/g')_${QFORMAT_MODIFIED}${KV_CACHE_QUANT:+_kv_${KV_CACHE_QUANT}}

SAVE_PATH=${ROOT_SAVE_PATH}/saved_models_${MODEL_NAME}

MODEL_CONFIG=${SAVE_PATH}/config.json

mkdir -p $SAVE_PATH

if [ "${REMOVE_EXISTING_MODEL_CONFIG,,}" = "true" ]; then
    rm -f $MODEL_CONFIG
fi

PTQ_ARGS=""

if [ "$LOW_MEMORY_MODE" = "true" ]; then
    PTQ_ARGS+=" --low_memory_mode "
fi

if [ -n "$AUTO_QUANTIZE_BITS" ]; then
    PTQ_ARGS+=" --auto_quantize_bits=$AUTO_QUANTIZE_BITS "
fi

if [ -n "$AUTO_QUANTIZE_METHOD" ]; then
    PTQ_ARGS+=" --auto_quantize_method=$AUTO_QUANTIZE_METHOD "
fi

if [ -n "$AUTO_QUANTIZE_SCORE_SIZE" ]; then
    PTQ_ARGS+=" --auto_quantize_score_size=$AUTO_QUANTIZE_SCORE_SIZE "
fi

# Automatically generate auto_quantize checkpoint path if not provided
if [ -n "$AUTO_QUANTIZE_BITS" ] && [ -z "$AUTO_QUANTIZE_CHECKPOINT" ]; then
    # Create a descriptive checkpoint name based on model and quantization settings
    AQ_METHOD=${AUTO_QUANTIZE_METHOD:-gradient}
    AUTO_QUANTIZE_CHECKPOINT="${ROOT_SAVE_PATH}/auto_quantize_checkpoints/${MODEL_NAME}_${AQ_METHOD}.pth"
    mkdir -p $(dirname $AUTO_QUANTIZE_CHECKPOINT)
    echo "Auto-generated auto_quantize checkpoint path: $AUTO_QUANTIZE_CHECKPOINT"
fi

if [ -n "$AUTO_QUANTIZE_BITS" ]; then
    PTQ_ARGS+=" --auto_quantize_checkpoint=$AUTO_QUANTIZE_CHECKPOINT "
fi

if [ -n "$CALIB_DATASET" ]; then
    PTQ_ARGS+=" --dataset=$CALIB_DATASET "
fi

if [ -n "$KV_CACHE_QUANT" ]; then
    PTQ_ARGS+=" --kv_cache_qformat=$KV_CACHE_QUANT "
fi

if $TRUST_REMOTE_CODE; then
    PTQ_ARGS+=" --trust_remote_code "
fi

if $USE_SEQ_DEVICE_MAP; then
    PTQ_ARGS+=" --use_seq_device_map "
fi

if [ -n "$GPU_MAX_MEM_PERCENTAGE" ]; then
    PTQ_ARGS+=" --gpu_max_mem_percentage=$GPU_MAX_MEM_PERCENTAGE "
fi

if [ -n "$CALIB_SEQ" ]; then
    PTQ_ARGS+=" --calib_seq=$CALIB_SEQ "
fi

if ! $VERBOSE; then
    PTQ_ARGS+=" --no-verbose "
fi

AWQ_ARGS=""
if [ -n "$AWQ_BLOCK_SIZE" ]; then
    AWQ_ARGS+="--awq_block_size=$AWQ_BLOCK_SIZE"
fi

if [[ -f $MODEL_CONFIG ]] || [[ -f "$SAVE_PATH/encoder/config.json" && -f "$SAVE_PATH/decoder/config.json" ]]; then
    MODEL_CONFIG_EXIST=true
else
    MODEL_CONFIG_EXIST=false
fi

if [[ $TASKS =~ "quant" ]] || [[ ! -d "$SAVE_PATH" ]] || [[ ! $(ls -A $SAVE_PATH) ]]; then

    if [ "$qformat" == "bf16" ] || [ "$qformat" == "fp16" ]; then
        if [ -d "$MODEL_PATH" ]; then
            MODEL_CONFIG_EXIST=true
            MODEL_CONFIG=$MODEL_PATH/config.json
            for file in $MODEL_PATH/*; do ln -sf "$file" $SAVE_PATH/; done
        else
            echo "Please use the model directory where the config.json file is present."
            exit 1
        fi
    fi

    if [[ "$MODEL_CONFIG_EXIST" == false ]]; then
        echo "Quantizing original model..."
        python hf_ptq.py \
            --pyt_ckpt_path=$MODEL_PATH \
            --export_path=$SAVE_PATH \
            --sparsity_fmt=$SPARSITY_FMT \
            --qformat="${QFORMAT// /,}" \
            --calib_size=$CALIB_SIZE \
            --batch_size=$CALIB_BATCH_SIZE \
            --inference_tensor_parallel=$TP \
            --inference_pipeline_parallel=$PP \
            $PTQ_ARGS \
            $AWQ_ARGS
    else
        echo "Quantized model config $MODEL_CONFIG exists, skipping the quantization stage"
    fi

    # for enc-dec model, users need to refer TRT-LLM example for deployment
    if [[ -f "$SAVE_PATH/encoder/config.json" && -f "$SAVE_PATH/decoder/config.json" && ! -f $MODEL_CONFIG ]]; then
        echo "Please continue to deployment with the TRT-LLM enc_dec example, https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/enc_dec. Checkpoint export_path: $SAVE_PATH"
        exit 0
    fi

    if [[ "$SPARSITY_FMT" != "dense" ]]; then
        echo "Sparse quantization detected (SPARSITY_FMT=$SPARSITY_FMT). Please deploy with the TRT-LLM using trtllm-build. Checkpoint export_path: $SAVE_PATH"
        exit 0
    fi

    if [[ "$QFORMAT" == *"nvfp4"* ]] || [[ "$KV_CACHE_QUANT" == *"nvfp4"* ]]; then
        cuda_major=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0 | cut -d. -f1)

        if [ "$cuda_major" -lt 10 ]; then
            echo "Please deploy the NVFP4 checkpoint on a Blackwell GPU. Checkpoint export_path: $SAVE_PATH"
            exit 0
        fi
    fi

    if [[ ! " fp8 nvfp4 bf16 fp16 " =~ " ${QFORMAT} " ]]; then
        echo "Quant $QFORMAT specified. Please read TensorRT-LLM quantization support matrix https://nvidia.github.io/TensorRT-LLM/features/quantization.html#quantization-in-tensorrt-llm and use TensorRT-LLM for deployment. Checkpoint export_path: $SAVE_PATH"
        exit 0
    fi

    if $TRUST_REMOTE_CODE; then
        RUN_ARGS+=" --trust_remote_code "
    fi

    python run_tensorrt_llm.py --checkpoint_dir=$SAVE_PATH $RUN_ARGS
fi

if [[ -d "${MODEL_PATH}" ]]; then
    MODEL_ABS_PATH=$(realpath ${MODEL_PATH})
else
    # model_path is a HF reference, not a local directory, no need to make the path absolute
    MODEL_ABS_PATH=${MODEL_PATH}
fi

if [[ $TASKS =~ "lm_eval" ]]; then

    if [ -z "$LM_EVAL_TASKS" ]; then
        echo "lm_eval_tasks not specified"
        exit 1
    fi

    lm_eval_flags=""
    if [[ "$LM_EVAL_TASKS" == *"llama"* ]]; then
        # Flags instructed by https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/llama3#paper
        lm_eval_flags+=" --fewshot_as_multiturn --apply_chat_template "
    fi

    if [ -n "$LM_EVAL_LIMIT" ]; then
        lm_eval_flags+=" --limit $LM_EVAL_LIMIT "
    fi

    if $TRUST_REMOTE_CODE; then
        lm_eval_flags+=" --trust_remote_code "
    fi

    LM_EVAL_RESULT=${SAVE_PATH}/lm_eval.txt
    echo "Evaluating lm_eval, result saved to $LM_EVAL_RESULT..."

    pushd ../llm_eval/

    pip install -r requirements.txt

    echo "Using the following config: max output $BUILD_MAX_OUTPUT_LEN max batch $BUILD_MAX_BATCH_SIZE"

    python lm_eval_tensorrt_llm.py \
        --model trt-llm \
        --model_args tokenizer=$MODEL_PATH,checkpoint_dir=$SAVE_PATH,max_gen_toks=$BUILD_MAX_OUTPUT_LEN \
        --tasks $LM_EVAL_TASKS \
        --batch_size $BUILD_MAX_BATCH_SIZE $lm_eval_flags | tee $LM_EVAL_RESULT

    popd

fi

if [[ $TASKS =~ "mmlu" ]]; then

    MMLU_RESULT=${SAVE_PATH}/mmlu.txt
    echo "Evaluating MMLU, result saved to $MMLU_RESULT..."

    pushd ../llm_eval/

    pip install -r requirements.txt

    if [ -z "$MMLU_DATA_PATH" ]; then
        MMLU_DATA_PATH=data/mmlu
    fi
    if [[ ! -d "$MMLU_DATA_PATH" ]] || [[ ! $(ls -A $MMLU_DATA_PATH) ]]; then
        echo "Preparing the MMLU test data"
        wget https://people.eecs.berkeley.edu/~hendrycks/data.tar -O /tmp/mmlu.tar
        mkdir -p data
        tar -xf /tmp/mmlu.tar -C data && mv data/data $MMLU_DATA_PATH
    fi

    python mmlu.py \
        --model_name causal \
        --model_path $MODEL_ABS_PATH \
        --checkpoint_dir $SAVE_PATH \
        --data_dir $MMLU_DATA_PATH | tee $MMLU_RESULT
    popd

fi

if [[ $TASKS =~ "mtbench" ]]; then

    pushd ../llm_eval/

    bash run_fastchat.sh -h $MODEL_ABS_PATH -e $SAVE_PATH
    find data/mt_bench/model_answer/ -type f -name '*.jsonl' -exec mv {} $SAVE_PATH \;

    JSONL_PATH=$(readlink -f $(find $SAVE_PATH -type f -name '*.jsonl'))
    echo "FastChat generation complete. The results are saved under $JSONL_PATH . Please run the judge(https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge) to evaluate the quality of the responses."

    popd

fi

if [[ $TASKS =~ "livecodebench" || $TASKS =~ "simple_eval" ]]; then
    # Clean a previous session if exists
    pkill -f "trtllm-serve" && while pgrep -f "trtllm-serve" >/dev/null; do sleep 1; done
    HASH=$(echo -n "$SAVE_PATH" | md5sum | awk '{print $1}')
    PORT=$((10000 + (0x${HASH:0:4} % 50001)))
    echo "Starting trtllm-serve on $PORT"
    trtllm-serve $SAVE_PATH --host 0.0.0.0 --port $PORT >$SAVE_PATH/serve.txt 2>&1 &
    SERVE_PID=$!

    tail -f $SAVE_PATH/serve.txt | while read line; do
        if echo "$line" | grep -q "Application startup complete"; then
            echo "Application startup complete."
            break
        fi
        if ! kill -0 $SERVE_PID 2>/dev/null; then
            echo "trtllm-serve has exited."
            exit 1
        fi
    done

    pushd ../llm_eval/

    if [[ $TASKS =~ "livecodebench" ]]; then
        echo "Using the following config: max output $BUILD_MAX_OUTPUT_LEN max batch $BUILD_MAX_BATCH_SIZE"
        bash run_livecodebench.sh $MODEL_NAME $BUILD_MAX_BATCH_SIZE $BUILD_MAX_OUTPUT_LEN $PORT | tee $SAVE_PATH/livecodebench.txt
        mkdir -p $SAVE_PATH/livecodebench
        mv LiveCodeBench/output/$MODEL_NAME/* $SAVE_PATH/livecodebench
        echo "LiveCodeBench results are saved under $SAVE_PATH/livecodebench."

    fi

    if [[ $TASKS =~ "simple_eval" ]]; then
        echo "Using the following config: max output $BUILD_MAX_OUTPUT_LEN max batch $BUILD_MAX_BATCH_SIZE"
        bash run_simple_eval.sh $MODEL_NAME $SIMPLE_EVAL_TASKS $BUILD_MAX_OUTPUT_LEN $PORT | tee $SAVE_PATH/simple_eval.txt
        echo "Simple eval results are saved under $SAVE_PATH/simple_eval.txt."
    fi

    popd

    kill $SERVE_PID
fi


popd
