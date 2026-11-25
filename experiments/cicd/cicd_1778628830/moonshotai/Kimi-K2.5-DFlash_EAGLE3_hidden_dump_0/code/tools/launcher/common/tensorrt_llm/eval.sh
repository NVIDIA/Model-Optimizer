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

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

###################################################################################################

if [[ -z ${HF_MODEL_CKPT} ]]; then
    export HF_MODEL_CKPT=/scratchspace/export
fi

if [[ -z ${TP} ]]; then
    TP=4
fi

if [[ -z ${EP} ]]; then
    EP=4
fi

if [[ -z ${EXTRA_LLM_API_OPTIONS} ]]; then
    EXTRA_LLM_API_OPTIONS=common/tensorrt_llm/extra_llm_api_options.yaml
fi


TARGET_FILENAME="config.json"


# Find all files matching the target filename, print their paths null-terminated
find "${HF_MODEL_CKPT}" -type f -name "$TARGET_FILENAME" -print0 | while IFS= read -r -d '' filepath; do
    # Extract the directory path from the full file path
    dir_path=$(dirname "$filepath")
    
    echo "Processing model: $dir_path"
    # Place your commands here to run within or on the $dir_path
    # Example: cd "$dir_path" && some_command

    trtllm-llmapi-launch trtllm-eval \
        --model ${dir_path} \
        --disable_kv_cache_reuse \
        --tp_size ${TP} \
        --ep_size ${EP} \
        --trust_remote_code \
        --extra_llm_api_options ${EXTRA_LLM_API_OPTIONS} \
        mmlu
done
