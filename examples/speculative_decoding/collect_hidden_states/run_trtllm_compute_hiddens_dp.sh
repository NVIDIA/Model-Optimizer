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

# Example usage of the script to compute the hidden states for a conversation dataset
# This script computes hidden states using TensorRT-LLM and saves them to
# the specified output directory. It does so in a data-parallel manner across 8 GPUs, by splitting
# the input file into 8 parts and running 8 processes in parallel, one on each GPU.

# Note: depending on the write-throughput of the destination disk, this is not guaranteed
# to yield a speed improvement compared to running the model-parallel version. Consider
# benchmarking on a smaller dataset before launching a large run.

INPUT_FILE=synthetic_conversations/daring-anteater.jsonl
OUTPUT_DIR=/mnt/md0/eagle-hidden-states/llama1b/daring_anteater/
DP_SIZE=8
MODEL=meta-llama/Llama-3.2-1B-Instruct
export TLLM_LOG_LEVEL="error";

split -n l/$DP_SIZE --numeric-suffixes=0 -d --additional-suffix=.jsonl $INPUT_FILE /tmp/part-

for i in $(seq 0 $((DP_SIZE-1)))
do

export CUDA_VISIBLE_DEVICES=$i;  python3 collect_hidden_states/compute_hidden_states_trtllm.py --model $MODEL --input-file /tmp/part-0${i}.jsonl --output-dir $OUTPUT_DIR --dp-rank $i &

done
wait

rm /tmp/part-*.jsonl
