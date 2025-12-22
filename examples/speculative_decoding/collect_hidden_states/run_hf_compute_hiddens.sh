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
# This script computes hidden states using a Hugging Face model and saves them to
# the specified output directory.

python3 collect_hidden_states/compute_hidden_states_hf.py \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --input-file synthetic_conversations/daring-anteater.jsonl \
  --output-dir /mnt/md0/eagle-hidden-states/llama1b/daring_anteater/