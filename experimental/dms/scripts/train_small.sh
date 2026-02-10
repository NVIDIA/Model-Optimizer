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


# Example debug launch script for single-GPU training with limited memory.

set -e

CONFIG=${1:-configs/qwen3_1.7b.yaml}
test -f "$CONFIG" || { echo "Config not found: $CONFIG"; exit 1; }


# to handle limited memory on GPUs
export PYTORCH_ALLOC_CONF=expandable_segments:True


# Single-node, single-GPU FSDP configuration
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export CUDA_VISIBLE_DEVICES=0


echo "=== Launching single-GPU training with FSDP offloading ==="
python3 -m models.qwen3.train --config "$CONFIG"
