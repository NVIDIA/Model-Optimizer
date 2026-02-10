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

# DMS adapter training script for Qwen3-8B.
#
# Usage:
#   bash scripts/train.sh configs/qwen3_8b.yaml
#
# This first prepares the dataset with a single process,
# then launches distributed training with accelerate.

set -e

CONFIG=${1:-configs/qwen3_8b.yaml}
test -f "$CONFIG" || { echo "Config not found: $CONFIG"; exit 1; }

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Preparing dataset (single process) ==="
python -m models.qwen3.train --config "$CONFIG" --prepare-dataset-only

echo "=== Launching distributed training ==="
accelerate launch -m models.qwen3.train --config "$CONFIG"
