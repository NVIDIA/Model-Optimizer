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

set -ex

FLOPS_PERCENT=50
BASE_DIR=results/bert_large_pruned_${FLOPS_PERCENT}_percent

# Export to ONNX on a single GPU
python3 bert_prune_distill_quantize.py \
    --model_name_or_path ${BASE_DIR}/int8_qat/ \
    --onnx_export_path ${BASE_DIR}/pruned_model_int8.onnx \
