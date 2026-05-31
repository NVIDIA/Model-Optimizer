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

# Megatron-Bridge HF -> Megatron checkpoint import.
# Assumes nvcr.io/nvidia/nemo:26.02+ container (megatron-bridge preinstalled at /opt/Megatron-Bridge).
#
# Required env: HF_MODEL_ID  (e.g. nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16)
# Optional env:
#   OUTPUT_DIR  Parent dir for the MCore checkpoint (default: cwd).
#   TORCH_DTYPE Model dtype for HF load (default: bfloat16).
#
# Writes MCore checkpoint to ${OUTPUT_DIR}/<basename(HF_MODEL_ID)>-MCore

set -e

if [[ -z "${HF_MODEL_ID}" ]]; then
    echo "[ERROR] HF_MODEL_ID is required" >&2
    exit 1
fi

OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)}"
MODEL_NAME="$(basename "${HF_MODEL_ID}")"
MEGATRON_PATH="${OUTPUT_DIR}/${MODEL_NAME}-MCore"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"

mkdir -p "${OUTPUT_DIR}"

exec python /opt/Megatron-Bridge/examples/conversion/convert_checkpoints.py import \
    --hf-model "${HF_MODEL_ID}" \
    --megatron-path "${MEGATRON_PATH}" \
    --torch-dtype "${TORCH_DTYPE}"
