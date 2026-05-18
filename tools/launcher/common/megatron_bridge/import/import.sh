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

# Megatron-Bridge HF -> Megatron checkpoint import (CPU-capable).
#
# Required env: HF_MODEL_ID  (e.g. nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16)
# Optional env:
#   OUTPUT_DIR  Parent dir for the MCore checkpoint (default: cwd).
#   TORCH_DTYPE Model dtype for HF load (default: bfloat16).
#
# Writes MCore checkpoint to ${OUTPUT_DIR}/<basename(HF_MODEL_ID)>-MCore
#
# Runs:
#   python examples/conversion/convert_checkpoints.py import \
#       --hf-model $HF_MODEL_ID \
#       --megatron-path $OUTPUT_DIR/<model>-MCore \
#       --torch-dtype $TORCH_DTYPE

set -e

if [[ -z "${HF_MODEL_ID}" ]]; then
    echo "[ERROR] HF_MODEL_ID is required" >&2
    exit 1
fi

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
LAUNCHER_DIR="${SCRIPT_DIR}/../../.."
BRIDGE_DIR="${LAUNCHER_DIR}/modules/Megatron-Bridge"
MLM_DIR="${LAUNCHER_DIR}/modules/Megatron-LM"

if ! python -c "import megatron.bridge" 2>/dev/null; then
    echo "[INFO] Installing megatron-bridge from ${BRIDGE_DIR}"
    unset PIP_CONSTRAINT
    pip install -e "${BRIDGE_DIR}"
fi

if [[ -n "${EXTRA_PIP_DEPS}" ]]; then
    echo "[INFO] Installing extra deps: ${EXTRA_PIP_DEPS}"
    unset PIP_CONSTRAINT
    read -r -a _deps <<< "${EXTRA_PIP_DEPS}"
    # --no-build-isolation: mamba-ssm/causal-conv1d need torch visible at build time.
    pip install --no-build-isolation "${_deps[@]}"
fi

# Megatron-Bridge needs newer megatron.core (incl. megatron.core.distributed.fsdp).
# Prepend local Megatron-LM to PYTHONPATH so its sources shadow installed megatron-core.
export PYTHONPATH="${MLM_DIR}:${PYTHONPATH}"

OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)}"
MODEL_NAME="$(basename "${HF_MODEL_ID}")"
MEGATRON_PATH="${OUTPUT_DIR}/${MODEL_NAME}-MCore"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"

mkdir -p "${OUTPUT_DIR}"

cd "${BRIDGE_DIR}"
exec python examples/conversion/convert_checkpoints.py import \
    --hf-model "${HF_MODEL_ID}" \
    --megatron-path "${MEGATRON_PATH}" \
    --torch-dtype "${TORCH_DTYPE}"
