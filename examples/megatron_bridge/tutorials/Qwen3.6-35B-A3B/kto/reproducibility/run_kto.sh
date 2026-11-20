#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Run KTO using NeMo RL's preference-training loop.
# Fill in the variables below before launching with the bundled kto.yaml.
# The bundled quantized recipe needs a checkpoint containing ModelOpt quantization state.
#
# Example, from the parent tutorial directory after filling in the variables:
#   bash kto/reproducibility/run_kto.sh

# Stop on command errors, unset variables and pipeline failures.
set -euo pipefail

# NeMo RL source tree containing examples/configs/dpo.yaml.
export NEMO_RL_SOURCE=/path/to/nemo-rl
# Hugging Face model ID or local architecture/tokenizer directory.
export HF_MODEL=/path/to/model-and-tokenizer
# Starting Megatron checkpoint, including ModelOpt state for quantized training.
export PRETRAINED_CHECKPOINT=/path/to/pretrained-megatron-checkpoint
# Training JSONL: one correct and one looping response per row.
# Format example: data/training.example.jsonl.
export KTO_TRAIN_DATA=/path/to/training.jsonl
# Validation JSONL used for validation-loss checks.
export KTO_VALIDATION_DATA=/path/to/validation.jsonl
# Output directory for checkpoints and logs.
export KTO_OUTPUT_DIR=/path/to/new-kto-output

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python "$SCRIPT_DIR/kto_train.py" --dpo-config "$NEMO_RL_SOURCE/examples/configs/dpo.yaml" "$@"
