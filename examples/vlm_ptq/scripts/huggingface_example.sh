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

# DEPRECATED: examples/vlm_ptq has been consolidated into examples/hf_ptq.
# This shim forwards all arguments to the hf_ptq script with the --vlm flag so existing
# commands keep working. Please migrate to:
#
#   cd examples/hf_ptq
#   scripts/huggingface_example.sh --model <model> --quant <qformat> --vlm
#
# See examples/hf_ptq/README.md#vlm-quantization for details.

set -e

echo "WARNING: examples/vlm_ptq is deprecated and will be removed in a future release." >&2
echo "         Forwarding to examples/hf_ptq/scripts/huggingface_example.sh --vlm" >&2
echo "         See examples/hf_ptq/README.md#vlm-quantization" >&2

script_dir="$(dirname "$(readlink -f "$0")")"

exec "$script_dir/../../hf_ptq/scripts/huggingface_example.sh" --vlm "$@"
