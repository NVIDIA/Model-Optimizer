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

set -euo pipefail

if (($# < 5 || $# > 6)); then
    echo "Usage: $0 SOURCE NEW_VENV TORCH_VERSION TRANSFORMERS_VERSION TORCHVISION_VERSION [--allow-cusparselt-sbsa]" >&2
    exit 2
fi
check_flags=()
if (($# == 6)); then
    [[ "$6" == --allow-cusparselt-sbsa ]] || { echo "Unknown option: $6" >&2; exit 2; }
    check_flags=(--allow-cusparselt-sbsa)
fi
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_dir=$(realpath "$1")
venv_dir=$2
torch_version=$3
transformers_version=$4
torchvision_version=$5
[[ ! -e "$venv_dir" ]] || { echo "Refusing to reuse an existing environment" >&2; exit 1; }
# Keep inherited vendor constraints and user packages outside this environment.
unset PIP_CONSTRAINT PIP_BUILD_CONSTRAINT PYTHONPATH PYTHONHOME
export PIP_CONFIG_FILE=/dev/null PYTHONNOUSERSITE=1
python3 -m venv "$venv_dir"
python="$venv_dir/bin/python"
"$python" -m pip install --no-compile --upgrade pip setuptools wheel "setuptools-scm>=8,<10"
constraint="$venv_dir/ptq-constraints.txt"
printf 'torch==%s\ntransformers==%s\ntorchvision==%s\n' \
    "$torch_version" "$transformers_version" "$torchvision_version" >"$constraint"
"$python" -m pip install --no-compile -c "$constraint" "torch==$torch_version" "torchvision==$torchvision_version"
# DeepSpeed builds no optional operators; its declared dependencies still resolve.
DS_BUILD_OPS=0 "$python" -m pip install --no-compile --no-build-isolation -c "$constraint" \
    "$source_dir[hf]" "transformers==$transformers_version"
# SDPA needs no FlashAttention. The example never imports the legacy streaming
# package, whose removed Transformers APIs are incompatible with Transformers 5.
requirements="$venv_dir/ptq-example-requirements.txt"
sed -e '/^flash-attn/d' -e '/^transformers_stream_generator$/d' "$source_dir/examples/hf_ptq/requirements.txt" >"$requirements"
"$python" -m pip install --no-compile -c "$constraint" -r "$requirements"
"$python" "$script_dir/verify_environment.py" --check-packages-only "${check_flags[@]}" >"$venv_dir/ptq-package-check.json"
"$python" -m pip freeze --all >"$venv_dir/ptq-packages.txt"
