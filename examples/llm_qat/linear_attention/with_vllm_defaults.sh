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

if [[ $# -eq 0 ]]; then
    echo "Usage: bash $0 <command> [args...]" >&2
    exit 2
fi

# Establish the same arithmetic before training or serving imports vLLM.
for setting in \
    FLA_USE_FAST_OPS=0 \
    USE_DEFAULT_FLA_NORM=0 \
    FLA_GDN_FIX_BT=0 \
    FLA_USE_CUDA_GRAPH=0 \
    FLA_TRIL_PRECISION=ieee; do
    export "$setting"
    printf '%s\n' "$setting" >&2
done

exec "$@"
