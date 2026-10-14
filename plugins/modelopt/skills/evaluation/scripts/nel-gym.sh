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

# Exact known-good pin for the NeMo Gym path; follow references/gym.md before
# bumping it.
readonly NEL_GYM_VERSION="0.2.6"
readonly NEL_GYM_SPEC="nemo-evaluator-launcher[all]==${NEL_GYM_VERSION}"

case "${1:-}" in
  -h|--help)
    echo "usage: nel-gym.sh <nel arguments>"
    echo "       nel-gym.sh --version"
    exit 0
    ;;
esac

command -v uvx >/dev/null 2>&1 || {
  echo "ERROR: 'uvx' is required to run the pinned NeMo Gym launcher" >&2
  exit 1
}

exec uvx --python 3.10 --from "$NEL_GYM_SPEC" nel "$@"
