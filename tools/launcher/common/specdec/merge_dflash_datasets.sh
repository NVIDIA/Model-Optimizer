#!/bin/bash

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

# Merge generated sources into one DFlash training JSONL.
#
# Resolves every image/video reference to an absolute path (training then runs
# with data.vlm_img_dir=/) and drops near-duplicate completions, which the
# temperature sweep produces in bulk.
#
# All args pass through to merge_dflash_datasets.py, so sources stay in the YAML:
#   script: common/specdec/merge_dflash_datasets.sh
#   args:
#     - --source pai_understanding=/scratchspace/data/pai_outputs
#     - --media-root pai_understanding=/scratchspace/data/pai_understanding
#     - --output /scratchspace/data/train.jsonl
#     - --jobs 8

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source ${SCRIPT_DIR}/../service_utils.sh

trap 'error_handler $0 $LINENO' ERR
trap 'exit_handler' EXIT

set -euo pipefail

MERGE=modules/Model-Optimizer/examples/speculative_decoding/recipes/merge_dflash_datasets.py

set -x
python3 "$MERGE" "$@"
set +x

# Surface the merged row count; a silent drop to near-zero means the sources
# were empty or every record failed media resolution.
prev_arg=""
for arg in "$@"; do
    [ "$prev_arg" = "--output" ] && wc -l "$arg"
    prev_arg="$arg"
done
