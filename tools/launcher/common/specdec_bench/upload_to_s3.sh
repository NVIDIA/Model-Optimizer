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

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source ${SCRIPT_DIR}/../service_utils.sh

trap 'error_handler $0 $LINENO' ERR
trap 'exit_handler' EXIT

###################################################################################################
# Upload a specdec_bench results directory to S3. Thin wrapper around
# examples/specdec_bench/upload_to_s3.py.
#
# YAML usage:
#   task_2:
#     script: common/specdec_bench/upload_to_s3.sh
#     args:
#       - /scratchspace/specdec_bench
#       - s3://team-specdec-workgroup/results
#       - --skip-existing                  # optional
#       - --allow-incomplete-provenance    # optional, for runs without CONTAINER_IMAGE set
#
# Required env (or pass via --endpoint / --key-id / --secret to the underlying script):
#   S3_ENDPOINT, S3_KEY_ID, S3_SECRET

# Install boto3 if not already in the container. Warm pipelines where an
# earlier specdec_bench task ran will already have it from run.sh.
if ! pip show boto3 >/dev/null 2>&1; then
    if ! pip install -r modules/Model-Optimizer/examples/specdec_bench/requirements.txt; then
        report_result "FAIL: upload_to_s3: pip install requirements.txt failed"
        exit 1
    fi
fi

if ! python3 modules/Model-Optimizer/examples/specdec_bench/upload_to_s3.py "${@}"; then
    report_result "FAIL: upload_to_s3: upload_to_s3.py exited non-zero"
    exit 1
fi

report_result "PASS: upload_to_s3 completed"
