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

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source ${SCRIPT_DIR}/../service_utils.sh

trap 'error_handler $0 $LINENO' ERR
trap 'exit_handler' EXIT

###################################################################################################
# Backend-agnostic specdec_bench entrypoint. The caller's YAML supplies --engine
# (VLLM | SGLANG | TRTLLM | NONE) and the dataset / sweep flags via "args"; this
# script just sources service_utils.sh, installs the speed-bench deps, and execs
# examples/specdec_bench/run.py.
#
# Required env: HF_MODEL_CKPT
# Optional env: HF_DRAFT_MODEL_CKPT (consumed by --draft_model_dir if the YAML passes it)

# Skip the install when the deps are already present (warm container or
# previous task in the pipeline). Saves a few minutes per task and avoids
# silently drifting versions if upstream wheels move between launches.
install_deps() {
    if ! pip show boto3 >/dev/null 2>&1 || \
       ! pip show datasets >/dev/null 2>&1 || \
       ! pip show seaborn >/dev/null 2>&1; then
        if ! pip install -r modules/Model-Optimizer/examples/specdec_bench/requirements.txt; then
            report_result "FAIL: specdec_bench: pip install requirements.txt failed"
            exit 1
        fi
    fi
}

if [[ -n "${SLURM_PROCID:-}" ]]; then
    flock /tmp/specdec_bench_deps.lock bash -c "$(declare -f install_deps); install_deps"
else
    install_deps
fi

# TensorRT-LLM jobs need to run through the launcher wrapper so spawned MPI
# workers inherit the Slurm/container context. Other backends leave this unset.
TRTLLM_LAUNCH_CMD=()
if [[ -n "${TRTLLM_LAUNCH_SCRIPT:-}" ]]; then
    # shellcheck disable=SC2206
    TRTLLM_LAUNCH_CMD=(${TRTLLM_LAUNCH_SCRIPT})
fi

if ! "${TRTLLM_LAUNCH_CMD[@]}" python3 modules/Model-Optimizer/examples/specdec_bench/run.py \
    --model_dir ${HF_MODEL_CKPT} \
    --tokenizer ${HF_MODEL_CKPT} \
    "${@}"; then
    report_result "FAIL: specdec_bench: run.py exited non-zero"
    exit 1
fi

report_result "PASS: specdec_bench run completed"
