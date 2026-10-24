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
# Speculative-decoding benchmark with AIPerf (examples/specdec_aiperf): starts
# `vllm serve ${HF_MODEL_CKPT}` with the args before `--` plus per-request spec-decode metrics,
# then runs bench.py against it with the args after `--`.
#
# Required env: HF_MODEL_CKPT
# Optional env: AIPERF_VENV (default /tmp/aiperf-venv; created from the example's requirements.txt
#               when missing, so point it at a shared directory to install only once),
#               VLLM_STARTUP_TIMEOUT (seconds, default 1800)

EXAMPLE_DIR=modules/Model-Optimizer/examples/specdec_aiperf
AIPERF_VENV=${AIPERF_VENV:-/tmp/aiperf-venv}
PORT=8000

SERVE_ARGS=()
BENCH_ARGS=()
past_separator=false
for arg in "$@"; do
    if [ "$arg" = "--" ]; then
        past_separator=true
    elif [ "$past_separator" = true ]; then
        BENCH_ARGS+=("$arg")
    else
        SERVE_ARGS+=("$arg")
    fi
done

# A separate venv: installing AIPerf into an inference image can clash with its packages.
if [ ! -x "${AIPERF_VENV}/bin/aiperf" ]; then
    if ! { python3 -m venv "${AIPERF_VENV}" && \
           "${AIPERF_VENV}/bin/pip" install -q -r ${EXAMPLE_DIR}/requirements.txt; }; then
        report_result "FAIL: specdec_aiperf: installing AIPerf failed"
        exit 1
    fi
fi

vllm serve ${HF_MODEL_CKPT} --port ${PORT} --per-request-spec-decode-metrics summary \
    "${SERVE_ARGS[@]}" &
SERVER_PID=$!

MAX_WAIT=${VLLM_STARTUP_TIMEOUT:-1800}
WAITED=0
until curl -sf "http://localhost:${PORT}/health" >/dev/null; do
    if ! kill -0 ${SERVER_PID} 2>/dev/null || [ ${WAITED} -ge ${MAX_WAIT} ]; then
        report_result "FAIL: specdec_aiperf: vllm serve did not come up in ${WAITED}s"
        kill ${SERVER_PID} 2>/dev/null || true
        exit 1
    fi
    sleep 10
    WAITED=$((WAITED + 10))
    if [ $((WAITED % 60)) -eq 0 ]; then
        echo "Waiting for vllm serve: ${WAITED}s"
    fi
done

STATUS=PASS
if ! ${AIPERF_VENV}/bin/python ${EXAMPLE_DIR}/bench.py --url http://localhost:${PORT} \
        --tokenizer ${HF_MODEL_CKPT} "${BENCH_ARGS[@]}"; then
    STATUS=FAIL
fi

kill ${SERVER_PID} 2>/dev/null || true
wait ${SERVER_PID} 2>/dev/null || true
report_result "${STATUS}: specdec_aiperf"
[ "${STATUS}" = PASS ] || exit 1
