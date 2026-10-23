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
# specdec_bench through AIPerf: starts one `run.py --serve` vLLM server from the YAML's args, then
# runs `run.py --aiperf_url` against it for every temperature in TEMPERATURES and concurrency in
# CONCURRENCIES, writing <--save_dir>/t<T>_c<C>/. The engine is built once for all of them.
#
# Required env: HF_MODEL_CKPT
# Optional env: TEMPERATURES (default "0"), CONCURRENCIES (default "32"),
#               SERVE_ARGS (extra `vllm serve` flags), AIPERF_VENV (default /tmp/aiperf-venv,
#               created with aiperf==${AIPERF_VERSION:-0.13.0} when missing)

RUN_PY=modules/Model-Optimizer/examples/specdec_bench/run.py
TEMPERATURES=${TEMPERATURES:-0}
CONCURRENCIES=${CONCURRENCIES:-32}
AIPERF_VENV=${AIPERF_VENV:-/tmp/aiperf-venv}
PORT=8000

SAVE_DIR=.
ARGS=()
while [ $# -gt 0 ]; do
    if [ "$1" = "--save_dir" ]; then
        SAVE_DIR=$2
        shift 2
    else
        ARGS+=("$1")
        shift
    fi
done

if ! pip show boto3 >/dev/null 2>&1 || \
   ! pip show datasets >/dev/null 2>&1 || \
   ! pip show seaborn >/dev/null 2>&1; then
    if ! pip install -r modules/Model-Optimizer/examples/specdec_bench/requirements.txt; then
        report_result "FAIL: specdec_bench aiperf: pip install requirements.txt failed"
        exit 1
    fi
fi

# A separate venv: installing aiperf into an inference image can clash with its packages.
if [ ! -x "${AIPERF_VENV}/bin/aiperf" ]; then
    if ! { python3 -m venv "${AIPERF_VENV}" && \
           "${AIPERF_VENV}/bin/pip" install -q "aiperf==${AIPERF_VERSION:-0.13.0}"; }; then
        report_result "FAIL: specdec_bench aiperf: installing aiperf failed"
        exit 1
    fi
fi

MAX_CONCURRENCY=$(printf '%s\n' ${CONCURRENCIES} | sort -n | tail -1)
python3 ${RUN_PY} --serve --port ${PORT} --model_dir ${HF_MODEL_CKPT} --tokenizer ${HF_MODEL_CKPT} \
    "${ARGS[@]}" --concurrency ${MAX_CONCURRENCY} ${SERVE_ARGS} &
SERVER_PID=$!

MAX_WAIT=${VLLM_STARTUP_TIMEOUT:-1800}
WAITED=0
until curl -sf "http://localhost:${PORT}/health" >/dev/null; do
    if ! kill -0 ${SERVER_PID} 2>/dev/null || [ ${WAITED} -ge ${MAX_WAIT} ]; then
        report_result "FAIL: specdec_bench aiperf: vllm serve did not come up in ${WAITED}s"
        kill ${SERVER_PID} 2>/dev/null || true
        exit 1
    fi
    sleep 10
    WAITED=$((WAITED + 10))
done

STATUS=PASS
for T in ${TEMPERATURES}; do
    for C in ${CONCURRENCIES}; do
        if ! python3 ${RUN_PY} --aiperf_url http://localhost:${PORT} \
                --aiperf_bin ${AIPERF_VENV}/bin/aiperf \
                --model_dir ${HF_MODEL_CKPT} --tokenizer ${HF_MODEL_CKPT} "${ARGS[@]}" \
                --temperature ${T} --concurrency ${C} --save_dir ${SAVE_DIR}/t${T}_c${C}; then
            report_result "FAIL: specdec_bench aiperf: temperature ${T}, concurrency ${C}"
            STATUS=FAIL
        fi
    done
done

kill ${SERVER_PID} 2>/dev/null || true
wait ${SERVER_PID} 2>/dev/null || true
[ ${STATUS} = PASS ] || exit 1
report_result "PASS: specdec_bench aiperf runs completed"
