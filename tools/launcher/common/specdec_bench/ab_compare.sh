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
# Matched A|B speculative-decoding benchmark in one job, so both arms share the node, image,
# prompts and sampling. For each repeat and concurrency it runs examples/specdec_bench/run.py
# twice: A = target alone (--speculative_algorithm NONE) and B = target + draft, A first on
# odd repeats and B first on even ones, so run order does not favor one arm. It then
# writes ${SAVE_ROOT}/ab_summary.json via examples/specdec_bench/compare_ab.py (speedup, AL,
# and A-vs-B exact match at concurrency 1 and temperature 0). Fails if an arm fails or the
# summary lists problems (missing arm, appended results, turn count mismatch).
#
# Required env:
#   HF_MODEL_CKPT    target model
#   DRAFT_MODEL      exported draft checkpoint
#   SAVE_ROOT        output root; results land in ${SAVE_ROOT}/r<repeat>/{A,B}_c<concurrency>/
# Optional env:
#   ENGINE           run.py --engine (default: VLLM)
#   SPEC_ALGO        run.py --speculative_algorithm for arm B (default: DFLASH)
#   NUM_SPEC_TOKENS  speculative tokens; --block_size for DFLASH/DSPARK, else --draft_length
#                    (default: 7)
#   CONCURRENCIES    comma-separated list (default: "1,8,32"); spaces also work
#   REPEATS          how many times to run the whole sweep (default: 1)
#   TRTLLM_LAUNCH_SCRIPT  command prefix for run.py, e.g. trtllm-llmapi-launch
# Arguments are passed to every run.py call (dataset, output length, TP size, ...).

: "${HF_MODEL_CKPT:?HF_MODEL_CKPT is required}"
: "${DRAFT_MODEL:?DRAFT_MODEL is required}"
: "${SAVE_ROOT:?SAVE_ROOT is required}"
# Checked up front: otherwise arm A runs for minutes before arm B fails to load the draft.
# Only local paths: a Hugging Face repo ID (org/name) is resolved by the engine.
case "$DRAFT_MODEL" in
    /*|./*|../*)
        if [ ! -d "$DRAFT_MODEL" ]; then
            report_result "FAIL: ab_compare: DRAFT_MODEL=${DRAFT_MODEL} is not a directory"
            exit 1
        fi
        ;;
esac
ENGINE=${ENGINE:-VLLM}
SPEC_ALGO=${SPEC_ALGO:-DFLASH}
NUM_SPEC_TOKENS=${NUM_SPEC_TOKENS:-7}
CONCURRENCIES=${CONCURRENCIES:-1,8,32}
REPEATS=${REPEATS:-1}
BENCH=modules/Model-Optimizer/examples/specdec_bench

case "$SPEC_ALGO" in
    DFLASH|DSPARK) SPEC_ARGS=(--block_size "$NUM_SPEC_TOKENS") ;;
    *) SPEC_ARGS=(--draft_length "$NUM_SPEC_TOKENS") ;;
esac

if ! pip show boto3 >/dev/null 2>&1 || ! pip show datasets >/dev/null 2>&1 || \
   ! pip show seaborn >/dev/null 2>&1; then
    if ! pip install -r ${BENCH}/requirements.txt; then
        report_result "FAIL: ab_compare: pip install requirements.txt failed"
        exit 1
    fi
fi

# run.py appends to existing result files, so a reused directory would mix runs.
if [ -n "$(ls -A "$SAVE_ROOT" 2>/dev/null)" ]; then
    report_result "FAIL: ab_compare: SAVE_ROOT=${SAVE_ROOT} is not empty"
    exit 1
fi
mkdir -p "$SAVE_ROOT"
python3 -c 'import json, socket, subprocess, sys
gpus = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.splitlines()
json.dump({"host": socket.gethostname(), "gpus": gpus}, open(sys.argv[1], "w"), indent=2)' \
    "${SAVE_ROOT}/env.json"

# Stops the job at the first failed arm: a sweep with a hole cannot be compared.
run_arm() {  # arm (A or B), repeat, concurrency, then the run.py args shared by every arm
    local arm=$1 repeat=$2 conc=$3
    shift 3
    local save_dir="${SAVE_ROOT}/r${repeat}/${arm}_c${conc}"
    local arm_args=(--speculative_algorithm NONE)
    if [ "$arm" = B ]; then
        arm_args=(--speculative_algorithm "$SPEC_ALGO" --draft_model_dir "$DRAFT_MODEL" "${SPEC_ARGS[@]}")
    fi
    echo "=== ${save_dir} ==="
    if ! ${TRTLLM_LAUNCH_SCRIPT:-} python3 ${BENCH}/run.py \
        --model_dir "${HF_MODEL_CKPT}" \
        --tokenizer "${HF_MODEL_CKPT}" \
        --engine "${ENGINE}" \
        --save_dir "${save_dir}" \
        "${arm_args[@]}" --concurrency "$conc" "$@"; then
        report_result "FAIL: ab_compare: arm ${arm} failed (repeat ${repeat}, concurrency ${conc})"
        exit 1
    fi
}

for repeat in $(seq 1 "$REPEATS"); do
    order="A B"
    [ $((repeat % 2)) -eq 0 ] && order="B A"
    for conc in ${CONCURRENCIES//,/ }; do
        for arm in $order; do
            run_arm "$arm" "$repeat" "$conc" "$@"
        done
    done
done

# compare_ab.py exits 1 when a row lists problems (missing arm, appended results, turn
# count mismatch); the summary is written either way.
if ! python3 ${BENCH}/compare_ab.py "$SAVE_ROOT" --json "${SAVE_ROOT}/ab_summary.json"; then
    report_result "FAIL: ab_compare: problems in ${SAVE_ROOT}/ab_summary.json"
    exit 1
fi
report_result "PASS: ab_compare: ${SAVE_ROOT}/ab_summary.json"
