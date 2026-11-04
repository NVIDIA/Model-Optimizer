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

# Measure speculative-decoding acceptance through a served endpoint with AIPerf.
#
# specdec_bench's own path (run.py) drives an in-process engine and derives
# acceptance from its own token bookkeeping. That is accurate, but it makes a
# number comparable only to other specdec_bench numbers. This script measures
# the same quantity through AIPerf, the standard LLM benchmarking client, so
# acceptance is comparable to anything else measured with it.
#
# The two agree. See "Cross-validating against the in-process harness" in the
# README for a same-node comparison and compare_harnesses.py to reproduce it.
#
# Requirements:
#   - vLLM with speculative decoding support for your method, exposing
#     --per-request-spec-decode-metrics
#   - aiperf built from main; the spec_al_* public datasets are not in a
#     released tag yet
#
# Environment:
#   MODEL             target checkpoint (path or HF id)
#   DRAFT_MODEL       draft checkpoint
#   SPEC_METHOD       vLLM speculative method, e.g. eagle3, dflash, mtp
#   NUM_SPEC_TOKENS   draft tokens proposed per step (K)
#   TP_SIZE           tensor parallel size
#   MAX_MODEL_LEN     server sequence-length cap
#   OUTPUT_LENGTH     max generated tokens per request
#   REQUEST_COUNT     requests to profile (MT-Bench: 80 conversations x 2 turns)
#   TEMPERATURE       sampling temperature; 0 for greedy
#   DATASET           AIPerf public dataset, default spec_al_mtbench
#   ARTIFACT_DIR      where the server log and AIPerf exports land
#   EXTRA_SERVE_ARGS  appended to `vllm serve` verbatim

set -o pipefail

MODEL=${MODEL:?Set MODEL to the target checkpoint}
DRAFT_MODEL=${DRAFT_MODEL:?Set DRAFT_MODEL to the draft checkpoint}
SPEC_METHOD=${SPEC_METHOD:-eagle3}
NUM_SPEC_TOKENS=${NUM_SPEC_TOKENS:-3}
TP_SIZE=${TP_SIZE:-1}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
OUTPUT_LENGTH=${OUTPUT_LENGTH:-4096}
REQUEST_COUNT=${REQUEST_COUNT:-160}
WARMUP_COUNT=${WARMUP_COUNT:-3}   # must be > 0; AIPerf rejects 0
TEMPERATURE=${TEMPERATURE:-0}
DATASET=${DATASET:-spec_al_mtbench}
SERVED_NAME=${SERVED_NAME:-specdec-target}
PORT=${PORT:-8000}
ARTIFACT_DIR=${ARTIFACT_DIR:-./aiperf_acceptance}
EXPORT_LEVEL=${EXPORT_LEVEL:-records}
# Off by default. --trust-remote-code lets a checkpoint's own Python run in this
# process, which is fine for a local checkpoint you produced and not fine as an
# unconditional default when MODEL may name a remote repo. Opt in per run.
TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-0}

TRUST_FLAG=""
if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
    TRUST_FLAG="--trust-remote-code"
    echo "note: --trust-remote-code enabled; ${MODEL} may execute its own code"
fi

mkdir -p "${ARTIFACT_DIR}"

echo "=== serving ${MODEL} + ${SPEC_METHOD} K=${NUM_SPEC_TOKENS} ==="
vllm serve "${MODEL}" \
    --served-model-name "${SERVED_NAME}" \
    --speculative-config "{\"method\":\"${SPEC_METHOD}\",\"model\":\"${DRAFT_MODEL}\",\"num_speculative_tokens\":${NUM_SPEC_TOKENS}}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --per-request-spec-decode-metrics detailed \
    ${TRUST_FLAG} \
    --port "${PORT}" ${EXTRA_SERVE_ARGS} \
    > "${ARTIFACT_DIR}/vllm_server.log" 2>&1 &
SERVER_PID=$!

# Reap the server before returning, and *wait* for it. Without the wait the
# script exits while vLLM still holds the GPUs, so a caller that immediately
# starts a second measurement -- exactly what the README's run-both-on-one-node
# recipe does -- races it for memory and fails in a way that looks like an OOM
# in the second run rather than a teardown problem in the first.
reap_server() {
    [[ -n "${SERVER_PID:-}" ]] || return 0
    kill -0 "${SERVER_PID}" 2>/dev/null || return 0
    kill "${SERVER_PID}" 2>/dev/null || true
    for _ in $(seq 1 "${SERVER_SHUTDOWN_WAIT:-60}"); do
        kill -0 "${SERVER_PID}" 2>/dev/null || break
        sleep 1
    done
    # vLLM's engine does not always unwind on SIGTERM; do not leave GPUs pinned.
    if kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "server ${SERVER_PID} ignored SIGTERM; sending SIGKILL"
        kill -9 "${SERVER_PID}" 2>/dev/null || true
    fi
    wait "${SERVER_PID}" 2>/dev/null || true
}
trap reap_server EXIT

echo "waiting for the server to become healthy..."
for _ in $(seq 1 180); do
    curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1 && break
    kill -0 ${SERVER_PID} 2>/dev/null || { echo "server exited early; see ${ARTIFACT_DIR}/vllm_server.log"; exit 1; }
    sleep 10
done
curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1 || {
    echo "server never became healthy; see ${ARTIFACT_DIR}/vllm_server.log"; exit 1; }

# Probe before committing the full run. `detailed` is only honoured on a build
# that carries the flag, and only for n == 1. Without this check an unsupported
# build produces a complete run of records with no acceptance field, and the
# problem surfaces only at analysis time, after the GPUs are gone.
echo "=== probing metrics.speculative_decoding ==="
curl -sf "http://localhost:${PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${SERVED_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"Count to five.\"}],\"max_tokens\":64,\"temperature\":${TEMPERATURE}}" \
    > "${ARTIFACT_DIR}/probe.json"
python3 - <<PY || exit 1
import json, sys
d = json.load(open("${ARTIFACT_DIR}/probe.json"))
sd = (d.get("metrics") or {}).get("speculative_decoding")
if not sd:
    print("FAIL: no metrics.speculative_decoding in the response.")
    print("This build does not expose per-request acceptance.")
    sys.exit(1)
# Presence is not enough. A block that is present but zeroed means speculation
# did not actually run -- a mis-specified draft, or a method the build accepted
# and then ignored. The sweep would still complete and report AL ~1.0, which
# reads as a real measurement of a useless draft rather than as a broken setup.
steps = sd.get("num_spec_steps") or 0
drafted = sd.get("num_draft_tokens") or 0
if steps <= 0 or drafted <= 0:
    print(f"FAIL: speculative_decoding present but inert (num_spec_steps={steps}, "
          f"num_draft_tokens={drafted}).")
    print("Speculation is not running; the sweep would report AL ~1.0 as if measured.")
    sys.exit(1)
print("probe OK:", json.dumps({k: v for k, v in sd.items() if not isinstance(v, list)}))
PY

echo "=== AIPerf: ${DATASET}, ${REQUEST_COUNT} requests, concurrency 1 ==="
# ignore_eos:false is explicit. Several client paths set ignore_eos:true
# whenever an output length is given, which turns natural-EOS generation into
# fixed-length generation and changes acceptance.
aiperf profile \
    --model "${SERVED_NAME}" \
    --tokenizer "${MODEL}" \
    --url "http://localhost:${PORT}" \
    --endpoint-type chat \
    --streaming \
    --public-dataset "${DATASET}" \
    --export-level "${EXPORT_LEVEL}" \
    --concurrency 1 \
    --request-count "${REQUEST_COUNT}" \
    --warmup-request-count "${WARMUP_COUNT}" \
    --osl "${OUTPUT_LENGTH}" \
    --extra-inputs ignore_eos:false \
    --extra-inputs "temperature:${TEMPERATURE}" \
    --artifact-dir "${ARTIFACT_DIR}"

echo "=== artifacts ==="
ls -la "${ARTIFACT_DIR}"
[[ -f "${ARTIFACT_DIR}/profile_export_aiperf.json" ]] || {
    echo "FAIL: no AIPerf export produced"; exit 1; }
echo "PASS: acceptance measured for ${SPEC_METHOD} K=${NUM_SPEC_TOKENS}"
