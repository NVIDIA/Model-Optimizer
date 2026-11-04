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

mkdir -p "${ARTIFACT_DIR}"

echo "=== serving ${MODEL} + ${SPEC_METHOD} K=${NUM_SPEC_TOKENS} ==="
vllm serve "${MODEL}" \
    --served-model-name "${SERVED_NAME}" \
    --speculative-config "{\"method\":\"${SPEC_METHOD}\",\"model\":\"${DRAFT_MODEL}\",\"num_speculative_tokens\":${NUM_SPEC_TOKENS}}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --per-request-spec-decode-metrics detailed \
    --trust-remote-code \
    --port "${PORT}" ${EXTRA_SERVE_ARGS} \
    > "${ARTIFACT_DIR}/vllm_server.log" 2>&1 &
SERVER_PID=$!
trap 'kill ${SERVER_PID} 2>/dev/null || true' EXIT

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
print("probe OK:", json.dumps(sd)[:300])
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
