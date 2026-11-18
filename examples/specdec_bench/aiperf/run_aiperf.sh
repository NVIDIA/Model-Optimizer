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

# Measure speculative-decoding acceptance and throughput through a served
# endpoint with AIPerf, in a layout the Speed visualizer can publish.
#
# specdec_bench's own path (run.py) drives an in-process engine and derives
# acceptance from its own token bookkeeping. That is accurate, but it makes a
# number comparable only to other specdec_bench numbers, and it cannot measure
# the serving stack under load. This script measures both through AIPerf, the
# standard LLM benchmarking client.
#
# Acceptance agrees with the in-process path; see "Cross-validating against the
# in-process harness" in the README.
#
# Output layout
# -------------
# One directory per dataset (a "sweep"), one subdirectory per concurrency (a
# "cell"), shaped for specdec-visualizer's btk_results.py:
#
#   ${ARTIFACT_ROOT}/<dataset>/            <- publish this directory
#     c<concurrency>/
#       metadata.json                      <- client_type, serving command, hw
#       recipe.json                        <- model/draft revisions, GPU count
#       aiperf0_aiperf_results/
#         profile_export_aiperf.json       <- AIPerf's own export, untouched
#
# Then:
#   python3 btk_results.py ${ARTIFACT_ROOT}/<dataset> --name <sweep> --publish
#
# Datasets are the OUTER loop on purpose. btk_results.py requires one identity
# per sweep and counts the dataset as part of the workload identity, so mixing
# datasets in one directory is rejected as "Mixed model, hardware, serving
# configuration, or workload in one sweep". Concurrency is the dimension that
# is allowed to vary within a sweep -- it is the Speed view's `batch` axis.
#
# The server starts once and stays up across every dataset and concurrency.
# Restarting per point would both waste the allocation and risk a different
# serving configuration between cells, which the publisher rejects.
#
# Requirements:
#   - vLLM with speculative decoding for your method, exposing
#     --per-request-spec-decode-metrics
#   - aiperf built from main; the spec_al_* and speed_bench_* public datasets
#     are not in a released tag yet
#
# Environment:
#   MODEL / DRAFT_MODEL        target and draft checkpoints
#   MODEL_REVISION / DRAFT_REVISION   recorded for provenance; strongly advised
#   SPEC_METHOD                vLLM speculative method (eagle3, dflash, mtp),
#                              or "none" for a matched baseline arm
#   NUM_SPEC_TOKENS            draft tokens proposed per step (K)
#   TP_SIZE                    tensor parallel size
#   GPU_COUNT                  whole-deployment GPU count (defaults to TP_SIZE)
#   DATASETS                   space-separated AIPerf public datasets
#   CONCURRENCIES              space-separated concurrency values
#   REQUEST_COUNT              profiling requests per cell (raised to at least
#                              the concurrency; the publisher rejects a budget
#                              smaller than the client count)
#   OUTPUT_LENGTH / TEMPERATURE / MAX_MODEL_LEN
#   TRUST_REMOTE_CODE=1        opt in to running checkpoint-provided code
#   ARTIFACT_ROOT              parent for the sweep directories
#   STUDY_ID                   optional grouping id carried into recipe.json

set -o pipefail

MODEL=${MODEL:?Set MODEL to the target checkpoint}
SPEC_METHOD=${SPEC_METHOD:-eagle3}
DRAFT_MODEL=${DRAFT_MODEL:-}
if [[ "${SPEC_METHOD}" != "none" && -z "${DRAFT_MODEL}" ]]; then
    echo "FAIL: set DRAFT_MODEL, or SPEC_METHOD=none for a baseline arm"; exit 1
fi
MODEL_REVISION=${MODEL_REVISION:-}
DRAFT_REVISION=${DRAFT_REVISION:-}
NUM_SPEC_TOKENS=${NUM_SPEC_TOKENS:-3}
TP_SIZE=${TP_SIZE:-1}
GPU_COUNT=${GPU_COUNT:-${TP_SIZE}}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
OUTPUT_LENGTH=${OUTPUT_LENGTH:-4096}
TEMPERATURE=${TEMPERATURE:-0}
# The Speed view's `batch` axis. Override for a cheaper run; a single value is
# the right choice when you only want acceptance, which is a per-step property
# and is measured cleanest without queueing.
CONCURRENCIES=${CONCURRENCIES:-"1 4 16 32 64 128 256 512"}
DATASETS=${DATASETS:-"spec_al_mtbench"}
REQUEST_COUNT=${REQUEST_COUNT:-160}
WARMUP_COUNT=${WARMUP_COUNT:-3}   # must be > 0; AIPerf rejects 0
SERVED_NAME=${SERVED_NAME:-specdec-target}
PORT=${PORT:-8000}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-./aiperf_results}
EXPORT_LEVEL=${EXPORT_LEVEL:-records}
STUDY_ID=${STUDY_ID:-}
CLIENT_INSTANCE=${CLIENT_INSTANCE:-aiperf0}
# Off by default. --trust-remote-code lets a checkpoint's own Python run in this
# process, which is fine for a local checkpoint you produced and not fine as an
# unconditional default when MODEL may name a remote repo. Opt in per run.
TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-0}

TRUST_FLAG=""
if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
    TRUST_FLAG="--trust-remote-code"
    echo "note: --trust-remote-code enabled; ${MODEL} may execute its own code"
fi

# Fail before reserving anything rather than quietly degrading a cell. Raising
# the budget per cell to meet the client count would make the highest
# concurrencies run roughly one request per client, which is not a steady-state
# throughput measurement: the batch axis would stop being comparable across
# cells while still publishing as though it were.
MAX_CONC=0
for conc in ${CONCURRENCIES}; do
    [[ "${conc}" =~ ^[0-9]+$ && "${conc}" -gt 0 ]] || {
        echo "FAIL: CONCURRENCIES must be positive integers; got '${conc}'"; exit 1; }
    (( conc > MAX_CONC )) && MAX_CONC=${conc}
done
if (( REQUEST_COUNT < MAX_CONC )); then
    echo "FAIL: REQUEST_COUNT=${REQUEST_COUNT} is below the largest concurrency (${MAX_CONC})."
    echo "      Every client would serve fewer than one request and the sweep's"
    echo "      batch axis would not be comparable. Raise REQUEST_COUNT, or drop"
    echo "      the high concurrencies from CONCURRENCIES."
    exit 1
fi

mkdir -p "${ARTIFACT_ROOT}"

# Build the serving command as an argv array, not a string. The publisher
# recovers the speculative config with shlex.split(build_command) followed by
# json.loads, and shlex *removes* quotes: a config recorded as bare
# --speculative-config {"method":"dflash"} round-trips to {method:dflash} and
# fails to parse. Recording argv and re-joining it with shlex.join below keeps
# the JSON quoting intact, and keeps the executed and recorded commands from
# drifting apart.
SERVE_ARGS=(vllm serve "${MODEL}" --served-model-name "${SERVED_NAME}")
# A revision is only a fact if the server was told to honour it. vLLM's
# --revision selects a Hugging Face Hub commit; for a local directory there is
# nothing to select and the flag is ignored, so passing it would not make the
# recorded revision true. Record which case applies instead of publishing an
# unenforceable claim: for a local checkpoint the revision is asserted by
# whoever staged it, and the artifact should say so.
REVISION_ENFORCED=false
if [[ -n "${MODEL_REVISION}" ]]; then
    if [[ -d "${MODEL}" ]]; then
        echo "note: ${MODEL} is a local directory; MODEL_REVISION is recorded as asserted by staging, not enforced by the server"
    else
        SERVE_ARGS+=(--revision "${MODEL_REVISION}" --tokenizer-revision "${MODEL_REVISION}")
        REVISION_ENFORCED=true
    fi
fi
if [[ -n "${DRAFT_REVISION}" && -d "${DRAFT_MODEL}" ]]; then
    echo "note: ${DRAFT_MODEL} is a local directory; DRAFT_REVISION is likewise asserted, not enforced"
fi
if [[ "${SPEC_METHOD}" != "none" ]]; then
    SERVE_ARGS+=(--speculative-config "{\"method\":\"${SPEC_METHOD}\",\"model\":\"${DRAFT_MODEL}\",\"num_speculative_tokens\":${NUM_SPEC_TOKENS}}")
fi
SERVE_ARGS+=(--tensor-parallel-size "${TP_SIZE}"
             --max-model-len "${MAX_MODEL_LEN}"
             --per-request-spec-decode-metrics detailed)
[[ -n "${TRUST_FLAG}" ]] && SERVE_ARGS+=("${TRUST_FLAG}")
SERVE_ARGS+=(--port "${PORT}")

ARGV_FILE="${ARTIFACT_ROOT}/.serve_argv"
printf '%s\n' "${SERVE_ARGS[@]}" > "${ARGV_FILE}"

echo "=== serving ${MODEL} (${SPEC_METHOD} K=${NUM_SPEC_TOKENS}) ==="
"${SERVE_ARGS[@]}" > "${ARTIFACT_ROOT}/vllm_server.log" 2>&1 &
SERVER_PID=$!

# Reap the server before returning, and *wait* for it. Without the wait the
# script exits while vLLM still holds the GPUs, so a caller that immediately
# starts another measurement races it for memory and fails in a way that looks
# like an OOM in the second run rather than a teardown problem in the first.
reap_server() {
    [[ -n "${SERVER_PID:-}" ]] || return 0
    kill -0 "${SERVER_PID}" 2>/dev/null || return 0
    kill "${SERVER_PID}" 2>/dev/null || true
    for _ in $(seq 1 "${SERVER_SHUTDOWN_WAIT:-60}"); do
        kill -0 "${SERVER_PID}" 2>/dev/null || break
        sleep 1
    done
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
    kill -0 ${SERVER_PID} 2>/dev/null || { echo "server exited early; see ${ARTIFACT_ROOT}/vllm_server.log"; exit 1; }
    sleep 10
done
curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1 || {
    echo "server never became healthy; see ${ARTIFACT_ROOT}/vllm_server.log"; exit 1; }

# Probe before committing the sweep. `detailed` is only honoured on a build that
# carries the flag, and only for n == 1; and a block that is present but zeroed
# means speculation did not actually run. Either way the sweep would otherwise
# complete and report AL ~1.0 as a real measurement of a useless draft.
if [[ "${SPEC_METHOD}" != "none" ]]; then
    echo "=== probing metrics.speculative_decoding (streaming, as the sweep runs) ==="
    # Probe the transport the sweep actually uses. AIPerf runs --streaming, and
    # acceptance metrics ride on the final SSE chunk; a unary probe can succeed
    # on a build that never attaches them to a stream, which is precisely the
    # failure this gate exists to catch.
    curl -sf -N "http://localhost:${PORT}/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"${SERVED_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"Count to five.\"}],\"max_tokens\":64,\"temperature\":${TEMPERATURE},\"stream\":true,\"stream_options\":{\"include_usage\":true}}" \
        > "${ARTIFACT_ROOT}/probe.sse"
    python3 - <<PY || exit 1
import json, sys
# Walk the SSE stream and keep the last chunk that carries the metrics block;
# vLLM attaches it to a trailing chunk, not to every delta.
sd = None
for line in open("${ARTIFACT_ROOT}/probe.sse"):
    line = line.strip()
    if not line.startswith("data:"):
        continue
    payload = line[5:].strip()
    if payload == "[DONE]":
        break
    try:
        chunk = json.loads(payload)
    except json.JSONDecodeError:
        continue
    found = (chunk.get("metrics") or {}).get("speculative_decoding")
    if found:
        sd = found
json.dump(sd or {}, open("${ARTIFACT_ROOT}/probe.json", "w"), indent=2)
if not sd:
    print("FAIL: no metrics.speculative_decoding on any streamed chunk.")
    print("This build does not expose per-request acceptance over streaming,")
    print("which is the transport the sweep uses.")
    sys.exit(1)
steps = sd.get("num_spec_steps") or 0
drafted = sd.get("num_draft_tokens") or 0
if steps <= 0 or drafted <= 0:
    print(f"FAIL: speculative_decoding present but inert (num_spec_steps={steps}, "
          f"num_draft_tokens={drafted}).")
    print("Speculation is not running; the sweep would report AL ~1.0 as if measured.")
    sys.exit(1)
print("probe OK:", json.dumps({k: v for k, v in sd.items() if not isinstance(v, list)}))
PY
fi

# Hardware and framework identity, recorded once so every cell of the sweep
# agrees. btk_results.py rejects a sweep whose cells disagree on these.
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
GPU_NAME=${GPU_NAME:-unknown}
# Read the version off the serving process itself rather than importing vllm
# here. `python3` on PATH is not necessarily the interpreter vLLM runs under --
# a client installed into its own venv earlier in PATH shadows it, and the
# import then fails and silently records "unknown". The engine banner is
# authoritative because it comes from the process actually serving the sweep.
VLLM_VERSION=$(grep -m1 -oE "V1 LLM engine \(v[^)]+\)" "${ARTIFACT_ROOT}/vllm_server.log" 2>/dev/null \
    | sed -E 's/.*\(v?//; s/\)//')
if [[ -z "${VLLM_VERSION}" ]]; then
    VLLM_VERSION=$(vllm --version 2>/dev/null | tr -d '\r' | tail -1)
fi
VLLM_VERSION=${VLLM_VERSION:-unknown}
if [[ "${VLLM_VERSION}" == "unknown" ]]; then
    # Not fatal, but say so loudly: the runtime version is the first thing
    # asked when two measurements of the same checkpoint disagree.
    echo "WARNING: could not determine the vLLM version; published provenance will be incomplete"
fi
echo "hardware: ${GPU_NAME} x${GPU_COUNT} | vllm ${VLLM_VERSION}"

failed=""
for dataset in ${DATASETS}; do
    sweep_dir="${ARTIFACT_ROOT}/${dataset}"
    mkdir -p "${sweep_dir}"
    for conc in ${CONCURRENCIES}; do
        requests=${REQUEST_COUNT}
        cell="${sweep_dir}/c${conc}"
        results="${cell}/${CLIENT_INSTANCE}_aiperf_results"
        # Start each cell empty. A re-run, or an arm pointed at the same root
        # with a different configuration, would otherwise leave a stale export
        # sitting beside freshly written metadata -- a published cell claiming
        # to describe a measurement it did not come from.
        rm -rf "${cell}"
        mkdir -p "${results}"
        # Snapshot the serving argv into the cell rather than referencing one
        # shared file, for the same reason: the cell must be self-describing.
        cp "${ARGV_FILE}" "${cell}/.serve_argv"

        echo
        echo "=== ${dataset} @ concurrency ${conc} (${requests} requests) ==="
        # ignore_eos:false is explicit. Several client paths set it true
        # whenever an output length is given, which turns natural-EOS
        # generation into fixed-length generation and changes acceptance.
        # Built as an argv array so the exact client invocation can be
        # recorded alongside the export. Without it a published cell -- a
        # SPEC_METHOD=none control especially -- cannot be audited from the
        # artifact alone: the concurrency and budget that produced it would
        # only exist in a job log that is not published with the result.
        CLIENT_ARGS=(aiperf profile
            --model "${SERVED_NAME}"
            --tokenizer "${MODEL}"
            --url "http://localhost:${PORT}"
            --endpoint-type chat
            --streaming
            --public-dataset "${dataset}"
            --export-level "${EXPORT_LEVEL}"
            --concurrency "${conc}"
            --request-count "${requests}"
            --warmup-request-count "${WARMUP_COUNT}"
            --osl "${OUTPUT_LENGTH}"
            --extra-inputs ignore_eos:false
            --extra-inputs "temperature:${TEMPERATURE}"
            --artifact-dir "${results}")
        printf '%s\n' "${CLIENT_ARGS[@]}" > "${cell}/.client_argv"

        if "${CLIENT_ARGS[@]}"; then
            :
        else
            echo "FAIL: ${dataset} @ concurrency ${conc}"
            failed="${failed} ${dataset}@${conc}"
            continue
        fi

        # Describe the cell for btk_results.py. Written after the measurement so
        # a cell only claims to exist once it has an export to back it.
        MD_CELL="${cell}" MD_INSTANCE="${CLIENT_INSTANCE}" MD_ARGV_FILE="${ARGV_FILE}" \
        MD_GPU="${GPU_NAME}" MD_SW_VERSION="${VLLM_VERSION}" MD_MODEL="${MODEL}" \
        MD_MODEL_REVISION="${MODEL_REVISION}" MD_DRAFT="${DRAFT_MODEL}" \
        MD_DRAFT_REVISION="${DRAFT_REVISION}" MD_ALGO="${SPEC_METHOD}" \
        MD_K="${NUM_SPEC_TOKENS}" MD_GPU_COUNT="${GPU_COUNT}" MD_STUDY="${STUDY_ID}" \
        MD_REVISION_ENFORCED="${REVISION_ENFORCED}" \
        python3 - <<'PY'
import json, os, shlex
from pathlib import Path

cell = Path(os.environ["MD_CELL"])
algo = os.environ["MD_ALGO"]
disabled = algo == "none"

# shlex.join is the exact inverse of the publisher's shlex.split, so the
# speculative config it recovers is byte-identical to the one the server was
# started with -- quotes and all.
argv = Path(os.environ["MD_ARGV_FILE"]).read_text().splitlines()

metadata = {
    "client_type": "aiperf",
    "client_instance_name": os.environ["MD_INSTANCE"],
    "build_command": shlex.join(argv),
    "gpu": os.environ["MD_GPU"],
    "sw": "vllm",
    "sw_version": os.environ["MD_SW_VERSION"],
}
recipe = {
    "model": os.environ["MD_MODEL"],
    "revision": os.environ["MD_MODEL_REVISION"] or None,
    "draft_model": (os.environ["MD_DRAFT"] or None) if not disabled else None,
    "draft_revision": (os.environ["MD_DRAFT_REVISION"] or None) if not disabled else None,
    # "disabled" is how the publisher recognises a matched baseline arm; it
    # maps the algorithm to NONE rather than demanding a draft model.
    "speculation": "disabled" if disabled else "enabled",
    "speculative_algorithm": None if disabled else algo,
    "draft_length": None if disabled else int(os.environ["MD_K"]),
    "gpu_count_used": int(os.environ["MD_GPU_COUNT"]),
    "disaggregated": False,
    # Whether the server was actually told to honour the recorded revisions, or
    # whether they are an assertion about what was staged. A consumer comparing
    # two runs needs to know which, because an unenforced revision cannot rule
    # out that the weights differed.
    "revision_enforced": os.environ["MD_REVISION_ENFORCED"] == "true",
}
if os.environ["MD_STUDY"]:
    recipe["study_id"] = os.environ["MD_STUDY"]

(cell / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
(cell / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
print(f"wrote {cell}/metadata.json and recipe.json")
PY
    done
done

echo
echo "=== sweeps ==="
for dataset in ${DATASETS}; do
    sweep_dir="${ARTIFACT_ROOT}/${dataset}"
    echo "${sweep_dir}"
    ls -d "${sweep_dir}"/c* 2>/dev/null | sed 's/^/    /'
    echo "    publish:  python3 btk_results.py ${sweep_dir} --name <sweep-name> --publish"
done

if [[ -n "${failed}" ]]; then
    echo "FAIL: cells failed:${failed}"
    exit 1
fi
echo "PASS: ${SPEC_METHOD} K=${NUM_SPEC_TOKENS} over [${DATASETS}] x [${CONCURRENCIES}]"
