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

set -Eeuo pipefail

: "${CAMPAIGN_DIR:?set CAMPAIGN_DIR}"
: "${CONFIG_PATH:?set CONFIG_PATH}"
: "${WORLD_SIZE:?set WORLD_SIZE to one worker-group world size}"
: "${WORKER_COUNT:?set WORKER_COUNT to the number of worker groups}"
: "${PUZZLETRON_GROUP_INDEX:=${PUZZLETRON_TASK_INDEX:-${SLURM_PROCID:-}}}"
: "${PUZZLETRON_GROUP_INDEX:?run this script as one orchestrator worker-group task}"
: "${PUZZLETRON_GROUP_RANK:=0}"

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
GROUP_INDEX="${PUZZLETRON_GROUP_INDEX}"
GROUP_RANK="${PUZZLETRON_GROUP_RANK}"
JOB_ID="${SLURM_JOB_ID:-local}"
WORKER_PREFIX="${JOB_ID}-depth-"
MANIFEST_PATH="${CAMPAIGN_DIR}/manifest.json"
worker_pid=""

override_args=()
if [[ -n "${DISTRIBUTED_EVAL_OVERRIDES:-}" ]]; then
  while IFS= read -r override; do
    [[ -n "${override}" ]] && override_args+=(--override "${override}")
  done <<< "${DISTRIBUTED_EVAL_OVERRIDES}"
fi

drain_workers() {
  [[ -f "${MANIFEST_PATH}" ]] || return 0
  CUDA_VISIBLE_DEVICES="" "${PYTHON_BIN}" \
    -m modelopt.torch.puzzletron.distributed_eval.cli drain \
    --campaign-dir "${CAMPAIGN_DIR}" \
    --stale-seconds "${STALE_SECONDS:-45}" || true
}

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  set +e
  if [[ "${GROUP_INDEX}" == "0" && "${GROUP_RANK}" == "0" ]]; then
    drain_workers
  fi
  if [[ -n "${worker_pid}" ]] && kill -0 "${worker_pid}" 2>/dev/null; then
    kill -TERM "${worker_pid}" 2>/dev/null
    wait "${worker_pid}" 2>/dev/null
  fi
  exit "${rc}"
}
trap cleanup EXIT INT TERM

# Rank 0 creates the shared campaign before any worker attempts to open it.
if [[ "${GROUP_INDEX}" == "0" && "${GROUP_RANK}" == "0" && ! -f "${MANIFEST_PATH}" ]]; then
  CUDA_VISIBLE_DEVICES="" "${PYTHON_BIN}" \
    -m modelopt.torch.puzzletron.distributed_eval.cli init \
    --campaign-dir "${CAMPAIGN_DIR}" \
    --config "${CONFIG_PATH}" \
    --world-size "${WORLD_SIZE}" \
    --stage depth \
    --evaluator-revision "${EVALUATOR_REVISION:-puzzletron-depth-v1}" \
    "${override_args[@]}"
fi

manifest_deadline=$((SECONDS + ${MANIFEST_TIMEOUT_SECONDS:-300}))
while [[ ! -f "${MANIFEST_PATH}" ]]; do
  if ((SECONDS >= manifest_deadline)); then
    echo "Timed out waiting for depth campaign manifest: ${MANIFEST_PATH}" >&2
    exit 1
  fi
  sleep 1
done

# Every scheduler task owns one GPU slice and starts one worker group. Multiple
# independent worker groups may share a node.
export NPROC_PER_NODE="${NPROC_PER_NODE:-${WORLD_SIZE}}"
export WORKER_GROUP_INDEX="${GROUP_INDEX}"
export WORKER_ID="${WORKER_PREFIX}${GROUP_INDEX}"
export WORKER_HOST="${WORKER_HOST:-$(hostname -f)}"
export WORKER_PORT="${WORKER_PORT:-$((5010 + GROUP_INDEX))}"

bash "${SCRIPT_DIR}/run_worker.sh" &
worker_pid=$!

coordinator_rc=0
if [[ "${GROUP_INDEX}" == "0" && "${GROUP_RANK}" == "0" ]]; then
  # Do not start depth iteration zero until every resident model is ready.
  CUDA_VISIBLE_DEVICES="" "${PYTHON_BIN}" - \
      "${CAMPAIGN_DIR}" \
      "${WORKER_COUNT}" \
      "${WORKER_PREFIX}" \
      "${STALE_SECONDS:-45}" \
      "${WORKER_START_TIMEOUT_SECONDS:-1800}" <<'PY' &
import sys
import time

from modelopt.torch.puzzletron.distributed_eval.campaign import Campaign

campaign_dir, expected_text, worker_prefix, stale_text, timeout_text = sys.argv[1:]
campaign = Campaign.open(campaign_dir)
expected = int(expected_text)
stale_seconds = float(stale_text)
deadline = time.monotonic() + float(timeout_text)
last_count = -1
while True:
    workers = [
        worker
        for worker in campaign.registry.list_workers(
            campaign.manifest,
            stale_seconds=stale_seconds,
        )
        if worker.worker_id.startswith(worker_prefix)
    ]
    if len(workers) != last_count:
        print(
            f"[depth-pool] ready workers: {len(workers)}/{expected}",
            flush=True,
        )
        last_count = len(workers)
    if len(workers) >= expected:
        break
    if time.monotonic() >= deadline:
        raise TimeoutError(
            f"Timed out waiting for depth workers: {len(workers)}/{expected}"
        )
    time.sleep(2)
PY
  readiness_pid=$!
  while kill -0 "${readiness_pid}" 2>/dev/null; do
    if ! kill -0 "${worker_pid}" 2>/dev/null; then
      set +e
      wait "${worker_pid}"
      worker_rc=$?
      worker_pid=""
      kill -TERM "${readiness_pid}" 2>/dev/null
      wait "${readiness_pid}" 2>/dev/null
      set -e
      echo "Depth worker 0 exited before the pool became ready." >&2
      exit "$((worker_rc == 0 ? 1 : worker_rc))"
    fi
    sleep 2
  done
  wait "${readiness_pid}"

  CUDA_VISIBLE_DEVICES="" bash "${SCRIPT_DIR}/run_depth_coordinator.sh" &
  coordinator_pid=$!
  while kill -0 "${coordinator_pid}" 2>/dev/null; do
    if ! kill -0 "${worker_pid}" 2>/dev/null; then
      set +e
      wait "${worker_pid}"
      worker_rc=$?
      worker_pid=""
      kill -TERM "${coordinator_pid}" 2>/dev/null
      wait "${coordinator_pid}" 2>/dev/null
      set -e
      echo "Depth worker 0 exited while the coordinator was active." >&2
      exit "$((worker_rc == 0 ? 1 : worker_rc))"
    fi
    sleep 2
  done
  set +e
  wait "${coordinator_pid}"
  coordinator_rc=$?
  set -e
  drain_workers
fi

set +e
wait "${worker_pid}"
worker_rc=$?
set -e
worker_pid=""
trap - EXIT INT TERM

if ((coordinator_rc != 0)); then
  exit "${coordinator_rc}"
fi
exit "${worker_rc}"
