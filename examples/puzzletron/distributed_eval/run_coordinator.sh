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
: "${WORLD_SIZE:?set WORLD_SIZE to one torchrun worker-group world size}"
: "${SOLUTIONS_PATH:?set SOLUTIONS_PATH}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
EVALUATOR_REVISION="${EVALUATOR_REVISION:-puzzletron-distributed-replace-block-v1}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

override_args=()
if [[ -n "${DISTRIBUTED_EVAL_OVERRIDES:-}" ]]; then
  while IFS= read -r override; do
    [[ -n "${override}" ]] && override_args+=(--override "${override}")
  done <<< "${DISTRIBUTED_EVAL_OVERRIDES}"
fi

if [[ "${PREPARE_SUBBLOCK_SOLUTIONS:-0}" == "1" && ! -f "${SOLUTIONS_PATH}" ]]; then
  : "${PUZZLE_DIR:?set PUZZLE_DIR to backfill subblock solutions}"
  : "${REPLACEMENT_LIBRARY_PATH:?set REPLACEMENT_LIBRARY_PATH}"
  : "${TEACHER_DIR:?set TEACHER_DIR}"
  prepare_args=(
    --puzzle-dir "${PUZZLE_DIR}"
    --replacement-library "${REPLACEMENT_LIBRARY_PATH}"
    --teacher-dir "${TEACHER_DIR}"
    --solutions-output "${SOLUTIONS_PATH}"
  )
  if [[ -n "${SUBBLOCK_MANIFEST_PATH:-}" ]]; then
    prepare_args+=(--manifest-output "${SUBBLOCK_MANIFEST_PATH}")
  fi
  if [[ "${TRUST_REMOTE_CODE:-0}" == "1" ]]; then
    prepare_args+=(--trust-remote-code)
  fi
  "${PYTHON_BIN}" "${SCRIPT_DIR}/../prepare_subblock_replacement_scoring.py" \
    "${prepare_args[@]}"
fi

if [[ ! -f "${CAMPAIGN_DIR}/manifest.json" ]]; then
  "${PYTHON_BIN}" -m modelopt.torch.puzzletron.distributed_eval.cli init \
    --campaign-dir "${CAMPAIGN_DIR}" \
    --config "${CONFIG_PATH}" \
    --world-size "${WORLD_SIZE}" \
    --evaluator-revision "${EVALUATOR_REVISION}" \
    "${override_args[@]}"
fi

compatibility_args=()
if [[ -n "${COMPATIBILITY_OUTPUT_DIR:-}" ]]; then
  compatibility_args+=(--compatibility-output-dir "${COMPATIBILITY_OUTPUT_DIR}")
fi

solution_args=()
if [[ -n "${SOLUTION_IDS:-}" ]]; then
  IFS=',' read -ra ids <<< "${SOLUTION_IDS}"
  for id in "${ids[@]}"; do
    solution_args+=(--solution-id "${id}")
  done
fi

"${PYTHON_BIN}" -m modelopt.torch.puzzletron.distributed_eval.cli coordinator \
  --campaign-dir "${CAMPAIGN_DIR}" \
  --solutions "${SOLUTIONS_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --stale-seconds "${STALE_SECONDS:-45}" \
  --connect-timeout-seconds "${CONNECT_TIMEOUT_SECONDS:-10}" \
  --task-timeout-seconds "${TASK_TIMEOUT_SECONDS:-7200}" \
  --retry-initial-seconds "${RETRY_INITIAL_SECONDS:-5}" \
  --retry-max-seconds "${RETRY_MAX_SECONDS:-60}" \
  "${solution_args[@]}" \
  "${compatibility_args[@]}"

if [[ "${FINALIZE_REPLACEMENT_SCORING:-0}" == "1" ]]; then
  if [[ -n "${FINALIZE_COMPLETION_DIR:-}" ]]; then
    : "${FINALIZE_COMPLETION_MARKER:?set FINALIZE_COMPLETION_MARKER}"
    : "${FINALIZE_EXPECTED_COMPLETIONS:?set FINALIZE_EXPECTED_COMPLETIONS}"
    "${PYTHON_BIN}" - \
      "${FINALIZE_COMPLETION_DIR}" \
      "${FINALIZE_COMPLETION_MARKER}" \
      "${FINALIZE_EXPECTED_COMPLETIONS}" \
      "${SCRIPT_DIR}/../finalize_replacement_scoring.py" \
      "${FINALIZE_CONFIG_PATH:-${CONFIG_PATH}}" \
      "${FINALIZE_PUZZLE_DIR:-${PUZZLE_DIR}}" <<'PY'
import fcntl
from pathlib import Path
import subprocess
import sys

from examples.puzzletron.finalize_replacement_scoring import (
    finalization_marker_is_current,
    write_finalization_marker,
)

(
    completion_dir_text,
    marker_name,
    expected_text,
    finalizer,
    config_path,
    puzzle_dir,
) = sys.argv[1:]
if not marker_name or Path(marker_name).name != marker_name:
    raise ValueError(f"invalid replacement-scoring completion marker: {marker_name!r}")
completion_dir = Path(completion_dir_text)
completion_dir.mkdir(parents=True, exist_ok=True)
(completion_dir / f"{marker_name}.done").touch()
with (completion_dir / ".finalize.lock").open("a+") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    finalized = completion_dir / "finalized"
    root = Path(puzzle_dir)
    root_summary = root / "artifacts" / "replacement_scoring" / "summary.json"
    root_manifest = root / "manifests" / "replacement_scoring.json"
    if finalization_marker_is_current(finalized, root_manifest, root_summary):
        raise SystemExit(0)
    finalized.unlink(missing_ok=True)
    completed = tuple(completion_dir.glob("*.done"))
    expected = int(expected_text)
    if len(completed) < expected:
        print(
            f"[replacement-pool] completed widths: {len(completed)}/{expected}; "
            "deferring root finalization",
            flush=True,
        )
        raise SystemExit(0)
    subprocess.run(
        [
            sys.executable,
            finalizer,
            "--config",
            config_path,
            "--puzzle-dir",
            puzzle_dir,
        ],
        check=True,
    )
    write_finalization_marker(finalized, root_manifest)
PY
  else
    "${PYTHON_BIN}" "${SCRIPT_DIR}/../finalize_replacement_scoring.py" \
      --config "${FINALIZE_CONFIG_PATH:-${CONFIG_PATH}}" \
      --puzzle-dir "${FINALIZE_PUZZLE_DIR:-${PUZZLE_DIR}}"
  fi
fi
