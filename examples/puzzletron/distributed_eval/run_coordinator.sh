#!/usr/bin/env bash
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
  "${PYTHON_BIN}" "${SCRIPT_DIR}/../finalize_replacement_scoring.py" \
    --config "${FINALIZE_CONFIG_PATH:-${CONFIG_PATH}}" \
    --puzzle-dir "${FINALIZE_PUZZLE_DIR:-${PUZZLE_DIR}}"
fi
