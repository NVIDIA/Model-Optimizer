#!/usr/bin/env bash
set -Eeuo pipefail

: "${CAMPAIGN_DIR:?set CAMPAIGN_DIR}"
: "${CONFIG_PATH:?set CONFIG_PATH}"
: "${WORLD_SIZE:?set WORLD_SIZE to one torchrun worker-group world size}"

PYTHON_BIN="${PYTHON_BIN:-python}"

override_args=()
if [[ -n "${DISTRIBUTED_EVAL_OVERRIDES:-}" ]]; then
  while IFS= read -r override; do
    [[ -n "${override}" ]] && override_args+=(--override "${override}")
  done <<< "${DISTRIBUTED_EVAL_OVERRIDES}"
fi

if [[ ! -f "${CAMPAIGN_DIR}/manifest.json" ]]; then
  "${PYTHON_BIN}" -m modelopt.torch.puzzletron.distributed_eval.cli init \
    --campaign-dir "${CAMPAIGN_DIR}" \
    --config "${CONFIG_PATH}" \
    --world-size "${WORLD_SIZE}" \
    --stage depth \
    --evaluator-revision "${EVALUATOR_REVISION:-puzzletron-depth-v1}" \
    "${override_args[@]}"
fi

exec "${PYTHON_BIN}" -m modelopt.torch.puzzletron.distributed_eval.cli depth-coordinator \
  --campaign-dir "${CAMPAIGN_DIR}" \
  --config "${CONFIG_PATH}" \
  --output-dir "${OUTPUT_DIR:-}" \
  --stale-seconds "${STALE_SECONDS:-45}" \
  --connect-timeout-seconds "${CONNECT_TIMEOUT_SECONDS:-10}" \
  --task-timeout-seconds "${TASK_TIMEOUT_SECONDS:-7200}" \
  --retry-initial-seconds "${RETRY_INITIAL_SECONDS:-5}" \
  --retry-max-seconds "${RETRY_MAX_SECONDS:-60}" \
  "${override_args[@]}"
