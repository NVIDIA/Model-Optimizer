#!/usr/bin/env bash
set -Eeuo pipefail

: "${CAMPAIGN_DIR:?set CAMPAIGN_DIR}"
: "${CONFIG_PATH:?set CONFIG_PATH}"

PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN="${TORCHRUN:-torchrun}"
NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NODE_RANK="${NODE_RANK:-0}"
RDZV_ID="${RDZV_ID:-distributed-eval-${SLURM_JOB_ID:-local}}"
RDZV_ENDPOINT="${RDZV_ENDPOINT:-127.0.0.1:29500}"
WORKER_HOST="${WORKER_HOST:-$(hostname -f)}"
WORKER_PORT="${WORKER_PORT:-5010}"
WORKER_ID="${WORKER_ID:-${SLURM_JOB_ID:-local}-group-${WORKER_GROUP_INDEX:-0}}"

override_args=()
if [[ -n "${DISTRIBUTED_EVAL_OVERRIDES:-}" ]]; then
  while IFS= read -r override; do
    [[ -n "${override}" ]] && override_args+=(--override "${override}")
  done <<< "${DISTRIBUTED_EVAL_OVERRIDES}"
fi

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONUNBUFFERED=1

exec "${TORCHRUN}" \
  --nnodes "${NNODES}" \
  --nproc-per-node "${NPROC_PER_NODE}" \
  --node-rank "${NODE_RANK}" \
  --rdzv-backend c10d \
  --rdzv-id "${RDZV_ID}" \
  --rdzv-endpoint "${RDZV_ENDPOINT}" \
  -m modelopt.torch.puzzletron.distributed_eval.cli worker \
  --campaign-dir "${CAMPAIGN_DIR}" \
  --config "${CONFIG_PATH}" \
  --host "${WORKER_HOST}" \
  --port "${WORKER_PORT}" \
  --worker-id "${WORKER_ID}" \
  --heartbeat-seconds "${HEARTBEAT_SECONDS:-10}" \
  "${override_args[@]}"
