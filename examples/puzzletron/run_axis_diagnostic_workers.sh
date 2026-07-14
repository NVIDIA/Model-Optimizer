#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 CONFIG" >&2
  exit 2
fi

config=$(realpath -m "$1")
root=${PUZZLETRON_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
: "${PUZZLETRON_IMAGE:?export PUZZLETRON_IMAGE with the container image path}"
: "${PUZZLETRON_CONTAINER_MOUNTS:?export PUZZLETRON_CONTAINER_MOUNTS for the container}"
image=${PUZZLETRON_IMAGE}
log_dir=${PUZZLETRON_LOG_DIR:-${root}/.runtime/sanity_logs}
mkdir -p "$log_dir"
export PUZZLETRON_ROOT="$root" PUZZLETRON_LOG_DIR="$log_dir"
export PUZZLETRON_SETUP_ENV=${PUZZLETRON_SETUP_ENV:-}

srun \
  --exclusive \
  --kill-on-bad-exit=0 \
  --nodes=2 \
  --ntasks=8 \
  --ntasks-per-node=4 \
  --gpus-per-task=2 \
  --gpu-bind=none \
  --container-image="$image" \
  --container-mounts="${PUZZLETRON_CONTAINER_MOUNTS}" \
  --container-workdir="$root" \
  --mpi=pmix \
  /bin/bash -lc '
set -Eeuo pipefail
if [[ -n "${PUZZLETRON_SETUP_ENV:-}" ]]; then source "${PUZZLETRON_SETUP_ENV}"; fi
source "${PUZZLETRON_ROOT}/.venv/bin/activate"
export PYTHONPATH="${PUZZLETRON_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
examples/puzzletron/run_axis_diagnostic_task.sh "$1" 2>&1 \
  | tee "${PUZZLETRON_LOG_DIR}/axis_diagnostic_${SLURM_JOB_ID}_task${SLURM_PROCID}.log"
' -- "$config"

srun \
  --exclusive \
  --nodes=1 \
  --ntasks=1 \
  --gpus-per-task=1 \
  --container-image="$image" \
  --container-mounts="${PUZZLETRON_CONTAINER_MOUNTS}" \
  --container-workdir="$root" \
  --mpi=pmix \
  /bin/bash -lc '
set -Eeuo pipefail
if [[ -n "${PUZZLETRON_SETUP_ENV:-}" ]]; then source "${PUZZLETRON_SETUP_ENV}"; fi
source "${PUZZLETRON_ROOT}/.venv/bin/activate"
export PYTHONPATH="${PUZZLETRON_ROOT}:${PYTHONPATH:-}"
python examples/puzzletron/run_axis_diagnostic_worker.py --config "$1" --finalize 2>&1 \
  | tee "${PUZZLETRON_LOG_DIR}/axis_diagnostic_${SLURM_JOB_ID}_finalize.log"
' -- "$config"
