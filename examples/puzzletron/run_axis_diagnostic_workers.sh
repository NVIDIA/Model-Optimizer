#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 CONFIG" >&2
  exit 2
fi

config=$(realpath -m "$1")
root=${PUZZLETRON_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
PUZZLETRON_VENV=${PUZZLETRON_VENV:-"${root}/.venv_new"}
: "${PUZZLETRON_IMAGE:?export PUZZLETRON_IMAGE with the container image path}"
: "${PUZZLETRON_CONTAINER_MOUNTS:?export PUZZLETRON_CONTAINER_MOUNTS for the container}"
image=${PUZZLETRON_IMAGE}
log_dir=${PUZZLETRON_LOG_DIR:-${root}/.runtime/sanity_logs}
AXIS_DIAGNOSTIC_NPROC_PER_NODE=${AXIS_DIAGNOSTIC_NPROC_PER_NODE:-8}
AXIS_DIAGNOSTIC_GPUS_PER_WORKER=${AXIS_DIAGNOSTIC_GPUS_PER_WORKER:-${AXIS_DIAGNOSTIC_NPROC_PER_NODE}}
AXIS_DIAGNOSTIC_WORKERS=${AXIS_DIAGNOSTIC_WORKERS:-${SLURM_NNODES:-1}}
if (( AXIS_DIAGNOSTIC_NPROC_PER_NODE < 1 || AXIS_DIAGNOSTIC_GPUS_PER_WORKER < 1 )); then
  echo "axis diagnostic worker GPU counts must be positive" >&2
  exit 2
fi
if (( AXIS_DIAGNOSTIC_NPROC_PER_NODE != AXIS_DIAGNOSTIC_GPUS_PER_WORKER )); then
  echo "one axis worker must expose every allocated GPU to its model instance" >&2
  exit 2
fi
if (( AXIS_DIAGNOSTIC_GPUS_PER_WORKER > 8 )); then
  echo "multi-node axis workers must use run_multinode_stage.sh with the stage-owned mesh" >&2
  exit 2
fi
mkdir -p "$log_dir"
export PUZZLETRON_ROOT="$root" PUZZLETRON_LOG_DIR="$log_dir" PUZZLETRON_VENV
export AXIS_DIAGNOSTIC_NPROC_PER_NODE AXIS_DIAGNOSTIC_GPUS_PER_WORKER
export PUZZLETRON_SETUP_ENV=${PUZZLETRON_SETUP_ENV:-}

srun_args=(
  --kill-on-bad-exit=0 \
  --nodes="${AXIS_DIAGNOSTIC_WORKERS}" \
  --ntasks="${AXIS_DIAGNOSTIC_WORKERS}" \
  --ntasks-per-node=1 \
  --gpus-per-task="${AXIS_DIAGNOSTIC_GPUS_PER_WORKER}" \
  --gpu-bind=none \
)

srun "${srun_args[@]}" \
  --container-image="$image" \
  --container-mounts="${PUZZLETRON_CONTAINER_MOUNTS}" \
  --container-workdir="$root" \
  --mpi=pmix \
  /bin/bash -lc '
set -Eeuo pipefail
if [[ -n "${PUZZLETRON_SETUP_ENV:-}" ]]; then source "${PUZZLETRON_SETUP_ENV}"; fi
source "${PUZZLETRON_VENV}/bin/activate"
export PYTHONPATH="${PUZZLETRON_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
examples/puzzletron/run_axis_diagnostic_task.sh "$1" 2>&1 \
  | tee "${PUZZLETRON_LOG_DIR}/axis_diagnostic_${SLURM_JOB_ID}_task${SLURM_PROCID}.log"
' -- "$config"

srun \
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
source "${PUZZLETRON_VENV}/bin/activate"
export PYTHONPATH="${PUZZLETRON_ROOT}:${PYTHONPATH:-}"
python examples/puzzletron/run_axis_diagnostic_worker.py --config "$1" --finalize 2>&1 \
  | tee "${PUZZLETRON_LOG_DIR}/axis_diagnostic_${SLURM_JOB_ID}_finalize.log"
' -- "$config"
