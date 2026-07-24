#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 STAGE CONFIG_PATH [OVERRIDE ...]" >&2
  exit 2
fi

STAGE=$1
CONFIG_PATH=$(realpath -m "$2")
shift 2
OVERRIDES=("$@")

SCRIPT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ROOT=${PUZZLETRON_ROOT:-${SLURM_SUBMIT_DIR:-"${SCRIPT_ROOT}"}}
PUZZLETRON_VENV=${PUZZLETRON_VENV:-"${ROOT}/.venv_new"}
: "${PUZZLETRON_IMAGE:?export PUZZLETRON_IMAGE with the container image path}"
: "${PUZZLETRON_CONTAINER_MOUNTS:?export PUZZLETRON_CONTAINER_MOUNTS for the container}"
IMAGE=${PUZZLETRON_IMAGE}
LOG_DIR=${PUZZLETRON_LOG_DIR:-"${ROOT}/puzzle_runs/logs"}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}

: "${SLURM_JOB_ID:?run this script with sbatch or inside a Slurm allocation}"
: "${SLURM_JOB_NODELIST:?Slurm did not provide the allocated node list}"

mkdir -p "${LOG_DIR}"
MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
MASTER_PORT=${MASTER_PORT:-$((29500 + SLURM_JOB_ID % 1000))}
export ROOT STAGE CONFIG_PATH LOG_DIR NPROC_PER_NODE MASTER_ADDR MASTER_PORT PUZZLETRON_VENV
export PUZZLETRON_SETUP_ENV=${PUZZLETRON_SETUP_ENV:-}

srun_args=(--nodes="${SLURM_NNODES}" --ntasks="${SLURM_NNODES}" --ntasks-per-node=1)

srun "${srun_args[@]}" \
  --container-image="${IMAGE}" \
  --container-mounts="${PUZZLETRON_CONTAINER_MOUNTS}" \
  --container-workdir="${ROOT}" \
  --mpi=pmix \
  /bin/bash -lc '
set -Eeuo pipefail
if [[ -n "${PUZZLETRON_SETUP_ENV:-}" ]]; then source "${PUZZLETRON_SETUP_ENV}"; fi
source "${PUZZLETRON_VENV}/bin/activate"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

log="${LOG_DIR}/multinode_${STAGE}_${SLURM_JOB_ID}_node${SLURM_NODEID}.log"
stage_option=--stage
if [[ "${PUZZLETRON_DIRECT_WORKER:-0}" == "1" ]]; then
  stage_option=--worker-stage
fi
args=(
  "${ROOT}/examples/puzzletron/main.py"
  --config "${CONFIG_PATH}"
  "${stage_option}" "${STAGE}"
)
if [[ "${PUZZLETRON_FORCE_STAGE:-0}" == "1" ]]; then
  args+=(--force)
fi
while [[ $# -gt 0 ]]; do
  args+=(--override "$1")
  shift
done

torchrun \
  --nnodes="${SLURM_NNODES}" \
  --nproc-per-node="${NPROC_PER_NODE}" \
  --node-rank="${SLURM_NODEID}" \
  --rdzv-backend=c10d \
  --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  --max-restarts=0 \
  "${args[@]}" 2>&1 | tee "${log}"
' -- "${OVERRIDES[@]}"
