#!/usr/bin/env bash
#SBATCH --job-name=puzzletron-eval-worker
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=puzzle_runs/logs/distributed_eval_worker_%j.log

set -Eeuo pipefail

: "${REPO_ROOT:?export REPO_ROOT with sbatch --export}"
: "${CAMPAIGN_DIR:?export CAMPAIGN_DIR}"
: "${CONFIG_PATH:?export CONFIG_PATH}"
: "${CONTAINER_IMAGE:?export CONTAINER_IMAGE with sbatch --export}"
: "${PUZZLETRON_CONTAINER_MOUNTS:?export PUZZLETRON_CONTAINER_MOUNTS with sbatch --export}"

PUZZLETRON_VENV="${PUZZLETRON_VENV:-${REPO_ROOT}/.venv_new}"
PUZZLETRON_SETUP_ENV="${PUZZLETRON_SETUP_ENV:-}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NNODES="${SLURM_NNODES}"
MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
MASTER_PORT="${MASTER_PORT:-$((29500 + SLURM_JOB_ID % 1000))}"

cd "${REPO_ROOT}"

export MASTER_ADDR MASTER_PORT NNODES NPROC_PER_NODE PUZZLETRON_CONTAINER_MOUNTS
export PUZZLETRON_SETUP_ENV PUZZLETRON_VENV
export RDZV_ID="distributed-eval-${SLURM_JOB_ID}"
export RDZV_ENDPOINT="${MASTER_ADDR}:${MASTER_PORT}"
export WORKER_HOST="${MASTER_ADDR}"
export WORKER_ID="${SLURM_JOB_ID}-group-${WORKER_GROUP_INDEX:-0}"

srun \
  --nodes "${NNODES}" \
  --ntasks "${NNODES}" \
  --ntasks-per-node 1 \
  --container-image "${CONTAINER_IMAGE}" \
  --container-mounts "${PUZZLETRON_CONTAINER_MOUNTS}" \
  --container-workdir "${REPO_ROOT}" \
  --mpi=pmix \
  bash -lc '
    set -Eeuo pipefail
    if [[ -n "${PUZZLETRON_SETUP_ENV:-}" ]]; then
      source "${PUZZLETRON_SETUP_ENV}"
    fi
    source "${PUZZLETRON_VENV}/bin/activate"
    export PYTHONPATH=$(pwd):${PYTHONPATH:-}
    export NODE_RANK=${SLURM_NODEID}
    exec bash examples/puzzletron/distributed_eval/run_worker.sh
  '
