#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 CONFIG" >&2
  exit 2
fi

local_task=${SLURM_LOCALID:?SLURM_LOCALID is required}
if (( local_task < 0 || local_task > 3 )); then
  echo "axis diagnostic expects four tasks per 8-GPU node; got SLURM_LOCALID=${local_task}" >&2
  exit 2
fi
first_gpu=$((2 * local_task))
export CUDA_VISIBLE_DEVICES="${first_gpu},$((first_gpu + 1))"
echo "axis diagnostic binding: node=${SLURMD_NODENAME:-unknown} task=${SLURM_PROCID} local_task=${local_task} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

exec torchrun \
  --standalone \
  --nproc-per-node=2 \
  examples/puzzletron/run_axis_diagnostic_worker.py \
  --config "$1" \
  --axis-index "${SLURM_PROCID:?SLURM_PROCID is required}"
