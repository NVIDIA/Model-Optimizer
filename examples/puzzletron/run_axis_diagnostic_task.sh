#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 CONFIG" >&2
  exit 2
fi

: "${AXIS_DIAGNOSTIC_NPROC_PER_NODE:?AXIS_DIAGNOSTIC_NPROC_PER_NODE is required}"
echo "axis diagnostic binding: node=${SLURMD_NODENAME:-unknown} task=${SLURM_PROCID:?SLURM_PROCID is required} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

exec torchrun \
  --standalone \
  "--nproc-per-node=${AXIS_DIAGNOSTIC_NPROC_PER_NODE}" \
  examples/puzzletron/run_axis_diagnostic_worker.py \
  --config "$1" \
  --axis-index "${SLURM_PROCID:?SLURM_PROCID is required}"
