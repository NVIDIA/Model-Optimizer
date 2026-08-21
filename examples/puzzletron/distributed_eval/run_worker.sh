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

PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN="${TORCHRUN:-torchrun}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
if [[ -n "${PUZZLETRON_GROUP_SIZE:-}" ]]; then
  : "${PUZZLETRON_GROUP_RANK:?set PUZZLETRON_GROUP_RANK with task identity}"
  : "${PUZZLETRON_RENDEZVOUS_ENDPOINT:?set PUZZLETRON_RENDEZVOUS_ENDPOINT with task identity}"
  : "${PUZZLETRON_RENDEZVOUS_ID:?set PUZZLETRON_RENDEZVOUS_ID with task identity}"
  NNODES="${PUZZLETRON_GROUP_SIZE}"
  NODE_RANK="${PUZZLETRON_GROUP_RANK}"
  RDZV_ENDPOINT="${PUZZLETRON_RENDEZVOUS_ENDPOINT}"
  RDZV_ID="${PUZZLETRON_RENDEZVOUS_ID}"
else
  NNODES="${NNODES:-1}"
  NODE_RANK="${NODE_RANK:-0}"
  RDZV_ID="${RDZV_ID:-distributed-eval-${SLURM_JOB_ID:-local}}"
  RDZV_ENDPOINT="${RDZV_ENDPOINT:-127.0.0.1:29500}"
fi
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
