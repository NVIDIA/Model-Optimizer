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
: "${PUZZLETRON_WORKER_WORLD_SIZE:=${WORLD_SIZE:-}}"
: "${PUZZLETRON_WORKER_WORLD_SIZE:?set PUZZLETRON_WORKER_WORLD_SIZE to one worker-group world size}"

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
    --world-size "${PUZZLETRON_WORKER_WORLD_SIZE}" \
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
