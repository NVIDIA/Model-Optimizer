#!/bin/bash
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

# One node of on-policy rejection-memory training (run inside the vLLM image, one copy per node):
# rollout servers on GPUs 0-2 and this node's trainer rank on GPU 3, which joins the other nodes'
# ranks through torchrun. Restarts resume from the latest complete checkpoint in OUT: the servers
# serve its export and the trainer continues from it.
#
# Env: CONFIG, OUT, NNODES, NODE_RANK, MASTER_ADDR; optional MASTER_PORT, EXTRA (key=value
# overrides for servers and trainer), SERVE_PYTHONPATH / TRAIN_PYTHONPATH (after this repo), LOG_DIR.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
ROOT=$(cd "$HERE/../../.." && pwd) # this ModelOpt tree goes first: the image may ship another one
: "${CONFIG:?}" "${OUT:?}" "${NNODES:?}" "${NODE_RANK:?}" "${MASTER_ADDR:?}"
MASTER_PORT=${MASTER_PORT:-65110}
PORT0=65101 # service ports above the ephemeral range
LOG_DIR=${LOG_DIR:-$OUT/logs}
read -r -a EXTRA_ARGS <<< "${EXTRA:-}"
CAPTURE=/dev/shm/rejmem-capture-${SLURM_JOB_ID:-$$}-$NODE_RANK # node-local record files
mkdir -p "$LOG_DIR" && rm -rf "$CAPTURE" && mkdir -p "$CAPTURE"
cleanup() {
  kill $(jobs -p) 2> /dev/null || true
  rm -rf "$CAPTURE"
}
trap cleanup EXIT

export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
for v in HF_HOME XDG_CACHE_HOME TRITON_CACHE_DIR FLASHINFER_WORKSPACE_BASE CUDA_CACHE_PATH HF_DATASETS_CACHE; do
  export "$v=/tmp/rejmem-cache/$v"
done

CKPT=$(ls -d "$OUT"/checkpoint-*[0-9] 2> /dev/null | awk -F- '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2- || true)
echo "node $NODE_RANK: resume from ${CKPT:-<init checkpoint>}"

URLS=""
for g in 0 1 2; do
  port=$((PORT0 + g))
  CUDA_VISIBLE_DEVICES=$g VLLM_CACHE_ROOT=/tmp/rejmem-cache/vllm$g TORCHINDUCTOR_CACHE_DIR=/tmp/rejmem-cache/ind$g \
    PYTHONPATH=$ROOT${SERVE_PYTHONPATH:+:$SERVE_PYTHONPATH} \
    python3 "$HERE/rollout.py" serve --config "$CONFIG" --port $port --capture_dir "$CAPTURE" \
    ${CKPT:+--draft "$CKPT/export"} "${EXTRA_ARGS[@]}" > "$LOG_DIR/serve${NODE_RANK}_g$g.log" 2>&1 &
  URLS="$URLS${URLS:+,}http://127.0.0.1:$port"
done
for g in 0 1 2; do
  until curl -fs "http://127.0.0.1:$((PORT0 + g))/health" > /dev/null; do
    if [ "$(jobs -rp | wc -l)" -lt 3 ]; then
      echo "a rollout server on node $NODE_RANK exited; see $LOG_DIR/serve${NODE_RANK}_g*.log"
      exit 1
    fi
    sleep 5
  done
done
echo "node $NODE_RANK: servers up after ${SECONDS}s"

CUDA_VISIBLE_DEVICES=3 TORCHINDUCTOR_CACHE_DIR=/tmp/rejmem-cache/ind-train \
  PYTHONPATH=$ROOT${TRAIN_PYTHONPATH:+:$TRAIN_PYTHONPATH} \
  python3 -m torch.distributed.run --nnodes "$NNODES" --node_rank "$NODE_RANK" --nproc_per_node 1 \
  --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT" \
  "$HERE/train_onpolicy.py" --config "$CONFIG" ${CKPT:+--resume_from "$CKPT"} \
  training.output_dir="$OUT" onpolicy.capture_dir="$CAPTURE" "onpolicy.server_urls=[$URLS]" "${EXTRA_ARGS[@]}"
