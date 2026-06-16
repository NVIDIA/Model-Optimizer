#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# EAGLE3 streaming training: a `vllm serve` (KV-transfer hidden-states producer)
# runs alongside the trainer, moving captured hidden states straight to the trainer
# over NIXL RDMA (no disk round-trip; a small HTTP sidecar only hands out metadata).
#
# CANONICAL TOPOLOGY/DISPATCH (per-example YAMLs cross-reference here). Topology is
# auto-chosen from the Slurm allocation (yaml `nodes:`) and $SERVE_NODES; nemo_run
# runs this script once per node, branching on $SLURM_NODEID:
#   nodes == 1  -> co-located: vllm serve on $SERVE_GPU, trainer on the rest.
#   nodes >= 2  -> split: nodes 0..SERVE_NODES-1 each run an independent whole-node
#                 vllm serve replica; nodes SERVE_NODES..NNODES-1 are multi-node-DDP
#                 trainers. SERVE_NODES default 1. Rendezvous over shared
#                 /scratchspace: each serve i publishes .serve_addr.i; head trainer
#                 (first trainer node = accelerate machine_rank 0) publishes its IP;
#                 trainers collect every serve address.
# Map-style dataset: DistributedSampler shards the corpus across trainer ranks, each
# rank fetches only its shard round-robin across the SERVE_NODES replicas
# (data.streaming_server_url = comma-joined list).
#
# Env vars (required):
#   HF_MODEL_CKPT       Target model path; vllm serve model arg (= served-model-name)
#                       and trainer data.streaming_model_name.
#   EAGLE_CAPTURE_IDS   JSON 1-based layer ids to capture = default_eagle_aux_layer_ids(L)
#                       +1, plus final layer L. Qwen3-8B (L=36): [1,17,32]->[2,18,33,36].
#
# Env vars (optional):
#   SERVE_NODES         multi-node: dedicated serve replica nodes (0..SERVE_NODES-1). default 1
#   SERVE_GPU_MEM_UTIL  default 0.4 single-node / 0.9 multi-node serve node
#   SERVE_READY_TIMEOUT server startup wait, seconds. default 900
#   SERVE_EXTRA_ARGS    extra `vllm serve` flags (e.g. --trust-remote-code)
#   SERVE_CPU_OFFLOAD_GB  GB/GPU offloaded to host RAM (fits big models on too-few GPUs; slower)
#   SERVE_MAX_MODEL_LEN   cap context length (trims KV/activation)
#   SERVE_MAX_NUM_SEQS    cap concurrent sequences (trims KV/activation)
#   SERVE_HOST          single-node: bind/connect host. default 127.0.0.1
#   SERVE_GPU           single-node: CUDA_VISIBLE_DEVICES for vllm. default "0"
#   SERVE_TP            tensor-parallel size. default 1 single-node / all serve-node GPUs
#   TRAIN_GPUS          single-node: trainer CUDA_VISIBLE_DEVICES. default = all but SERVE_GPU
#   SERVE_ADVERTISE_IP  multi-node: address node 1 dials. default node 0's routable IP

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "${SCRIPT_DIR}/../service_utils.sh"

###################################################################################################
# Container provisioning: the vllm image lacks modelopt's .dist-info and the real
# pyproject, so synthesize a minimal pyproject (scoped `include` avoids setuptools'
# flat-layout error) and `pip install -e .`.

TOML=modules/Model-Optimizer/pyproject.toml
if [ ! -f "$TOML" ]; then
    cat > "$TOML" <<'EOF'
[build-system]
requires = ["setuptools>=80"]
build-backend = "setuptools.build_meta"

[project]
name = "nvidia-modelopt"
version = "0.0.0"
dependencies = [
    "omegaconf>=2.3.0",
    "PyYAML>=6.0",
    "pulp<4.0",
    "pydantic>=2.0",
    "regex",
    "rich",
    "safetensors",
    "scipy",
    "nvidia-ml-py>=12",
    "packaging",
    "tqdm",
]

[tool.setuptools.packages.find]
include = ["modelopt*", "modelopt_recipes*"]
EOF
fi
pip install --no-cache-dir -e modules/Model-Optimizer/
pip install --no-cache-dir -r modules/Model-Optimizer/examples/speculative_decoding/requirements.txt
pip install --no-cache-dir 'datasets' 'huggingface-hub>=1.2.1'
export PATH=$PATH:/workspace/.local/bin

###################################################################################################

trap 'error_handler $0 $LINENO' ERR

if [ -z "$HF_MODEL_CKPT" ]; then
    echo "ERROR: HF_MODEL_CKPT must be set." >&2; exit 1
fi
if [ -z "$EAGLE_CAPTURE_IDS" ]; then
    echo "ERROR: EAGLE_CAPTURE_IDS must be set (e.g. '[2, 18, 33, 36]' for Qwen3-8B)." >&2; exit 1
fi

# Forwarded verbatim to the trainer; capture before the helpers below run.
SCRIPT_ARGS=("$@")

SERVE_PORT="${SERVE_PORT:-8765}"
SERVE_READY_TIMEOUT="${SERVE_READY_TIMEOUT:-900}"
SERVE_NODES="${SERVE_NODES:-1}"
SERVE_LOG="/scratchspace/vllm_serve.log"   # serve nodes override with a per-node path
# Namespace rendezvous/sentinel files per Slurm job (SLURM_JOB_ID: same across an
# allocation's nodes, unique across allocations) so concurrent allocations on the
# shared mount don't clobber each other's addresses. Fixed token off-Slurm.
RUN_ID="${SLURM_JOB_ID:-local}"
SERVE_ADDR_FILE="/scratchspace/.serve_addr.${RUN_ID}"
DONE_FILE="/scratchspace/.training_done.${RUN_ID}"
SERVE_PID=""

cleanup() {
    [ -n "$SERVE_PID" ] || return 0
    echo "Cleaning up vllm serve (PID=$SERVE_PID)..."
    kill "$SERVE_PID" 2>/dev/null || true
    wait "$SERVE_PID" 2>/dev/null || true
}

gpus_on_node() { nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -n1; }

# Resolve a routable IP (other nodes must dial it). `hostname -I` can list a
# link-local/loopback first, so prefer the Slurm node name, then first non-lo/non-ll IP.
#   $1 = optional override (SERVE_ADVERTISE_IP / TRAINER_ADVERTISE_IP)
resolve_routable_ip() {
    local ip="$1"
    [ -z "$ip" ] && ip=$(getent hosts "${SLURMD_NODENAME:-$(hostname)}" 2>/dev/null | awk '{print $1}' | head -1)
    [ -z "$ip" ] && ip=$(hostname -I | tr ' ' '\n' | grep -vE '^(127\.|169\.254\.|fe80:|::1)' | head -1)
    [ -z "$ip" ] && ip=$(hostname -I | awk '{print $1}')
    echo "$ip"
}

# Start vllm serve in the background. Sets SERVE_PID.
#   $1 = bind host   $2 = tensor-parallel size   $3 = CUDA_VISIBLE_DEVICES ("" -> all)
launch_vllm() {
    local bind_host="$1" tp="$2" cvd="$3"
    echo "Launching vllm serve on ${bind_host}:${SERVE_PORT} (TP=${tp}, CUDA_VISIBLE_DEVICES=${cvd:-all}, mem=${SERVE_GPU_MEM_UTIL}, log: $SERVE_LOG)..."
    # Pin GPUs only for a non-empty set; empty CUDA_VISIBLE_DEVICES hides ALL, so unset = whole node.
    local -a gpu_env=()
    [ -n "$cvd" ] && gpu_env=(env "CUDA_VISIBLE_DEVICES=$cvd")
    # Optional memory knobs (see header). Space-free env values to survive nemo_run's unquoted export.
    local -a opt_args=()
    [ -n "${SERVE_CPU_OFFLOAD_GB:-}" ] && opt_args+=(--cpu-offload-gb "$SERVE_CPU_OFFLOAD_GB")
    [ -n "${SERVE_MAX_MODEL_LEN:-}" ]  && opt_args+=(--max-model-len "$SERVE_MAX_MODEL_LEN")
    [ -n "${SERVE_MAX_NUM_SEQS:-}" ]   && opt_args+=(--max-num-seqs "$SERVE_MAX_NUM_SEQS")
    # --no-enable-chunked-prefill / --no-enable-prefix-caching: connector captures hidden states during prefill; both skip recomputing cached/partial prefixes, yielding short/empty hidden_states. Required.
    # --no-enable-flashinfer-autotune: on NVFP4 MoE the autotuner re-tunes on the first serving step and stalls a worker past vLLM's execute-model timeout, killing EngineCore.
    # Hidden states move serve -> trainer over NIXL RDMA (no disk round-trip): one
    # pre-registered pinned pool per serve, a tiny HTTP sidecar hands out per-request
    # transfer descriptors. Replicated across TP ranks, so only rank 0 owns the pool.
    KVCFG="{\"kv_connector\":\"RdmaHiddenStatesConnector\",\"kv_connector_module_path\":\"modelopt.torch.speculative.plugins.rdma_hidden_states_connector\",\"kv_role\":\"kv_producer\",\"kv_connector_extra_config\":{\"sidecar_port\":\"${HS_SIDECAR_PORT:-18999}\",\"pool_slots\":\"${HS_POOL_SLOTS:-16}\",\"max_tokens\":\"${HS_MAX_TOKENS:-4096}\"}}"
    "${gpu_env[@]}" vllm serve "$HF_MODEL_CKPT" \
        --host "$bind_host" \
        --port "$SERVE_PORT" \
        --tensor-parallel-size "$tp" \
        --enforce-eager \
        --no-enable-chunked-prefill \
        --no-enable-prefix-caching \
        --no-enable-flashinfer-autotune \
        --gpu-memory-utilization "$SERVE_GPU_MEM_UTIL" \
        "${opt_args[@]}" \
        ${SERVE_EXTRA_ARGS:-} \
        --speculative-config "{
            \"method\":\"extract_hidden_states\",
            \"num_speculative_tokens\":1,
            \"draft_model_config\":{
                \"hf_config\":{
                    \"eagle_aux_hidden_state_layer_ids\":$EAGLE_CAPTURE_IDS
                }
            }
        }" \
        --kv-transfer-config "$KVCFG" \
        > "$SERVE_LOG" 2>&1 &
    SERVE_PID=$!
}

# Poll until the server answers (or, if we own it, dies). $1 = base URL.
wait_vllm_ready() {
    local url="$1" tries=$(( SERVE_READY_TIMEOUT / 5 ))
    echo "Waiting for vllm serve at ${url} to become ready (up to ${SERVE_READY_TIMEOUT}s)..."
    for ((i = 0; i < tries; i++)); do
        if curl -fsS "${url}/v1/models" > /dev/null 2>&1; then echo "vllm serve ready."; return 0; fi
        if [ -n "$SERVE_PID" ] && ! kill -0 "$SERVE_PID" 2>/dev/null; then
            echo "vllm serve died early. Tail of $SERVE_LOG:"; tail -100 "$SERVE_LOG"; return 1
        fi
        sleep 5
    done
    echo "Server not ready in ${SERVE_READY_TIMEOUT}s. Tail:"; tail -100 "$SERVE_LOG"; return 1
}

# Run the trainer then export the HF checkpoint.
#   $1 = streaming server base URL   $2 = CUDA_VISIBLE_DEVICES ("" -> all)
# DataLoader workers = in-flight fetches per rank; keep modest so (ranks x workers) stays near the serve's max_num_seqs.
run_trainer_and_export() {
    local url="$1" cvd="$2"
    # Optional multi-node trainer routing (see dispatch). Defaults: 1 node, no --num_nodes, export on rank 0.
    local num_tnodes="${3:-1}" head_ip="${4:-}" mrank="${5:-0}"
    echo "Launching trainer (server=${url}, CUDA_VISIBLE_DEVICES=${cvd:-all}, trainer_nodes=${num_tnodes}, machine_rank=${mrank})..."
    # Empty cvd -> all GPUs (don't set the var; "" hides all).
    local -a gpu_env=()
    [ -n "$cvd" ] && gpu_env=(env "CUDA_VISIBLE_DEVICES=$cvd")
    # accelerate multi-node routing only when >1 trainer node.
    local -a mn_args=()
    if [ "${num_tnodes}" -gt 1 ]; then
        mn_args=(--num_nodes "$num_tnodes" --head_node_ip "$head_ip" --machine_rank "$mrank")
    fi
    "${gpu_env[@]}" bash modules/Model-Optimizer/examples/speculative_decoding/launch_train.sh \
        "${SCRIPT_ARGS[@]}" \
        "${mn_args[@]}" \
        data.streaming_server_url="$url" \
        data.streaming_model_name="$HF_MODEL_CKPT" \
        training.dataloader_num_workers="${STREAMING_NUM_WORKERS:-4}" \
        || { echo "ERROR: trainer failed." >&2; return 1; }

    # Export only on the head trainer (machine_rank 0); non-head nodes would race the same export dir. Export reads training.output_dir, not the serve.
    if [ "${mrank}" -ne 0 ]; then
        echo "machine_rank=${mrank}: training done, skipping export (head trainer handles it)."
        return 0
    fi

    # Derive checkpoint dir from the forwarded training.output_dir= dotlist (EAGLE default)
    # so EAGLE/DFlash runs each export their own dir. EXPORT_EXTRA_ARGS lets DFlash on a
    # custom-modeling base (e.g. Kimi) pass --trust_remote_code; empty by default.
    local out_dir
    out_dir=$(printf '%s\n' "${SCRIPT_ARGS[@]}" | sed -n 's/^training\.output_dir=//p' | tail -1)
    # Fail loud rather than guess a default: a wrong dir would silently export the
    # wrong checkpoint. Every streaming yaml already forwards training.output_dir=.
    if [ -z "$out_dir" ]; then
        echo "ERROR: no training.output_dir= forwarded in SCRIPT_ARGS; cannot locate checkpoint to export." >&2
        return 1
    fi
    python3 modules/Model-Optimizer/examples/speculative_decoding/scripts/export_hf_checkpoint.py \
        --model_path "$out_dir" \
        --export_path "${EXPORT_PATH:-/scratchspace/export}" \
        ${EXPORT_EXTRA_ARGS:-}
}

# Topology dispatch (see header): branch on $SLURM_NNODES / $SLURM_NODEID.
NNODES="${SLURM_NNODES:-1}"
NODEID="${SLURM_NODEID:-0}"

# Need >=1 trainer node: with SERVE_NODES >= NNODES every node takes the serve branch,
# so nobody publishes the rendezvous/DONE_FILE and serve nodes block forever.
if [ "$NNODES" -gt 1 ] && [ "$SERVE_NODES" -ge "$NNODES" ]; then
    echo "ERROR: SERVE_NODES ($SERVE_NODES) must be < SLURM_NNODES ($NNODES); need >=1 trainer node." >&2
    exit 1
fi

if [ "$NNODES" -le 1 ]; then
    # ----------------------------- single node -----------------------------
    SERVE_HOST="${SERVE_HOST:-127.0.0.1}"
    SERVE_GPU="${SERVE_GPU:-0}"
    SERVE_TP="${SERVE_TP:-1}"
    SERVE_GPU_MEM_UTIL="${SERVE_GPU_MEM_UTIL:-0.4}"

    if [ -z "$TRAIN_GPUS" ]; then
        TRAIN_GPUS=$(python3 - <<PY
total = int("$(gpus_on_node)")
exclude = {int(x) for x in "$SERVE_GPU".split(",") if x != ""}
print(",".join(str(i) for i in range(total) if i not in exclude))
PY
)
    fi
    if [ -z "$TRAIN_GPUS" ]; then
        echo "ERROR: no GPUs left for the trainer (SERVE_GPU=$SERVE_GPU consumed them all)." >&2; exit 1
    fi

    trap cleanup INT TERM EXIT
    launch_vllm "$SERVE_HOST" "$SERVE_TP" "$SERVE_GPU"
    wait_vllm_ready "http://${SERVE_HOST}:${SERVE_PORT}" || exit 1
    run_trainer_and_export "http://${SERVE_HOST}:${SERVE_PORT}" "$TRAIN_GPUS" || exit 1

elif [ "$NODEID" -lt "$SERVE_NODES" ]; then
    # ---------------------- multi-node: serve node(s) ----------------------
    # Each runs a whole-node vllm serve replica and publishes ${SERVE_ADDR_FILE}.${NODEID}.
    SERVE_GPU_MEM_UTIL="${SERVE_GPU_MEM_UTIL:-0.9}"     # dedicated node -> use most of it
    SERVE_TP="${SERVE_TP:-$(gpus_on_node)}"              # default: all GPUs on this node
    SERVE_LOG="/scratchspace/vllm_serve.${NODEID}.log"  # per-node log (avoid collision)
    rm -f "${SERVE_ADDR_FILE}.${NODEID}"                 # clear own stale address
    [ "$NODEID" -eq 0 ] && rm -f "$DONE_FILE"            # node 0 clears the shared sentinel once

    trap cleanup INT TERM EXIT
    launch_vllm "0.0.0.0" "$SERVE_TP" ""
    wait_vllm_ready "http://127.0.0.1:${SERVE_PORT}" || exit 1

    serve_addr=$(resolve_routable_ip "${SERVE_ADVERTISE_IP:-}")
    echo "$serve_addr" > "${SERVE_ADDR_FILE}.${NODEID}"
    echo "Serve node ${NODEID}/${SERVE_NODES} published ${serve_addr}; holding up until training signals done..."
    while [ ! -f "$DONE_FILE" ]; do sleep 10; done
    echo "Training-done sentinel seen; serve node ${NODEID} exiting (EXIT trap stops vllm)."

else
    # -------------------- multi-node: trainer node(s) ----------------------
    # Trainer nodes SERVE_NODES..NNODES-1 -> 0-based accelerate machine ranks.
    NUM_TRAINER_NODES=$(( NNODES - SERVE_NODES ))
    TRAINER_RANK=$(( NODEID - SERVE_NODES ))
    TRAINER_ADDR_FILE="/scratchspace/.trainer_addr.${RUN_ID}"  # per-job (see RUN_ID)

    # Only head trainer (rank 0) signals serves to release on exit; a non-head node
    # exiting first must NOT tear them down early.
    if [ "$TRAINER_RANK" -eq 0 ]; then
        trap 'touch "$DONE_FILE" 2>/dev/null || true' EXIT
        rm -f "$TRAINER_ADDR_FILE"                 # clear stale rendezvous state
    fi

    # Collect serve addresses into the comma-joined URL list the dataset round-robins across.
    echo "Trainer node (rank ${TRAINER_RANK}/${NUM_TRAINER_NODES}) waiting for ${SERVE_NODES} serve address(es)..."
    URLS=""
    for ((s = 0; s < SERVE_NODES; s++)); do
        af="${SERVE_ADDR_FILE}.${s}"
        for ((i = 0; i < SERVE_READY_TIMEOUT; i++)); do
            [ -f "$af" ] && break
            sleep 1
        done
        [ -f "$af" ] || { echo "ERROR: serve node ${s} never published its address." >&2; exit 1; }
        surl="http://$(cat "$af"):${SERVE_PORT}"
        wait_vllm_ready "$surl" || exit 1
        URLS="${URLS:+$URLS,}$surl"
    done
    echo "Trainer rank ${TRAINER_RANK} using serve URLs: ${URLS}"

    if [ "$NUM_TRAINER_NODES" -le 1 ]; then
        # 1 trainer node: single-node DDP.
        run_trainer_and_export "$URLS" "" || exit 1
    else
        # >1 trainer node: head publishes its routable IP for accelerate rendezvous (29500); all read and join.
        if [ "$TRAINER_RANK" -eq 0 ]; then
            head_addr=$(resolve_routable_ip "${TRAINER_ADVERTISE_IP:-}")
            echo "$head_addr" > "$TRAINER_ADDR_FILE"
            echo "Head trainer (rank 0) published ${head_addr} for accelerate rendezvous."
        else
            echo "Trainer rank ${TRAINER_RANK} waiting for head-trainer address..."
            for ((i = 0; i < SERVE_READY_TIMEOUT; i++)); do
                [ -f "$TRAINER_ADDR_FILE" ] && break
                sleep 1
            done
            [ -f "$TRAINER_ADDR_FILE" ] || { echo "ERROR: head trainer never published its address." >&2; exit 1; }
        fi
        HEAD_IP=$(cat "$TRAINER_ADDR_FILE")
        run_trainer_and_export "$URLS" "" "$NUM_TRAINER_NODES" "$HEAD_IP" "$TRAINER_RANK" || exit 1
    fi
fi

###################################################################################################
