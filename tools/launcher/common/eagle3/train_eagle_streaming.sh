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

# EAGLE3 streaming training: runs a `vllm serve` (KV-transfer producer of hidden
# states) alongside the trainer and routes hidden states over HTTP rather than
# dumping to disk. Sibling of train_eagle.sh.
#
# Topology is chosen automatically from the Slurm allocation (the launcher yaml's
# `nodes:` field); nemo_run runs this script once per node, so it branches on
# $SLURM_NODEID:
#   nodes == 1  -> co-located: vllm serve on $SERVE_GPU, trainer on the rest of
#                  the local GPUs (original single-node behavior).
#   nodes == 2  -> split: node 0 runs vllm serve on all its GPUs, node 1 runs
#                  the trainer on all its GPUs. Roles rendezvous through the
#                  shared /scratchspace mount (node 0 publishes its serve
#                  address; the trainer signals completion).
#   nodes >= 3  -> 1 serve node (node 0) + N trainer nodes (nodes 1..NNODES-1)
#                  doing multi-node DDP. The head trainer (node 1, accelerate
#                  machine_rank 0) publishes its IP for accelerate's c10d
#                  rendezvous; all trainer nodes read both the serve address and
#                  the head-trainer address from /scratchspace. NOTE: only global
#                  rank 0 fetches hidden states from the single serve and
#                  broadcasts to the rest (DataLoaderDispatcher), so the single
#                  serve is the throughput ceiling — adding trainer nodes scales
#                  effective batch / compute, not data-production throughput.
#
# Env vars (required):
#   HF_MODEL_CKPT       Target model path. Used by both vllm serve (as the
#                       model arg, becomes the served-model-name) and the
#                       trainer (data.streaming_model_name).
#   EAGLE_CAPTURE_IDS   JSON list of 1-based layer ids vllm should capture.
#                       Must equal default_eagle_aux_layer_ids(L) shifted by +1,
#                       plus the final layer L. For Qwen3-8B (L=36):
#                       default = [1,17,32] -> capture = [2,18,33,36].
#
# Env vars (optional):
#   SERVE_PORT          default 8765
#   SERVE_GPU_MEM_UTIL  default 0.4 (single-node) / 0.9 (multi-node serve node)
#   SERVE_READY_TIMEOUT seconds to wait for the server to come up. default 900
#   SERVE_EXTRA_ARGS    extra flags appended to `vllm serve` (e.g. --trust-remote-code)
#   SERVE_CPU_OFFLOAD_GB  GB of weights/GPU to offload to host RAM (fits big models
#                         on too-few GPUs; slower). e.g. "10"
#   SERVE_MAX_MODEL_LEN   cap vllm context length (trims KV/activation). e.g. "4096"
#   SERVE_MAX_NUM_SEQS    cap concurrent sequences (trims KV/activation). e.g. "8"
#   SERVE_HOST          single-node only: bind/connect host. default 127.0.0.1
#   SERVE_GPU           single-node only: CUDA_VISIBLE_DEVICES for vllm. default "0"
#   SERVE_TP            tensor-parallel size. default 1 (single-node) / all GPUs
#                       on the serve node (multi-node)
#   TRAIN_GPUS          single-node only: CUDA_VISIBLE_DEVICES for the trainer.
#                       default = all local GPUs except SERVE_GPU.
#   SERVE_ADVERTISE_IP  multi-node only: address node 1 should dial. default is
#                       node 0's routable IP (its resolved Slurm node name, else
#                       its first non-loopback / non-link-local IP).
#
# All script args are forwarded to launch_train.sh (typically: --config <yaml>
# plus OmegaConf dotlist overrides).

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "${SCRIPT_DIR}/../service_utils.sh"

###################################################################################################
# Container provisioning
#
# vllm/vllm-openai:* has vllm and torch but not modelopt or the speculative
# trainer's deps. modelopt is bind-mounted at
# /usr/local/lib/python3.12/dist-packages/modelopt, but it has no .dist-info
# (so `importlib.metadata.version('nvidia-modelopt')` would fail). nemo_run
# only ships modelopt subdirs, not the real pyproject.toml, so we synthesize
# a minimal one with a correctly-scoped setuptools.packages.find include —
# without `include = ["modelopt*"]`, setuptools sees both `modelopt/` and
# `modelopt_recipes/` at the top level and refuses with a "flat-layout"
# error. We then `pip install -e .` to register the dist-info.

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

# Everything passed to this script (--config <yaml> + OmegaConf dotlist) is
# forwarded verbatim to the trainer. Capture it before the helpers below run.
SCRIPT_ARGS=("$@")

SERVE_PORT="${SERVE_PORT:-8765}"
SERVE_READY_TIMEOUT="${SERVE_READY_TIMEOUT:-900}"
SERVE_SCRATCH="/scratchspace/streaming_serve_scratch"
SERVE_LOG="/scratchspace/vllm_serve.log"
# Multi-node rendezvous over the shared /scratchspace mount (lustre, visible on
# every node): node 0 publishes its address here, node 1 signals completion here.
SERVE_ADDR_FILE="/scratchspace/.serve_addr"
DONE_FILE="/scratchspace/.training_done"
SERVE_PID=""
mkdir -p "$SERVE_SCRATCH"

cleanup() {
    [ -n "$SERVE_PID" ] || return 0
    echo "Cleaning up vllm serve (PID=$SERVE_PID)..."
    kill "$SERVE_PID" 2>/dev/null || true
    wait "$SERVE_PID" 2>/dev/null || true
}

gpus_on_node() { nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -n1; }

# Start vllm serve in the background. Sets SERVE_PID.
#   $1 = bind host   $2 = tensor-parallel size   $3 = CUDA_VISIBLE_DEVICES ("" -> all)
launch_vllm() {
    local bind_host="$1" tp="$2" cvd="$3"
    echo "Launching vllm serve on ${bind_host}:${SERVE_PORT} (TP=${tp}, CUDA_VISIBLE_DEVICES=${cvd:-all}, mem=${SERVE_GPU_MEM_UTIL}, log: $SERVE_LOG)..."
    # Only pin GPUs when a non-empty set is given; an empty CUDA_VISIBLE_DEVICES
    # would expose *zero* GPUs (not all), so leave it unset to use the whole node.
    local -a gpu_env=()
    [ -n "$cvd" ] && gpu_env=(env "CUDA_VISIBLE_DEVICES=$cvd")
    # Optional single-value memory knobs (see header), assembled into --flag
    # value pairs. Each is a space-free env value so it survives nemo_run's
    # unquoted `export FOO=value`.
    local -a opt_args=()
    [ -n "${SERVE_CPU_OFFLOAD_GB:-}" ] && opt_args+=(--cpu-offload-gb "$SERVE_CPU_OFFLOAD_GB")
    [ -n "${SERVE_MAX_MODEL_LEN:-}" ]  && opt_args+=(--max-model-len "$SERVE_MAX_MODEL_LEN")
    [ -n "${SERVE_MAX_NUM_SEQS:-}" ]   && opt_args+=(--max-num-seqs "$SERVE_MAX_NUM_SEQS")
    # --no-enable-chunked-prefill / --no-enable-prefix-caching: the
    # ExampleHiddenStatesConnector captures hidden states during prefill; both
    # features skip recomputing cached/partial prefixes, which yields short or
    # empty hidden_states. Required, not optional.
    # --no-enable-flashinfer-autotune: on big NVFP4 MoE (Kimi) the flashinfer
    # trtllm_fp4_block_scale_moe autotuner re-tunes on the first real serving
    # step and stalls a worker past vLLM's execute-model timeout -> EngineCore
    # dies with "RPC call to sample_tokens timed out" -> 500s -> trainer aborts.
    # Disabling autotune keeps kernels static (and pairs with the larger
    # VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS set in the example env).
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
        --kv-transfer-config "{
            \"kv_connector\":\"ExampleHiddenStatesConnector\",
            \"kv_role\":\"kv_producer\",
            \"kv_connector_extra_config\":{\"shared_storage_path\":\"$SERVE_SCRATCH\"}
        }" \
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
# The streaming dataset is map-style now, so fetch concurrency comes from the
# DataLoader's workers (each worker = one in-flight request). STREAMING_NUM_WORKERS
# sets that; keep it modest so (ranks-per-server x workers) stays near the server's
# max_num_seqs (flooding a cold NVFP4 MoE server kills EngineCore). 0 disables
# prefetch (serialized fetches) and is usually too slow.
run_trainer_and_export() {
    local url="$1" cvd="$2"
    # Optional multi-node trainer routing (see dispatch section). Defaults keep
    # the original single-trainer-node behavior: no --num_nodes, export on rank 0.
    local num_tnodes="${3:-1}" head_ip="${4:-}" mrank="${5:-0}"
    echo "Launching trainer (server=${url}, CUDA_VISIBLE_DEVICES=${cvd:-all}, trainer_nodes=${num_tnodes}, machine_rank=${mrank})..."
    # Empty cvd -> use all GPUs on the node (don't set the var; "" would hide all).
    local -a gpu_env=()
    [ -n "$cvd" ] && gpu_env=(env "CUDA_VISIBLE_DEVICES=$cvd")
    # Engage accelerate multi-node routing only when >1 trainer node; a single
    # trainer node keeps the original invocation (no --num_nodes) verbatim.
    local -a mn_args=()
    if [ "${num_tnodes}" -gt 1 ]; then
        mn_args=(--num_nodes "$num_tnodes" --head_node_ip "$head_ip" --machine_rank "$mrank")
    fi
    "${gpu_env[@]}" bash modules/Model-Optimizer/examples/speculative_decoding/launch_train.sh \
        "${SCRIPT_ARGS[@]}" \
        "${mn_args[@]}" \
        data.streaming_server_url="$url" \
        data.streaming_model_name="$HF_MODEL_CKPT" \
        data.streaming_shared_storage_path="$SERVE_SCRATCH" \
        training.dataloader_num_workers="${STREAMING_NUM_WORKERS:-4}" \
        || { echo "ERROR: trainer failed." >&2; return 1; }

    # Export only on the head trainer (machine_rank 0); non-head trainer nodes
    # would race writing the same export dir. The export reads the saved
    # checkpoint (training.output_dir), not the serve, so it is serve-independent.
    if [ "${mrank}" -ne 0 ]; then
        echo "machine_rank=${mrank}: training done, skipping export (head trainer handles it)."
        return 0
    fi

    # Export the trained draft to HF format. Derive the checkpoint dir from the
    # forwarded `training.output_dir=` dotlist (defaulting to the EAGLE
    # convention) so EAGLE and DFlash runs each export their own output_dir.
    # EXPORT_EXTRA_ARGS lets DFlash on a custom-modeling base (e.g. Kimi) pass
    # --trust_remote_code; empty by default so EAGLE behavior is unchanged.
    local out_dir
    out_dir=$(printf '%s\n' "${SCRIPT_ARGS[@]}" | sed -n 's/^training\.output_dir=//p' | tail -1)
    out_dir="${out_dir:-/scratchspace/eagle3}"
    python3 modules/Model-Optimizer/examples/speculative_decoding/scripts/export_hf_checkpoint.py \
        --model_path "$out_dir" \
        --export_path "${EXPORT_PATH:-/scratchspace/export}" \
        ${EXPORT_EXTRA_ARGS:-}
}

# ---------------------------------------------------------------------------
# Topology dispatch (see header): nemo_run runs this script once per node, so
# branch on $SLURM_NNODES / $SLURM_NODEID. Per-branch detail in section heads.
# ---------------------------------------------------------------------------
NNODES="${SLURM_NNODES:-1}"
NODEID="${SLURM_NODEID:-0}"

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

elif [ "$NODEID" -eq 0 ]; then
    # ----------------------- multi-node: serve node ------------------------
    SERVE_GPU_MEM_UTIL="${SERVE_GPU_MEM_UTIL:-0.9}"   # dedicated node -> use most of it
    SERVE_TP="${SERVE_TP:-$(gpus_on_node)}"            # default: all GPUs on this node
    rm -f "$SERVE_ADDR_FILE" "$DONE_FILE"              # clear stale rendezvous state

    trap cleanup INT TERM EXIT
    launch_vllm "0.0.0.0" "$SERVE_TP" ""
    wait_vllm_ready "http://127.0.0.1:${SERVE_PORT}" || exit 1

    # Publish a *routable* address for the trainer node. `hostname -I` can list a
    # link-local (169.254.x) or loopback address first, which is unreachable from
    # the other node, so resolve the Slurm node name and fall back to the first
    # non-link-local / non-loopback IP.
    serve_addr="${SERVE_ADVERTISE_IP:-}"
    if [ -z "$serve_addr" ]; then
        serve_addr=$(getent hosts "${SLURMD_NODENAME:-$(hostname)}" 2>/dev/null | awk '{print $1}' | head -1)
    fi
    if [ -z "$serve_addr" ]; then
        serve_addr=$(hostname -I | tr ' ' '\n' | grep -vE '^(127\.|169\.254\.|fe80:|::1)' | head -1)
    fi
    [ -z "$serve_addr" ] && serve_addr=$(hostname -I | awk '{print $1}')
    echo "$serve_addr" > "$SERVE_ADDR_FILE"
    echo "Serve node published ${serve_addr}; holding the server up until the trainer signals done..."
    while [ ! -f "$DONE_FILE" ]; do sleep 10; done
    echo "Training-done sentinel seen; serve node exiting (EXIT trap stops vllm)."

elif [ "$NODEID" -ge 1 ]; then
    # -------------------- multi-node: trainer node(s) ----------------------
    # Node 0 is the vllm serve; trainer nodes are SLURM nodes 1..NNODES-1, which
    # map to 0-based accelerate machine ranks (head trainer = SLURM node 1).
    NUM_TRAINER_NODES=$(( NNODES - 1 ))
    TRAINER_RANK=$(( NODEID - 1 ))
    TRAINER_ADDR_FILE="/scratchspace/.trainer_addr"

    # Only the head trainer (rank 0) signals the serve node to release on exit;
    # a non-head node exiting first must NOT tear the serve down early.
    if [ "$TRAINER_RANK" -eq 0 ]; then
        trap 'touch "$DONE_FILE" 2>/dev/null || true' EXIT
        rm -f "$TRAINER_ADDR_FILE"                 # clear stale rendezvous state
    fi

    echo "Trainer node (rank ${TRAINER_RANK}/${NUM_TRAINER_NODES}) waiting for the serve address..."
    for ((i = 0; i < SERVE_READY_TIMEOUT; i++)); do
        [ -f "$SERVE_ADDR_FILE" ] && break
        sleep 1
    done
    [ -f "$SERVE_ADDR_FILE" ] || { echo "ERROR: serve node never published its address." >&2; exit 1; }
    URL="http://$(cat "$SERVE_ADDR_FILE"):${SERVE_PORT}"
    wait_vllm_ready "$URL" || exit 1

    if [ "$NUM_TRAINER_NODES" -le 1 ]; then
        # Original 1-serve + 1-trainer topology: single-node DDP, unchanged.
        run_trainer_and_export "$URL" "" || exit 1
    else
        # >1 trainer node: head (rank 0) publishes its routable IP for accelerate's
        # c10d rendezvous (port 29500); all trainer nodes read it and join. Reuse
        # the serve node's IP-resolution logic (avoid link-local / loopback).
        if [ "$TRAINER_RANK" -eq 0 ]; then
            head_addr="${TRAINER_ADVERTISE_IP:-}"
            [ -z "$head_addr" ] && head_addr=$(getent hosts "${SLURMD_NODENAME:-$(hostname)}" 2>/dev/null | awk '{print $1}' | head -1)
            [ -z "$head_addr" ] && head_addr=$(hostname -I | tr ' ' '\n' | grep -vE '^(127\.|169\.254\.|fe80:|::1)' | head -1)
            [ -z "$head_addr" ] && head_addr=$(hostname -I | awk '{print $1}')
            echo "$head_addr" > "$TRAINER_ADDR_FILE"
            echo "Head trainer (rank 0) published ${head_addr} for c10d rendezvous."
        else
            echo "Trainer rank ${TRAINER_RANK} waiting for head-trainer address..."
            for ((i = 0; i < SERVE_READY_TIMEOUT; i++)); do
                [ -f "$TRAINER_ADDR_FILE" ] && break
                sleep 1
            done
            [ -f "$TRAINER_ADDR_FILE" ] || { echo "ERROR: head trainer never published its address." >&2; exit 1; }
        fi
        HEAD_IP=$(cat "$TRAINER_ADDR_FILE")
        run_trainer_and_export "$URL" "" "$NUM_TRAINER_NODES" "$HEAD_IP" "$TRAINER_RANK" || exit 1
    fi
fi

###################################################################################################

#exit_handler $0
