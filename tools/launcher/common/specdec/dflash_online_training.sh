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

# DFlash online training script for the ModelOpt Launcher.
# Trains a DFlash draft model using accelerate launch + main.py --config.
#
# All training config comes from the YAML recipe (--config) and OmegaConf overrides.
# All args are passed directly to main.py (--config + key=value overrides).
#
# Multi-node env vars (set by Slurm or user):
#   NUM_NODES       — number of nodes (default: 1)
#   HEAD_NODE_IP    — head node IP (auto-detected if not set)
#
# Usage from YAML:
#   script: common/dflash/online_training.sh
#   args:
#     - --config modules/Model-Optimizer/modelopt_recipes/general/speculative_decoding/dflash.yaml
#     - model.model_name_or_path=/hf-local/Qwen/Qwen3-8B
#     - data.data_path=/path/to/data.jsonl
#     - training.output_dir=/scratchspace/dflash
#   environment:
#     - NUM_NODES: "8"

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source ${SCRIPT_DIR}/../service_utils.sh

OUTPUT_DIR=$(python3 -c "
import sys
for arg in sys.argv[1:]:
    if arg.startswith('training.output_dir='):
        print(arg.split('=', 1)[1])
        break
" "$@")
# A verdict left in a reused output_dir by an earlier run must never be read as this run's,
# so remove it before anything below can stop the script.
if [ "${SLURM_PROCID:-0}" = "0" ] && [ -n "$OUTPUT_DIR" ]; then
    rm -f "${OUTPUT_DIR}/regression_check.json"
fi

# Checked before any work: a typo such as "true" must not cost a full training run.
case "${REGRESSION_STRICT:-0}" in
    0|1) ;;
    *) echo "ERROR: REGRESSION_STRICT must be 0 or 1, got '${REGRESSION_STRICT}'" >&2; exit 1 ;;
esac

pip install -r modules/Model-Optimizer/examples/speculative_decoding/requirements.txt
pip install huggingface-hub>=1.2.1
export PATH=$PATH:/workspace/.local/bin

# Some trust_remote_code MoE models pin an older transformers (e.g. MiniMax-M2.7
# needs 4.57.x; its modeling code is incompatible with the 5.x the recipe defaults
# pull in). Override here when the model needs it.
if [ -n "${OVERRIDE_TRANSFORMERS:-}" ]; then
    pip install "transformers==${OVERRIDE_TRANSFORMERS}"
fi

###################################################################################################

trap 'error_handler $0 $LINENO' ERR
# Without this, the ERR trap sets FAIL_EXIT=1 but the script still exits 0,
# masking training failures (e.g. CUDA OOM) as Slurm SUCCEEDED.
trap 'exit_handler' EXIT

# Auto-detect head node IP for multi-node training
NUM_NODES=${NUM_NODES:-1}
if [ -z "$HEAD_NODE_IP" ] && [[ "$NUM_NODES" != "1" ]]; then
    HEAD_NODE_IP=$(scontrol show hostnames "$SLURM_JOB_NODELIST" 2>/dev/null | head -1)
    HEAD_NODE_IP=${HEAD_NODE_IP:-$SLURM_LAUNCH_NODE_IPADDR}
    if [ -z "$HEAD_NODE_IP" ] && [ -n "$SLURM_JOB_NODELIST" ]; then
        HEAD_NODE_IP=$(python3 -c "
import socket, re, os
nl = os.environ.get('SLURM_JOB_NODELIST', '')
m = re.match(r'([a-zA-Z0-9-]+?)(?:\[(\d+))?', nl)
if m:
    host = m.group(1) + (m.group(2) or '')
    try:
        print(socket.gethostbyname(host))
    except:
        print(host)
" 2>/dev/null)
    fi
    if [ -z "$HEAD_NODE_IP" ] && [ "${SLURM_PROCID:-0}" = "0" ]; then
        HEAD_NODE_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
    fi
    export HEAD_NODE_IP
    echo "Auto-detected HEAD_NODE_IP: ${HEAD_NODE_IP}"
fi

# Build accelerate launch command
MAIN_PY=modules/Model-Optimizer/examples/speculative_decoding/main.py

if [[ "$NUM_NODES" != "1" ]]; then
    if [ -z "$HEAD_NODE_IP" ]; then
        echo "ERROR: HEAD_NODE_IP is empty. Cannot launch multi-node training."
        exit 1
    fi
    GPU_PER_NODE=${GPU_PER_NODE:-$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)}
    TOTAL_GPU=$((NUM_NODES * GPU_PER_NODE))
    echo "Total GPUs: $TOTAL_GPU (NUM_NODES: $NUM_NODES, GPU_PER_NODE: $GPU_PER_NODE)"
    MULTI_NODE_ARGS="--num_processes $TOTAL_GPU \
                     --num_machines $NUM_NODES \
                     --machine_rank $SLURM_PROCID \
                     --rdzv_backend c10d \
                     --main_process_ip $HEAD_NODE_IP \
                     --main_process_port 29500"
else
    TOTAL_GPU=$(python3 -c "import torch; print(torch.cuda.device_count())")
    echo "Total GPUs: $TOTAL_GPU (single node)"
    MULTI_NODE_ARGS=""
fi

export TOKENIZERS_PARALLELISM=False

# ACCELERATE_CONFIG selects an explicit accelerate config file (e.g. an FSDP2 YAML).
# Required for trust_remote_code MoE models that need FSDP2 via accelerate config
# rather than transformers-native ParallelismConfig (MiniMax-M2.7 on transformers 4.57.x).
ACCEL_CONFIG_ARGS=""
if [ -n "${ACCELERATE_CONFIG:-}" ] && [ -f "${ACCELERATE_CONFIG}" ]; then
    ACCEL_CONFIG_ARGS="--config_file ${ACCELERATE_CONFIG}"
    echo "Using accelerate config: ${ACCELERATE_CONFIG}"
fi

# Config creation precedes serialization, so existence alone does not prove completion.
export_complete() {
    python3 - "$1/config.json" <<'PY'
import json
import sys

try:
    with open(sys.argv[1]) as f:
        config = json.load(f)
except (OSError, ValueError):
    sys.exit(1)

sys.exit(0 if isinstance(config, dict) and config else 1)
PY
}

recover_export() {
    local destination=$1 backup="$1.old"

    [ -e "$backup" ] || return 0
    if [ -e "$destination" ]; then
        # Only a complete publication supersedes the backup.
        export_complete "$destination" || return 1
        rm -rf "$backup" || return 1
    else
        mv -T "$backup" "$destination" || return 1
    fi
}

export_draft() {
    local source=$1 destination=$2
    local scratch="$2.tmp" backup="$2.old"

    # Callers use ||, which suppresses ERR handling inside this function.
    recover_export "$destination" || return 1
    rm -rf "$scratch" || return 1
    echo "=== Exporting: ${source} → ${destination} ==="
    CUDA_VISIBLE_DEVICES=0 python3 modules/Model-Optimizer/examples/speculative_decoding/scripts/export_hf_checkpoint.py \
        --model_path "$source" \
        --export_path "$scratch" \
        --trust_remote_code || return 1
    if ! export_complete "$scratch"; then
        echo "ERROR: export produced an invalid config: ${scratch}" >&2
        return 1
    fi
    if [ -e "$destination" ]; then
        mv -T "$destination" "$backup" || return 1
    fi
    if ! mv -T "$scratch" "$destination"; then
        recover_export "$destination" || return 1
        return 1
    fi
    rm -rf "$backup" || return 1
}

set -x
start_time=$(date +%s)
# The ERR trap only records a failure and lets the script go on, so test the status here:
# a crashed run must not export or write a regression verdict from whatever checkpoints
# and trainer_state.json it, or an earlier run, left behind.
if accelerate launch $ACCEL_CONFIG_ARGS --mixed_precision "${MIXED_PRECISION:-bf16}" $MULTI_NODE_ARGS $MAIN_PY "$@"; then
    TRAIN_STATUS=0
else
    # The if suppresses the ERR trap, so report as it would; $? is still the launch status.
    error_handler "$0" "$LINENO"
    TRAIN_STATUS=1
fi
echo "Training time: $(( $(date +%s) - start_time )) seconds"
set +x

if [ "$TRAIN_STATUS" != 0 ]; then
    echo "ERROR: training failed; skipping export and regression check." >&2
# Export last checkpoint to deployment format (rank 0 only, single GPU)
elif [ "${SLURM_PROCID:-0}" = "0" ]; then
    if [ -n "$OUTPUT_DIR" ]; then
        # Export all checkpoints that don't already have an exported version
        EXPORTED=0
        for BACKUP in "${OUTPUT_DIR}"/exported-checkpoint-*.old; do
            [ -d "$BACKUP" ] || continue
            recover_export "${BACKUP%.old}" || exit 1
        done
        for CKPT in "${OUTPUT_DIR}"/checkpoint-*; do
            [ -d "$CKPT" ] || continue
            STEP=$(basename "$CKPT" | sed 's/checkpoint-//')
            EXPORT_DIR="${OUTPUT_DIR}/exported-checkpoint-${STEP}"
            if export_complete "$EXPORT_DIR"; then
                echo "Already exported: ${EXPORT_DIR}"
                continue
            fi
            if compgen -G "${CKPT}/pytorch_model_fsdp_*" >/dev/null; then
                echo "WARNING: cannot re-export FSDP shards in ${CKPT}; checking the required export below." >&2
                continue
            fi
            export_draft "$CKPT" "$EXPORT_DIR" || exit 1
            EXPORTED=$((EXPORTED + 1))
        done
        LATEST_CKPT=$(latest_checkpoint "$OUTPUT_DIR" checkpoint-)
        LATEST_SAVE="${LATEST_CKPT}/trainer_state.json"
        if [ ! -f "$LATEST_SAVE" ]; then
            LATEST_SAVE="$LATEST_CKPT"
        fi

        # Equal timestamps cannot establish ordering; only a newer checkpoint makes the root stale.
        FINAL_IS_CURRENT=0
        if [ -f "${OUTPUT_DIR}/modelopt_state.pth" ] && {
            [ -z "$LATEST_CKPT" ] ||
            ! [ "$LATEST_SAVE" -nt "${OUTPUT_DIR}/modelopt_state.pth" ]
        }; then
            FINAL_IS_CURRENT=1
        fi

        # Skipped sharded exports are safe only if the current serving artifact is available.
        if [ "$FINAL_IS_CURRENT" != 1 ] && [ -n "$LATEST_CKPT" ] &&
            ! export_complete "${OUTPUT_DIR}/exported-${LATEST_CKPT##*/}"; then
            echo "ERROR: newest checkpoint has no complete export: ${LATEST_CKPT}" >&2
            exit 1
        fi

        FINAL_EXPORT="${OUTPUT_DIR}/exported-checkpoint-final"
        if [ "$FINAL_IS_CURRENT" = 1 ]; then
            if ! export_complete "$FINAL_EXPORT" ||
                [ "${OUTPUT_DIR}/modelopt_state.pth" -nt "${FINAL_EXPORT}/config.json" ]; then
                export_draft "$OUTPUT_DIR" "$FINAL_EXPORT" || exit 1
                EXPORTED=$((EXPORTED + 1))
            fi
        elif [ -e "$FINAL_EXPORT" ]; then
            # Retire an obsolete final only once its replacement is available.
            if [ -z "$LATEST_CKPT" ] ||
                ! export_complete "${OUTPUT_DIR}/exported-${LATEST_CKPT##*/}"; then
                echo "ERROR: obsolete final export has no complete newest-step replacement." >&2
                exit 1
            fi
            echo "Removing obsolete final export: ${FINAL_EXPORT}"
            rm -rf "$FINAL_EXPORT" || exit 1
        fi
        if [ "$EXPORTED" -eq 0 ]; then
            echo "No new checkpoints to export in ${OUTPUT_DIR}"
        fi
    fi

    # Regression check (uses env vars MAX_FINAL_LOSS, MIN_FINAL_ACC, etc.). The verdict
    # lands next to the exports for downstream gates. A regression only warns unless
    # REGRESSION_STRICT=1, which fails the task (via the ERR trap) so Slurm reports it.
    if [ -n "$OUTPUT_DIR" ]; then
        REGRESSION_ARGS=("${OUTPUT_DIR}" --json "${OUTPUT_DIR}/regression_check.json")
        if [ "${REGRESSION_STRICT:-0}" = 1 ]; then
            python3 common/check_regression.py "${REGRESSION_ARGS[@]}"
        else
            python3 common/check_regression.py "${REGRESSION_ARGS[@]}" || true
        fi
    fi
fi
