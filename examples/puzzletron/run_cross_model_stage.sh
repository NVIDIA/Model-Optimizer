#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ROOT=${PUZZLETRON_ROOT:-${SCRIPT_ROOT}}
CAMPAIGN_ROOT=${PUZZLETRON_CAMPAIGN_ROOT:-${ROOT}/puzzle_runs/clean/acceptance/cross-model-stage-matrix}
MODEL_ID=${1:?usage: run_cross_model_stage.sh MODEL_ID STAGE}
STAGE=${2:?usage: run_cross_model_stage.sh MODEL_ID STAGE}
EXTRA_ARGS=("${@:3}")
CFG=${CAMPAIGN_ROOT}/configs/${MODEL_ID}.yaml
MODEL_ROOT=${CAMPAIGN_ROOT}/models/${MODEL_ID}
MAIN=${ROOT}/examples/puzzletron/main.py
PUZZLETRON_VENV=${PUZZLETRON_VENV:-"${ROOT}/.venv_new"}

if [[ "${PUZZLETRON_ENV_READY:-0}" != 1 ]]; then
  if [[ -n "${PUZZLETRON_SETUP_ENV:-}" ]]; then
    source "${PUZZLETRON_SETUP_ENV}"
  fi
  source "${PUZZLETRON_VENV}/bin/activate"
fi
export PUZZLETRON_SETUP_ENV=${PUZZLETRON_SETUP_ENV:-}
# The NeMo container may expose a different CUDA-major PyTorch under
# /usr/local than the active venv.  Native extensions such as Transformer
# Engine have unversioned libtorch dependencies, so resolve those from the
# active venv first and let PyTorch's RPATH select its matching CUDA runtime.
PYTHON_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
VENV_TORCH_LIB="${VIRTUAL_ENV}/lib/python${PYTHON_VERSION}/site-packages/torch/lib"
if [[ -d "${VENV_TORCH_LIB}" ]]; then
  export LD_LIBRARY_PATH="${VENV_TORCH_LIB}:${LD_LIBRARY_PATH:-}"
fi
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PUZZLETRON_TRITON_CACHE_NAMESPACE=cross_model_stage_matrix
if [[ -z "${AIPERF_EXECUTABLE:-}" && -x "${ROOT}/../aiperf/.venv/bin/aiperf" ]]; then
  export AIPERF_EXECUTABLE="${ROOT}/../aiperf/.venv/bin/aiperf"
fi

# Some model-local compiled kernels cannot safely be reused after a stage
# repeatedly materializes different tensor shapes.  The descriptor writes the
# affected stages into the generated config; the launcher only implements the
# generic policy and contains no model-family special cases.
if python - "${CFG}" "${STAGE}" <<'PY'
import sys
import yaml

config = yaml.safe_load(open(sys.argv[1]))
disabled = config.get("execution", {}).get("torch_compile_disabled_stages", [])
raise SystemExit(0 if sys.argv[2] in disabled else 1)
PY
then
  export TORCH_COMPILE_DISABLE=1
fi
if [[ "${STAGE}" == convert ]]; then
  # hf-xet can leave multi-GB shards and HTTPS sockets permanently in
  # CLOSE-WAIT on compute nodes.  The standard Hub HTTP path is slower but
  # resumable and has proven reliable for the same pinned snapshots.
  export HF_HUB_DISABLE_XET=1
fi

mkdir -p "${MODEL_ROOT}/logs" "${MODEL_ROOT}/manifests/executions"
LOG=${MODEL_ROOT}/logs/${STAGE}_${SLURM_JOB_ID:-local}_node${SLURM_NODEID:-0}.log
exec > >(tee -a "${LOG}") 2>&1

python - "${CFG}" "${MODEL_ROOT}" "${MODEL_ID}" "${STAGE}" "${LOG}" "${EXTRA_ARGS[@]}" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

config, root, model_id, stage, log = map(Path, sys.argv[1:6])
extra_args = sys.argv[6:]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None

def git(repo, *args):
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None

repo = Path.cwd()
automodel = repo.parent / "Automodel"
payload = {
    "version": 1,
    "model_id": model_id.name,
    "stage": stage.name,
    "config": {"path": str(config), "sha256": sha(config)},
    "command": [
        "examples/puzzletron/run_cross_model_stage.sh", model_id.name, stage.name, *extra_args
    ],
    "environment": {
        key: os.environ.get(key)
        for key in (
            "SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_NNODES", "SLURM_NTASKS",
            "SLURM_GPUS_ON_NODE", "SLURM_NODEID", "CUDA_VISIBLE_DEVICES",
            "TRITON_CACHE_DIR", "TILELANG_CACHE_DIR",
        )
    },
    "revisions": {
        "modelopt_head": git(repo, "rev-parse", "HEAD"),
        "modelopt_dirty": bool(git(repo, "status", "--porcelain")),
        "automodel_head": git(automodel, "rev-parse", "HEAD"),
        "automodel_dirty": bool(git(automodel, "status", "--porcelain")),
        "setup_env_sha256": (
            sha(Path(os.environ["PUZZLETRON_SETUP_ENV"]))
            if os.environ.get("PUZZLETRON_SETUP_ENV")
            else None
        ),
    },
    "log": str(log),
}
job = os.environ.get("SLURM_JOB_ID", "local")
node = os.environ.get("SLURM_NODEID", "0")
path = root / "manifests" / "executions" / f"{stage.name}_{job}_node{node}.json"
temporary = path.with_suffix(".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
PY

master_addr() {
  if [[ -n "${MASTER_ADDR:-}" ]]; then
    echo "${MASTER_ADDR}"
  else
    scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1
  fi
}

distributed_stage() {
  case "$1" in
    activation|sort|sort_equivalence|activation_diagnostic|bypass|bypass_overfit|scoring|depth|evaluation|distillation) return 0 ;;
    *) return 1 ;;
  esac
}

if distributed_stage "${STAGE}"; then
  NNODES=${CAMPAIGN_NNODES:?CAMPAIGN_NNODES is required}
  NPROC=${CAMPAIGN_NPROC_PER_NODE:?CAMPAIGN_NPROC_PER_NODE is required}
  NODE_RANK=${SLURM_NODEID:-0}
  DEFAULT_PORT=$((29500 + ${SLURM_JOB_ID:-0} % 1000))
  # A stage may be retried inside one long-lived interactive allocation.  Let
  # the caller select a fresh rendezvous port so a cancelled torchrun cannot
  # poison the next attempt through the job-derived default.
  PORT=${MASTER_PORT:-${DEFAULT_PORT}}
  torchrun \
    --nnodes="${NNODES}" \
    --nproc-per-node="${NPROC}" \
    --node-rank="${NODE_RANK}" \
    --rdzv-backend=c10d \
    --rdzv-endpoint="$(master_addr):${PORT}" \
    --max-restarts=0 \
    "${MAIN}" --config "${CFG}" --stage "${STAGE}" \
    "${EXTRA_ARGS[@]}"
else
  if [[ "${SLURM_NODEID:-0}" == 0 ]]; then
    python "${MAIN}" --config "${CFG}" --stage "${STAGE}" "${EXTRA_ARGS[@]}"
  fi
fi
