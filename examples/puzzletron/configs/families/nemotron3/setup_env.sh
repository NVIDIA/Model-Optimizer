#!/usr/bin/env bash
# Shared setup script for the Nemotron3 Puzzletron environment.
#
# Used by the Dockerfile (--deps before COPY, --modelopt after COPY) and
# can be run standalone to set up a bare-metal environment (no arguments).
#
# Usage:
#   ./setup_env.sh [--deps | --modelopt]
#     --deps      System packages, git clones, venv, pip deps — safe to Docker-cache
#     --modelopt  Install the local ModelOpt source only (run after --deps)
#     (no args)   Full setup — suitable for bare-metal
#
# Configurable via environment variables (all have defaults):
#   MODEL_OPT_ROOT         Path to the ModelOpt repo root
#                          Default: 5 directories above this script
#   VLLM_ROOT              Where to clone vLLM         (default: /workspace/vllm)
#   AUTOMODEL_ROOT         Where to clone Automodel    (default: /workspace/Automodel)
#   VIRTUAL_ENV            Python venv path            (default: /venv)
#   SKIP_APT               Set to 1 to skip apt-get   (default: 0)
#   FORCE_CUDA             Build CUDA exts without a live GPU (default: 1)
#   TORCH_CUDA_ARCH_LIST   Architectures to compile for
#                          (default: "8.0;8.6;9.0;10.0")

set -euo pipefail

MODE="${1:-all}"

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# examples/puzzletron/configs/families/nemotron3 → 5 levels up to repo root
MODEL_OPT_ROOT="${MODEL_OPT_ROOT:-$(cd "${SCRIPT_DIR}/../../../../.." && pwd)}"
VLLM_ROOT="${VLLM_ROOT:-/workspace/vllm}"
AUTOMODEL_ROOT="${AUTOMODEL_ROOT:-/workspace/Automodel}"
VIRTUAL_ENV="${VIRTUAL_ENV:-/venv}"

# ── Build flags ────────────────────────────────────────────────────────────────
SKIP_APT="${SKIP_APT:-0}"
export FORCE_CUDA="${FORCE_CUDA:-1}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;9.0;10.0}"

activate_venv() {
    export VIRTUAL_ENV
    export PATH="${VIRTUAL_ENV}/bin:${PATH}"
}

# ══════════════════════════════════════════════════════════════════════════════
# DEPS phase — nothing here depends on the local ModelOpt source.
# In Docker this runs before `COPY .` so it stays cached across source changes.
# ══════════════════════════════════════════════════════════════════════════════
install_deps() {
    if [[ "${SKIP_APT}" != "1" ]]; then
        apt-get update
        apt-get install -y build-essential cmake curl git ninja-build \
            python3 python3-dev python3-pip python3-venv
    fi

    # Clone external repos (idempotent)
    [[ -d "${VLLM_ROOT}" ]] || \
        git clone --branch feature/add_anymodel_to_vllm --single-branch \
            https://github.com/Separius/vllm.git "${VLLM_ROOT}"
    [[ -d "${AUTOMODEL_ROOT}" ]] || \
        git clone --branch puzzletron --single-branch \
            https://github.com/Separius/Automodel.git "${AUTOMODEL_ROOT}"

    [[ -d "${VIRTUAL_ENV}" ]] || python3 -m venv "${VIRTUAL_ENV}"
    activate_venv

    python -m pip install --upgrade pip \
        "setuptools>=80,<81" "setuptools-scm>=8" setuptools-rust wheel \
        "packaging>=24.2" "cmake>=3.26.1" ninja jinja2 hydra-core immutabledict

    python -m pip install \
        torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
        --index-url https://download.pytorch.org/whl/cu129

    VLLM_PRECOMPILED_WHEEL_VARIANT=cu129 \
    python -m pip install --no-build-isolation -e "${VLLM_ROOT}"

    python -m pip install -e "${AUTOMODEL_ROOT}"
    python -m pip install aiperf

    # CUDA extension packages placed here (before COPY .) so they stay cached
    # across ModelOpt source changes in Docker builds.
    python -m pip install --no-build-isolation \
        "git+https://github.com/fanshiqing/grouped_gemm@v1.1.4"
    python -m pip install "mamba-ssm[causal-conv1d]" --no-build-isolation
    python -m pip install "flash-linear-attention[cuda]"
}

# ══════════════════════════════════════════════════════════════════════════════
# MODELOPT phase — installs the local ModelOpt source.
# In Docker this runs after `COPY .` and re-runs on every source change.
# ══════════════════════════════════════════════════════════════════════════════
install_modelopt() {
    activate_venv
    python -m pip install -e "${MODEL_OPT_ROOT}[hf]"
}

# ── Dispatch ───────────────────────────────────────────────────────────────────
case "${MODE}" in
    --deps)
        install_deps
        ;;
    --modelopt)
        install_modelopt
        ;;
    all)
        install_deps
        install_modelopt
        echo ""
        echo "Setup complete. Activate your environment with:"
        echo "  source ${VIRTUAL_ENV}/bin/activate"
        ;;
    *)
        echo "Unknown argument: ${MODE}" >&2
        echo "Usage: $0 [--deps | --modelopt]" >&2
        exit 1
        ;;
esac
