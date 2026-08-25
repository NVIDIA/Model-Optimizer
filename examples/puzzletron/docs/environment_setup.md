# Environment setup

Puzzletron uses two environments:

- a lightweight control environment for the setup wizard and orchestrator;
- a GPU worker environment for ModelOpt, the patched vLLM fork, AutoModel, and
  AIPerf.

The runner file connects them. `runner.execution_contract.venv` selects the
worker virtual environment, and `runner.execution_contract.container` selects
an optional Slurm container.

## Control environment

The setup wizard and orchestrator do not import PyTorch or initialize CUDA.
Create one environment for both:

```bash
python3 -m venv .venv-puzzletron-control
source .venv-puzzletron-control/bin/activate
python -m pip install \
  -r examples/puzzletron/requirements-setup.txt \
  -r examples/puzzletron/requirements-orchestrator.txt
```

A Slurm login node also needs `sbatch`, `squeue`, and `sacct`. It does not need
ModelOpt, CUDA, the worker container, or the worker virtual environment.

## Worker environment

Use one Python environment for ModelOpt, patched vLLM, AutoModel, and official
AIPerf. Install PyTorch first and build every CUDA extension against that
installation. Mixing PyTorch or CUDA builds can cause import failures or
incorrect GPU execution.

### Choose a container or host environment

This CUDA image provides a reproducible bootstrap:

```text
nvcr.io/nvidia/cuda:12.9.2-cudnn-devel-ubuntu24.04
```

The image is an example, not a required runner image. Slurm campaigns can set
`runner.execution_contract.container` to an image or path accepted by the
site, or omit it to execute directly in the worker environment. Bare-metal
runners use the host environment selected by `runner.execution_contract.venv`.

The commands below assume a container. For bare metal, skip the Docker,
`/workspace`, and `apt-get` steps. Install equivalent Python and build tools
through the host-environment tooling and adapt the paths.

```bash
export PUZZLETRON_WORKSPACE=/absolute/path/to/workspace
docker run --gpus all --ipc=host --rm -it \
  -v "${PUZZLETRON_WORKSPACE}:/workspace" \
  -w /workspace \
  nvcr.io/nvidia/cuda:12.9.2-cudnn-devel-ubuntu24.04 bash
```

Inside the container, install Python and the build tools used by editable
packages and optional CUDA extensions:

```bash
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  build-essential cmake git ninja-build \
  python3 python3-dev python3-pip python3-venv
```

### Clone the tracked forks

Keep ModelOpt and the two Puzzletron forks as siblings. The machine-readable
[CI environment](../ci_environment.json) records the shared compatibility pins.

```bash
export MODEL_OPT_ROOT=/workspace/modelopt
export VLLM_ROOT=/workspace/vllm
export AUTOMODEL_ROOT=/workspace/Automodel
export PUZZLETRON_CI_ENVIRONMENT="${MODEL_OPT_ROOT}/examples/puzzletron/ci_environment.json"
export AUTOMODEL_REF="$(python3 -c \
  'import json, sys; print(json.load(open(sys.argv[1]))["nemo_automodel"]["commit"])' \
  "${PUZZLETRON_CI_ENVIRONMENT}")"

git clone --branch feature/add_anymodel_to_vllm --single-branch \
  https://github.com/Separius/vllm.git "${VLLM_ROOT}"
git clone --branch puzzletron --single-branch \
  https://github.com/Separius/Automodel.git "${AUTOMODEL_ROOT}"
git -C "${AUTOMODEL_ROOT}" checkout --detach "${AUTOMODEL_REF}"
```

```text
/workspace/
├── modelopt/
├── vllm/
└── Automodel/
```

### Install runtime packages

The patched vLLM branch uses the PyTorch version recorded in the CI environment
with CUDA 12.9. Install that combination before compiling CUDA code:

```bash
python3 -m venv /workspace/.venv
source /workspace/.venv/bin/activate

export PUZZLETRON_TORCH_VERSION="$(python -c \
  'import json, sys; print(json.load(open(sys.argv[1]))["torch"])' \
  "${PUZZLETRON_CI_ENVIRONMENT}")"
export PUZZLETRON_TORCHVISION_VERSION="$(python -c \
  'import json, sys; print(json.load(open(sys.argv[1]))["torchvision"])' \
  "${PUZZLETRON_CI_ENVIRONMENT}")"
export PUZZLETRON_TRANSFORMERS_VERSION="$(python -c \
  'import json, sys; print(json.load(open(sys.argv[1]))["transformers"])' \
  "${PUZZLETRON_CI_ENVIRONMENT}")"

python -m pip install --upgrade \
  pip "setuptools>=80,<81" "setuptools-scm>=8" setuptools-rust \
  wheel "packaging>=24.2" "cmake>=3.26.1" ninja jinja2

python -m pip install \
  "torch==${PUZZLETRON_TORCH_VERSION}" \
  "torchvision==${PUZZLETRON_TORCHVISION_VERSION}" \
  "torchaudio==${PUZZLETRON_TORCH_VERSION}" \
  --index-url https://download.pytorch.org/whl/cu129

VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_VARIANT=cu129 \
  python -m pip install --no-build-isolation -e "${VLLM_ROOT}"

python -m pip install -e "${AUTOMODEL_ROOT}"
python -m pip install aiperf
python -m pip install -e "${MODEL_OPT_ROOT}[hf,puzzletron]"
python -m pip install "transformers==${PUZZLETRON_TRANSFORMERS_VERSION}"
python -m pip install -r "${MODEL_OPT_ROOT}/examples/puzzletron/requirements.txt"
```

Do not add `--no-deps`; these packages need their declared Python dependencies.
`--no-build-isolation` makes compiled extensions use the active PyTorch
installation. It does not disable dependency installation.

Install only the kernels required by the target architecture:

```bash
# Mixture of experts
python -m pip install --no-build-isolation \
  "git+https://github.com/fanshiqing/grouped_gemm@v1.1.4"

# Mamba
python -m pip install "mamba-ssm[causal-conv1d]" --no-build-isolation

# Linear attention
python -m pip install "flash-linear-attention[cuda]"
```

## Verify the worker environment

Run these checks inside the same container and virtual environment used by
Puzzletron jobs:

```bash
test "$(git -C "${VLLM_ROOT}" remote get-url origin)" = \
  "https://github.com/Separius/vllm.git"
test "$(git -C "${VLLM_ROOT}" branch --show-current)" = \
  "feature/add_anymodel_to_vllm"
test "$(git -C "${AUTOMODEL_ROOT}" remote get-url origin)" = \
  "https://github.com/Separius/Automodel.git"
test "$(git -C "${AUTOMODEL_ROOT}" rev-parse HEAD)" = "${AUTOMODEL_REF}"

git -C "${MODEL_OPT_ROOT}" rev-parse HEAD
git -C "${VLLM_ROOT}" rev-parse HEAD
git -C "${AUTOMODEL_ROOT}" rev-parse HEAD
```

```bash
python - <<'PY'
import importlib.metadata as metadata
import json
import os

from packaging.version import Version

import aiperf
import lmms_eval
import modelopt
import nemo_automodel
import torch
import transformers
import vllm

with open(os.environ["PUZZLETRON_CI_ENVIRONMENT"], encoding="utf-8") as stream:
    ci_environment = json.load(stream)

for package in (
    "torch",
    "vllm",
    "nemo-automodel",
    "aiperf",
    "lmms-eval",
    "nvidia-modelopt",
):
    print(package, metadata.version(package))

print("torch CUDA", torch.version.cuda)
print("CUDA available", torch.cuda.is_available())
print("modelopt", modelopt.__file__)
print("vllm", vllm.__file__)

assert Version(torch.__version__).release == Version(ci_environment["torch"]).release
assert Version(metadata.version("torchvision")).release == Version(
    ci_environment["torchvision"]
).release
assert transformers.__version__ == ci_environment["transformers"]
assert metadata.version("lmms-eval") == ci_environment["lmms_eval"]
assert Version(metadata.version("nemo-automodel")).base_version == (
    ci_environment["nemo_automodel"]["base_version"]
)
assert torch.version.cuda == "12.9"
assert torch.cuda.is_available()
PY

python -m pip check
```

Record the three source revisions and verification output with the campaign.
Repeat verification after pulling either fork or rebuilding a CUDA extension.
