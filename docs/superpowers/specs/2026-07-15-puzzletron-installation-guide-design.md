# Puzzletron Installation Guide Design

## Goal

Replace the generic installation block in `examples/puzzletron/README.md` with
a reproducible CUDA 12.9 setup that keeps PyTorch, vLLM, AutoModel, ModelOpt,
and optional architecture-specific CUDA extensions ABI-compatible.

## Environment baseline

- Recommend `nvcr.io/nvidia/cuda:12.9.2-cudnn-devel-ubuntu24.04` as the clean
  and supported starting image.
- Use Python 3.12 in a virtual environment.
- Treat the patched vLLM checkout as the PyTorch compatibility authority. Its
  current metadata pins PyTorch 2.11.0 and its default binary variant uses CUDA
  12.9.
- Clone vLLM from `https://github.com/Separius/vllm.git` and track the
  `feature/add_anymodel_to_vllm` branch. Do not pin the installation guide to a
  single commit, but record the resolved commit for each production run.
- Clone AutoModel from `https://github.com/Separius/Automodel.git` and track the
  `puzzletron` branch under the same record-without-pinning policy.

## Installation order

1. Install the Ubuntu compiler, Python, Git, and virtual-environment tools.
2. Clone the AnyModel vLLM fork from the `feature/add_anymodel_to_vllm` branch
   and the AutoModel fork from the `puzzletron` branch, both with
   `--single-branch`.
3. Create and activate a Python 3.12 virtual environment.
4. Install Python build tooling and the CUDA 12.9 builds of PyTorch 2.11.0,
   torchvision 0.26.0, and torchaudio 2.11.0.
5. Install the patched vLLM checkout in editable mode, using its precompiled
   extension path so ordinary Python changes do not trigger a full CUDA build.
6. Install the tracked AutoModel fork and ModelOpt from sibling checkouts in
   editable mode, and install the published AIPerf package with
   `pip install aiperf`.
7. Install only the optional kernels required by the target architecture:
   grouped GEMM for MoE, Mamba plus causal-conv1d for Mamba models, and
   flash-linear-attention for linear-attention models.
8. Verify the vLLM and AutoModel remotes and active branches, record their
   resolved commits, and check package versions, import locations, CUDA
   availability, and dependency consistency.

## Dependency flags

- Do not use blanket `--no-deps` for vLLM, AutoModel, ModelOpt, AIPerf, or
  flash-linear-attention; their Python runtime dependencies are required.
- Use `--no-build-isolation` for CUDA extensions that inspect or compile against
  the active PyTorch installation, including grouped GEMM and Mamba. This keeps
  compilation tied to PyTorch 2.11.0+cu129 instead of a temporary build
  environment.
- Explain that a full vLLM CUDA/C++ source build is a separate workflow; the
  default guide targets Python-level development with matching precompiled vLLM
  extensions.

## README scope

Only the Installation section will change. The rest of the Puzzletron workflow
documentation and launch commands remain unchanged. The guide will use a
generic sibling layout containing `modelopt`, `vllm`, and `Automodel`; AIPerf
will no longer be described as a required sibling checkout. The vLLM entry will
be created from the explicit Separius fork and AnyModel feature branch, and the
AutoModel entry from the Separius fork's `puzzletron` branch.

## Validation

- Check Markdown formatting and line length through the repository pre-commit
  hook for the changed README.
- Inspect the final diff for copy-paste correctness.
- The guide will include runtime commands that print Python, Torch, CUDA, vLLM,
  AutoModel, AIPerf, and ModelOpt versions or import paths, followed by
  `python -m pip check`.
- The guide will verify that vLLM's `origin` points to the Separius fork, its
  active branch is `feature/add_anymodel_to_vllm`, and its resolved commit is
  captured without constraining future installs to that commit.
- The same checks will verify AutoModel's Separius `origin`, active `puzzletron`
  branch, and resolved commit.
