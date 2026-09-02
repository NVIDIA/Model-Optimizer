# Environment setup

Puzzletron uses two environments:

- one lightweight local Python environment for the setup wizard and campaign
  commands; and
- a GPU worker environment for ModelOpt, the patched vLLM fork, AutoModel, and
  AIPerf.

The runner file connects them. `runner.execution_contract.venv` selects the
worker virtual environment, and `runner.execution_contract.container` selects
an optional Slurm container.

## Local Puzzletron environment

The setup wizard and `orchestrate.py` do not import PyTorch or initialize CUDA.
Create one environment for both:

```bash
python3 -m venv .venv-puzzletron
source .venv-puzzletron/bin/activate
python -m pip install -r examples/puzzletron/requirements-setup.txt
```

Only one local virtual environment is needed for a first campaign.
`requirements-setup.txt` includes the packages required to generate, launch,
and monitor a campaign.
`requirements-orchestrator.txt` is the smaller subset for a machine that only
launches or monitors an existing campaign. Neither set requires PyTorch.

A Slurm login node also needs `sbatch`, `squeue`, and `sacct`. It does not need
ModelOpt, CUDA, the worker container, or the worker virtual environment.

## Worker environment

The repository [`Dockerfile`](../Dockerfile) builds the worker image. It
installs ModelOpt, the pinned vLLM and AutoModel sources, AIPerf, LMMS-Eval,
the required CUDA extensions, and the teacher-evaluation resources. Its
LMMS-Eval install includes the pinned native Qwen 3.5 image and video backend.
Do not maintain a second set of worker installation commands or evaluator
overlays outside the Dockerfile.

Build the Linux amd64 image from the repository root by following the
[image build and validation guide](worker_image.md). That guide provides the
build command and the revision-specific image tag.

The amd64 platform is required because the current CUDA extension set and
Linux `eva-decord 0.6.1` dependency do not have a validated ARM build path.

Using the `image` variable from that guide, run the image locally with GPU
access:

```bash
docker run --gpus all --ipc=host --rm -it \
  "${image}"
```

Inside the image, the runner contract is:

- `repository: /opt/puzzletron/src/modelopt`
- `venv: /venv`
- `container: <registry reference or cluster-readable image path>`

Add site-specific data, model, cache, and result mounts through
`container_mounts`. A registry upload or conversion to a cluster container
format changes how the image is delivered, not how its Python environment is
created.

CI jobs that need the Puzzletron worker stack should use this image and its
`/venv`; they should not reinstall a separate environment.
