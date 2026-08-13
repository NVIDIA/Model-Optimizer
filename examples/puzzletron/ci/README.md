# Puzzletron GPU lifecycle CI image

This directory defines the execution image for the hermetic one-GPU Puzzletron
lifecycle gate. The image supplies the pinned CUDA and Python dependencies;
the `gpu_puzzletron` Nox session installs the checked-out ModelOpt revision and
runs the sole reusable tiny-Qwen lifecycle test.

The base image is pinned by OCI digest in both
[`Dockerfile`](Dockerfile) and [`ci_environment.json`](../ci_environment.json).
The environment file also owns the exact Torch, Transformers, LMMS-Eval,
AutoModel, AIPerf, and Nox versions. The image preinstalls ModelOpt's declared
runtime and test dependencies. The setup script verifies the pins when the
image is built and again after installing only the checked-out ModelOpt source,
without resolving dependencies or using build isolation at runtime.

Build the image from the repository root:

```bash
docker build \
  --platform linux/amd64 \
  --file examples/puzzletron/ci/Dockerfile \
  --tag modelopt-puzzletron-ci:local \
  .
```

Run the same entry point used by CI with one visible GPU:

```bash
docker run --gpus device=0 --ipc=host --rm \
  --volume "${PWD}:/workspace/modelopt" \
  --workdir /workspace/modelopt \
  modelopt-puzzletron-ci:local \
  nox -s gpu_puzzletron
```

Publishing the reviewed image is a separate registry operation. Publish it with
anonymous pull access, then configure `PUZZLETRON_GPU_CI_IMAGE` with the
complete immutable `nvcr.io/...@sha256:...` reference. Until the variable is
configured, the workflow reports an explicit non-error skip and does not
allocate a GPU runner. Once configured, invalid tags, mutable references, and
non-NVCR images fail before GPU allocation. The workflow intentionally does not
expose registry credentials to copied pull-request branches.
