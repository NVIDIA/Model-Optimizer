# Build the Puzzletron worker image

The [`Dockerfile`](../Dockerfile) contains the worker installation steps.
[`ci_environment.json`](../ci_environment.json) stores the versions, source
revisions, CUDA targets, and downloaded-file checksums used by those steps.
The image checker reads the same file, so installation and validation use the
same values.

## Build

Run this command from the repository root on a Linux amd64 system with Docker:

```bash
test -z "$(git status --porcelain)"
revision="$(git rev-parse HEAD)"
image="modelopt-puzzletron:amd64-sha-$(git rev-parse --short=12 HEAD)"
docker build \
  --platform linux/amd64 \
  --file examples/puzzletron/Dockerfile \
  --build-arg MODELOPT_REVISION="${revision}" \
  --tag "${image}" \
  .
```

The tag includes the platform and source commit. The image also records the
full commit in its `org.opencontainers.image.revision` label.

## Check

In the same shell, check the installed packages, source revisions, CUDA
version, imports, and evaluation data:

```bash
docker run --rm "${image}" \
  python /opt/puzzletron/verify_image_environment.py \
    --environment /opt/puzzletron/ci_environment.json
```

On a host with an NVIDIA GPU, also check CUDA access:

```bash
docker run --gpus all --ipc=host --rm "${image}" \
  python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
```

## Use

The image contains the worker environment at `/venv` and the ModelOpt checkout
at `/opt/puzzletron/src/modelopt`. Use the image directly with Docker, publish
it to a registry, or convert it to the format accepted by the target Slurm
container plugin.

The current image supports Linux amd64 only. Its CUDA extensions and
`eva-decord 0.6.1` dependency have not been validated on Linux ARM.

The source tag identifies the recipe revision, but rebuilding that revision may
resolve newer transitive Python dependencies. Record the registry digest when
the exact built image must be reused. This repository does not publish the
image automatically.
