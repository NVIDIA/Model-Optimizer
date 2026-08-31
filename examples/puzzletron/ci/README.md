# Build the Puzzletron worker image

The [`Dockerfile`](../Dockerfile) contains the worker installation steps.
[`ci_environment.json`](../ci_environment.json) stores the versions, source
revisions, CUDA targets, and downloaded-file checksums used by those steps.
The image checker reads the same file, so installation and validation use the
same values.

## Build

Run the build command from a clean repository checkout on a Linux amd64 system
with Docker:

```bash
python examples/puzzletron/build_worker_image.py
```

The command builds the image, checks its installed environment, and prints its
local Docker name. That name is only a convenience. The full source commit is
recorded in the image, and exported files use the same readable commit-based
filename.

## Export

Add `--archive` for a portable Docker archive, `--sqsh` for an Enroot/Pyxis
image, or both:

```bash
python examples/puzzletron/build_worker_image.py \
  --output-dir /path/to/output \
  --archive \
  --sqsh
```

Creating a Docker archive also requires `zstd`. Creating a SquashFS image
requires Enroot and Docker on the same build host.

Both formats use the same source identity:

```text
modelopt-puzzletron-linux-amd64-git-<12-character-commit>.tar.zst
modelopt-puzzletron-linux-amd64-git-<12-character-commit>.sqsh
```

Each image file has a matching `.sha256` file.

Verify an exported file from its output directory with:

```bash
sha256sum --check modelopt-puzzletron-linux-amd64-git-<12-character-commit>.sqsh.sha256
```

## Check

The build command already checks the installed packages, source revisions,
CUDA version, imports, and evaluation data. To repeat that check later, first
set the local Docker name printed by the build command:

```bash
image="modelopt-puzzletron:linux-amd64-git-$(git rev-parse --short=12 HEAD)"
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

The artifact filename identifies the recipe revision. Keep the image and its
checksum together; those files identify the exact export without relying on a
local Docker tag. Rebuilding that revision may still resolve newer transitive
Python dependencies. Record the registry digest if the image is later
published. This repository does not publish the image automatically.
