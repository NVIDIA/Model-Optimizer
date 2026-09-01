# Build the Puzzletron worker image

The [`Dockerfile`](../Dockerfile) contains the worker installation steps.
[`ci_environment.json`](../ci_environment.json) stores the versions, source
revisions, CUDA targets, and downloaded-file checksums used by those steps.
The Dockerfile reads the same file for installation and its build-time checks.

## Build

Run one repository command from a clean checkout on a Linux amd64 system with
Docker:

```bash
python examples/puzzletron/build_worker_image.py
```

The command always uses `examples/puzzletron/Dockerfile`, the repository root as
the build context, Linux amd64 as the platform, and the current full Git
revision. It checks the installed modules, CUDA version, and required evaluation
data. The full source revision is recorded in the image label and at
`/opt/puzzletron/modelopt_revision`. The Dockerfile remains directly usable by
standard Docker tools, but users do not need to assemble these arguments.

## Export for another runtime

The same command can build and export a revision-named Enroot/Pyxis image in one
step. No separate Docker build is required:

```bash
python examples/puzzletron/build_worker_image.py \
  --output-dir /path/to/output \
  --sqsh
```

The helper runs the same controlled Docker build before exporting. Docker reuses
its build cache when the image is already current. Creating a SquashFS image
requires Enroot and Docker on the same Linux amd64 host. Use node-local or other
large storage for the output directory. The local Docker image remains
available after export and can be removed with normal Docker image-management
commands when it is no longer needed.

Use `--archive` instead of `--sqsh` for a compressed Docker archive, or pass
both flags to create both formats.

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

## GPU check

The build does not require a GPU. On a host with an NVIDIA GPU, check CUDA
access using the local Docker name printed by the build command:

```bash
image="modelopt-puzzletron:linux-amd64-git-$(git rev-parse --short=12 HEAD)"
docker run --gpus all --ipc=host --rm "${image}" \
  python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
```

## Use

The image contains the worker environment at `/venv` and the ModelOpt checkout
at `/opt/puzzletron/src/modelopt`. Use the image directly with Docker, publish
it to a registry, or export it for Enroot, Pyxis, or the target Slurm container
plugin.

The current image supports Linux amd64 only. Its CUDA extensions and
`eva-decord 0.6.1` dependency have not been validated on Linux ARM.

The artifact filename identifies the recipe revision. Keep the image and its
checksum together; those files identify the exact export without relying on a
local Docker tag. Rebuilding that revision may still resolve newer transitive
Python dependencies. Record the registry digest if the image is later
published. This repository does not publish the image automatically.
