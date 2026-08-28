# Puzzletron image validation

The root [`Dockerfile`](../Dockerfile) is the canonical Puzzletron worker and
GPU CI environment. [`ci_environment.json`](../ci_environment.json) records
its immutable VCS inputs, package versions, CUDA architecture targets, binary
and NLTK resource checksums, and the reviewed Mamba compatibility patch.

Build and verify the image from the repository root:

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

docker run --rm "${image}" \
  python /opt/puzzletron/verify_image_environment.py \
    --environment /opt/puzzletron/ci_environment.json
```

The `amd64-sha-<12-character commit>` tag identifies the platform and gives
people a compact source reference. The `org.opencontainers.image.revision`
label retains the full source commit that identifies the exact Dockerfile and
repository inputs. When an image is published, retain the commit tag and record
the registry digest; consumers should prefer the digest when they need an
immutable reference.

The image is Linux amd64-only because the current CUDA extension set and Linux
`eva-decord 0.6.1` dependency do not have a validated ARM build path.
The verifier checks package versions and sources, CUDA compatibility, worker
imports, LMMS-Eval task configs, and the NLTK resources used by teacher
evaluation. The image build therefore fails when the recorded worker contract
is incomplete.

Use the resulting image directly with Docker, publish it to a registry, or
materialize it in the format accepted by the target Slurm container plugin.
Workers and GPU CI jobs use the same `/venv` environment and the repository at
`/opt/puzzletron/src/modelopt`. Publication and full workload validation are
separate steps; they do not require another package installation recipe.

The `Puzzletron worker image` GitHub workflow builds and smoke-tests the image
for relevant pull-request updates, changes merged into `feature/puzzletron_v2`,
and manual dispatches. It records the runner-local image ID and source revision
but does not publish the image to a registry. The smoke test verifies CUDA
access; teacher evaluation remains a separate integration test.
