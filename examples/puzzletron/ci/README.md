# Puzzletron image validation

The root [`Dockerfile`](../Dockerfile) is the canonical Puzzletron worker and
GPU CI environment. [`ci_environment.json`](../ci_environment.json) records
its immutable VCS inputs, package versions, CUDA architecture targets, binary
and NLTK resource checksums, and the reviewed Mamba compatibility patch.

Build and verify the image from the repository root:

```bash
docker build \
  --platform linux/amd64 \
  --file examples/puzzletron/Dockerfile \
  --build-arg MODELOPT_REVISION="$(git rev-parse HEAD)" \
  --tag modelopt-puzzletron-worker:amd64-local \
  .

docker run --rm modelopt-puzzletron-worker:amd64-local \
  python /opt/puzzletron/verify_image_environment.py \
    --environment /opt/puzzletron/ci_environment.json
```

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
