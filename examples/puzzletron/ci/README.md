# Puzzletron image validation

The root [`Dockerfile`](../Dockerfile) and
[`ci_environment.json`](../ci_environment.json) define an initial pinned
Puzzletron CUDA environment. The manifest records immutable VCS inputs, package
versions, CUDA architecture targets, and the reviewed Mamba compatibility
patch.

Build and verify the image from the repository root:

```bash
docker build \
  --platform linux/amd64 \
  --file examples/puzzletron/Dockerfile \
  --build-arg MODELOPT_REVISION="$(git rev-parse HEAD)" \
  --tag modelopt-puzzletron-runtime:local \
  .

docker run --rm modelopt-puzzletron-runtime:local \
  python /opt/puzzletron/verify_image_environment.py \
    --environment /opt/puzzletron/ci_environment.json
```

The verifier checks the recorded environment contract. It does not prove that
every downstream workload is ready. Checkpoint teacher evaluation currently
needs follow-up work for LMMS-Eval task assets and optional dependencies. GitHub
image builds, registry publication, and digest-consuming GPU jobs are outside
this initial recipe. Known manual additions include `decord`, `langdetect`, and
NLTK's `punkt_tab` data.
