# Puzzletron image validation and consumption

Puzzletron has one repository-owned environment image. The root
[`Dockerfile`](../Dockerfile) installs the complete validated Qwen and Nemotron
runtime used by lifecycle CI, runtime-stat collection, serving, AIPerf, and
external users. It also bakes in the ModelOpt source revision recorded in the
image metadata, including Mamba, causal-convolution, grouped-GEMM, and the
reviewed TileLang compatibility patch needed by the pinned sources.

The base image is pinned by OCI digest in both the Dockerfile and
[`ci_environment.json`](../ci_environment.json). The manifest owns the exact
Torch, Transformers, LMMS-Eval, AutoModel, patched vLLM, AIPerf, Nox,
linear-attention, Mamba, causal-convolution, grouped-GEMM, and CUDA-architecture
inputs. The
[`verify_image_environment.py`](verify_image_environment.py) verifier checks
that recorded compatibility contract during the build and again in a fresh
container. Secondary and transitive dependencies are resolved by pip from the
repository requirements; the image is not claimed to be bit-for-bit
reproducible across rebuild dates.

The Dockerfile is the sole third-party installation recipe. There is no
separate CI Dockerfile or host setup script. CI uses the same full image as
runtime jobs. For pull requests, the checked-out ModelOpt source is mounted over
the baked source and installed with `--no-deps`; this changes only the source
under test and preserves the verified image environment. During the immutable
digest transition, the existing lifecycle job checks the shared CI subset; the
image workflow separately checks the complete runtime profile before running
that lifecycle job.

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
    --environment /opt/puzzletron/ci_environment.json \
    --profile runtime
```

The image workflow also mounts the current checkout and runs the focused
one-GPU lifecycle test. That gate proves the full image can replace the prior
lean CI environment; it does not publish an image.

Publication is a separate trusted registry operation. The publication workflow
should push the verified image to an approved NGC repository, resolve the
resulting digest, and make the complete immutable `nvcr.io/...@sha256:...`
reference available to CI and users. The resolver rejects tags and non-NVCR
references before a GPU runner is allocated.
