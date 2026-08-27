FROM nvidia/cuda:12.9.2-cudnn-devel-ubuntu24.04@sha256:b4db213759eb86d55a7271909bdad891fab300ec5700fb4f4656463b2f51980f

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG DEBIAN_FRONTEND=noninteractive

ENV VIRTUAL_ENV=/venv
ENV PATH=/venv/bin/:$PATH
ENV PIP_NO_CACHE_DIR=1
ENV PUZZLETRON_CI_ENVIRONMENT=/opt/puzzletron/ci_environment.json
ENV PUZZLETRON_REQUIREMENTS=/opt/puzzletron/requirements.txt
ENV PUZZLETRON_VERIFY_SCRIPT=/opt/puzzletron/verify_image_environment.py
ENV PUZZLETRON_ROOT=/opt/puzzletron/src/modelopt
ENV PUZZLETRON_VENV=/venv
ENV PUZZLETRON_VLLM_ANYMODEL=1
ENV PYTHONPATH=/opt/puzzletron/src/modelopt
ENV PYTHONUNBUFFERED=1

COPY examples/puzzletron/ci_environment.json /opt/puzzletron/ci_environment.json
COPY examples/puzzletron/requirements.txt /opt/puzzletron/requirements.txt
COPY examples/puzzletron/ci_environment.py /opt/puzzletron/src/modelopt/examples/puzzletron/ci_environment.py
COPY examples/puzzletron/ci/verify_image_environment.py /opt/puzzletron/verify_image_environment.py
COPY examples/puzzletron/patches /opt/puzzletron/patches
COPY pyproject.toml LICENSE_HEADER /opt/modelopt-dependencies/

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential ca-certificates cmake git ninja-build \
        python3 python3-dev python3-pip python3-venv && \
    rm -rf /var/lib/apt/lists/* && \
    python3 -m venv "${VIRTUAL_ENV}" && \
    python -m pip install --upgrade \
        pip "setuptools>=80,<81" "setuptools-scm>=8,<10" setuptools-rust wheel \
        "packaging>=24.2" "cmake>=3.26.1" ninja jinja2 && \
    python "${PUZZLETRON_VERIFY_SCRIPT}" \
        --environment "${PUZZLETRON_CI_ENVIRONMENT}" \
        --profile runtime \
        --manifest-only

RUN torch_version="$(python -c \
        'import json, os; print(json.load(open(os.environ["PUZZLETRON_CI_ENVIRONMENT"]))["torch"])')" && \
    torchvision_version="$(python -c \
        'import json, os; print(json.load(open(os.environ["PUZZLETRON_CI_ENVIRONMENT"]))["torchvision"])')" && \
    python -m pip install \
        "torch==${torch_version}" \
        "torchvision==${torchvision_version}" \
        "torchaudio==${torch_version}" \
        --index-url https://download.pytorch.org/whl/cu129

RUN automodel_repository="$(python -c \
        'import json, os; print(json.load(open(os.environ["PUZZLETRON_CI_ENVIRONMENT"]))["nemo_automodel"]["repository"])')" && \
    automodel_revision="$(python -c \
        'import json, os; print(json.load(open(os.environ["PUZZLETRON_CI_ENVIRONMENT"]))["nemo_automodel"]["commit"])')" && \
    aiperf_version="$(python -c \
        'import json, os; print(json.load(open(os.environ["PUZZLETRON_CI_ENVIRONMENT"]))["gpu_image"]["aiperf"])')" && \
    nox_version="$(python -c \
        'import json, os; print(json.load(open(os.environ["PUZZLETRON_CI_ENVIRONMENT"]))["gpu_image"]["nox"])')" && \
    transformers_version="$(python -c \
        'import json, os; print(json.load(open(os.environ["PUZZLETRON_CI_ENVIRONMENT"]))["transformers"])')" && \
    python -m pip install \
        -r "${PUZZLETRON_REQUIREMENTS}" \
        "nemo-automodel @ git+${automodel_repository}@${automodel_revision}" \
        "aiperf==${aiperf_version}" \
        "nox==${nox_version}" && \
    python -m pip install "transformers==${transformers_version}" && \
    python -m pip check

RUN vllm_repository="$(python -c \
        'import json, os; print(json.load(open(os.environ["PUZZLETRON_CI_ENVIRONMENT"]))["vllm"]["repository"])')" && \
    vllm_revision="$(python -c \
        'import json, os; print(json.load(open(os.environ["PUZZLETRON_CI_ENVIRONMENT"]))["vllm"]["commit"])')" && \
    recorded_cuda_architectures="$(python -c \
        'import json, os; print(json.load(open(os.environ["PUZZLETRON_CI_ENVIRONMENT"]))["runtime_image"]["torch_cuda_arch_list"])')" && \
    export FORCE_CUDA=1 && \
    export TORCH_CUDA_ARCH_LIST="${recorded_cuda_architectures}" && \
    python -m pip install --no-build-isolation \
        "vllm @ git+${vllm_repository}@${vllm_revision}" && \
    python -m pip check

RUN causal_conv1d_version="$(python -c \
        'import json, os; print(json.load(open(os.environ["PUZZLETRON_CI_ENVIRONMENT"]))["runtime_image"]["causal_conv1d"])')" && \
    grouped_gemm_repository="$(python -c \
        'import json, os; print(json.load(open(os.environ["PUZZLETRON_CI_ENVIRONMENT"]))["runtime_image"]["grouped_gemm"]["repository"])')" && \
    grouped_gemm_revision="$(python -c \
        'import json, os; print(json.load(open(os.environ["PUZZLETRON_CI_ENVIRONMENT"]))["runtime_image"]["grouped_gemm"]["commit"])')" && \
    grouped_gemm_distribution="$(python -c \
        'import json, os; print(json.load(open(os.environ["PUZZLETRON_CI_ENVIRONMENT"]))["runtime_image"]["grouped_gemm"]["distribution"])')" && \
    grouped_gemm_cuda_architectures="$(python -c \
        'import json, os; print(json.load(open(os.environ["PUZZLETRON_CI_ENVIRONMENT"]))["runtime_image"]["grouped_gemm_cuda_arch_list"])')" && \
    linear_attention_version="$(python -c \
        'import json, os; print(json.load(open(os.environ["PUZZLETRON_CI_ENVIRONMENT"]))["runtime_image"]["flash_linear_attention"])')" && \
    mamba_ssm_repository="$(python -c \
        'import json, os; print(json.load(open(os.environ["PUZZLETRON_CI_ENVIRONMENT"]))["runtime_image"]["mamba_ssm"]["repository"])')" && \
    mamba_ssm_revision="$(python -c \
        'import json, os; print(json.load(open(os.environ["PUZZLETRON_CI_ENVIRONMENT"]))["runtime_image"]["mamba_ssm"]["commit"])')" && \
    mamba_ssm_patch="$(python -c \
        'import json, os; print(json.load(open(os.environ["PUZZLETRON_CI_ENVIRONMENT"]))["runtime_image"]["mamba_ssm"]["compatibility_patch"])')" && \
    mamba_ssm_patch_sha256="$(python -c \
        'import json, os; print(json.load(open(os.environ["PUZZLETRON_CI_ENVIRONMENT"]))["runtime_image"]["mamba_ssm"]["compatibility_patch_sha256"])')" && \
    recorded_cuda_architectures="$(python -c \
        'import json, os; print(json.load(open(os.environ["PUZZLETRON_CI_ENVIRONMENT"]))["runtime_image"]["torch_cuda_arch_list"])')" && \
    export FORCE_CUDA=1 && \
    export TORCH_CUDA_ARCH_LIST="${recorded_cuda_architectures}" && \
    python -m pip install --no-build-isolation \
        "causal-conv1d==${causal_conv1d_version}" && \
    echo "${mamba_ssm_patch_sha256}  /opt/puzzletron/patches/${mamba_ssm_patch}" | \
        sha256sum --check --strict && \
    git clone --filter=blob:none --no-checkout \
        "${mamba_ssm_repository}" /tmp/mamba-ssm && \
    git -C /tmp/mamba-ssm checkout --detach "${mamba_ssm_revision}" && \
    test "$(git -C /tmp/mamba-ssm rev-parse HEAD)" = "${mamba_ssm_revision}" && \
    git -C /tmp/mamba-ssm apply "/opt/puzzletron/patches/${mamba_ssm_patch}" && \
    MAMBA_FORCE_BUILD=TRUE python -m pip install --no-build-isolation \
        /tmp/mamba-ssm && \
    rm -rf /tmp/mamba-ssm && \
    python -m pip install \
        "flash-linear-attention[cuda]==${linear_attention_version}" && \
    export TORCH_CUDA_ARCH_LIST="${grouped_gemm_cuda_architectures}" && \
    python -m pip install --no-build-isolation \
        "${grouped_gemm_distribution} @ git+${grouped_gemm_repository}@${grouped_gemm_revision}" && \
    python -m pip check

RUN mkdir -p /opt/modelopt-dependencies/modelopt && \
    touch /opt/modelopt-dependencies/modelopt/__init__.py && \
    python -m pip install "/opt/modelopt-dependencies[hf,puzzletron,dev-test]" && \
    python -m pip uninstall -y nvidia-modelopt && \
    python -m pip check

# Keep the immutable source revision below dependency compilation so source-only
# rebuilds reuse the pinned CUDA dependency layers.
ARG MODELOPT_REVISION
RUN [[ "${MODELOPT_REVISION}" =~ ^[0-9a-f]{40}$ ]]

COPY pyproject.toml LICENSE_HEADER README.md /opt/puzzletron/src/modelopt/
COPY modelopt /opt/puzzletron/src/modelopt/modelopt
COPY modelopt_recipes /opt/puzzletron/src/modelopt/modelopt_recipes
COPY puzzletron_orchestrator /opt/puzzletron/src/modelopt/puzzletron_orchestrator
COPY puzzletron_setup /opt/puzzletron/src/modelopt/puzzletron_setup
COPY examples/__init__.py /opt/puzzletron/src/modelopt/examples/__init__.py
COPY examples/puzzletron /opt/puzzletron/src/modelopt/examples/puzzletron

RUN python -m pip install --no-build-isolation --no-deps -e \
        "/opt/puzzletron/src/modelopt[hf,puzzletron,dev-test]" && \
    python -m pip check && \
    python "${PUZZLETRON_VERIFY_SCRIPT}" \
        --environment "${PUZZLETRON_CI_ENVIRONMENT}" \
        --profile runtime && \
    python -c "import aiperf, causal_conv1d, fla, grouped_gemm, lmms_eval, mamba_ssm, modelopt, nemo_automodel, puzzletron_orchestrator, puzzletron_setup, tilelang, torch, transformers, vllm"

LABEL org.opencontainers.image.source="https://github.com/NVIDIA/Model-Optimizer" \
    org.opencontainers.image.revision="${MODELOPT_REVISION}" \
    com.nvidia.modelopt.puzzletron.environment="examples/puzzletron/ci_environment.json" \
    com.nvidia.modelopt.puzzletron.environment-recipe="examples/puzzletron/Dockerfile"

WORKDIR /opt/puzzletron/src/modelopt

CMD ["bash"]
