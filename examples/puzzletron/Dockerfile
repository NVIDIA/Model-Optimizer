FROM nvidia/cuda:12.9.2-cudnn-devel-ubuntu24.04@sha256:b4db213759eb86d55a7271909bdad891fab300ec5700fb4f4656463b2f51980f

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
ARG TARGETPLATFORM

ENV VIRTUAL_ENV=/venv
ENV PATH=/venv/bin/:$PATH
ENV PIP_NO_CACHE_DIR=1
ENV NLTK_DATA=/opt/puzzletron/nltk_data
ENV PUZZLETRON_CI_ENVIRONMENT=/opt/puzzletron/ci_environment.json
ENV PUZZLETRON_REQUIREMENTS=/opt/puzzletron/requirements.txt
ENV PUZZLETRON_ROOT=/opt/puzzletron/src/modelopt
ENV PUZZLETRON_VENV=/venv
ENV PUZZLETRON_VLLM_ANYMODEL=1
ENV PYTHONPATH=/opt/puzzletron/src/modelopt
ENV PYTHONUNBUFFERED=1

COPY examples/puzzletron/ci_environment.json /opt/puzzletron/ci_environment.json
COPY examples/puzzletron/requirements.txt /opt/puzzletron/requirements.txt
COPY examples/__init__.py /opt/puzzletron/src/modelopt/examples/__init__.py
COPY examples/puzzletron/patches /opt/puzzletron/patches
COPY pyproject.toml LICENSE_HEADER /opt/modelopt-dependencies/

# Install base build tools before compiling CUDA extensions.
RUN test "${TARGETPLATFORM}" = "linux/amd64" || \
        (echo "Puzzletron requires linux/amd64 because its CUDA stack and video decoder are not validated for Linux ARM" >&2; exit 1) && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential ca-certificates cmake curl git jq ninja-build \
        python3 python3-dev python3-pip python3-venv unzip && \
    rm -rf /var/lib/apt/lists/* && \
    python3 -m venv "${VIRTUAL_ENV}" && \
    python -m pip install --upgrade \
        pip "setuptools>=80,<81" "setuptools-scm>=8,<10" setuptools-rust wheel \
        "packaging>=24.2" "cmake>=3.26.1" ninja jinja2

# `pin` keeps ci_environment.json as the single source of dependency versions
# while leaving the install commands readable.
RUN pin() { jq -er ".${1}" "${PUZZLETRON_CI_ENVIRONMENT}"; } && \
    python -m pip install \
        "torch==$(pin torch)" \
        "torchvision==$(pin torchvision)" \
        "torchaudio==$(pin torch)" \
        --index-url https://download.pytorch.org/whl/cu129

# Evaluation and worker dependencies. Install Transformers last so the pinned
# worker version wins over transitive requirements.
RUN pin() { jq -er ".${1}" "${PUZZLETRON_CI_ENVIRONMENT}"; } && \
    python -m pip install \
        -r "${PUZZLETRON_REQUIREMENTS}" \
        "nemo-automodel @ git+$(pin nemo_automodel.repository)@$(pin nemo_automodel.commit)" \
        "aiperf==$(pin gpu_image.aiperf)" \
        "langdetect==$(pin gpu_image.langdetect)" \
        "nltk==$(pin gpu_image.nltk)" \
        "nox==$(pin gpu_image.nox)" && \
    mkdir -p "${NLTK_DATA}/tokenizers" && \
    while read -r nltk_resource nltk_resource_sha256; do \
        nltk_archive="/tmp/${nltk_resource}.zip" && \
        curl --fail --location --silent --show-error \
            --output "${nltk_archive}" \
            "https://raw.githubusercontent.com/nltk/nltk_data/$(pin gpu_image.nltk_data_commit)/packages/tokenizers/${nltk_resource}.zip" && \
        echo "${nltk_resource_sha256}  ${nltk_archive}" | sha256sum --check --strict && \
        unzip -q "${nltk_archive}" -d "${NLTK_DATA}/tokenizers" && \
        rm "${nltk_archive}"; \
    done < <(jq -r \
        '.gpu_image.nltk_resources[] as $name | "\($name) \(.gpu_image.nltk_resource_sha256[$name])"' \
        "${PUZZLETRON_CI_ENVIRONMENT}") && \
    python -m pip install "transformers==$(pin transformers)"

# Compile the pinned AnyModel vLLM fork for every supported worker GPU.
RUN pin() { jq -er ".${1}" "${PUZZLETRON_CI_ENVIRONMENT}"; } && \
    export FORCE_CUDA=1 && \
    export TORCH_CUDA_ARCH_LIST="$(pin runtime_image.torch_cuda_arch_list)" && \
    python -m pip install --no-build-isolation \
        "vllm @ git+$(pin vllm.repository)@$(pin vllm.commit)"

# Compile the remaining CUDA extensions. Mamba is built from its pinned source
# because the compatibility patch is not present in the upstream release.
RUN pin() { jq -er ".${1}" "${PUZZLETRON_CI_ENVIRONMENT}"; } && \
    mamba_patch="$(pin runtime_image.mamba_ssm.compatibility_patch)" && \
    export FORCE_CUDA=1 && \
    export TORCH_CUDA_ARCH_LIST="$(pin runtime_image.torch_cuda_arch_list)" && \
    python -m pip install --no-build-isolation \
        "causal-conv1d==$(pin runtime_image.causal_conv1d)" && \
    echo "$(pin runtime_image.mamba_ssm.compatibility_patch_sha256)  /opt/puzzletron/patches/${mamba_patch}" | \
        sha256sum --check --strict && \
    git clone --filter=blob:none --no-checkout \
        "$(pin runtime_image.mamba_ssm.repository)" /tmp/mamba-ssm && \
    git -C /tmp/mamba-ssm checkout --detach "$(pin runtime_image.mamba_ssm.commit)" && \
    test "$(git -C /tmp/mamba-ssm rev-parse HEAD)" = \
        "$(pin runtime_image.mamba_ssm.commit)" && \
    git -C /tmp/mamba-ssm apply "/opt/puzzletron/patches/${mamba_patch}" && \
    MAMBA_FORCE_BUILD=TRUE python -m pip install --no-build-isolation \
        /tmp/mamba-ssm && \
    rm -rf /tmp/mamba-ssm && \
    python -m pip install \
        "flash-linear-attention[cuda]==$(pin runtime_image.flash_linear_attention)" && \
    export TORCH_CUDA_ARCH_LIST="$(pin runtime_image.grouped_gemm_cuda_arch_list)" && \
    python -m pip install --no-build-isolation \
        "$(pin runtime_image.grouped_gemm.distribution) @ git+$(pin runtime_image.grouped_gemm.repository)@$(pin runtime_image.grouped_gemm.commit)"

# Cache ModelOpt's ordinary Python dependencies separately from its source.
RUN mkdir -p /opt/modelopt-dependencies/modelopt && \
    touch /opt/modelopt-dependencies/modelopt/__init__.py && \
    python -m pip install "/opt/modelopt-dependencies[hf,puzzletron,dev-test]" && \
    python -m pip uninstall -y nvidia-modelopt

# Keep the immutable source revision below dependency compilation so source-only
# rebuilds reuse the pinned CUDA dependency layers.
ARG MODELOPT_REVISION
RUN [[ "${MODELOPT_REVISION}" =~ ^[0-9a-f]{40}$ ]] && \
    printf '%s\n' "${MODELOPT_REVISION}" > /opt/puzzletron/modelopt_revision

COPY pyproject.toml LICENSE_HEADER README.md /opt/puzzletron/src/modelopt/
COPY modelopt /opt/puzzletron/src/modelopt/modelopt
COPY modelopt_recipes /opt/puzzletron/src/modelopt/modelopt_recipes
COPY puzzletron_orchestrator /opt/puzzletron/src/modelopt/puzzletron_orchestrator
COPY puzzletron_setup /opt/puzzletron/src/modelopt/puzzletron_setup
COPY examples/puzzletron /opt/puzzletron/src/modelopt/examples/puzzletron

# Smoke-test the installed environment. Pins and source revisions are checked
# by the repository tests and by the install commands above.
RUN python -m pip install --no-build-isolation --no-deps -e \
        "/opt/puzzletron/src/modelopt[hf,puzzletron,dev-test]" && \
    pin() { jq -er ".${1}" "${PUZZLETRON_CI_ENVIRONMENT}"; } && \
    for module in \
        aiperf causal_conv1d decord fla grouped_gemm langdetect lmms_eval \
        mamba_ssm modelopt nemo_automodel nltk puzzletron_orchestrator \
        puzzletron_setup tilelang torch transformers vllm; do \
        python -c "import ${module}"; \
    done && \
    test "$(python -c 'import torch; print(torch.version.cuda)')" = \
        "$(pin gpu_image.torch_cuda)" && \
    while read -r nltk_resource; do \
        python -c "import nltk; nltk.data.find('tokenizers/${nltk_resource}')"; \
    done < <(jq -r '.gpu_image.nltk_resources[]' "${PUZZLETRON_CI_ENVIRONMENT}") && \
    lmms_root="$(python -c 'import lmms_eval; print(next(iter(lmms_eval.__path__)))')" && \
    while read -r task_config; do \
        test -f "${lmms_root}/${task_config}"; \
    done < <(jq -r '.lmms_eval.task_configs[]' "${PUZZLETRON_CI_ENVIRONMENT}")

LABEL org.opencontainers.image.source="https://github.com/NVIDIA/Model-Optimizer" \
    org.opencontainers.image.revision="${MODELOPT_REVISION}" \
    com.nvidia.modelopt.puzzletron.environment="examples/puzzletron/ci_environment.json" \
    com.nvidia.modelopt.puzzletron.environment-recipe="examples/puzzletron/Dockerfile"

WORKDIR /opt/puzzletron/src/modelopt

CMD ["bash"]
