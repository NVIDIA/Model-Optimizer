FROM nvidia/cuda:12.9.2-cudnn-devel-ubuntu24.04@sha256:b4db213759eb86d55a7271909bdad891fab300ec5700fb4f4656463b2f51980f

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG DEBIAN_FRONTEND=noninteractive

ENV VIRTUAL_ENV=/venv
ENV PATH=/venv/bin/:$PATH
ENV PIP_NO_CACHE_DIR=1
ENV PUZZLETRON_CI_ENVIRONMENT=/opt/puzzletron/ci_environment.json
ENV PUZZLETRON_REQUIREMENTS=/opt/puzzletron/requirements.txt
ENV PYTHONUNBUFFERED=1

COPY examples/puzzletron/ci_environment.json /opt/puzzletron/ci_environment.json
COPY examples/puzzletron/requirements.txt /opt/puzzletron/requirements.txt
COPY examples/puzzletron/ci/setup_env.sh /opt/puzzletron/setup_env.sh
COPY pyproject.toml LICENSE_HEADER /opt/modelopt-dependencies/

RUN bash /opt/puzzletron/setup_env.sh --deps
RUN mkdir -p /opt/modelopt-dependencies/modelopt && \
    touch /opt/modelopt-dependencies/modelopt/__init__.py && \
    python -m pip install "/opt/modelopt-dependencies[hf,puzzletron,dev-test]" && \
    python -m pip uninstall -y nvidia-modelopt && \
    python -m pip check && \
    bash /opt/puzzletron/setup_env.sh --verify

LABEL org.opencontainers.image.source="https://github.com/NVIDIA/Model-Optimizer" \
    com.nvidia.modelopt.puzzletron.environment="examples/puzzletron/ci_environment.json" \
    com.nvidia.modelopt.puzzletron.environment-recipe="examples/puzzletron/ci/setup_env.sh"

WORKDIR /workspace/modelopt
