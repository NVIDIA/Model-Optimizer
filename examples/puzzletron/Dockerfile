ARG BUILD_BASE_IMAGE=nvcr.io/nvidia/pytorch:26.07-py3

FROM ${BUILD_BASE_IMAGE} as BASE

ARG DEBIAN_FRONTEND=noninteractive

ARG MODEL_OPT_ROOT=/workspace/modelopt
ARG VLLM_ROOT=/workspace/vllm
ARG AUTOMODEL_ROOT=/workspace/Automodel

COPY examples/puzzletron/setup_packages.sh /tmp/setup_packages.sh

ENV VIRTUAL_ENV=/venv

ARG DOCKER_BUILD=1
RUN DOCKER_BUILD="$DOCKER_BUILD" bash /tmp/setup_packages.sh
