ARG CUDA_VERSION=12.9.2
ARG BUILD_BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04

FROM ${BUILD_BASE_IMAGE} AS base

ARG DEBIAN_FRONTEND=noninteractive

ARG MODEL_OPT_ROOT=/workspace/modelopt
ARG VLLM_ROOT=/workspace/vllm
ARG AUTOMODEL_ROOT=/workspace/Automodel

# Copy only the setup script first so that the expensive dep-install layer
# stays cached as long as the script hasn't changed, regardless of source edits.
COPY examples/puzzletron/configs/families/nemotron3/setup_env.sh /tmp/setup_env.sh

# Tell the build system to proceed without a live GPU
ARG FORCE_CUDA=1
# Avoid GPU detection at compile time
ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0;10.0"

RUN MODEL_OPT_ROOT="${MODEL_OPT_ROOT}" \
    VLLM_ROOT="${VLLM_ROOT}" \
    AUTOMODEL_ROOT="${AUTOMODEL_ROOT}" \
    bash /tmp/setup_env.sh --deps

ENV VIRTUAL_ENV=/venv
ENV PATH=/venv/bin/:$PATH

# Copy the full repo after deps so source changes don't bust the cache above.
COPY . "${MODEL_OPT_ROOT}"/

RUN MODEL_OPT_ROOT="${MODEL_OPT_ROOT}" bash /tmp/setup_env.sh --modelopt

