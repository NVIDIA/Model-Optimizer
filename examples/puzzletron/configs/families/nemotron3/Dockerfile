ARG CUDA_VERSION=12.9.2
ARG BUILD_BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04

FROM ${BUILD_BASE_IMAGE} AS base

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install -y build-essential cmake git ninja-build python3 python3-dev python3-pip python3-venv

ARG MODEL_OPT_ROOT=/workspace/modelopt
ARG VLLM_ROOT=/workspace/vllm
ARG AUTOMODEL_ROOT=/workspace/Automodel

RUN git clone --branch feature/add_anymodel_to_vllm --single-branch https://github.com/Separius/vllm.git "${VLLM_ROOT}"
RUN git clone --branch puzzletron --single-branch https://github.com/Separius/Automodel.git "${AUTOMODEL_ROOT}"

RUN python3 -m venv /venv
ENV VIRTUAL_ENV=/venv
ENV PATH=/venv/bin/:$PATH

RUN python3 -m pip install --upgrade pip "setuptools>=80,<81" "setuptools-scm>=8" setuptools-rust wheel "packaging>=24.2" "cmake>=3.26.1" ninja jinja2

RUN python -m pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu129

RUN VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_VARIANT=cu129 python -m pip install --no-build-isolation -e "${VLLM_ROOT}"

RUN python -m pip install -e "${AUTOMODEL_ROOT}"
RUN python -m pip install aiperf

COPY . "${MODEL_OPT_ROOT}"/
RUN python -m pip install -e "${MODEL_OPT_ROOT}[hf]"

# Mixture of experts
RUN python -m pip install --no-build-isolation "git+https://github.com/fanshiqing/grouped_gemm@v1.1.4"

# Mamba
RUN python -m pip install "mamba-ssm[causal-conv1d]" --no-build-isolation

# Linear attention
RUN python -m pip install "flash-linear-attention[cuda]"
