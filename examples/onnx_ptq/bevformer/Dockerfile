# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM nvcr.io/nvidia/pytorch:26.01-py3

ARG NUM_JOBS=8
ARG ORT_VERSION=1.24.0.dev20260123002
ARG DL4AGX_COMMIT=9f7b29104c253d5bc68334e7b83b3eecb72d4572
ARG BEVFORMER_TRT_COMMIT=303d3140c14016047c07f9db73312af364f0dd7c
ARG TENSORRT_COMMIT=3b4ddc1d45f11167f2ea53dc6046bba8df800e2d
ARG MMCV_COMMIT=235c0253ab8806a2a2ee6954b4258a95358497ac
ARG MMDET_COMMIT=3b72b12fe9b14de906d1363982b9fba05e7d47c1

ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9+PTX"
ENV TRT_LIBPATH=/usr/lib/x86_64-linux-gnu
ENV LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-13.1/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY . /opt/Model-Optimizer
COPY examples/onnx_ptq/bevformer/requirements.txt /tmp/bevformer-requirements.txt
# Stable ONNX Runtime wheels do not support the base image's CUDA 13 runtime.
RUN env -u PIP_CONSTRAINT python -m pip install --no-cache-dir \
        --upgrade pip setuptools wheel && \
    env -u PIP_CONSTRAINT python -m pip install --no-cache-dir cuda-python==13.1.0 && \
    env -u PIP_CONSTRAINT python -m pip install --no-cache-dir \
        -e "/opt/Model-Optimizer[onnx]" && \
    env -u PIP_CONSTRAINT python -m pip install --no-cache-dir --pre \
        --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/ \
        --force-reinstall --no-deps \
        onnxruntime-gpu==${ORT_VERSION} && \
    env -u PIP_CONSTRAINT python -m pip install --no-cache-dir \
        -r /tmp/bevformer-requirements.txt

RUN git clone https://github.com/NVIDIA/DL4AGX.git /workspace/DL4AGX && \
    git -C /workspace/DL4AGX checkout ${DL4AGX_COMMIT} && \
    git clone https://github.com/DerryHub/BEVFormer_tensorrt.git \
        /workspace/BEVFormer_tensorrt && \
    git -C /workspace/BEVFormer_tensorrt checkout ${BEVFORMER_TRT_COMMIT} && \
    git -C /workspace/BEVFormer_tensorrt apply \
        /workspace/DL4AGX/AV-Solutions/bevformer-int8-eq/bevformer_trt10.patch

RUN git clone https://github.com/NVIDIA/TensorRT.git /workspace/TensorRT && \
    git -C /workspace/TensorRT checkout ${TENSORRT_COMMIT} && \
    cd /workspace/TensorRT/tools/pytorch-quantization && \
    MAX_JOBS=${NUM_JOBS} env -u PIP_CONSTRAINT python -m pip install \
        --no-cache-dir . --no-build-isolation

RUN git clone https://github.com/open-mmlab/mmcv.git \
        /workspace/BEVFormer_tensorrt/third_party/mmcv && \
    git -C /workspace/BEVFormer_tensorrt/third_party/mmcv checkout ${MMCV_COMMIT} && \
    sed -i 's/c++14/c++17/g' \
        /workspace/BEVFormer_tensorrt/third_party/mmcv/setup.py && \
    env -u PIP_CONSTRAINT python -m pip install --no-cache-dir \
        -r /workspace/BEVFormer_tensorrt/third_party/mmcv/requirements/optional.txt && \
    cd /workspace/BEVFormer_tensorrt/third_party/mmcv && \
    MAX_JOBS=${NUM_JOBS} MMCV_WITH_OPS=1 \
        env -u PIP_CONSTRAINT python -m pip install --no-cache-dir -v -e . \
        --no-build-isolation

RUN git clone https://github.com/open-mmlab/mmdetection.git \
        /workspace/BEVFormer_tensorrt/third_party/mmdetection && \
    git -C /workspace/BEVFormer_tensorrt/third_party/mmdetection checkout ${MMDET_COMMIT} && \
    env -u PIP_CONSTRAINT python -m pip install --no-cache-dir -v \
        /workspace/BEVFormer_tensorrt/third_party/mmdetection --no-build-isolation && \
    cd /workspace/BEVFormer_tensorrt/third_party/bev_mmdet3d && \
    MAX_JOBS=${NUM_JOBS} env -u PIP_CONSTRAINT \
        python -m pip install --no-cache-dir -v -e . \
        --no-build-isolation

WORKDIR /workspace/BEVFormer_tensorrt
