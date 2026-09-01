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

FROM nvcr.io/nvidia/pytorch:22.06-py3@sha256:6f9a1fdfcbc1d1aa6f28791ed7dc41d651d7c47d634c6a11b4f0692c50c8a664 AS evaluator

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 && \
    rm -rf /var/lib/apt/lists/*

RUN env -u PIP_CONSTRAINT PIP_CONFIG_FILE=/dev/null python -m pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu117 \
        torch==1.13.1+cu117 \
        torchvision==0.14.1+cu117

COPY examples/onnx_ptq/requirements-evaluator.txt /tmp/requirements-evaluator.txt
# Avoid the training and development dependencies declared by the legacy projects.
RUN env -u PIP_CONSTRAINT PIP_CONFIG_FILE=/dev/null FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=8 \
        python -m pip install --no-cache-dir --no-build-isolation \
        -r /tmp/requirements-evaluator.txt && \
    env -u PIP_CONSTRAINT PIP_CONFIG_FILE=/dev/null python -m pip install \
        --no-cache-dir --no-build-isolation --no-deps \
        av2==0.2.1 \
        lyft-dataset-sdk==0.0.8 \
        mmdet3d==1.0.0rc6 \
        nuscenes-devkit==1.1.11 \
        refile==0.4.1

# Both TensorRT packages own the Python namespace, so install the CUDA wrapper last.
RUN env -u PIP_CONSTRAINT PIP_CONFIG_FILE=/dev/null python -m pip uninstall -y tensorrt && \
    env -u PIP_CONSTRAINT PIP_CONFIG_FILE=/dev/null python -m pip install \
        --no-cache-dir --force-reinstall \
        tensorrt==11.1.0.106 && \
    env -u PIP_CONSTRAINT PIP_CONFIG_FILE=/dev/null python -m pip install \
        --no-cache-dir --force-reinstall --no-deps \
        tensorrt-cu13==11.1.0.106 && \
    python -c "import tensorrt as trt; assert trt.__version__ == '11.1.0.106'"

COPY examples/onnx_ptq /opt/Model-Optimizer/examples/onnx_ptq
WORKDIR /workspace


FROM nvcr.io/nvidia/pytorch:26.07-py3@sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c AS modelopt

COPY . /opt/Model-Optimizer
RUN env -u PIP_CONSTRAINT PIP_CONFIG_FILE=/dev/null python -m pip install --no-cache-dir \
        "/opt/Model-Optimizer[onnx]" \
        "onnxruntime-gpu[cuda,cudnn]~=1.24.2" && \
    python -c "import modelopt, tensorrt as trt; assert trt.__version__ == '11.1.0.106'"

WORKDIR /workspace
