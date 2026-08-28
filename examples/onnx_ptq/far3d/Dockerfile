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

# TensorRT 10.11 from the base image builds and runs the FAR3D engines.
FROM nvcr.io/nvidia/pytorch:25.06-py3

ENV LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ARG TORCH_CUDA_ARCH_LIST=8.9+PTX

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends libgl1 && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir uv==0.12.3 && \
    UV_PYTHON_INSTALL_DIR=/opt/python uv venv --seed --python 3.9.23 /opt/far3d

COPY examples/onnx_ptq/install-evaluator-cu129.sh \
    examples/onnx_ptq/mmcv-torch2.patch \
    examples/onnx_ptq/mmdet3d-runtime.patch \
    examples/onnx_ptq/requirements-evaluator-cu129.txt \
    examples/onnx_ptq/requirements-modelopt-25.06.txt \
    examples/onnx_ptq/requirements-torch-cu129.txt \
    examples/onnx_ptq/far3d/requirements.txt \
    /tmp/evaluator/
RUN bash /tmp/evaluator/install-evaluator-cu129.sh \
        /opt/far3d "${TORCH_CUDA_ARCH_LIST}" \
        -r /tmp/evaluator/requirements.txt && \
    env -u PIP_CONSTRAINT python -m pip install --no-cache-dir \
        -r /tmp/evaluator/requirements-modelopt-25.06.txt

COPY . /opt/Model-Optimizer
RUN env -u PIP_CONSTRAINT python -m pip install --no-cache-dir --no-deps \
    -e /opt/Model-Optimizer
