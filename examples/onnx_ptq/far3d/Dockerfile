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

# Pinned to TensorRT 10.11 to match the DL4AGX reference and documented results.
FROM nvcr.io/nvidia/pytorch:25.06-py3

ENV LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

ENV UV_PYTHON_INSTALL_DIR=/opt/python

RUN python -m pip install --no-cache-dir uv && \
    uv python install 3.8 && \
    uv venv --seed --python 3.8 /opt/far3d

COPY examples/onnx_ptq/far3d/requirements*.txt /tmp/far3d-requirements/
RUN env -u PIP_CONSTRAINT /opt/far3d/bin/python -m pip install --no-cache-dir \
        -r /tmp/far3d-requirements/requirements-torch.txt && \
    env -u PIP_CONSTRAINT /opt/far3d/bin/python -m pip install --no-cache-dir \
        -r /tmp/far3d-requirements/requirements.txt && \
    env -u PIP_CONSTRAINT /opt/far3d/bin/python -m pip install --no-cache-dir \
        --no-build-isolation \
        -r /tmp/far3d-requirements/requirements-mmdet3d.txt && \
    mkdir -p /opt/far3d/lib/python3.8/site-packages/tensorrt && \
    cp /opt/far3d/lib/python3.8/site-packages/tensorrt_bindings/__init__.py \
        /opt/far3d/lib/python3.8/site-packages/tensorrt/__init__.py && \
    cp /opt/far3d/lib/python3.8/site-packages/tensorrt_bindings/tensorrt.so \
        /opt/far3d/lib/python3.8/site-packages/tensorrt/tensorrt.so

COPY . /opt/Model-Optimizer
RUN env -u PIP_CONSTRAINT python -m pip install --no-cache-dir \
        -r /tmp/far3d-requirements/requirements-modelopt-onnx.txt && \
    cd /opt/Model-Optimizer && \
    env -u PIP_CONSTRAINT python -m pip install --no-cache-dir --no-deps -e ".[onnx]"
