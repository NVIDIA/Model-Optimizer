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

FROM nvcr.io/nvidia/pytorch:26.07-py3

ENV LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git \
        libgl1 \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

ENV UV_PYTHON_INSTALL_DIR=/opt/python

RUN python -m pip install --no-cache-dir uv==0.12.3 && \
    uv python install 3.8.20 && \
    uv venv --seed --python 3.8.20 /opt/petr

COPY examples/onnx_ptq/petr/requirements*.txt \
    examples/onnx_ptq/petr/mmdet3d-nuscenes-only.patch \
    /tmp/petr-requirements/
# PETR uses nuScenes; remove mmdet3d's eager Lyft integration and unused PLY dependency.
RUN unset PIP_CONSTRAINT && \
    /opt/petr/bin/python -m pip install --no-cache-dir \
        -r /tmp/petr-requirements/requirements-torch.txt && \
    /opt/petr/bin/python -m pip install --no-cache-dir \
        -r /tmp/petr-requirements/requirements.txt && \
    git clone --branch v1.0.0rc6 --depth 1 \
        https://github.com/open-mmlab/mmdetection3d.git /tmp/mmdetection3d && \
    test "$(git -C /tmp/mmdetection3d rev-parse HEAD)" = \
        47285b3f1e9dba358e98fcd12e523cfd0769c876 && \
    git -C /tmp/mmdetection3d apply --check --index --unidiff-zero \
        /tmp/petr-requirements/mmdet3d-nuscenes-only.patch && \
    git -C /tmp/mmdetection3d apply --index --unidiff-zero \
        /tmp/petr-requirements/mmdet3d-nuscenes-only.patch && \
    /opt/petr/bin/python -m pip install --no-cache-dir \
        --no-build-isolation \
        /tmp/mmdetection3d && \
    /opt/petr/bin/python -m pip check && \
    rm -rf /tmp/mmdetection3d && \
    mkdir -p /opt/petr/lib/python3.8/site-packages/tensorrt && \
    cp /opt/petr/lib/python3.8/site-packages/tensorrt_bindings/__init__.py \
        /opt/petr/lib/python3.8/site-packages/tensorrt/__init__.py && \
    cp /opt/petr/lib/python3.8/site-packages/tensorrt_bindings/tensorrt.so \
        /opt/petr/lib/python3.8/site-packages/tensorrt/tensorrt.so

COPY . /opt/Model-Optimizer
RUN cd /opt/Model-Optimizer && \
    env -u PIP_CONSTRAINT python -m pip install --no-cache-dir \
        -e ".[onnx]" \
        "onnxruntime-gpu[cuda,cudnn]~=1.24.2"
