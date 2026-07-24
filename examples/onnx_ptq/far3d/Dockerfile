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

RUN env -u PIP_CONSTRAINT /opt/far3d/bin/python -m pip install --no-cache-dir \
        torch==1.13.1+cu117 \
        torchvision==0.14.1+cu117 \
        --extra-index-url https://download.pytorch.org/whl/cu117

RUN env -u PIP_CONSTRAINT /opt/far3d/bin/python -m pip install --no-cache-dir \
        "numpy<1.24" \
        opencv-python==4.5.5.64 \
        yapf==0.32.0 \
        mmcv-full==1.7.0 \
        -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/index.html

RUN env -u PIP_CONSTRAINT /opt/far3d/bin/python -m pip install --no-cache-dir \
        av2==0.2.1 \
        einops \
        "ipython<9" \
        kornia==0.6.12 \
        mmdet==2.28.2 \
        mmsegmentation==0.30.0 \
        onnx \
        onnx-graphsurgeon==0.6.1 \
        onnxruntime \
        onnxsim \
        refile \
        tensorrt-cu12-bindings==10.11.0.33 \
        --extra-index-url https://pypi.nvidia.com && \
    mkdir -p /opt/far3d/lib/python3.8/site-packages/tensorrt && \
    cp /opt/far3d/lib/python3.8/site-packages/tensorrt_bindings/__init__.py \
        /opt/far3d/lib/python3.8/site-packages/tensorrt/__init__.py && \
    cp /opt/far3d/lib/python3.8/site-packages/tensorrt_bindings/tensorrt.so \
        /opt/far3d/lib/python3.8/site-packages/tensorrt/tensorrt.so

RUN env -u PIP_CONSTRAINT /opt/far3d/bin/python -m pip install --no-cache-dir \
        "setuptools<81" && \
    env -u PIP_CONSTRAINT /opt/far3d/bin/python -m pip install --no-cache-dir \
        --no-build-isolation \
        mmdet3d==1.0.0rc6

COPY . /opt/modelopt
RUN env -u PIP_CONSTRAINT python -m pip install --no-cache-dir \
        lief \
        ml_dtypes \
        omegaconf \
        "onnx-graphsurgeon>=0.6.1" \
        "onnx~=1.21.0" \
        "onnxruntime-gpu~=1.24.2" \
        onnxscript \
        "onnxslim>=0.1.76" \
        "setuptools>=80" && \
    env -u PIP_CONSTRAINT python -m pip install --no-cache-dir --no-deps -e /opt/modelopt
