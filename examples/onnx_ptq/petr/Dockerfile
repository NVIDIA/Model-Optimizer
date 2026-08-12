# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM nvcr.io/nvidia/pytorch:26.07-py3

ENV LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

ENV UV_PYTHON_INSTALL_DIR=/opt/python

RUN python -m pip install --no-cache-dir uv && \
    uv python install 3.8 && \
    uv venv --seed --python 3.8 /opt/petr

COPY examples/onnx_ptq/petr/requirements*.txt /tmp/petr-requirements/
RUN env -u PIP_CONSTRAINT /opt/petr/bin/python -m pip install --no-cache-dir \
        -r /tmp/petr-requirements/requirements-torch.txt && \
    env -u PIP_CONSTRAINT /opt/petr/bin/python -m pip install --no-cache-dir \
        -r /tmp/petr-requirements/requirements.txt && \
    env -u PIP_CONSTRAINT /opt/petr/bin/python -m pip install --no-cache-dir \
        --no-build-isolation \
        -r /tmp/petr-requirements/requirements-mmdet3d.txt && \
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
