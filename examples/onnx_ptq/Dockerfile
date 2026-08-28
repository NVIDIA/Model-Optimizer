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

ARG TORCH_CUDA_ARCH_LIST=8.9+PTX

COPY examples/onnx_ptq/install-evaluator.sh \
    examples/onnx_ptq/mmcv-torch2.patch \
    examples/onnx_ptq/mmdet3d-runtime.patch \
    examples/onnx_ptq/requirements-25.06.txt \
    /tmp/evaluator/
RUN bash /tmp/evaluator/install-evaluator.sh "${TORCH_CUDA_ARCH_LIST}"

COPY examples/onnx_ptq/far3d/requirements.txt /tmp/evaluator/requirements-far3d.txt
ARG EXAMPLE
# AV2 declares development-only tooling and a second OpenCV distribution.
RUN case "${EXAMPLE}" in \
        far3d) python -m pip install --no-cache-dir \
            -r /tmp/evaluator/requirements-far3d.txt && \
            python -m pip install --no-cache-dir --no-deps av2==0.2.1 ;; \
        petr) ;; \
        *) echo "EXAMPLE must be far3d or petr" >&2; exit 1 ;; \
    esac

COPY . /opt/Model-Optimizer
RUN env -u PIP_CONSTRAINT python -m pip install --no-cache-dir --no-deps \
    -e /opt/Model-Optimizer
