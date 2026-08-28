#!/usr/bin/env bash
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

set -euo pipefail

prefix=$1
cuda_arch_list=$2
shift 2
python="${prefix}/bin/python"

unset PIP_CONSTRAINT
"${python}" -m pip install --no-cache-dir \
    -r /tmp/evaluator/requirements-torch-cu129.txt
"${python}" -m pip install --no-cache-dir \
    -r /tmp/evaluator/requirements-evaluator-cu129.txt "$@"

git clone --branch v1.7.0 --depth 1 https://github.com/open-mmlab/mmcv.git /tmp/mmcv
test "$(git -C /tmp/mmcv rev-parse HEAD)" = 270c293c9b4bfa90211ccab212d55ccd27bdc09f
git -C /tmp/mmcv apply --unidiff-zero /tmp/evaluator/mmcv-torch2.patch
MMCV_WITH_OPS=1 FORCE_CUDA=1 MAX_JOBS=8 TORCH_CUDA_ARCH_LIST="${cuda_arch_list}" \
    "${python}" -m pip install --no-cache-dir --no-build-isolation --no-deps /tmp/mmcv

git clone --branch v1.0.0rc6 --depth 1 \
    https://github.com/open-mmlab/mmdetection3d.git /tmp/mmdetection3d
test "$(git -C /tmp/mmdetection3d rev-parse HEAD)" = \
    47285b3f1e9dba358e98fcd12e523cfd0769c876
git -C /tmp/mmdetection3d apply --unidiff-zero /tmp/evaluator/mmdet3d-runtime.patch
"${python}" -m pip install --no-cache-dir --no-build-isolation /tmp/mmdetection3d
"${python}" -m pip check

site_packages=$("${python}" -c 'import site; print(site.getsitepackages()[0])')
ln -s tensorrt_bindings "${site_packages}/tensorrt"
rm -rf /tmp/mmcv /tmp/mmdetection3d
