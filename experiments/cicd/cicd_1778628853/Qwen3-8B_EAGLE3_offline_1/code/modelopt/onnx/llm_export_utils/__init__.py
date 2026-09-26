# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Deprecated shim for the legacy ``modelopt.onnx.llm_export_utils`` package.

The in-repo LLM ONNX export pipeline (formerly ``examples/torch_onnx/llm_export.py``
plus this package) was removed in 0.44.0rc1 in favor of
`TensorRT-Edge-LLM <https://github.com/NVIDIA/TensorRT-Edge-LLM>`_, which provides
a more complete and actively maintained pipeline.

This package is preserved only as a compatibility shim so external consumers that
still import ``modelopt.onnx.llm_export_utils`` (notably TensorRT-Edge-LLM 0.6.1
and earlier) continue to work. It will be removed in a future release.

New code should migrate to:

* ``modelopt.onnx.export`` — quant exporters (``FP8QuantExporter``, ``NVFP4QuantExporter``, etc.)
* ``modelopt.onnx.graph_surgery`` — graph transforms (GQA replacement, BF16 conversion, etc.)
* `TensorRT-Edge-LLM <https://github.com/NVIDIA/TensorRT-Edge-LLM>`_ — end-to-end LLM export.
"""

import warnings

warnings.warn(
    "modelopt.onnx.llm_export_utils is deprecated and will be removed in a future "
    "release. Use modelopt.onnx.export and modelopt.onnx.graph_surgery, or migrate "
    "to TensorRT-Edge-LLM (https://github.com/NVIDIA/TensorRT-Edge-LLM).",
    DeprecationWarning,
    stacklevel=2,
)
