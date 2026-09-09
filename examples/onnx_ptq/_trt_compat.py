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

from collections.abc import Iterable
from pathlib import Path

import onnx

from modelopt.onnx.quantization.ort_utils import _check_for_trtexec

_DYNAMIC_NVFP4_OP = "TRT_FP4DynamicQuantize"
_DYNAMIC_NVFP4_MIN_TRT_VERSION = "11.0"
_DYNAMIC_NVFP4_AUTO_FORMATS = {"nvfp4_awq_lite"}
_DYNAMIC_NVFP4_TRT_ERROR = (
    "Dynamic NVFP4 (W4A4) TensorRT engine builds require TensorRT 11.0 or newer. "
    "Upgrade TensorRT, or re-export with `--qformat=fp8` and without `--recipe`. "
    "ONNX export without `--trt_build` remains supported."
)


def request_needs_dynamic_nvfp4_check(
    qformat: str,
    auto_quantization_formats: list[str],
    *,
    recipe_provided: bool,
    trt_build: bool,
) -> bool:
    if not trt_build or recipe_provided:
        return False
    return qformat == "nvfp4" or (
        qformat == "auto"
        and bool(_DYNAMIC_NVFP4_AUTO_FORMATS.intersection(auto_quantization_formats))
    )


def _nodes_use_dynamic_nvfp4(nodes: Iterable[onnx.NodeProto]) -> bool:
    for node in nodes:
        if node.op_type == _DYNAMIC_NVFP4_OP:
            return True
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                nested_graphs = (attribute.g,)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                nested_graphs = attribute.graphs
            else:
                continue
            if any(_nodes_use_dynamic_nvfp4(nested.node) for nested in nested_graphs):
                return True
    return False


def onnx_uses_dynamic_nvfp4(onnx_path: str | Path) -> bool:
    model = onnx.load(str(onnx_path), load_external_data=False)
    return _nodes_use_dynamic_nvfp4(model.graph.node) or any(
        _nodes_use_dynamic_nvfp4(function.node) for function in model.functions
    )


def check_dynamic_nvfp4_trt_support() -> None:
    try:
        _check_for_trtexec(min_version=_DYNAMIC_NVFP4_MIN_TRT_VERSION)
    except ImportError as error:
        raise ImportError(f"{_DYNAMIC_NVFP4_TRT_ERROR} ({error})") from error
