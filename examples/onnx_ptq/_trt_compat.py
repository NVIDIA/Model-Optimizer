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

from pathlib import Path

import onnx

from modelopt.onnx.quantization.ort_utils import _check_for_trtexec
from modelopt.onnx.utils import has_node_op_type

DYNAMIC_NVFP4_OP = "TRT_FP4DynamicQuantize"
DYNAMIC_NVFP4_MIN_TRT_VERSION = "11.0"
W4A16_NVFP4_RECIPE = "w4a16_nvfp4"

_DYNAMIC_NVFP4_FORMATS = {"nvfp4", "nvfp4_awq_lite"}
_DYNAMIC_NVFP4_TRT_ERROR = (
    "Dynamic NVFP4 (W4A4) TensorRT engine builds require TensorRT 11.0 or newer. "
    "Upgrade TensorRT, or re-export with "
    "`--quantize_mode=nvfp4 --recipe=w4a16_nvfp4` to use the weight-only NVFP4 "
    "recipe on TensorRT 10.16."
)


def request_uses_dynamic_nvfp4(
    quantize_mode: str, recipe: str | None, auto_quantization_formats: list[str]
) -> bool:
    """Return whether the requested quantization can emit dynamic NVFP4 activations."""
    if quantize_mode == "nvfp4":
        return recipe != W4A16_NVFP4_RECIPE
    return quantize_mode == "auto" and bool(
        _DYNAMIC_NVFP4_FORMATS.intersection(auto_quantization_formats)
    )


def onnx_uses_dynamic_nvfp4(onnx_path: str | Path) -> bool:
    """Return whether an ONNX graph contains dynamic NVFP4 activation quantization."""
    model = onnx.load(str(onnx_path), load_external_data=False)
    return has_node_op_type(model.graph, DYNAMIC_NVFP4_OP)


def check_dynamic_nvfp4_trt_support() -> None:
    """Require a TensorRT release that reliably compiles dynamic NVFP4 graphs."""
    try:
        _check_for_trtexec(min_version=DYNAMIC_NVFP4_MIN_TRT_VERSION)
    except ImportError as e:
        raise ImportError(f"{_DYNAMIC_NVFP4_TRT_ERROR} ({e})") from e
