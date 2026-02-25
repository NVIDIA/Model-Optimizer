# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""LTX-2 quantization plugin."""

import contextlib

import torch

from modelopt.torch.quantization.nn.modules.quant_linear import _QuantLinear
from modelopt.torch.quantization.nn.modules.quant_module import QuantModuleRegistry
from modelopt.torch.quantization.utils import is_torch_export_mode

_FP8_DTYPES = tuple(
    dtype
    for dtype_name in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz")
    if (dtype := getattr(torch, dtype_name, None)) is not None
)


def _upcast_fp8_weight(
    weight: torch.Tensor, target_dtype: torch.dtype, seed: int = 0
) -> torch.Tensor:
    if target_dtype is torch.bfloat16:
        try:
            from ltx_core.loader.fuse_loras import fused_add_round_launch

            return fused_add_round_launch(
                torch.zeros_like(weight, dtype=target_dtype),
                weight,
                seed,
            )
        except Exception:
            pass
    return weight.to(target_dtype)


class _QuantLTX2Linear(_QuantLinear):
    """Quantized Linear with FP8 upcast before weight quantization."""

    @staticmethod
    def _get_quantized_weight(module: "_QuantLTX2Linear", weight: torch.Tensor) -> torch.Tensor:
        if _FP8_DTYPES and weight.dtype in _FP8_DTYPES:
            weight = _upcast_fp8_weight(weight, torch.bfloat16, 0)
        if module._enable_weight_quantization or is_torch_export_mode():
            return module.weight_quantizer(weight)
        return weight


def register_ltx2_quant_linear() -> None:
    """Register the LTX-2 quantized Linear, overriding the default mapping."""
    with contextlib.suppress(KeyError):
        QuantModuleRegistry.unregister(torch.nn.Linear)
    QuantModuleRegistry.register({torch.nn.Linear: "nn.Linear"})(_QuantLTX2Linear)
