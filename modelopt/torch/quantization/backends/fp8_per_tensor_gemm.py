# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""This module provides a GEMM function for fp8 per tensor quantization."""

from typing import Any

import torch
from torch.autograd import Function

from modelopt.torch.quantization.backends.gemm_registry import gemm_registry
from modelopt.torch.quantization.config import FP8_DEFAULT_CFG
from modelopt.torch.quantization.nn.modules.quant_linear import RealQuantLinear
from modelopt.torch.quantization.qtensor import FP8QTensor, QTensorWrapper
from modelopt.torch.quantization.utils import reduce_amax

from .utils import fp8_compatible

FP8_MIN = torch.finfo(torch.float8_e4m3fn).min
FP8_MAX = torch.finfo(torch.float8_e4m3fn).max


@torch.compile(dynamic=True)
def _to_fp8(x, scale):
    return (x / scale).clamp(FP8_MIN, FP8_MAX).to(torch.float8_e4m3fn)


@torch.compile(dynamic=True)
def _fp8_gemm_impl(input, weight_fp8, scale_a, scale_b, bias=None):
    input_shape = input.shape
    input_fp8 = _to_fp8(input, scale_a).reshape(-1, input_shape[-1])
    weight_fp8_t = weight_fp8.reshape(-1, weight_fp8.shape[-1]).t()
    output = torch._scaled_mm(
        input_fp8,
        weight_fp8_t,
        scale_a=scale_a,
        scale_b=scale_b,
        bias=bias,
        out_dtype=input.dtype,
        use_fast_accum=True,
    )
    return output.reshape(*input_shape[:-1], output.shape[-1])


def fp8_per_tensor_gemm(quant_module, input, bias=None):
    """GEMM function for fp8 per tensor quantization."""
    cached_scale_a = (
        hasattr(quant_module, "_scale_a") and quant_module.input_quantizer.amax is not None
    )

    if not cached_scale_a:
        input_amax = quant_module.input_quantizer.amax
        if input_amax is None:
            input_amax = reduce_amax(input)
        assert input_amax != 0
        quant_module._scale_a = (input_amax.float() / 448.0).to(device=input.device)

    cached_scale_b = (
        hasattr(quant_module, "_scale_b") and quant_module.weight_quantizer.amax is not None
    )

    if not cached_scale_b:
        weight_amax = quant_module.weight_quantizer.amax
        if weight_amax is None:
            weight_amax = reduce_amax(quant_module.weight)
        assert weight_amax != 0
        quant_module._scale_b = (weight_amax.float() / 448.0).to(device=quant_module.weight.device)

    if quant_module.weight.dtype != torch.float8_e4m3fn:
        weight_fp8 = _to_fp8(quant_module.weight, quant_module._scale_b)
    else:
        weight_fp8 = quant_module.weight.data

    output = _fp8_gemm_impl(
        input,
        weight_fp8,
        scale_a=quant_module._scale_a,
        scale_b=quant_module._scale_b,
        bias=bias if input.dtype != torch.float32 else None,
    )
    # _scaled_mm does not support bias for float32 input, so we add it manually
    if input.dtype == torch.float32 and bias is not None:
        output += bias
    return output


def _fp8_availability_check(module, input, args, kwargs):
    """Comprehensive check for FP8 GEMM availability."""
    # Quantizer configs
    quant_cfg: dict[str, Any] = FP8_DEFAULT_CFG["quant_cfg"]
    input_cfg = quant_cfg["*input_quantizer"]
    weight_cfg = quant_cfg["*weight_quantizer"]

    # Check hardware support
    if not torch.cuda.is_available() or not fp8_compatible():
        return False

    # Check module type
    if not isinstance(module, RealQuantLinear):
        return False

    # Check quantizer presence and configuration
    if not hasattr(module, "input_quantizer") or not hasattr(module, "weight_quantizer"):
        return False

    # Check input quantizer config
    for key, value in input_cfg.items():
        if (
            not hasattr(module.input_quantizer, key)
            or getattr(module.input_quantizer, key) != value
        ):
            return False

    # Check weight quantizer config
    for key, value in weight_cfg.items():
        if (
            not hasattr(module.weight_quantizer, key)
            or getattr(module.weight_quantizer, key) != value
        ):
            return False

    return True


class Fp8PerTensorLinear(Function):
    """Linear layer with FP8 per tensor quantization."""

    @staticmethod
    def forward(
        ctx,
        quant_module,
        input_tensor,
        weight,
        bias=None,
        allreduce_dgrad=False,
        tp_group=None,
    ):
        """Forward method."""
        ctx.save_for_backward(
            input_tensor if weight.requires_grad else None,
            weight if input_tensor.requires_grad else None,
            getattr(quant_module.weight_quantizer, "_scale", None),
        )
        ctx.compute_bias_grad = bias is not None and bias.requires_grad
        ctx.block_sizes = getattr(quant_module.weight_quantizer, "_block_sizes", None)

        ctx.allreduce_dgrad = allreduce_dgrad
        ctx.tp_group = tp_group

        ret = fp8_per_tensor_gemm(quant_module, input_tensor, bias)
        return ret

    @staticmethod
    def backward(ctx, grad_outputs):
        """Backward method.

        For input, we will save the unquantized input and use it directly to compute the weight
        gradient. For weight, if the weight is compressed, we will save the quantized weight and
        dequantize it to compute the input gradient. If the weight is not compressed, we will save
        the unquantized weight and use it directly to compute the input gradient.
        """
        input_tensor, weight, scale = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        if weight is not None:
            if isinstance(weight, QTensorWrapper):
                weight = weight.get_qtensor()
                assert isinstance(weight, FP8QTensor)
                weight = weight.dequantize(scale=scale, block_sizes=ctx.block_sizes)
            grad_input = grad_outputs @ weight
        if input_tensor is not None:
            grad_weight = grad_outputs.reshape(-1, grad_outputs.shape[-1]).T @ input_tensor.reshape(
                -1, input_tensor.shape[-1]
            )
        if ctx.compute_bias_grad:
            # Sum all dimensions except the last one
            grad_bias = grad_outputs.sum(dim=list(range(grad_outputs.dim() - 1)))

        if ctx.allreduce_dgrad:
            # All-reduce. Note: here async and sync are effectively the same.
            torch.distributed.all_reduce(grad_input, group=ctx.tp_group)

        return None, grad_input, grad_weight, grad_bias, None, None

    @classmethod
    def apply(cls, *args, **kwargs):
        """Get rid of kwargs because super does not support kwargs."""
        additional_args = tuple(kwargs.values())
        return super().apply(*args, *additional_args)


# Register default implementations
gemm_registry.register(
    gemm_func=Fp8PerTensorLinear.apply,
    availability_check=_fp8_availability_check,
)
