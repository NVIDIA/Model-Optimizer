# Adapted from https://github.com/pytorch/pytorch/blob/e48ee2cf50d86d87ef7c7d0839267dbed4903ebf/torch/onnx/symbolic_opset14.py#L137-L207

# From PyTorch:

# Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
# Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
# Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
# Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
# Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
# Copyright (c) 2011-2013 NYU                      (Clement Farabet)
# Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
# Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
# Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

# From Caffe2:

# Copyright (c) 2016-present, Facebook Inc. All rights reserved.

# All contributions by Facebook:
# Copyright (c) 2016 Facebook Inc.

# All contributions by Google:
# Copyright (c) 2015 Google Inc.
# All rights reserved.

# All contributions by Yangqing Jia:
# Copyright (c) 2015 Yangqing Jia
# All rights reserved.

# All contributions by Kakao Brain:
# Copyright 2019-2020 Kakao Brain

# All contributions by Cruise LLC:
# Copyright (c) 2022 Cruise LLC.
# All rights reserved.

# All contributions by Tri Dao:
# Copyright (c) 2024 Tri Dao.
# All rights reserved.

# All contributions by Arm:
# Copyright (c) 2021, 2023-2024 Arm Limited and/or its affiliates

# All contributions from Caffe:
# Copyright(c) 2013, 2014, 2015, the respective contributors
# All rights reserved.

# All other contributions:
# Copyright(c) 2015, 2016 the respective contributors
# All rights reserved.

# Caffe2 uses a copyright model similar to Caffe: each contributor holds
# copyright over their contributions to Caffe2. The project versioning records
# all such contribution and copyright details. If a contributor wants to further
# mark their specific copyright on a particular contribution, they should
# indicate their copyright solely in the commit message of the change when it is
# committed.

# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.

# 3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
#    and IDIAP Research Institute nor the names of its contributors may be
#    used to endorse or promote products derived from this software without
#    specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

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

"""Utility to export a quantized torch model to quantized ONNX."""

import contextlib
from typing import TYPE_CHECKING

import onnx
import torch
from torch.onnx import symbolic_helper
from torch.onnx import symbolic_helper as sym_help

if TYPE_CHECKING:
    if hasattr(torch.onnx._internal, "jit_utils"):
        from torch.onnx._internal.jit_utils import GraphContext
    else:  # torch >= 2.9
        from torch.onnx._internal.torchscript_exporter.jit_utils import GraphContext

onnx_dtype_map = {
    "BFloat16": onnx.TensorProto.BFLOAT16,
    "Float": onnx.TensorProto.FLOAT,
    "Float8": onnx.TensorProto.FLOAT8E4M3FN,
    "Half": onnx.TensorProto.FLOAT16,
    "INT8": onnx.TensorProto.INT8,
    "UINT8": onnx.TensorProto.UINT8,
}
mha_valid_precisions = {"Half", "BFloat16"}

torch_dtype_map = {"Float": torch.float32, "Half": torch.float16, "BFloat16": torch.bfloat16}


def export_int8(
    g: "GraphContext",
    inputs: torch.Value,
    amax: torch.Tensor,
    num_bits: int,
    unsigned: bool,
    narrow_range: bool,
    trt_high_precision_dtype: str | None,
):
    """Export quantized model to INT8 ONNX."""
    assert num_bits == 8, "Number of bits must be 8 for INT8 ONNX export."
    output_shape = sym_help._get_tensor_sizes(inputs)
    maxbound = (1 << (num_bits - 1 + int(unsigned))) - 1

    input_type = inputs.type().scalarType()
    if trt_high_precision_dtype is None:
        trt_high_precision_dtype = input_type

    if amax.numel() == 1:
        zero_point, axis = torch.tensor(0.0, device=amax.device), None
    else:
        amax_init_shape = amax.shape
        amax = amax.squeeze().data
        assert len(amax.shape) == 1, "ONNX does not support multi-axis quantization."
        zero_point = torch.zeros_like(amax, dtype=torch.int32).data
        axis = list(amax_init_shape).index(next(iter(amax.shape)))

    zero_point = g.op("Constant", value_t=zero_point.to(torch_dtype_map[trt_high_precision_dtype]))

    if not unsigned:
        assert not narrow_range, "ONNX does not support unsigned narrow range INT8."
        zero_point = g.op("Cast", zero_point, to_i=onnx_dtype_map["INT8"])
    else:
        zero_point = g.op("Cast", zero_point, to_i=onnx_dtype_map["UINT8"])

    amax = amax.to(torch_dtype_map[trt_high_precision_dtype])
    scale = amax / maxbound
    scale.masked_fill_(scale == 0, 1.0)
    scale = g.op("Constant", value_t=scale)

    assert trt_high_precision_dtype in (input_type, "Float", "BFloat16"), (
        "TRT StronglyType requires both weights and amax to be in the BF16/FP16, or the QDQ in Float."
    )

    # custom ops, so cast the input if needed.
    if trt_high_precision_dtype != input_type:
        inputs = g.op("Cast", inputs, to_i=onnx_dtype_map[trt_high_precision_dtype])
    quantized = g.op("QuantizeLinear", inputs, scale, zero_point, axis_i=axis)
    out = g.op("DequantizeLinear", quantized, scale, zero_point, axis_i=axis).setType(
        inputs.type().with_dtype(torch_dtype_map[trt_high_precision_dtype]).with_sizes(output_shape)
    )

    # custom ops, so cast the output if needed.
    if trt_high_precision_dtype != input_type:
        inputs = g.op("Cast", inputs, to_i=onnx_dtype_map[input_type])

    return out


def export_int4(
    g: "GraphContext",
    inputs: torch.Value,
    amax: torch.Tensor,
    num_bits: int,
    trt_high_precision_dtype: str | None,
    block_size: int,
    axis: int,
):
    """Export quantized model to INT4 ONNX."""
    assert num_bits == 4, "Number of bits must be 4 for INT4 ONNX export."
    scale_inv = amax / 7.0
    scale_inv_op = g.op("Constant", value_t=scale_inv)
    otype = inputs.type().scalarType()
    output_shape = sym_help._get_tensor_sizes(inputs)
    if trt_high_precision_dtype is None:
        trt_high_precision_dtype = otype
    return g.op(
        "trt::DequantizeLinear", inputs, scale_inv_op, axis_i=axis, block_size_i=block_size
    ).setType(
        inputs.type().with_dtype(torch_dtype_map[trt_high_precision_dtype]).with_sizes(output_shape)
    )


def _fp8_quantize(
    g: "GraphContext",
    inputs: torch.Value,
    scale_inv: float,
    trt_high_precision_dtype: str,
):
    """Helper Function for Quantization."""
    output_shape = sym_help._get_tensor_sizes(inputs)

    # TRT StronglyType only supports FP16 QDQs
    # custom ops, so cast the input if needed.
    input_type = inputs.type().scalarType()
    assert trt_high_precision_dtype in (input_type, "Float"), (
        "TRT StronglyType requires both weights and amax to be in the BF16/FP16, or the QDQ in Float."
    )
    if trt_high_precision_dtype != input_type:
        inputs = g.op("Cast", inputs, to_i=onnx_dtype_map[trt_high_precision_dtype])

    scale = g.op(
        "Constant",
        value_t=torch.tensor(scale_inv).to(torch_dtype_map[trt_high_precision_dtype]),
    )
    q_op = g.op("trt::TRT_FP8QuantizeLinear", inputs, scale).setType(
        inputs.type().with_dtype(torch.uint8).with_sizes(output_shape)
    )
    return q_op


def _fp8_dequantize(
    g: "GraphContext",
    inputs: torch.Value,
    scale_inv: float,
    trt_high_precision_dtype: str,
    otype: str | None = None,
):
    """Helper Function for Dequantization."""
    output_shape = sym_help._get_tensor_sizes(inputs)
    assert trt_high_precision_dtype in (otype, "Float"), (
        "TRT StronglyType requires both weights and amax to be in the BF16/FP16, or the QDQ in Float."
    )
    scale = g.op(
        "Constant",
        value_t=torch.tensor(scale_inv, dtype=torch_dtype_map[otype]),  # type: ignore[index]
    )
    out = g.op("trt::TRT_FP8DequantizeLinear", inputs, scale).setType(
        inputs.type().with_dtype(torch_dtype_map[trt_high_precision_dtype]).with_sizes(output_shape)
    )

    # DQ outputs are currently constrained to FP32 due to a similar limitation in ORT
    # custom ops, so cast the output if needed.
    if trt_high_precision_dtype != otype:
        out = g.op("Cast", out, to_i=onnx_dtype_map[otype])  # type: ignore[index]
    return out


def export_fp8(
    g: "GraphContext",
    inputs: torch.Value,
    amax: float,
    trt_high_precision_dtype: str | None,
):
    """Export quantized model to FP8 ONNX."""
    scale = 1.0 if amax is None else 448.0 / float(amax)
    otype = inputs.type().scalarType()
    if trt_high_precision_dtype is None:
        trt_high_precision_dtype = otype

    q_tensor = _fp8_quantize(g, inputs, 1.0 / scale, trt_high_precision_dtype)
    return _fp8_dequantize(g, q_tensor, 1.0 / scale, trt_high_precision_dtype, otype)


def scaled_dot_product_attention(
    g: "GraphContext",
    query: "torch._C.Value",
    key: "torch._C.Value",
    value: "torch._C.Value",
    attn_mask: "torch._C.Value | None" = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: "torch._C.Value | None" = None,
    enable_gqa: bool = False,
):
    """Perform scaled dot product attention."""
    if hasattr(torch.onnx, "_type_utils"):
        from torch.onnx._type_utils import JitScalarType
    else:  # torch >= 2.9
        from torch.onnx._internal.torchscript_exporter import JitScalarType

    if hasattr(torch.onnx, "symbolic_opset14"):
        from torch.onnx.symbolic_opset14 import _attention_scale, _causal_attention_mask
    else:  # torch >= 2.9
        from torch.onnx._internal.torchscript_exporter.symbolic_opset14 import (
            _attention_scale,
            _causal_attention_mask,
        )

    assert (not is_causal) or (is_causal and symbolic_helper._is_none(attn_mask)), (
        "is_causal and attn_mask cannot be set at the same time"
    )
    assert not enable_gqa, (
        "conversion of scaled_dot_product_attention not implemented if enable_gqa is True"
    )

    if symbolic_helper._is_none(scale):
        scale = _attention_scale(g, query)

    if is_causal:
        attn_mask = _causal_attention_mask(g, query, key)

    # Swap the last two axes of key
    # NOTE: onnx-script has different logic here, because the attribute perms in
    # transpose needs list of ints
    key_shape_builtin = symbolic_helper._get_tensor_rank(key)
    key_transposed_axes = list(range(key_shape_builtin))
    key_transposed_axes[-1], key_transposed_axes[-2] = (
        key_transposed_axes[-2],
        key_transposed_axes[-1],
    )
    key_transposed = g.op("Transpose", key, perm_i=key_transposed_axes)

    # https://github.com/pytorch/pytorch/blob/12da0c70378b5be9135c6fda62a9863bce4a4818/aten/src/ATen/native/transformers/attention.cpp#L653
    # Scale q, k before matmul for stability see https://tinyurl.com/sudb9s96 for math
    query_scaled = g.op("Mul", query, g.op("Sqrt", scale))
    key_transposed_scaled = g.op("Mul", key_transposed, g.op("Sqrt", scale))
    mul_qk = g.op("MatMul", query_scaled, key_transposed_scaled)

    if symbolic_helper._is_none(attn_mask):
        mul_qk_add = mul_qk
    elif JitScalarType.from_value(attn_mask) == JitScalarType.BOOL:
        # Turn the Boolean mask to float: attn_mask.masked_fill(not attn_mask, -float('inf'))
        const_zero = g.op("Constant", value_t=torch.tensor([0.0]))
        const_neg_inf = g.op("Constant", value_t=torch.tensor([-float("inf")]))
        attn_mask = g.op("Where", attn_mask, const_zero, const_neg_inf)
        mul_qk_add = g.op("Add", mul_qk, attn_mask)
    elif JitScalarType.from_value(attn_mask) in (
        JitScalarType.FLOAT,
        JitScalarType.HALF,
        JitScalarType.BFLOAT16,
    ):
        mul_qk_add = g.op("Add", mul_qk, attn_mask)
    else:
        raise ValueError(f"Unsupported type for attn_mask: {JitScalarType.from_value(attn_mask)}")

    attn_weight = g.op("Softmax", mul_qk_add, axis_i=-1)

    if dropout_p != 0:
        attn_weight = g.op(
            "Dropout",
            attn_weight,
            g.op("Constant", value_t=torch.tensor(dropout_p, dtype=torch.float)),
        )

    return g.op("MatMul", attn_weight, value)


def export_fp8_mha(
    g: "GraphContext",
    query: "torch._C.Value",
    key: "torch._C.Value",
    value: "torch._C.Value",
    attn_mask: "torch._C.Value | None" = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: "torch._C.Value | None" = None,
    q_quantized_scale: float = 1.0,
    k_quantized_scale: float = 1.0,
    v_quantized_scale: float = 1.0,
    high_precision_flag: str = "Half",
    disable_fp8_mha: bool = True,
):
    r"""Export quantized fMHA to FP8 ONNX.

    FP8 ONNX graph::

        Q           K          V
        |           |          |
        \          /           |
        QDQ      QDQ           |
          \      /             |
         Cast   Cast           |
           \    /              |
            BMM1               |
             \                 |
            Cast              QDQ
               \               |
              SoftMax          |
                 |             |
                QDQ            |
                  \            |
                   Cast      Cast
                       \     /
                        BMM2
                         |
                        Cast
    """
    if hasattr(torch.onnx, "_type_utils"):
        from torch.onnx._type_utils import JitScalarType
    else:  # torch >= 2.9
        from torch.onnx._internal.torchscript_exporter._type_utils import JitScalarType

    if hasattr(torch.onnx, "symbolic_opset14"):
        from torch.onnx.symbolic_opset14 import _attention_scale, _causal_attention_mask
    else:  # torch >= 2.9
        from torch.onnx._internal.torchscript_exporter.symbolic_opset14 import (
            _attention_scale,
            _causal_attention_mask,
        )

    # Pass all arguments, including x, to the custom ONNX operator
    assert (not is_causal) or (is_causal and sym_help._is_none(attn_mask)), (
        "is_causal and attn_mask cannot be set at the same time"
    )

    scale = sym_help._maybe_get_const(scale, "f")
    if sym_help._is_none(scale):
        scale = _attention_scale(g, query)

    if is_causal:
        attn_mask = _causal_attention_mask(g, query, key)

    # Swap the last two axes of key
    # NOTE: onnx-script has different logic here, because the attribute perms in
    # transpose needs list of ints
    key_shape_builtin = sym_help._get_tensor_rank(key)
    key_transposed_axes = list(range(key_shape_builtin))
    key_transposed_axes[-1], key_transposed_axes[-2] = (
        key_transposed_axes[-2],
        key_transposed_axes[-1],
    )
    key_transposed = g.op("Transpose", key, perm_i=key_transposed_axes)

    # https://github.com/pytorch/pytorch/blob/12da0c70378b5be9135c6fda62a9863bce4a4818/aten/src/ATen/native/transformers/attention.cpp#L653
    # Scale q, k before matmul for stability see https://tinyurl.com/sudb9s96 for math
    query_scaled = g.op("Mul", query, g.op("Sqrt", scale))
    key_transposed_scaled = g.op("Mul", key_transposed, g.op("Sqrt", scale))
    if not disable_fp8_mha:
        if high_precision_flag not in mha_valid_precisions:
            raise ValueError(
                "The Quantized config setting doesn't match TRT's fusion pattern; the qdqs must be in 16 bits."
            )
        q_input_dtype = query.type().scalarType()
        k_input_dtype = key.type().scalarType()
        v_input_dtype = value.type().scalarType()
        if {q_input_dtype, k_input_dtype, v_input_dtype} != {high_precision_flag}:
            raise ValueError("The quantized MHA must have 16-bit inputs.")
        query_scaled = export_fp8(g, query_scaled, q_quantized_scale, high_precision_flag)
        query_scaled = g.op("Cast", query_scaled, to_i=onnx_dtype_map["Float"])
        key_transposed_scaled = export_fp8(
            g, key_transposed_scaled, k_quantized_scale, high_precision_flag
        )
        key_transposed_scaled = g.op("Cast", key_transposed_scaled, to_i=onnx_dtype_map["Float"])
    mul_qk = g.op("MatMul", query_scaled, key_transposed_scaled)
    if not disable_fp8_mha:
        mul_qk = g.op("Cast", mul_qk, to_i=onnx_dtype_map[high_precision_flag])

    if sym_help._is_none(attn_mask):
        mul_qk_add = mul_qk
    elif JitScalarType.from_value(attn_mask) == JitScalarType.BOOL:
        # Turn the Boolean mask to float: attn_mask.masked_fill(not attn_mask, -float('inf'))
        const_zero = g.op("Constant", value_t=torch.tensor([0.0]))
        const_neg_inf = g.op("Constant", value_t=torch.tensor([-float("inf")]))
        attn_mask = g.op("Where", attn_mask, const_zero, const_neg_inf)
        mul_qk_add = g.op("Add", mul_qk, attn_mask)
    elif JitScalarType.from_value(attn_mask) in (
        JitScalarType.FLOAT,
        JitScalarType.HALF,
        JitScalarType.BFLOAT16,
    ):
        mul_qk_add = g.op("Add", mul_qk, attn_mask)
    else:
        raise ValueError(f"Unsupported type for attn_mask: {JitScalarType.from_value(attn_mask)}")

    attn_weight = g.op("Softmax", mul_qk_add, axis_i=-1)

    if not disable_fp8_mha:
        # Softmax's output scale is hard coded to 1.0
        attn_weight = export_fp8(g, attn_weight, 1.0, high_precision_flag)
        attn_weight = g.op("Cast", attn_weight, to_i=onnx_dtype_map["Float"])

    if dropout_p != 0:
        attn_weight = g.op(
            "Dropout",
            attn_weight,
            g.op("Constant", value_t=torch.tensor(dropout_p, dtype=torch.float)),
        )
    if not disable_fp8_mha:
        value = export_fp8(g, value, v_quantized_scale, high_precision_flag)
        value = g.op("Cast", value, to_i=onnx_dtype_map["Float"])
        return g.op(
            "Cast",
            g.op("MatMul", attn_weight, value),
            to_i=onnx_dtype_map[high_precision_flag],
        )
    else:
        return g.op("MatMul", attn_weight, value)


def _fp4_dynamic_quantize(
    g: "GraphContext",
    inputs: torch.Value,
    scale: float,
    trt_high_precision_dtype: str | None,
    block_size: int,
    axis: int = -1,
    scale_type: int = onnx_dtype_map["Float8"],
):
    """Helper Function for Dynamic Quantization."""
    # TRT StronglyType only supports FP16 QDQ ops, so cast the input if needed.
    input_type = inputs.type().scalarType()
    if trt_high_precision_dtype is None:
        trt_high_precision_dtype = input_type

    if trt_high_precision_dtype != input_type:
        inputs = g.op("Cast", inputs, to_i=onnx_dtype_map[trt_high_precision_dtype])

    scale = g.op(
        "Constant",
        value_t=torch.tensor(scale).to(torch_dtype_map["Float"]),
    )
    # This is a TensorRT local function, it dynamically quantizes the input tensor to FP4.
    xf4, sx_f8 = g.op(
        "trt::TRT_FP4DynamicQuantize",
        inputs,
        scale,
        axis_i=axis,
        block_size_i=block_size,
        scale_type_i=scale_type,
        outputs=2,
    )

    return xf4, sx_f8


def _fp4_dequantize(
    g: "GraphContext",
    inputs: torch.Value,
    scale: float | torch.Value,
    trt_high_precision_dtype: str | None,
):
    """Helper Function for Dequantization."""
    if isinstance(scale, float):
        scale = g.op(
            "Constant",
            value_t=torch.tensor(scale, dtype=torch_dtype_map["Float"]),
        )
    return g.op("DequantizeLinear", inputs, scale)


def _fp4_dequantize_2(
    g: "GraphContext",
    inputs: torch.Value,
    dyn_scale: torch.Value,
    block_size: int,
    axis: int = -1,
):
    """Helper Function for Dequantization."""
    return g.op("DequantizeLinear", inputs, dyn_scale, axis_i=axis, block_size_i=block_size)


def _mxfp8_dynamic_quantize(
    g: "GraphContext",
    inputs: torch.Value,
    block_size: int,
    axis: int = -1,
):
    x_f8, sx_ui8 = g.op(
        "trt::TRT_MXFP8DynamicQuantize",
        inputs,
        axis_i=axis,
        block_size_i=block_size,
        outputs=2,
        output_dtype_i=onnx_dtype_map["Float8"],
    )

    return x_f8, sx_ui8


def _mxfp8_dequantize(
    g: "GraphContext",
    inputs: torch.Value,
    scale: torch.Value,
    block_size: int,
    axis: int = -1,
    input_dtype: str = "Half",
):
    return g.op(
        "trt::TRT_MXFP8DequantizeLinear",
        inputs,
        scale,
        axis_i=axis,
        block_size_i=block_size,
        output_dtype_i=onnx_dtype_map[input_dtype],
    )


def export_mxfp8(
    g: "GraphContext",
    inputs: torch.Tensor,
    onnx_quantizer_type: str,
    block_size: int,
    axis: int = -1,
):
    """Export quantized model to MXFP8 ONNX."""
    input_dtype = inputs.type().scalarType()
    if onnx_quantizer_type == "dynamic":
        x_f8, sx_ui8 = _mxfp8_dynamic_quantize(g, inputs, block_size, axis=axis)

        return _mxfp8_dequantize(g, x_f8, sx_ui8, block_size, axis=axis, input_dtype=input_dtype)
    else:
        scale = torch.tensor(1.0, dtype=torch_dtype_map[input_dtype])
        return _mxfp8_dequantize(g, inputs, scale, block_size, axis=axis, input_dtype=input_dtype)


def export_fp4(
    g: "GraphContext",
    inputs: torch.Value,
    block_size: int,
    amax: torch.Value,
    num_bits: tuple[int, int],
    trt_high_precision_dtype: str | None,
    onnx_quantizer_type: str,
):
    """Export quantized model to FP4 ONNX."""
    if onnx_quantizer_type == "dynamic":
        amax = sym_help._get_const(amax, "f", "amax")
        if amax is None or amax == 0:
            sx_f32_per_tensor = 1.0
        else:
            assert num_bits == (2, 1)
            sx_f32_per_tensor = float(amax) / 6.0 / 448.0

        x_f4, sx_f8 = _fp4_dynamic_quantize(
            g, inputs, sx_f32_per_tensor, trt_high_precision_dtype, block_size
        )
        dq_scale = _fp4_dequantize(g, sx_f8, sx_f32_per_tensor, trt_high_precision_dtype)
        return _fp4_dequantize_2(g, x_f4, dq_scale, block_size)
    else:
        # This is a dummy custom op, we post-process the exported ONNX model to replace this node
        # with two DQ nodes as described in the static double quantization recipe.
        return g.op("trt::TRT_FP4QDQ", inputs, block_size_i=block_size)


@contextlib.contextmanager
def configure_linear_module_onnx_quantizers(model):
    """Sets the onnx export attributes for the given model."""
    for _, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            module.input_quantizer._onnx_quantizer_type = "dynamic"
            module.weight_quantizer._onnx_quantizer_type = "static"
    yield
