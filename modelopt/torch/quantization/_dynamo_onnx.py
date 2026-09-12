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

"""Private ONNXScript translations for quantized Dynamo export."""

import onnx
import onnxscript
import torch
from onnxscript.function_libs.torch_lib.tensor_typing import TFloat
from onnxscript.onnx_types import FLOAT4E2M1, FLOAT8E4M3FN

from .export_onnx import onnx_dtype_map

_OPSET = onnxscript.opset21
_TRT_OPSET = onnxscript.values.Opset(domain="trt", version=1)


@onnxscript.script(_TRT_OPSET)
def _fp8_qdq(inputs: TFloat, scale: TFloat) -> TFloat:
    quantized = _TRT_OPSET.TRT_FP8QuantizeLinear(inputs, scale)
    return _TRT_OPSET.TRT_FP8DequantizeLinear(quantized, scale)


@onnxscript.script(_TRT_OPSET)
def _int4_dq(inputs: TFloat, scale: TFloat, axis: int, block_size: int) -> TFloat:
    return _TRT_OPSET.DequantizeLinear(inputs, scale, axis=axis, block_size=block_size)


@onnxscript.script(_TRT_OPSET)
def _fp4_qdq(inputs: TFloat, block_size: int) -> TFloat:
    return _TRT_OPSET.TRT_FP4QDQ(inputs, block_size=block_size)


@onnxscript.script(_TRT_OPSET)
def _fp4_dynamic_quantize(
    inputs: TFloat, scale: TFloat, block_size: int
) -> tuple[FLOAT4E2M1[...], FLOAT8E4M3FN[...]]:
    quantized, dynamic_scale = _TRT_OPSET.TRT_FP4DynamicQuantize(
        inputs, scale, axis=-1, block_size=block_size, scale_type=17
    )
    return quantized, dynamic_scale


@onnxscript.script(_TRT_OPSET)
def _fp4_dq(inputs: FLOAT4E2M1[...], scale: TFloat, block_size: int) -> TFloat:
    return _TRT_OPSET.DequantizeLinear(inputs, scale, axis=-1, block_size=block_size)


@onnxscript.script(_TRT_OPSET)
def _mxfp8_dynamic_qdq(inputs: TFloat, block_size: int, output_dtype: int) -> TFloat:
    quantized, scale = _TRT_OPSET.TRT_MXFP8DynamicQuantize(
        inputs, axis=-1, block_size=block_size, output_dtype=17
    )
    return _TRT_OPSET.TRT_MXFP8DequantizeLinear(
        quantized,
        scale,
        axis=-1,
        block_size=block_size,
        output_dtype=output_dtype,
    )


@onnxscript.script(_TRT_OPSET)
def _mxfp8_static_dq(inputs: TFloat, scale: TFloat, block_size: int, output_dtype: int) -> TFloat:
    return _TRT_OPSET.TRT_MXFP8DequantizeLinear(
        inputs,
        scale,
        axis=-1,
        block_size=block_size,
        output_dtype=output_dtype,
    )


def _cast(inputs, dtype: int):
    return inputs if int(inputs.dtype) == dtype else _OPSET.Cast(inputs, to=dtype)


def _resolve_dtype(inputs, high_precision_dtype: str | None) -> int:
    return (
        int(inputs.dtype) if high_precision_dtype is None else onnx_dtype_map[high_precision_dtype]
    )


def _static_shape(value) -> list[int]:
    try:
        return [int(dim) for dim in value.shape]
    except (AttributeError, TypeError, ValueError):
        raise NotImplementedError("Dynamo ONNX export does not support dynamic shapes.") from None


def _block_activation_shape(
    inputs, block_size: int, quantizer_type: str | None
) -> list[int] | None:
    if quantizer_type != "dynamic":
        return None
    shape = _static_shape(inputs)
    if len(shape) not in (2, 3):
        raise NotImplementedError("Dynamo ONNX block activation export supports rank 2 or 3 only.")
    if shape[-1] % block_size:
        raise NotImplementedError(
            f"Dynamo ONNX block size {block_size} must divide the last dimension {shape[-1]}."
        )
    return shape


def _translate_quantize_op(
    inputs,
    amax,
    num_bits: int,
    exponent_bits: int,
    unsigned: bool,
    narrow_range: bool,
    high_precision_dtype: str | None = None,
    block_size: int | None = None,
    axis: int | None = None,
):
    if num_bits == 8 and exponent_bits == 4:
        if amax is None:
            scale = _OPSET.CastLike(1.0, inputs)
        elif any(dim != 1 for dim in _static_shape(amax)):
            raise AssertionError(
                "E4M3 supports ONNX export only for per-tensor quantization with scalar amax."
            )
        else:
            scale = _OPSET.CastLike(_OPSET.Div(amax, 448.0), inputs)
        return _fp8_qdq(inputs, scale)

    output_dtype = _resolve_dtype(inputs, high_precision_dtype)
    if num_bits == 8 and exponent_bits == 0:
        source_dtype = int(inputs.dtype)
        if not unsigned:
            assert not narrow_range, "ONNX does not support signed narrow-range INT8."
        assert output_dtype in (
            source_dtype,
            onnx.TensorProto.FLOAT,
            onnx.TensorProto.BFLOAT16,
        ), "TensorRT strongly typed mode requires Q/DQ in the input dtype, FP32, or BF16."
        inputs = _cast(inputs, output_dtype)
        quantized_axes = [index for index, dim in enumerate(_static_shape(amax)) if dim != 1]
        if len(quantized_axes) > 1:
            raise AssertionError("ONNX does not support multi-axis quantization.")
        quantized_axis = quantized_axes[0] if quantized_axes else None
        amax = _OPSET.Squeeze(_OPSET.Cast(amax, to=output_dtype))
        scale = _OPSET.Div(amax, float((1 << (7 + int(unsigned))) - 1))
        scale = _OPSET.Where(_OPSET.Equal(scale, 0.0), _OPSET.CastLike(1.0, scale), scale)
        zero_point_dtype = onnx.TensorProto.UINT8 if unsigned else onnx.TensorProto.INT8
        zero_point = _OPSET.Cast(_OPSET.Mul(amax, 0.0), to=zero_point_dtype)
        if quantized_axis is None:
            quantized = _OPSET.QuantizeLinear(inputs, scale, zero_point)
            output = _OPSET.DequantizeLinear(quantized, scale, zero_point)
        else:
            quantized = _OPSET.QuantizeLinear(inputs, scale, zero_point, axis=quantized_axis)
            output = _OPSET.DequantizeLinear(quantized, scale, zero_point, axis=quantized_axis)
        return output if output_dtype == source_dtype else _OPSET.Cast(output, to=source_dtype)

    if num_bits == 4 and exponent_bits == 0:
        if unsigned:
            raise NotImplementedError("Dynamo ONNX export supports signed INT4 only.")
        if block_size is None or axis is None:
            raise ValueError("INT4 ONNX export requires block_size and axis.")
        scale = _OPSET.Div(_OPSET.Cast(amax, to=output_dtype), 7.0)
        output = _int4_dq(inputs, scale, axis, block_size)
        return output if int(inputs.dtype) == output_dtype else _OPSET.Cast(output, to=output_dtype)

    raise NotImplementedError(
        f"Unsupported num_bits={num_bits}, exponent_bits={exponent_bits} for ONNX export."
    )


def _translate_dynamic_block_quantize_op(
    inputs,
    block_size: int,
    amax,
    num_bits: int,
    exponent_bits: int,
    scale_num_bits: int,
    scale_exponent_bits: int,
    high_precision_dtype: str | None = None,
    quantizer_type: str | None = None,
):
    activation_shape = _block_activation_shape(inputs, block_size, quantizer_type)
    format_bits = (num_bits, exponent_bits, scale_num_bits, scale_exponent_bits)

    if format_bits == (4, 2, 8, 4):
        if quantizer_type != "dynamic":
            return _OPSET.Identity(_fp4_qdq(inputs, block_size))
        assert activation_shape is not None
        output_dtype = _resolve_dtype(inputs, high_precision_dtype)
        inputs = _cast(inputs, output_dtype)
        if amax is None:
            scale = _OPSET.Constant(value_float=1.0)
        else:
            scale = _OPSET.Div(_OPSET.Cast(amax, to=onnx.TensorProto.FLOAT), 2688.0)
            scale = _OPSET.Where(_OPSET.Equal(scale, 0.0), _OPSET.CastLike(1.0, scale), scale)
        quantized, dynamic_scale = _fp4_dynamic_quantize(inputs, scale, block_size)
        quantized.dtype = onnxscript.ir.DataType.FLOAT4E2M1
        dynamic_scale.dtype = onnxscript.ir.DataType.FLOAT8E4M3FN
        quantized.shape = onnxscript.ir.Shape(activation_shape)
        scale_shape = [*activation_shape[:-1], activation_shape[-1] // block_size]
        dynamic_scale.shape = onnxscript.ir.Shape(scale_shape)
        dequantized_scale = _OPSET.DequantizeLinear(dynamic_scale, scale)
        output = _fp4_dq(quantized, dequantized_scale, block_size)
        return (
            output
            if output_dtype == onnx.TensorProto.FLOAT
            else _OPSET.Cast(output, to=output_dtype)
        )

    if format_bits == (8, 4, 9, 8):
        output_dtype = int(inputs.dtype)
        if quantizer_type == "dynamic":
            output = _mxfp8_dynamic_qdq(inputs, block_size, output_dtype)
        else:
            output = _mxfp8_static_dq(
                inputs, _OPSET.CastLike(1.0, inputs), block_size, output_dtype
            )
        return _OPSET.Identity(output)

    raise NotImplementedError(
        "Unsupported block format "
        f"({exponent_bits}, {num_bits - exponent_bits - 1}) with scale format "
        f"({scale_exponent_bits}, {scale_num_bits - scale_exponent_bits - 1})."
    )


def _get_dynamo_onnx_translation_table() -> dict:
    dynamic_op = torch.ops.tensorrt.dynamic_block_quantize_op
    return {
        torch.ops.tensorrt.quantize_op.default: _translate_quantize_op,
        dynamic_op.default: _translate_dynamic_block_quantize_op,
        dynamic_op.overload: _translate_dynamic_block_quantize_op,
    }
