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

"""Basic tensor quantization functions."""

import warnings

import torch
from torch.autograd import Function
from torch.onnx import symbolic_helper

import modelopt.torch.quantization.triton as triton_kernel

from .config import QuantizerAttributeConfig
from .extensions import get_cuda_ext, get_cuda_ext_fp8, get_cuda_ext_mx

mx_format_map = {
    (4, 3): "E4M3",
    (5, 2): "E5M2",
    (3, 2): "E3M2",
    (2, 3): "E2M3",
    8: "INT8",
    (8, 0): "E8M0",
    (2, 1): "E2M1",
    (1, 2): "E1M2",
    (0, 3): "E0M3",
    (3, 0): "E3M0",
}

DISABLE_TRITON_KERNEL = False


def _fp8_eager(x, amax=None):
    dtype = x.dtype
    if amax is not None:
        scale = 448.0 / (amax.to(torch.float32))
        scale_inv = 1 / scale
        x = x.to(torch.float32) * scale
    x = x.to(torch.float8_e4m3fn)
    if amax is not None:
        x = x.to(torch.float32) * scale_inv
    return x.to(dtype)


def fp8_eager(x, amax):
    """Eager mode implementation of FP8 quantization."""
    return _fp8_eager(x, amax)


def scaled_e4m3_impl(
    inputs: torch.Tensor,
    amax: torch.Tensor | None = None,
) -> torch.Tensor:
    """Implementation of fake quantizing input to FP8.

    Args:
        inputs: Torch tensor.
        amax: Absolute max range of the input tensor.

    Returns:
        Input tensors faked quantized to FP8.
    """
    if (not inputs.is_cuda) or amax is None or amax.squeeze().ndim > 1:
        return fp8_eager(inputs, amax)

    cuda_ext_fp8 = get_cuda_ext_fp8(raise_if_failed=False)
    if cuda_ext_fp8 is None:
        return fp8_eager(inputs, amax)

    if amax.numel() == 1:
        outputs = cuda_ext_fp8.fake_e4m3fy(inputs, amax)
    elif amax.squeeze().ndim == 1:
        axis = amax.shape.index(amax.numel())
        outputs = cuda_ext_fp8.fake_e4m3fy_with_axis(inputs, amax.squeeze(), axis)
    return outputs


def fake_quant_impl(
    inputs: torch.Tensor,
    amax: torch.Tensor,
    num_bits=8,
    unsigned=False,
    narrow_range=True,
):
    """Implementation of fake quantizing input according to number of bits."""
    cuda_ext = get_cuda_ext()

    if amax.numel() == 1:
        outputs = cuda_ext.fake_tensor_quant(inputs, amax, num_bits, unsigned, narrow_range)
    else:
        axis = amax.shape.index(amax.numel())
        outputs = cuda_ext.fake_tensor_quant_with_axis(
            inputs, amax.squeeze(), axis, num_bits, unsigned, narrow_range
        )
    return outputs


def _quantize_impl(
    inputs: torch.Tensor,
    amax: torch.Tensor,
    num_bits: int = 8,
    exponent_bits: int = 0,
    unsigned: bool = False,
    narrow_range: bool = True,
):
    if num_bits == 8 and exponent_bits == 4:
        return scaled_e4m3_impl(inputs=inputs, amax=amax)
    elif isinstance(num_bits, int):
        return fake_quant_impl(
            inputs=inputs,
            amax=amax,
            num_bits=num_bits,
            unsigned=unsigned,
            narrow_range=narrow_range,
        )
    else:
        raise ValueError(
            f"Invalid combination of (num_bits, exponent_bits): ({num_bits}, {exponent_bits})."
        )


def _quantize_impl_abstract(
    input: torch.Tensor,
    amax: torch.Tensor,
    num_bits: int = 8,
    exponent_bits: int = 0,
    unsigned: bool = False,
    narrow_range: bool = True,
) -> torch.Tensor:
    """Register an abstract implementation for quantizing tensor.

    This abstract function returns an empty tensor with the same shape and dtype.
    """
    output = torch.empty_like(input)

    return output


# Argument types: Tensor, int, NoneType, int, int, int, int,
def _dynamic_block_quantize_impl(
    inputs: torch.Tensor,
    block_size: int,
    amax: torch.Tensor,
    num_bits: int,
    exponent_bits: int,
    scale_num_bits: int,
    scale_exponent_bits: int,
):
    scale_bits = (scale_exponent_bits, scale_num_bits - scale_exponent_bits - 1)
    if exponent_bits != 0:
        num_bits = (exponent_bits, num_bits - exponent_bits - 1)
    if num_bits in mx_format_map:
        assert scale_bits in mx_format_map, f"Scale bits should be in {mx_format_map.keys()}"
        if scale_bits != (8, 0):
            assert amax.is_cuda, "amax must be a CUDA tensor for dynamic block quantization."
            if amax.numel() != 1:
                amax = amax.amax()
        if (
            num_bits == (2, 1)  # type: ignore[comparison-overlap]
            and scale_bits == (4, 3)
            and triton_kernel.IS_AVAILABLE
            and not DISABLE_TRITON_KERNEL
            and amax is not None
        ):
            return triton_kernel.fp4_fake_quant_block(inputs, amax)
        cuda_ext_mx = get_cuda_ext_mx(raise_if_failed=True)
        return cuda_ext_mx.fused_amax_convert(
            inputs,
            block_size,
            getattr(cuda_ext_mx.Types, mx_format_map[num_bits]),
            getattr(cuda_ext_mx.Types, mx_format_map[scale_bits]),
            amax,
        )
    else:
        raise NotImplementedError(
            f"Unsupported num_bits: {num_bits}, scale_bits: {scale_bits} for dynamic block quantization."
        )


def _dynamic_block_quantize_impl_abstract(
    inputs: torch.Tensor,
    block_size: int,
    amax: torch.Tensor,
    num_bits: int,
    exponent_bits: int,
    scale_num_bits: int,
    scale_exponent_bits: int,
):
    """Register an abstract implementation for dynamic block quantization.

    This abstract function returns an empty tensor with the same shape and dtype.
    """
    output = torch.empty_like(inputs)

    return output


quantize_op = _quantize_impl
dynamic_block_quantize_op = _dynamic_block_quantize_impl
# Define custom operators via torch.library if supported:
# 1. quantize_op: Applies static quantization to the input tensor using a specified amax.
# 2. dynamic_block_quantize_op: Performs blockwise double quantization with dynamically
#    determined scales, governed by the given quantization format (scale_num_bits, scale_exponent_bits).
try:
    torch.library.define(
        "tensorrt::quantize_op",
        "(Tensor input, Tensor amax, int num_bits, int exponent_bits, "
        "bool unsigned, bool narrow_range) -> Tensor",
    )
    torch.library.define(
        "tensorrt::dynamic_block_quantize_op",
        "(Tensor input, int block_size, Tensor amax, int num_bits, int exponent_bits, "
        "int scale_num_bits, int scale_exponent_bits) -> Tensor",
    )
    torch.library.define(
        "tensorrt::dynamic_block_quantize_op.overload",
        "(Tensor input, int block_size, None amax, int num_bits, int exponent_bits, "
        "int scale_num_bits, int scale_exponent_bits) -> Tensor",
    )

    # Implement the None amax case
    def _dynamic_block_quantize_impl_none_amax(
        inputs: torch.Tensor,
        block_size: int,
        amax: None,
        num_bits: int,
        exponent_bits: int,
        scale_num_bits: int,
        scale_exponent_bits: int,
    ):
        return torch.empty_like(inputs)

    # Register the implementation for both CPU and CUDA
    torch.library.impl("tensorrt::quantize_op", ["cpu", "cuda"])(_quantize_impl)
    torch.library.impl("tensorrt::dynamic_block_quantize_op", ["cpu", "cuda"])(
        _dynamic_block_quantize_impl
    )
    torch.library.impl("tensorrt::dynamic_block_quantize_op.overload", ["cpu", "cuda"])(
        _dynamic_block_quantize_impl_none_amax
    )

    # Register the fake implementation
    torch.library.register_fake("tensorrt::quantize_op")(_quantize_impl_abstract)
    torch.library.register_fake("tensorrt::dynamic_block_quantize_op")(
        _dynamic_block_quantize_impl_abstract
    )
    torch.library.register_fake("tensorrt::dynamic_block_quantize_op.overload")(
        _dynamic_block_quantize_impl_abstract
    )

    quantize_op = torch.ops.tensorrt.quantize_op
    dynamic_block_quantize_op = torch.ops.tensorrt.dynamic_block_quantize_op
except (AttributeError, RuntimeError) as e:
    # torch.library is an experimental feature, the function signatures may change overtime.
    warnings.warn(
        "Unable to register operators with torch.library. Exporting quantized models with"
        f" torch.export will not be supported.\n{e}"
    )

# Predefined descriptors
QUANT_DESC_8BIT_PER_TENSOR = QuantizerAttributeConfig(num_bits=8)
QUANT_DESC_UNSIGNED_8BIT_PER_TENSOR = QuantizerAttributeConfig(num_bits=8, unsigned=True)
QUANT_DESC_8BIT_CONV1D_WEIGHT_PER_CHANNEL = QuantizerAttributeConfig(num_bits=8, axis=(0))
QUANT_DESC_8BIT_CONV2D_WEIGHT_PER_CHANNEL = QuantizerAttributeConfig(num_bits=8, axis=(0))
QUANT_DESC_8BIT_CONV3D_WEIGHT_PER_CHANNEL = QuantizerAttributeConfig(num_bits=8, axis=(0))
QUANT_DESC_8BIT_LINEAR_WEIGHT_PER_ROW = QuantizerAttributeConfig(num_bits=8, axis=(0))
QUANT_DESC_8BIT_CONVTRANSPOSE1D_WEIGHT_PER_CHANNEL = QuantizerAttributeConfig(num_bits=8, axis=(0))
QUANT_DESC_8BIT_CONVTRANSPOSE2D_WEIGHT_PER_CHANNEL = QuantizerAttributeConfig(num_bits=8, axis=(0))
QUANT_DESC_8BIT_CONVTRANSPOSE3D_WEIGHT_PER_CHANNEL = QuantizerAttributeConfig(num_bits=8, axis=(0))


@torch.jit.script
def _fake_tensor_quant_backward(inputs, amax: torch.Tensor | None, grad_outputs):
    # Skip clip for MX formats
    if amax is None:
        return grad_outputs

    zero = grad_outputs.new_zeros(1)
    grad_inputs = torch.where(inputs.abs() <= amax, grad_outputs, zero)
    return grad_inputs


def _fake_quant_backward_function(ctx, grad_outputs, num_args=1):
    saved_tensors = ctx.saved_tensors
    if len(saved_tensors) == 0:
        return (grad_outputs,) + (None,) * (num_args - 1)
    inputs, amax = saved_tensors
    return (_fake_tensor_quant_backward(inputs, amax, grad_outputs),) + (None,) * (num_args - 1)


def _save_for_backward_if_needed(ctx, pass_through_bwd, inputs, amax):
    if not pass_through_bwd and amax is not None:
        amax = (
            amax
            if isinstance(amax, torch.Tensor)
            else torch.tensor(amax, device=inputs.device, dtype=inputs.dtype)
        )
        ctx.save_for_backward(inputs, amax)


class FakeTensorQuantFunction(Function):
    """Fake version of TensorQuantFunction use CUDA extension."""

    @staticmethod
    @symbolic_helper.parse_args("v", "t", "t", "i", "b", "b", "s", "b", "i", "i")
    def symbolic(
        g,
        inputs,
        amax,
        bias=None,
        num_bits=8,
        unsigned=False,
        narrow_range=True,
        trt_high_precision_dtype=None,
        pass_through_bwd=False,
        block_size=None,
        axis=None,
    ):
        """ONNX symbolic function."""
        from .export_onnx import export_int4, export_int8

        if num_bits == 4:
            return export_int4(
                g, inputs, amax, num_bits, trt_high_precision_dtype, block_size, axis
            )

        return export_int8(
            g, inputs, amax, num_bits, unsigned, narrow_range, trt_high_precision_dtype
        )

    @staticmethod
    def forward(
        ctx,
        inputs,
        amax,
        bias=None,
        num_bits=8,
        unsigned=False,
        narrow_range=True,
        trt_high_precision_dtype=None,
        pass_through_bwd=False,
        block_size=None,
        axis=None,
    ):
        """Forward method."""
        if bias is not None:
            inputs = inputs - bias

        _save_for_backward_if_needed(ctx, pass_through_bwd, inputs, amax)

        def legacy_quant_func():
            # The LegacyFakeTensorQuantFunction support cpu and amax with any shape that can be broadcasted to inputs.
            outputs = _tensor_quant(inputs, amax, num_bits, unsigned, narrow_range)
            return outputs

        if not inputs.is_cuda:
            outputs = legacy_quant_func()
        else:
            try:
                outputs = quantize_op(
                    inputs,
                    amax,
                    num_bits=num_bits,
                    exponent_bits=0,
                    unsigned=unsigned,
                    narrow_range=narrow_range,
                )
            except (AttributeError, ValueError):
                # AttributeError: cuda_ext is not imported, possibly due to CPU only installation
                # ValueError: cuda_ext is installed, but trying to perform multidimensional quantization (amax dim > 1)
                outputs = legacy_quant_func()

        if bias is not None:
            outputs = outputs + bias

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        """Implements straight through estimation with clipping."""
        return _fake_quant_backward_function(ctx, grad_outputs, num_args=10)


class ScaledE4M3Function(Function):
    """E4M3fy input with scale."""

    @staticmethod
    @symbolic_helper.parse_args("v", "t", "t", "i", "i", "s", "b")
    def symbolic(
        g,
        inputs,
        amax=None,
        bias=None,
        E=4,  # noqa: N803
        M=3,  # noqa: N803
        trt_high_precision_dtype=None,
        pass_through_bwd=False,
    ):
        """ONNX symbolic function."""
        from .export_onnx import export_fp8

        return export_fp8(g, inputs, amax, trt_high_precision_dtype)

    @staticmethod
    # Default values could cause errors from TorchDynamo during torch.export
    def forward(
        ctx,
        inputs,
        amax,
        bias,
        E,  # noqa: N803
        M,  # noqa: N803
        trt_high_precision_dtype=None,
        pass_through_bwd=False,
    ):
        """Forward method."""
        if E != 4 or M != 3:
            raise NotImplementedError("Only support E=4 & M=3 for now.")

        if bias is not None:
            inputs = inputs - bias

        _save_for_backward_if_needed(ctx, pass_through_bwd, inputs, amax)

        outputs = quantize_op(
            inputs,
            amax,
            num_bits=8,
            exponent_bits=4,
            unsigned=False,
            narrow_range=False,
        )

        if bias is not None:
            outputs = outputs + bias

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        """Implements straight through estimation with clipping."""
        return _fake_quant_backward_function(ctx, grad_outputs, num_args=7)


def _dynamic_block_quantize_forward(
    ctx,
    inputs,
    block_size,
    amax,
    num_bits,
    scale_bits,
    trt_high_precision_dtype=None,
    onnx_quantizer_type="dynamic",
    pass_through_bwd=True,
):
    """Forward method."""
    if isinstance(num_bits, int):
        # special case for INT dynamic block quantization, e.g. MXINT8
        exponent_bits = 0
    else:
        assert isinstance(num_bits, tuple) and len(num_bits) == 2
        exponent_bits = num_bits[0]
        num_bits = num_bits[0] + num_bits[1] + 1
    assert isinstance(scale_bits, tuple) and len(scale_bits) == 2
    scale_exponent_bits = scale_bits[0]
    scale_num_bits = scale_bits[0] + scale_bits[1] + 1
    outputs = dynamic_block_quantize_op(
        inputs,
        block_size,
        amax,
        num_bits,
        exponent_bits,
        scale_num_bits,
        scale_exponent_bits,
    )
    return outputs


class DynamicBlockQuantizationFunction(Function):
    """Dynamic block quantization functional."""

    @staticmethod
    @symbolic_helper.parse_args("v", "i", "v", "t", "is", "is", "s", "s", "b")
    def symbolic(
        g,
        inputs,
        block_size,
        amax,
        bias,
        num_bits,
        scale_bits,
        trt_high_precision_dtype=None,
        onnx_quantizer_type="dynamic",
        pass_through_bwd=True,
    ):
        """ONNX symbolic function."""
        from .export_onnx import export_fp4, export_mxfp8

        if num_bits == (2, 1) and scale_bits == (4, 3):
            return export_fp4(
                g,
                inputs,
                block_size,
                amax,
                num_bits,
                trt_high_precision_dtype,
                onnx_quantizer_type,
            )
        if num_bits == (4, 3) and scale_bits == (8, 0):
            return export_mxfp8(
                g,
                inputs,
                onnx_quantizer_type,
                block_size,
            )
        raise NotImplementedError(
            f"Unsupported num_bits: {num_bits} and scale_bits: {scale_bits} for ONNX export."
        )

    @staticmethod
    def forward(
        ctx,
        inputs,
        block_size,
        amax,
        bias,
        num_bits,
        scale_bits,
        trt_high_precision_dtype=None,
        onnx_quantizer_type="dynamic",
        pass_through_bwd=True,
    ):
        """Forward method."""
        _save_for_backward_if_needed(ctx, pass_through_bwd, inputs, amax)
        return _dynamic_block_quantize_forward(
            ctx,
            inputs,
            block_size,
            amax,
            num_bits,
            scale_bits,
            trt_high_precision_dtype,
            onnx_quantizer_type,
            pass_through_bwd,
        )

    @staticmethod
    def backward(ctx, grad_outputs):
        """Implements straight through estimation with clipping."""
        return _fake_quant_backward_function(ctx, grad_outputs, num_args=9)


def _tensor_quant(inputs, amax, num_bits=8, unsigned=False, narrow_range=True):
    """Shared function body between TensorQuantFunction and FakeTensorQuantFunction."""
    # Fine scale, per channel scale will be handled by broadcasting, which could be tricky. Pop a warning.
    if unsigned and inputs.min() < 0.0:
        raise TypeError("Negative values encountered in unsigned quantization.")

    # Computation can be done in FP32 to prevent potential over flow.
    input_dtype = inputs.dtype
    inputs = inputs.float()
    amax = amax.float()

    min_amax = amax.min()
    if min_amax < 0:
        raise ValueError("Negative values in amax")

    max_bound = torch.tensor((2.0 ** (num_bits - 1 + int(unsigned))) - 1.0, device=amax.device)
    if unsigned:
        min_bound = 0
    elif narrow_range:
        min_bound = -max_bound
    else:
        min_bound = -max_bound - 1
    scale = max_bound / amax

    epsilon = 1.0 / (1 << 24)
    if min_amax <= epsilon:  # Treat amax smaller than minimum representable of fp16 0
        zero_amax_mask = amax <= epsilon
        scale[zero_amax_mask] = 0  # Value quantized with amax=0 should all be 0

    outputs = torch.clamp((inputs * scale).round_(), min_bound, max_bound)

    if min_amax <= epsilon:
        scale[zero_amax_mask] = (
            1.0  # Return 1 makes more sense for values quantized to 0 with amax=0
        )
    outputs = outputs / scale

    outputs = outputs.to(input_dtype)
    return outputs


fake_tensor_quant = FakeTensorQuantFunction.apply
scaled_e4m3 = ScaledE4M3Function.apply
dynamic_block_quant = DynamicBlockQuantizationFunction.apply
