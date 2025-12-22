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

"""Provides some basic utilities that can be used in quantize() methods."""

import math
from collections.abc import Sequence

import numpy as np

INT4_MIN = -8
INT4_MAX = 7
UINT4_MIN = 0
UINT4_MAX = 15
# following min-value for clip is taken from AutoAWQ where zero-point based quantization is
# supported and working
CLIP_MIN = 1e-5


def pack_float32_to_4bit_optimized(array: np.ndarray | Sequence, signed: bool) -> np.ndarray:
    """Convert an array of float32 value to a 4bit data-type and pack every two consecutive elements in a byte.

    This is the optimized version of pack_float32_to_4bit() utility in ONNX helper file. The basic optimizations
    done here mainly rely on moving some common code out of the per-element function calls or loops, thereby making
    them per-input-array, instead of per-input-element. The remaining logic should largely remain as is.

    Args:
        array: array of float to convert and pack
        signed: Whether the 4 bit variant is signed or unsigned

    Returns:
        Packed array with size `ceil(array.size/2)` (single dimension).
    """
    if not isinstance(array, np.ndarray):
        array = np.asarray(array, dtype=np.float32)

    array_flat = array.ravel()
    is_odd_volume = np.prod(array.shape) % 2 == 1
    if is_odd_volume:
        array_flat = np.append(array_flat, np.array([0]))

    inp_arr_len = array_flat.size
    dtype = np.int8 if signed else np.uint8
    clip_low = INT4_MIN if signed else UINT4_MIN
    clip_high = INT4_MAX if signed else UINT4_MAX
    array_flat = np.clip(array_flat, clip_low, clip_high)
    array_flat = np.rint(array_flat).astype(dtype)
    assert len(array_flat) % 2 == 0, "array length must be even at this point"
    assert len(array_flat) == inp_arr_len, "output-length must match the input-length"
    output_list = [
        ((array_flat[i + 1] << 4) | (array_flat[i] & 0x0F)) for i in range(0, inp_arr_len, 2)
    ]
    arr = np.array(output_list)
    return arr.astype(np.uint8)


def pack_float32_to_4bit_cpp_based(array: np.ndarray | Sequence, signed: bool) -> np.ndarray:
    """Convert an array of float32 value to a 4bit data-type and pack every two consecutive elements in a byte.

    This is the optimized version of pack_float32_to_4bit() utility in ONNX helper file. The basic optimizations
    here is to implement this round_and_pack logic in C++, which is supposed to be faster.

    Args:
        array: array of float to convert and pack
        signed: Whether the 4 bit variant is signed or unsigned

    Returns:
        Packed array with size `ceil(array.size/2)` (single dimension).
    """
    from .extensions import round_and_pack_ext

    if not isinstance(array, np.ndarray):
        array = np.asarray(array, dtype=np.float32)

    # - Currently, FP32, FP64, UINT8 and INT8 dtypes have C++ implementation of the round-and-pack
    # logic.
    # - With above, FP16 should also get "implicitly" supported due to possible type promotion to higher
    # precision float types (e.g. float32 or float64). So, it is mentioned as supported below.
    # - We can add support for other dtypes as and when needed.
    use_python_version = False
    if round_and_pack_ext is None or array.dtype not in [
        "float",
        "float16",
        "float32",
        "float64",
        "int8",
        "uint8",
    ]:
        use_python_version = True

    array_flat = array.ravel()
    is_odd_volume = np.prod(array.shape) % 2 == 1
    if is_odd_volume:
        array_flat = np.append(array_flat, np.array([0], array_flat.dtype))

    inp_arr_len = array_flat.size

    assert inp_arr_len % 2 == 0, "input array length must be even at this point"

    if use_python_version:
        numpy_out = pack_float32_to_4bit_optimized(array_flat, signed)
    else:
        numpy_out = np.zeros([1, int(inp_arr_len / 2)], dtype=np.int8)
        numpy_out = numpy_out.ravel()
        ret = round_and_pack_ext.round_and_pack(
            signed, array_flat, array_flat.size, numpy_out, numpy_out.size
        )
        assert ret == inp_arr_len / 2, "Unexpected output length"
        numpy_out = numpy_out.astype(np.uint8)

    return numpy_out


def get_weights_scaling_factor_2(input: np.ndarray):
    """Returns per tensor weight scaling factor."""
    per_block_scale_amax = np.max(np.abs(input)) / 6.0
    per_block_quant_scale = per_block_scale_amax / 448.0
    if per_block_quant_scale == 0:
        per_block_quant_scale = 1.0
    return np.float32(per_block_quant_scale)


def get_weights_scaling_factor(
    input: np.ndarray, block_size: int, weights_scaling_factor_2: np.float32
):
    """Returns quantized per block weight scaling factor."""
    # Get per_block amax
    [n, k] = input.shape
    assert block_size != 0, "Block size is zero. Cannot return per_block amax for given input."

    assert k % block_size == 0, (
        "Weight shape is not divisible for block size for block quantization."
    )

    input = input.reshape(n, k // block_size, block_size)
    # Get per block amax
    per_block_amax = np.max(np.abs(input), axis=-1)
    # Get per-block-scale
    per_block_scale = per_block_amax / 6.0
    # Quantize per_block_scale to FP8
    q_per_block_scale = per_block_scale / weights_scaling_factor_2
    # Set all zero values in scale to 1.0
    zero_mask = per_block_scale == 0
    q_per_block_scale[zero_mask] = 1.0
    return q_per_block_scale.astype(np.float32)


def update_block_size(
    block_size: int,
    layer_info: dict[str, dict] | None = None,
    name: str | None = None,
    quantize_axis: int = 0,
    w: np.ndarray = None,
) -> int:
    """Update the block size for quantization.

    Args:
        num_bits (int): Number of bits for quantization.
        layer_info (dict[str, dict] | None): Optional dictionary mapping tensor names
            to layer configuration dict.
        name (str | None): Name of the tensor.
        block_size (int): Current block size.
        quantize_axis (int): Axis along which to quantize.
        w (np.ndarray): Weight tensor to be quantized.

    Returns:
        int: Updated block size.
    """
    if layer_info and name in layer_info:
        block_size = layer_info[name]["block_size"]
        quantize_axis = layer_info[name]["axis"]
    if block_size is not None and block_size == -1:
        return w.shape[quantize_axis]
    return block_size


def get_num_bits(layer_info: dict[str, dict] | None = None, name: str | None = None) -> int:
    """Determine the layer configuration for quantization from layer_info.

    Args:
        layer_info (dict[str, dict] | None): Optional dictionary mapping tensor names
            to layer configuration dict.
        name (str | None): Name of the tensor.

    Returns:
        int: Number of bits to use for quantization. Defaults to 4 if not specified.
    """
    if layer_info and name in layer_info:
        num_bits = layer_info[name]["precision"]
    else:
        num_bits = 4
    return num_bits


def get_layer_block_size(
    layer_info: dict[str, dict] | None = None,
    name: str | None = None,
    default_block_size: int | None = None,
) -> int | None:
    """Get the block size for a specific layer from layer_info.

    Args:
        layer_info (dict[str, dict] | None): Optional dictionary mapping tensor names to layer configuration.
        name (str | None): Name of the tensor.
        default_block_size (int | None): Default block size if not specified. Defaults to None.

    Returns:
        int: Block size to use for quantization.
    """
    if layer_info and name in layer_info:
        return layer_info[name].get("block_size", default_block_size)
    return default_block_size


def get_layer_axis(
    layer_info: dict[str, dict] | None = None,
    name: str | None = None,
    default_axis: int | None = None,
) -> int | None:
    """Get the quantization axis for a specific layer from layer_info.

    Args:
        layer_info (dict[str, dict] | None): Optional dictionary mapping tensor names to layer configuration.
        name (str | None): Name of the tensor.
        default_axis (int | None): Default axis if not specified. Defaults to None.

    Returns:
        int: Quantization axis to use.
    """
    if layer_info and name in layer_info:
        return layer_info[name].get("axis", default_axis)
    return default_axis


def _next_block_size_multiple(x: float, block_size: int) -> float:
    return math.ceil(x / block_size) * block_size


def _pad(w: np.ndarray, block_size: int, quantize_axis: int = 0) -> np.ndarray:
    """Pads `w` to next largest multiple of block_size, on quantize_axis."""
    assert 0 <= quantize_axis < len(w.shape), (
        f"incorrect quantize-axis {quantize_axis}, w-shape={w.shape}"
    )

    if w.shape[quantize_axis] % block_size == 0:
        return w

    pad_width = (
        _next_block_size_multiple(w.shape[quantize_axis], block_size) - w.shape[quantize_axis]
    )
    pads = [(0, 0) for _ in range(len(w.shape))]
    pads[quantize_axis] = (0, pad_width)
    return np.pad(w, pads, mode="constant", constant_values=0)


def _depad(w: np.ndarray, orig_shape: tuple, quantize_axis: int = 0) -> np.ndarray:
    """Depad quantize_axis to original shape."""
    if w.shape == orig_shape:
        return w
    ans = None
    if quantize_axis == 0:
        ans = w[0 : orig_shape[0], ...]
    elif quantize_axis == 1:
        ans = w[..., 0 : orig_shape[1]]
    else:
        raise ValueError("Incorrect Quantize-axis: it must be 0 or 1 for a 2D array")
    return ans


def reshape_scales_for_per_channel_nodes(
    scales_map: dict[str, np.ndarray],
    block_size: int,
    layer_info: dict[str, dict] | None = None,
):
    """Update the scale map for per-channel nodes. For per channel quantization the scale needs to be 1D.

    Args:
        scales_map (dict[str, np.ndarray]): Dictionary mapping weight names to scale arrays.
        layer_info (dict[str, dict] | None): Optional dictionary mapping tensor names
            to layer configuration dict.

    Returns:
        dict[str, np.ndarray]: Updated scales map.
    """
    for name in scales_map:
        layer_block_size = block_size
        if layer_info and name in layer_info and isinstance(layer_info[name], dict):
            layer_block_size = layer_info[name].get("block_size", block_size)
        is_per_channel = layer_block_size == -1
        scales_map[name] = scales_map[name].reshape(-1) if is_per_channel else scales_map[name]
    return scales_map


def find_scales(
    w: np.ndarray,
    block_size: int,
    quantize_axis: int = 0,
    alpha: float = 1.0,
    use_zero_point: bool = False,
    num_bits: int = 4,
):
    """Find scale factors for `w` via `s = max(w.block(block_size)) / 7`."""
    w = _pad(w, block_size, quantize_axis)
    if quantize_axis == 0:
        w = w.T

    s_last_dim = w.shape[-1] // block_size
    s_shape = list(w.shape)
    s_shape[-1] = s_last_dim
    z = None
    if not use_zero_point:
        scale = 2 ** (num_bits - 1) - 1
        w_amax = np.abs(w.reshape(-1, block_size)).max(axis=-1)
        s = (w_amax * alpha) / scale
        s = np.clip(s, CLIP_MIN, None).reshape(s_shape)
    else:
        max_val = w.reshape(-1, block_size).max(axis=-1)
        min_val = w.reshape(-1, block_size).min(axis=-1)
        max_int = (2**num_bits) - 1
        min_int = 0
        s = (max_val - min_val).clip(min=CLIP_MIN) / max_int
        # z = -np.round(temp).clip(min=min_int, max=max_int)    # gives 0 - need to check
        temp = min_val / s
        temp = np.round(temp)
        temp = -temp
        temp = temp.clip(min=min_int, max=max_int)
        z = temp
        # Validate zero-point values are within expected range
        if not np.all((z >= min_int) & (z <= max_int)):
            raise ValueError(
                f"Zero-point values out of range [{min_int}, {max_int}]: min={np.min(z)}, max={np.max(z)}"
            )
        assert s.shape == z.shape, "s and z shape mismatch"
        s = s.reshape(s_shape)
        z = z.reshape(s_shape)
    assert z is None or use_zero_point is True, "zero-point value and use-zero-point not in sync"
    if quantize_axis == 0:
        s = s.T
        if z is not None:
            z = z.T
    return s, z


def rtn(
    w: np.ndarray,
    s: np.ndarray,
    block_size: int,
    quantize_axis: int = 0,
    zp: np.ndarray = None,
    num_bits: int = 4,
) -> np.ndarray:
    """Quantizes `w` with scale factors `s` via Round-to-Nearest.

    Ties are broken by rounding to the nearest even number.
    """
    w_padded = _pad(w, block_size, quantize_axis)
    num_blocks = w_padded.shape[quantize_axis] // s.shape[quantize_axis]
    if zp is None:
        maxq = 2 ** (num_bits - 1) - 1
        minq = -(2 ** (num_bits - 1))
        w_padded = (
            np.rint(w_padded / s.repeat(num_blocks, axis=quantize_axis))
            .clip(minq, maxq)
            .astype(np.int8)
        )
    else:
        maxq = (2**num_bits) - 1
        minq = 0
        w_padded = (
            (
                np.rint(w_padded / s.repeat(num_blocks, axis=quantize_axis))
                + zp.repeat(num_blocks, axis=quantize_axis)
            )
            .clip(minq, maxq)
            .astype(np.int8)
        )
    return _depad(w_padded, w.shape, quantize_axis)


def dq_tensor(
    w: np.ndarray,
    s: np.ndarray,
    block_size: int,
    quantize_axis: int = 0,
    zp: np.ndarray = None,
) -> np.ndarray:
    """Dequantizes `w` with scale factors `s`."""
    w_padded = _pad(w, block_size, quantize_axis)
    num_blocks = w_padded.shape[quantize_axis] // s.shape[quantize_axis]
    if zp is None:
        w_padded = w_padded * s.repeat(num_blocks, axis=quantize_axis)
    else:
        w_padded = (w_padded - zp.repeat(num_blocks, axis=quantize_axis)) * s.repeat(
            num_blocks, axis=quantize_axis
        )
    return _depad(w_padded, w.shape, quantize_axis)


def quant_tensor(
    w: np.ndarray,
    block_size: int,
    quantize_axis: int = 0,
    alpha: float = 1.0,
    use_zero_point: bool = False,
    num_bits: int = 4,
):
    """Quantize a tensor using alpha etc. and return the quantized tensor.

    Returns:
        tuple: A tuple containing:
            - wq: The quantized weight tensor (np.ndarray)
            - scale: The scale factors used for quantization (np.ndarray)
            - zp: The zero-point values (np.ndarray or None if not using zero-point)
    """
    scale, zp = find_scales(w, block_size, quantize_axis, alpha, use_zero_point, num_bits)
    wq = rtn(w, scale, block_size, quantize_axis, zp, num_bits)
    return wq, scale, zp


def quantize(
    input: np.ndarray,
    block_size: int,
    weights_scaling_factor: np.ndarray,
    weights_scaling_factor_2: np.ndarray,
):
    """Converting a tensor to a quantized format based on NVFP4 quantization."""
    # Reshape the weight and scale factors
    input = input.reshape((*tuple(input.shape[:-1]), -1, block_size))

    # Scale weights
    scaled_weight = input / (
        np.expand_dims(weights_scaling_factor, axis=-1) * weights_scaling_factor_2
    )

    # Reshape weights to original and return
    return scaled_weight.reshape(scaled_weight.shape[0], -1)


def pack_weights_to_int4(
    weight: np.ndarray,
) -> np.ndarray:
    """Converts ONNX model weights from high precision to INT4 precision."""
    weight_shape_int8 = weight.shape
    assert weight_shape_int8[0] % 2 == 0, "weight_shape[0] must be divisible by 2"
    weight_shape_int4 = (weight_shape_int8[0] // 2, *weight_shape_int8[1:])
    weight = weight.flatten().round()
    weights_int8_np = np.clip(weight, -8, 7).astype(np.int8)
    weights_int4_np = np.zeros(weights_int8_np.shape[0] // 2, dtype=np.int8)
    weights_int4_np = (((weights_int8_np[1::2]) << 4) | (weights_int8_np[::2] & 0xF)).astype(
        np.uint8
    )
    weights_int4_np = weights_int4_np.reshape(weight_shape_int4)
    return weights_int4_np


def get_amax(weight: np.ndarray, quant_axis: int, block_size: int) -> np.ndarray:
    """Returns the amax of the weight tensor along the specified axis for a given block size.

    Only 2D and 3D tensors are supported.

    Args:
        weight: The weight tensor.
        quant_axis: The axis to quantize.
        block_size: The block size.

    Returns:
        The amax of the weight tensor.
    """
    rank = weight.ndim
    if quant_axis == -1:
        quant_axis = rank - 1
    assert rank in [2, 3], "Weight must be a 2D or 3D tensor"
    if rank == 3:
        d0, d1, d2 = weight.shape
        if quant_axis == 2:
            assert d2 % block_size == 0, (
                f"Weight dimension {d2} must be divisible by block size {block_size}"
            )
            amax = np.abs(weight.reshape(d0, d1 * d2 // block_size, block_size)).max(axis=2)
        elif quant_axis == 1:
            assert d1 % block_size == 0, (
                f"Weight dimension {d1} must be divisible by block size {block_size}"
            )
            amax = np.abs(weight.reshape(d0 * d1 // block_size, block_size, d2)).max(axis=1)
        else:
            raise ValueError(f"Unsupported weight axis: {quant_axis}")
    else:
        d0, d1 = weight.shape
        if quant_axis == 1:
            assert d1 % block_size == 0, (
                f"Weight dimension {d1} must be divisible by block size {block_size}"
            )
            amax = np.abs(weight.reshape(d0 * d1 // block_size, block_size)).max(axis=1)
        elif quant_axis == 0:
            assert d0 % block_size == 0, (
                f"Weight dimension {d0} must be divisible by block size {block_size}"
            )
            amax = np.abs(weight.reshape(block_size, d0 * d1 // block_size)).max(axis=0)
        else:
            raise ValueError(f"Unsupported weight axis: {quant_axis}")
    return amax


def compute_e8m0(
    amax: np.ndarray, weight_shape: tuple[int, ...], quant_axis: int, block_size: int
) -> np.ndarray:
    """Computes the e8m0 value for the weight tensor.

    Args:
        amax: The amax of the weight tensor.
        weight_shape: The shape of the weight tensor.
        quant_axis: The axis to compute the e8m0 value.
        block_size: The block size.

    Returns:
        The e8m0 value for the weight tensor.
    """
    rank = len(weight_shape)
    if quant_axis == -1:
        quant_axis = rank - 1
    q_max = 448  # Largest value of FP8
    e8m0_bias = 127
    amax = amax / q_max
    min_value = -127.0
    e8m0_unbiased_exp = np.ceil(np.maximum(np.log2(amax), min_value))
    e8m0 = e8m0_unbiased_exp + e8m0_bias
    if not np.all((e8m0 >= 0) & (e8m0 <= 255)):
        raise ValueError(f"e8m0 values out of range: min={np.min(e8m0)}, max={np.max(e8m0)}")
    if rank == 2:
        d0, d1 = weight_shape
        if quant_axis == 1:
            e8m0 = e8m0.reshape(d0, d1 // block_size)
        elif quant_axis == 0:
            e8m0 = e8m0.reshape(d0 // block_size, d1)
        else:
            raise ValueError(f"Unsupported axis: {quant_axis}")
    elif rank == 3:
        d0, d1, d2 = weight_shape
        if quant_axis == 2:
            e8m0 = e8m0.reshape(d0, d1, d2 // block_size)
        elif quant_axis == 1:
            e8m0 = e8m0.reshape(d0, d1 // block_size, d2)
        else:
            raise ValueError(f"Unsupported axis: {quant_axis}")
    else:
        raise ValueError(f"Unsupported weight shape: {weight_shape}")
    return e8m0
