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

"""Implements FP8 quantization for efficient tensor storage and computation."""

import math

import torch

from ..qtensor.base_qtensor import BaseQuantizedTensor
from ..utils import (
    convert_quantization_axis_to_reduce_axis,
    reduce_amax,
    reduce_block_amax,
    reduce_block_padding,
)

__all__ = ["FP8QTensor"]


class FP8QTensor(BaseQuantizedTensor):
    """Implements the FP8 quantization on tensors for more efficient storage or computation.

    Attributes:
        quantized_data (torch.Tensor): The quantized data stored as a packed fp8 tensor.
    """

    @classmethod
    def quantize(
        cls,
        input: torch.Tensor,
        scales: torch.Tensor = None,
        axis: tuple | int | None = None,
        block_sizes: dict | None = None,
    ) -> tuple:
        """Converting a tensor to a quantized format based on FP8 quantization. Only E4M3 is supported.

        Args:
            input (torch.Tensor): The input tensor to be quantized.
            scales (torch.Tensor): The scales for quantization.
            axis: The dimensions to reduce for quantization. None or int or tuple of ints.
            block_sizes (dict): A dictionary specifying the block size for each dimension.

        Note: One can only provide axis or block_sizes for FP8 quantization.

        Returns:
            tuple: FP8QTensor, scales
        """
        original_input = input

        # If block_sizes is provided, pad the input so that each dimension is divisible by the block size.
        if block_sizes:
            input = reduce_block_padding(input, block_sizes)

        # Compute scales if not provided
        if scales is None:
            if block_sizes:
                amax = reduce_block_amax(input, block_sizes)
            else:
                reduce_axis = convert_quantization_axis_to_reduce_axis(input, axis)
                amax = reduce_amax(input, axis=reduce_axis)
            scales = amax / 448.0  # Consider parameterizing the divisor if needed

        # Determine the expected scales shape from the (possibly padded) input
        expected_shape = list(input.shape)
        expanded_scales = scales.clone()
        if block_sizes:
            for dim, block_size in block_sizes.items():
                # Convert negative indices to positive ones.
                dim = dim if dim >= 0 else len(input.shape) + dim
                # After padding, this should always hold.
                assert input.shape[dim] % block_size == 0, (
                    f"Tensor dimension {dim}, {input.shape[dim]} is not divisible by {block_size} even after padding."
                )
                # The scales tensor is expected to have size equal to input.shape[dim] // block_size.
                expected_shape[dim] = input.shape[dim] // block_size

            # If we get amax from tensor_quantizer, it might not be the expected shape but has the
            # same number of elements, reshape it.
            if scales.shape != tuple(expected_shape) and scales.numel() == math.prod(
                expected_shape
            ):
                scales = scales.reshape(expected_shape)
                expanded_scales = expanded_scales.reshape(expected_shape)

            assert scales.shape == tuple(expected_shape), (
                f"Mismatch in expected scale shape: {scales.shape} vs {tuple(expected_shape)}"
            )

            # Expand scales along each block dimension for broadcasting.
            for dim, block_size in block_sizes.items():
                expanded_scales = expanded_scales.repeat_interleave(block_size, dim=dim)

        # Perform quantization using FP8 (E4M3) format.
        quantized_data = (input / expanded_scales).to(torch.float8_e4m3fn)

        # Crop quantized_data back to the original shape (if padding was added).
        slices = tuple(slice(0, dim) for dim in original_input.shape)
        quantized_data_cropped = quantized_data[slices]

        return cls(original_input.shape, original_input.dtype, quantized_data_cropped), scales

    def dequantize(self, dtype: torch.dtype = None, **kwarg):
        """Dequantze FP8 packed tensor to a target dtype."""
        if dtype is None:
            dtype = self.metadata["dtype"]
        assert "scale" in kwarg, "Require scale for FP8 dequantization."

        # get args
        scales = kwarg["scale"]
        block_sizes = kwarg.get("block_sizes")

        quantized_data = self._quantized_data
        if block_sizes:
            # pad the weight if needed
            quantized_data = reduce_block_padding(quantized_data, block_sizes)
            shape = quantized_data.shape

            # Compute expanded shape for broadcasting scales
            expanded_shape = list(shape)
            for dim, block_size in block_sizes.items():
                assert shape[dim] % block_size == 0, (
                    f"Dimension {shape[dim]} is not divisible by {block_size}."
                )
                expanded_shape[dim] //= block_size  # Reduce the dimension size for blocks

            assert tuple(expanded_shape) == scales.shape, (
                f"Scales shape {scales.shape} must match expected {tuple(expanded_shape)}."
            )

            # Expand scales for broadcasting
            for dim, block_size in block_sizes.items():
                scales = scales.repeat_interleave(block_size, dim=dim)

        # handle padded tensors
        slices = tuple(slice(0, dim) for dim in self.metadata["shape"])
        scales = scales.to(self._quantized_data.device)

        return (quantized_data.to(dtype) * scales.to(dtype))[slices]
