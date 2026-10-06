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
"""Implements INT8 quantization for efficient tensor storage and computation."""

import torch

from ..qtensor.base_qtensor import BaseQuantizedTensor
from ..utils import (
    convert_quantization_axis_to_reduce_axis,
    reduce_amax,
    reduce_block_amax,
    reduce_block_padding,
)


class INT8QTensor(BaseQuantizedTensor):
    """Implements the INT8 quantization on tensors for more efficient storage or computation.

    Attributes:
        quantized_data (torch.Tensor): The quantized data stored as an INT8 tensor.
    """

    @classmethod
    def quantize(
        cls,
        input: torch.Tensor,
        scales: torch.Tensor | None = None,
        axis: tuple | int | None = None,
        block_sizes: dict | None = None,
    ) -> tuple:
        """Converting a tensor to a quantized format based on INT8 quantization.

        Args:
            input (torch.Tensor): The input tensor to be quantized.
            scales (torch.Tensor): The scales for quantization.
            axis: The dimensions to reduce for quantization. None or int or tuple of ints.
            block_sizes (dict): A dictionary specifying the block size for each dimension.
        Note: One can only provide axis or block_sizes for INT8 quantization.

        Returns:
            tuple: INT8QTensor, scales
        """
        original_input = input
        if scales is None:
            if block_sizes:
                input = reduce_block_padding(input, block_sizes)
                amax = reduce_block_amax(input, block_sizes)
            else:
                reduce_axis = convert_quantization_axis_to_reduce_axis(input, axis)
                amax = reduce_amax(input, axis=reduce_axis)
            scales = amax / 127.0

        # Calculate the scale shape and make sure it aligns with input and block_sizes
        expected_shape = list(input.shape)
        expanded_scales = scales.clone()
        if block_sizes:
            for dim, block_size in block_sizes.items():
                dim = dim if dim >= 0 else len(input.shape) + dim  # Convert negative index
                assert input.shape[dim] % block_size == 0, (
                    f"Tensor dimension {dim}, {input.shape[dim]} is not divisible by {block_size}."
                )
                expected_shape[dim] = (
                    input.shape[dim] // block_size
                )  # Adjust expected shape for blocks

            # Assert the shape of `scales` matches expected reduced dimensions
            assert scales.shape == tuple(expected_shape), (
                f"Mismatch in expected scale shape: {scales.shape} vs {tuple(expected_shape)}"
            )

            # Expand scales for broadcasting
            for dim, block_size in block_sizes.items():
                expanded_scales = expanded_scales.repeat_interleave(block_size, dim=dim)

        # Quantization
        quantized_data = (input / expanded_scales).round().clamp(-128, 127).to(torch.int8)

        return cls(original_input.shape, original_input.dtype, quantized_data), scales

    def dequantize(self, dtype: torch.dtype = None, **kwarg):
        """Dequantize INT8 packed tensor to a target dtype."""
        if dtype is None:
            dtype = self.metadata["dtype"]
        assert "scale" in kwarg, "Require scale for INT8 dequantization."

        # Get args
        scales = kwarg["scale"]
        block_sizes = kwarg.get("block_sizes")

        shape = self._quantized_data.shape
        if block_sizes:
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

        # Handle padded tensors
        slices = tuple(slice(0, dim) for dim in self.metadata["shape"])
        scales = scales.to(self._quantized_data.device)

        return (self._quantized_data.view(torch.int8).to(dtype) * scales.to(dtype))[slices]
