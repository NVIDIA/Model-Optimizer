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

"""Implements NF4 quantization for efficient tensor storage and computation."""

import numpy as np
import torch

from ..extensions import get_cuda_ext
from ..qtensor.base_qtensor import BaseQuantizedTensor
from ..utils import reduce_amax, reduce_block_padding

nf4_table = torch.tensor(
    [
        -1.0000,
        -0.6962,
        -0.5251,
        -0.3949,
        -0.2844,
        -0.1848,
        -0.0911,
        0.0000,
        0.0796,
        0.1609,
        0.2461,
        0.3379,
        0.4407,
        0.5626,
        0.7230,
        1.0000,
    ],
    dtype=torch.bfloat16,
)

__all__ = ["NF4QTensor"]


def _dequantize_scalers(scales, double_scale, scale_zeros, dtype):
    return (scales / double_scale.unsqueeze(-1)).to(dtype) + scale_zeros


def _quantize_to_nearest_lut(flatten_tensor: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    """Quantize a float16 tensor to nearest value and return the indices of the look-up table."""
    assert flatten_tensor.dim() == 1, (
        f"Expect flatten tensor but got input with {flatten_tensor.dim()} dimensions."
    )
    diff = (flatten_tensor[:, None] - lut).abs()  # Shape: (numel, 16)
    indices = diff.argmin(dim=-1)
    return indices


def _nf4_lookup(quantized_idx: torch.Tensor) -> torch.Tensor:
    """Dequantize uint4 index to nf4 value."""
    return nf4_table.to(quantized_idx.device)[quantized_idx]


class NF4QTensor(BaseQuantizedTensor):
    """Implements the NF4 quantization on tensors for more efficient storage or computation.

    Attributes:
        quantized_data (torch.Tensor): The quantized data stored as a packed uint8 tensor.
    """

    @classmethod
    def quantize(
        cls, input: torch.Tensor, block_size: int, scale_block_size: int | None = None
    ) -> tuple:
        """Converting a tensor to a quantized format based on NF4 double quantization.

        Args:
            input (torch.Tensor): The input tensor to be quantized.
            block_size (int): The size of each block for quantization.
            scale_block_size (int): The block size for scaling during quantization.

        Returns:
            tuple: Contains quantized data, input quantization config, and scale quantization config.
        """
        cuda_ext = get_cuda_ext()

        # pad the input if needed
        original_input = input
        input = reduce_block_padding(input.view(-1), block_sizes={-1: block_size})

        # get scales for each block
        block_input = input.view(-1, block_size)
        scales = reduce_amax(block_input, -1)

        if cuda_ext and input.is_cuda:
            packed_output_uint8 = cuda_ext.NF4_quantize(input, scales, block_size)
        else:
            # expand scalers to match shape of input
            scales = scales.view(block_input.shape[0], -1)  # shape: (block_input.shape[0], 1)
            scaled_blocks = block_input / scales

            quantized_output = torch.empty(
                block_input.numel(), dtype=torch.uint8, device=block_input.device
            )
            flattened = scaled_blocks.flatten()
            quantized_output = _quantize_to_nearest_lut(
                flattened, nf4_table.to(device=input.device, dtype=input.dtype)
            )
            quantized_output_uint8 = quantized_output.to(torch.uint8)

            # pack the int4 weights into a uint8 tensor
            # packing format: w0, w1, w2, w3, w4, w5, ...
            #               | byte  | byte  | byte  |
            packed_output_uint8 = quantized_output_uint8[::2] << 4 | quantized_output_uint8[1::2]

        # pad the scales if needed
        scales = reduce_block_padding(scales.view(-1), block_sizes={-1: scale_block_size})
        return cls(original_input.shape, original_input.dtype, packed_output_uint8), scales

    @classmethod
    def double_quantization(cls, scales: torch.Tensor, scale_block_size: int, num_scale_bits: int):
        """Perform double quantization on the scales.

        Unlike the `quantize` method quantizing input data, this function quantizes float scales into
        int8 to further reduce memory usage of scales.
        """
        # Double quantization for the scales, int8 per-block quantization
        assert scales.numel() % scale_block_size == 0, (
            "Number of scales elements is not divisible by the scale block size."
        )
        scale_quant_maxbound = 2 ** (num_scale_bits - 1) - 1
        block_scales = scales.view(-1, scale_block_size)
        num_scale_blocks = block_scales.shape[0]
        scalers_zero_point = block_scales.mean()
        block_scales = block_scales - scalers_zero_point

        double_quant_scales = scale_quant_maxbound / reduce_amax(block_scales, -1)
        quantized_scales = block_scales * double_quant_scales.expand(
            num_scale_blocks, scale_block_size
        )
        quantized_scales = (
            quantized_scales.round()
            .clamp(-scale_quant_maxbound, scale_quant_maxbound)
            .to(torch.int8)
        )

        return quantized_scales, double_quant_scales.flatten(), scalers_zero_point

    def dequantize(self, dtype: torch.dtype = None, **kwarg):
        """Dequantze NF4 packed tensor to a target dtype."""
        if dtype is None:
            dtype = self.metadata["dtype"]
        cuda_ext = get_cuda_ext()

        # get kwargs
        scales = kwarg["scale"]
        block_sizes = kwarg["block_sizes"]
        double_scale = kwarg["double_scale"]
        scale_zeros = kwarg["scale_zeros"]

        # unpadd the scales if needed
        scales = scales.view(-1)[: (self._quantized_data.numel() * 2) // block_sizes[-1]]

        if cuda_ext and self._quantized_data.is_cuda:
            # with a custom cuda kernel
            scales = _dequantize_scalers(scales, double_scale, scale_zeros, dtype).flatten()
            output = cuda_ext.NF4_dequantize(self._quantized_data, scales, block_sizes[-1])
            return (
                output.view(-1)[: np.prod(self.metadata["shape"])]  # handle padding
                .reshape(self.metadata["shape"])
                .to(dtype)
            )
        else:
            # de-qauntize scales
            scales = _dequantize_scalers(scales, double_scale, scale_zeros, dtype).flatten()
            # indexing in torch required long dtype, we may need to optimize this with customized kernels
            first_half_idx = (self._quantized_data >> 4).to(torch.long)
            second_half_idx = (self._quantized_data & 0x0F).to(torch.long)
            scaled_first_half = _nf4_lookup(first_half_idx)  # w0, w2, w4...
            scaled_second_half = _nf4_lookup(second_half_idx)  # w1, w3, w5...

            # de-quantize tensor
            first_half = scaled_first_half.view(-1, block_sizes[-1] // 2) * scales.view(-1, 1)
            second_half = scaled_second_half.view(-1, block_sizes[-1] // 2) * scales.view(-1, 1)

            # merge the interleaving elements
            first_half = first_half.flatten().unsqueeze(-1).transpose(0, 1)
            second_half = second_half.flatten().unsqueeze(-1).transpose(0, 1)
            return (
                torch.stack([first_half, second_half], dim=-1)
                .view(-1)[: np.prod(self.metadata["shape"])]  # handle padding
                .reshape(self.metadata["shape"])
                .to(dtype)
            )
