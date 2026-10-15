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

"""Implements MXFP4 quantization for efficient tensor storage and computation."""

import torch

from ..qtensor.base_qtensor import BaseQuantizedTensor

__all__ = ["MXFP4QTensor"]


class MXFP4QTensor(BaseQuantizedTensor):
    """Implements the MXFP4 quantization on tensors for more efficient storage or computation.

    Attributes:
        quantized_data (torch.Tensor): The quantized data stored as a packed fp8 tensor.
    """

    E2M1_max = 6.0

    E2M1_values = [0, 0.5, 1, 1.5, 2, 3, 4, 6]
    E2M1_bounds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5])
    E8M0_MIN_EXPONENT = -127
    E8M0_MAX_EXPONENT = 127

    @classmethod
    def encode_e8m0_scale(cls, scale: torch.Tensor) -> torch.Tensor:
        """Encode positive floating-point scales as E8M0 exponent bytes."""
        min_scale = 2.0**cls.E8M0_MIN_EXPONENT
        max_scale = 2.0**cls.E8M0_MAX_EXPONENT
        exponent = torch.ceil(torch.log2(scale.float().clamp(min=min_scale, max=max_scale)))
        exponent = exponent.clamp(cls.E8M0_MIN_EXPONENT, cls.E8M0_MAX_EXPONENT)
        return (exponent + 127).to(torch.uint8)

    @classmethod
    def decode_e8m0_scale(cls, scale: torch.Tensor) -> torch.Tensor:
        """Decode E8M0 exponent bytes to floating-point powers of two."""
        return torch.exp2(scale.float() - 127)

    @classmethod
    def project_e8m0_scale_ste(cls, scale: torch.Tensor) -> torch.Tensor:
        """Project scales to E8M0 in forward while preserving identity gradients."""
        projected = cls.decode_e8m0_scale(cls.encode_e8m0_scale(scale))
        return scale.float() + (projected - scale.float()).detach()

    @staticmethod
    def _cast_fp4(x: torch.Tensor) -> torch.Tensor:
        sign = torch.sign(x)
        sign_bit = (2 - sign) // 2
        ord_ = torch.sum(
            (x.abs().unsqueeze(-1) - MXFP4QTensor.E2M1_bounds.to(x.device)) > 0, dim=-1
        )
        return (sign_bit * 0b1000 + ord_).to(torch.uint8)

    @staticmethod
    def _fuse_uint4_to_uint8(x: torch.Tensor) -> torch.Tensor:
        left_side = x[..., 0::2]
        right_side = x[..., 1::2]
        new_data = right_side.clone() << 4
        new_data[..., : left_side.shape[-1]] += left_side
        return new_data

    @classmethod
    def quantize_with_scale(
        cls, input: torch.Tensor, scale: torch.Tensor, block_size: int | None
    ) -> "MXFP4QTensor":
        """Quantize using caller-provided E8M0 scale bytes."""
        if block_size is None:
            block_size = 32
        if not isinstance(block_size, int) or block_size <= 0:
            raise ValueError(f"Block size must be a positive integer, received {block_size!r}.")
        if input.shape[-1] % block_size:
            raise ValueError(
                f"Input last dimension {input.shape[-1]} is not divisible by block size {block_size}."
            )
        if scale.dtype != torch.uint8:
            raise ValueError(f"E8M0 scale must use torch.uint8, received {scale.dtype}.")

        original_shape = input.shape
        original_dtype = input.dtype
        input_blocks = input.reshape(-1, block_size)
        scale = scale.reshape(-1)
        if scale.numel() != input_blocks.shape[0]:
            raise ValueError(
                f"Expected {input_blocks.shape[0]} E8M0 scales, received {scale.numel()}."
            )
        decoded_scale = cls.decode_e8m0_scale(scale).reshape(-1, 1)
        input_q = cls._cast_fp4(input_blocks.float() / decoded_scale)
        input_q = cls._fuse_uint4_to_uint8(input_q.reshape(original_shape))
        return cls(original_shape, original_dtype, input_q)

    @classmethod
    def quantize(cls, input: torch.Tensor, block_size: int | None) -> tuple:
        """Convert a tensor to MXFP4 with dynamically computed E8M0 scales.

        Args:
            input (torch.Tensor): The input tensor to be quantized.
            block_size (int | None): The block size for quantization.
        """
        if block_size is None:
            block_size = 32

        input_blocks = input.reshape(-1, block_size)
        input_amax = input_blocks.float().abs().max(dim=-1, keepdim=True).values
        e8m0_scale = cls.encode_e8m0_scale(input_amax / cls.E2M1_max)
        return cls.quantize_with_scale(input, e8m0_scale, block_size), e8m0_scale

    def dequantize(self, dtype: torch.dtype = None, **kwarg):
        """Dequantze MXFP4 packed tensor to a target dtype."""

        def unfuse_uint8_to_uint4(x):
            """Unfuse uint8 values back to uint4 values.

            This is the inverse operation of fuse_uint4_to_uint8.
            """
            # Extract the lower 4 bits (even indices)
            left_side = x & 0x0F

            # Extract the upper 4 bits (odd indices)
            right_side = (x >> 4) & 0x0F

            # Create a new tensor with alternating values
            shape = list(x.shape)
            shape[-1] = shape[-1] * 2
            result = torch.zeros(shape, dtype=torch.uint8, device=x.device)

            # Fill in the values - even indices get low bits, odd indices get high bits
            result[..., 0::2] = left_side  # Even indices from low bits
            result[..., 1::2] = right_side  # Odd indices from high bits

            return result

        e8m0_scale = kwarg["scale"]
        block_size = kwarg["block_sizes"][-1]
        if dtype is None:
            dtype = self.metadata["dtype"]

        # Unfuse the uint8 values back to uint4
        x_unfused = unfuse_uint8_to_uint4(self._quantized_data)
        # Extract sign and magnitude
        sign = 1 - 2 * ((x_unfused & 0b1000) >> 3).to(
            torch.float32
        )  # Extract sign bit and convert to +1/-1
        magnitude = x_unfused & 0b0111  # Extract magnitude bits
        magnitude = magnitude.to(torch.long)

        # Create a tensor with the E2M1 values
        values = torch.tensor(MXFP4QTensor.E2M1_values, device=self._quantized_data.device)

        # Use gather to index the values tensor properly
        # We need to reshape magnitude to match the dimensions we want to gather along
        original_shape = magnitude.shape
        x_float = values[magnitude.reshape(-1)].reshape(original_shape)

        # Apply sign and scale
        x_float = sign.float() * x_float

        # Reshape to apply block-wise scaling
        x_float = x_float.reshape(-1, block_size)

        # Apply the E8M0 scale
        scale_factor = self.decode_e8m0_scale(e8m0_scale)
        scale_factor = scale_factor.reshape(-1, 1)  # Reshape for proper broadcasting

        # Apply scaling and reshape back to original shape
        x_float = x_float * scale_factor

        # Reshape back to the original shape
        return x_float.reshape(original_shape).to(dtype)
