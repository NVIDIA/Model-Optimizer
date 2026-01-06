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

"""Implements MXFP8 quantization for efficient tensor storage and computation."""

import torch

from ..qtensor.base_qtensor import BaseQuantizedTensor

__all__ = ["MXFP8QTensor"]


class MXFP8QTensor(BaseQuantizedTensor):
    """Implements the MXFP8 quantization on tensors for more efficient storage or computation.

    MXFP8 uses:
    - FP8 E4M3 format for elements
    - E8M0 format for shared scales (power-of-2 only, stored as biased uint8 exponent)
    - Block size of 32 elements along the last dimension

    Attributes:
        quantized_data (torch.Tensor): The quantized data stored as float8_e4m3fn tensor.
    """

    E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0
    BLOCK_SIZE = 32

    @classmethod
    def _compute_e8m0_exponent(cls, amax: torch.Tensor) -> torch.Tensor:
        """Compute E8M0 exponent from per-block amax values.

        Args:
            amax: Per-block absolute max values.

        Returns:
            torch.Tensor: Float tensor of E8M0 exponents (unbiased, range [-127, 127]).
        """
        # Compute E8M0 scale: scale = 2^ceil(log2(amax / E4M3_max))
        descale = amax / cls.E4M3_MAX

        # Handle zero/inf/nan cases
        min_value = torch.tensor(-127.0, device=descale.device)
        log2_descale = torch.where(
            descale > 0,
            torch.log2(descale),
            min_value,
        )

        # Ceil to get power-of-2 scale
        e8m0_exponent = torch.ceil(log2_descale)

        # Clamp exponent to valid E8M0 range [-127, 127]
        return torch.clamp(e8m0_exponent, min=-127, max=127)

    @classmethod
    def get_weights_scaling_factor(
        cls,
        weight: torch.Tensor,
        block_size: int | None = None,
    ) -> torch.Tensor:
        """Returns E8M0 scale (uint8 biased exponent) for weight tensor.

        Args:
            weight: The weight tensor to compute scale for. Must be 2D.
            block_size: The block size for quantization. Defaults to 32.

        Returns:
            torch.Tensor: E8M0 scale as uint8 tensor with shape [out_dim, in_dim // block_size].
        """
        if block_size is None:
            block_size = cls.BLOCK_SIZE

        assert block_size > 0, f"block_size must be positive, got {block_size}"
        assert weight.dim() >= 2, f"Weight must be at least 2D, got {weight.dim()}D"

        out_dim, in_dim = weight.shape[-2], weight.shape[-1]

        assert in_dim % block_size == 0, (
            f"Weight inner dimension ({in_dim}) must be divisible by block_size ({block_size})"
        )

        # Reshape to [out_dim, num_blocks, block_size]
        weight_reshaped = weight.view(out_dim, in_dim // block_size, block_size)

        # Compute amax per block
        amax = weight_reshaped.float().abs().max(dim=-1)[0]

        # Compute E8M0 exponent and convert to biased uint8 (bias = 127)
        e8m0_exponent = cls._compute_e8m0_exponent(amax)
        return (e8m0_exponent + 127).to(torch.uint8)

    @classmethod
    def get_weights_scaling_factor_from_quantizer(
        cls,
        weight: torch.Tensor,
        weight_quantizer,
    ) -> torch.Tensor:
        """Returns E8M0 scale from quantizer or computes from weight.

        This method handles extracting the scale from a weight quantizer,
        with proper format conversion and shape correction.

        Args:
            weight: The weight tensor.
            weight_quantizer: The weight quantizer with block_sizes and optional _scale.

        Returns:
            torch.Tensor: E8M0 scale as uint8 tensor with shape [out_dim, in_dim // block_size].
        """
        assert hasattr(weight_quantizer, "block_sizes"), (
            "weight_quantizer must have 'block_sizes' attribute"
        )
        assert weight.dim() >= 2, f"Weight must be at least 2D, got {weight.dim()}D"

        block_size = weight_quantizer.block_sizes[-1]
        out_dim, in_dim = weight.shape[-2], weight.shape[-1]
        expected_shape = (out_dim, in_dim // block_size)

        if hasattr(weight_quantizer, "_scale") and weight_quantizer._scale is not None:
            scale = weight_quantizer._scale

            # If already uint8 E8M0 format, return as-is (with shape correction if needed)
            if scale.dtype == torch.uint8:
                if (
                    scale.shape != expected_shape
                    and scale.numel() == expected_shape[0] * expected_shape[1]
                ):
                    scale = scale.reshape(expected_shape)
                return scale

            # Legacy float32 scale - convert to E8M0 uint8
            if scale.shape != expected_shape:
                if scale.numel() == expected_shape[0] * expected_shape[1]:
                    scale = scale.reshape(expected_shape)
                else:
                    # Shape mismatch, recompute from weight
                    return cls.get_weights_scaling_factor(weight, block_size)

            # Convert float32 scale to E8M0 uint8
            e8m0_exponent = torch.ceil(torch.log2(scale.clamp(min=2**-127)))
            e8m0_exponent = torch.clamp(e8m0_exponent, min=-127, max=127)
            return (e8m0_exponent + 127).to(torch.uint8)

        # No scale in quantizer, compute from weight
        return cls.get_weights_scaling_factor(weight, block_size)

    @classmethod
    def quantize(cls, input: torch.Tensor, block_size: int | None = None) -> tuple:
        """Convert a tensor to MXFP8 quantized format.

        Args:
            input (torch.Tensor): The input tensor to be quantized.
            block_size (int | None): The block size for quantization. Defaults to 32.

        Returns:
            tuple: (MXFP8QTensor, e8m0_scale) where e8m0_scale is uint8 biased exponent.
        """
        if block_size is None:
            block_size = cls.BLOCK_SIZE

        assert block_size > 0, f"block_size must be positive, got {block_size}"
        assert input.numel() > 0, "Input tensor must not be empty"
        assert input.dim() >= 1, f"Input must have at least 1 dimension, got {input.dim()}D"

        original_shape = input.shape
        original_dtype = input.dtype

        # Pad last dimension if not divisible by block_size
        last_dim = original_shape[-1]
        if last_dim % block_size != 0:
            pad_size = block_size - (last_dim % block_size)
            input = torch.nn.functional.pad(input, (0, pad_size), mode="constant", value=0)

        # Flatten to [num_blocks, block_size] for block-wise quantization
        input_flat = input.view(-1, block_size)

        # Compute amax per block
        input_amax = input_flat.float().abs().max(dim=-1, keepdim=True).values

        # Compute E8M0 exponent and scale factor
        e8m0_exponent = cls._compute_e8m0_exponent(input_amax)
        scale_factor = torch.exp2(-e8m0_exponent)

        # Apply scale and quantize to FP8 E4M3
        scaled_input = input_flat * scale_factor

        # Clamp to E4M3 range [-448, 448] and convert
        scaled_input = torch.clamp(scaled_input, min=-cls.E4M3_MAX, max=cls.E4M3_MAX)
        quantized_data = scaled_input.to(torch.float8_e4m3fn)

        # Reshape: account for padded shape, then crop back to original
        padded_shape = list(original_shape)
        padded_shape[-1] = input.shape[-1]
        quantized_data = quantized_data.view(padded_shape)
        quantized_data = quantized_data[..., :last_dim]

        # Convert exponent to biased uint8 (bias = 127)
        e8m0_scale = (e8m0_exponent + 127).to(torch.uint8)

        # Reshape scale to preserve leading dimensions: (*original_shape[:-1], padded_last_dim // block_size)
        scale_shape = [*original_shape[:-1], input.shape[-1] // block_size]
        e8m0_scale = e8m0_scale.view(scale_shape)

        return cls(original_shape, original_dtype, quantized_data), e8m0_scale

    def dequantize(self, dtype: torch.dtype = None, **kwargs) -> torch.Tensor:
        """Dequantize MXFP8 tensor back to the target dtype.

        Args:
            dtype (torch.dtype | None): Target dtype for dequantization. Defaults to original dtype.
            **kwargs: Must contain 'scale' (E8M0 biased uint8) and 'block_sizes'.

        Returns:
            torch.Tensor: Dequantized tensor in the target dtype.
        """
        assert "scale" in kwargs, "dequantize requires 'scale' in kwargs"
        assert "block_sizes" in kwargs, "dequantize requires 'block_sizes' in kwargs"

        e8m0_scale = kwargs["scale"]
        block_size = kwargs["block_sizes"][-1]

        assert block_size > 0, f"block_size must be positive, got {block_size}"

        if dtype is None:
            dtype = self.metadata["dtype"]

        original_shape = self.metadata["shape"]
        last_dim = original_shape[-1]
        quantized_data = self._quantized_data

        # Validate scale shape matches expected number of blocks
        padded_last_dim = last_dim + (block_size - last_dim % block_size) % block_size
        expected_num_blocks = (quantized_data.numel() // last_dim) * (padded_last_dim // block_size)
        assert e8m0_scale.numel() == expected_num_blocks, (
            f"Scale has {e8m0_scale.numel()} elements but expected {expected_num_blocks} blocks"
        )

        # Pad last dimension if not divisible by block_size
        if last_dim % block_size != 0:
            pad_size = block_size - (last_dim % block_size)
            quantized_data = torch.nn.functional.pad(
                quantized_data.float(), (0, pad_size), mode="constant", value=0
            )
        else:
            quantized_data = quantized_data.float()

        # Flatten to [num_blocks, block_size] for block-wise dequantization
        quantized_flat = quantized_data.view(-1, block_size)

        # Convert E8M0 biased exponent back to scale factor: descale = 2^(exponent - 127)
        descale = torch.exp2(e8m0_scale.float() - 127)

        # Flatten scale to (num_blocks, 1) for broadcasting with quantized_flat
        descale = descale.view(-1, 1)

        # Apply descale
        dequantized = quantized_flat * descale

        # Reshape: account for padded shape, then crop back to original
        padded_shape = list(original_shape)
        padded_shape[-1] = quantized_data.shape[-1]
        dequantized = dequantized.view(padded_shape)
        dequantized = dequantized[..., :last_dim]

        return dequantized.to(dtype)
