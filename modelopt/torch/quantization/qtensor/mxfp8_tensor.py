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
    SCALE_DTYPE = torch.uint8  # E8M0 format stores biased exponent as uint8

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
    def get_weights_scaling_factor(cls, weight: torch.Tensor) -> torch.Tensor:
        """Returns E8M0 scale (uint8 biased exponent) for weight tensor.

        Args:
            weight: The weight tensor to compute scale for. Must be at least 2D.

        Returns:
            torch.Tensor: E8M0 scale as uint8 tensor with shape [..., out_dim, in_dim // 32].
        """
        assert weight.dim() >= 2, f"Weight must be at least 2D, got {weight.dim()}D"

        in_dim = weight.shape[-1]

        assert in_dim % cls.BLOCK_SIZE == 0, (
            f"Weight inner dimension ({in_dim}) must be divisible by MXFP8 block size ({cls.BLOCK_SIZE})"
        )

        # Reshape to [..., num_blocks, block_size]
        weight_reshaped = weight.view(*weight.shape[:-1], in_dim // cls.BLOCK_SIZE, cls.BLOCK_SIZE)

        # Compute amax per block
        amax = weight_reshaped.float().abs().max(dim=-1)[0]

        # Compute E8M0 exponent and convert to biased uint8 (bias = 127)
        e8m0_exponent = cls._compute_e8m0_exponent(amax)
        return (e8m0_exponent + 127).to(cls.SCALE_DTYPE)

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
            torch.Tensor: E8M0 scale as uint8 tensor with shape [out_dim, in_dim // 32].
        """
        assert hasattr(weight_quantizer, "block_sizes"), (
            "weight_quantizer must have 'block_sizes' attribute"
        )
        assert weight_quantizer.block_sizes[-1] == cls.BLOCK_SIZE, (
            f"MXFP8 requires block size {cls.BLOCK_SIZE}, got {weight_quantizer.block_sizes[-1]}"
        )
        assert weight.dim() >= 2, f"Weight must be at least 2D, got {weight.dim()}D"

        out_dim, in_dim = weight.shape[-2], weight.shape[-1]
        expected_shape = (out_dim, in_dim // cls.BLOCK_SIZE)

        if hasattr(weight_quantizer, "_scale") and weight_quantizer._scale is not None:
            scale = weight_quantizer._scale

            assert scale.dtype == cls.SCALE_DTYPE, (
                f"MXFP8 scale must be {cls.SCALE_DTYPE} (E8M0 format), got {scale.dtype}"
            )

            # Reshape if needed (same number of elements but wrong shape)
            if (
                scale.shape != expected_shape
                and scale.numel() == expected_shape[0] * expected_shape[1]
            ):
                scale = scale.reshape(expected_shape)
            return scale

        # No scale in quantizer, compute from weight
        return cls.get_weights_scaling_factor(weight)

    @classmethod
    def quantize_with_e8m0_scale(
        cls,
        weight: torch.Tensor,
        e8m0_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Quantize weight tensor using a pre-computed E8M0 scale.

        This method is useful for export paths where the scale has already been computed.

        Args:
            weight: The weight tensor to quantize. Must be at least 2D.
            e8m0_scale: E8M0 scale as uint8 biased exponent (bias = 127).
                Shape should be [..., out_dim, in_dim // 32].

        Returns:
            torch.Tensor: Quantized weight as float8_e4m3fn with same shape as input.
        """
        assert weight.dim() >= 2, f"Weight must be at least 2D, got {weight.dim()}D"
        assert e8m0_scale.dtype == cls.SCALE_DTYPE, (
            f"e8m0_scale must be {cls.SCALE_DTYPE} (E8M0 format), got {e8m0_scale.dtype}"
        )

        in_dim = weight.shape[-1]
        num_blocks = in_dim // cls.BLOCK_SIZE

        assert in_dim % cls.BLOCK_SIZE == 0, (
            f"Weight inner dimension ({in_dim}) must be divisible by MXFP8 block size ({cls.BLOCK_SIZE})"
        )

        # Reshape scale if needed (same number of elements but wrong shape)
        expected_shape = (*weight.shape[:-1], num_blocks)
        if e8m0_scale.shape != expected_shape:
            if e8m0_scale.numel() == weight.numel() // cls.BLOCK_SIZE:
                e8m0_scale = e8m0_scale.reshape(expected_shape)

        # Convert E8M0 biased exponent to scale factor: scale = 2^(127 - exponent)
        scale_factor = torch.exp2(127 - e8m0_scale.float())

        # NOTE: vLLM/flashinfer may require this behavior:
        # scale_factor = torch.where(
        #    e8m0_scale == 0,
        #    1.0,
        #    torch.exp2(127 - e8m0_scale.float())
        # )

        # Reshape weight to [..., out_dim, num_blocks, block_size]
        weight_reshaped = weight.view(*weight.shape[:-1], num_blocks, cls.BLOCK_SIZE)

        # Apply scale and quantize to FP8 E4M3
        scale_factor_expanded = scale_factor.unsqueeze(-1)
        scaled_weight = weight_reshaped * scale_factor_expanded
        scaled_weight = torch.clamp(scaled_weight, min=-cls.E4M3_MAX, max=cls.E4M3_MAX)
        quantized_weight = scaled_weight.to(torch.float8_e4m3fn)

        # Reshape back to original shape
        return quantized_weight.view(weight.shape)

    @classmethod
    def quantize(cls, input: torch.Tensor) -> tuple:
        """Convert a tensor to MXFP8 quantized format.

        Args:
            input (torch.Tensor): The input tensor to be quantized.

        Returns:
            tuple: (MXFP8QTensor, e8m0_scale) where e8m0_scale is uint8 biased exponent.
        """
        assert input.numel() > 0, "Input tensor must not be empty"
        assert input.dim() >= 1, f"Input must have at least 1 dimension, got {input.dim()}D"
        assert input.is_floating_point(), f"Input must be floating point, got {input.dtype}"

        original_shape = input.shape
        original_dtype = input.dtype

        # Pad last dimension if not divisible by block_size
        last_dim = original_shape[-1]
        if last_dim % cls.BLOCK_SIZE != 0:
            pad_size = cls.BLOCK_SIZE - (last_dim % cls.BLOCK_SIZE)
            input = torch.nn.functional.pad(input, (0, pad_size), mode="constant", value=0)

        # Flatten to [num_blocks, block_size] for block-wise quantization
        input_flat = input.view(-1, cls.BLOCK_SIZE)

        # Compute amax per block and E8M0 scale
        input_amax = input_flat.float().abs().max(dim=-1, keepdim=True).values
        e8m0_exponent = cls._compute_e8m0_exponent(input_amax)
        e8m0_scale = (e8m0_exponent + 127).to(cls.SCALE_DTYPE)

        # Reshape scale to match padded input shape for quantize_with_e8m0_scale
        padded_shape = list(original_shape)
        padded_shape[-1] = input.shape[-1]
        scale_shape = [*original_shape[:-1], input.shape[-1] // cls.BLOCK_SIZE]
        e8m0_scale = e8m0_scale.view(scale_shape)

        # Use quantize_with_e8m0_scale for the actual quantization (single source of truth)
        quantized_data = cls.quantize_with_e8m0_scale(input.view(padded_shape), e8m0_scale)

        # Crop back to original shape
        quantized_data = quantized_data[..., :last_dim]

        return cls(original_shape, original_dtype, quantized_data), e8m0_scale

    def dequantize(self, dtype: torch.dtype = None, **kwargs) -> torch.Tensor:
        """Dequantize MXFP8 tensor back to the target dtype.

        Args:
            dtype (torch.dtype | None): Target dtype for dequantization. Defaults to original dtype.
            **kwargs: Must contain 'scale' (E8M0 biased uint8).

        Returns:
            torch.Tensor: Dequantized tensor in the target dtype.
        """
        assert "scale" in kwargs, "dequantize requires 'scale' in kwargs"

        e8m0_scale = kwargs["scale"]
        assert e8m0_scale.dtype == self.SCALE_DTYPE, (
            f"e8m0_scale must be {self.SCALE_DTYPE} (E8M0 format), got {e8m0_scale.dtype}"
        )

        if dtype is None:
            dtype = self.metadata["dtype"]

        original_shape = self.metadata["shape"]
        last_dim = original_shape[-1]
        quantized_data = self._quantized_data

        # Validate scale shape matches expected number of blocks
        padded_last_dim = (
            last_dim + (self.BLOCK_SIZE - last_dim % self.BLOCK_SIZE) % self.BLOCK_SIZE
        )
        expected_num_blocks = (quantized_data.numel() // last_dim) * (
            padded_last_dim // self.BLOCK_SIZE
        )
        assert e8m0_scale.numel() == expected_num_blocks, (
            f"Scale has {e8m0_scale.numel()} elements but expected {expected_num_blocks} blocks"
        )

        # Pad last dimension if not divisible by block_size
        if last_dim % self.BLOCK_SIZE != 0:
            pad_size = self.BLOCK_SIZE - (last_dim % self.BLOCK_SIZE)
            quantized_data = torch.nn.functional.pad(
                quantized_data.float(), (0, pad_size), mode="constant", value=0
            )
        else:
            quantized_data = quantized_data.float()

        # Flatten to [num_blocks, block_size] for block-wise dequantization
        quantized_flat = quantized_data.view(-1, self.BLOCK_SIZE)

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
