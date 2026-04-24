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

    @classmethod
    def quantize(cls, input: torch.Tensor, block_size: int | None) -> tuple:
        """Converting a tensor to a quantized format based on MXFP4 quantization. Only E4M3 is supported.

        Args:
            input (torch.Tensor): The input tensor to be quantized.
            block_sizes (dict | None): The block sizes for quantization.
        """

        def cast_fp4(x):
            sign = torch.sign(x)
            sign_bit = (2 - sign) // 2
            ord_ = torch.sum(
                (x.abs().unsqueeze(-1) - MXFP4QTensor.E2M1_bounds.to(x.device)) > 0, dim=-1
            )
            fp4_val = (sign_bit * 0b1000 + ord_).to(torch.uint8)
            return fp4_val

        def fuse_uint4_to_uint8(x):
            # If the last dimension is odd, pad with zeros
            # If this behavior is not desired, please modify the code accordingly
            left_side = x[..., 0::2]  # Even indices (0, 2, 4...)
            right_side = x[..., 1::2]  # Odd indices (1, 3, 5...)
            new_data = right_side.clone() << 4  # Put odd indices (higher addresses) in high bits
            new_data[..., : left_side.shape[-1]] += left_side  # Put even indices in low bits
            return new_data

        if block_size is None:
            block_size = 32

        original_shape = input.shape
        original_dtype = input.dtype
        input = input.view(-1, block_size)
        # get scales
        # Casting scales to float yields better match with fakequant kernel (needs further investigation)
        input_amax = input.float().abs().max(dim=-1, keepdim=True).values
        descale = input_amax / cls.E2M1_max
        min_value = torch.tensor(-127.0, device=descale.device)
        e8m0_scale = torch.ceil(torch.maximum(torch.log2(descale), min_value))

        input = (input / torch.exp2(e8m0_scale)).view(original_shape)
        input_q = cast_fp4(input)
        input_q = fuse_uint4_to_uint8(input_q)
        e8m0_scale = (e8m0_scale + 127).to(torch.uint8)
        return cls(original_shape, original_dtype, input_q), e8m0_scale

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
        scale_factor = torch.exp2(e8m0_scale.float() - 127)
        scale_factor = scale_factor.reshape(-1, 1)  # Reshape for proper broadcasting

        # Apply scaling and reshape back to original shape
        x_float = x_float * scale_factor

        # Reshape back to the original shape
        return x_float.reshape(original_shape).to(dtype)

    # --- Signed E2M1 lookup, indexed by the full 4-bit pattern.
    # Bit 3 is the sign; bits 2..0 index the 8-entry magnitude table.
    _E2M1_SIGNED_VALUES = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ]
    _e2m1_signed_cache: dict = {}

    @classmethod
    def _get_signed_e2m1_lut(cls, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (device, dtype)
        if key not in cls._e2m1_signed_cache:
            cls._e2m1_signed_cache[key] = torch.tensor(
                cls._E2M1_SIGNED_VALUES, dtype=dtype, device=device
            )
        return cls._e2m1_signed_cache[key]

    @classmethod
    def dequantize_packed(
        cls,
        blocks: torch.Tensor,
        scales: torch.Tensor,
        *,
        block_size: int = 32,
        dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """Dequantize MXFP4-packed bytes to ``dtype`` without a QTensor wrapper.

        Input layout (DeepSeek-V4 checkpoint convention — group axis and packing
        axis fused in a single trailing dim):

            blocks : ``[..., K // 2]``, dtype ``uint8`` or ``int8``.
                     Low nibble = even element, high nibble = odd element.
            scales : ``[..., K // block_size]``, dtype ``uint8`` or
                     ``torch.float8_e8m0fnu``. UE8M0: byte ``e`` maps to
                     ``2 ** (e - 127)``.

        Returns a tensor of shape ``[..., K]`` in the requested ``dtype``.

        The GPT-OSS layout stores blocks and scales as
        ``blocks.shape == (..., G, 16)`` and ``scales.shape == (..., G)``.
        To feed such inputs here, reshape blocks to ``(..., G * 16)`` so that
        ``blocks.shape[:-1] == scales.shape[:-1]`` holds and the last dim of
        blocks is ``K // 2``. This helper does *no* trailing transpose, so the
        result is in the natural ``(out, in)`` orientation, suitable for
        feeding a standard ``nn.Linear`` or a downstream weight quantizer.

        UE8M0 note: per the OCP MX spec byte ``0xFF`` is NaN; we match
        ``transformers.integrations.mxfp4._convert_moe_packed_tensors`` by
        treating it as exponent ``+128``, which overflows bf16 to ``+Inf``.
        Real MXFP4 checkpoints do not use ``0xFF``.
        """
        # Local tensor only — DTensor/other wrappers would bypass ``view(uint8)``'s
        # byte reinterpretation; the caller should unwrap first (see the FP8 plugin
        # at ``_QuantFP8Linear._get_weight_and_scale_inv``).
        assert not isinstance(blocks, torch.distributed.tensor.DTensor) and not isinstance(
            scales, torch.distributed.tensor.DTensor
        ), "dequantize_packed expects local tensors; unwrap DTensor via ._local_tensor first"

        b = blocks.contiguous().view(torch.uint8)
        assert b.shape[:-1] == scales.shape[:-1], (
            f"Prefix shapes must match: blocks {tuple(b.shape)} vs scales {tuple(scales.shape)}"
        )
        K_half = b.shape[-1]
        G = scales.shape[-1]
        assert 2 * K_half == G * block_size, (
            f"Incompatible shapes: 2 * blocks.shape[-1] ({2 * K_half}) != "
            f"scales.shape[-1] * block_size ({G * block_size})"
        )

        lut = cls._get_signed_e2m1_lut(b.device, dtype)

        low = (b & 0x0F).long()
        high = (b >> 4).long()

        out = torch.empty(*b.shape[:-1], 2 * K_half, dtype=dtype, device=b.device)
        out[..., 0::2] = lut[low]
        out[..., 1::2] = lut[high]

        # Expose the per-group axis, apply the UE8M0 exponent via ldexp, then fold back.
        out = out.unflatten(-1, (G, block_size))
        exp = scales.contiguous().view(torch.uint8).to(torch.int32) - 127
        out = torch.ldexp(out, exp.unsqueeze(-1))
        return out.flatten(-2)
