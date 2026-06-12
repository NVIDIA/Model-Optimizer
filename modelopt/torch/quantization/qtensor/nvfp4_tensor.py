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

"""Implements NVFP4 quantization for efficient tensor storage and computation."""

import torch

from ..backends.utils import fp4_compatible
from ..qtensor.base_qtensor import BaseQuantizedTensor
from ..utils import reduce_amax, reduce_block_amax, reduce_block_padding

# Define conversion tables
e2m1_bounds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5])
e2m1_values = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])

# Four Over Six (4/6) adaptive block scaling — paper arXiv:2512.02010v5.
# scales to either 4 or 6 per block, therefore the FP8 block scales are either 448 or 256.
FP4_E2M1_MAX = 6.0
FP4_E2M1_MAX_M4 = 4.0
F8_E4M3_MAX = 448.0
F8_E4M3_MAX_46 = 256.0

__all__ = ["NVFP4QTensor"]


def _cast_per_block_scale_to_fp8(
    per_block_scale: torch.Tensor,
    per_block_scale_max: torch.Tensor | None = None,
    fp8_max_for_normalization: float = F8_E4M3_MAX,
) -> torch.Tensor:
    """Clamp to FP8 E4M3FN range [2**-9, 448] and cast — avoids underflow→0 / overflow→NaN.

    When ``per_block_scale_max`` is provided, first rescales as
    ``per_block_scale.float() * fp8_max_for_normalization / per_block_scale_max`` — the static-export
    path needs this because the ``[==0]=1.0`` safety net combined with a small
    ``global_amax`` can drive the rescaled value above 448 (see PR #1397).
    """
    if per_block_scale_max is not None:
        per_block_scale = per_block_scale.float() * fp8_max_for_normalization / per_block_scale_max
    return per_block_scale.clamp(min=2**-9, max=448.0).to(torch.float8_e4m3fn)


class NVFP4QTensor(BaseQuantizedTensor):
    """Implements the INT4 quantization on tensors for more efficient storage or computation.

    Attributes:
        quantized_data (torch.Tensor): The quantized data stored as a packed uint8 tensor.
    """

    e2m1_values_on_device = {}
    e2m1_bounds_on_device = {}

    @classmethod
    def get_e2m1_values(cls, device):
        """Returns the e2m1 values on the device."""
        if device not in cls.e2m1_values_on_device:
            cls.e2m1_values_on_device[device] = e2m1_values.to(device)
        return cls.e2m1_values_on_device[device]

    @classmethod
    def get_e2m1_bounds(cls, device):
        """Returns the e2m1 values on the device."""
        if device not in cls.e2m1_bounds_on_device:
            cls.e2m1_bounds_on_device[device] = e2m1_bounds.to(device)
        return cls.e2m1_bounds_on_device[device]

    @classmethod
    def _is_static_quantizer(cls, weight_quantizer) -> bool:
        """Check if the weight quantizer is a static NVFP4 quantizer with pre-computed amax."""
        global_amax = cls._get_static_global_amax(weight_quantizer)
        return global_amax is not None

    @classmethod
    def _get_static_global_amax(cls, weight_quantizer):
        """Return global amax from live or restored static NVFP4 quantizers."""
        global_amax = getattr(weight_quantizer, "global_amax", None)
        if global_amax is None:
            global_amax = getattr(weight_quantizer, "_global_amax", None)
        return global_amax

    @classmethod
    def _is_four_over_six(cls, weight_quantizer) -> bool:
        """Return True if 4/6 adaptive block scaling is enabled on this quantizer."""
        bs = getattr(weight_quantizer, "block_sizes", None) or {}
        return bool(bs.get("four_over_six", False))

    @classmethod
    def get_weights_scaling_factor_2_from_quantizer(cls, weight_quantizer):
        """Returns per tensor weight scaling factor from the weight_quantizer.

        Handles both static NVFP4 quantizers (using global_amax) and
        dynamic quantizers (using _amax).

        Args:
            weight_quantizer: The weight quantizer (static or dynamic).

        Returns:
            The global scaling factor as a float tensor.
        """
        m_fp8 = F8_E4M3_MAX_46 if cls._is_four_over_six(weight_quantizer) else F8_E4M3_MAX
        global_amax = cls._get_static_global_amax(weight_quantizer)
        if global_amax is not None:
            return global_amax.float() / (FP4_E2M1_MAX * m_fp8)
        else:
            assert hasattr(weight_quantizer, "_amax"), (
                "Weight quantizer does not have attribute amax"
            )
            return weight_quantizer._amax.float() / (FP4_E2M1_MAX * m_fp8)

    @classmethod
    def get_weights_scaling_factor_from_quantizer(
        cls,
        weight_quantizer,
        weight: torch.Tensor,
        weights_scaling_factor_2: torch.Tensor | None = None,
        keep_high_precision: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns quantized per block weight scaling factor from quantizer.

        Handles both static NVFP4 quantizers (with pre-computed per-block amax)
        and dynamic quantizers (computing from weight tensor).

        Args:
            weight_quantizer: The weight quantizer (static or dynamic).
            weight: The weight tensor (used for shape in static, values in dynamic).
            weights_scaling_factor_2: Optional pre-computed global scale.
            keep_high_precision: Whether to keep scales in high precision.

        Returns:
            Tuple of (per_block_scale, weights_scaling_factor_2).
        """
        block_size = weight_quantizer.block_sizes[-1]

        if weights_scaling_factor_2 is None:
            weights_scaling_factor_2 = cls.get_weights_scaling_factor_2_from_quantizer(
                weight_quantizer
            )

        is_four_over_six = cls._is_four_over_six(weight_quantizer)
        fp8_max_for_normalization = F8_E4M3_MAX_46 if is_four_over_six else F8_E4M3_MAX

        if cls._is_static_quantizer(weight_quantizer):
            # Static path: use pre-computed per-block amax values from quantizer
            global_amax = cls._get_static_global_amax(weight_quantizer).float()
            per_block_amax = weight_quantizer._amax.float()

            # Compute scales in float
            per_block_scale_max = global_amax / FP4_E2M1_MAX
            per_block_scale = per_block_amax / FP4_E2M1_MAX
            per_block_scale[per_block_scale == 0] = 1.0

            # Reshape per_block_scale to match weight's block structure
            num_blocks_per_row = weight.shape[-1] // block_size
            expected_shape = (*weight.shape[:-1], num_blocks_per_row)
            per_block_scale = per_block_scale.view(expected_shape)

            if is_four_over_six:
                per_block_scale = cls._select_four_over_six_scale(
                    weight,
                    per_block_scale,
                    weights_scaling_factor_2,
                    block_size,
                    per_block_scale_max,
                    fp8_max_for_normalization=fp8_max_for_normalization,
                )

            if not keep_high_precision:
                per_block_scale = _cast_per_block_scale_to_fp8(
                    per_block_scale,
                    per_block_scale_max,
                    fp8_max_for_normalization=fp8_max_for_normalization,
                )
            return per_block_scale, weights_scaling_factor_2
        else:
            # Dynamic path: compute from weight tensor
            return cls.get_weights_scaling_factor(
                weight,
                block_size,
                weights_scaling_factor_2,
                keep_high_precision,
                four_over_six=is_four_over_six,
            )

    @classmethod
    def get_weights_scaling_factor(
        cls,
        input: torch.Tensor,
        block_size: int,
        weights_scaling_factor_2: torch.Tensor | None = None,
        keep_high_precision: bool = False,
        four_over_six: bool = False,
    ):
        """Returns quantized per block weight scaling factor from weight tensor.

        This is the dynamic path that computes scales directly from the weight values.
        For quantizers with pre-computed amax, use get_weights_scaling_factor_from_quantizer.
        """
        if weights_scaling_factor_2 is None:
            weights_scaling_factor_2 = cls.get_weights_scaling_factor_2(
                input, four_over_six=four_over_six
            )

        # Get per_block amax
        assert block_size != 0, "Block size is zero. Cannot return per_block amax for given input."

        assert input.shape[-1] % block_size == 0, (
            "Weight shape is not divisible for block size for block quantization."
        )

        # Get per block amax
        per_block_amax = reduce_block_amax(input, block_sizes={-1: block_size}).float()
        # Get per-block-scale (default M=6)
        per_block_scale = per_block_amax / (
            FP4_E2M1_MAX * weights_scaling_factor_2.to(per_block_amax.device)
        )
        # Set all zero values in scale to 1.0
        per_block_scale[per_block_scale == 0] = 1.0

        if four_over_six:
            per_block_scale = cls._select_four_over_six_scale(
                input, per_block_scale, weights_scaling_factor_2, block_size
            )

        if not keep_high_precision:
            per_block_scale = _cast_per_block_scale_to_fp8(per_block_scale)
        return per_block_scale, weights_scaling_factor_2

    @classmethod
    def get_weights_scaling_factor_2(cls, input: torch.Tensor, four_over_six: bool = False):
        """Returns per tensor weight scaling factor."""
        m_fp8 = F8_E4M3_MAX_46 if four_over_six else F8_E4M3_MAX
        return reduce_amax(input).float() / (FP4_E2M1_MAX * m_fp8)

    @classmethod
    def _select_four_over_six_scale(
        cls,
        weight: torch.Tensor,
        per_block_scale_m6: torch.Tensor,
        weights_scaling_factor_2: torch.Tensor,
        block_size: int,
        per_block_scale_max: torch.Tensor | None = None,
        fp8_max_for_normalization: float = F8_E4M3_MAX,
    ) -> torch.Tensor:
        """Pick M=4 or M=6 per block by per-block MSE (paper §3.1, arXiv:2512.02010v5).

        Both candidates share the per-block amax: the M=4 scale equals the M=6 scale times 6/4.
        We round both candidates onto the E2M1 grid (after F8 quantization of the block scale)
        and pick whichever yields lower per-block MSE against the BF16/F32 weight values.

        Inputs:
            weight: original weight tensor [..., features], features divisible by block_size.
            per_block_scale_m6: F32 per-block scale under the default M=6 rule.
                Shape [..., num_blocks].
            weights_scaling_factor_2: per-tensor F32 alpha. Must already use the 4/6-adjusted
                denominator (FP4_E2M1_MAX * F8_E4M3_MAX_46), set by get_weights_scaling_factor_2*.
            block_size: block length (16 for NVFP4).
            per_block_scale_max: optional max scale value for the static-export F8 rescale
                (see _cast_per_block_scale_to_fp8). Pass-through only.

        Returns the per-block scale in F32, with M=4 blocks scaled by 6/4 vs M=6 blocks.
        Same shape as per_block_scale_m6. The caller is responsible for the subsequent
        F8_E4M3 cast.
        """
        ratio = FP4_E2M1_MAX / FP4_E2M1_MAX_M4  # 1.5
        per_block_scale_m4 = per_block_scale_m6 * ratio

        # Round candidate per-block scales to F8_E4M3 precision before scoring — the saved scales
        # are F8 quantized, so MSE under F8-rounded scales is what eventually gets deployed.
        scale_m6_f8 = _cast_per_block_scale_to_fp8(
            per_block_scale_m6,
            per_block_scale_max,
            fp8_max_for_normalization=fp8_max_for_normalization,
        ).to(torch.float32)
        scale_m4_f8 = _cast_per_block_scale_to_fp8(
            per_block_scale_m4,
            per_block_scale_max,
            fp8_max_for_normalization=fp8_max_for_normalization,
        ).to(torch.float32)

        # Quantize-then-dequantize both candidates on the actual weight, compare per-block MSE.
        alpha = weights_scaling_factor_2.to(weight.device).to(torch.float32)
        deq_m6 = cls._fake_quant_to_e2m1(weight, scale_m6_f8, alpha, block_size)
        deq_m4 = cls._fake_quant_to_e2m1(weight, scale_m4_f8, alpha, block_size)

        w_blocks = weight.to(torch.float32).view(*weight.shape[:-1], -1, block_size)
        mse_m6 = ((w_blocks - deq_m6) ** 2).mean(dim=-1)
        mse_m4 = ((w_blocks - deq_m4) ** 2).mean(dim=-1)
        chose_m4 = mse_m4 < mse_m6

        return torch.where(chose_m4, per_block_scale_m4, per_block_scale_m6)

    @classmethod
    def _fake_quant_to_e2m1(
        cls,
        weight: torch.Tensor,
        per_block_scale_f32: torch.Tensor,
        alpha: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        """Round-trip quantize one candidate (scale_block ⊗ alpha) and return dequantized blocks.

        Returns shape [..., num_blocks, block_size] in float32.
        """
        device = weight.device
        w_blocks = weight.to(torch.float32).view(*weight.shape[:-1], -1, block_size)
        scale = per_block_scale_f32.view(*per_block_scale_f32.shape, 1).to(device)
        alpha_v = alpha.to(torch.float32)
        if alpha_v.dim() == 0:
            divisor = scale * alpha_v
        else:
            divisor = scale * alpha_v.view(*alpha_v.shape, *([1] * (scale.dim() - alpha_v.dim())))
        scaled = w_blocks / divisor

        # Sign + abs, then round abs to E2M1 grid using the same bounds as _cast_fp4. Values
        # whose magnitude exceeds the implicit grid max (6.0) are clamped before rounding.
        sign = torch.sign(scaled)
        abs_v = scaled.abs().clamp_(max=FP4_E2M1_MAX)
        bounds = cls.get_e2m1_bounds(device)
        ord_ = torch.searchsorted(bounds, abs_v, out_int32=True)
        # Mirror the equals-bound nudge in _cast_fp4 (round-half-up at odd-indexed bounds)
        odd_bounds = bounds[[1, 3, 5]]
        nudge = torch.any(abs_v.unsqueeze(-1) == odd_bounds, dim=-1).to(ord_.dtype)
        ord_ = ord_ + nudge
        # Map ordinal → magnitude
        e2m1_pos = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=device, dtype=torch.float32
        )
        ord_ = ord_.clamp_(0, 7)
        mag = e2m1_pos[ord_.long()]
        return sign * mag * divisor

    @classmethod
    def get_activation_scaling_factor(cls, quantizer):
        """Returns the activation scaling factor for export."""
        # TODO: Update to use module and not quantizer
        if not quantizer.is_enabled:
            return None

        amax = quantizer.export_amax()

        if amax is None:
            return None

        activation_scaling_factor = amax.float() / (quantizer.maxbound * 448.0)

        assert torch.all(activation_scaling_factor > 0), (
            f" activation scaling factor {activation_scaling_factor} not positive."
        )

        return activation_scaling_factor

    @classmethod
    def _cast_fp4(cls, weight: torch.Tensor):
        """Converts tensor to uint4."""
        device = weight.device

        # Extract sign and compute absolute values in one pass
        sign_bit = (weight < 0).to(torch.uint8)
        weight_abs = weight.abs_()

        # Get bounds and compute ordinal values
        e2m1_bounds = cls.get_e2m1_bounds(device)
        ord = torch.searchsorted(e2m1_bounds, weight_abs, out_int32=True).to(torch.uint8)

        # Efficiently check for rounding at odd-indexed bounds [0.75, 1.75, 2.5]
        # Only need to check bounds at indices 1, 3, 5
        odd_bounds = e2m1_bounds[[1, 3, 5]]  # [0.75, 1.75, 2.5]
        equals_odd_bounds = torch.any(weight_abs.unsqueeze(-1) == odd_bounds, dim=-1).to(
            torch.uint8
        )

        # Combine sign, ordinal, and rounding adjustment
        return (sign_bit << 3) + ord + equals_odd_bounds

    @classmethod
    def quantize(
        cls,
        input: torch.Tensor,
        block_size: int,
        weights_scaling_factor: torch.Tensor | None = None,
        weights_scaling_factor_2: torch.Tensor | None = None,
        keep_high_precision: bool = False,
        try_tensorrt: bool = False,
        four_over_six: bool = False,
    ):
        """Converting a tensor to a quantized format based on NVFP4 quantization.

        Args:
            input (torch.Tensor): The input tensor to be quantized.
            block_size (int): The size of each block for quantization.
            weights_scaling_factor (torch.Tensor): The scaling factor for the weights.
            weights_scaling_factor_2 (torch.Tensor): The scaling factor for the weights.
            keep_high_precision (bool): Whether to keep output scales at high precision.
            four_over_six (bool): Enable per-block M=4 vs M=6 adaptive selection
                (paper arXiv:2512.02010v5 §3.1). Only consulted when
                ``weights_scaling_factor`` is None — otherwise the caller-provided scale
                already encodes the M choice.

        Returns:
        tuple: Contains quantized data, quantized per block scaling factor, and per tensor scaling factor.
        """
        # Get original input shape
        input_shape = input.shape
        input_dtype = input.dtype

        # pad the input if needed
        input = reduce_block_padding(input, block_sizes={-1: block_size})

        if weights_scaling_factor_2 is None:
            weights_scaling_factor_2 = cls.get_weights_scaling_factor_2(
                input, four_over_six=four_over_six
            )

        # try call trtllm fp4 quantization if possible
        if (
            fp4_compatible()
            and weights_scaling_factor is None
            and try_tensorrt
            and not four_over_six
            and block_size == 16
            and input.is_cuda
            and input.dtype in [torch.half, torch.bfloat16]
        ):
            try:
                import tensorrt_llm  # noqa: F401

                # Make sure this utils is available for dequantize
                from tensorrt_llm._torch.auto_deploy.utils.quantization_utils import (
                    cutlass_fp4_scale_to_modelopt_fp4_scale,  # noqa: F401
                )

                packed_weight, weights_scaling_factor = torch.ops.trtllm.fp4_quantize(
                    input, 1.0 / weights_scaling_factor_2, block_size, False
                )
                # weights_scaling_factor is ready for nvfp4_gemm to use;
                # however, it is different from the non trtllm version, so when dequantize,
                # it will be converted.
                return (
                    cls(input_shape, input_dtype, packed_weight),
                    weights_scaling_factor,
                    weights_scaling_factor_2,
                )
            except ImportError:
                pass

        if weights_scaling_factor is None:
            weights_scaling_factor, _ = cls.get_weights_scaling_factor(
                input, block_size, weights_scaling_factor_2, four_over_six=four_over_six
            )

        # Reshape the weight and scale factors
        original_shape = input.shape
        input = input.view((*tuple(input.shape[:-1]), -1, block_size))

        # Scale weights
        scaled_weight = input / (
            (weights_scaling_factor.to(torch.float32) * weights_scaling_factor_2).unsqueeze(-1)
        )

        # Reshape weights to original
        scaled_weight = scaled_weight.view(original_shape)

        if keep_high_precision:
            return scaled_weight
        # Cast weights to fp4
        q_weight = cls._cast_fp4(scaled_weight)
        # Pack weights
        packed_weight = (q_weight[..., 1::2] << 4) | q_weight[..., 0::2]
        return (
            cls(input_shape, input_dtype, packed_weight),
            weights_scaling_factor,
            weights_scaling_factor_2,
        )

    def dequantize(self, dtype: torch.dtype = None, fast=False, **kwarg):
        """Dequantze NVFP4 packed tensor to a target dtype."""
        if dtype is None:
            dtype = self.metadata["dtype"]

        def _unpack_tensor(input: torch.Tensor):
            # Initialize storage for unpacked tensor
            unpacked_shape = list(input.shape)
            unpacked_shape[-1] = unpacked_shape[-1] * 2
            unpacked = torch.empty(unpacked_shape, dtype=dtype, device=input.device)

            unpacked[..., 1::2] = input >> 4
            unpacked[..., 0::2] = input & 0x0F

            unpacked = unpacked.reshape(-1)
            unpacked = self.get_e2m1_values(input.device)[unpacked.long()]

            return unpacked.reshape(unpacked_shape)

        # Get scales from kwargs
        if kwarg["scale"].dtype == torch.uint8 and kwarg["scale"].ndim == 1:
            # If quantization is done by trtllm, convert cutlass fp4 scale to modelopt fp4 scale
            try:
                from tensorrt_llm._torch.auto_deploy.utils.quantization_utils import (
                    cutlass_fp4_scale_to_modelopt_fp4_scale,
                )

                kwarg["scale"] = cutlass_fp4_scale_to_modelopt_fp4_scale(
                    kwarg["scale"], self.metadata["shape"][-2:]
                )
            except ImportError as e:
                raise ImportError(
                    "This tensor is quantized by trtllm, but tensorrt_llm cannot be imported."
                ) from e

        if fast:
            from modelopt.torch.kernels.quantization.gemm.fp4_kernel import fp4_dequantize

            return fp4_dequantize(
                self._quantized_data,
                kwarg["scale"],
                kwarg["double_scale"],
                block_size=kwarg["block_sizes"][-1],
                dtype=dtype,
            ).reshape(self.metadata["shape"])
        else:
            q_per_block_scale = (
                kwarg["scale"].to(torch.float32)
                if kwarg["scale"].dtype == torch.float8_e4m3fn
                else kwarg["scale"]
            )
            block_size = kwarg["block_sizes"][-1]
            per_block_quant_scale = kwarg["double_scale"]

            # Dequantize scales
            per_block_scale = q_per_block_scale * per_block_quant_scale

            # Unpack and unscale weights
            deq_data = _unpack_tensor(self._quantized_data)

            deq_data = deq_data.view(
                (*tuple(deq_data.shape[:-1]), -1, block_size)
            ) * per_block_scale.unsqueeze(-1)
            return deq_data.reshape(self.metadata["shape"]).to(dtype)
