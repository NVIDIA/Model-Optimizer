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
FP8_E4M3_MAX = 448.0
FP8_E4M3_MAX_46 = 256.0

__all__ = ["NVFP4QTensor"]


def _cast_per_block_scale_to_fp8(
    per_block_scale: torch.Tensor,
    per_block_scale_max: torch.Tensor | None = None,
    fp8_max_for_normalization: float = FP8_E4M3_MAX,
) -> torch.Tensor:
    """Clamp to FP8 E4M3FN range [2**-9, 448] and cast — avoids underflow→0 / overflow→NaN.

    When ``per_block_scale_max`` is provided, first rescales as
    ``per_block_scale.float() * fp8_max_for_normalization / per_block_scale_max`` — the static-export
    path needs this because the ``[==0]=1.0`` safety net combined with a small
    ``global_amax`` can drive the rescaled value above 448 (see PR #1397).
    """
    if per_block_scale_max is not None:
        per_block_scale = per_block_scale.float() * fp8_max_for_normalization / per_block_scale_max
    return per_block_scale.clamp(min=2**-9, max=FP8_E4M3_MAX).to(torch.float8_e4m3fn)


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
        m_fp8 = FP8_E4M3_MAX_46 if cls._is_four_over_six(weight_quantizer) else FP8_E4M3_MAX
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
        fp8_max_for_normalization = FP8_E4M3_MAX_46 if is_four_over_six else FP8_E4M3_MAX

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

            # 4/6 M=4/M=6 is folded into _amax during MSE weight calibration.
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

        # four_over_six only selects the 256 FP8 normalization (via scaling_factor_2);
        # the M=4/M=6 choice is made during MSE calibration.
        if not keep_high_precision:
            per_block_scale = _cast_per_block_scale_to_fp8(per_block_scale)
        return per_block_scale, weights_scaling_factor_2

    @classmethod
    def get_weights_scaling_factor_2(cls, input: torch.Tensor, four_over_six: bool = False):
        """Returns per tensor weight scaling factor."""
        m_fp8 = FP8_E4M3_MAX_46 if four_over_six else FP8_E4M3_MAX
        return reduce_amax(input).float() / (FP4_E2M1_MAX * m_fp8)

    @classmethod
    def get_activation_scaling_factor(cls, quantizer):
        """Returns the activation scaling factor for export."""
        # TODO: Update to use module and not quantizer
        if not quantizer.is_enabled:
            return None

        amax = quantizer.export_amax()

        if amax is None:
            return None

        activation_scaling_factor = amax.float() / (quantizer.maxbound * FP8_E4M3_MAX)

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
