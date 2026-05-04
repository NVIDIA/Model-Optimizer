# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for NVFP4QTensor per-block FP8 scale underflow clamping."""

import torch

from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor

_FP8_E4M3FN_MIN = 2**-9  # 0.001953125 — smallest positive FP8 E4M3FN subnormal


class TestNVFP4ScaleClamping:
    """Per-block weight scales below the FP8 E4M3FN minimum must be clamped, not rounded to zero."""

    def test_no_zero_scales_for_tiny_weights(self):
        """Tiny per-block amax (<<FP8 min) must not underflow to zero after FP8 cast."""
        block_size = 16
        tiny_weight = torch.full((4, block_size), 1e-10)
        wsf2 = torch.tensor(1e-10 / (6.0 * 448.0))

        per_block_scale, _ = NVFP4QTensor.get_weights_scaling_factor(tiny_weight, block_size, wsf2)
        per_block_scale_f32 = per_block_scale.float()

        assert (per_block_scale_f32 > 0).all(), (
            f"Zero per-block scales found after FP8 cast: {per_block_scale_f32.tolist()}. "
            "FP8 scale underflow clamping likely regressed."
        )
        assert (per_block_scale_f32 >= _FP8_E4M3FN_MIN).all(), (
            "Per-block scales below FP8 minimum subnormal found after cast."
        )

    def test_normal_weights_unaffected_by_clamp(self):
        """Weights with typical magnitudes must not be affected by the underflow clamp."""
        block_size = 16
        torch.manual_seed(42)
        normal_weight = torch.randn(8, block_size)

        per_block_scale, _ = NVFP4QTensor.get_weights_scaling_factor(normal_weight, block_size)
        assert (per_block_scale.float() > 0).all(), "Normal weights produced zero scales."

    def test_mixed_weight_no_zeros(self):
        """Mixed-magnitude tensor (normal + tiny blocks) must have no zero scales."""
        block_size = 16
        weight = torch.cat(
            [
                torch.randn(4, block_size),
                torch.full((4, block_size), 1e-12),
            ],
            dim=0,
        )

        per_block_scale, _ = NVFP4QTensor.get_weights_scaling_factor(weight, block_size)
        assert (per_block_scale.float() > 0).all(), (
            "Zero scales in mixed-magnitude tensor after FP8 cast."
        )
