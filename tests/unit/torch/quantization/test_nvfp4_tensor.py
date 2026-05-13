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

"""Tests for NVFP4QTensor per-block FP8 scale clamping (underflow + overflow)."""

from types import SimpleNamespace

import torch

from modelopt.torch.quantization.qtensor.nvfp4_tensor import (
    NVFP4QTensor,
    _cast_per_block_scale_to_fp8,
)

_FP8_E4M3FN_MIN = 2**-9  # 0.001953125 — smallest positive FP8 E4M3FN subnormal
_FP8_E4M3FN_MAX = 448.0


class TestNVFP4ScaleClamping:
    """Per-block weight scales outside the FP8 E4M3FN range must be clamped, not turned into 0/NaN."""

    def test_no_zero_scales_for_tiny_weights(self):
        """Tiny per-block amax (<<FP8 min) must not underflow to zero after FP8 cast."""
        block_size = 16
        tiny_weight = torch.full((4, block_size), 1e-10)
        # wsf2=1.0 → per_block_scale = amax/(6*wsf2) ≈ 1.7e-11 << 2^-9, exercises FP8-min clamp
        wsf2 = torch.tensor(1.0)

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

    def test_helper_clamps_overflow_to_max(self):
        """Values above 448 must saturate to 448, not cast to NaN (fp8_e4m3fn has no Inf)."""
        oversized = torch.tensor([100.0, 448.0, 1e3, 1e6])
        out = _cast_per_block_scale_to_fp8(oversized).float()
        assert torch.isfinite(out).all(), f"FP8 cast produced non-finite values: {out.tolist()}"
        assert (out <= _FP8_E4M3FN_MAX).all(), f"FP8 cast values exceed 448: {out.tolist()}"

    def test_helper_clamps_underflow_to_min(self):
        """Values below the FP8 subnormal must clamp up, not collapse to 0."""
        tiny = torch.tensor([0.0, 1e-12, 1e-6, _FP8_E4M3FN_MIN / 2])
        out = _cast_per_block_scale_to_fp8(tiny).float()
        assert (out > 0).all(), f"FP8 cast produced zero scales: {out.tolist()}"

    def test_static_path_no_nan_when_block_amax_zero(self):
        """Static path: when a block's amax is 0 (all-zero weights), the `[==0]=1.0` safety net
        and a small global_amax push the pre-cast value above 448. Without the max clamp,
        fp8_e4m3fn would cast it to NaN — regression for the export-time NaN reported on this PR.
        """
        block_size = 16
        # global_amax small enough that 1.0 * 448 / (global_amax/6) >> 448.
        global_amax = torch.tensor(0.01)
        # One block with amax=0 (triggers safety net), three normal blocks.
        per_block_amax = torch.tensor([[0.0, 0.005, 0.008, 0.01]])
        weight = torch.randn(1, 4 * block_size)
        q = SimpleNamespace(
            global_amax=global_amax,
            _amax=per_block_amax,
            block_sizes={-1: block_size},
        )

        per_block_scale, _ = NVFP4QTensor.get_weights_scaling_factor_from_quantizer(q, weight)
        per_block_scale_f32 = per_block_scale.float()
        assert torch.isfinite(per_block_scale_f32).all(), (
            f"NaN/Inf in exported static per-block scale: {per_block_scale_f32.tolist()}"
        )
        assert (per_block_scale_f32 <= _FP8_E4M3FN_MAX).all(), (
            f"Static per-block scale exceeds FP8 max 448: {per_block_scale_f32.tolist()}"
        )
