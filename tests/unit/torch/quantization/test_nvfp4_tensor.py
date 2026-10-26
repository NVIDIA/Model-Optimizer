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

import pytest
import torch

import modelopt.torch.quantization.qtensor.nvfp4_tensor as nvfp4_tensor
from modelopt.torch.quantization.qtensor.nvfp4_tensor import (
    NVFP4QTensor,
    _cast_per_block_scale_to_fp8,
    cutlass_fp4_scale_to_modelopt_fp4_scale,
)

_FP8_E4M3FN_MIN = 2**-9  # 0.001953125 — smallest positive FP8 E4M3FN subnormal
_FP8_E4M3FN_MAX = 448.0


@pytest.mark.parametrize("try_tensorrt", [False, True])
def test_cpu_quantize_does_not_probe_cuda(monkeypatch, try_tensorrt):
    """The optional TRT-LLM path must not query GPU capability for a CPU tensor."""

    def fail_if_called():
        pytest.fail("fp4_compatible() was called for CPU quantization")

    monkeypatch.setattr(nvfp4_tensor, "fp4_compatible", fail_if_called)
    weight = torch.randn(2, 16, dtype=torch.bfloat16)

    quantized, _, _ = NVFP4QTensor.quantize(weight, block_size=16, try_tensorrt=try_tensorrt)

    assert quantized._quantized_data.shape == (2, 8)


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
            "Per-block scales with zero values found after FP8 cast "
            "(below the FP8 E4M3FN subnormal minimum — clamp would have prevented this)."
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
        """Static path: zero-amax block + small global_amax must clamp to 448, not cast to NaN."""
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


def _cublas_swizzle(scale):
    """Flat cuBLAS tiled layout of a ``[rows, cols]`` block-scale matrix (the 128x4-tile swizzle)."""
    rows, cols = scale.shape
    row_blocks, col_blocks = -(-rows // 128), -(-cols // 4)
    padded = torch.zeros(row_blocks * 128, col_blocks * 4, dtype=scale.dtype)
    padded[:rows, :cols] = scale
    blocks = padded.view(row_blocks, 128, col_blocks, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1)


@pytest.mark.parametrize("weight_shape", [(128, 64), (3, 16), (130, 80), (256, 128)])
def test_cutlass_fp4_scale_to_modelopt_fp4_scale_undoes_the_swizzle(weight_shape):
    m, n = weight_shape
    scale = torch.randint(0, 256, (m, n // 16), dtype=torch.uint8)

    restored = cutlass_fp4_scale_to_modelopt_fp4_scale(_cublas_swizzle(scale), weight_shape)

    assert restored.dtype == torch.float8_e4m3fn
    assert restored.is_contiguous()  # consumers index it flatly
    assert torch.equal(restored.view(torch.uint8), scale)
