# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Unit tests for the NVFP4 four-over-six fake quantization kernel."""

import pytest
import torch

from modelopt.torch.quantization.calib.fouroversix import nvfp4_4o6_fake_quant


def _amax(x: torch.Tensor) -> torch.Tensor:
    return x.abs().max().float()


class TestOutputShapeAndDtype:
    def test_output_shape_and_dtype(self):
        x = torch.randn(128, 64, dtype=torch.bfloat16)
        out = nvfp4_4o6_fake_quant(x, _amax(x))
        assert out.shape == x.shape
        assert out.dtype == x.dtype


class TestZeroTensor:
    def test_zero_tensor(self):
        x = torch.zeros(64, 32, dtype=torch.float32)
        out = nvfp4_4o6_fake_quant(x, _amax(x))
        assert out.shape == x.shape
        assert torch.all(out == 0)


class TestAllScaleRules:
    @pytest.mark.parametrize("scale_rule", ["mse", "mae", "abs_max"])
    def test_all_scale_rules(self, scale_rule):
        x = torch.randn(64, 64, dtype=torch.float32)
        out = nvfp4_4o6_fake_quant(x, _amax(x), scale_rule=scale_rule)
        assert out.shape == x.shape
        assert not torch.any(torch.isnan(out))


class TestMseLowerErrorThanStatic6:
    def test_mse_lower_error_than_static6(self):
        """4o6-MSE error should be <= static-6 MSE on a tensor with outlier blocks."""
        torch.manual_seed(0)
        # Create a tensor where some blocks have large outliers
        x = torch.randn(128, 64, dtype=torch.float32)
        # Inject outliers into several blocks
        x[0, :] *= 100
        x[16, :] *= 50
        x[64, :] *= 200

        x_amax = _amax(x)

        # 4o6 output
        out_4o6 = nvfp4_4o6_fake_quant(x, x_amax, scale_rule="mse")

        # Static-6 reference: use quantize_to_nvfp4 directly with no scale_expansion
        from modelopt.torch.quantization.calib.fouroversix import (
            _fake_quantize_to_e2m1,
            _quantize_to_nvfp4,
        )

        x_blocks = x.reshape(-1, 16).float()
        x_scaled_6, scales_6 = _quantize_to_nvfp4(x_blocks, x_amax)
        x_fq_6 = _fake_quantize_to_e2m1(x_scaled_6)
        denom = 6 * 256  # _E2M1_MAX * _E4M3_MAX_FOUROVERSIX
        out_static6 = (
            x_fq_6 * scales_6.unsqueeze(1).to(torch.float32) * x_amax / denom
        ).reshape_as(x)

        mse_4o6 = ((out_4o6 - x) ** 2).mean().item()
        mse_static6 = ((out_static6 - x) ** 2).mean().item()

        assert mse_4o6 <= mse_static6, (
            f"4o6 MSE ({mse_4o6:.6f}) should be <= static-6 MSE ({mse_static6:.6f})"
        )


class TestRoundtripNearIdentity:
    def test_roundtrip_near_identity(self):
        """For a slowly-varying tensor, output should be finite and close in magnitude.

        E2M1 4-bit float has only 8 positive representable levels (0, 0.5, 1, 1.5, 2, 3, 4, 6),
        so per-element relative error can reach ~25% even for well-conditioned inputs.
        We verify the mean relative error is reasonable (< 25%).
        """
        x = torch.linspace(0.1, 1.0, 128).reshape(8, 16).float()
        out = nvfp4_4o6_fake_quant(x, _amax(x))
        assert torch.all(torch.isfinite(out)), "Output contains non-finite values"
        mean_rel_err = ((out - x).abs() / (x.abs() + 1e-8)).mean().item()
        assert mean_rel_err < 0.25, f"Mean relative error {mean_rel_err:.4f} exceeds 25%"


class TestNonDivisibleLength:
    def test_non_divisible_length(self):
        """Input whose numel is not divisible by block_size should raise ValueError."""
        x = torch.randn(7, 5, dtype=torch.float32)  # 35 elements, not divisible by 16
        with pytest.raises(ValueError, match="divisible"):
            nvfp4_4o6_fake_quant(x, _amax(x))
