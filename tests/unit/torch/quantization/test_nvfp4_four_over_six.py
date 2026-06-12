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

"""CPU tests for NVFP4 Four-Over-Six (4/6) adaptive weight scaling.

4/6 is a weight-only refinement applied under max calibration: it uses an FP8
normalization max of 256 (instead of 448) and, per block, picks an M=4 scale
candidate (the M=6 scale times 6/4) when it lowers per-block reconstruction MSE
(arXiv:2512.02010). It is enabled via ``block_sizes={"four_over_six": True}`` on
(static, max-calibrated) weight quantizers.
"""

from types import SimpleNamespace

import torch

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.config import choices
from modelopt.torch.quantization.qtensor.nvfp4_tensor import (
    FP4_E2M1_MAX,
    FP4_E2M1_MAX_M4,
    FP8_E4M3_MAX,
    FP8_E4M3_MAX_46,
    NVFP4QTensor,
    _cast_per_block_scale_to_fp8,
)

BLOCK_SIZE = 16


def _per_block_amax(weight: torch.Tensor, block_size: int) -> torch.Tensor:
    """Per-block amax via plain reshape, matching reduce_block_amax on the last axis."""
    blocks = weight.abs().view(*weight.shape[:-1], -1, block_size)
    return blocks.amax(dim=-1).float()


class TestConstants:
    def test_fp8_and_e2m1_constants(self):
        assert FP8_E4M3_MAX == 448.0
        assert FP8_E4M3_MAX_46 == 256.0
        assert FP4_E2M1_MAX == 6.0
        assert FP4_E2M1_MAX_M4 == 4.0

    def test_m4_over_m6_ratio_is_1_5(self):
        assert FP4_E2M1_MAX / FP4_E2M1_MAX_M4 == 1.5


class TestIsFourOverSix:
    def test_flag_true(self):
        q = SimpleNamespace(block_sizes={-1: BLOCK_SIZE, "four_over_six": True})
        assert NVFP4QTensor._is_four_over_six(q) is True

    def test_flag_false(self):
        q = SimpleNamespace(block_sizes={-1: BLOCK_SIZE, "four_over_six": False})
        assert NVFP4QTensor._is_four_over_six(q) is False

    def test_flag_absent_defaults_false(self):
        q = SimpleNamespace(block_sizes={-1: BLOCK_SIZE})
        assert NVFP4QTensor._is_four_over_six(q) is False

    def test_missing_block_sizes_defaults_false(self):
        assert NVFP4QTensor._is_four_over_six(SimpleNamespace()) is False
        assert NVFP4QTensor._is_four_over_six(SimpleNamespace(block_sizes=None)) is False


class TestScalingFactor2:
    def test_256_vs_448_denominator(self):
        torch.manual_seed(0)
        w = torch.randn(8, 4 * BLOCK_SIZE)
        wsf2_default = NVFP4QTensor.get_weights_scaling_factor_2(w, four_over_six=False)
        wsf2_46 = NVFP4QTensor.get_weights_scaling_factor_2(w, four_over_six=True)
        # wsf2 = amax / (6 * m_fp8); only m_fp8 differs (448 vs 256).
        assert torch.allclose(wsf2_46 / wsf2_default, torch.tensor(448.0 / 256.0), rtol=1e-6)


class TestSelectFourOverSixScale:
    def _setup(self, seed=0, rows=8, n_blocks=4):
        torch.manual_seed(seed)
        weight = torch.randn(rows, n_blocks * BLOCK_SIZE)
        wsf2 = NVFP4QTensor.get_weights_scaling_factor_2(weight, four_over_six=True)
        per_block_amax = _per_block_amax(weight, BLOCK_SIZE)
        per_block_scale_m6 = per_block_amax / (FP4_E2M1_MAX * wsf2)
        per_block_scale_m6[per_block_scale_m6 == 0] = 1.0
        return weight, wsf2, per_block_scale_m6

    def test_returns_m6_or_m4_candidate(self):
        weight, wsf2, m6 = self._setup()
        selected = NVFP4QTensor._select_four_over_six_scale(weight, m6, wsf2, BLOCK_SIZE)
        m4 = m6 * (FP4_E2M1_MAX / FP4_E2M1_MAX_M4)
        is_m6 = torch.isclose(selected, m6)
        is_m4 = torch.isclose(selected, m4)
        assert (is_m6 | is_m4).all(), "Selected scale is neither the M=6 nor the M=4 candidate."

    def test_selection_never_increases_block_mse(self):
        """Adaptive M=4/M=6 selection must not raise per-block MSE vs M=6 only (same alpha)."""
        weight, wsf2, _ = self._setup(seed=3, rows=16, n_blocks=8)
        # Both candidates share the same per-tensor alpha (wsf2); only per-block scale differs.
        m6_scale, _ = NVFP4QTensor.get_weights_scaling_factor(
            weight, BLOCK_SIZE, wsf2, keep_high_precision=True, four_over_six=False
        )
        sel_scale, _ = NVFP4QTensor.get_weights_scaling_factor(
            weight, BLOCK_SIZE, wsf2, keep_high_precision=True, four_over_six=True
        )
        alpha = wsf2.float()
        deq_m6 = NVFP4QTensor._fake_quant_to_e2m1(
            weight, _cast_per_block_scale_to_fp8(m6_scale).float(), alpha, BLOCK_SIZE
        )
        deq_sel = NVFP4QTensor._fake_quant_to_e2m1(
            weight,
            _cast_per_block_scale_to_fp8(
                sel_scale, fp8_max_for_normalization=FP8_E4M3_MAX_46
            ).float(),
            alpha,
            BLOCK_SIZE,
        )
        w_blocks = weight.float().view(*weight.shape[:-1], -1, BLOCK_SIZE)
        mse_m6 = ((w_blocks - deq_m6) ** 2).mean(dim=-1)
        mse_sel = ((w_blocks - deq_sel) ** 2).mean(dim=-1)
        assert (mse_sel <= mse_m6 + 1e-12).all(), "4/6 selection increased per-block MSE."

    def test_chooses_m4_when_strictly_better(self):
        """At least one block should pick M=4 on random data (else selection is a no-op)."""
        weight, wsf2, _ = self._setup(seed=7, rows=32, n_blocks=8)
        m6_scale, _ = NVFP4QTensor.get_weights_scaling_factor(
            weight, BLOCK_SIZE, wsf2, keep_high_precision=True, four_over_six=False
        )
        sel_scale, _ = NVFP4QTensor.get_weights_scaling_factor(
            weight, BLOCK_SIZE, wsf2, keep_high_precision=True, four_over_six=True
        )
        assert not torch.allclose(sel_scale, m6_scale), "Expected some blocks to switch to M=4."


class TestRoundTripScales:
    def test_no_zero_or_nan_scales(self):
        torch.manual_seed(1)
        weight = torch.cat([torch.randn(4, BLOCK_SIZE), torch.full((4, BLOCK_SIZE), 1e-12)], dim=0)
        per_block_scale, _ = NVFP4QTensor.get_weights_scaling_factor(
            weight, BLOCK_SIZE, four_over_six=True
        )
        s = per_block_scale.float()
        assert torch.isfinite(s).all(), f"Non-finite 4/6 scales: {s.tolist()}"
        assert (s > 0).all(), f"Zero 4/6 scales: {s.tolist()}"


class TestNVFP4FourOverSixConfig:
    @staticmethod
    def _block_sizes(cfg, name):
        entry = next(e for e in cfg["quant_cfg"] if e["quantizer_name"] == name)
        return entry["cfg"]["block_sizes"]

    def test_weight_quantizer_is_static_with_four_over_six(self):
        bs = self._block_sizes(mtq.NVFP4_FOUR_OVER_SIX_CFG, "*weight_quantizer")
        assert bs.get("type") == "static"
        # Schema coerces the bool to int 1; the feature reads it truthily.
        assert bs.get("four_over_six")

    def test_input_quantizer_unchanged(self):
        bs = self._block_sizes(mtq.NVFP4_FOUR_OVER_SIX_CFG, "*input_quantizer")
        assert not bs.get("four_over_six", False)

    def test_registered_in_choices(self):
        assert "NVFP4_FOUR_OVER_SIX_CFG" in choices
