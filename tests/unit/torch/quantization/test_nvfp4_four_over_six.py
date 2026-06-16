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

4/6 is weight-only: the ``four_over_six: True`` block_sizes flag selects the 256 FP8
normalization max (vs 448); the per-block M=6 vs M=4 choice is made by MSE weight
calibration (arXiv:2512.02010).
"""

from types import SimpleNamespace

import torch

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.config import QuantizerAttributeConfig, choices
from modelopt.torch.quantization.nn import NVFP4StaticQuantizer
from modelopt.torch.quantization.qtensor.nvfp4_tensor import (
    FP4_E2M1_MAX,
    FP4_E2M1_MAX_M4,
    FP8_E4M3_MAX,
    FP8_E4M3_MAX_46,
    NVFP4QTensor,
)

BLOCK_SIZE = 16


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


class TestStaticQuantizerFourOverSixThreading:
    """NVFP4StaticQuantizer._fake_quantize threads fp8_max_for_normalization from the
    four_over_six flag: 256 when enabled, 448 otherwise.

    The per-block M=6/M=4 choice itself is made by MSE calibration.
    """

    @staticmethod
    def _make_static_quantizer(four_over_six: bool) -> NVFP4StaticQuantizer:
        block_sizes = {-1: BLOCK_SIZE, "type": "static", "scale_bits": (4, 3)}
        if four_over_six:
            block_sizes["four_over_six"] = True
        cfg = QuantizerAttributeConfig(num_bits=(2, 1), block_sizes=block_sizes)
        q = NVFP4StaticQuantizer(quant_attribute_cfg=cfg)
        q.amax = torch.full((1, 4), 0.5)
        q.global_amax = torch.tensor(2.0)
        return q

    def _captured_fp8_max(self, monkeypatch, four_over_six: bool) -> float:
        import modelopt.torch.quantization.nn.modules.tensor_quantizer as tqm

        captured = {}

        def spy(*args, **kwargs):
            # Call site: (inputs, amax, global_amax, quantize_block_scales,
            #             fp8_max_for_normalization, dtype, pass_through_bwd).
            # The 4/6 → 256 vs 448 selection happens before this call, so capturing the
            # threaded value is enough; return a passthrough to avoid the triton kernel
            # (unavailable on CPU) — this tests the threading, not the kernel.
            captured["fp8_max"] = args[4]
            return args[0]

        monkeypatch.setattr(tqm, "static_blockwise_fp4_fake_quant", spy)
        q = self._make_static_quantizer(four_over_six)
        q._fake_quantize(torch.randn(1, 4 * BLOCK_SIZE))
        return captured["fp8_max"]

    def test_four_over_six_threads_256(self, monkeypatch):
        assert self._captured_fp8_max(monkeypatch, four_over_six=True) == 256.0

    def test_default_threads_448(self, monkeypatch):
        assert self._captured_fp8_max(monkeypatch, four_over_six=False) == 448.0
