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

"""Tests for the ``nvfp4_act_headroom`` activation global-scale calibration."""

import pytest
import torch

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.calib import NVFP4ActHeadroomCalibrator
from modelopt.torch.quantization.model_calib import (
    _is_nvfp4_dynamic_input_quantizer,
    _swap_in_nvfp4_act_headroom_calibrators,
)
from modelopt.torch.quantization.nn import TensorQuantizer

NVFP4_CFG = {
    "num_bits": (2, 1),
    "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
}

ACT_ONLY_CFG = {
    "quant_cfg": [
        {"quantizer_name": "*", "enable": False},
        {"quantizer_name": "*input_quantizer", "cfg": NVFP4_CFG},
    ],
    "algorithm": {"method": "nvfp4_act_headroom", "anchor_percentile": 1},
}


class _Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(64, 64)
        self.fc2 = torch.nn.Linear(64, 64)

    def forward(self, x):
        return self.fc2(self.fc1(x))


def _calibrate(calibrator, x):
    calibrator.collect(x)
    return float(calibrator.compute_amax())


def test_amax_is_rho_times_anchor():
    """On a uniform per-block distribution the anchor term sets amax = rho * anchor."""
    torch.manual_seed(0)
    # All blocks share one magnitude, so anchor == floor and rho * anchor dominates.
    x = torch.full((256, 64), 0.5)
    amax = _calibrate(NVFP4ActHeadroomCalibrator(rho=1024.0, anchor_percentile=1.0), x)
    assert amax == pytest.approx(1024.0 * 0.5, rel=0.05)


def test_amax_never_below_calibrated_max():
    """A range wider than FP8 can hold warns and falls back to the no-clipping floor."""
    x = torch.zeros(64, 64)
    x[0, :16] = 100.0  # one large block, the rest tiny
    x[1:, :] = 1e-3
    with pytest.warns(UserWarning, match="exceeds the FP8 scale range"):
        amax = _calibrate(NVFP4ActHeadroomCalibrator(rho=1.0, anchor_percentile=1.0), x)
    assert amax >= 100.0 * 0.9


def test_lower_anchor_percentile_gives_smaller_amax():
    """anchor_percentile is tunable and monotone: a lower percentile anchors lower."""
    torch.manual_seed(0)
    x = torch.randn(512, 64).abs() + 1e-4
    amax_p1 = _calibrate(NVFP4ActHeadroomCalibrator(anchor_percentile=1.0), x)
    amax_p50 = _calibrate(NVFP4ActHeadroomCalibrator(anchor_percentile=50.0), x)
    assert amax_p1 < amax_p50


def test_headroom_exceeds_plain_max():
    """The calibrated scale leaves headroom above what plain max would pick."""
    torch.manual_seed(0)
    x = torch.randn(512, 64).abs() + 1e-4
    amax = _calibrate(NVFP4ActHeadroomCalibrator(anchor_percentile=1.0), x)
    assert amax > float(x.abs().max())


def test_all_zero_activation_yields_no_scale():
    """An all-zero activation carries no scale information, so no amax is inferred."""
    calibrator = NVFP4ActHeadroomCalibrator()
    calibrator.collect(torch.zeros(32, 64))
    assert calibrator.compute_amax() is None


def test_rho_out_of_range_rejected():
    with pytest.raises(ValueError, match="rho must be in"):
        NVFP4ActHeadroomCalibrator(rho=28672.0)


def test_reset_clears_state():
    calibrator = NVFP4ActHeadroomCalibrator()
    calibrator.collect(torch.randn(32, 64).abs() + 1e-3)
    calibrator.reset()
    assert calibrator.compute_amax() is None


def test_only_nvfp4_input_quantizers_are_selected():
    """The swap targets NVFP4 dynamic-block input quantizers, not weights or other formats."""
    nvfp4_in = TensorQuantizer(mtq.config.QuantizerAttributeConfig(**NVFP4_CFG))
    assert _is_nvfp4_dynamic_input_quantizer("layer.input_quantizer", nvfp4_in)
    # Same config but a weight quantizer -> not selected.
    assert not _is_nvfp4_dynamic_input_quantizer("layer.weight_quantizer", nvfp4_in)
    # FP8 input quantizer -> not selected.
    fp8_in = TensorQuantizer(mtq.config.QuantizerAttributeConfig(num_bits=(4, 3)))
    assert not _is_nvfp4_dynamic_input_quantizer("layer.input_quantizer", fp8_in)


def test_swap_installs_calibrator_with_config_values():
    model = mtq.quantize(_Net(), ACT_ONLY_CFG, lambda m: m(torch.randn(8, 64)))
    n = _swap_in_nvfp4_act_headroom_calibrators(model, anchor_percentile=5.0, rho=1024.0)
    assert n == 2  # fc1 and fc2 input quantizers
    cal = model.fc1.input_quantizer._calibrator
    assert isinstance(cal, NVFP4ActHeadroomCalibrator)
    assert cal._anchor_percentile == 5.0
    assert cal._rho == 1024.0


def test_activation_only_quantize_leaves_weights_untouched():
    """End-to-end: input quantizers get a scale, weight quantizers stay disabled."""
    torch.manual_seed(0)
    model = _Net()
    weights_before = {n: p.clone() for n, p in model.named_parameters()}

    model = mtq.quantize(model, ACT_ONLY_CFG, lambda m: m(torch.randn(16, 64)))

    for name in ("fc1", "fc2"):
        layer = getattr(model, name)
        assert layer.input_quantizer.is_enabled
        assert layer.input_quantizer.amax is not None
        assert float(layer.input_quantizer.amax) > 0
        assert not layer.weight_quantizer.is_enabled

    for n, p in model.named_parameters():
        if n in weights_before:
            assert torch.equal(p, weights_before[n]), f"{n} was modified"


def test_w4a4_weights_use_max_activations_use_headroom():
    """W4A4: weights fall back to plain max, only activations get the headroom scale."""
    torch.manual_seed(0)
    data = torch.randn(16, 64)
    w4a4_cfg = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {"quantizer_name": "*weight_quantizer", "cfg": NVFP4_CFG},
            {"quantizer_name": "*input_quantizer", "cfg": NVFP4_CFG},
        ],
        "algorithm": {"method": "nvfp4_act_headroom", "anchor_percentile": 1},
    }
    model = _Net()
    weight_max = float(model.fc1.weight.abs().max())
    model = mtq.quantize(model, w4a4_cfg, lambda m: m(data))

    # Weight quantizer is calibrated with plain max: amax == the literal weight max.
    assert model.fc1.weight_quantizer.is_enabled
    assert float(model.fc1.weight_quantizer.amax) == pytest.approx(weight_max, rel=1e-5)
    # Activation quantizer gets the headroom scale, well above the observed activation max.
    assert float(model.fc1.input_quantizer.amax) > float(data.abs().max())


def test_anchor_percentile_changes_model_scales():
    """The knob propagates through mtq.quantize to the calibrated activation scales."""
    torch.manual_seed(0)
    data = torch.randn(16, 64)

    def _amax_for(percentile):
        torch.manual_seed(0)
        cfg = {
            **ACT_ONLY_CFG,
            "algorithm": {"method": "nvfp4_act_headroom", "anchor_percentile": percentile},
        }
        model = mtq.quantize(_Net(), cfg, lambda m: m(data))
        return float(model.fc1.input_quantizer.amax)

    assert _amax_for(1) < _amax_for(50)
