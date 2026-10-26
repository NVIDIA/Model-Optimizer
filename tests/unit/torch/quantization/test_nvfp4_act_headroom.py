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

"""Tests for the ``nvfp4_act_headroom`` activation global-scale calibration."""

import warnings

import pytest
import torch

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization import model_calib
from modelopt.torch.quantization.calib import NVFP4ActHeadroomCalibrator
from modelopt.torch.quantization.model_calib import (
    _is_nvfp4_dynamic_input_quantizer,
    _swap_in_nvfp4_act_headroom_calibrators,
    max_calibrate,
)
from modelopt.torch.quantization.nn import SequentialQuantizer, TensorQuantizer
from modelopt.torch.quantization.utils import reduce_block_amax

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


def test_range_too_wide_for_rho_falls_back_to_the_calibrated_top():
    """A range too wide for rho warns and falls back to the top of the calibrated range."""
    x = torch.zeros(64, 64)
    x[0, :16] = 100.0  # one large block, the rest tiny
    x[1:, :] = 1e-3
    with pytest.warns(UserWarning, match="leaves no headroom"):
        amax = _calibrate(NVFP4ActHeadroomCalibrator(rho=1.0, anchor_percentile=1.0), x)
    assert amax == pytest.approx(100.0, rel=0.06)  # within one histogram bin of the top block


@pytest.mark.parametrize(
    "case",
    [
        "gaussian",
        "heavy_tail",
        "sparse",
        "outlier",
        "wide_dynamic_range",
        "constant",
    ],
)
def test_upper_percentile_100_never_clips(case):
    """upper_percentile=100 is the documented no-clipping setting; verify it holds."""
    g = torch.Generator().manual_seed(7)
    if case == "gaussian":
        x = torch.randn(512, 256, generator=g)
    elif case == "heavy_tail":
        x = torch.randn(512, 256, generator=g) * torch.exp(torch.randn(512, 256, generator=g) * 2)
    elif case == "sparse":
        x = torch.randn(512, 256, generator=g) * (torch.rand(512, 256, generator=g) > 0.9)
    elif case == "outlier":
        x = torch.randn(512, 256, generator=g)
        x[0, 0] = 5000.0
    elif case == "wide_dynamic_range":
        # Per-token magnitudes spanning ~6 decades: the regime that trips the guardrail.
        x = torch.randn(2048, 1024, generator=g) * torch.logspace(-4, 2, 2048).unsqueeze(1)
    else:
        x = torch.full((256, 256), 0.5)

    cal = NVFP4ActHeadroomCalibrator(anchor_percentile=1.0, upper_percentile=100.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        amax = _calibrate(cal, x)
    observed_max = float(reduce_block_amax(x, block_sizes={-1: 16}).max())
    assert amax >= observed_max, f"{case}: amax {amax} clips observed max {observed_max}"


def test_default_clips_rare_outliers_to_protect_the_bulk():
    """A lone freak block is clipped by default; chasing it would flush the tensor to zero."""
    g = torch.Generator().manual_seed(11)
    x = torch.randn(4096, 256, generator=g).abs() + 1e-3
    x[0, :16] = 3e7  # one block ~7 orders of magnitude above the rest

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        default = _calibrate(NVFP4ActHeadroomCalibrator(anchor_percentile=1.0), x)
        literal = _calibrate(
            NVFP4ActHeadroomCalibrator(anchor_percentile=1.0, upper_percentile=100.0), x
        )

    # The default ignores the outlier, keeping the scale near the bulk of the distribution.
    assert default < 3e7
    # Honouring it instead drags the scale orders of magnitude higher, which is exactly the
    # exposure upper_percentile=100 opts into.
    assert literal > 100 * default


def test_upper_percentile_out_of_range_rejected():
    with pytest.raises(ValueError, match="upper_percentile must be in"):
        NVFP4ActHeadroomCalibrator(upper_percentile=0.0)


def test_anchor_percentile_zero_rejected():
    """percentile 0 would defeat the low-tail mask, so it is rejected up front."""
    with pytest.raises(ValueError, match="anchor_percentile must be in"):
        NVFP4ActHeadroomCalibrator(anchor_percentile=0.0)


def test_ragged_last_dim_is_handled():
    """Activations whose last dim is not a multiple of the block size must not crash."""
    g = torch.Generator().manual_seed(3)
    x = torch.randn(32, 70, generator=g)  # 70 % 16 != 0
    amax = _calibrate(NVFP4ActHeadroomCalibrator(anchor_percentile=1.0), x)
    assert amax > 0
    assert amax >= float(x.abs().max())


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nan_inf_activations_rejected(bad):
    """NaN/Inf must be reported rather than silently mis-binned."""
    x = torch.randn(32, 64)
    x[0, 0] = bad
    with pytest.raises(AssertionError, match=r"nan|inf"):
        NVFP4ActHeadroomCalibrator().collect(x)


def test_calibrator_is_restored_after_calibration():
    """The swapped calibrator must not leak into later calibrations of the same model."""
    data = torch.randn(16, 64)
    model = mtq.quantize(_Net(), ACT_ONLY_CFG, lambda m: m(data))
    assert not isinstance(model.fc1.input_quantizer._calibrator, NVFP4ActHeadroomCalibrator)

    headroom_amax = float(model.fc1.input_quantizer.amax)
    # A subsequent plain-max calibration must produce a plain-max scale, not a headroom one.
    max_calibrate(model, lambda m: m(data))
    assert float(model.fc1.input_quantizer.amax) < headroom_amax


def _spy_on(name, calls, stand_in=None):
    """Replace a model_calib entry point with a recording spy, returning the original."""
    real = getattr(model_calib, name)
    run = stand_in if stand_in is not None else real

    def spy(model, forward_loop=None, **kwargs):
        calls.append((name, kwargs))
        return run(model, forward_loop)

    setattr(model_calib, name, spy)
    return real


def _quantize_with(weight_scale_algorithm, calls, spy_name, stand_in=None):
    cfg = {**ACT_ONLY_CFG, "algorithm": {**ACT_ONLY_CFG["algorithm"]}}
    if weight_scale_algorithm is not None:
        cfg["algorithm"]["weight_scale_algorithm"] = weight_scale_algorithm
    real = _spy_on(spy_name, calls, stand_in)
    try:
        torch.manual_seed(0)
        return mtq.quantize(_Net(), cfg, lambda m: m(torch.randn(16, 64)))
    finally:
        setattr(model_calib, spy_name, real)


def test_weight_scale_algorithm_defaults_to_max():
    """Choosing this activation policy must not change how weights are calibrated."""
    calls = []
    model = _quantize_with(None, calls, "max_calibrate")
    assert [c[0] for c in calls] == ["max_calibrate"]
    # Sparse dispatch: none of the nested config's unset defaults are forwarded as kwargs.
    assert calls[0][1] == {}
    assert float(model.fc1.input_quantizer.amax) > 0


def test_weight_scale_algorithm_dispatches_to_mse_with_its_options():
    """A different weight algorithm is selectable, and the activation scale is unaffected.

    The real MSE weight pass needs triton (static NVFP4) or CUDA (dynamic), so this asserts the
    dispatch and option plumbing and that the headroom activation scale still lands.
    """
    calls = []
    model = _quantize_with(
        {"method": "mse", "fp8_scale_sweep": True},
        calls,
        "mse_calibrate",
        stand_in=model_calib.max_calibrate,
    )
    assert [c[0] for c in calls] == ["mse_calibrate"]
    assert calls[0][1] == {"fp8_scale_sweep": True}
    data_max = float(torch.randn(16, 64).abs().max())
    assert float(model.fc1.input_quantizer.amax) > data_max  # still the headroom scale


def test_weight_scale_algorithm_owns_the_shared_state_knobs():
    """distributed_sync / shared_states / sync_expert_weight_amax belong to the weight pass."""
    calls = []
    _quantize_with(
        {
            "method": "max",
            "sync_expert_weight_amax": True,
            "shared_states": {"weight_global_amax": {"patterns": [r".*fc1"]}},
        },
        calls,
        "max_calibrate",
    )
    assert calls[0][1] == {
        "sync_expert_weight_amax": True,
        "shared_states": {"weight_global_amax": {"patterns": [r".*fc1"]}},
    }


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


def test_sequential_quantizer_activation_is_rejected():
    """SequentialQuantizer activations are unsupported and must fail loudly, not silently.

    Their leaves are submodules named ``...input_quantizer.<i>``, so a name-based match would
    skip them and leave those activations on plain max without any signal.
    """
    cfg = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {"quantizer_name": "*input_quantizer", "cfg": [NVFP4_CFG, NVFP4_CFG]},
        ],
        "algorithm": {"method": "nvfp4_act_headroom", "anchor_percentile": 1},
    }
    torch.manual_seed(0)
    with pytest.raises(NotImplementedError, match="does not support SequentialQuantizer"):
        mtq.quantize(_Net(), cfg, lambda m: m(torch.randn(16, 64)))


def test_rejection_leaves_the_model_untouched():
    """Validation happens before any calibrator is swapped, so a rejected model is unchanged."""
    torch.manual_seed(0)
    model = mtq.quantize(
        _Net(),
        {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {"quantizer_name": "*fc1*input_quantizer", "cfg": NVFP4_CFG},
                {"quantizer_name": "*fc2*input_quantizer", "cfg": [NVFP4_CFG, NVFP4_CFG]},
            ],
            "algorithm": "max",
        },
        lambda m: m(torch.randn(16, 64)),
    )
    before = model.fc1.input_quantizer._calibrator
    with pytest.raises(NotImplementedError, match="does not support SequentialQuantizer"):
        _swap_in_nvfp4_act_headroom_calibrators(
            model, anchor_percentile=1.0, upper_percentile=99.99, rho=16384.0
        )
    assert model.fc1.input_quantizer._calibrator is before


def test_non_nvfp4_sequential_activation_is_ignored():
    """Only SequentialQuantizers wrapping NVFP4 leaves are rejected; others are irrelevant."""
    torch.manual_seed(0)
    fp8 = {"num_bits": (4, 3)}
    model = mtq.quantize(
        _Net(),
        {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {"quantizer_name": "*input_quantizer", "cfg": [fp8, fp8]},
            ],
            "algorithm": "max",
        },
        lambda m: m(torch.randn(16, 64)),
    )
    assert isinstance(model.fc1.input_quantizer, SequentialQuantizer)
    # No NVFP4 leaf anywhere, so nothing to reject and nothing to swap.
    assert (
        _swap_in_nvfp4_act_headroom_calibrators(
            model, anchor_percentile=1.0, upper_percentile=99.99, rho=16384.0
        )
        == []
    )


def test_swap_installs_calibrator_with_config_values():
    model = mtq.quantize(_Net(), ACT_ONLY_CFG, lambda m: m(torch.randn(8, 64)))
    swapped = _swap_in_nvfp4_act_headroom_calibrators(
        model, anchor_percentile=5.0, upper_percentile=99.99, rho=1024.0
    )
    assert len(swapped) == 2  # fc1 and fc2 input quantizers
    cal = model.fc1.input_quantizer._calibrator
    assert isinstance(cal, NVFP4ActHeadroomCalibrator)
    assert cal._anchor_percentile == 5.0
    assert cal._upper_percentile == 99.99
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
