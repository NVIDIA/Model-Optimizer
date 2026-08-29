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

"""Tests for weighted-MSE (``wmse``) calibration (CPU)."""

import pytest
import torch
import torch.nn as nn
from _test_utils.torch.quantization.models import SimpleConv, SimpleLinear

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.config import WmseCalibConfig
from modelopt.torch.quantization.model_calib import (
    _LocalHessianAccumulator,
    local_hessian_calibrate,
    mse_calibrate,
    wmse_calibrate,
)
from modelopt.torch.quantization.nn import SequentialQuantizer, TensorQuantizer

# Weight-only INT8 per-channel; calibration is re-run explicitly per test.
INT8_WEIGHT_CFG = {
    "quant_cfg": [
        {"quantizer_name": "*", "enable": False},
        {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 8, "axis": 0}},
    ],
    "algorithm": "max",
}

INT8_W8A8_CFG = {
    "quant_cfg": [
        {"quantizer_name": "*", "enable": False},
        {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 8, "axis": 0}},
        {"quantizer_name": "*input_quantizer", "cfg": {"num_bits": 8, "axis": None}},
    ],
    "algorithm": "max",
}


def _weight_amaxes(model):
    return {
        n: m.amax
        for n, m in model.named_modules()
        if isinstance(m, TensorQuantizer) and m.is_enabled and m.amax is not None
    }


def _make_forward_loop(seed=0):
    def forward_loop(model):
        torch.manual_seed(seed)
        for _ in range(3):
            x = torch.randn(8, 16)
            x[:, 0] *= 40.0  # skew so the importance is non-trivial vs plain weight MSE
            model(x)

    return forward_loop


class TestWmseAccumulator:
    def test_accumulate_shape_samples_fp32_buffer(self):
        torch.manual_seed(0)
        acc = _LocalHessianAccumulator(8, 32, 16, diagonal=True)
        assert acc.is_enabled
        acc.accumulate(torch.randn(10, 32, dtype=torch.bfloat16))
        assert acc.hessian_per_block.shape == (2, 16)  # [n_blocks, block_size], not [n, bs, bs]
        assert acc.hessian_per_block.dtype == torch.float32  # fp32 despite bf16 input
        acc.accumulate(torch.randn(5, 32))
        assert acc.num_samples == 15
        assert acc.build_error_func() is not None
        assert acc.hessian_per_block is None  # raw buffer freed

    def test_importance_is_sum_of_squared_activations(self):
        torch.manual_seed(1)
        cin, bs = 32, 16
        x = torch.randn(7, cin)
        acc = _LocalHessianAccumulator(4, cin, bs, diagonal=True)
        acc.accumulate(x)
        expected = x.square().sum(0).reshape(cin // bs, bs)  # Imp_i = ‖X[:, i]‖²
        assert torch.allclose(acc.hessian_per_block, expected, atol=1e-5)
        assert torch.allclose(acc.normalized_hessian(), expected / 7, atol=1e-6)

    def test_importance_is_the_local_hessian_diagonal(self):
        torch.manual_seed(2)
        cin, bs = 32, 16
        x = torch.randn(9, cin)
        full = _LocalHessianAccumulator(4, cin, bs)
        diag = _LocalHessianAccumulator(4, cin, bs, diagonal=True)
        full.accumulate(x)
        diag.accumulate(x)
        assert torch.allclose(
            diag.hessian_per_block, torch.diagonal(full.hessian_per_block, dim1=-2, dim2=-1)
        )

    def test_error_func_matches_explicit_weighted_squared_error(self):
        torch.manual_seed(3)
        cout, cin, bs = 4, 32, 16
        n_blocks = cin // bs
        acc = _LocalHessianAccumulator(cout, cin, bs, diagonal=True)
        x = torch.randn(7, cin)
        acc.accumulate(x)
        error_func = acc.build_error_func()

        importance = x.square().sum(0).reshape(n_blocks, bs) / acc.num_samples
        w = torch.randn(cout * n_blocks, bs)
        wq = w + 0.05 * torch.randn_like(w)
        err = error_func(w, wq).view(-1, bs)

        assert err.shape == (cout * n_blocks, bs)
        assert torch.allclose(err, err[:, :1].expand(-1, bs))  # per-block scalar broadcast
        dw = (w - wq).view(cout, n_blocks, bs)
        expected = torch.einsum("cnb,nb->cn", dw * dw, importance).reshape(-1)
        assert torch.allclose(err[:, 0], expected, atol=1e-5)

    def test_error_func_matches_local_hessian_with_diagonal_hessian(self):
        """Pins eq. 13: WMSE(Imp) == local_hessian(diag(Imp)) block-for-block."""
        torch.manual_seed(4)
        cout, cin, bs = 5, 32, 16
        n_blocks = cin // bs
        x = torch.randn(11, cin)

        diag_acc = _LocalHessianAccumulator(cout, cin, bs, diagonal=True)
        diag_acc.accumulate(x)
        importance = diag_acc.normalized_hessian()
        wmse_error = diag_acc.build_error_func()

        # A full-Hessian accumulator whose Hessian is exactly diag(Imp).
        full_acc = _LocalHessianAccumulator(cout, cin, bs)
        full_acc.hessian_per_block = torch.diag_embed(importance)
        full_acc.num_samples = 1
        lh_error = full_acc.build_error_func()

        w = torch.randn(cout * n_blocks, bs)
        wq = w + 0.05 * torch.randn_like(w)
        assert torch.allclose(wmse_error(w, wq), lh_error(w, wq), atol=1e-5)

    def test_returns_none_when_disabled_or_no_samples(self):
        not_divisible = _LocalHessianAccumulator(8, 30, 16, diagonal=True)
        assert not not_divisible.is_enabled
        not_divisible.accumulate(torch.randn(4, 30))  # no-op
        assert not_divisible.build_error_func() is None
        # no samples
        assert _LocalHessianAccumulator(8, 32, 16, diagonal=True).build_error_func() is None

    def test_wmse_never_allocates_coupling(self):
        """``activation_error_coupling`` is local-Hessian-only; wmse must not carry it."""
        acc = _LocalHessianAccumulator(2, 4, 2, diagonal=True)
        acc.accumulate(torch.randn(3, 4))
        assert acc.coupling_per_block is None
        assert acc.normalized_coupling() is None


def test_wmse_config_and_preset():
    config = WmseCalibConfig()
    dumped = config.model_dump()
    assert dumped["method"] == "wmse"
    assert dumped["fp8_scale_sweep"] is True
    assert dumped["block_size"] == 16
    assert "activation_error_coupling" not in dumped  # local-Hessian-only feature
    with pytest.raises(ValueError):
        WmseCalibConfig(activation_error_coupling=True)

    preset = mtq.NVFP4_W4A4_WEIGHT_WMSE_CFG
    assert preset["algorithm"]["method"] == "wmse"
    assert preset["algorithm"]["fp8_scale_sweep"] is True
    assert preset["algorithm"]["layerwise"]["enable"] is True
    # Same numerics as the local-Hessian preset; only the scale search differs.
    assert preset["quant_cfg"] == mtq.NVFP4_W4A4_WEIGHT_LOCAL_HESSIAN_CFG["quant_cfg"]


class TestWmseCalibrateDense:
    def test_refines_amax_beyond_max_and_plain_mse(self):
        forward_loop = _make_forward_loop()
        torch.manual_seed(0)
        model_wmse = SimpleLinear()
        mtq.quantize(model_wmse, INT8_WEIGHT_CFG, forward_loop=forward_loop)
        max_amax = {n: a.clone() for n, a in _weight_amaxes(model_wmse).items()}
        wmse_calibrate(model_wmse, forward_loop, fp8_scale_sweep=False, debug=True)

        torch.manual_seed(0)
        model_mse = SimpleLinear()
        mtq.quantize(model_mse, INT8_WEIGHT_CFG, forward_loop=forward_loop)
        mse_calibrate(model_mse, forward_loop, fp8_scale_sweep=False)

        accs = model_wmse._local_hessian_accumulators
        assert accs and all(a.num_samples > 0 for a in accs.values())
        assert all(a.diagonal and a.normalized_hessian().dim() == 2 for a in accs.values())
        wmse, mse = _weight_amaxes(model_wmse), _weight_amaxes(model_mse)
        assert all(torch.isfinite(a).all() and (a > 0).all() for a in wmse.values())
        assert any(not torch.allclose(wmse[n], max_amax[n]) for n in wmse)  # refined past max-cal
        assert any(not torch.allclose(wmse[n], mse[n]) for n in wmse)  # weighting changed choice

    def test_matches_local_hessian_when_hessian_is_diagonal(self):
        """End-to-end equivalence: a diagonal input covariance makes the two agree exactly.

        Single linear so the calibrated layer's own input is the crafted one; one-hot rows
        scaled per channel give ``XᵀX = diag(scale²)`` exactly, so the local Hessian carries
        no information the diagonal importance does not and both must pick the same amax.
        """

        class _OneLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(16, 8)

            def forward(self, x):
                return self.fc(x)

        scale = torch.logspace(-1, 1, 16)
        x = torch.eye(16) * scale
        forward_loop = lambda m: m(x)  # noqa: E731

        amaxes = {}
        for name, calibrate in (("wmse", wmse_calibrate), ("lh", local_hessian_calibrate)):
            torch.manual_seed(0)
            model = _OneLinear()
            mtq.quantize(model, INT8_WEIGHT_CFG, forward_loop=forward_loop)
            calibrate(model, forward_loop, fp8_scale_sweep=False)
            amaxes[name] = _weight_amaxes(model)

        assert amaxes["wmse"] and amaxes["wmse"].keys() == amaxes["lh"].keys()
        assert all(torch.allclose(amaxes["wmse"][n], amaxes["lh"][n]) for n in amaxes["wmse"])

    def test_warns_with_module_name_when_cin_not_divisible(self):
        class _OddModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.odd = nn.Linear(24, 32)  # 24 not divisible by block_size 16

            def forward(self, x):
                return self.odd(x)

        torch.manual_seed(0)
        model = _OddModel()
        forward_loop = lambda m: m(torch.randn(4, 24))  # noqa: E731
        mtq.quantize(model, INT8_WEIGHT_CFG, forward_loop=forward_loop)
        with pytest.warns(UserWarning, match=r"wmse: odd input features \(24\) not divisible"):
            wmse_calibrate(model, forward_loop, fp8_scale_sweep=False)

    def test_no_forward_loop_is_skipped(self):
        torch.manual_seed(0)
        model = SimpleLinear()
        mtq.quantize(model, INT8_WEIGHT_CFG, forward_loop=_make_forward_loop())
        before = {n: a.clone() for n, a in _weight_amaxes(model).items()}
        with pytest.warns(UserWarning, match="forward_loop must be provided for wmse"):
            wmse_calibrate(model, forward_loop=None)
        assert all(torch.equal(before[n], a) for n, a in _weight_amaxes(model).items())

    @pytest.mark.parametrize("quant_cfg", [INT8_WEIGHT_CFG, INT8_W8A8_CFG])
    def test_importance_uses_input_quantizer_output(self, quant_cfg):
        torch.manual_seed(0)
        model = SimpleLinear()
        x = torch.randn(8, 16)
        x[:, 0] *= 40.0
        forward_loop = lambda m: m(x)  # noqa: E731
        mtq.quantize(model, quant_cfg, forward_loop=forward_loop)
        wmse_calibrate(model, forward_loop, fp8_scale_sweep=False, debug=True)

        linear = model.net[0]
        acc = model._local_hessian_accumulators[id(linear.weight_quantizer)]
        quantizer_output = linear.input_quantizer(x).float()
        expected = quantizer_output.square().sum(0).reshape(1, 16)
        assert torch.equal(acc.hessian_per_block, expected)
        assert acc.coupling_per_block is None


class TestWmseFallbacks:
    """Weights wmse can't pair with an input fall back to plain MSE (no importance)."""

    def test_conv_weight_falls_back_without_crash(self):
        torch.manual_seed(0)
        model = SimpleConv()  # 4-D conv weights — no single 2-D weight to pair
        forward_loop = lambda m: m(SimpleConv.get_input())  # noqa: E731
        mtq.quantize(model, INT8_WEIGHT_CFG, forward_loop=forward_loop)
        wmse_calibrate(model, forward_loop, fp8_scale_sweep=False, debug=True)
        conv = model.net[0]
        assert id(conv.weight_quantizer) not in model._local_hessian_accumulators
        assert conv.weight_quantizer.amax is not None  # still calibrated via plain MSE

    def test_sequential_quantizer_weight_falls_back_without_crash(self):
        torch.manual_seed(0)
        model = SimpleLinear()
        mtq.quantize(model, INT8_WEIGHT_CFG, forward_loop=_make_forward_loop())
        linear = model.net[0]
        linear.weight_quantizer = SequentialQuantizer(TensorQuantizer(), TensorQuantizer())
        wmse_calibrate(model, _make_forward_loop(), fp8_scale_sweep=False, debug=True)
        assert id(linear.weight_quantizer) not in model._local_hessian_accumulators
