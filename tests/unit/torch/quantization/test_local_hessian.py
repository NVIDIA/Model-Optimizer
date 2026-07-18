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

"""Tests for local Hessian-weighted MSE calibration (CPU)."""

import warnings

import pytest
import torch
import torch.nn as nn
from _test_utils.torch.quantization.models import SimpleConv, SimpleLinear

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization import calib
from modelopt.torch.quantization.config import LocalHessianCalibConfig, QuantizerAttributeConfig
from modelopt.torch.quantization.model_calib import (
    _FP8_SWEEP_CALIBRATOR_REGISTRY,
    _activation_error_coupling_bias,
    _activation_error_coupling_fallback_reason,
    _LocalHessianAccumulator,
    _make_weight_mse_calibrator,
    _register_fp8_sweep_calibrator,
    _warn_if_block_size_mismatch,
    local_hessian_calibrate,
    mse_calibrate,
)
from modelopt.torch.quantization.nn import SequentialQuantizer, TensorQuantizer
from modelopt.torch.quantization.nn.modules.tensor_quantizer import (
    _QUANT_FUNCTIONAL_BACKENDS,
    register_quant_backend,
)

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
            x[:, 0] *= 40.0  # skew so the Hessian is non-trivial vs plain weight MSE
            model(x)

    return forward_loop


class TestLocalHessianAccumulator:
    def test_accumulate_shape_samples_fp32_buffer(self):
        torch.manual_seed(0)
        acc = _LocalHessianAccumulator(8, 32, 16)
        assert acc.is_enabled
        acc.accumulate(torch.randn(10, 32, dtype=torch.bfloat16))
        assert acc.hessian_per_block.shape == (2, 16, 16)
        assert acc.hessian_per_block.dtype == torch.float32  # fp32 despite bf16 input
        acc.accumulate(torch.randn(5, 32))
        assert acc.num_samples == 15
        assert acc.build_error_func() is not None
        assert acc.hessian_per_block is None  # raw buffer freed

    def test_error_func_matches_explicit_hessian_weighted_loss(self):
        torch.manual_seed(1)
        cout, cin, bs = 4, 32, 16
        n_blocks = cin // bs
        acc = _LocalHessianAccumulator(cout, cin, bs)
        x = torch.randn(7, cin)
        acc.accumulate(x)
        error_func = acc.build_error_func()

        xb = x.reshape(-1, cin).T.reshape(n_blocks, bs, -1)
        hessian = (xb @ xb.transpose(-1, -2)) / acc.num_samples
        w = torch.randn(cout * n_blocks, bs)
        wq = w + 0.05 * torch.randn_like(w)
        err = error_func(w, wq).view(-1, bs)

        assert err.shape == (cout * n_blocks, bs)
        assert torch.allclose(err, err[:, :1].expand(-1, bs))  # per-block scalar broadcast
        dw = (w - wq).view(cout, n_blocks, bs)
        expected = torch.einsum("cnb,nbd,cnd->cn", dw, hessian, dw).reshape(-1)
        assert torch.allclose(err[:, 0], expected, atol=1e-5)

    def test_returns_none_when_disabled_or_no_samples(self):
        not_divisible = _LocalHessianAccumulator(8, 30, 16)
        assert not not_divisible.is_enabled
        not_divisible.accumulate(torch.randn(4, 30))  # no-op
        assert not_divisible.build_error_func() is None
        assert _LocalHessianAccumulator(8, 32, 16).build_error_func() is None  # no samples

    def test_activation_error_coupling_accumulation(self):
        torch.manual_seed(2)
        cout, cin, bs = 3, 8, 4
        x = torch.randn(6, cin, dtype=torch.bfloat16)
        xq = (x.float() + 0.1 * torch.randn_like(x.float())).to(torch.bfloat16)
        acc = _LocalHessianAccumulator(cout, cin, bs)
        acc.accumulate(xq, x)

        xb = x.float().T.reshape(cin // bs, bs, -1)
        xqb = xq.float().T.reshape(cin // bs, bs, -1)
        assert acc.coupling_per_block.dtype == torch.float32
        assert torch.equal(acc.hessian_per_block, xqb @ xqb.transpose(-1, -2))
        assert torch.equal(acc.coupling_per_block, xqb @ (xqb - xb).transpose(-1, -2))

    def test_activation_error_coupling_objective_matches_output_mse(self):
        torch.manual_seed(3)
        samples, cout, cin, bs = 7, 2, 2, 2
        xq = torch.randn(samples, cin)
        x = xq - 0.2 * torch.randn_like(xq)
        w0 = torch.randn(cout, cin)
        acc = _LocalHessianAccumulator(cout, cin, bs)
        acc.accumulate(xq, x)
        hessian = acc.normalized_hessian()[0]
        coupling = acc.normalized_coupling()
        bias = _activation_error_coupling_bias(w0, coupling, cout, bs)[:, 0]
        # The minimizer of the constant-dropped quadratic is deliberately negative.
        dw = -torch.linalg.solve(hessian, bias.T).T
        wq = w0 + dw

        objective = acc.build_error_func(keep_buffer=True)(w0, wq)[:, 0]
        deployed = xq @ wq.T
        reference = x @ w0.T
        constant = xq @ w0.T - reference
        expected = ((deployed - reference).square() - constant.square()).sum(0) / samples
        assert torch.allclose(objective, expected, atol=2e-5, rtol=2e-5)
        assert (objective < 0).all()

    def test_default_accumulation_does_not_allocate_coupling(self):
        acc = _LocalHessianAccumulator(2, 4, 2)
        x = torch.randn(3, 4)
        acc.accumulate(x)
        assert acc.coupling_per_block is None
        assert acc.normalized_coupling() is None


def test_activation_error_coupling_config_and_preset():
    config = LocalHessianCalibConfig(activation_error_coupling=True)
    assert config.model_dump()["activation_error_coupling"] is True
    assert (
        mtq.NVFP4_W4A4_WEIGHT_LOCAL_HESSIAN_ACT_ERROR_COUPLING_CFG["algorithm"][
            "activation_error_coupling"
        ]
        is True
    )


class TestLocalHessianCalibrateDense:
    def test_refines_amax_beyond_max_and_plain_mse(self):
        forward_loop = _make_forward_loop()
        torch.manual_seed(0)
        model_lh = SimpleLinear()
        mtq.quantize(model_lh, INT8_WEIGHT_CFG, forward_loop=forward_loop)
        max_amax = {n: a.clone() for n, a in _weight_amaxes(model_lh).items()}
        local_hessian_calibrate(model_lh, forward_loop, fp8_scale_sweep=False, debug=True)

        torch.manual_seed(0)
        model_mse = SimpleLinear()
        mtq.quantize(model_mse, INT8_WEIGHT_CFG, forward_loop=forward_loop)
        mse_calibrate(model_mse, forward_loop, fp8_scale_sweep=False)

        accs = model_lh._local_hessian_accumulators
        assert accs and all(a.num_samples > 0 for a in accs.values())
        lh, mse = _weight_amaxes(model_lh), _weight_amaxes(model_mse)
        assert all(torch.isfinite(a).all() and (a > 0).all() for a in lh.values())
        assert any(not torch.allclose(lh[n], max_amax[n]) for n in lh)  # refined past max-cal
        assert any(not torch.allclose(lh[n], mse[n]) for n in lh)  # Hessian changed the choice

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
        with pytest.warns(UserWarning, match=r"odd input features \(24\) not divisible"):
            local_hessian_calibrate(model, forward_loop, fp8_scale_sweep=False)

    def test_no_forward_loop_is_skipped(self):
        torch.manual_seed(0)
        model = SimpleLinear()
        mtq.quantize(model, INT8_WEIGHT_CFG, forward_loop=_make_forward_loop())
        before = {n: a.clone() for n, a in _weight_amaxes(model).items()}
        with pytest.warns(UserWarning, match="forward_loop must be provided"):
            local_hessian_calibrate(model, forward_loop=None)
        assert all(torch.equal(before[n], a) for n, a in _weight_amaxes(model).items())

    def test_activation_error_coupling_captures_quantized_inputs(self):
        torch.manual_seed(0)
        model = SimpleLinear()
        forward_loop = _make_forward_loop()
        mtq.quantize(model, INT8_W8A8_CFG, forward_loop=forward_loop)
        local_hessian_calibrate(
            model,
            forward_loop,
            fp8_scale_sweep=False,
            activation_error_coupling=True,
            debug=True,
        )
        routed = [a for a in model._local_hessian_accumulators.values() if a.num_samples]
        assert routed
        assert all(a.coupling_per_block is not None for a in routed)
        assert any(torch.count_nonzero(a.coupling_per_block) for a in routed)

    @pytest.mark.parametrize("quant_cfg", [INT8_WEIGHT_CFG, INT8_W8A8_CFG])
    @pytest.mark.parametrize("activation_error_coupling", [False, True])
    def test_hessian_uses_input_quantizer_output(self, quant_cfg, activation_error_coupling):
        torch.manual_seed(0)
        model = SimpleLinear()
        x = torch.randn(8, 16)
        x[:, 0] *= 40.0
        forward_loop = lambda m: m(x)  # noqa: E731
        mtq.quantize(model, quant_cfg, forward_loop=forward_loop)
        local_hessian_calibrate(
            model,
            forward_loop,
            fp8_scale_sweep=False,
            activation_error_coupling=activation_error_coupling,
            debug=True,
        )

        linear = model.net[0]
        acc = model._local_hessian_accumulators[id(linear.weight_quantizer)]
        quantizer_output = linear.input_quantizer(x).float().T.reshape(1, 16, -1)
        expected_hessian = quantizer_output @ quantizer_output.transpose(-1, -2)
        assert torch.equal(acc.hessian_per_block, expected_hessian)
        expect_coupling = activation_error_coupling and linear.input_quantizer.is_enabled
        assert (acc.coupling_per_block is not None) is expect_coupling


class TestLocalHessianFallbacks:
    """Weights local-Hessian can't pair with an input fall back to plain MSE (no Hessian)."""

    def test_conv_weight_falls_back_without_crash(self):
        torch.manual_seed(0)
        model = SimpleConv()  # 4-D conv weights — no single 2-D weight to pair
        forward_loop = lambda m: m(SimpleConv.get_input())  # noqa: E731
        mtq.quantize(model, INT8_WEIGHT_CFG, forward_loop=forward_loop)
        local_hessian_calibrate(model, forward_loop, fp8_scale_sweep=False, debug=True)
        conv = model.net[0]
        assert id(conv.weight_quantizer) not in model._local_hessian_accumulators
        assert conv.weight_quantizer.amax is not None  # still calibrated via plain MSE

    def test_sequential_quantizer_weight_falls_back_without_crash(self):
        torch.manual_seed(0)
        model = SimpleLinear()
        mtq.quantize(model, INT8_WEIGHT_CFG, forward_loop=_make_forward_loop())
        linear = model.net[0]
        linear.weight_quantizer = SequentialQuantizer(TensorQuantizer(), TensorQuantizer())
        local_hessian_calibrate(model, _make_forward_loop(), fp8_scale_sweep=False, debug=True)
        assert id(linear.weight_quantizer) not in model._local_hessian_accumulators

    @pytest.mark.parametrize(
        ("cfg", "reason"),
        [
            ({"num_bits": 8}, "pre_quant_scale"),
            (
                {
                    "num_bits": 8,
                    "rotate": {"enable": True, "mode": "rotate", "block_size": 8},
                },
                "rotation",
            ),
        ],
    )
    def test_activation_error_coupling_fallback_reasons(self, cfg, reason):
        q = TensorQuantizer(QuantizerAttributeConfig(**cfg))
        if reason == "pre_quant_scale":
            q.pre_quant_scale = torch.ones(16)
        assert reason in _activation_error_coupling_fallback_reason(q)


class TestBlockSizeMismatchWarning:
    def test_block_size_mismatch_warns_only_on_mismatch(self):
        def q(block):
            return TensorQuantizer(
                QuantizerAttributeConfig(
                    num_bits=(2, 1), block_sizes={-1: block, "type": "static", "scale_bits": (4, 3)}
                )
            )

        with pytest.warns(UserWarning, match="will not align"):
            _warn_if_block_size_mismatch(q(32), 16, "layer")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _warn_if_block_size_mismatch(q(16), 16, "layer")  # matching block
            per_channel = TensorQuantizer(QuantizerAttributeConfig(num_bits=8, axis=0))
            _warn_if_block_size_mismatch(per_channel, 16, "layer")  # no block_sizes


class TestMakeWeightMseCalibratorErrorFunc:
    def setup_method(self):
        self._orig_fp8_registry = dict(_FP8_SWEEP_CALIBRATOR_REGISTRY)
        self._orig_quant_backends = dict(_QUANT_FUNCTIONAL_BACKENDS)

    def teardown_method(self):
        _FP8_SWEEP_CALIBRATOR_REGISTRY.clear()
        _FP8_SWEEP_CALIBRATOR_REGISTRY.update(self._orig_fp8_registry)
        _QUANT_FUNCTIONAL_BACKENDS.clear()
        _QUANT_FUNCTIONAL_BACKENDS.update(self._orig_quant_backends)

    def _make_quantizer(self, backend=None):
        q = TensorQuantizer(QuantizerAttributeConfig(num_bits=8, axis=None, backend=backend))
        q.amax = torch.tensor(1.0)
        return q

    def test_error_func_threaded_to_mse_calibrator(self):
        marker = lambda x, xq: (x - xq) ** 2  # noqa: E731
        cal = _make_weight_mse_calibrator(
            self._make_quantizer(), 0.1, 0.25, 4.0, fp8_scale_sweep=False, error_func=marker
        )
        assert isinstance(cal, calib.MseCalibrator)
        assert cal._error_func is marker

    def test_registered_backend_with_error_func_is_skipped(self):
        register_quant_backend("_lh_test_backend", lambda x, tq: x)
        _register_fp8_sweep_calibrator(
            "_lh_test_backend",
            lambda amax, axis, qf: calib.MseCalibrator(amax=amax, axis=axis, quant_func=qf),
        )
        q = self._make_quantizer(backend="_lh_test_backend")
        with pytest.warns(UserWarning, match="does not support a custom error"):
            cal = _make_weight_mse_calibrator(
                q, 0.1, 0.25, 4.0, fp8_scale_sweep=True, error_func=lambda x, xq: x
            )
        assert cal is None
