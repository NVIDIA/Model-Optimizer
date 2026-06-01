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

import pytest
import torch
from _test_utils.torch.quantization.models import SimpleLinear

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization import calib
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.model_calib import (
    _FP8_SWEEP_CALIBRATOR_REGISTRY,
    _LocalHessianAccumulator,
    _make_weight_mse_calibrator,
    _register_fp8_sweep_calibrator,
    local_hessian_calibrate,
    mse_calibrate,
)
from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.quantization.nn.modules.tensor_quantizer import (
    _QUANT_FUNCTIONAL_BACKENDS,
    register_quant_backend,
)

# Weight-only INT8 per-channel config; calibration is re-run explicitly per test.
INT8_WEIGHT_CFG = {
    "quant_cfg": [
        {"quantizer_name": "*", "enable": False},
        {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 8, "axis": 0}},
    ],
    "algorithm": "max",
}


class TestLocalHessianAccumulator:
    def test_shape_samples_and_buffer_release(self):
        torch.manual_seed(0)
        cout, cin, bs = 8, 32, 16
        acc = _LocalHessianAccumulator(cout, cin, bs)
        assert acc.is_enabled

        x = torch.randn(10, cin)
        acc.accumulate(x)
        assert acc.hessian_per_block.shape == (cin // bs, bs, bs)
        assert acc.num_samples == 10

        # A second batch accumulates (sum over samples).
        acc.accumulate(torch.randn(5, cin))
        assert acc.num_samples == 15

        error_func = acc.build_error_func()
        assert error_func is not None
        # build_error_func releases the raw accumulator (keeps only the normalized copy).
        assert acc.hessian_per_block is None

    def test_error_func_matches_explicit_hessian_weighted_loss(self):
        torch.manual_seed(1)
        cout, cin, bs = 4, 32, 16
        n_blocks = cin // bs
        acc = _LocalHessianAccumulator(cout, cin, bs)

        x = torch.randn(7, cin)
        acc.accumulate(x)
        num_samples = acc.num_samples
        error_func = acc.build_error_func()

        # Reference normalized per-block Hessian.
        xb = x.reshape(-1, cin).T.reshape(n_blocks, bs, -1)
        hessian = (xb @ xb.transpose(-1, -2)) / num_samples

        total_blocks = cout * n_blocks
        w = torch.randn(total_blocks, bs)
        wq = w + 0.05 * torch.randn_like(w)
        err = error_func(w, wq)

        assert err.shape == w.shape
        # The per-block scalar loss is broadcast across the block's bs entries.
        err_blocks = err.view(-1, bs)
        assert torch.allclose(err_blocks, err_blocks[:, :1].expand(-1, bs))

        dw = (w - wq).view(cout, n_blocks, bs)
        expected = torch.einsum("cnb,nbd,cnd->cn", dw, hessian, dw).reshape(-1)
        assert torch.allclose(err_blocks[:, 0], expected, atol=1e-5)

    def test_disabled_when_cin_not_divisible(self):
        acc = _LocalHessianAccumulator(8, 30, 16)
        assert not acc.is_enabled
        acc.accumulate(torch.randn(4, 30))  # no-op
        assert acc.hessian_per_block is None
        assert acc.build_error_func() is None

    def test_no_samples_returns_none(self):
        acc = _LocalHessianAccumulator(8, 32, 16)
        assert acc.build_error_func() is None

    def test_accumulates_in_fp32_for_low_precision_input(self):
        acc = _LocalHessianAccumulator(4, 16, 16)
        acc.accumulate(torch.randn(8, 16, dtype=torch.bfloat16))
        assert acc.hessian_per_block.dtype == torch.float32


class TestBlockSizeMismatchWarning:
    def _block_quantizer(self, block):
        cfg = QuantizerAttributeConfig(
            num_bits=(2, 1), block_sizes={-1: block, "type": "static", "scale_bits": (4, 3)}
        )
        return TensorQuantizer(quant_attribute_cfg=cfg)

    def test_warns_on_mismatch(self):
        from modelopt.torch.quantization.model_calib import _warn_if_block_size_mismatch

        with pytest.warns(UserWarning, match="will not align"):
            _warn_if_block_size_mismatch(self._block_quantizer(32), 16, "layer")

    def test_silent_when_matching_or_no_block_sizes(self):
        import warnings

        from modelopt.torch.quantization.model_calib import _warn_if_block_size_mismatch

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _warn_if_block_size_mismatch(self._block_quantizer(16), 16, "layer")
            per_channel = TensorQuantizer(QuantizerAttributeConfig(num_bits=8, axis=0))
            _warn_if_block_size_mismatch(per_channel, 16, "layer")


def _make_forward_loop(seed=0, skew=True):
    def forward_loop(model):
        torch.manual_seed(seed)
        for _ in range(3):
            x = torch.randn(8, 16)
            if skew:
                # Skew one input feature so the activation Hessian is non-trivial and
                # the Hessian-weighted optimum diverges from the plain weight MSE.
                x[:, 0] *= 40.0
            model(x)

    return forward_loop


class TestLocalHessianCalibrateDense:
    def test_runs_and_refines_amax(self):
        torch.manual_seed(0)
        model = SimpleLinear()
        forward_loop = _make_forward_loop()
        mtq.quantize(model, INT8_WEIGHT_CFG, forward_loop=forward_loop)

        # Snapshot the post-max-calibration amax for each weight quantizer.
        max_amax = {
            name: module.amax.clone()
            for name, module in model.named_modules()
            if isinstance(module, TensorQuantizer) and module.is_enabled and module.amax is not None
        }
        assert max_amax, "expected enabled weight quantizers after quantize"

        local_hessian_calibrate(model, forward_loop, fp8_scale_sweep=False, debug=True)

        # Every enabled weight quantizer got a Hessian accumulator with collected samples.
        accumulators = model._local_hessian_accumulators
        assert accumulators
        assert all(acc.num_samples > 0 for acc in accumulators.values())

        # The Hessian-weighted search moved at least one amax away from the max value.
        changed = False
        for name, module in model.named_modules():
            if name in max_amax:
                assert torch.isfinite(module.amax).all() and (module.amax > 0).all()
                if not torch.allclose(module.amax, max_amax[name]):
                    changed = True
        assert changed, "local_hessian did not refine any amax away from max-calibration"

    def test_differs_from_plain_mse(self):
        forward_loop = _make_forward_loop(seed=3)

        torch.manual_seed(0)
        model_lh = SimpleLinear()
        mtq.quantize(model_lh, INT8_WEIGHT_CFG, forward_loop=forward_loop)
        local_hessian_calibrate(model_lh, forward_loop, fp8_scale_sweep=False)

        torch.manual_seed(0)
        model_mse = SimpleLinear()
        mtq.quantize(model_mse, INT8_WEIGHT_CFG, forward_loop=forward_loop)
        mse_calibrate(model_mse, forward_loop, fp8_scale_sweep=False)

        lh = {
            n: m.amax
            for n, m in model_lh.named_modules()
            if isinstance(m, TensorQuantizer) and m.is_enabled and m.amax is not None
        }
        mse = {
            n: m.amax
            for n, m in model_mse.named_modules()
            if isinstance(m, TensorQuantizer) and m.is_enabled and m.amax is not None
        }
        assert lh.keys() == mse.keys()
        # Hessian weighting should change the chosen scale for at least one quantizer.
        assert any(not torch.allclose(lh[n], mse[n]) for n in lh)

    def test_no_forward_loop_is_skipped(self):
        torch.manual_seed(0)
        model = SimpleLinear()
        mtq.quantize(model, INT8_WEIGHT_CFG, forward_loop=_make_forward_loop())
        before = {
            n: m.amax.clone()
            for n, m in model.named_modules()
            if isinstance(m, TensorQuantizer) and m.is_enabled and m.amax is not None
        }
        with pytest.warns(UserWarning, match="forward_loop must be provided"):
            local_hessian_calibrate(model, forward_loop=None)
        after = {
            n: m.amax
            for n, m in model.named_modules()
            if isinstance(m, TensorQuantizer) and m.is_enabled and m.amax is not None
        }
        for n in before:
            assert torch.equal(before[n], after[n])


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
        cfg = QuantizerAttributeConfig(num_bits=8, axis=None, backend=backend)
        q = TensorQuantizer(quant_attribute_cfg=cfg)
        q.amax = torch.tensor(1.0)
        return q

    def test_error_func_threaded_to_mse_calibrator(self):
        q = self._make_quantizer()
        marker = lambda x, xq: (x - xq) ** 2  # noqa: E731
        cal = _make_weight_mse_calibrator(
            q, 0.1, 0.25, 4.0, fp8_scale_sweep=False, error_func=marker
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
