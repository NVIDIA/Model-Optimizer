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

"""CPU unit tests for the LSQ algorithm using INT4 quantization."""

import types
from unittest.mock import Mock, create_autospec

import pytest
import torch
from torch import nn

import modelopt.torch.quantization.model_calib as model_calib_module
import modelopt.torch.quantization.nn.modules.tensor_quantizer as tensor_quantizer_module
from modelopt.torch.quantization.config import (
    LocalHessianCalibConfig,
    LSQConfig,
    MaxCalibConfig,
    MseCalibConfig,
)
from modelopt.torch.quantization.model_calib import lsq
from modelopt.torch.quantization.nn import QuantLinear
from modelopt.torch.quantization.nn.modules.tensor_quantizer import (
    _FP8_E4M3_MIN_POSITIVE,
    StaticBlockScaleQuantizer,
    TensorQuantizer,
)
from modelopt.torch.quantization.tensor_quant import int_cast_ste


def _make_int4_static_quantizer():
    tq = TensorQuantizer()
    tq._num_bits = 4
    tq._unsigned = False
    tq._narrow_range = True
    tq._disabled = False
    tq._block_sizes = {-1: 16}
    tq._pass_through_bwd = True
    tq.register_buffer("_amax", torch.ones(8))
    return StaticBlockScaleQuantizer.from_tensor_quantizer(tq)


def _skip_scale_calibration(monkeypatch):
    monkeypatch.setattr(
        "modelopt.torch.quantization.model_calib._run_scale_calibration",
        lambda *args, **kwargs: None,
    )


@pytest.mark.parametrize(
    ("num_bits", "expected_dispatch"),
    [pytest.param((2, 1), "nvfp4", id="nvfp4"), pytest.param((4, 3), "generic", id="fp8")],
)
def test_non_lsq_static_float_dispatches_only_nvfp4_to_fp4_kernel(
    monkeypatch, num_bits, expected_dispatch
):
    tq = TensorQuantizer()
    tq._num_bits = num_bits
    tq._block_sizes = {-1: 16, "type": "static", "scale_bits": (4, 3)}
    tq.register_buffer("_amax", torch.ones(4))
    quantizer = StaticBlockScaleQuantizer.from_tensor_quantizer(tq, global_amax=torch.tensor(1.0))
    dispatches = []

    def fake_nvfp4(inputs, *_args):
        dispatches.append("nvfp4")
        return inputs

    def fake_generic(_self, inputs):
        dispatches.append("generic")
        return inputs

    monkeypatch.setattr(tensor_quantizer_module, "static_blockwise_fp4_fake_quant", fake_nvfp4)
    monkeypatch.setattr(TensorQuantizer, "_fake_quantize", fake_generic)

    quantizer._fake_quantize(torch.ones(4, 16))

    assert dispatches == [expected_dispatch]


class TestLSQConfig:
    """Tests for LSQConfig validation."""

    def test_default_config(self):
        cfg = LSQConfig()
        assert cfg.method == "lsq"
        assert cfg.learnable_amax == ["post"]
        assert cfg.tied_amax is False
        assert cfg.quantize_pre_scale is True
        assert cfg.scale_algorithm is None

    @pytest.mark.parametrize(
        ("method", "config_type"),
        [
            ("max", MaxCalibConfig),
            ("mse", MseCalibConfig),
            ("local_hessian", LocalHessianCalibConfig),
        ],
    )
    def test_scale_algorithm(self, method, config_type):
        cfg = LSQConfig(scale_algorithm={"method": method})
        assert isinstance(cfg.scale_algorithm, config_type)

    def test_unsupported_scale_algorithm(self):
        with pytest.raises(ValueError):
            LSQConfig(scale_algorithm={"method": "smoothquant"})

    def test_scale_algorithm_preserves_sparse_dict(self, monkeypatch):
        cfg = LSQConfig(scale_algorithm={"method": "mse", "fp8_scale_sweep": True})
        assert cfg.model_dump()["scale_algorithm"] == {
            "method": "mse",
            "fp8_scale_sweep": True,
        }

        calibrate = create_autospec(model_calib_module.mse_calibrate)
        monkeypatch.setattr(model_calib_module, "mse_calibrate", calibrate)
        model = Mock()
        model_calib_module._run_scale_calibration(
            model, None, cfg.scale_algorithm, caller_name="lsq"
        )
        calibrate.assert_called_once_with(model, forward_loop=None, fp8_scale_sweep=True)

    @pytest.mark.parametrize(
        ("learnable_amax", "tied_amax"),
        [
            (["post"], False),
            (["pre"], False),
            (["pre", "post"], False),
            (["pre", "post"], True),
            ([], False),
            ([], True),
            ("post", False),
            ("pre", False),
        ],
    )
    def test_valid_combinations(self, learnable_amax, tied_amax):
        cfg = LSQConfig(learnable_amax=learnable_amax, tied_amax=tied_amax)
        assert cfg.tied_amax is tied_amax

    @pytest.mark.parametrize(
        "learnable_amax",
        [["post"], ["pre"], "post", "pre"],
    )
    def test_invalid_tied_with_single_learnable(self, learnable_amax):
        with pytest.raises(ValueError, match="tied_amax=True requires"):
            LSQConfig(learnable_amax=learnable_amax, tied_amax=True)


class TestEnableLSQ:
    """Tests for StaticBlockScaleQuantizer.enable_lsq() with INT4 format."""

    def _make_quantizer(self):
        """Create a StaticBlockScaleQuantizer configured for INT4."""
        sbsq = _make_int4_static_quantizer()
        assert sbsq._quant_max_bound == 7.0
        return sbsq

    def test_post_only_learnable(self):
        q = self._make_quantizer()
        amax = torch.ones(8) * 3.0
        q.enable_lsq(amax, quantize_scales=False, learnable_amax=["post"], tied_amax=False)
        assert q._lsq is True
        assert isinstance(q._amax_post, nn.Parameter)
        assert q._amax_post.requires_grad is True
        assert not isinstance(q._amax_pre, nn.Parameter)
        assert not q._amax_pre.requires_grad

    def test_pre_only_learnable(self):
        q = self._make_quantizer()
        amax = torch.ones(8) * 3.0
        q.enable_lsq(amax, quantize_scales=False, learnable_amax=["pre"], tied_amax=False)
        assert isinstance(q._amax_pre, nn.Parameter)
        assert q._amax_pre.requires_grad is True
        assert not isinstance(q._amax_post, nn.Parameter)

    def test_both_learnable(self):
        q = self._make_quantizer()
        amax = torch.ones(8) * 3.0
        q.enable_lsq(amax, quantize_scales=False, learnable_amax=["pre", "post"], tied_amax=False)
        assert isinstance(q._amax_pre, nn.Parameter)
        assert isinstance(q._amax_post, nn.Parameter)

    def test_tied_both_learnable(self):
        q = self._make_quantizer()
        amax = torch.ones(8) * 3.0
        q.enable_lsq(amax, quantize_scales=False, learnable_amax=["pre", "post"], tied_amax=True)
        assert q._tied_amax is True
        assert isinstance(q._amax_post, nn.Parameter)
        assert not hasattr(q, "_amax_pre")
        assert q.amax_pre is q._amax_post

    def test_frozen(self):
        q = self._make_quantizer()
        amax = torch.ones(8) * 3.0
        q.enable_lsq(amax, quantize_scales=False, learnable_amax=[], tied_amax=False)
        assert not isinstance(q._amax_post, nn.Parameter)
        assert not isinstance(q._amax_pre, nn.Parameter)

    def test_old_amax_deleted(self):
        q = self._make_quantizer()
        assert hasattr(q, "_amax")
        q.enable_lsq(torch.ones(8), quantize_scales=False)
        assert not hasattr(q, "_amax")

    def test_can_skip_pre_scale_quantization(self):
        q = self._make_quantizer()
        q.enable_lsq(
            torch.ones(8),
            quantize_scales=False,
            quantize_pre_scale=False,
        )
        assert q._quantize_pre_scale is False

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_learnable_amax_uses_input_dtype(self, dtype):
        q = self._make_quantizer()
        q.enable_lsq(
            torch.ones(8, dtype=dtype),
            quantize_scales=False,
            learnable_amax=["pre", "post"],
        )

        assert q._amax_pre.dtype == dtype
        assert q._amax_post.dtype == dtype

    def test_dtype_cast_updates_learnable_amax_dtype(self):
        q = self._make_quantizer()
        q.enable_lsq(
            torch.ones(8),
            quantize_scales=False,
            learnable_amax=["pre", "post"],
        )

        q.to(dtype=torch.bfloat16)

        assert q._amax_pre.dtype == torch.bfloat16
        assert q._amax_post.dtype == torch.bfloat16

    def test_align_lsq_amax_param_dtypes_uses_weight_dtype(self):
        pytest.importorskip("transformers")
        from modelopt.torch.quantization.plugins.transformers_trainer import (
            _align_lsq_amax_param_dtypes,
        )

        module = nn.Module()
        module.weight = nn.Parameter(torch.ones(8, 16, dtype=torch.bfloat16))
        module.weight_quantizer = self._make_quantizer()
        module.weight_quantizer.enable_lsq(
            torch.ones(8),
            quantize_scales=False,
            learnable_amax=["pre", "post"],
        )

        assert module.weight_quantizer._amax_pre.dtype == torch.float32
        assert module.weight_quantizer._amax_post.dtype == torch.float32

        _align_lsq_amax_param_dtypes(module)

        assert module.weight_quantizer._amax_pre.dtype == torch.bfloat16
        assert module.weight_quantizer._amax_post.dtype == torch.bfloat16


class TestLSQWeightIteration:
    """Tests LSQ conversion for each weight exposed by QuantModule's iterator contract."""

    def test_multiple_singular_weight_quantizers_use_their_weight_dtypes(self, monkeypatch):
        _skip_scale_calibration(monkeypatch)
        module = QuantLinear(16, 8, bias=False, dtype=torch.bfloat16)
        module.weight_quantizer = _make_int4_static_quantizer()
        module.proj = nn.Parameter(torch.ones(8, 16, dtype=torch.float16))
        module.proj_weight_quantizer = _make_int4_static_quantizer()

        lsq(module)

        assert module.weight_quantizer._lsq
        assert module.proj_weight_quantizer._lsq
        assert module.weight_quantizer._amax_post.dtype == torch.bfloat16
        assert module.proj_weight_quantizer._amax_post.dtype == torch.float16

    def test_plural_expert_weight_quantizers_enter_lsq(self, monkeypatch):
        _skip_scale_calibration(monkeypatch)
        module = QuantLinear(16, 8, bias=False)
        module.expert_weight = nn.Parameter(torch.ones(2, 8, 16))
        module.expert_weight_quantizers = nn.ModuleList(
            [_make_int4_static_quantizer(), _make_int4_static_quantizer()]
        )

        def iter_expert_weights(self):
            yield from zip(self.expert_weight, self.expert_weight_quantizers)

        module.iter_weights_for_calibration = types.MethodType(iter_expert_weights, module)

        lsq(module)

        assert all(quantizer._lsq for quantizer in module.expert_weight_quantizers)

    def test_shared_weight_quantizer_enters_lsq_once(self, monkeypatch):
        _skip_scale_calibration(monkeypatch)
        module = QuantLinear(16, 8, bias=False)
        shared_quantizer = _make_int4_static_quantizer()

        def mark_lsq_enabled(*_args, **_kwargs):
            shared_quantizer._lsq = True

        shared_quantizer.enable_lsq = Mock(side_effect=mark_lsq_enabled)
        module.weight_quantizer = shared_quantizer
        module.proj = nn.Parameter(torch.ones(8, 16))
        module.proj_weight_quantizer = shared_quantizer

        lsq(module)

        assert shared_quantizer._lsq
        assert shared_quantizer.enable_lsq.call_count == 1
        assert module.weight_quantizer is module.proj_weight_quantizer


class TestIntCastSTE:
    """Tests for int_cast_ste (INT4 STE function)."""

    def test_round_trip(self):
        x = torch.tensor([[-3.2, 1.8, 0.0, 6.5, -7.1]], requires_grad=True)
        y = int_cast_ste(x, 4)
        assert y.shape == x.shape
        max_bound = 7.0
        assert y.min() >= -max_bound
        assert y.max() <= max_bound
        y.sum().backward()
        assert x.grad is not None

    def test_ste_gradient(self):
        x = torch.tensor([[2.3, -2.3]], requires_grad=True)
        y = int_cast_ste(x, 4)
        y.sum().backward()
        assert torch.all(x.grad == 1.0)


class TestFakeQuantizeLSQ:
    """Tests for _fake_quantize() LSQ path with INT4."""

    def _make_lsq_quantizer(self, learnable_amax=("post",), tied_amax=False):
        tq = TensorQuantizer()
        tq._num_bits = 4
        tq._unsigned = False
        tq._narrow_range = True
        tq._disabled = False
        tq._block_sizes = {-1: 16}
        tq._pass_through_bwd = True
        tq.register_buffer("_amax", torch.ones(4))
        sbsq = StaticBlockScaleQuantizer.from_tensor_quantizer(tq)
        amax = torch.ones(4) * 3.5
        sbsq.enable_lsq(
            amax, quantize_scales=False, learnable_amax=learnable_amax, tied_amax=tied_amax
        )
        return sbsq

    def test_output_shape(self):
        q = self._make_lsq_quantizer()
        x = torch.randn(4, 16)
        out = q._fake_quantize(x)
        assert out.shape == x.shape

    def test_differentiable_post(self):
        q = self._make_lsq_quantizer(learnable_amax=["post"])
        x = torch.randn(4, 16)
        out = q._fake_quantize(x)
        out.sum().backward()
        assert q._amax_post.grad is not None
        assert q._amax_pre.grad is None

    def test_differentiable_pre(self):
        q = self._make_lsq_quantizer(learnable_amax=["pre"])
        x = torch.randn(4, 16)
        out = q._fake_quantize(x)
        out.sum().backward()
        assert q._amax_pre.grad is not None
        assert q._amax_post.grad is None

    def test_differentiable_both(self):
        q = self._make_lsq_quantizer(learnable_amax=["pre", "post"])
        x = torch.randn(4, 16)
        out = q._fake_quantize(x)
        out.sum().backward()
        assert q._amax_pre.grad is not None
        assert q._amax_post.grad is not None

    def test_tied_shares_tensor(self):
        q = self._make_lsq_quantizer(learnable_amax=["pre", "post"], tied_amax=True)
        x = torch.randn(4, 16)
        out = q._fake_quantize(x)
        out.sum().backward()
        assert q._amax_post.grad is not None

    def test_skip_pre_scale_quantization_still_quantizes_post(self, monkeypatch):
        q = self._make_lsq_quantizer()
        q._quantize_scales = True
        q._quantize_pre_scale = False
        q.register_buffer("_per_tensor_scale", torch.tensor(1.0))
        calls = []

        def spy_maybe_quantize_scale(scale_raw):
            calls.append(scale_raw)
            return scale_raw

        monkeypatch.setattr(q, "_maybe_quantize_scale", spy_maybe_quantize_scale)

        out = q._fake_quantize(torch.randn(4, 16))

        assert out.shape == (4, 16)
        assert len(calls) == 1

    def test_skip_pre_scale_quantization_uses_raw_scale_floor(self, monkeypatch):
        q = self._make_lsq_quantizer()
        q._quantize_scales = True
        q._quantize_pre_scale = False
        q.register_buffer("_per_tensor_scale", torch.tensor(1.0))
        min_values = []

        def fake_amax_to_scale(amax, maxbound, min_value=None):
            min_values.append(min_value)
            return torch.ones_like(amax)

        monkeypatch.setattr(
            "modelopt.torch.quantization.nn.modules.tensor_quantizer._amax_to_scale",
            fake_amax_to_scale,
        )
        monkeypatch.setattr(q, "_maybe_quantize_scale", lambda scale_raw: scale_raw)

        out = q._fake_quantize(torch.randn(4, 16))

        assert out.shape == (4, 16)
        assert torch.equal(min_values[0], torch.tensor([_FP8_E4M3_MIN_POSITIVE]))
        assert min_values[1] == 1e-8
