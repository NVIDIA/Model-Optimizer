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
    QuantizerAttributeConfig,
)
from modelopt.torch.quantization.model_calib import lsq, max_calibrate
from modelopt.torch.quantization.nn import QuantLinear
from modelopt.torch.quantization.nn.modules.tensor_quantizer import (
    _FP8_E4M3_MIN_POSITIVE,
    StaticBlockScaleQuantizer,
    TensorQuantizer,
    _amax_to_scale,
)
from modelopt.torch.quantization.tensor_quant import int_cast_ste
from modelopt.torch.quantization.utils.shared_input import SharedWeightGlobalAmaxState
from modelopt.torch.utils import to_empty_if_meta_device


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
        model_calib_module._run_scale_calibration(model, None, cfg.scale_algorithm)
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
        q.enable_lsq(quantize_scales=False, learnable_amax=["post"], tied_amax=False)
        assert q._lsq is True
        assert isinstance(q._amax_post, nn.Parameter)
        assert q._amax_post.requires_grad is True
        assert not isinstance(q._amax_pre, nn.Parameter)
        assert not q._amax_pre.requires_grad

    def test_pre_only_learnable(self):
        q = self._make_quantizer()
        q.enable_lsq(quantize_scales=False, learnable_amax=["pre"], tied_amax=False)
        assert isinstance(q._amax_pre, nn.Parameter)
        assert q._amax_pre.requires_grad is True
        assert not isinstance(q._amax_post, nn.Parameter)

    def test_both_learnable(self):
        q = self._make_quantizer()
        q.enable_lsq(quantize_scales=False, learnable_amax=["pre", "post"], tied_amax=False)
        assert isinstance(q._amax_pre, nn.Parameter)
        assert isinstance(q._amax_post, nn.Parameter)

    def test_tied_both_learnable(self):
        q = self._make_quantizer()
        q.enable_lsq(quantize_scales=False, learnable_amax=["pre", "post"], tied_amax=True)
        assert q._tied_amax is True
        assert isinstance(q._amax_post, nn.Parameter)
        assert not hasattr(q, "_amax_pre")
        assert q.amax_pre is q._amax_post

    def test_frozen(self):
        q = self._make_quantizer()
        q.enable_lsq(quantize_scales=False, learnable_amax=[], tied_amax=False)
        assert not isinstance(q._amax_post, nn.Parameter)
        assert not isinstance(q._amax_pre, nn.Parameter)

    def test_old_amax_deleted(self):
        q = self._make_quantizer()
        assert hasattr(q, "_amax")
        q.enable_lsq(quantize_scales=False)
        assert not hasattr(q, "_amax")

    def test_can_skip_pre_scale_quantization(self):
        q = self._make_quantizer()
        q.enable_lsq(
            quantize_scales=False,
            quantize_pre_scale=False,
        )
        assert q._quantize_pre_scale is False

    def test_quantize_scales_without_global_amax_raises(self):
        q = self._make_quantizer()
        assert q.global_amax is None
        with pytest.raises(AssertionError, match="global_amax"):
            q.enable_lsq(quantize_scales=True)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_learnable_amax_uses_input_dtype(self, dtype):
        q = self._make_quantizer()
        q.enable_lsq(
            quantize_scales=False,
            learnable_amax=["pre", "post"],
            dtype=dtype,
        )

        assert q._amax_pre.dtype == dtype
        assert q._amax_post.dtype == dtype

    def test_dtype_cast_updates_learnable_amax_dtype(self):
        q = self._make_quantizer()
        q.enable_lsq(
            quantize_scales=False,
            learnable_amax=["pre", "post"],
        )

        q.to(dtype=torch.bfloat16)

        assert q._amax_pre.dtype == torch.bfloat16
        assert q._amax_post.dtype == torch.bfloat16

    def test_to_empty_if_meta_device_materializes_static_amax(self):
        q = self._make_quantizer()
        q._amax = q._amax.to("meta")
        q.global_amax = torch.tensor(1.0, device="meta")

        to_empty_if_meta_device(q, device=torch.device("cpu"))

        assert q._amax.device.type == "cpu"
        assert q.global_amax.device.type == "cpu"


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

    @pytest.mark.parametrize("distributed_sync", [False, True])
    def test_max_calibrate_promotes_static_int_quantizer(self, distributed_sync):
        module = QuantLinear(16, 8, bias=False)
        config = QuantizerAttributeConfig(num_bits=4, block_sizes={-1: 16, "type": "static"})
        module.weight_quantizer.set_from_attribute_config(config)
        module.input_quantizer.set_from_attribute_config(config)

        max_calibrate(
            module,
            forward_loop=lambda model: model(torch.randn(2, 16)),
            distributed_sync=distributed_sync,
        )

        assert isinstance(module.weight_quantizer, StaticBlockScaleQuantizer)
        assert module.weight_quantizer.export_amax() is not None
        assert not isinstance(module.input_quantizer, StaticBlockScaleQuantizer)
        assert module.input_quantizer.amax is not None


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
        tq.register_buffer("_amax", torch.ones(4) * 3.5)
        sbsq = StaticBlockScaleQuantizer.from_tensor_quantizer(tq)
        sbsq.enable_lsq(quantize_scales=False, learnable_amax=learnable_amax, tied_amax=tied_amax)
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
        # per_tensor_scale of 1.0: INT4 _quant_max_bound is 7.0, so scale = global_amax / 7.
        q.global_amax = torch.tensor(float(q._quant_max_bound))
        quantize_flags = []
        orig_block_scale = q._block_scale_from_amax

        def spy_block_scale(amax, quantize):
            quantize_flags.append(quantize)
            return orig_block_scale(amax, quantize)

        monkeypatch.setattr(q, "_block_scale_from_amax", spy_block_scale)

        out = q._fake_quantize(torch.randn(4, 16))

        assert out.shape == (4, 16)
        # post scale is FP8-quantized, pre scale is not (quantize_pre_scale=False).
        assert quantize_flags == [True, False]

    def test_skip_pre_scale_quantization_uses_raw_scale_floor(self, monkeypatch):
        q = self._make_lsq_quantizer()
        q._quantize_scales = True
        q._quantize_pre_scale = False
        q.global_amax = torch.tensor(float(q._quant_max_bound))
        min_values = []

        def fake_amax_to_scale(amax, maxbound, min_value=1e-8):
            # Only record the per-block (shape-4) scale calls, not global scale derivation.
            if amax.numel() == 4:
                min_values.append(min_value)
            return torch.ones_like(amax)

        monkeypatch.setattr(
            "modelopt.torch.quantization.nn.modules.tensor_quantizer._amax_to_scale",
            fake_amax_to_scale,
        )

        out = q._fake_quantize(torch.randn(4, 16))

        assert out.shape == (4, 16)
        assert torch.equal(min_values[0], torch.tensor([_FP8_E4M3_MIN_POSITIVE]))
        assert min_values[1] == 1e-8


class TestLSQSharedGlobalAmax:
    """Regression: LSQ must honor the shared/tied weight global_amax invariant.

    A q/k/v-style fusible group ties ``_global_amax`` to a single shared buffer object.
    Since LSQ derives the per-tensor scale from ``global_amax`` at runtime (no snapshot),
    an in-place update of the shared buffer (e.g. export unification) must propagate to
    every member. Uses INT4 (FP8-quantized scales) members so the forward runs on CPU;
    the shared-buffer mechanism under test is format-agnostic.
    """

    def _make_member(self, amax_value=2.0):
        tq = TensorQuantizer()
        tq._num_bits = 4
        tq._unsigned = False
        tq._narrow_range = True
        tq._disabled = False
        tq._block_sizes = {-1: 16, "type": "static", "scale_bits": (4, 3)}
        tq._pass_through_bwd = True
        tq.register_buffer("_amax", torch.ones(4) * amax_value)
        return StaticBlockScaleQuantizer.from_tensor_quantizer(tq)

    def _make_tied_lsq_group(self, global_amax=3.0, n_members=3):
        members = [self._make_member(amax_value=2.0) for _ in range(n_members)]
        state = SharedWeightGlobalAmaxState()
        state.global_amax = torch.tensor(float(global_amax))
        for member in members:
            assert state.tie_member_quantizer(member)
        # All members must alias the single shared buffer object.
        assert all(m._global_amax is members[0]._global_amax for m in members)
        for member in members:
            member.enable_lsq(quantize_scales=True)
        return members

    def test_block_scale_tracks_shared_update(self):
        members = self._make_tied_lsq_group(global_amax=3.0)
        new_value = 5.0
        # Mutate the shared buffer in place, mimicking export unification.
        members[0]._global_amax.data.fill_(new_value)

        for member in members:
            expected = _amax_to_scale(torch.tensor(new_value), member._quant_max_bound)
            scale = member._block_scale_from_amax(member.amax_post, quantize=True)
            assert scale.shape == member.amax_post.shape
            assert torch.all(scale >= _FP8_E4M3_MIN_POSITIVE * expected)

    def test_members_produce_identical_output_after_shared_update(self):
        members = self._make_tied_lsq_group(global_amax=3.0)
        members[0]._global_amax.data.fill_(5.0)

        x = torch.randn(4, 16)
        outputs = [member._fake_quantize(x) for member in members]
        for out in outputs[1:]:
            assert torch.equal(out, outputs[0])
