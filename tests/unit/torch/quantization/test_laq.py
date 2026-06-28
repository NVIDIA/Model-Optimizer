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

"""CPU unit tests for the LAQ algorithm using INT4 quantization."""

import pytest
import torch
from torch import nn

from modelopt.torch.quantization.config import LAQConfig
from modelopt.torch.quantization.nn.modules.tensor_quantizer import (
    StaticBlockScaleQuantizer,
    TensorQuantizer,
)
from modelopt.torch.quantization.plugins.transformers_trainer import _align_laq_amax_param_dtypes
from modelopt.torch.quantization.tensor_quant import int_cast_ste


class TestLAQConfig:
    """Tests for LAQConfig validation."""

    def test_default_config(self):
        cfg = LAQConfig()
        assert cfg.method == "laq"
        assert cfg.learnable_amax == ["post"]
        assert cfg.tied_amax is False
        assert cfg.quantize_pre_scale is True
        assert cfg.scale_algorithm is None

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
        cfg = LAQConfig(learnable_amax=learnable_amax, tied_amax=tied_amax)
        assert cfg.tied_amax is tied_amax

    @pytest.mark.parametrize(
        "learnable_amax",
        [["post"], ["pre"], "post", "pre"],
    )
    def test_invalid_tied_with_single_learnable(self, learnable_amax):
        with pytest.raises(ValueError, match="tied_amax=True requires"):
            LAQConfig(learnable_amax=learnable_amax, tied_amax=True)


class TestEnableLAQ:
    """Tests for StaticBlockScaleQuantizer.enable_laq() with INT4 format."""

    def _make_quantizer(self):
        """Create a StaticBlockScaleQuantizer configured for INT4."""
        tq = TensorQuantizer()
        tq._num_bits = 4
        tq._unsigned = False
        tq._narrow_range = True
        tq._disabled = False
        tq._block_sizes = {-1: 16}
        tq._pass_through_bwd = True
        tq.register_buffer("_amax", torch.ones(8))
        sbsq = StaticBlockScaleQuantizer.from_tensor_quantizer(tq)
        assert sbsq._quant_max_bound == 7.0
        return sbsq

    def test_post_only_learnable(self):
        q = self._make_quantizer()
        amax = torch.ones(8) * 3.0
        q.enable_laq(amax, quantize_scales=False, learnable_amax=["post"], tied_amax=False)
        assert q._laq is True
        assert isinstance(q._amax_post, nn.Parameter)
        assert q._amax_post.requires_grad is True
        assert not isinstance(q._amax_pre, nn.Parameter)
        assert not q._amax_pre.requires_grad

    def test_pre_only_learnable(self):
        q = self._make_quantizer()
        amax = torch.ones(8) * 3.0
        q.enable_laq(amax, quantize_scales=False, learnable_amax=["pre"], tied_amax=False)
        assert isinstance(q._amax_pre, nn.Parameter)
        assert q._amax_pre.requires_grad is True
        assert not isinstance(q._amax_post, nn.Parameter)

    def test_both_learnable(self):
        q = self._make_quantizer()
        amax = torch.ones(8) * 3.0
        q.enable_laq(amax, quantize_scales=False, learnable_amax=["pre", "post"], tied_amax=False)
        assert isinstance(q._amax_pre, nn.Parameter)
        assert isinstance(q._amax_post, nn.Parameter)

    def test_tied_both_learnable(self):
        q = self._make_quantizer()
        amax = torch.ones(8) * 3.0
        q.enable_laq(amax, quantize_scales=False, learnable_amax=["pre", "post"], tied_amax=True)
        assert q._tied_amax is True
        assert isinstance(q._amax_post, nn.Parameter)
        assert not hasattr(q, "_amax_pre")
        assert q.amax_pre is q._amax_post

    def test_frozen(self):
        q = self._make_quantizer()
        amax = torch.ones(8) * 3.0
        q.enable_laq(amax, quantize_scales=False, learnable_amax=[], tied_amax=False)
        assert not isinstance(q._amax_post, nn.Parameter)
        assert not isinstance(q._amax_pre, nn.Parameter)

    def test_old_amax_deleted(self):
        q = self._make_quantizer()
        assert hasattr(q, "_amax")
        q.enable_laq(torch.ones(8), quantize_scales=False)
        assert not hasattr(q, "_amax")

    def test_can_skip_pre_scale_quantization(self):
        q = self._make_quantizer()
        q.enable_laq(
            torch.ones(8),
            quantize_scales=False,
            quantize_pre_scale=False,
        )
        assert q._quantize_pre_scale is False

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_learnable_amax_uses_input_dtype(self, dtype):
        q = self._make_quantizer()
        q.enable_laq(
            torch.ones(8, dtype=dtype),
            quantize_scales=False,
            learnable_amax=["pre", "post"],
        )

        assert q._amax_pre.dtype == dtype
        assert q._amax_post.dtype == dtype

    def test_frozen_amax_uses_fp32_storage(self):
        q = self._make_quantizer()
        q.enable_laq(
            torch.ones(8, dtype=torch.bfloat16),
            quantize_scales=False,
            learnable_amax=[],
        )

        assert q._amax_pre.dtype == torch.float32
        assert q._amax_post.dtype == torch.float32

    def test_dtype_cast_updates_learnable_amax_dtype(self):
        q = self._make_quantizer()
        q.enable_laq(
            torch.ones(8),
            quantize_scales=False,
            learnable_amax=["pre", "post"],
        )

        q.to(dtype=torch.bfloat16)

        assert q._amax_pre.dtype == torch.bfloat16
        assert q._amax_post.dtype == torch.bfloat16

    def test_align_laq_amax_param_dtypes_uses_weight_dtype(self):
        module = nn.Module()
        module.weight = nn.Parameter(torch.ones(8, 16, dtype=torch.bfloat16))
        module.weight_quantizer = self._make_quantizer()
        module.weight_quantizer.enable_laq(
            torch.ones(8),
            quantize_scales=False,
            learnable_amax=["pre", "post"],
        )

        assert module.weight_quantizer._amax_pre.dtype == torch.float32
        assert module.weight_quantizer._amax_post.dtype == torch.float32

        _align_laq_amax_param_dtypes(module)

        assert module.weight_quantizer._amax_pre.dtype == torch.bfloat16
        assert module.weight_quantizer._amax_post.dtype == torch.bfloat16


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


class TestFakeQuantizeLAQ:
    """Tests for _fake_quantize() LAQ path with INT4."""

    def _make_laq_quantizer(self, learnable_amax=("post",), tied_amax=False):
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
        sbsq.enable_laq(
            amax, quantize_scales=False, learnable_amax=learnable_amax, tied_amax=tied_amax
        )
        return sbsq

    def test_output_shape(self):
        q = self._make_laq_quantizer()
        x = torch.randn(4, 16)
        out = q._fake_quantize(x)
        assert out.shape == x.shape

    def test_differentiable_post(self):
        q = self._make_laq_quantizer(learnable_amax=["post"])
        x = torch.randn(4, 16)
        out = q._fake_quantize(x)
        out.sum().backward()
        assert q._amax_post.grad is not None
        assert q._amax_pre.grad is None

    def test_differentiable_pre(self):
        q = self._make_laq_quantizer(learnable_amax=["pre"])
        x = torch.randn(4, 16)
        out = q._fake_quantize(x)
        out.sum().backward()
        assert q._amax_pre.grad is not None
        assert q._amax_post.grad is None

    def test_differentiable_both(self):
        q = self._make_laq_quantizer(learnable_amax=["pre", "post"])
        x = torch.randn(4, 16)
        out = q._fake_quantize(x)
        out.sum().backward()
        assert q._amax_pre.grad is not None
        assert q._amax_post.grad is not None

    def test_tied_shares_tensor(self):
        q = self._make_laq_quantizer(learnable_amax=["pre", "post"], tied_amax=True)
        x = torch.randn(4, 16)
        out = q._fake_quantize(x)
        out.sum().backward()
        assert q._amax_post.grad is not None

    def test_skip_pre_scale_quantization_still_quantizes_post(self, monkeypatch):
        q = self._make_laq_quantizer()
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
        q = self._make_laq_quantizer()
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
        assert torch.equal(min_values[0], torch.tensor([0.002]))
        assert min_values[1] == 1e-8
