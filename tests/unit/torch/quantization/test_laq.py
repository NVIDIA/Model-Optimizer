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
from modelopt.torch.quantization.tensor_quant import int_cast_ste


class TestLAQConfig:
    """Tests for LAQConfig validation."""

    def test_default_config(self):
        cfg = LAQConfig()
        assert cfg.method == "laq"
        assert cfg.learnable_amax == ["post"]
        assert cfg.tied_amax is False
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
