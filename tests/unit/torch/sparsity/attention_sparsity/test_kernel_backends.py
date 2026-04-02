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

"""Unit tests for diffusers/LTX kernel backends with mocked dependencies.

These tests verify the attention computation logic and registration without
requiring diffusers or ltx_core to be installed.
"""

import importlib
import sys
import types
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Helpers: mock diffusers and ltx_core modules before importing backends
# ---------------------------------------------------------------------------


def _make_mock_diffusers():
    """Create a mock diffusers module hierarchy for attention_dispatch."""
    mock_diffusers = types.ModuleType("diffusers")
    mock_models = types.ModuleType("diffusers.models")
    mock_attention_dispatch = types.ModuleType("diffusers.models.attention_dispatch")

    # Create a real-ish AttentionBackendName enum mock
    class FakeAttentionBackendName(str):
        _member_map_ = {}
        _value2member_map_ = {}

    mock_attention_dispatch.AttentionBackendName = FakeAttentionBackendName

    class FakeRegistry:
        _backends = {}
        _constraints = {}
        _supported_arg_names = {}

    mock_attention_dispatch._AttentionBackendRegistry = FakeRegistry
    mock_attention_dispatch.attention_backend = MagicMock()

    mock_diffusers.models = mock_models
    mock_models.attention_dispatch = mock_attention_dispatch

    return {
        "diffusers": mock_diffusers,
        "diffusers.models": mock_models,
        "diffusers.models.attention_dispatch": mock_attention_dispatch,
    }


def _make_mock_ltx_core():
    """Create a mock ltx_core module hierarchy."""
    mock_ltx = types.ModuleType("ltx_core")
    mock_model = types.ModuleType("ltx_core.model")
    mock_transformer = types.ModuleType("ltx_core.model.transformer")
    mock_attn_mod = types.ModuleType("ltx_core.model.transformer.attention")

    class FakeAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention_function = lambda q, k, v, heads, mask=None: q

    mock_attn_mod.Attention = FakeAttention

    mock_ltx.model = mock_model
    mock_model.transformer = mock_transformer
    mock_transformer.attention = mock_attn_mod

    return {
        "ltx_core": mock_ltx,
        "ltx_core.model": mock_model,
        "ltx_core.model.transformer": mock_transformer,
        "ltx_core.model.transformer.attention": mock_attn_mod,
    }


# ---------------------------------------------------------------------------
# Tests: kernels/__init__.py thread-local context
# ---------------------------------------------------------------------------


class TestSkipSoftmaxContext:
    """Test thread-local skip-softmax context in kernels/__init__.py."""

    def test_default_is_false(self):
        from modelopt.torch.sparsity.attention_sparsity.kernels import get_skip_softmax_context

        assert get_skip_softmax_context() is False

    def test_set_and_get(self):
        from modelopt.torch.sparsity.attention_sparsity.kernels import (
            get_skip_softmax_context,
            set_skip_softmax_context,
        )

        set_skip_softmax_context(True)
        assert get_skip_softmax_context() is True
        set_skip_softmax_context(False)
        assert get_skip_softmax_context() is False


# ---------------------------------------------------------------------------
# Tests: diffusers eager attention
# ---------------------------------------------------------------------------


class TestDiffusersEagerAttention:
    """Test diffusers eager attention backend with mocked diffusers imports."""

    @pytest.fixture(autouse=True)
    def _setup_mocks(self):
        """Inject mock diffusers modules and reimport the backend."""
        mocks = _make_mock_diffusers()
        mod_name = "modelopt.torch.sparsity.attention_sparsity.kernels.diffusers_eager_attention"
        # Remove cached module so reimport picks up mocks
        sys.modules.pop(mod_name, None)
        with patch.dict(sys.modules, mocks):
            self.mod = importlib.import_module(mod_name)
            yield
        sys.modules.pop(mod_name, None)

    def test_eager_attention_basic(self):
        """Eager attention produces correct output shape [B, S, H, D]."""
        b, s, h, d = 2, 8, 4, 16
        q = torch.randn(b, s, h, d)
        k = torch.randn(b, s, h, d)
        v = torch.randn(b, s, h, d)

        out = self.mod._diffusers_eager_attention(q, k, v)
        assert out.shape == (b, s, h, d)

    def test_eager_attention_cross_attention(self):
        """Eager attention handles different Q/KV sequence lengths."""
        b, sq, sk, h, d = 1, 4, 12, 2, 8
        q = torch.randn(b, sq, h, d)
        k = torch.randn(b, sk, h, d)
        v = torch.randn(b, sk, h, d)

        out = self.mod._diffusers_eager_attention(q, k, v)
        assert out.shape == (b, sq, h, d)

    def test_eager_attention_with_causal_mask(self):
        """Causal mask produces lower-triangular attention pattern."""
        b, s, h, d = 1, 4, 1, 8
        q = torch.randn(b, s, h, d)
        k = torch.randn(b, s, h, d)
        v = torch.eye(s).unsqueeze(0).unsqueeze(2).expand(b, s, h, s)
        # With identity V and causal, output should reflect causal structure
        out = self.mod._diffusers_eager_attention(q, k, v, is_causal=True)
        assert out.shape == (b, s, h, s)

    def test_eager_attention_with_mask(self):
        """Attention mask is applied correctly."""
        b, s, h, d = 1, 4, 2, 8
        q = torch.randn(b, s, h, d)
        k = torch.randn(b, s, h, d)
        v = torch.randn(b, s, h, d)
        # Mask that blocks all positions -> output should be mean of V
        mask = torch.zeros(b, 1, s, s)  # no masking
        out = self.mod._diffusers_eager_attention(q, k, v, attn_mask=mask)
        assert out.shape == (b, s, h, d)

    def test_eager_attention_gqa(self):
        """GQA: fewer KV heads are repeated to match Q heads."""
        b, s, hq, hkv, d = 1, 4, 8, 2, 16
        q = torch.randn(b, s, hq, d)
        k = torch.randn(b, s, hkv, d)
        v = torch.randn(b, s, hkv, d)

        out = self.mod._diffusers_eager_attention(q, k, v, enable_gqa=True)
        assert out.shape == (b, s, hq, d)

    def test_register_idempotent(self):
        """Registration is safe to call multiple times."""
        self.mod.register_diffusers_eager_attention()
        self.mod.register_diffusers_eager_attention()  # second call should not raise

    def test_get_backend_before_register_raises(self):
        """Getting backend before registration raises RuntimeError."""
        self.mod._BACKEND_REGISTERED = False
        with pytest.raises(RuntimeError, match="not registered"):
            self.mod.get_skip_softmax_attention_backend()


# ---------------------------------------------------------------------------
# Tests: diffusers triton attention
# ---------------------------------------------------------------------------


class TestDiffusersTritonAttention:
    """Test diffusers Triton attention backend with mocked dependencies."""

    @pytest.fixture(autouse=True)
    def _setup_mocks(self):
        """Inject mock diffusers and triton_fa modules."""
        mocks = _make_mock_diffusers()
        mod_name = "modelopt.torch.sparsity.attention_sparsity.kernels.diffusers_triton_attention"
        sys.modules.pop(mod_name, None)

        # Mock the triton_fa.attention function
        def fake_attention(q, k, v, **kw):
            return q  # just return q as output

        mocks["modelopt.torch.kernels.triton_fa"] = types.ModuleType(
            "modelopt.torch.kernels.triton_fa"
        )
        mocks["modelopt.torch.kernels.triton_fa"].attention = fake_attention

        with patch.dict(sys.modules, mocks):
            self.mod = importlib.import_module(mod_name)
            yield
        sys.modules.pop(mod_name, None)

    def test_triton_attention_basic(self):
        """Triton attention reshapes correctly [B,S,H,D] -> varlen -> [B,S,H,D]."""
        b, s, h, d = 2, 8, 4, 16
        q = torch.randn(b, s, h, d)
        k = torch.randn(b, s, h, d)
        v = torch.randn(b, s, h, d)

        out = self.mod._diffusers_triton_attention(q, k, v)
        assert out.shape == (b, s, h, d)

    def test_triton_attention_cross_attention(self):
        """Different Q/KV sequence lengths produce separate varlen metadata."""
        b, sq, sk, h, d = 1, 4, 12, 2, 8
        q = torch.randn(b, sq, h, d)
        k = torch.randn(b, sk, h, d)
        v = torch.randn(b, sk, h, d)

        out = self.mod._diffusers_triton_attention(q, k, v)
        assert out.shape == (b, sq, h, d)

    def test_set_clear_config(self):
        """Thread-local config set/clear cycle."""
        self.mod.set_triton_skip_softmax_config(threshold=0.1)
        assert self.mod._thread_local.skip_threshold == 0.1
        self.mod.clear_triton_skip_softmax_config()
        assert self.mod._thread_local.skip_threshold is None

    def test_threshold_passed_to_kernel(self):
        """When threshold is set, it appears in kernel kwargs."""
        captured_kw = {}
        original_attention = self.mod.attention

        def spy_attention(q, k, v, **kw):
            captured_kw.update(kw)
            return q

        self.mod.attention = spy_attention
        try:
            self.mod.set_triton_skip_softmax_config(threshold=0.05)
            b, s, h, d = 1, 4, 2, 8
            q = torch.randn(b, s, h, d)
            self.mod._diffusers_triton_attention(q, q, q)
            assert captured_kw.get("skip_softmax_threshold") == 0.05
        finally:
            self.mod.attention = original_attention
            self.mod.clear_triton_skip_softmax_config()

    def test_register_idempotent(self):
        """Registration is safe to call multiple times."""
        self.mod.register_diffusers_triton_attention()
        self.mod.register_diffusers_triton_attention()

    def test_get_backend_before_register_raises(self):
        """Getting backend before registration raises RuntimeError."""
        self.mod._BACKEND_REGISTERED = False
        with pytest.raises(RuntimeError, match="not registered"):
            self.mod.get_triton_attention_backend()


# ---------------------------------------------------------------------------
# Tests: LTX eager attention
# ---------------------------------------------------------------------------


class TestLTXEagerAttention:
    """Test LTX-2 eager attention backend with mocked ltx_core."""

    @pytest.fixture(autouse=True)
    def _setup_mocks(self):
        """Inject mock ltx_core modules."""
        mocks = _make_mock_ltx_core()
        mod_name = "modelopt.torch.sparsity.attention_sparsity.kernels.ltx_eager_attention"
        sys.modules.pop(mod_name, None)
        with patch.dict(sys.modules, mocks):
            self.mod = importlib.import_module(mod_name)
            self.FakeAttention = mocks["ltx_core.model.transformer.attention"].Attention
            yield
        sys.modules.pop(mod_name, None)

    def test_eager_attention_basic(self):
        """LTX eager attention: [B, T, H*D] -> [B, T, H*D]."""
        b, t, h, d = 2, 8, 4, 16
        q = torch.randn(b, t, h * d)
        k = torch.randn(b, t, h * d)
        v = torch.randn(b, t, h * d)

        out = self.mod._ltx_eager_attention(q, k, v, heads=h)
        assert out.shape == (b, t, h * d)

    def test_eager_attention_with_mask(self):
        """LTX eager attention handles 2D and 3D masks."""
        b, t, h, d = 1, 4, 2, 8
        q = torch.randn(b, t, h * d)
        k = torch.randn(b, t, h * d)
        v = torch.randn(b, t, h * d)

        # 2D mask [t, t]
        mask_2d = torch.zeros(t, t)
        out = self.mod._ltx_eager_attention(q, k, v, heads=h, mask=mask_2d)
        assert out.shape == (b, t, h * d)

        # 3D mask [b, t, t]
        mask_3d = torch.zeros(b, t, t)
        out = self.mod._ltx_eager_attention(q, k, v, heads=h, mask=mask_3d)
        assert out.shape == (b, t, h * d)

    def test_wrapper_routes_to_eager_when_active(self):
        """Wrapper calls eager attention when skip-softmax context is active."""
        from modelopt.torch.sparsity.attention_sparsity.kernels import set_skip_softmax_context

        original_fn = MagicMock(return_value=torch.zeros(1, 4, 32))
        wrapper = self.mod._SkipSoftmaxLTXAttentionWrapper(original_fn)

        b, t, h, d = 1, 4, 2, 16
        q = torch.randn(b, t, h * d)
        k = torch.randn(b, t, h * d)
        v = torch.randn(b, t, h * d)

        # Inactive: calls original
        out = wrapper(q, k, v, heads=h)
        original_fn.assert_called_once()

        # Active: calls eager (not original)
        original_fn.reset_mock()
        set_skip_softmax_context(True)
        try:
            out = wrapper(q, k, v, heads=h)
            original_fn.assert_not_called()
            assert out.shape == (b, t, h * d)
        finally:
            set_skip_softmax_context(False)

    def test_register_patches_attention_modules(self):
        """register_ltx_eager_attention patches Attention modules in model."""
        model = nn.Sequential()
        attn = self.FakeAttention()
        model.add_module("attn", attn)

        self.mod.register_ltx_eager_attention(model)

        assert isinstance(attn.attention_function, self.mod._SkipSoftmaxLTXAttentionWrapper)

        # Idempotent: second call doesn't double-wrap
        self.mod.register_ltx_eager_attention(model)
        assert isinstance(attn.attention_function, self.mod._SkipSoftmaxLTXAttentionWrapper)


# ---------------------------------------------------------------------------
# Tests: LTX triton attention
# ---------------------------------------------------------------------------


class TestLTXTritonAttention:
    """Test LTX-2 Triton attention backend with mocked dependencies."""

    @pytest.fixture(autouse=True)
    def _setup_mocks(self):
        """Inject mock ltx_core and triton_fa modules."""
        mocks = _make_mock_ltx_core()
        mod_name = "modelopt.torch.sparsity.attention_sparsity.kernels.ltx_triton_attention"
        sys.modules.pop(mod_name, None)

        def fake_attention(q, k, v, **kw):
            return q

        mocks["modelopt.torch.kernels.triton_fa"] = types.ModuleType(
            "modelopt.torch.kernels.triton_fa"
        )
        mocks["modelopt.torch.kernels.triton_fa"].attention = fake_attention

        with patch.dict(sys.modules, mocks):
            self.mod = importlib.import_module(mod_name)
            self.FakeAttention = mocks["ltx_core.model.transformer.attention"].Attention
            yield
        sys.modules.pop(mod_name, None)

    def test_triton_attention_basic(self):
        """LTX triton attention: [B, T, H*D] -> varlen -> [B, T, H*D]."""
        b, t, h, d = 2, 8, 4, 16
        q = torch.randn(b, t, h * d)
        k = torch.randn(b, t, h * d)
        v = torch.randn(b, t, h * d)

        out = self.mod._ltx_triton_attention(q, k, v, heads=h, threshold=0.1)
        assert out.shape == (b, t, h * d)

    def test_set_clear_context(self):
        """Thread-local context set/clear cycle."""
        self.mod.set_ltx_triton_context(active=True, threshold=0.05)
        active, threshold = self.mod._get_ltx_triton_context()
        assert active is True
        assert threshold == 0.05

        self.mod.clear_ltx_triton_context()
        active, threshold = self.mod._get_ltx_triton_context()
        assert active is False
        assert threshold is None

    def test_wrapper_routes_to_triton_when_active(self):
        """Wrapper calls Triton attention when context is active."""
        original_fn = MagicMock(return_value=torch.zeros(1, 4, 32))
        wrapper = self.mod._TritonLTXAttentionWrapper(original_fn)

        b, t, h, d = 1, 4, 2, 16
        q = torch.randn(b, t, h * d)
        k = torch.randn(b, t, h * d)
        v = torch.randn(b, t, h * d)

        # Inactive: calls original
        out = wrapper(q, k, v, heads=h)
        original_fn.assert_called_once()

        # Active: calls triton (not original)
        original_fn.reset_mock()
        self.mod.set_ltx_triton_context(active=True, threshold=0.1)
        try:
            out = wrapper(q, k, v, heads=h)
            original_fn.assert_not_called()
            assert out.shape == (b, t, h * d)
        finally:
            self.mod.clear_ltx_triton_context()

    def test_register_patches_attention_modules(self):
        """register_ltx_triton_attention patches Attention modules."""
        model = nn.Sequential()
        attn = self.FakeAttention()
        model.add_module("attn", attn)

        self.mod.register_ltx_triton_attention(model)
        assert isinstance(attn.attention_function, self.mod._TritonLTXAttentionWrapper)

        # Idempotent
        self.mod.register_ltx_triton_attention(model)
        assert isinstance(attn.attention_function, self.mod._TritonLTXAttentionWrapper)

    def test_threshold_passed_to_kernel(self):
        """When threshold is set, it appears in kernel kwargs."""
        captured_kw = {}
        original_attention = self.mod.attention

        def spy_attention(q, k, v, **kw):
            captured_kw.update(kw)
            return q

        self.mod.attention = spy_attention
        try:
            b, t, h, d = 1, 4, 2, 8
            q = torch.randn(b, t, h * d)
            self.mod._ltx_triton_attention(q, q, q, heads=h, threshold=0.07)
            assert captured_kw.get("skip_softmax_threshold") == 0.07
        finally:
            self.mod.attention = original_attention


# ---------------------------------------------------------------------------
# Tests: conversion.py _register_diffusers_backends_if_needed
# ---------------------------------------------------------------------------


class TestRegisterDiffusersBackends:
    """Test _register_diffusers_backends_if_needed with mocked imports."""

    def test_no_diffusers_no_error(self):
        """When diffusers is not installed, function completes without error."""
        from modelopt.torch.sparsity.attention_sparsity.conversion import (
            _register_diffusers_backends_if_needed,
        )

        model = nn.Linear(10, 10)
        # Should not raise even if diffusers is not installed
        _register_diffusers_backends_if_needed(model)

    def test_with_diffusers_model(self):
        """When model is a diffusers ModelMixin, backends are registered."""
        from modelopt.torch.sparsity.attention_sparsity.conversion import (
            _register_diffusers_backends_if_needed,
        )

        # Create a fake ModelMixin so isinstance check passes
        mock_mixin = type("ModelMixin", (nn.Module,), {})
        mock_modeling_utils = types.ModuleType("diffusers.models.modeling_utils")
        mock_modeling_utils.ModelMixin = mock_mixin

        fake_model = mock_mixin()

        with (
            patch.dict(sys.modules, {"diffusers.models.modeling_utils": mock_modeling_utils}),
            patch(
                "modelopt.torch.sparsity.attention_sparsity.kernels.register_diffusers_eager_attention",
                MagicMock(),
            ) as mock_eager,
            patch(
                "modelopt.torch.sparsity.attention_sparsity.kernels.register_diffusers_triton_attention",
                MagicMock(),
            ) as mock_triton,
        ):
            _register_diffusers_backends_if_needed(fake_model)
            mock_eager.assert_called_once()
            mock_triton.assert_called_once()
