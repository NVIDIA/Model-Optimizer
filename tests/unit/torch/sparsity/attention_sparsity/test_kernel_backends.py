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
requiring diffusers, ltx_core, or a GPU (triton driver).
"""

import importlib
import sys
import types
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Module names that must be cleaned from sys.modules between tests
# ---------------------------------------------------------------------------
_KERNELS_PKG = "modelopt.torch.sparsity.attention_sparsity.kernels"
_ALL_KERNEL_MODS = [
    _KERNELS_PKG,
    f"{_KERNELS_PKG}.diffusers_eager_attention",
    f"{_KERNELS_PKG}.diffusers_triton_attention",
    f"{_KERNELS_PKG}.ltx_eager_attention",
    f"{_KERNELS_PKG}.ltx_triton_attention",
]


def _purge_kernel_modules():
    """Remove all kernel backend modules from sys.modules."""
    for name in _ALL_KERNEL_MODS:
        sys.modules.pop(name, None)


# ---------------------------------------------------------------------------
# Helpers: build mock module dicts
# ---------------------------------------------------------------------------


def _make_base_mocks():
    """Mocks needed by every test: modelopt.torch.kernels + triton_fa."""
    mock_kernels = types.ModuleType("modelopt.torch.kernels")

    def fake_attention(q, k, v, **kw):
        return q

    mock_kernels.IS_AVAILABLE = True
    mock_kernels.attention = fake_attention
    mock_kernels.register_triton_attention = None

    mock_triton_fa = types.ModuleType("modelopt.torch.kernels.triton_fa")
    mock_triton_fa.attention = fake_attention

    return {
        "modelopt.torch.kernels": mock_kernels,
        "modelopt.torch.kernels.triton_fa": mock_triton_fa,
    }


def _make_mock_diffusers():
    """Mock diffusers.models.attention_dispatch."""
    mock_diffusers = types.ModuleType("diffusers")
    mock_models = types.ModuleType("diffusers.models")
    mock_ad = types.ModuleType("diffusers.models.attention_dispatch")

    class FakeBackendName(str):
        _member_map_: dict = {}
        _value2member_map_: dict = {}

    mock_ad.AttentionBackendName = FakeBackendName

    class FakeRegistry:
        _backends: dict = {}
        _constraints: dict = {}
        _supported_arg_names: dict = {}

    mock_ad._AttentionBackendRegistry = FakeRegistry
    mock_ad.attention_backend = MagicMock()

    mock_diffusers.models = mock_models
    mock_models.attention_dispatch = mock_ad
    return {
        "diffusers": mock_diffusers,
        "diffusers.models": mock_models,
        "diffusers.models.attention_dispatch": mock_ad,
    }


def _make_mock_ltx_core():
    """Mock ltx_core.model.transformer.attention."""
    mock_ltx = types.ModuleType("ltx_core")
    mock_model = types.ModuleType("ltx_core.model")
    mock_tf = types.ModuleType("ltx_core.model.transformer")
    mock_attn = types.ModuleType("ltx_core.model.transformer.attention")

    class FakeAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention_function = lambda q, k, v, heads, mask=None: q

    mock_attn.Attention = FakeAttention
    mock_ltx.model = mock_model
    mock_model.transformer = mock_tf
    mock_tf.attention = mock_attn
    return {
        "ltx_core": mock_ltx,
        "ltx_core.model": mock_model,
        "ltx_core.model.transformer": mock_tf,
        "ltx_core.model.transformer.attention": mock_attn,
    }


def _import_fresh(mod_name: str, extra_mocks: dict):
    """Purge kernel modules, patch sys.modules, and reimport ``mod_name``."""
    _purge_kernel_modules()
    mocks = {**_make_base_mocks(), **extra_mocks}
    with patch.dict(sys.modules, mocks):
        # Reimport the parent package first so submodule imports resolve
        kernels_pkg = importlib.import_module(_KERNELS_PKG)
        mod = importlib.import_module(mod_name)
    return kernels_pkg, mod


# ---------------------------------------------------------------------------
# Tests: kernels/__init__.py thread-local context
# ---------------------------------------------------------------------------


class TestSkipSoftmaxContext:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.kernels, _ = _import_fresh(_KERNELS_PKG, {})
        yield
        _purge_kernel_modules()

    def test_default_is_false(self):
        assert self.kernels.get_skip_softmax_context() is False

    def test_set_and_get(self):
        self.kernels.set_skip_softmax_context(True)
        assert self.kernels.get_skip_softmax_context() is True
        self.kernels.set_skip_softmax_context(False)
        assert self.kernels.get_skip_softmax_context() is False


# ---------------------------------------------------------------------------
# Tests: diffusers eager attention
# ---------------------------------------------------------------------------


class TestDiffusersEagerAttention:
    @pytest.fixture(autouse=True)
    def _setup(self):
        _, self.mod = _import_fresh(
            f"{_KERNELS_PKG}.diffusers_eager_attention", _make_mock_diffusers()
        )
        yield
        _purge_kernel_modules()

    def test_basic(self):
        b, s, h, d = 2, 8, 4, 16
        q = torch.randn(b, s, h, d)
        out = self.mod._diffusers_eager_attention(q, q, q)
        assert out.shape == (b, s, h, d)

    def test_cross_attention(self):
        b, sq, sk, h, d = 1, 4, 12, 2, 8
        q = torch.randn(b, sq, h, d)
        k = torch.randn(b, sk, h, d)
        v = torch.randn(b, sk, h, d)
        out = self.mod._diffusers_eager_attention(q, k, v)
        assert out.shape == (b, sq, h, d)

    def test_causal_mask(self):
        b, s, h, d = 1, 4, 1, 8
        q = torch.randn(b, s, h, d)
        v = torch.eye(s).unsqueeze(0).unsqueeze(2).expand(b, s, h, s)
        out = self.mod._diffusers_eager_attention(q, q, v, is_causal=True)
        assert out.shape == (b, s, h, s)

    def test_attn_mask(self):
        b, s, h, d = 1, 4, 2, 8
        q = torch.randn(b, s, h, d)
        mask = torch.zeros(b, 1, s, s)
        out = self.mod._diffusers_eager_attention(q, q, q, attn_mask=mask)
        assert out.shape == (b, s, h, d)

    def test_gqa(self):
        b, s, hq, hkv, d = 1, 4, 8, 2, 16
        q = torch.randn(b, s, hq, d)
        k = torch.randn(b, s, hkv, d)
        v = torch.randn(b, s, hkv, d)
        out = self.mod._diffusers_eager_attention(q, k, v, enable_gqa=True)
        assert out.shape == (b, s, hq, d)

    def test_register_idempotent(self):
        self.mod.register_diffusers_eager_attention()
        self.mod.register_diffusers_eager_attention()

    def test_get_backend_before_register_raises(self):
        self.mod._BACKEND_REGISTERED = False
        with pytest.raises(RuntimeError, match="not registered"):
            self.mod.get_skip_softmax_attention_backend()


# ---------------------------------------------------------------------------
# Tests: diffusers triton attention
# ---------------------------------------------------------------------------


class TestDiffusersTritonAttention:
    @pytest.fixture(autouse=True)
    def _setup(self):
        _, self.mod = _import_fresh(
            f"{_KERNELS_PKG}.diffusers_triton_attention", _make_mock_diffusers()
        )
        yield
        _purge_kernel_modules()

    def test_basic(self):
        b, s, h, d = 2, 8, 4, 16
        q = torch.randn(b, s, h, d)
        out = self.mod._diffusers_triton_attention(q, q, q)
        assert out.shape == (b, s, h, d)

    def test_cross_attention(self):
        b, sq, sk, h, d = 1, 4, 12, 2, 8
        q = torch.randn(b, sq, h, d)
        k = torch.randn(b, sk, h, d)
        v = torch.randn(b, sk, h, d)
        out = self.mod._diffusers_triton_attention(q, k, v)
        assert out.shape == (b, sq, h, d)

    def test_set_clear_config(self):
        self.mod.set_triton_skip_softmax_config(threshold=0.1)
        assert self.mod._thread_local.skip_threshold == 0.1
        self.mod.clear_triton_skip_softmax_config()
        assert self.mod._thread_local.skip_threshold is None

    def test_threshold_forwarded(self):
        captured = {}
        orig = self.mod.attention
        self.mod.attention = lambda q, k, v, **kw: (captured.update(kw), q)[1]
        try:
            self.mod.set_triton_skip_softmax_config(threshold=0.05)
            q = torch.randn(1, 4, 2, 8)
            self.mod._diffusers_triton_attention(q, q, q)
            assert captured.get("skip_softmax_threshold") == 0.05
        finally:
            self.mod.attention = orig
            self.mod.clear_triton_skip_softmax_config()

    def test_register_idempotent(self):
        self.mod.register_diffusers_triton_attention()
        self.mod.register_diffusers_triton_attention()

    def test_get_backend_before_register_raises(self):
        self.mod._BACKEND_REGISTERED = False
        with pytest.raises(RuntimeError, match="not registered"):
            self.mod.get_triton_attention_backend()


# ---------------------------------------------------------------------------
# Tests: LTX eager attention
# ---------------------------------------------------------------------------


class TestLTXEagerAttention:
    @pytest.fixture(autouse=True)
    def _setup(self):
        ltx_mocks = _make_mock_ltx_core()
        self.kernels, self.mod = _import_fresh(f"{_KERNELS_PKG}.ltx_eager_attention", ltx_mocks)
        self.FakeAttention = ltx_mocks["ltx_core.model.transformer.attention"].Attention
        yield
        _purge_kernel_modules()

    def test_basic(self):
        b, t, h, d = 2, 8, 4, 16
        q = torch.randn(b, t, h * d)
        out = self.mod._ltx_eager_attention(q, q, q, heads=h)
        assert out.shape == (b, t, h * d)

    def test_masks(self):
        b, t, h, d = 1, 4, 2, 8
        q = torch.randn(b, t, h * d)
        out = self.mod._ltx_eager_attention(q, q, q, heads=h, mask=torch.zeros(t, t))
        assert out.shape == (b, t, h * d)
        out = self.mod._ltx_eager_attention(q, q, q, heads=h, mask=torch.zeros(b, t, t))
        assert out.shape == (b, t, h * d)

    def test_wrapper_routing(self):
        original_fn = MagicMock(return_value=torch.zeros(1, 4, 32))
        wrapper = self.mod._SkipSoftmaxLTXAttentionWrapper(original_fn)
        b, t, h, d = 1, 4, 2, 16
        q = torch.randn(b, t, h * d)

        wrapper(q, q, q, heads=h)
        original_fn.assert_called_once()

        original_fn.reset_mock()
        self.kernels.set_skip_softmax_context(True)
        try:
            out = wrapper(q, q, q, heads=h)
            original_fn.assert_not_called()
            assert out.shape == (b, t, h * d)
        finally:
            self.kernels.set_skip_softmax_context(False)

    def test_register_idempotent(self):
        model = nn.Sequential()
        model.add_module("attn", self.FakeAttention())
        self.mod.register_ltx_eager_attention(model)
        assert isinstance(model.attn.attention_function, self.mod._SkipSoftmaxLTXAttentionWrapper)
        self.mod.register_ltx_eager_attention(model)
        assert isinstance(model.attn.attention_function, self.mod._SkipSoftmaxLTXAttentionWrapper)


# ---------------------------------------------------------------------------
# Tests: LTX triton attention
# ---------------------------------------------------------------------------


class TestLTXTritonAttention:
    @pytest.fixture(autouse=True)
    def _setup(self):
        ltx_mocks = _make_mock_ltx_core()
        _, self.mod = _import_fresh(f"{_KERNELS_PKG}.ltx_triton_attention", ltx_mocks)
        self.FakeAttention = ltx_mocks["ltx_core.model.transformer.attention"].Attention
        yield
        _purge_kernel_modules()

    def test_basic(self):
        b, t, h, d = 2, 8, 4, 16
        q = torch.randn(b, t, h * d)
        out = self.mod._ltx_triton_attention(q, q, q, heads=h, threshold=0.1)
        assert out.shape == (b, t, h * d)

    def test_set_clear_context(self):
        self.mod.set_ltx_triton_context(active=True, threshold=0.05)
        active, threshold = self.mod._get_ltx_triton_context()
        assert active is True
        assert threshold == 0.05
        self.mod.clear_ltx_triton_context()
        active, threshold = self.mod._get_ltx_triton_context()
        assert active is False
        assert threshold is None

    def test_wrapper_routing(self):
        original_fn = MagicMock(return_value=torch.zeros(1, 4, 32))
        wrapper = self.mod._TritonLTXAttentionWrapper(original_fn)
        b, t, h, d = 1, 4, 2, 16
        q = torch.randn(b, t, h * d)

        wrapper(q, q, q, heads=h)
        original_fn.assert_called_once()

        original_fn.reset_mock()
        self.mod.set_ltx_triton_context(active=True, threshold=0.1)
        try:
            out = wrapper(q, q, q, heads=h)
            original_fn.assert_not_called()
            assert out.shape == (b, t, h * d)
        finally:
            self.mod.clear_ltx_triton_context()

    def test_register_idempotent(self):
        model = nn.Sequential()
        model.add_module("attn", self.FakeAttention())
        self.mod.register_ltx_triton_attention(model)
        assert isinstance(model.attn.attention_function, self.mod._TritonLTXAttentionWrapper)
        self.mod.register_ltx_triton_attention(model)
        assert isinstance(model.attn.attention_function, self.mod._TritonLTXAttentionWrapper)

    def test_threshold_forwarded(self):
        captured = {}
        orig = self.mod.attention
        self.mod.attention = lambda q, k, v, **kw: (captured.update(kw), q)[1]
        try:
            q = torch.randn(1, 4, 16)
            self.mod._ltx_triton_attention(q, q, q, heads=2, threshold=0.07)
            assert captured.get("skip_softmax_threshold") == 0.07
        finally:
            self.mod.attention = orig


# ---------------------------------------------------------------------------
# Tests: conversion.py _register_diffusers_backends_if_needed
# ---------------------------------------------------------------------------


class TestRegisterDiffusersBackends:
    def test_no_diffusers_no_error(self):
        from modelopt.torch.sparsity.attention_sparsity.conversion import (
            _register_diffusers_backends_if_needed,
        )

        _register_diffusers_backends_if_needed(nn.Linear(10, 10))

    def test_with_diffusers_model(self):
        from modelopt.torch.sparsity.attention_sparsity.conversion import (
            _register_diffusers_backends_if_needed,
        )

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
