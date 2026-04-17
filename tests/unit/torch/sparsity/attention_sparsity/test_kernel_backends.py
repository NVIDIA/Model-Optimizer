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

"""Unit tests for diffusers kernel backends and thread-local context."""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn


def _mock_diffusers():
    """Mock diffusers.models.attention_dispatch for testing without real diffusers."""
    m = types.ModuleType("diffusers.models.attention_dispatch")

    class FakeBackendName(str):
        _member_map_: dict = {}
        _value2member_map_: dict = {}

    m.AttentionBackendName = FakeBackendName

    class FakeReg:
        _backends: dict = {}
        _constraints: dict = {}
        _supported_arg_names: dict = {}

    m._AttentionBackendRegistry = FakeReg
    m.attention_backend = MagicMock()
    return {
        "diffusers": types.ModuleType("diffusers"),
        "diffusers.models": types.ModuleType("diffusers.models"),
        "diffusers.models.attention_dispatch": m,
    }


# ---------------------------------------------------------------------------
# Tests: thread-local skip-softmax context
# ---------------------------------------------------------------------------


class TestSkipSoftmaxContext:
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
    @pytest.fixture(autouse=True)
    def _setup(self):
        with patch.dict(sys.modules, _mock_diffusers()):
            from modelopt.torch.sparsity.attention_sparsity.kernels.diffusers_eager_attention import (
                _diffusers_eager_attention,
                get_skip_softmax_attention_backend,
                register_diffusers_eager_attention,
            )

            self._fn = _diffusers_eager_attention
            self._register = register_diffusers_eager_attention
            self._get_backend = get_skip_softmax_attention_backend

            import modelopt.torch.sparsity.attention_sparsity.kernels.diffusers_eager_attention as mod

            mod._BACKEND_REGISTERED = False
            yield

    def test_basic_shape(self):
        q = torch.randn(2, 8, 4, 16)
        assert self._fn(q, q, q).shape == (2, 8, 4, 16)

    def test_cross_attention(self):
        q = torch.randn(1, 4, 2, 8)
        k = torch.randn(1, 12, 2, 8)
        assert self._fn(q, k, k).shape == (1, 4, 2, 8)

    def test_causal(self):
        q = torch.randn(1, 4, 1, 8)
        assert self._fn(q, q, q, is_causal=True).shape == (1, 4, 1, 8)

    def test_gqa(self):
        q = torch.randn(1, 4, 8, 16)
        k = torch.randn(1, 4, 2, 16)
        assert self._fn(q, k, k, enable_gqa=True).shape == (1, 4, 8, 16)

    def test_register_idempotent(self):
        self._register()
        self._register()

    def test_get_backend_before_register_raises(self):
        with pytest.raises(RuntimeError, match="not registered"):
            self._get_backend()


# ---------------------------------------------------------------------------
# Tests: diffusers triton attention
# ---------------------------------------------------------------------------


class TestDiffusersTritonAttention:
    @pytest.fixture(autouse=True)
    def _setup(self):
        mocks = _mock_diffusers()
        mk = types.ModuleType("modelopt.torch.kernels")
        mk.attention = lambda q, k, v, **kw: q
        mk.attention_calibrate = None
        mk.IS_AVAILABLE = True
        mk.register_triton_attention = None
        mocks["modelopt.torch.kernels"] = mk

        with patch.dict(sys.modules, mocks):
            from modelopt.torch.sparsity.attention_sparsity.kernels.diffusers_triton_attention import (
                _diffusers_triton_attention,
                clear_triton_skip_softmax_config,
                get_triton_attention_backend,
                register_diffusers_triton_attention,
                set_triton_skip_softmax_config,
            )

            self._fn = _diffusers_triton_attention
            self._set = set_triton_skip_softmax_config
            self._clear = clear_triton_skip_softmax_config
            self._register = register_diffusers_triton_attention
            self._get_backend = get_triton_attention_backend

            import modelopt.torch.sparsity.attention_sparsity.kernels.diffusers_triton_attention as mod

            mod._BACKEND_REGISTERED = False
            yield

    def test_set_clear_config(self):
        self._set(threshold=0.1)
        self._clear()

    def test_register_idempotent(self):
        self._register()
        self._register()

    def test_get_backend_before_register_raises(self):
        with pytest.raises(RuntimeError, match="not registered"):
            self._get_backend()


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
        mock_utils = types.ModuleType("diffusers.models.modeling_utils")
        mock_utils.ModelMixin = mock_mixin

        with (
            patch.dict(sys.modules, {"diffusers.models.modeling_utils": mock_utils}),
            patch(
                "modelopt.torch.sparsity.attention_sparsity.kernels.register_diffusers_eager_attention",
                MagicMock(),
            ) as mock_eager,
            patch(
                "modelopt.torch.sparsity.attention_sparsity.kernels.register_diffusers_triton_attention",
                MagicMock(),
            ) as mock_triton,
        ):
            _register_diffusers_backends_if_needed(mock_mixin())
            mock_eager.assert_called_once()
            mock_triton.assert_called_once()
