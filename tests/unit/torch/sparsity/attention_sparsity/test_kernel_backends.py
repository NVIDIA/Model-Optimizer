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
# Tests: _diffusers_triton_attention forward and counter helpers
# ---------------------------------------------------------------------------


class TestDiffusersTritonForward:
    """Exercise _diffusers_triton_attention with a fake Triton backend."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        import torch

        self.torch = torch

        def _fake_attention(q, k, v, **kw):
            out = q.clone()
            if kw.get("measure_sparsity"):
                out._sparsity_total = 7
                out._sparsity_skipped = 3
            return out

        def _fake_attention_calibrate(q, k, v, threshold_trials, **kw):
            num_thresholds = len(threshold_trials)
            counters = torch.zeros(num_thresholds, 2, dtype=torch.int64)
            counters[:, 0] = 20
            counters[:, 1] = 7
            return q.clone(), counters

        # Import the real module and monkey-patch its bound attention symbols.
        # This sidesteps Python's module-attribute caching on the parent package
        # which can defeat sys.modules-based patching.
        from modelopt.torch.sparsity.attention_sparsity.kernels import (
            diffusers_triton_attention as mod,
        )

        orig_attn = mod.attention
        orig_calib = mod.attention_calibrate
        mod.attention = _fake_attention
        mod.attention_calibrate = _fake_attention_calibrate
        self.mod = mod
        try:
            yield
        finally:
            mod.attention = orig_attn
            mod.attention_calibrate = orig_calib
            mod.clear_triton_skip_softmax_config()

    def _make_qkv(self, b=1, seq_q=8, seq_k=8, h=2, d=4):
        q = self.torch.randn(b, seq_q, h, d)
        k = self.torch.randn(b, seq_k, h, d)
        v = self.torch.randn(b, seq_k, h, d)
        return q, k, v

    def test_inference_mode_default(self):
        """Default path: no skip-softmax, just pass-through."""
        q, k, v = self._make_qkv()
        out = self.mod._diffusers_triton_attention(q, k, v)
        assert out.shape == q.shape

    def test_inference_mode_with_threshold(self):
        """Static threshold configuration."""
        self.mod.set_triton_skip_softmax_config(threshold=0.05)
        q, k, v = self._make_qkv()
        out = self.mod._diffusers_triton_attention(q, k, v)
        assert out.shape == q.shape
        self.mod.clear_triton_skip_softmax_config()

    def test_inference_mode_with_raw_threshold(self):
        """Raw threshold takes precedence."""
        self.mod.set_triton_skip_softmax_config(raw_threshold=-5.0)
        q, k, v = self._make_qkv()
        out = self.mod._diffusers_triton_attention(q, k, v)
        assert out.shape == q.shape
        self.mod.clear_triton_skip_softmax_config()

    def test_inference_mode_with_scale_factor(self):
        """Scale factor used for dynamic threshold."""
        self.mod.set_triton_skip_softmax_config(scale_factor=2.0)
        q, k, v = self._make_qkv()
        out = self.mod._diffusers_triton_attention(q, k, v)
        assert out.shape == q.shape
        self.mod.clear_triton_skip_softmax_config()

    def test_different_seq_q_seq_k(self):
        """Cross-attention: seq_q != seq_k."""
        q, k, v = self._make_qkv(seq_q=8, seq_k=16)
        out = self.mod._diffusers_triton_attention(q, k, v)
        assert out.shape == q.shape

    def test_is_causal_passthrough(self):
        """is_causal flag is forwarded to the kernel wrapper."""
        q, k, v = self._make_qkv()
        out = self.mod._diffusers_triton_attention(q, k, v, is_causal=True)
        assert out.shape == q.shape

    def test_measure_sparsity_accumulates_counters(self):
        """measure_sparsity=True accumulates runtime counters."""
        self.mod.set_triton_skip_softmax_config(threshold=0.05, measure_sparsity=True)
        q, k, v = self._make_qkv()
        self.mod._diffusers_triton_attention(q, k, v)
        total, skipped = self.mod.get_sparsity_counters()
        assert total == 7
        assert skipped == 3
        # Second call accumulates
        self.mod._diffusers_triton_attention(q, k, v)
        total2, skipped2 = self.mod.get_sparsity_counters()
        assert total2 == 14
        assert skipped2 == 6
        self.mod.clear_triton_skip_softmax_config()

    def test_calibration_mode_accumulates(self):
        """Calibration mode stores and sums counters across calls."""
        self.mod.set_triton_skip_softmax_config(
            calibration_mode=True,
            threshold_trials=[0.01, 0.1],
        )
        q, k, v = self._make_qkv()
        self.mod._diffusers_triton_attention(q, k, v)
        counters1 = self.mod.get_calibration_counters()
        assert counters1 is not None
        assert counters1.shape == (2, 2)
        seq_k = self.mod.get_calibration_seq_k()
        assert seq_k == q.shape[1]

        # Second call accumulates
        self.mod._diffusers_triton_attention(q, k, v)
        counters2 = self.mod.get_calibration_counters()
        assert (counters2 == counters1 * 2).all()

        self.mod.clear_triton_skip_softmax_config()

    def test_clear_resets_counters(self):
        """clear_triton_skip_softmax_config resets all counters."""
        self.mod.set_triton_skip_softmax_config(threshold=0.05, measure_sparsity=True)
        q, k, v = self._make_qkv()
        self.mod._diffusers_triton_attention(q, k, v)
        assert self.mod.get_sparsity_counters() != (0, 0)
        self.mod.clear_triton_skip_softmax_config()
        assert self.mod.get_sparsity_counters() == (0, 0)
        assert self.mod.get_calibration_counters() is None
        assert self.mod.get_calibration_seq_k() is None

    def test_register_backend_and_get(self):
        """After registration, get_triton_attention_backend returns a context."""
        self.mod._BACKEND_REGISTERED = False
        self.mod.register_diffusers_triton_attention()
        # Even after registering, our mock returns a MagicMock — just call it.
        backend_ctx = self.mod.get_triton_attention_backend()
        assert backend_ctx is not None


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
                "modelopt.torch.sparsity.attention_sparsity.kernels.register_diffusers_triton_attention",
                MagicMock(),
            ) as mock_triton,
        ):
            _register_diffusers_backends_if_needed(mock_mixin())
            mock_triton.assert_called_once()
