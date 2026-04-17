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

"""Unit tests for LTX Triton attention thread-local context and wrapper (no GPU)."""

import contextlib
import sys
import types
from unittest.mock import MagicMock, patch

import pytest
import torch


def _fake_attention(q, k, v, **kw):
    out = q.clone()
    if kw.get("measure_sparsity"):
        out._sparsity_total = 5
        out._sparsity_skipped = 2
    return out


def _fake_attention_calibrate(q, k, v, threshold_trials, **kw):
    num_thresholds = len(threshold_trials)
    counters = torch.zeros(num_thresholds, 2, dtype=torch.int64)
    counters[:, 0] = 10
    counters[:, 1] = 5
    return q.clone(), counters


@pytest.fixture
def ltx_mod():
    """Import ltx_triton_attention and patch its bound attention symbols."""
    from modelopt.torch.sparsity.attention_sparsity.kernels import ltx_triton_attention as mod

    orig_attn = mod.attention
    orig_calib = mod.attention_calibrate
    mod.attention = _fake_attention
    mod.attention_calibrate = _fake_attention_calibrate
    try:
        yield mod
    finally:
        mod.attention = orig_attn
        mod.attention_calibrate = orig_calib
        mod.clear_ltx_triton_context()


class TestThreadLocalContext:
    """Test set/clear/get thread-local context functions."""

    def test_set_context_populates_fields(self, ltx_mod):
        ltx_mod.set_ltx_triton_context(
            active=True,
            threshold=0.1,
            calibration_mode=False,
            threshold_trials=[0.01, 0.1],
            scale_factor=2.0,
            raw_threshold=-5.0,
        )
        active, threshold, scale_factor = ltx_mod._get_ltx_triton_context()
        assert active is True
        assert threshold == 0.1
        assert scale_factor == 2.0

    def test_set_context_without_calibration_mode_clears_counters(self, ltx_mod):
        """Setting non-calibration mode resets calibration_counters to None."""
        ltx_mod._thread_local.calibration_counters = torch.tensor([[1, 2]])
        ltx_mod.set_ltx_triton_context(active=True, calibration_mode=False)
        assert ltx_mod._thread_local.calibration_counters is None

    def test_set_context_in_calibration_mode_preserves_counters(self, ltx_mod):
        """Setting calibration mode does NOT clear the existing counters."""
        existing = torch.tensor([[5, 3]])
        ltx_mod._thread_local.calibration_counters = existing
        ltx_mod.set_ltx_triton_context(active=True, calibration_mode=True)
        assert ltx_mod._thread_local.calibration_counters is existing

    def test_clear_context_resets_all(self, ltx_mod):
        ltx_mod.set_ltx_triton_context(active=True, threshold=0.1, scale_factor=2.0)
        ltx_mod.clear_ltx_triton_context()
        active, threshold, scale_factor = ltx_mod._get_ltx_triton_context()
        assert active is False
        assert threshold is None
        assert scale_factor is None

    def test_get_calibration_counters_returns_none_initially(self, ltx_mod):
        ltx_mod.clear_ltx_triton_context()
        assert ltx_mod.get_calibration_counters() is None
        assert ltx_mod.get_calibration_seq_k() is None

    def test_get_calibration_counters_after_set(self, ltx_mod):
        fake = torch.tensor([[10, 5], [10, 8]], dtype=torch.int64)
        ltx_mod._thread_local.calibration_counters = fake
        ltx_mod._thread_local.calibration_seq_k = 4096
        assert ltx_mod.get_calibration_counters() is fake
        assert ltx_mod.get_calibration_seq_k() == 4096


class TestLTXTritonAttention:
    """Test _ltx_triton_attention wrapper with shape translation."""

    def _make_qkv(self, b=1, seq_q=16, seq_k=16, heads=2, dim_head=8):
        dim = heads * dim_head
        q = torch.randn(b, seq_q, dim)
        k = torch.randn(b, seq_k, dim)
        v = torch.randn(b, seq_k, dim)
        return q, k, v

    def test_inference_mode_shape_roundtrip(self, ltx_mod):
        """Forward returns [B, T, H*D]."""
        ltx_mod.clear_ltx_triton_context()
        q, k, v = self._make_qkv()
        out = ltx_mod._ltx_triton_attention(q, k, v, heads=2)
        assert out.shape == q.shape

    def test_inference_mode_different_seq_q_k(self, ltx_mod):
        """Forward with different Q and K seq lengths (cross-attention)."""
        ltx_mod.clear_ltx_triton_context()
        q, k, v = self._make_qkv(seq_q=16, seq_k=32)
        out = ltx_mod._ltx_triton_attention(q, k, v, heads=2)
        assert out.shape == q.shape

    def test_inference_mode_with_raw_threshold(self, ltx_mod):
        """raw_threshold path is exercised when set."""
        ltx_mod.set_ltx_triton_context(active=True, raw_threshold=-4.0)
        q, k, v = self._make_qkv()
        out = ltx_mod._ltx_triton_attention(q, k, v, heads=2)
        assert out.shape == q.shape

    def test_inference_mode_with_scale_factor(self, ltx_mod):
        """scale_factor path is exercised when raw_threshold is None."""
        ltx_mod.set_ltx_triton_context(active=True, scale_factor=2.0)
        q, k, v = self._make_qkv()
        out = ltx_mod._ltx_triton_attention(q, k, v, heads=2)
        assert out.shape == q.shape

    def test_inference_mode_with_static_threshold(self, ltx_mod):
        """Static threshold path when only `threshold` passed."""
        ltx_mod.clear_ltx_triton_context()
        q, k, v = self._make_qkv()
        out = ltx_mod._ltx_triton_attention(q, k, v, heads=2, threshold=0.1)
        assert out.shape == q.shape

    def test_calibration_mode_accumulates_counters(self, ltx_mod):
        """Calibration mode collects multi-threshold stats and accumulates."""
        ltx_mod.set_ltx_triton_context(
            active=True,
            calibration_mode=True,
            threshold_trials=[0.01, 0.1, 0.5],
        )
        q, k, v = self._make_qkv(seq_q=16, seq_k=32, heads=2, dim_head=8)

        out1 = ltx_mod._ltx_triton_attention(q, k, v, heads=2)
        assert out1.shape == q.shape
        counters1 = ltx_mod.get_calibration_counters()
        assert counters1 is not None
        assert counters1.shape == (3, 2)
        assert ltx_mod.get_calibration_seq_k() == 32

        # A second call accumulates
        ltx_mod._ltx_triton_attention(q, k, v, heads=2)
        counters2 = ltx_mod.get_calibration_counters()
        assert (counters2 == counters1 * 2).all()


class TestRegisterLTXTritonAttention:
    """Test register_ltx_triton_attention patches ltx_core Attention modules."""

    def test_no_ltx_core_no_error(self, ltx_mod):
        """If ltx_core is absent, the patch attempt raises ImportError cleanly."""
        # ltx_core may not be installed — register_ltx_triton_attention tries to
        # import it. Either it raises ImportError (no ltx_core) or it patches
        # nothing on a Linear module (no Attention instances present).
        with contextlib.suppress(ImportError, ModuleNotFoundError):
            ltx_mod.register_ltx_triton_attention(torch.nn.Linear(4, 4))

    def test_patches_ltx_attention_modules(self, ltx_mod):
        """When ltx_core.Attention exists, modules get wrapped."""

        # Build a minimal fake ltx_core.model.transformer.attention module
        class FakeAttention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attention_function = lambda q, k, v, heads, mask=None: q

        fake_attn_mod = types.ModuleType("ltx_core.model.transformer.attention")
        fake_attn_mod.Attention = FakeAttention

        ltx_core = types.ModuleType("ltx_core")
        ltx_core_model = types.ModuleType("ltx_core.model")
        ltx_core_model_xf = types.ModuleType("ltx_core.model.transformer")

        patched = {
            "ltx_core": ltx_core,
            "ltx_core.model": ltx_core_model,
            "ltx_core.model.transformer": ltx_core_model_xf,
            "ltx_core.model.transformer.attention": fake_attn_mod,
        }
        with patch.dict(sys.modules, patched):

            class Parent(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.attn1 = FakeAttention()
                    self.attn2 = FakeAttention()

            parent = Parent()
            ltx_mod.register_ltx_triton_attention(parent)

            from modelopt.torch.sparsity.attention_sparsity.kernels.ltx_triton_attention import (
                _TritonLTXAttentionWrapper,
            )

            assert isinstance(parent.attn1.attention_function, _TritonLTXAttentionWrapper)
            assert isinstance(parent.attn2.attention_function, _TritonLTXAttentionWrapper)

            # Idempotent: re-registering doesn't double-wrap
            ltx_mod.register_ltx_triton_attention(parent)
            assert not isinstance(
                parent.attn1.attention_function._original_fn,
                _TritonLTXAttentionWrapper,
            )


class TestWrapperDispatch:
    """Test _TritonLTXAttentionWrapper dispatch based on thread-local context."""

    def test_inactive_calls_original(self, ltx_mod):
        ltx_mod.clear_ltx_triton_context()
        original = MagicMock(return_value=torch.zeros(1, 4, 16))
        wrapper = ltx_mod._TritonLTXAttentionWrapper(original)
        q = torch.randn(1, 4, 16)
        k = torch.randn(1, 4, 16)
        v = torch.randn(1, 4, 16)
        wrapper(q, k, v, heads=2)
        original.assert_called_once()

    def test_active_calls_triton(self, ltx_mod):
        ltx_mod.set_ltx_triton_context(active=True, threshold=0.1)
        original = MagicMock()
        wrapper = ltx_mod._TritonLTXAttentionWrapper(original)
        q = torch.randn(1, 4, 16)
        k = torch.randn(1, 4, 16)
        v = torch.randn(1, 4, 16)
        out = wrapper(q, k, v, heads=2)
        original.assert_not_called()
        assert out.shape == q.shape
