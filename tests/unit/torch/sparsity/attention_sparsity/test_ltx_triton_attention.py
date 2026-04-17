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

"""Unit tests for LTX Triton attention thread-local context and registration.

The ``_ltx_triton_attention`` wrapper forward pass is exercised end-to-end in
``tests/gpu/torch/sparsity/attention_sparsity/test_diffusers_triton_attention.py``.
These CPU tests only cover the thread-local context helpers and registration.
"""

import contextlib
import sys
import types
from unittest.mock import patch

import pytest
import torch


@pytest.fixture
def ltx_mod():
    """Import ltx_triton_attention and ensure thread-local state is reset."""
    from modelopt.torch.sparsity.attention_sparsity.kernels import ltx_triton_attention as mod

    mod.clear_ltx_triton_context()
    try:
        yield mod
    finally:
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
        assert ltx_mod.get_calibration_counters() is None
        assert ltx_mod.get_calibration_seq_k() is None


class TestRegisterLTXTritonAttention:
    """Test register_ltx_triton_attention patches ltx_core Attention modules."""

    def test_no_ltx_core_no_error(self, ltx_mod):
        """If ltx_core is absent, the patch attempt raises ImportError cleanly."""
        with contextlib.suppress(ImportError, ModuleNotFoundError):
            ltx_mod.register_ltx_triton_attention(torch.nn.Linear(4, 4))

    def test_patches_ltx_attention_modules(self, ltx_mod):
        """When ltx_core.Attention exists, modules get wrapped."""

        class FakeAttention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attention_function = lambda q, k, v, heads, mask=None: q

        fake_attn_mod = types.ModuleType("ltx_core.model.transformer.attention")
        fake_attn_mod.Attention = FakeAttention

        patched = {
            "ltx_core": types.ModuleType("ltx_core"),
            "ltx_core.model": types.ModuleType("ltx_core.model"),
            "ltx_core.model.transformer": types.ModuleType("ltx_core.model.transformer"),
            "ltx_core.model.transformer.attention": fake_attn_mod,
        }
        with patch.dict(sys.modules, patched):

            class Parent(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.attn1 = FakeAttention()

            parent = Parent()
            ltx_mod.register_ltx_triton_attention(parent)

            from modelopt.torch.sparsity.attention_sparsity.kernels.ltx_triton_attention import (
                _TritonLTXAttentionWrapper,
            )

            assert isinstance(parent.attn1.attention_function, _TritonLTXAttentionWrapper)
