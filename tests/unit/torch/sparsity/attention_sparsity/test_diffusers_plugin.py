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

"""Unit tests for the diffusers WAN sparse attention plugin.

Tests cover:
- Plugin registration: WanAttention is registered with SparseAttentionRegistry
- Processor replacement: ModelOptWanAttnProcessor is installed after sparsify()
- Config validation: "diffusers_triton" backend is accepted
- Forward shape: output shape matches input shape after sparsify()
- Enable/disable: is_enabled flag propagates to the processor
- Both methods: triton_sparse_softmax and triton_skip_softmax
"""

import pytest

pytest.importorskip("diffusers")

import torch
import torch.nn as nn

import modelopt.torch.sparsity.attention_sparsity as mtsa
from modelopt.torch.sparsity.attention_sparsity.plugins.diffusers import (
    ModelOptWanAttnProcessor,
    WanSparseAttentionModule,
    register_wan_sparse_attention,
)
from modelopt.torch.sparsity.attention_sparsity.sparse_attention import SparseAttentionModule

# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

_SPARSE_CFG = {
    "sparse_cfg": {
        "*": {
            "method": "triton_sparse_softmax",
            "sparsity_n": 2,
            "sparsity_m": 4,
            "num_sink_tokens": 0,
            "dense_window_size": 0,
            "backend": "diffusers_triton",
            "enable": True,
        },
        "default": {"enable": False},
    }
}

_SKIP_CFG = {
    "sparse_cfg": {
        "*": {
            "method": "triton_skip_softmax",
            "skip_softmax_threshold": 0.1,
            "backend": "diffusers_triton",
            "enable": True,
        },
        "default": {"enable": False},
    }
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wan_attention(dim=64, heads=4, dim_head=16):
    """Create a small WanAttention module (CPU, float32)."""
    from diffusers.models.transformers.transformer_wan import WanAttention

    return WanAttention(dim=dim, heads=heads, dim_head=dim_head)


def _make_wan_model(num_layers=2, dim=64, heads=4, dim_head=16):
    """Tiny nn.Sequential wrapping several WanAttention modules."""
    from diffusers.models.transformers.transformer_wan import WanAttention

    class _WanModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList(
                [WanAttention(dim=dim, heads=heads, dim_head=dim_head) for _ in range(num_layers)]
            )

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    return _WanModel()


def _rand_hidden(batch=1, seq=16, dim=64):
    return torch.randn(batch, seq, dim)


# ---------------------------------------------------------------------------
# Tests: config validation
# ---------------------------------------------------------------------------


class TestDiffusersTritonBackend:
    """Validate that the "diffusers_triton" backend is accepted by the config system."""

    def test_backend_accepted_sparse(self):
        from modelopt.torch.sparsity.attention_sparsity.config import SparseAttentionAttributeConfig

        cfg = SparseAttentionAttributeConfig(
            backend="diffusers_triton", method="triton_sparse_softmax"
        )
        assert cfg.backend == "diffusers_triton"

    def test_backend_accepted_skip(self):
        from modelopt.torch.sparsity.attention_sparsity.config import SparseAttentionAttributeConfig

        cfg = SparseAttentionAttributeConfig(
            backend="diffusers_triton", method="triton_skip_softmax"
        )
        assert cfg.backend == "diffusers_triton"

    def test_invalid_backend_still_rejected(self):
        from pydantic import ValidationError

        from modelopt.torch.sparsity.attention_sparsity.config import SparseAttentionAttributeConfig

        with pytest.raises(ValidationError):
            SparseAttentionAttributeConfig(backend="unknown_backend")


# ---------------------------------------------------------------------------
# Tests: plugin registration
# ---------------------------------------------------------------------------


class TestPluginRegistration:
    """register_wan_sparse_attention() should register WanAttention with the registry."""

    def test_register_on_model_with_wan_attention(self):
        model = _make_wan_model()
        result = register_wan_sparse_attention(model)
        assert result is True

    def test_register_returns_false_for_non_wan_model(self):
        model = nn.Linear(64, 64)
        result = register_wan_sparse_attention(model)
        assert result is False

    def test_register_is_idempotent(self):
        """Calling register twice should not raise or double-register."""
        model = _make_wan_model()
        register_wan_sparse_attention(model)
        register_wan_sparse_attention(model)  # second call — no error


# ---------------------------------------------------------------------------
# Tests: sparsify() replaces modules and processor
# ---------------------------------------------------------------------------


class TestSparsifyReplacement:
    """After mtsa.sparsify(), WanAttention modules become WanSparseAttentionModule."""

    def test_module_type_after_sparsify(self):
        model = _make_wan_model(num_layers=2)
        mtsa.sparsify(model, _SPARSE_CFG)

        sparse_modules = [m for m in model.modules() if isinstance(m, SparseAttentionModule)]
        assert len(sparse_modules) == 2
        for m in sparse_modules:
            assert isinstance(m, WanSparseAttentionModule)

    def test_processor_type_after_sparsify(self):
        model = _make_wan_model(num_layers=1)
        mtsa.sparsify(model, _SPARSE_CFG)

        sparse_modules = [m for m in model.modules() if isinstance(m, WanSparseAttentionModule)]
        assert len(sparse_modules) == 1
        proc = sparse_modules[0].processor
        assert isinstance(proc, ModelOptWanAttnProcessor)

    def test_processor_has_correct_sparse_kw_sparse(self):
        model = _make_wan_model(num_layers=1)
        mtsa.sparsify(model, _SPARSE_CFG)

        sparse_mod = next(m for m in model.modules() if isinstance(m, WanSparseAttentionModule))
        kw = sparse_mod.processor.sparse_kw
        assert kw["sparsity_n"] == 2
        assert kw["sparsity_m"] == 4

    def test_processor_has_correct_sparse_kw_skip(self):
        model = _make_wan_model(num_layers=1)
        mtsa.sparsify(model, _SKIP_CFG)

        sparse_mod = next(m for m in model.modules() if isinstance(m, WanSparseAttentionModule))
        kw = sparse_mod.processor.sparse_kw
        assert "skip_softmax_threshold" in kw
        assert kw["skip_softmax_threshold"] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Tests: forward pass output shape (CPU, no Triton kernel — uses SDPA fallback)
# ---------------------------------------------------------------------------


class TestForwardShape:
    """Forward pass through a sparsified model returns the correct output shape.

    The Triton kernel is not available on CPU, so ModelOptWanAttnProcessor falls
    back to dispatch_attention_fn (standard SDPA) for these tests.  We force
    this by setting _enabled=False on the processor, which exercises the SDPA
    fallback path explicitly.
    """

    def _run_with_sdpa_fallback(self, model, hidden):
        """Disable sparse modules so the SDPA fallback is used (works on CPU)."""
        for m in model.modules():
            if isinstance(m, WanSparseAttentionModule):
                m.disable()
        return model(hidden)

    def test_output_shape_self_attention(self):
        model = _make_wan_model(num_layers=1)
        mtsa.sparsify(model, _SPARSE_CFG)
        hidden = _rand_hidden(batch=2, seq=16, dim=64)
        out = self._run_with_sdpa_fallback(model, hidden)
        assert out.shape == hidden.shape

    def test_output_shape_multiple_layers(self):
        model = _make_wan_model(num_layers=3)
        mtsa.sparsify(model, _SPARSE_CFG)
        hidden = _rand_hidden(batch=1, seq=32, dim=64)
        out = self._run_with_sdpa_fallback(model, hidden)
        assert out.shape == hidden.shape

    def test_output_shape_skip_method(self):
        model = _make_wan_model(num_layers=1)
        mtsa.sparsify(model, _SKIP_CFG)
        hidden = _rand_hidden(batch=1, seq=16, dim=64)
        out = self._run_with_sdpa_fallback(model, hidden)
        assert out.shape == hidden.shape


# ---------------------------------------------------------------------------
# Tests: enable / disable
# ---------------------------------------------------------------------------


class TestEnableDisable:
    """is_enabled propagates into the processor._enabled flag on forward."""

    def test_enabled_by_default(self):
        model = _make_wan_model(num_layers=1)
        mtsa.sparsify(model, _SPARSE_CFG)
        sparse_mod = next(m for m in model.modules() if isinstance(m, WanSparseAttentionModule))
        assert sparse_mod.is_enabled is True

    def test_disable_sets_processor_flag(self):
        model = _make_wan_model(num_layers=1)
        mtsa.sparsify(model, _SPARSE_CFG)
        sparse_mod = next(m for m in model.modules() if isinstance(m, WanSparseAttentionModule))

        sparse_mod.disable()

        # Run a forward to trigger the flag sync
        hidden = _rand_hidden(batch=1, seq=8, dim=64)
        import contextlib

        with contextlib.suppress(Exception):
            sparse_mod(hidden)  # may fail without Triton; we only care about the flag

        assert sparse_mod.processor._enabled is False

    def test_enable_after_disable(self):
        model = _make_wan_model(num_layers=1)
        mtsa.sparsify(model, _SPARSE_CFG)
        sparse_mod = next(m for m in model.modules() if isinstance(m, WanSparseAttentionModule))

        sparse_mod.disable()
        sparse_mod.enable()

        hidden = _rand_hidden(batch=1, seq=8, dim=64)
        import contextlib

        with contextlib.suppress(Exception):
            sparse_mod(hidden)

        assert sparse_mod.processor._enabled is True
