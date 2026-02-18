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

"""Tests for _is_sparse_moe_block and _QuantSparseMoe."""

import pytest
import torch
import torch.nn as nn

pytest.importorskip("transformers")

from _test_utils.torch.transformers_models import get_tiny_qwen3_moe

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.nn import QuantModuleRegistry
from modelopt.torch.quantization.plugins.huggingface import (
    TRANSFORMERS_VERSION_GE_5_0,
    _is_sparse_moe_block,
    register_sparse_moe_on_the_fly,
)


# ---------------------------------------------------------------------------
# Helpers: lightweight mock modules for _is_sparse_moe_block
# ---------------------------------------------------------------------------
class _FakeGateWithRouter(nn.Module):
    """Mimics a v5.x TopKRouter gate with top_k and num_experts."""

    def __init__(self, top_k=2, num_experts=4):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.linear = nn.Linear(8, num_experts)

    def forward(self, x):
        return self.linear(x)


class _FakeExperts(nn.ModuleList):
    def __init__(self, n=4):
        super().__init__([nn.Linear(8, 8) for _ in range(n)])
        self.num_experts = n


class _MoEBlockWithGateRouter(nn.Module):
    """Matches the primary detection path: gate.top_k + gate.num_experts."""

    def __init__(self, num_experts=4, top_k=2):
        super().__init__()
        self.gate = _FakeGateWithRouter(top_k=top_k, num_experts=num_experts)
        self.experts = _FakeExperts(num_experts)

    def forward(self, hidden_states):
        logits = self.gate(hidden_states)
        routing_weights, selected = torch.topk(logits, self.gate.top_k, dim=-1)
        out = torch.zeros_like(hidden_states)
        for i in range(self.gate.num_experts):
            mask = (selected == i).any(dim=-1)
            if mask.any():
                out[mask] += self.experts[i](hidden_states[mask])
        return out


class _MoEBlockFallback(nn.Module):
    """Matches the fallback path: top_k + num_experts on the block itself."""

    def __init__(self, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(8, num_experts)
        self.experts = _FakeExperts(num_experts)

    def forward(self, hidden_states):
        logits = self.gate(hidden_states)
        routing_weights, selected = torch.topk(logits, self.top_k, dim=-1)
        out = torch.zeros_like(hidden_states)
        for i in range(self.num_experts):
            mask = (selected == i).any(dim=-1)
            if mask.any():
                out[mask] += self.experts[i](hidden_states[mask])
        return out


# ---------------------------------------------------------------------------
# Tests for _is_sparse_moe_block
# ---------------------------------------------------------------------------
class TestIsSparseBlock:
    def test_no_experts_returns_false(self):
        module = nn.Linear(8, 8)
        assert _is_sparse_moe_block(module) is False

    def test_experts_but_no_gate_or_topk_returns_false(self):
        module = nn.Module()
        module.experts = nn.ModuleList([nn.Linear(8, 8)])
        assert _is_sparse_moe_block(module) is False

    def test_gate_with_router_attrs_returns_true(self):
        block = _MoEBlockWithGateRouter(num_experts=4, top_k=2)
        assert _is_sparse_moe_block(block) is True

    def test_fallback_block_level_attrs_returns_true(self):
        block = _MoEBlockFallback(num_experts=4, top_k=2)
        assert _is_sparse_moe_block(block) is True

    def test_gate_missing_num_experts_returns_false(self):
        """gate.top_k present but gate.num_experts absent -> primary path fails."""
        module = nn.Module()
        module.experts = nn.ModuleList([nn.Linear(8, 8)])
        gate = nn.Module()
        gate.top_k = 2
        module.gate = gate
        assert _is_sparse_moe_block(module) is False

    def test_gate_missing_top_k_returns_false(self):
        """gate.num_experts present but gate.top_k absent -> primary path fails."""
        module = nn.Module()
        module.experts = nn.ModuleList([nn.Linear(8, 8)])
        gate = nn.Module()
        gate.num_experts = 4
        module.gate = gate
        assert _is_sparse_moe_block(module) is False

    def test_block_level_only_top_k_returns_false(self):
        """Only top_k on block (no num_experts) -> fallback fails."""
        module = nn.Module()
        module.experts = nn.ModuleList([nn.Linear(8, 8)])
        module.top_k = 2
        assert _is_sparse_moe_block(module) is False

    def test_block_level_only_num_experts_returns_false(self):
        """Only num_experts on block (no top_k) -> fallback fails."""
        module = nn.Module()
        module.experts = nn.ModuleList([nn.Linear(8, 8)])
        module.num_experts = 4
        assert _is_sparse_moe_block(module) is False

    def test_glm4_like_block_rejected(self):
        """A module with n_routed_experts instead of num_experts should be rejected."""
        module = nn.Module()
        module.experts = nn.ModuleList([nn.Linear(8, 8)])
        gate = nn.Module()
        gate.top_k = 2
        gate.n_routed_experts = 4  # different attr name
        module.gate = gate
        assert _is_sparse_moe_block(module) is False


# ---------------------------------------------------------------------------
# Tests for _QuantSparseMoe
# ---------------------------------------------------------------------------
class TestQuantSparseMoe:
    """Tests for _QuantSparseMoe using a real tiny Qwen3Moe model."""

    @staticmethod
    def _get_moe_block(model):
        """Return the first MoE block from the model."""
        for module in model.modules():
            if _is_sparse_moe_block(module):
                return module
        raise RuntimeError("No MoE block found in model")

    def test_register_sparse_moe_on_the_fly(self):
        model = get_tiny_qwen3_moe()
        moe_block = self._get_moe_block(model)
        moe_type = type(moe_block)

        if QuantModuleRegistry.get(moe_type) is not None:
            pytest.skip("MoE type already registered (upstream change)")

        register_sparse_moe_on_the_fly(model)
        assert QuantModuleRegistry.get(moe_type) is not None

    def test_setup_creates_expert_token_count(self):
        model = get_tiny_qwen3_moe()
        moe_block = self._get_moe_block(model)
        moe_type = type(moe_block)

        if QuantModuleRegistry.get(moe_type) is None:
            register_sparse_moe_on_the_fly(model)

        converted = QuantModuleRegistry.convert(moe_block)
        assert hasattr(converted, "expert_token_count")
        expected_num_experts = moe_block.num_experts if hasattr(moe_block, "num_experts") else 0
        assert converted.expert_token_count.shape == (expected_num_experts,)
        assert converted.expert_token_count.dtype == torch.long
        assert (converted.expert_token_count == 0).all()

    def test_setup_count_expert_tokens_default_false(self):
        model = get_tiny_qwen3_moe()
        moe_block = self._get_moe_block(model)
        moe_type = type(moe_block)

        if QuantModuleRegistry.get(moe_type) is None:
            register_sparse_moe_on_the_fly(model)

        converted = QuantModuleRegistry.convert(moe_block)
        assert converted._count_expert_tokens is False

    def test_forward_no_calib_matches_original(self):
        """When calibration is off, _QuantSparseMoe should produce the same output as the original."""
        model = get_tiny_qwen3_moe()
        moe_block = self._get_moe_block(model)
        moe_type = type(moe_block)

        if QuantModuleRegistry.get(moe_type) is None:
            register_sparse_moe_on_the_fly(model)

        ref_block = self._get_moe_block(get_tiny_qwen3_moe())
        ref_block.load_state_dict(moe_block.state_dict())

        converted = QuantModuleRegistry.convert(moe_block)

        torch.manual_seed(42)
        x = torch.randn(1, 4, 32)
        with torch.no_grad():
            out_ref = ref_block(x)
            out_test = converted(x)

        if isinstance(out_ref, tuple):
            out_ref = out_ref[0]
        if isinstance(out_test, tuple):
            out_test = out_test[0]
        assert torch.allclose(out_ref, out_test, atol=1e-5)

    def test_forward_calib_sends_all_tokens_to_all_experts(self):
        """During calibration, all experts should see tokens (expert_token_count all > 0)."""
        model = get_tiny_qwen3_moe()
        register_sparse_moe_on_the_fly(model)

        def calib_fn(model):
            x = model.dummy_inputs["input_ids"]
            model(x)

        mtq.quantize(model, mtq.INT8_DEFAULT_CFG, calib_fn)

        for name, module in model.named_modules():
            if hasattr(module, "expert_token_count") and module.expert_token_count.numel() > 0:
                assert (module.expert_token_count > 0).all(), (
                    f"Not all experts received tokens in {name}: {module.expert_token_count}"
                )

    def test_forward_calib_restores_top_k(self):
        """After calibration forward, top_k should be restored to its original value."""
        model = get_tiny_qwen3_moe()
        moe_block = self._get_moe_block(model)
        moe_type = type(moe_block)

        if QuantModuleRegistry.get(moe_type) is None:
            register_sparse_moe_on_the_fly(model)

        if TRANSFORMERS_VERSION_GE_5_0:
            original_top_k = moe_block.gate.top_k
        else:
            original_top_k = moe_block.top_k

        converted = QuantModuleRegistry.convert(moe_block)

        # Simulate calibration mode: set _if_calib on a child TensorQuantizer
        for m in converted.experts.modules():
            if hasattr(m, "_if_calib"):
                m._if_calib = True
                break

        x = torch.randn(1, 4, 32)
        with torch.no_grad():
            converted(x)

        if TRANSFORMERS_VERSION_GE_5_0:
            assert converted.gate.top_k == original_top_k
        else:
            assert converted.top_k == original_top_k

    def test_gate_forward_hook_counts_tokens(self):
        """Verify the gate forward hook correctly counts expert token assignments."""
        model = get_tiny_qwen3_moe()
        moe_block = self._get_moe_block(model)
        moe_type = type(moe_block)

        if QuantModuleRegistry.get(moe_type) is None:
            register_sparse_moe_on_the_fly(model)

        converted = QuantModuleRegistry.convert(moe_block)

        # Reset counts and enable counting
        converted.expert_token_count.zero_()
        converted._count_expert_tokens = True

        hidden_size = converted.gate.in_features
        x = torch.randn(8, hidden_size)
        with torch.no_grad():
            converted.gate(x)

        # After one gate call with counting enabled, total assigned tokens should equal
        # num_tokens * top_k
        top_k = converted.top_k if hasattr(converted, "top_k") else converted.gate.top_k
        total_assigned = converted.expert_token_count.sum().item()
        assert total_assigned == 8 * top_k

        # Disable counting and verify counts don't change
        converted._count_expert_tokens = False
        prev_counts = converted.expert_token_count.clone()
        with torch.no_grad():
            converted.gate(x)
        assert torch.equal(converted.expert_token_count, prev_counts)
