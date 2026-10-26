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

"""Tests for _is_sparse_sequaential_moe_block and _QuantSparseSequentialMoe."""

import copy

import pytest
import torch
import torch.nn as nn

pytest.importorskip("transformers")

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.nn import QuantModuleRegistry, TensorQuantizer
from modelopt.torch.quantization.plugins.huggingface import (
    _is_sparse_sequaential_moe_block,
    register_sparse_moe_on_the_fly,
)


# ---------------------------------------------------------------------------
# Helpers: lightweight mock modules for _is_sparse_sequaential_moe_block
# ---------------------------------------------------------------------------
class _FakeGateWithRouter(nn.Module):
    """Mimics a router gate with top_k and num_experts that returns logits."""

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


class _RoutedExpertsGate(nn.Module):
    """A router gate that names its expert count ``n_routed_experts`` and returns logits."""

    def __init__(self, top_k=2, num_experts=4):
        super().__init__()
        self.top_k = top_k
        self.n_routed_experts = num_experts
        self.linear = nn.Linear(8, num_experts)

    def forward(self, x):
        return self.linear(x)


class _MoEBlockRoutedExpertsGate(nn.Module):
    """Matches the primary detection path through gate.top_k + gate.n_routed_experts."""

    def __init__(self, num_experts=4, top_k=2):
        super().__init__()
        self.gate = _RoutedExpertsGate(top_k=top_k, num_experts=num_experts)
        self.experts = nn.ModuleList([nn.Linear(8, 8) for _ in range(num_experts)])

    def forward(self, hidden_states):
        _, selected = torch.topk(self.gate(hidden_states), self.gate.top_k, dim=-1)
        out = torch.zeros_like(hidden_states)
        for i, expert in enumerate(self.experts):
            mask = (selected == i).any(dim=-1)
            if mask.any():
                out[mask] += expert(hidden_states[mask])
        return out


# ---------------------------------------------------------------------------
# Tests for _is_sparse_sequaential_moe_block
# ---------------------------------------------------------------------------
class TestIsSparseBlock:
    def test_no_experts_returns_false(self):
        module = nn.Linear(8, 8)
        assert _is_sparse_sequaential_moe_block(module) is False

    def test_experts_but_no_gate_or_topk_returns_false(self):
        module = nn.Module()
        module.experts = nn.ModuleList([nn.Linear(8, 8)])
        assert _is_sparse_sequaential_moe_block(module) is False

    def test_gate_with_router_attrs_returns_true(self):
        block = _MoEBlockWithGateRouter(num_experts=4, top_k=2)
        assert _is_sparse_sequaential_moe_block(block) is True

    def test_fallback_block_level_attrs_returns_true(self):
        block = _MoEBlockFallback(num_experts=4, top_k=2)
        assert _is_sparse_sequaential_moe_block(block) is True

    def test_gate_missing_num_experts_returns_false(self):
        """gate.top_k present but gate.num_experts absent -> primary path fails."""
        module = nn.Module()
        module.experts = nn.ModuleList([nn.Linear(8, 8)])
        gate = nn.Module()
        gate.top_k = 2
        module.gate = gate
        assert _is_sparse_sequaential_moe_block(module) is False

    def test_gate_missing_top_k_returns_false(self):
        """gate.num_experts present but gate.top_k absent -> primary path fails."""
        module = nn.Module()
        module.experts = nn.ModuleList([nn.Linear(8, 8)])
        gate = nn.Module()
        gate.num_experts = 4
        module.gate = gate
        assert _is_sparse_sequaential_moe_block(module) is False

    def test_block_level_top_k_infers_num_experts(self):
        """top_k on block + experts with __len__ -> num_experts is inferred, returns True."""
        module = nn.Module()
        module.experts = nn.ModuleList([nn.Linear(8, 8)])
        module.top_k = 2
        assert _is_sparse_sequaential_moe_block(module) is True
        assert module.num_experts == 1

    def test_block_level_top_k_no_len_returns_false(self):
        """top_k on block but experts has no __len__ -> cannot infer num_experts, returns False."""
        module = nn.Module()
        module.experts = nn.Module()
        module.top_k = 2
        assert _is_sparse_sequaential_moe_block(module) is False

    def test_block_level_only_num_experts_returns_false(self):
        """Only num_experts on block (no top_k) -> fallback fails."""
        module = nn.Module()
        module.experts = nn.ModuleList([nn.Linear(8, 8)])
        module.num_experts = 4
        assert _is_sparse_sequaential_moe_block(module) is False

    def test_n_routed_experts_accepted(self):
        """A module with n_routed_experts (NemotronH-style) should be accepted."""
        module = nn.Module()
        module.experts = nn.ModuleList([nn.Linear(8, 8)])
        gate = nn.Module()
        gate.top_k = 2
        gate.n_routed_experts = 4
        module.gate = gate
        assert _is_sparse_sequaential_moe_block(module) is True


# ---------------------------------------------------------------------------
# Tests for _QuantSparseSequentialMoe
# ---------------------------------------------------------------------------
class TestQuantSparseSequentialMoe:
    """Tests for _QuantSparseSequentialMoe on a sequential (per-expert ``nn.Linear``) MoE block."""

    @staticmethod
    def _convert(block):
        if QuantModuleRegistry.get(type(block)) is None:
            register_sparse_moe_on_the_fly(block)
        return QuantModuleRegistry.convert(block)

    def test_register_sparse_moe_on_the_fly(self):
        block = _MoEBlockWithGateRouter()
        register_sparse_moe_on_the_fly(block)
        assert QuantModuleRegistry.get(type(block)) is not None

    def test_setup_config_knobs_default(self):
        """_setup should only initialize config knobs, no buffer or hook."""
        converted = self._convert(_MoEBlockWithGateRouter())
        assert converted._moe_calib_experts_ratio is None
        assert not hasattr(converted, "expert_token_count")

    def test_forward_default_config_passthrough(self):
        """With default config (both features off), forward should be a direct pass-through."""
        block = _MoEBlockWithGateRouter()
        ref_block = copy.deepcopy(block)
        converted = self._convert(block)

        x = torch.randn(4, 8)
        with torch.no_grad():
            assert torch.allclose(ref_block(x), converted(x), atol=1e-5)
        assert not hasattr(converted, "expert_token_count")

    def test_forward_calib_restores_top_k(self):
        """After calibration forward with moe_calib_experts_ratio, top_k should be restored."""
        converted = self._convert(_MoEBlockWithGateRouter(top_k=2))
        converted._moe_calib_experts_ratio = 1.0
        converted.experts[0]._if_calib = True  # simulate calibration mode

        with torch.no_grad():
            converted(torch.randn(4, 8))
        assert converted.gate.top_k == 2

    def test_token_counting_lazy_init(self):
        """When moe_calib_experts_ratio > 0, token counting infra is lazy-inited."""
        converted = self._convert(_MoEBlockWithGateRouter(num_experts=4, top_k=2))
        converted._moe_calib_experts_ratio = 0.5
        assert not hasattr(converted, "expert_token_count")

        converted.experts[0]._if_calib = True  # lazy-init triggers during a calibration forward
        with torch.no_grad():
            converted(torch.randn(4, 8))
        assert converted.expert_token_count.numel() == 4

        # Manually enable counting and call gate to verify the hook counts top_k per token
        converted._count_expert_tokens = True
        converted.expert_token_count.zero_()
        with torch.no_grad():
            converted.gate(torch.randn(8, 8))
        assert converted.expert_token_count.sum().item() == 8 * 2

    @pytest.mark.parametrize(
        ("block_cls", "top_k_owner"),
        [
            (_MoEBlockWithGateRouter, "gate"),
            (_MoEBlockFallback, "block"),  # top_k + num_experts on the block, nn.Linear gate
            (_MoEBlockRoutedExpertsGate, "gate"),  # gate.n_routed_experts instead of num_experts
        ],
    )
    def test_calib_forward_supports_detected_layouts(self, block_cls, top_k_owner):
        """Every layout detection accepts must also calibrate with token forcing."""
        converted = self._convert(block_cls(num_experts=4, top_k=1))
        owner = converted.gate if top_k_owner == "gate" else converted
        converted._moe_calib_experts_ratio = 0.75
        converted.experts[0]._if_calib = True

        with torch.no_grad():
            converted(torch.randn(16, 8))
        assert owner.top_k == 1  # restored after the forced pass
        # The forced pass routed each token to round(4 * 0.75) = 3 experts.
        assert converted.expert_token_count.sum().item() == 16 * 3


class _TinySequentialMoEModel(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x):
        return self.block(x)


@pytest.mark.parametrize("block_cls", [_MoEBlockWithGateRouter, _MoEBlockFallback])
def test_quantize_sequential_moe_with_token_forcing(block_cls):
    """End-to-end: mtq.quantize calibrates every expert of a sequential MoE via token forcing."""
    torch.manual_seed(0)
    model = _TinySequentialMoEModel(block_cls(num_experts=4, top_k=1))
    calib_data = [torch.randn(32, 8) for _ in range(2)]
    quant_cfg = copy.deepcopy(mtq.INT8_DEFAULT_CFG)
    quant_cfg["algorithm"] = {"method": "max", "moe_calib_experts_ratio": 0.75}

    def forward_loop(m):
        for x in calib_data:
            m(x)

    mtq.quantize(model, quant_cfg, forward_loop)

    block = model.block
    assert block._moe_calib_experts_ratio == 0.75
    # Each calibration batch routed every token to round(4 * 0.75) = 3 experts.
    assert block.expert_token_count.sum().item() == 2 * 32 * 3
    assert (block.expert_token_count > 0).all()
    for name, module in block.experts.named_modules():
        if isinstance(module, TensorQuantizer) and module.is_enabled:
            assert module.amax is not None, f"expert quantizer {name} was not calibrated"
    owner = block.gate if hasattr(block.gate, "top_k") else block
    assert owner.top_k == 1  # routing restored after calibration
    with torch.no_grad():
        assert torch.isfinite(model(torch.randn(4, 8))).all()
