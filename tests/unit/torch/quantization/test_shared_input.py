# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Unit tests for SharedQuantState — group-level quantization state on parent modules.

These tests use a hand-built CPU model with Q/K/V siblings under a dummy
``self_attn`` parent, then drive ``max_calibrate`` directly so the run is
fast and deterministic without needing real attention/MoE layers.
"""

import pytest
import torch
import torch.nn as nn

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.model_calib import max_calibrate
from modelopt.torch.quantization.nn import NVFP4StaticQuantizer, TensorQuantizer
from modelopt.torch.quantization.utils import (
    SharedQuantState,
    attach_shared_quant_states,
    find_shared_input_groups,
    populate_shared_state,
    quantizer_attr_names,
    reduce_amax,
)

# Test-only patterns: lists of regexes (the per-kind value of ``shared_patterns``,
# e.g. the ``"weight"`` list). ``re.fullmatch``-ed against module FQNs; capture groups
# define the grouping key. ``(?:(.*)\.)?`` captures the immediate parent (or None at the
# model root, since these test models hold roles directly) -> per-parent / per-expert.
SIBLING_PATTERNS = [
    r"(?:(.*)\.)?(?:q_proj|k_proj|v_proj)",
    r"(?:(.*)\.)?(?:gate_proj|up_proj)",
    r"(?:(.*)\.)?(?:w1|w3)",
]


NVFP4_BLOCK = 16


def _make_nvfp4_static_cfg() -> QuantizerAttributeConfig:
    """NVFP4 static block quantization config (E2M1 weights + E4M3 per-block scales)."""
    return QuantizerAttributeConfig(
        num_bits=(2, 1),
        block_sizes={-1: NVFP4_BLOCK, "type": "static", "scale_bits": (4, 3)},
    )


class _DummyAttention(nn.Module):
    """A toy parent module that exposes ``q_proj``, ``k_proj``, ``v_proj`` siblings."""

    def __init__(self, in_features: int = 32, out_features: int = 32) -> None:
        super().__init__()
        self.q_proj = nn.Linear(in_features, out_features, bias=False)
        self.k_proj = nn.Linear(in_features, out_features, bias=False)
        self.v_proj = nn.Linear(in_features, out_features, bias=False)


def _populate_amax(linear: nn.Module, value: float) -> None:
    """Directly set ``_amax`` on a linear's weight_quantizer for deterministic testing."""
    wq_attr = quantizer_attr_names("weight").weight_quantizer
    wq = getattr(linear, wq_attr)
    # Match the per-block shape the real calibrator would produce
    out_features, in_features = linear.weight.shape
    n_blocks = in_features // NVFP4_BLOCK
    wq._amax = torch.full(
        (out_features, n_blocks), value, dtype=torch.float32, device=linear.weight.device
    )


class TestSharedQuantStateBasics:
    """Direct exercise of the SharedQuantState container and its helpers."""

    def test_init_unset(self):
        s = SharedQuantState()
        assert s.weight_global_amax is None

    def test_attach_creates_state_on_parent(self):
        attn = _DummyAttention()
        # Configure with NVFP4-static weight quantizers and seed _amax so attach finds them.
        mtq.replace_quant_module(attn)
        cfg = _make_nvfp4_static_cfg()
        for proj in (attn.q_proj, attn.k_proj, attn.v_proj):
            proj.weight_quantizer.set_from_attribute_config(cfg)
            _populate_amax(proj, value=1.0)

        n = attach_shared_quant_states(attn, patterns=SIBLING_PATTERNS)
        assert n == 1, f"expected one new state, got {n}"
        assert hasattr(attn, "_shared_quant_state")
        for proj in (attn.q_proj, attn.k_proj, attn.v_proj):
            assert proj.weight_quantizer._shared_quant_state_ref is attn._shared_quant_state

    def test_find_groups_skips_singletons(self):
        """A parent with only one matching child must NOT form a group."""

        class _OnlyOne(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(16, 16, bias=False)

        m = _OnlyOne()
        mtq.replace_quant_module(m)
        cfg = _make_nvfp4_static_cfg()
        m.q_proj.weight_quantizer.set_from_attribute_config(cfg)
        _populate_amax(m.q_proj, value=1.0)

        groups = find_shared_input_groups(m, patterns=SIBLING_PATTERNS)
        assert groups == [], "single sibling must not form a group"

    def test_patterns_override_hooks(self):
        """When patterns are given, hooks are skipped (patterns are the sole source)."""
        attn = _DummyAttention()
        mtq.replace_quant_module(attn)
        cfg = _make_nvfp4_static_cfg()
        for proj in (attn.q_proj, attn.k_proj, attn.v_proj):
            proj.weight_quantizer.set_from_attribute_config(cfg)
            _populate_amax(proj, value=1.0)

        def fwd(m):
            x = torch.randn(2, 32)
            for proj in (m.q_proj, m.k_proj, m.v_proj):
                proj(x)

        # Hooks alone would group all three (shared input). A q/k-only pattern must
        # win — v_proj absent proves hooks are skipped when patterns are provided.
        groups = find_shared_input_groups(
            attn, forward_loop=fwd, patterns=[r"(?:(.*)\.)?(?:q_proj|k_proj)"]
        )
        assert len(groups) == 1
        _parent, members = groups[0]
        assert len(members) == 2 and attn.v_proj not in members

    def test_populate_writes_max_across_siblings(self):
        attn = _DummyAttention()
        mtq.replace_quant_module(attn)
        cfg = _make_nvfp4_static_cfg()
        # Seed deterministic _amax values: q=1.0, k=3.0, v=2.0 → max=3.0
        attn.q_proj.weight_quantizer.set_from_attribute_config(cfg)
        attn.k_proj.weight_quantizer.set_from_attribute_config(cfg)
        attn.v_proj.weight_quantizer.set_from_attribute_config(cfg)
        _populate_amax(attn.q_proj, value=1.0)
        _populate_amax(attn.k_proj, value=3.0)
        _populate_amax(attn.v_proj, value=2.0)

        attach_shared_quant_states(attn, patterns=SIBLING_PATTERNS)
        n_groups = populate_shared_state(attn)

        assert n_groups == 1
        shared = attn._shared_quant_state.weight_global_amax
        assert shared is not None
        assert torch.isclose(shared, torch.tensor(3.0)), f"expected 3.0, got {shared.item()}"


class _MoEExpert(nn.Module):
    """A toy MoE expert with Mixtral-style ``w1`` (gate) and ``w3`` (up) projections."""

    def __init__(self, hidden=32, intermediate=32) -> None:
        super().__init__()
        self.w1 = nn.Linear(hidden, intermediate, bias=False)
        self.w3 = nn.Linear(hidden, intermediate, bias=False)


class _MoEBlock(nn.Module):
    """MoE block holding experts in an ``nn.ModuleList``."""

    def __init__(self, n_experts: int = 4) -> None:
        super().__init__()
        self.experts = nn.ModuleList(_MoEExpert() for _ in range(n_experts))


class TestMoESharedState:
    """SharedQuantState groups ``w1``/``w3`` per expert via sibling-scope patterns.

    Cross-expert grouping is intentionally not covered here: weight amax is
    per-expert (gate==up *within* an expert), so a single scale across experts
    is only meaningful for the input quantizer — to be added with that feature.
    """

    def _setup_moe(self, n_experts: int = 4) -> _MoEBlock:
        block = _MoEBlock(n_experts=n_experts)
        mtq.replace_quant_module(block)
        cfg = _make_nvfp4_static_cfg()
        for i, expert in enumerate(block.experts):
            # Distinct amax values so the max is determined and identifiable.
            for j, proj in enumerate((expert.w1, expert.w3)):
                proj.weight_quantizer.set_from_attribute_config(cfg)
                _populate_amax(proj, value=1.0 + i + 0.1 * j)
        return block

    def test_sibling_patterns_group_per_expert(self):
        """Sibling-scope w1/w3 patterns group each expert independently (per-expert)."""
        block = self._setup_moe(n_experts=4)
        groups = find_shared_input_groups(block, patterns=SIBLING_PATTERNS)
        # One [w1, w3] group per expert, parented at that expert (not the block).
        assert len(groups) == 4
        experts = list(block.experts)
        for parent, members in groups:
            assert parent in experts
            assert len(members) == 2

        attach_shared_quant_states(block, patterns=SIBLING_PATTERNS)
        n_groups = populate_shared_state(block)
        assert n_groups == 4
        assert not hasattr(block, "_shared_quant_state")  # state is on each expert, not the block
        # Each expert's max is within-expert: max(w1=1.0+i, w3=1.0+i+0.1) = 1.1 + i.
        for i, expert in enumerate(block.experts):
            shared = expert._shared_quant_state.weight_global_amax
            assert torch.isclose(shared, torch.tensor(1.1 + i)), f"expert {i}: {shared.item()}"


class TestMaxCalibrateEndToEnd:
    """End-to-end through ``max_calibrate``: same shared global_amax across siblings."""

    def _setup_attention(self, scales=(0.5, 2.0, 1.0)) -> _DummyAttention:
        """Build an attention block with NVFP4-static weight quantizers; distinct weight scales."""
        attn = _DummyAttention(in_features=32, out_features=32)
        # Bias weight magnitudes so per-projection amaxes differ.
        with torch.no_grad():
            attn.q_proj.weight.mul_(scales[0])
            attn.k_proj.weight.mul_(scales[1])
            attn.v_proj.weight.mul_(scales[2])
        mtq.replace_quant_module(attn)
        cfg = _make_nvfp4_static_cfg()
        for proj in (attn.q_proj, attn.k_proj, attn.v_proj):
            proj.weight_quantizer.set_from_attribute_config(cfg)
            # Other quantizers on the wrapped module would interfere with the
            # weight-only calibration path, so disable them.
            for name in ("input_quantizer", "output_quantizer"):
                q = getattr(proj, name, None)
                if isinstance(q, TensorQuantizer):
                    q.disable()
        return attn

    @pytest.mark.parametrize("distributed_sync", [True, False])
    def test_qkv_share_global_amax_via_max_calibrate(self, distributed_sync):
        """After max_calibrate, q/k/v_proj have identical global_amax (the group max)."""
        attn = self._setup_attention()

        # Drive a forward pass so input shape gets observed by the weight calibrators.
        def fwd(m):
            x = torch.randn(2, 32)
            m.q_proj(x)
            m.k_proj(x)
            m.v_proj(x)

        max_calibrate(attn, forward_loop=fwd, distributed_sync=distributed_sync)

        # All siblings should be promoted to NVFP4StaticQuantizer with same value.
        global_amaxes = []
        for proj in (attn.q_proj, attn.k_proj, attn.v_proj):
            assert isinstance(proj.weight_quantizer, NVFP4StaticQuantizer)
            ga = proj.weight_quantizer.global_amax
            assert ga is not None
            global_amaxes.append(ga.item())

        assert global_amaxes[0] == global_amaxes[1] == global_amaxes[2], (
            f"siblings should share global_amax, got {global_amaxes}"
        )

        # The shared value must equal the max over each child's own _amax reduction.
        per_child_max = max(
            reduce_amax(proj.weight_quantizer._amax, axis=None).item()
            for proj in (attn.q_proj, attn.k_proj, attn.v_proj)
        )
        assert global_amaxes[0] == pytest.approx(per_child_max), (
            f"shared global_amax {global_amaxes[0]} != per-child max {per_child_max}"
        )

    def test_standalone_linear_no_shared_state(self):
        """A linear without siblings has no shared state attached."""

        class _Lonely(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(32, 32, bias=False)

        m = _Lonely()
        mtq.replace_quant_module(m)
        m.proj.weight_quantizer.set_from_attribute_config(_make_nvfp4_static_cfg())
        for name in ("input_quantizer", "output_quantizer"):
            q = getattr(m.proj, name, None)
            if isinstance(q, TensorQuantizer):
                q.disable()

        max_calibrate(m, forward_loop=lambda m: m.proj(torch.randn(2, 32)), distributed_sync=False)

        # Promoted, but no shared state attached because there's no sibling group.
        assert isinstance(m.proj.weight_quantizer, NVFP4StaticQuantizer)
        assert not hasattr(m.proj.weight_quantizer, "_shared_quant_state_ref")
        # Standalone global_amax should equal its own reduce_amax(_amax).
        expected = reduce_amax(m.proj.weight_quantizer._amax, axis=None).item()
        assert m.proj.weight_quantizer.global_amax.item() == pytest.approx(expected)

    def test_shared_state_survives_state_dict_round_trip(self, tmp_path):
        """``SharedQuantState`` is an nn.Module; its buffer must round-trip through state_dict."""
        attn = self._setup_attention()

        def fwd(m):
            x = torch.randn(2, 32)
            m.q_proj(x)
            m.k_proj(x)
            m.v_proj(x)

        max_calibrate(attn, forward_loop=fwd, distributed_sync=False)

        # Shared state survives — its buffer lives in the parent's state_dict
        # under ``_shared_quant_state.weight_global_amax`` exactly once.
        assert hasattr(attn, "_shared_quant_state")
        sd = attn.state_dict()
        shared_buffer_keys = [k for k in sd if k.endswith("_shared_quant_state.weight_global_amax")]
        assert shared_buffer_keys == ["_shared_quant_state.weight_global_amax"], (
            f"buffer should appear exactly once on the parent, got {shared_buffer_keys}"
        )

        # Each weight_quantizer holds the back-reference (via object.__setattr__)
        # but it does NOT register as a child submodule — otherwise the buffer
        # would be duplicated under each member's prefix.
        for proj in (attn.q_proj, attn.k_proj, attn.v_proj):
            assert proj.weight_quantizer._shared_quant_state_ref is attn._shared_quant_state
            assert "_shared_quant_state_ref" not in proj.weight_quantizer._modules

        # state_dict round-trips under ``weights_only=True``.
        path = tmp_path / "sd.pt"
        torch.save(sd, path)
        loaded = torch.load(path, weights_only=True)
        assert loaded["_shared_quant_state.weight_global_amax"].item() == pytest.approx(
            sd["_shared_quant_state.weight_global_amax"].item()
        )


class _AttnQKV(nn.Module):
    """Attention-like module whose q/k/v consume the same input tensor."""

    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        return self.q_proj(x) + self.k_proj(x) + self.v_proj(x)


class _GLUExpert(nn.Module):
    """Expert whose w1/w3 consume the same (per-expert) input tensor."""

    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.w1 = nn.Linear(hidden, hidden, bias=False)
        self.w3 = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        return self.w1(x) * self.w3(x)


class _AttnMoE(nn.Module):
    """q/k/v attention + MoE experts; each expert gets a distinct input slice."""

    def __init__(self, n_experts: int = 2, hidden: int = 16) -> None:
        super().__init__()
        self.self_attn = _AttnQKV(hidden)
        self.experts = nn.ModuleList(_GLUExpert(hidden) for _ in range(n_experts))

    def forward(self, x):
        a = self.self_attn(x)
        for i, expert in enumerate(self.experts):
            expert(a[i : i + 1])  # distinct slice per expert -> per-expert hook grouping
        return a


class TestHookPatternEquivalence:
    """The sibling regex patterns reproduce exactly what hook discovery finds."""

    def test_hooks_match_sibling_patterns(self):
        model = _AttnMoE(n_experts=2, hidden=16)
        mtq.replace_quant_module(model)
        cfg = _make_nvfp4_static_cfg()
        linears = [model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj]
        for expert in model.experts:
            linears += [expert.w1, expert.w3]
        for lin in linears:
            lin.weight_quantizer.set_from_attribute_config(cfg)
            _populate_amax(lin, value=1.0)

        def fwd(m):
            m(torch.randn(2, 16))

        hook_groups = find_shared_input_groups(model, forward_loop=fwd)
        pattern_groups = find_shared_input_groups(model, patterns=SIBLING_PATTERNS)

        def normalize(groups):
            # Group identity = (parent module, set of member modules), order-independent.
            return {(id(parent), frozenset(id(m) for m in members)) for parent, members in groups}

        assert normalize(hook_groups) == normalize(pattern_groups)
        # Sanity: one q/k/v group + one w1/w3 group per expert (per-expert, not cross).
        assert len(hook_groups) == 3
