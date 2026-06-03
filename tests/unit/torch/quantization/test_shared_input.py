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

"""Unit tests for SharedQuantState — group-level quantization state on parent modules.

These tests use a hand-built CPU model with Q/K/V siblings under a dummy
``self_attn`` parent, then drive ``max_calibrate`` directly so the run is
fast and deterministic without needing real attention/MoE layers.
"""

import pytest
import torch
import torch.nn as nn

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.config import MaxCalibConfig, QuantizerAttributeConfig
from modelopt.torch.quantization.model_calib import max_calibrate
from modelopt.torch.quantization.nn import NVFP4StaticQuantizer, TensorQuantizer
from modelopt.torch.quantization.utils import (
    DEFAULT_WEIGHT_SHARED_PATTERNS,
    SharedQuantState,
    attach_shared_quant_states,
    find_shared_input_groups,
    populate_shared_state,
    promote_nvfp4_static_quantizers,
    quantizer_attr_names,
    reduce_amax,
)

# The production default patterns (q/k/v, gate/up, w1/w3) are exactly what these tests
# need; reuse them so the tests also exercise the real default. ``re.fullmatch``-ed
# against module FQNs; ``(?:(.*)\.)?`` captures the immediate parent (or None at the
# model root, since these test models hold roles directly) -> per-parent / per-expert.
SIBLING_PATTERNS = DEFAULT_WEIGHT_SHARED_PATTERNS


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

    def forward(self, x):
        return self.q_proj(x) + self.k_proj(x) + self.v_proj(x)


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

    def test_default_patterns_skip_non_fusible_gate(self):
        """A gate sharing the block input but never fused (e.g. ``shared_expert_gate``)
        must NOT be grouped with the gate_proj/up_proj pair.

        This is why grouping is name/pattern-based, not shared-input-hook-based: a hook
        would lump ``shared_expert_gate`` in with the GLU pair (same input tensor) and
        wrongly unify its global_amax. The default patterns match only the fused roles.
        """

        class _MLPWithGate(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = nn.Linear(32, 32, bias=False)
                self.up_proj = nn.Linear(32, 32, bias=False)
                self.shared_expert_gate = nn.Linear(32, 1, bias=False)  # shares input, not fused

        m = _MLPWithGate()
        mtq.replace_quant_module(m)
        cfg = _make_nvfp4_static_cfg()
        for lin in (m.gate_proj, m.up_proj, m.shared_expert_gate):
            lin.weight_quantizer.set_from_attribute_config(cfg)
            _populate_amax(lin, value=1.0)

        groups = find_shared_input_groups(m, patterns=DEFAULT_WEIGHT_SHARED_PATTERNS)
        assert len(groups) == 1
        _parent, members = groups[0]
        assert set(members) == {m.gate_proj, m.up_proj}
        assert m.shared_expert_gate not in members

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

    def test_populate_skips_meta_amax(self):
        """Meta (no-data) ``_amax`` must not become a meta ``weight_global_amax`` buffer.

        Quantizing an ``init_empty_weights`` model produces meta ``_amax``; aggregating it
        would make ``weight_global_amax`` a meta buffer that breaks the later meta->device
        ``.to()`` during dispatch. The group is skipped instead, leaving the buffer ``None``.
        """
        attn = _DummyAttention()
        mtq.replace_quant_module(attn)
        cfg = _make_nvfp4_static_cfg()
        for proj in (attn.q_proj, attn.k_proj, attn.v_proj):
            proj.weight_quantizer.set_from_attribute_config(cfg)
            out_features, in_features = proj.weight.shape
            proj.weight_quantizer._amax = torch.empty(
                (out_features, in_features // NVFP4_BLOCK), device="meta"
            )

        attach_shared_quant_states(attn, patterns=SIBLING_PATTERNS)  # groups q/k/v (no amax needed)
        n_groups = populate_shared_state(attn)

        assert n_groups == 0  # nothing real to aggregate
        assert attn._shared_quant_state.weight_global_amax is None  # not a meta tensor

    def test_promote_ignores_shared_state_outside_root(self):
        """Promoting a submodule must ignore a back-ref whose owning state is outside it.

        ``promote_nvfp4_static_quantizers`` also runs on submodules/individual linears; a
        quantizer may still carry ``_shared_quant_state_ref`` from an earlier full-model
        run. If the owning ``_shared_quant_state`` is not within the promotion root, the
        quantizer must fall back to its OWN amax, not the stale group value.
        """
        attn = _DummyAttention()
        mtq.replace_quant_module(attn)
        cfg = _make_nvfp4_static_cfg()
        for proj, val in ((attn.q_proj, 1.0), (attn.k_proj, 3.0), (attn.v_proj, 2.0)):
            proj.weight_quantizer.set_from_attribute_config(cfg)
            _populate_amax(proj, value=val)
        # Group on the parent → shared weight_global_amax = max = 3.0.
        attach_shared_quant_states(attn, patterns=SIBLING_PATTERNS)
        populate_shared_state(attn)
        assert torch.isclose(attn._shared_quant_state.weight_global_amax, torch.tensor(3.0))

        # Promote with q_proj as the root: it does NOT contain attn._shared_quant_state
        # (that lives on the parent), so the stale ref is ignored → own amax (1.0), not 3.0.
        promote_nvfp4_static_quantizers(attn.q_proj)
        ga = attn.q_proj.weight_quantizer.global_amax
        assert torch.isclose(ga, torch.tensor(1.0)), f"expected own amax 1.0, got {ga.item()}"


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

    def test_shared_state_buffer_is_non_persistent(self):
        """The shared buffer is a calibration-time artifact and must NOT be in state_dict.

        The scale is carried by each member's promoted quantizer (``_global_amax``), which
        IS serialized. If the shared buffer were persistent it would add
        ``_shared_quant_state.weight_global_amax`` keys that restore can't match (the
        submodule isn't re-created on load) — the regression covered end-to-end below.
        """
        attn = self._setup_attention()

        def fwd(m):
            x = torch.randn(2, 32)
            m.q_proj(x)
            m.k_proj(x)
            m.v_proj(x)

        max_calibrate(attn, forward_loop=fwd, distributed_sync=False)

        assert hasattr(attn, "_shared_quant_state")  # exists at runtime (calibration artifact)
        sd = attn.state_dict()
        # Non-persistent: the shared buffer must NOT appear in state_dict.
        assert not [k for k in sd if k.endswith("_shared_quant_state.weight_global_amax")]
        # The value lives on each member's quantizer (``_global_amax``), which IS persisted,
        # and the runtime back-reference is set (but not as a child submodule).
        for role in ("q_proj", "k_proj", "v_proj"):
            proj = getattr(attn, role)
            assert proj.weight_quantizer._shared_quant_state_ref is attn._shared_quant_state
            assert "_shared_quant_state_ref" not in proj.weight_quantizer._modules
            assert f"{role}.weight_quantizer._global_amax" in sd

    def test_modelopt_save_restore_with_shared_state(self, tmp_path):
        """``mtq.quantize`` -> ``mto.save`` -> ``mto.restore`` on a FRESH model round-trips.

        Regression for two save/restore bugs the shared state introduced: (a) the runtime
        back-ref must be excluded from ``get_modelopt_state`` (else save pickles a live
        ``QuantLinear``), and (b) the shared buffer must be non-persistent (else
        ``load_state_dict`` on the fresh, submodule-less model fails on the unexpected key).
        """
        cfg = {
            "quant_cfg": [
                {"enable": False, "quantizer_name": "*"},
                {
                    "cfg": {
                        "num_bits": (2, 1),
                        "block_sizes": {-1: NVFP4_BLOCK, "type": "static", "scale_bits": (4, 3)},
                    },
                    "quantizer_name": "*weight_quantizer",
                },
            ],
            "algorithm": "max",
        }
        attn = _DummyAttention()  # fresh (un-quantized) — mtq.quantize rejects re-quantizing
        mtq.quantize(attn, cfg, lambda m: m(torch.randn(2, 32)))

        # Grouping happened, and each member carries its own promoted global_amax.
        assert hasattr(attn, "_shared_quant_state")
        expected = {
            role: getattr(attn, role).weight_quantizer.global_amax.item()
            for role in ("q_proj", "k_proj", "v_proj")
        }

        # Save must not raise (pickling metadata.quantizer_state).
        path = tmp_path / "model.pth"
        mto.save(attn, path)

        # Restore into a fresh model must not raise (no _shared_quant_state.* key to match).
        restored = _DummyAttention()
        mto.restore(restored, path)

        for role in ("q_proj", "k_proj", "v_proj"):
            wq = getattr(restored, role).weight_quantizer
            assert isinstance(wq, NVFP4StaticQuantizer)
            assert wq.global_amax.item() == pytest.approx(expected[role])

    def test_empty_weight_patterns_disable_grouping(self):
        """``shared_patterns={"weight": []}`` disables grouping (key presence, not truthiness)."""
        attn = self._setup_attention(scales=(0.5, 2.0, 1.0))  # distinct per-proj amaxes

        def fwd(m):
            x = torch.randn(2, 32)
            m.q_proj(x)
            m.k_proj(x)
            m.v_proj(x)

        max_calibrate(
            attn, forward_loop=fwd, distributed_sync=False, shared_patterns={"weight": []}
        )

        # No sibling group: no shared state, and each proj keeps its OWN global_amax.
        assert not hasattr(attn, "_shared_quant_state")
        gas = []
        for proj in (attn.q_proj, attn.k_proj, attn.v_proj):
            assert isinstance(proj.weight_quantizer, NVFP4StaticQuantizer)
            assert not hasattr(proj.weight_quantizer, "_shared_quant_state_ref")
            gas.append(proj.weight_quantizer.global_amax.item())
        assert len(set(gas)) > 1, f"grouping should be disabled, but global_amax all equal: {gas}"

    def test_config_rejects_invalid_shared_patterns(self):
        """Bad keys and bad regexes are rejected when the config is parsed, not at calib time."""
        MaxCalibConfig(shared_patterns={"weight": [r"(?:(.*)\.)?(?:q_proj|k_proj)"]})  # valid
        MaxCalibConfig(shared_patterns={"weight": []})  # empty list is valid (disables grouping)
        with pytest.raises(ValueError, match="unsupported quantizer kind"):
            MaxCalibConfig(shared_patterns={"weigth": [r".*"]})  # typo'd key
        with pytest.raises(ValueError, match="invalid regex"):
            MaxCalibConfig(shared_patterns={"weight": ["("]})  # unbalanced paren
