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

"""Tests for _QuantFusedExperts: generic fused MoE quantization and export."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

pytest.importorskip("transformers")

import modelopt.torch.quantization as mtq
from modelopt.torch.export.moe_utils import _export_fused_experts
from modelopt.torch.export.quant_utils import get_quant_config
from modelopt.torch.quantization.conversion import _normalize_fused_experts_quantizer_name
from modelopt.torch.quantization.model_calib import local_hessian_calibrate
from modelopt.torch.quantization.nn import QuantModuleRegistry
from modelopt.torch.quantization.plugins.huggingface import (
    _is_fused_experts_module,
    _is_sparse_sequaential_moe_block,
    _QuantFusedExperts,
    normalize_experts_impl_on_the_fly,
    register_fused_experts_on_the_fly,
    register_sparse_moe_on_the_fly,
)
from modelopt.torch.quantization.utils.core_utils import weight_attr_names

# ---------------------------------------------------------------------------
# Synthetic fused expert module matching the HF transformers 5.0+ pattern
# ---------------------------------------------------------------------------
NUM_EXPERTS = 4
HIDDEN_DIM = 32
INTERMEDIATE_DIM = 16
TOP_K = 2


class _SyntheticFusedExperts(nn.Module):
    """Mimics MixtralExperts / Qwen3MoeExperts / DeepseekV3NaiveMoe from transformers 5.x."""

    def __init__(self):
        super().__init__()
        self.num_experts = NUM_EXPERTS
        self.hidden_dim = HIDDEN_DIM
        self.intermediate_dim = INTERMEDIATE_DIM
        self.gate_up_proj = nn.Parameter(
            torch.randn(NUM_EXPERTS, 2 * INTERMEDIATE_DIM, HIDDEN_DIM) * 0.02
        )
        self.down_proj = nn.Parameter(torch.randn(NUM_EXPERTS, HIDDEN_DIM, INTERMEDIATE_DIM) * 0.02)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = F.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = (
                current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            )
            final_hidden_states.index_add_(
                0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
            )
        return final_hidden_states


class _SyntheticTopKRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.top_k = TOP_K
        self.num_experts = NUM_EXPERTS
        self.weight = nn.Parameter(torch.randn(NUM_EXPERTS, HIDDEN_DIM) * 0.02)

    def forward(self, hidden_states):
        router_logits = F.linear(hidden_states, self.weight)
        router_logits = F.softmax(router_logits.float(), dim=-1)
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)
        router_top_value = router_top_value / router_top_value.sum(dim=-1, keepdim=True)
        return router_logits, router_top_value, router_indices


class _SyntheticSparseMoeBlock(nn.Module):
    """Mimics MixtralSparseMoeBlock / Qwen3MoeSparseMoeBlock."""

    def __init__(self):
        super().__init__()
        self.gate = _SyntheticTopKRouter()
        self.experts = _SyntheticFusedExperts()

    def forward(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        _, top_k_weights, top_k_index = self.gate(hidden_states)
        hidden_states = self.experts(hidden_states, top_k_index, top_k_weights)
        return hidden_states.reshape(batch_size, sequence_length, hidden_dim)


class _TinyMoEModel(nn.Module):
    """Minimal model containing a single MoE block."""

    def __init__(self):
        super().__init__()
        self.moe = _SyntheticSparseMoeBlock()

    def forward(self, x):
        return self.moe(x)


# ---------------------------------------------------------------------------
# grouped_mm backend fixtures: a synthetic fused-experts module that can run
# either the eager (F.linear) forward or the real transformers grouped_mm
# forward, selected by ``config._experts_implementation``. Used to verify
# grouped_mm calibration matches eager and that local-Hessian capture works in
# grouped mode. The class avoids the ``@use_experts_implementation`` decorator so
# it imports on transformers builds that predate the grouped_mm backend.
# ---------------------------------------------------------------------------
try:
    import transformers.integrations.moe as _moe_integration

    HAS_GROUPED_MM = hasattr(_moe_integration, "_grouped_linear")
except ImportError:
    HAS_GROUPED_MM = False


class _ExpertsConfig:
    """Minimal config carrying the experts-implementation selector."""

    def __init__(self, impl="grouped_mm"):
        self._experts_implementation = impl
        self.hidden_act = "silu"


class _SyntheticGroupedExperts(nn.Module):
    """Fused experts dispatching to eager or the real grouped_mm forward.

    Mirrors the HF transformers 5.x fused-experts contract
    (``has_gate``/``has_bias``/``is_transposed``/``_apply_gate`` + 3-D
    ``gate_up_proj``/``down_proj``) that ``grouped_mm_experts_forward`` expects.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts = NUM_EXPERTS
        self.hidden_dim = HIDDEN_DIM
        self.intermediate_dim = INTERMEDIATE_DIM
        self.has_gate = True
        self.has_bias = False
        self.is_transposed = False
        self.is_concatenated = True
        self.gate_up_proj = nn.Parameter(
            torch.randn(NUM_EXPERTS, 2 * INTERMEDIATE_DIM, HIDDEN_DIM) * 0.02
        )
        self.down_proj = nn.Parameter(torch.randn(NUM_EXPERTS, HIDDEN_DIM, INTERMEDIATE_DIM) * 0.02)
        self.act_fn = nn.SiLU()

    def _apply_gate(self, gate_up_out):
        gate, up = gate_up_out.chunk(2, dim=-1)
        return self.act_fn(gate) * up

    def _eager_forward(self, hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = F.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = (
                current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            )
            final_hidden_states.index_add_(
                0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
            )
        return final_hidden_states

    def forward(self, hidden_states, top_k_index, top_k_weights):
        if self.config._experts_implementation == "grouped_mm":
            return _moe_integration.grouped_mm_experts_forward(
                self, hidden_states, top_k_index, top_k_weights
            )
        return self._eager_forward(hidden_states, top_k_index, top_k_weights)


class _GroupedSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = _SyntheticTopKRouter()
        self.experts = _SyntheticGroupedExperts(config)

    def forward(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        _, top_k_weights, top_k_index = self.gate(hidden_states)
        hidden_states = self.experts(hidden_states, top_k_index, top_k_weights)
        return hidden_states.reshape(batch_size, sequence_length, hidden_dim)


class _GroupedMoEModel(nn.Module):
    """Minimal MoE model whose experts run the selected backend."""

    def __init__(self, impl):
        super().__init__()
        self.config = _ExpertsConfig(impl=impl)
        self.moe = _GroupedSparseMoeBlock(self.config)

    def forward(self, x):
        return self.moe(x)


# ---------------------------------------------------------------------------
# Tests for _is_fused_experts_module
# ---------------------------------------------------------------------------
class TestIsFusedExpertsModule:
    def test_synthetic_fused_experts_detected(self):
        module = _SyntheticFusedExperts()
        assert _is_fused_experts_module(module) is True

    def test_plain_module_not_detected(self):
        assert _is_fused_experts_module(nn.Linear(8, 8)) is False

    def test_module_with_2d_gate_up_not_detected(self):
        module = nn.Module()
        module.gate_up_proj = nn.Parameter(torch.randn(16, 8))
        module.down_proj = nn.Parameter(torch.randn(8, 16))
        module.num_experts = 4
        module.act_fn = nn.SiLU()
        assert _is_fused_experts_module(module) is False

    def test_module_missing_act_fn_not_detected(self):
        module = nn.Module()
        module.gate_up_proj = nn.Parameter(torch.randn(4, 16, 8))
        module.down_proj = nn.Parameter(torch.randn(4, 8, 16))
        module.num_experts = 4
        assert _is_fused_experts_module(module) is False

    def test_sparse_moe_block_not_detected_as_fused(self):
        block = _SyntheticSparseMoeBlock()
        assert _is_fused_experts_module(block) is False

    def test_fused_moe_block_not_detected_as_sequential(self):
        """Fused MoE blocks (non-iterable experts) should not be detected as sequential."""
        block = _SyntheticSparseMoeBlock()
        assert _is_sparse_sequaential_moe_block(block) is False


# ---------------------------------------------------------------------------
# Tests for registration and quantization
# ---------------------------------------------------------------------------
class TestQuantFusedExperts:
    @staticmethod
    def _cleanup_registry(mod_type):
        if QuantModuleRegistry.get(mod_type) is not None:
            QuantModuleRegistry.unregister(mod_type)

    def test_register_fused_experts_on_the_fly(self):
        model = _TinyMoEModel()
        expert_type = type(model.moe.experts)
        self._cleanup_registry(expert_type)
        self._cleanup_registry(type(model.moe))

        register_fused_experts_on_the_fly(model)
        assert QuantModuleRegistry.get(expert_type) is not None
        self._cleanup_registry(expert_type)

    def test_fused_experts_only_registration(self):
        """Fused MoE: only the expert module is registered, not the block (non-iterable experts)."""
        model = _TinyMoEModel()
        expert_type = type(model.moe.experts)
        block_type = type(model.moe)
        self._cleanup_registry(expert_type)
        self._cleanup_registry(block_type)

        register_fused_experts_on_the_fly(model)
        register_sparse_moe_on_the_fly(model)
        assert QuantModuleRegistry.get(expert_type) is not None
        # Block is NOT registered because fused experts are not iterable
        assert QuantModuleRegistry.get(block_type) is None
        self._cleanup_registry(expert_type)

    def test_convert_creates_quantizers(self):
        """After conversion, fused experts should have shared input and per-expert weight quantizers."""
        model = _TinyMoEModel()
        expert_type = type(model.moe.experts)
        self._cleanup_registry(expert_type)

        register_fused_experts_on_the_fly(model)
        converted = QuantModuleRegistry.convert(model.moe.experts)

        # Shared input quantizers (single TensorQuantizer, not ModuleList)
        assert hasattr(converted, "gate_up_proj_input_quantizer")
        assert hasattr(converted, "down_proj_input_quantizer")
        # Per-expert weight quantizers (ModuleList)
        assert hasattr(converted, "gate_up_proj_weight_quantizers")
        assert hasattr(converted, "down_proj_weight_quantizers")
        assert len(converted.gate_up_proj_weight_quantizers) == NUM_EXPERTS
        assert len(converted.down_proj_weight_quantizers) == NUM_EXPERTS
        self._cleanup_registry(expert_type)

    def test_forward_passthrough_matches(self):
        """Forward through _QuantFusedExperts should match the original (quantizers disabled)."""
        model = _TinyMoEModel()
        expert_type = type(model.moe.experts)
        self._cleanup_registry(expert_type)

        ref_experts = _SyntheticFusedExperts()
        ref_experts.load_state_dict(model.moe.experts.state_dict())

        register_fused_experts_on_the_fly(model)
        converted = QuantModuleRegistry.convert(model.moe.experts)

        seq_len = 8
        hidden_states = torch.randn(seq_len, HIDDEN_DIM)
        top_k_index = torch.randint(0, NUM_EXPERTS, (seq_len, TOP_K))
        top_k_weights = torch.softmax(torch.randn(seq_len, TOP_K), dim=-1)

        with torch.no_grad():
            out_ref = ref_experts(hidden_states, top_k_index, top_k_weights)
            out_test = converted(hidden_states, top_k_index, top_k_weights)

        assert torch.allclose(out_ref, out_test, atol=1e-4), (
            f"Max diff: {(out_ref - out_test).abs().max().item()}"
        )
        self._cleanup_registry(expert_type)

    def test_expert_index_recovery(self):
        """Storage-offset based expert index recovery should be correct."""
        experts = _SyntheticFusedExperts()
        expert_type = type(experts)
        self._cleanup_registry(expert_type)

        register_fused_experts_on_the_fly(_TinyMoEModel())
        converted = QuantModuleRegistry.convert(experts)

        for idx in range(NUM_EXPERTS):
            weight_slice = converted.gate_up_proj[idx]
            recovered_idx = converted._get_expert_idx_from_gate_up(weight_slice)
            assert recovered_idx == idx, f"Expected {idx}, got {recovered_idx}"
        self._cleanup_registry(expert_type)


# ---------------------------------------------------------------------------
# Tests for export
# ---------------------------------------------------------------------------
class TestExportFusedExperts:
    @staticmethod
    def _cleanup_registry(mod_type):
        if QuantModuleRegistry.get(mod_type) is not None:
            QuantModuleRegistry.unregister(mod_type)

    def test_export_creates_per_expert_submodules(self):
        """_export_fused_experts should create per-expert submodules with standard naming."""
        model = _TinyMoEModel()
        expert_type = type(model.moe.experts)
        self._cleanup_registry(expert_type)

        quant_cfg = {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {
                    "quantizer_name": "*gate_up_proj_input_quantizer",
                    "cfg": {"num_bits": 8, "axis": None},
                },
                {
                    "quantizer_name": "*down_proj_input_quantizer",
                    "cfg": {"num_bits": 8, "axis": None},
                },
                {
                    "quantizer_name": "*gate_up_proj_weight_quantizer",
                    "cfg": {"num_bits": 8, "axis": 0},
                },
                {
                    "quantizer_name": "*down_proj_weight_quantizer",
                    "cfg": {"num_bits": 8, "axis": 0},
                },
            ],
            "algorithm": "max",
        }

        def forward_loop(m):
            torch.manual_seed(0)
            for _ in range(2):
                x = torch.randn(1, 4, HIDDEN_DIM)
                m(x)

        mtq.quantize(model, quant_cfg, forward_loop=forward_loop)
        converted = model.moe.experts

        _export_fused_experts(converted, torch.float16)

        # Verify per-expert submodules exist
        for idx in range(NUM_EXPERTS):
            expert_mod = getattr(converted, str(idx), None)
            assert expert_mod is not None, f"Missing expert submodule {idx}"
            assert hasattr(expert_mod, "gate_proj"), f"Expert {idx} missing gate_proj"
            assert hasattr(expert_mod, "up_proj"), f"Expert {idx} missing up_proj"
            assert hasattr(expert_mod, "down_proj"), f"Expert {idx} missing down_proj"

            assert expert_mod.gate_proj.weight.shape == (INTERMEDIATE_DIM, HIDDEN_DIM)
            assert expert_mod.up_proj.weight.shape == (INTERMEDIATE_DIM, HIDDEN_DIM)
            assert expert_mod.down_proj.weight.shape == (HIDDEN_DIM, INTERMEDIATE_DIM)

        # Verify fused params are removed
        assert not hasattr(converted, "gate_up_proj")
        assert not hasattr(converted, "down_proj")
        assert not hasattr(converted, "gate_up_proj_weight_quantizers")

        self._cleanup_registry(expert_type)

    def test_uncalibrated_expert_gate_up_share_amax(self, monkeypatch):
        """gate_proj and up_proj must share weight_scale_2 even when an expert
        was never routed during calibration.

        Regression for the bug where ``_export_fused_experts``'s per-projection
        fallback computed amax independently from the gate and up halves of the
        fused tensor — producing mismatched ``weight_scale_2`` values for any
        uncalibrated expert. vLLM fuses W1 (gate) and W3 (up) at load time and
        asserts a single shared scale; mismatched scales corrupted MoE output.
        The fix derives the fallback amax once from the fused ``gate_up[idx]``
        tensor before the deepcopies, so gate's clone and up's clone start with
        the same amax.
        """
        # Build experts where gate and up have very different magnitudes —
        # any per-half fallback would clearly produce different amaxes.
        experts = _SyntheticFusedExperts()
        gate = torch.randn(NUM_EXPERTS, INTERMEDIATE_DIM, HIDDEN_DIM) * 0.02
        up = torch.randn(NUM_EXPERTS, INTERMEDIATE_DIM, HIDDEN_DIM) * 0.20
        with torch.no_grad():
            experts.gate_up_proj.copy_(torch.cat([gate, up], dim=1))

        expert_type = type(experts)
        if QuantModuleRegistry.get(expert_type) is None:
            QuantModuleRegistry.register({expert_type: "test.SyntheticFusedExperts"})(
                _QuantFusedExperts
            )
        try:
            converted = QuantModuleRegistry.convert(experts)

            # Leave every expert weight quantizer uncalibrated (no _amax).
            # Mark them enabled to exercise the export-time fallback path.
            for q in converted.gate_up_proj_weight_quantizers:
                q._disabled = False
            for q in converted.down_proj_weight_quantizers:
                q._disabled = False

            # Capture the amax each per-projection wrapper carries into the
            # FP4 quantization step. Patching here avoids needing CUDA / FP4.
            seen = {}  # (expert_idx, proj_name) -> amax tensor

            def _spy_export(wrapper, dtype):
                # Identify which expert/projection this wrapper belongs to by
                # matching the weight tensor against the fused parameters.
                w = wrapper.weight.data
                # gate_up_proj is (N, 2*INTER, HIDDEN); split halves are
                # contiguous .data views or .contiguous() copies — we can match
                # by shape and value identity for this synthetic case.
                amax = wrapper.weight_quantizer._amax.detach().clone()
                # Identify by matching against gate vs. up slices of each expert.
                for idx in range(NUM_EXPERTS):
                    g_slice = converted.gate_up_proj.data[idx, :INTERMEDIATE_DIM, :]
                    u_slice = converted.gate_up_proj.data[idx, INTERMEDIATE_DIM:, :]
                    d_slice = converted.down_proj.data[idx]
                    if w.shape == g_slice.shape and torch.equal(w, g_slice):
                        seen[(idx, "gate_proj")] = amax
                        return
                    if w.shape == u_slice.shape and torch.equal(w, u_slice):
                        seen[(idx, "up_proj")] = amax
                        return
                    if w.shape == d_slice.shape and torch.equal(w, d_slice):
                        seen[(idx, "down_proj")] = amax
                        return

            monkeypatch.setattr(
                "modelopt.torch.export.unified_export_hf._export_quantized_weight",
                _spy_export,
            )

            _export_fused_experts(converted, torch.float16)

            # Assert: for every expert, gate's amax matches up's amax.
            for idx in range(NUM_EXPERTS):
                g_amax = seen.get((idx, "gate_proj"))
                u_amax = seen.get((idx, "up_proj"))
                assert g_amax is not None and u_amax is not None, (
                    f"Expert {idx}: missing recorded amax (gate={g_amax}, up={u_amax})"
                )
                assert torch.allclose(g_amax, u_amax), (
                    f"Expert {idx}: gate amax {g_amax.item()} != up amax {u_amax.item()}. "
                    f"Uncalibrated fused experts must share gate/up amax so that "
                    f"weight_scale_2 stays consistent across the fusion."
                )
        finally:
            if QuantModuleRegistry.get(expert_type) is not None:
                QuantModuleRegistry.unregister(expert_type)

    def test_per_block_amax_reshape_for_fused_export(self, monkeypatch):
        """Per-block ``_amax`` (NVFP4 static, row axis collapsed) must be reshaped
        before dim-0 slicing so gate's blocks and up's blocks are split correctly.

        Regression for the bug where a flat per-block ``_amax`` of shape
        ``(fused_total * blocks_per_row,)`` was sliced naively, producing wrong
        per-projection scales. The fix reshapes to ``(fused_total, blocks_per_row)``
        before slicing on dim-0 when ``amax.numel() % fused_total == 0``.
        """
        experts = _SyntheticFusedExperts()
        expert_type = type(experts)
        if QuantModuleRegistry.get(expert_type) is None:
            QuantModuleRegistry.register({expert_type: "test.SyntheticFusedExperts"})(
                _QuantFusedExperts
            )
        try:
            converted = QuantModuleRegistry.convert(experts)

            # Per-block amax: 4 blocks per row. Distinct values per row so we can
            # detect whether the reshape correctly preserves the row→block layout.
            blocks_per_row = 4
            fused_total = 2 * INTERMEDIATE_DIM  # gate_up rows
            for idx in range(NUM_EXPERTS):
                # Gate rows take values 1..INTERMEDIATE_DIM, up rows 101..101+INTERMEDIATE_DIM.
                gate_amax = (
                    torch.arange(1, INTERMEDIATE_DIM + 1).float().repeat_interleave(blocks_per_row)
                )
                up_amax = (
                    torch.arange(101, 101 + INTERMEDIATE_DIM)
                    .float()
                    .repeat_interleave(blocks_per_row)
                )
                # Flat shape (fused_total * blocks_per_row,) — row axis collapsed.
                flat = torch.cat([gate_amax, up_amax])
                assert flat.numel() == fused_total * blocks_per_row

                wq = converted.gate_up_proj_weight_quantizers[idx]
                wq._disabled = False
                wq.amax = flat

                # down_proj quantizers also need to look calibrated (otherwise
                # the export-time fallback would compute amax from each weight
                # slice and we'd skip the new reshape branch). Set a 1-D per-row
                # amax that matches dim-0 of down_proj (so amax.numel() == fused_total
                # for down). That intentionally does NOT exercise the new branch
                # for down — we only want to exercise it for gate_up.
                dwq = converted.down_proj_weight_quantizers[idx]
                dwq._disabled = False
                dwq.amax = torch.ones(HIDDEN_DIM)

            seen = {}

            def _spy_export(wrapper, dtype):
                w = wrapper.weight.data
                wq = wrapper.weight_quantizer
                amax = wq._amax.detach().clone() if hasattr(wq, "_amax") else None
                for idx in range(NUM_EXPERTS):
                    g_slice = converted.gate_up_proj.data[idx, :INTERMEDIATE_DIM, :]
                    u_slice = converted.gate_up_proj.data[idx, INTERMEDIATE_DIM:, :]
                    if w.shape == g_slice.shape and torch.equal(w, g_slice):
                        seen[(idx, "gate_proj")] = amax
                        return
                    if w.shape == u_slice.shape and torch.equal(w, u_slice):
                        seen[(idx, "up_proj")] = amax
                        return

            monkeypatch.setattr(
                "modelopt.torch.export.unified_export_hf._export_quantized_weight",
                _spy_export,
            )

            _export_fused_experts(converted, torch.float16)

            # gate's amax should contain values 1..INTERMEDIATE_DIM repeated
            # blocks_per_row times, reshaped to (INTERMEDIATE_DIM, blocks_per_row);
            # up's amax should contain 101..101+INTERMEDIATE_DIM same shape.
            for idx in range(NUM_EXPERTS):
                g_amax = seen.get((idx, "gate_proj"))
                u_amax = seen.get((idx, "up_proj"))
                assert g_amax is not None and u_amax is not None, (
                    f"Expert {idx}: missing recorded amax"
                )
                assert g_amax.shape[0] == INTERMEDIATE_DIM, (
                    f"Expert {idx} gate amax dim-0 should be {INTERMEDIATE_DIM} "
                    f"after reshape+slice, got {g_amax.shape}"
                )
                assert u_amax.shape[0] == INTERMEDIATE_DIM, (
                    f"Expert {idx} up amax dim-0 should be {INTERMEDIATE_DIM}, got {u_amax.shape}"
                )
                # First block of first row carries the marker value.
                assert g_amax.flatten()[0].item() == 1.0, (
                    f"Expert {idx} gate amax[0,0] should be 1.0 (gate row 0 marker), "
                    f"got {g_amax.flatten()[0].item()} — reshape probably didn't restore row axis"
                )
                assert u_amax.flatten()[0].item() == 101.0, (
                    f"Expert {idx} up amax[0,0] should be 101.0 (up row 0 marker), "
                    f"got {u_amax.flatten()[0].item()} — slice probably didn't separate gate from up"
                )
        finally:
            if QuantModuleRegistry.get(expert_type) is not None:
                QuantModuleRegistry.unregister(expert_type)


# ---------------------------------------------------------------------------
# Tests for normalize_experts_impl_on_the_fly
# ---------------------------------------------------------------------------
class _StubConfig:
    """Minimal stand-in for HF PretrainedConfig with optional nested sub-configs."""

    def __init__(self, impl=None, **nested):
        if impl is not None:
            self._experts_implementation = impl
        for key, value in nested.items():
            setattr(self, key, value)


class _TinyMoEModelWithConfig(_TinyMoEModel):
    def __init__(self, config):
        super().__init__()
        self.config = config


class _NonMoEModelWithConfig(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.config = config


class TestNormalizeExpertsImpl:
    def test_redirects_unsupported_backend_to_grouped_mm(self):
        """An unsupported backend on an MoE model gets redirected to grouped_mm."""
        cfg = _StubConfig(impl="kernels")
        model = _TinyMoEModelWithConfig(cfg)
        normalize_experts_impl_on_the_fly(model)
        assert cfg._experts_implementation == "grouped_mm"

    def test_recurses_into_nested_configs(self):
        """VLM-style nested text_config / vision_config are also normalized."""
        text_cfg = _StubConfig(impl="grouped_mm")  # already supported -> unchanged
        vision_cfg = _StubConfig(impl="batched_mm")  # unsupported -> grouped_mm
        root_cfg = _StubConfig(text_config=text_cfg, vision_config=vision_cfg)
        model = _TinyMoEModelWithConfig(root_cfg)
        normalize_experts_impl_on_the_fly(model)
        assert text_cfg._experts_implementation == "grouped_mm"
        assert vision_cfg._experts_implementation == "grouped_mm"

    def test_skips_model_without_fused_experts(self):
        """Non-MoE models must not have their config silently mutated."""
        cfg = _StubConfig(impl="kernels")
        model = _NonMoEModelWithConfig(cfg)
        normalize_experts_impl_on_the_fly(model)
        assert cfg._experts_implementation == "kernels"

    def test_no_crash_when_config_missing(self):
        """Model without a ``config`` attribute must not raise."""
        normalize_experts_impl_on_the_fly(_TinyMoEModel())  # no-op, no error

    def test_no_crash_when_impl_attr_missing(self):
        """Config without ``_experts_implementation`` must not raise."""
        cfg = _StubConfig()  # no impl attr
        model = _TinyMoEModelWithConfig(cfg)
        normalize_experts_impl_on_the_fly(model)
        assert not hasattr(cfg, "_experts_implementation")

    def test_leaves_eager_value_unchanged(self):
        cfg = _StubConfig(impl="eager")
        model = _TinyMoEModelWithConfig(cfg)
        normalize_experts_impl_on_the_fly(model)
        assert cfg._experts_implementation == "eager"

    def test_leaves_grouped_mm_value_unchanged(self):
        cfg = _StubConfig(impl="grouped_mm")
        model = _TinyMoEModelWithConfig(cfg)
        normalize_experts_impl_on_the_fly(model)
        assert cfg._experts_implementation == "grouped_mm"

    def test_leaves_none_value_unchanged(self):
        """Explicit None (model's eager F.linear forward) is supported -> untouched."""
        cfg = _StubConfig()
        cfg._experts_implementation = None
        model = _TinyMoEModelWithConfig(cfg)
        normalize_experts_impl_on_the_fly(model)
        assert cfg._experts_implementation is None


# ---------------------------------------------------------------------------
# End-to-end PTQ calibration test — guards the full fused-experts path:
#   register_fused_experts_on_the_fly → _QuantFusedExperts.{_setup, forward} →
#   plural ModuleList name normalization in conversion._match_quantizer →
#   TensorQuantizer amax collection via the F.linear hook.
# If any link breaks, quantizer `amax` stays None and this test fails.
# ---------------------------------------------------------------------------
class TestFusedExpertsCalibration:
    @staticmethod
    def _cleanup_registry(mod_type):
        if QuantModuleRegistry.get(mod_type) is not None:
            QuantModuleRegistry.unregister(mod_type)

    def test_calibration_populates_all_expert_quantizers(self):
        """After PTQ, every input/weight quantizer on the fused-experts module has amax set."""
        model = _TinyMoEModel()
        expert_type = type(model.moe.experts)
        self._cleanup_registry(expert_type)

        quant_cfg = {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {
                    "quantizer_name": "*gate_up_proj_input_quantizer",
                    "cfg": {"num_bits": 8, "axis": None},
                },
                {
                    "quantizer_name": "*down_proj_input_quantizer",
                    "cfg": {"num_bits": 8, "axis": None},
                },
                {
                    "quantizer_name": "*gate_up_proj_weight_quantizer",
                    "cfg": {"num_bits": 8, "axis": 0},
                },
                {
                    "quantizer_name": "*down_proj_weight_quantizer",
                    "cfg": {"num_bits": 8, "axis": 0},
                },
            ],
            "algorithm": "max",
        }

        def forward_loop(m):
            torch.manual_seed(0)
            for _ in range(2):
                x = torch.randn(1, 4, HIDDEN_DIM)
                m(x)

        mtq.quantize(model, quant_cfg, forward_loop=forward_loop)

        experts = model.moe.experts
        assert experts.gate_up_proj_input_quantizer.amax is not None, (
            "Shared gate_up_proj input quantizer was not calibrated — "
            "F.linear hook likely bypassed by non-eager experts_implementation."
        )
        assert experts.down_proj_input_quantizer.amax is not None, (
            "Shared down_proj input quantizer was not calibrated."
        )
        for idx in range(NUM_EXPERTS):
            assert experts.gate_up_proj_weight_quantizers[idx].amax is not None, (
                f"gate_up_proj_weight_quantizers[{idx}].amax is None — "
                "plural ModuleList name normalization in _match_quantizer likely broken."
            )
            assert experts.down_proj_weight_quantizers[idx].amax is not None, (
                f"down_proj_weight_quantizers[{idx}].amax is None."
            )

        self._cleanup_registry(expert_type)

    def test_local_hessian_refines_per_expert_weights(self):
        """local_hessian captures each expert's routed activations and refines its weight amax."""
        model = _TinyMoEModel()
        expert_type = type(model.moe.experts)
        self._cleanup_registry(expert_type)

        weight_quant = {"num_bits": 8, "axis": 0}
        quant_cfg = {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {"quantizer_name": "*gate_up_proj_weight_quantizer", "cfg": weight_quant},
                {"quantizer_name": "*down_proj_weight_quantizer", "cfg": weight_quant},
            ],
            "algorithm": "max",
        }

        def forward_loop(m):
            torch.manual_seed(0)
            for _ in range(3):
                m(torch.randn(1, 8, HIDDEN_DIM))

        mtq.quantize(model, quant_cfg, forward_loop=forward_loop)
        experts = model.moe.experts
        expert_quantizers = list(experts.gate_up_proj_weight_quantizers) + list(
            experts.down_proj_weight_quantizers
        )
        max_amax = {id(q): q.amax.clone() for q in expert_quantizers if q.amax is not None}
        # Expected (cout, cin) keyed by quantizer id, to verify each Hessian pairs with its
        # own expert's weight slice (catches gate_up/down swaps and stale-index mis-pairing).
        expected_shape = {}
        for quantizers, weight in (
            (experts.gate_up_proj_weight_quantizers, experts.gate_up_proj),
            (experts.down_proj_weight_quantizers, experts.down_proj),
        ):
            for i, q in enumerate(quantizers):
                expected_shape[id(q)] = (weight[i].shape[0], weight[i].shape[1])

        local_hessian_calibrate(model, forward_loop, fp8_scale_sweep=False, debug=True)

        # Each captured Hessian is keyed to a real per-expert quantizer with the matching weight
        # shape, spans multiple distinct experts, and the refinement moved at least one amax.
        routed = {qid: a for qid, a in model._local_hessian_accumulators.items() if a.num_samples}
        assert len(routed) >= 2, "expected multiple distinct experts to capture Hessians"
        for qid, acc in routed.items():
            assert (acc.cout, acc.cin) == expected_shape[qid]
        assert all(q.amax is not None and torch.isfinite(q.amax).all() for q in expert_quantizers)
        assert any(
            id(q) in max_amax and not torch.allclose(q.amax, max_amax[id(q)])
            for q in expert_quantizers
        )

        self._cleanup_registry(expert_type)

    def test_max_calibrate_populates_dead_static_nvfp4_expert_quantizers(self):
        """max calibration fills static NVFP4 ``_amax`` on experts the forward never routed to.

        Regression for the dead-expert MSE skip: with partial routing during max
        calibration, never-routed experts' weight quantizers stay with
        ``_amax=None`` unless static NVFP4 finalization bootstraps them from the
        per-expert weight slice.
        """
        model = _TinyMoEModel()
        expert_type = type(model.moe.experts)
        self._cleanup_registry(expert_type)

        quant_cfg = {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {
                    "quantizer_name": "*gate_up_proj_weight_quantizer",
                    "cfg": {
                        "num_bits": (2, 1),
                        "block_sizes": {-1: 16, "type": "static", "scale_bits": (4, 3)},
                    },
                },
                {
                    "quantizer_name": "*down_proj_weight_quantizer",
                    "cfg": {
                        "num_bits": (2, 1),
                        "block_sizes": {-1: 16, "type": "static", "scale_bits": (4, 3)},
                    },
                },
            ],
            "algorithm": "max",
        }

        # Forward loop that routes only to experts 0 and 1 (deterministic).
        # Bypasses the router and calls experts directly with crafted indices.
        live = {0, 1}
        dead = {idx for idx in range(NUM_EXPERTS) if idx not in live}
        assert dead, "Test requires at least one dead expert"

        def partial_forward(m):
            torch.manual_seed(0)
            seq_len = 8
            hidden = torch.randn(seq_len, HIDDEN_DIM)
            top_k_index = torch.zeros(seq_len, TOP_K, dtype=torch.long)
            top_k_index[:, 0] = 0
            top_k_index[:, 1] = 1
            top_k_weights = torch.ones(seq_len, TOP_K) / TOP_K
            with torch.no_grad():
                m.moe.experts(hidden, top_k_index, top_k_weights)

        mtq.quantize(model, quant_cfg, forward_loop=partial_forward)

        experts = model.moe.experts

        # Static NVFP4 finalization in max_calibrate bootstraps every expert.
        for idx in range(NUM_EXPERTS):
            gu_q = experts.gate_up_proj_weight_quantizers[idx]
            d_q = experts.down_proj_weight_quantizers[idx]
            assert gu_q._amax is not None and not torch.all(gu_q._amax == 0), (
                f"Expert {idx} gate_up_proj _amax not populated after max_calibrate"
            )
            assert d_q._amax is not None and not torch.all(d_q._amax == 0), (
                f"Expert {idx} down_proj _amax not populated after max_calibrate"
            )

        # For dead experts, bootstrap reads blockwise max(|weight|). Sanity-check it
        # matches the actual weight tensor's per-block max over hidden_dim.
        for idx in dead:
            expected = experts.gate_up_proj.data[idx].reshape(2 * INTERMEDIATE_DIM, 2, 16)
            expected = expected.abs().amax(dim=2).flatten()
            got = experts.gate_up_proj_weight_quantizers[idx]._amax.flatten()
            assert torch.allclose(got, expected, atol=1e-4), (
                f"Expert {idx} amax should equal blockwise max(|weight|); "
                f"max diff {(got - expected).abs().max().item()}"
            )

        self._cleanup_registry(expert_type)


# ---------------------------------------------------------------------------
# Tests for export enumeration — guards the bug where fused-experts were
# silently skipped by get_quant_config because their weight quantizers live
# on a plural nn.ModuleList instead of the singular *_weight_quantizer attr.
# Missed enumeration → experts don't appear in quantized_layers →
# quantization_formats has only 1 entry from the non-expert modules →
# quant_algo lands on that format instead of "MIXED_PRECISION".
# ---------------------------------------------------------------------------
class _MixedPrecisionModel(nn.Module):
    """A model with both a fused-experts block AND a standard Linear, so a
    mixed-precision recipe should produce two distinct format groups."""

    def __init__(self):
        super().__init__()
        self.moe = _SyntheticSparseMoeBlock()
        self.dense = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)

    def forward(self, x):
        return self.dense(self.moe(x))


class TestMixedPrecisionExport:
    @staticmethod
    def _cleanup_registry(mod_type):
        if QuantModuleRegistry.get(mod_type) is not None:
            QuantModuleRegistry.unregister(mod_type)

    def test_weight_attr_names_yields_fused_expert_params(self):
        """weight_attr_names must yield gate_up_proj / down_proj on fused experts
        even though their quantizers are a plural ModuleList, not singular."""
        model = _TinyMoEModel()
        expert_type = type(model.moe.experts)
        self._cleanup_registry(expert_type)

        register_fused_experts_on_the_fly(model)
        converted = QuantModuleRegistry.convert(model.moe.experts)

        yielded = list(weight_attr_names(converted))
        assert set(yielded) == {"gate_up_proj", "down_proj"}, (
            f"Expected both fused weight attrs, got {yielded}. "
            "Likely regression in representative_weight_quantizer plural fallback."
        )

        self._cleanup_registry(expert_type)

    def test_mixed_precision_config_export(self):
        """Mixed-precision recipe (experts FP8 + dense Linear FP8 per-channel) should
        show both modules in quantized_layers. Using two distinct formats would
        trigger MIXED_PRECISION; using same-format still exercises enumeration."""
        model = _MixedPrecisionModel()
        expert_type = type(model.moe.experts)
        self._cleanup_registry(expert_type)

        # FP8 per-tensor for experts; FP8 per-channel for dense — two distinct
        # format strings in quantization_formats, so quant_algo must become
        # MIXED_PRECISION.
        quant_cfg = {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {
                    "quantizer_name": "*gate_up_proj_input_quantizer",
                    "cfg": {"num_bits": (4, 3), "axis": None},
                },
                {
                    "quantizer_name": "*down_proj_input_quantizer",
                    "cfg": {"num_bits": (4, 3), "axis": None},
                },
                {
                    "quantizer_name": "*gate_up_proj_weight_quantizer",
                    "cfg": {"num_bits": (4, 3), "axis": None},
                },
                {
                    "quantizer_name": "*down_proj_weight_quantizer",
                    "cfg": {"num_bits": (4, 3), "axis": None},
                },
                {
                    "quantizer_name": "*dense.input_quantizer",
                    "cfg": {"num_bits": (4, 3), "axis": None},
                },
                {
                    "quantizer_name": "*dense.weight_quantizer",
                    "cfg": {"num_bits": (4, 3), "axis": 0},  # per-channel → FP8_PC_PT
                },
            ],
            "algorithm": "max",
        }

        def forward_loop(m):
            torch.manual_seed(0)
            for _ in range(2):
                x = torch.randn(1, 4, HIDDEN_DIM)
                m(x)

        mtq.quantize(model, quant_cfg, forward_loop=forward_loop)

        cfg = get_quant_config(model)
        q = cfg["quantization"]

        # The fused-experts module MUST appear in quantized_layers. This is the
        # central guard: regressions of weight_attr_names plural fallback would
        # make experts disappear here.
        layer_names = set(q.get("quantized_layers", {}).keys())
        assert any("moe.experts" in n for n in layer_names), (
            f"Fused-experts module missing from quantized_layers: {layer_names}. "
            "weight_attr_names likely not yielding plural-ModuleList weight attrs."
        )
        assert any(n.endswith("dense") for n in layer_names), (
            f"Dense Linear missing from quantized_layers: {layer_names}."
        )

        # Two distinct formats → MIXED_PRECISION at top level.
        assert q["quant_algo"] == "MIXED_PRECISION", (
            f"Expected MIXED_PRECISION (fused-experts FP8 per-tensor + dense "
            f"FP8 per-channel), got quant_algo={q['quant_algo']}. "
            f"quantized_layers={q.get('quantized_layers')}"
        )

        self._cleanup_registry(expert_type)


# ---------------------------------------------------------------------------
# Tests for the fused-experts quantizer-name normalizer used by
# conversion._match_quantizer. Covers both plural (actual _QuantFusedExperts
# layout) and singular (defensive: future variants may name the ModuleList
# without the trailing `s`) forms.
# ---------------------------------------------------------------------------
class TestNormalizeFusedExpertsQuantizerName:
    def test_plural_weight_quantizers_stripped(self):
        assert (
            _normalize_fused_experts_quantizer_name("moe.experts.gate_up_proj_weight_quantizers.7")
            == "moe.experts.gate_up_proj_weight_quantizer"
        )

    def test_plural_input_quantizers_stripped(self):
        assert (
            _normalize_fused_experts_quantizer_name("moe.experts.down_proj_input_quantizers.3")
            == "moe.experts.down_proj_input_quantizer"
        )

    def test_singular_weight_quantizer_with_index_stripped(self):
        """Defensive: handle variants that name the ModuleList singular."""
        assert (
            _normalize_fused_experts_quantizer_name("moe.experts.gate_up_proj_weight_quantizer.2")
            == "moe.experts.gate_up_proj_weight_quantizer"
        )

    def test_singular_input_quantizer_with_index_stripped(self):
        assert (
            _normalize_fused_experts_quantizer_name("moe.experts.down_proj_input_quantizer.0")
            == "moe.experts.down_proj_input_quantizer"
        )

    def test_non_indexed_name_unchanged(self):
        """Plain singular names (no index) must be passed through untouched."""
        assert (
            _normalize_fused_experts_quantizer_name("moe.experts.gate_up_proj_weight_quantizer")
            == "moe.experts.gate_up_proj_weight_quantizer"
        )

    def test_unrelated_dotted_number_unchanged(self):
        """Dotted numbers that aren't inside a quantizer-list context are left alone."""
        assert (
            _normalize_fused_experts_quantizer_name("moe.layers.3.gate.weight")
            == "moe.layers.3.gate.weight"
        )


# ---------------------------------------------------------------------------
# Tests for the grouped_mm backend: _QuantFusedExperts intercepts the real
# transformers grouped_mm forward (via _grouped_linear), and grouped calibration
# must produce the same per-expert quantizer state as the eager F.linear path.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not HAS_GROUPED_MM, reason="transformers grouped_mm backend unavailable")
class TestGroupedMMExperts:
    @staticmethod
    def _cleanup_registry(mod_type):
        if QuantModuleRegistry.get(mod_type) is not None:
            QuantModuleRegistry.unregister(mod_type)

    @staticmethod
    def _quant_cfg():
        return {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {
                    "quantizer_name": "*gate_up_proj_input_quantizer",
                    "cfg": {"num_bits": 8, "axis": None},
                },
                {
                    "quantizer_name": "*down_proj_input_quantizer",
                    "cfg": {"num_bits": 8, "axis": None},
                },
                {
                    "quantizer_name": "*gate_up_proj_weight_quantizer",
                    "cfg": {"num_bits": 8, "axis": 0},
                },
                {
                    "quantizer_name": "*down_proj_weight_quantizer",
                    "cfg": {"num_bits": 8, "axis": 0},
                },
            ],
            "algorithm": "max",
        }

    @staticmethod
    def _forward_loop_all_experts(m):
        # Route every token to all experts (round-robin) so eager and grouped both
        # calibrate every expert's weight quantizer — eager only sees routed experts.
        torch.manual_seed(0)
        seq = 16
        hidden = torch.randn(seq, HIDDEN_DIM)
        top_k_index = torch.stack(
            [torch.arange(seq) % NUM_EXPERTS, (torch.arange(seq) + 1) % NUM_EXPERTS], dim=1
        )
        top_k_weights = torch.softmax(torch.randn(seq, TOP_K), dim=1)
        with torch.no_grad():
            m.moe.experts(hidden, top_k_index, top_k_weights)

    def test_intercept_fires_twice_and_uses_full_3d_weight(self):
        """The grouped interceptor runs once per projection with the full 3-D weight."""
        self._cleanup_registry(_SyntheticGroupedExperts)
        model = _GroupedMoEModel("grouped_mm")
        register_fused_experts_on_the_fly(model)
        converted = QuantModuleRegistry.convert(model.moe.experts)
        assert converted._running_grouped_mm is True

        shapes = []
        orig = _moe_integration._grouped_linear

        def _spy(input, weight, offs, bias=None, is_transposed=False):
            shapes.append(tuple(weight.shape))
            return orig(input, weight, offs, bias=bias, is_transposed=is_transposed)

        _moe_integration._grouped_linear = _spy
        try:
            seq = 8
            hidden = torch.randn(seq, HIDDEN_DIM)
            top_k_index = torch.randint(0, NUM_EXPERTS, (seq, TOP_K))
            top_k_weights = torch.softmax(torch.randn(seq, TOP_K), dim=1)
            with torch.no_grad():
                converted(hidden, top_k_index, top_k_weights)
        finally:
            _moe_integration._grouped_linear = orig

        assert shapes == [
            (NUM_EXPERTS, 2 * INTERMEDIATE_DIM, HIDDEN_DIM),  # gate_up
            (NUM_EXPERTS, HIDDEN_DIM, INTERMEDIATE_DIM),  # down
        ]
        self._cleanup_registry(_SyntheticGroupedExperts)

    def test_forward_passthrough_matches(self):
        """Converted grouped experts (quantizers disabled) match the unconverted forward."""
        self._cleanup_registry(_SyntheticGroupedExperts)
        torch.manual_seed(0)
        model = _GroupedMoEModel("grouped_mm")
        ref = _SyntheticGroupedExperts(_ExpertsConfig("grouped_mm"))
        ref.load_state_dict(model.moe.experts.state_dict())

        register_fused_experts_on_the_fly(model)
        converted = QuantModuleRegistry.convert(model.moe.experts)

        seq = 8
        hidden = torch.randn(seq, HIDDEN_DIM)
        top_k_index = torch.randint(0, NUM_EXPERTS, (seq, TOP_K))
        top_k_weights = torch.softmax(torch.randn(seq, TOP_K), dim=1)
        with torch.no_grad():
            out_ref = ref(hidden, top_k_index, top_k_weights)
            out_test = converted(hidden, top_k_index, top_k_weights)
        assert torch.allclose(out_ref, out_test, atol=1e-4), (
            f"Max diff: {(out_ref - out_test).abs().max().item()}"
        )
        self._cleanup_registry(_SyntheticGroupedExperts)

    def test_calibration_matches_eager(self):
        """grouped_mm PTQ calibration yields the same quantizer state as eager PTQ.

        Identical weights + identical routing ⇒ the shared input quantizers see the
        same activation set and each per-expert weight quantizer sees the same slice,
        so amax must match (max abs is order-free; only the down-proj input flows
        through a GEMM, hence the small tolerance there).
        """
        self._cleanup_registry(_SyntheticGroupedExperts)
        torch.manual_seed(0)
        eager_model = _GroupedMoEModel("eager")
        grouped_model = _GroupedMoEModel("grouped_mm")
        grouped_model.load_state_dict(eager_model.state_dict())

        mtq.quantize(eager_model, self._quant_cfg(), forward_loop=self._forward_loop_all_experts)
        mtq.quantize(grouped_model, self._quant_cfg(), forward_loop=self._forward_loop_all_experts)

        e = eager_model.moe.experts
        g = grouped_model.moe.experts

        assert g._running_grouped_mm is True
        assert e._running_grouped_mm is False

        # Shared input quantizers (down flows through a GEMM -> small tolerance).
        for q_name in ("gate_up_proj_input_quantizer", "down_proj_input_quantizer"):
            ea = getattr(e, q_name).amax
            ga = getattr(g, q_name).amax
            assert ea is not None and ga is not None, f"{q_name} not calibrated"
            assert torch.allclose(ea, ga, atol=1e-3, rtol=1e-2), (
                f"{q_name} amax mismatch: eager={ea} grouped={ga}"
            )

        # Per-expert weight amax (weights identical -> tight match), all experts calibrated.
        for idx in range(NUM_EXPERTS):
            for q_list in ("gate_up_proj_weight_quantizers", "down_proj_weight_quantizers"):
                eq = getattr(e, q_list)[idx].amax
                gq = getattr(g, q_list)[idx].amax
                assert eq is not None and gq is not None, f"{q_list}[{idx}] not calibrated"
                assert torch.allclose(eq, gq, atol=1e-5), f"{q_list}[{idx}] weight amax mismatch"
        self._cleanup_registry(_SyntheticGroupedExperts)

    def test_local_hessian_captures_per_expert_in_grouped_mode(self):
        """local_hessian splits the expert-sorted grouped batch by offsets and refines
        each expert's weight amax — the grouped analogue of the eager per-expert hook."""
        self._cleanup_registry(_SyntheticGroupedExperts)
        model = _GroupedMoEModel("grouped_mm")

        weight_quant = {"num_bits": 8, "axis": 0}
        quant_cfg = {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {"quantizer_name": "*gate_up_proj_weight_quantizer", "cfg": weight_quant},
                {"quantizer_name": "*down_proj_weight_quantizer", "cfg": weight_quant},
            ],
            "algorithm": "max",
        }

        def forward_loop(m):
            torch.manual_seed(0)
            for _ in range(3):
                m(torch.randn(1, 8, HIDDEN_DIM))

        mtq.quantize(model, quant_cfg, forward_loop=forward_loop)
        experts = model.moe.experts
        assert experts._running_grouped_mm is True

        expert_quantizers = list(experts.gate_up_proj_weight_quantizers) + list(
            experts.down_proj_weight_quantizers
        )
        max_amax = {id(q): q.amax.clone() for q in expert_quantizers if q.amax is not None}
        expected_shape = {}
        for quantizers, weight in (
            (experts.gate_up_proj_weight_quantizers, experts.gate_up_proj),
            (experts.down_proj_weight_quantizers, experts.down_proj),
        ):
            for i, q in enumerate(quantizers):
                expected_shape[id(q)] = (weight[i].shape[0], weight[i].shape[1])

        local_hessian_calibrate(model, forward_loop, fp8_scale_sweep=False, debug=True)

        routed = {qid: a for qid, a in model._local_hessian_accumulators.items() if a.num_samples}
        assert len(routed) >= 2, "expected multiple distinct experts to capture Hessians"
        for qid, acc in routed.items():
            assert (acc.cout, acc.cin) == expected_shape[qid]
        assert all(q.amax is not None and torch.isfinite(q.amax).all() for q in expert_quantizers)
        assert any(
            id(q) in max_amax and not torch.allclose(q.amax, max_amax[id(q)])
            for q in expert_quantizers
        )
        self._cleanup_registry(_SyntheticGroupedExperts)
