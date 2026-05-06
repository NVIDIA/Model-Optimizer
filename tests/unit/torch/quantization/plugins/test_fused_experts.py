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

from modelopt.torch.quantization.conversion import _normalize_fused_experts_quantizer_name
from modelopt.torch.quantization.nn import QuantModuleRegistry
from modelopt.torch.quantization.plugins.huggingface import (
    _is_fused_experts_module,
    _is_sparse_sequaential_moe_block,
    _QuantFusedExperts,
    force_eager_experts_impl_on_the_fly,
    register_fused_experts_on_the_fly,
    register_sparse_moe_on_the_fly,
)

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
    def test_export_creates_per_expert_submodules(self):
        """_export_fused_experts should create per-expert submodules with standard naming."""
        from modelopt.torch.export.moe_utils import _export_fused_experts

        experts = _SyntheticFusedExperts()
        expert_type = type(experts)

        # Manually register and convert
        if QuantModuleRegistry.get(expert_type) is None:
            QuantModuleRegistry.register({expert_type: "test.SyntheticFusedExperts"})(
                _QuantFusedExperts
            )
        converted = QuantModuleRegistry.convert(experts)

        # Run a forward pass to calibrate (set amaxes)
        seq_len = 16
        hidden_states = torch.randn(seq_len, HIDDEN_DIM)
        top_k_index = torch.randint(0, NUM_EXPERTS, (seq_len, TOP_K))
        top_k_weights = torch.softmax(torch.randn(seq_len, TOP_K), dim=-1)
        with torch.no_grad():
            converted(hidden_states, top_k_index, top_k_weights)

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

        if QuantModuleRegistry.get(expert_type) is not None:
            QuantModuleRegistry.unregister(expert_type)


# ---------------------------------------------------------------------------
# Tests for force_eager_experts_impl_on_the_fly
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


class TestForceEagerExpertsImpl:
    def test_sets_eager_on_moe_model(self):
        """Non-eager backend on an MoE model gets flipped to eager."""
        cfg = _StubConfig(impl="kernels")
        model = _TinyMoEModelWithConfig(cfg)
        force_eager_experts_impl_on_the_fly(model)
        assert cfg._experts_implementation == "eager"

    def test_recurses_into_nested_configs(self):
        """VLM-style nested text_config / vision_config are also flipped."""
        text_cfg = _StubConfig(impl="grouped_mm")
        vision_cfg = _StubConfig(impl="bmm")
        root_cfg = _StubConfig(text_config=text_cfg, vision_config=vision_cfg)
        model = _TinyMoEModelWithConfig(root_cfg)
        force_eager_experts_impl_on_the_fly(model)
        assert text_cfg._experts_implementation == "eager"
        assert vision_cfg._experts_implementation == "eager"

    def test_skips_model_without_fused_experts(self):
        """Non-MoE models must not have their config silently mutated."""
        cfg = _StubConfig(impl="kernels")
        model = _NonMoEModelWithConfig(cfg)
        force_eager_experts_impl_on_the_fly(model)
        assert cfg._experts_implementation == "kernels"

    def test_no_crash_when_config_missing(self):
        """Model without a ``config`` attribute must not raise."""
        force_eager_experts_impl_on_the_fly(_TinyMoEModel())  # no-op, no error

    def test_no_crash_when_impl_attr_missing(self):
        """Config without ``_experts_implementation`` must not raise."""
        cfg = _StubConfig()  # no impl attr
        model = _TinyMoEModelWithConfig(cfg)
        force_eager_experts_impl_on_the_fly(model)
        assert not hasattr(cfg, "_experts_implementation")

    def test_leaves_eager_value_unchanged(self):
        cfg = _StubConfig(impl="eager")
        model = _TinyMoEModelWithConfig(cfg)
        force_eager_experts_impl_on_the_fly(model)
        assert cfg._experts_implementation == "eager"


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
        import modelopt.torch.quantization as mtq

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
        from modelopt.torch.quantization.utils.core_utils import weight_attr_names

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
        import modelopt.torch.quantization as mtq
        from modelopt.torch.export.quant_utils import get_quant_config

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
