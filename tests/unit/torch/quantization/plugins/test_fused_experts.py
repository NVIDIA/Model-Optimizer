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

"""Tests for _QuantFusedExperts: generic fused MoE quantization and export."""

import copy
import warnings
from contextlib import nullcontext
from unittest import mock

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor, Shard, distribute_tensor
from torch.distributed.tensor.debug import CommDebugMode

pytest.importorskip("transformers")

from _test_utils.torch.quantization.tied_modules import tie_fused_experts_3d_params
from transformers import PretrainedConfig, Qwen3MoeConfig
from transformers.integrations.finegrained_fp8 import FP8Experts
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts

import modelopt.torch.quantization as mtq
import modelopt.torch.quantization.nn.modules.tensor_quantizer as tensor_quantizer_module
from modelopt.torch.export.model_utils import _release_exported_tensors
from modelopt.torch.export.moe_utils import _export_fused_experts, _fallback_amax
from modelopt.torch.export.quant_utils import get_quant_config, get_quantization_format
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.conversion import _normalize_fused_experts_quantizer_name
from modelopt.torch.quantization.model_calib import (
    _register_weight_input_hooks,
    local_hessian_calibrate,
)
from modelopt.torch.quantization.nn import QuantModuleRegistry, SequentialQuantizer, TensorQuantizer
from modelopt.torch.quantization.plugins.huggingface import (
    _fused_experts_wrapper,
    _is_fused_experts_module,
    _is_sparse_sequaential_moe_block,
    _QuantFusedExperts,
    _QuantNemotronHExperts,
    normalize_experts_impl_on_the_fly,
    register_fused_experts_on_the_fly,
    register_sparse_moe_on_the_fly,
)
from modelopt.torch.quantization.utils.calib_utils import GPTQHelper
from modelopt.torch.quantization.utils.core_utils import module_name_maps, weight_attr_names
from modelopt.torch.utils.moe import torch_a2a_experts_forward


def test_native_fp8_experts_dequantized_before_nvfp4_export():
    """Calibrate and pack scaled FP8 gate/up and down weights, preserving their BF16 values."""
    generator = torch.Generator().manual_seed(42)
    experts = FP8Experts(
        PretrainedConfig(hidden_size=256, intermediate_size=256, num_experts=2, hidden_act="silu"),
        block_size=(128, 128),
    )
    reference = {}
    for name, scale_range in (("gate_up_proj", (0.01, 0.08)), ("down_proj", (0.15, 0.3))):
        weight = getattr(experts, name)
        scales = getattr(experts, f"{name}_scale_inv")
        assert weight.dtype == torch.float8_e4m3fn
        with torch.no_grad():
            weight.copy_(torch.randn(weight.shape, generator=generator).to(weight.dtype))
            scales.copy_(torch.linspace(*scale_range, scales.numel()).reshape(scales.shape))
        # Build only the reference here; ModelOpt must consume the untouched FP8 source.
        blocks = weight.float().reshape(2, weight.shape[1] // 128, 128, 2, 128)
        reference[name] = (blocks * scales[:, :, None, :, None]).reshape(weight.shape).bfloat16()

    registered_before = QuantModuleRegistry.get(FP8Experts)
    try:
        mtq.quantize(experts, mtq.NVFP4_DEFAULT_CFG)
        for name, expected in reference.items():
            assert getattr(experts, name).dtype == torch.float8_e4m3fn
            quantizers = getattr(experts, f"{name}_weight_quantizers")
            for idx, quantizer in enumerate(quantizers):
                assert quantizer.is_enabled and quantizer.num_bits == (2, 1)
                torch.testing.assert_close(
                    quantizer.amax.float(),
                    expected[idx].float().abs().amax(),
                    rtol=0.004,
                    atol=0,
                )

        _export_fused_experts(experts, torch.bfloat16)
        levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])
        for idx in range(2):
            exported = getattr(experts, str(idx))
            torch.testing.assert_close(
                exported.gate_proj.weight_scale_2, exported.up_proj.weight_scale_2
            )
            for name, expected in (
                ("gate_proj", reference["gate_up_proj"][idx, :256]),
                ("up_proj", reference["gate_up_proj"][idx, 256:]),
                ("down_proj", reference["down_proj"][idx]),
            ):
                projection = getattr(exported, name)
                packed = projection.weight
                assert packed.dtype == torch.uint8
                codes = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(expected.shape)
                decoded = (
                    levels[codes.long()]
                    * projection.weight_scale.float().repeat_interleave(16, dim=-1)
                    * projection.weight_scale_2.float()
                )
                relative_l2 = torch.linalg.vector_norm(
                    decoded - expected.float()
                ) / torch.linalg.vector_norm(expected.float())
                assert relative_l2 < 0.15, f"Expert {idx} {name}: relative L2 {relative_l2}"
    finally:
        if registered_before is None and QuantModuleRegistry.get(FP8Experts) is not None:
            QuantModuleRegistry.unregister(FP8Experts)


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


class _SyntheticNonGatedFusedExperts(nn.Module):
    """Mimics NemotronHExperts (transformers 5.5+): non-gated fused experts.

    A single ``up_proj`` (no gate half) + ``down_proj``, both 3-D ``nn.Parameter`` s,
    with the forward calling ``F.linear`` exactly twice per expert (up then down).
    """

    def __init__(self):
        super().__init__()
        self.num_experts = NUM_EXPERTS
        self.hidden_dim = HIDDEN_DIM
        self.intermediate_dim = INTERMEDIATE_DIM
        self.up_proj = nn.Parameter(torch.randn(NUM_EXPERTS, INTERMEDIATE_DIM, HIDDEN_DIM) * 0.02)
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
            current_hidden_states = F.linear(current_state, self.up_proj[expert_idx])
            current_hidden_states = self.act_fn(current_hidden_states)
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


class _SyntheticNonGatedSparseMoeBlock(nn.Module):
    """Mimics NemotronHMoE: a router + non-gated fused experts."""

    def __init__(self):
        super().__init__()
        self.gate = _SyntheticTopKRouter()
        self.experts = _SyntheticNonGatedFusedExperts()

    def forward(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        _, top_k_weights, top_k_index = self.gate(hidden_states)
        hidden_states = self.experts(hidden_states, top_k_index, top_k_weights)
        return hidden_states.reshape(batch_size, sequence_length, hidden_dim)


class _TinyNonGatedMoEModel(nn.Module):
    """Minimal model containing a single non-gated MoE block."""

    def __init__(self):
        super().__init__()
        self.moe = _SyntheticNonGatedSparseMoeBlock()

    def forward(self, x):
        return self.moe(x)


def _route_once_to_each_expert(model):
    """Call fused experts directly with deterministic routing that covers every expert."""
    assert NUM_EXPERTS % TOP_K == 0
    seq_len = NUM_EXPERTS // TOP_K
    hidden_states = torch.randn(seq_len, HIDDEN_DIM)
    top_k_index = torch.arange(NUM_EXPERTS, dtype=torch.long).reshape(seq_len, TOP_K)
    top_k_weights = torch.ones(seq_len, TOP_K) / TOP_K
    model.moe.experts(hidden_states, top_k_index, top_k_weights)


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
        self.ep_group = None  # the expert-parallel group a torch_a2a forward dispatches over


class _SyntheticGroupedExperts(nn.Module):
    """Fused experts dispatching to eager or the real grouped_mm forward.

    Mirrors the HF transformers 5.x fused-experts contract
    (``has_gate``/``has_bias``/``is_transposed``/``_apply_gate`` + 3-D
    ``gate_up_proj``/``down_proj``) that ``grouped_mm_experts_forward`` expects. ``gated=False``
    builds NemotronHExperts' ungated layout instead, ``down_proj(act_fn(up_proj(x)))`` with a single
    ``up_proj``, which ``grouped_mm_experts_forward`` takes through its ``has_gate`` branch and
    ``_QuantNemotronHExperts`` quantizes.
    """

    def __init__(self, config, gated=True):
        super().__init__()
        self.config = config
        self.num_experts = NUM_EXPERTS
        self.hidden_dim = HIDDEN_DIM
        self.intermediate_dim = INTERMEDIATE_DIM
        self.has_gate = gated
        self.has_bias = False
        self.is_transposed = False
        self.is_concatenated = True
        # What transformers' @use_experts_implementation sets (5.18+ reads it in grouped_mm).
        self._is_expert_parallel = False
        first = torch.randn(NUM_EXPERTS, (2 if gated else 1) * INTERMEDIATE_DIM, HIDDEN_DIM) * 0.02
        setattr(self, "gate_up_proj" if gated else "up_proj", nn.Parameter(first))
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
            if self.has_gate:
                gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
                current_hidden_states = self.act_fn(gate) * up
            else:
                current_hidden_states = self.act_fn(
                    F.linear(current_state, self.up_proj[expert_idx])
                )
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
        if self.config._experts_implementation == "torch_a2a":
            return torch_a2a_experts_forward(
                self, hidden_states, top_k_index, top_k_weights, self.config.ep_group
            )
        return self._eager_forward(hidden_states, top_k_index, top_k_weights)


class _GroupedSparseMoeBlock(nn.Module):
    def __init__(self, config, gated=True):
        super().__init__()
        self.gate = _SyntheticTopKRouter()
        self.experts = _SyntheticGroupedExperts(config, gated=gated)

    def forward(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        _, top_k_weights, top_k_index = self.gate(hidden_states)
        hidden_states = self.experts(hidden_states, top_k_index, top_k_weights)
        return hidden_states.reshape(batch_size, sequence_length, hidden_dim)


class _GroupedMoEModel(nn.Module):
    """Minimal MoE model whose experts run the selected backend."""

    def __init__(self, impl, gated=True):
        super().__init__()
        self.config = _ExpertsConfig(impl=impl)
        self.moe = _GroupedSparseMoeBlock(self.config, gated=gated)

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

    def test_module_missing_act_fn_still_detected(self):
        module = nn.Module()
        module.gate_up_proj = nn.Parameter(torch.randn(4, 16, 8))
        module.down_proj = nn.Parameter(torch.randn(4, 8, 16))
        module.num_experts = 4
        assert _is_fused_experts_module(module) is True

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

    def test_transposed_fused_experts_are_left_unquantized(self):
        """Matched by shape, a transposed container would export with gate/up split along its input
        dim; registration leaves it unquantized, with a warning, rather than let export write wrong
        weights or abort the rest of the model's quantization."""

        class _TransposedExperts(_SyntheticFusedExperts):
            is_transposed = True

        model = nn.Module()
        model.experts = _TransposedExperts()

        with pytest.warns(UserWarning, match="store their weights transposed"):
            register_fused_experts_on_the_fly(model)
        assert QuantModuleRegistry.get(_TransposedExperts) is None

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
            recovered_idx = converted._get_expert_idx_from_first_proj(weight_slice)
            assert recovered_idx == idx, f"Expected {idx}, got {recovered_idx}"
        self._cleanup_registry(expert_type)

    def _make_rotated_fused_experts(self, monkeypatch):
        monkeypatch.setattr(
            tensor_quantizer_module,
            "normalized_hadamard_transform",
            lambda inputs, rotate_fp32=False, block_size=None: inputs,
        )
        model = _TinyMoEModel()
        expert_type = type(model.moe.experts)
        self._cleanup_registry(expert_type)
        register_fused_experts_on_the_fly(model)
        converted = QuantModuleRegistry.convert(model.moe.experts)
        expert_quantizers = list(converted.gate_up_proj_weight_quantizers) + list(
            converted.down_proj_weight_quantizers
        )
        for q in expert_quantizers:
            q.set_from_attribute_config(
                QuantizerAttributeConfig(num_bits=8, rotate={"enable": True})
            )
            q.amax = torch.tensor(6.0)
        return converted, expert_quantizers, expert_type

    def test_fold_weight_disables_per_expert_quantizers_and_rotation(self, monkeypatch):
        converted, expert_quantizers, expert_type = self._make_rotated_fused_experts(monkeypatch)
        try:
            converted.fold_weight()
            for q in expert_quantizers:
                assert not q.is_enabled
                assert not q.rotate_is_enabled
                assert not hasattr(q, "_amax")
        finally:
            self._cleanup_registry(expert_type)

    def test_fold_weight_keep_attrs_keeps_amax_disables_rotation(self, monkeypatch):
        converted, expert_quantizers, expert_type = self._make_rotated_fused_experts(monkeypatch)
        try:
            converted.fold_weight(keep_attrs=True)
            for q in expert_quantizers:
                assert not q.is_enabled
                assert not q.rotate_is_enabled
                assert hasattr(q, "_amax")
        finally:
            self._cleanup_registry(expert_type)


# ---------------------------------------------------------------------------
# Tests for export
# ---------------------------------------------------------------------------
class TestIterWeightQuantizersForCalibration:
    """The quantizer-only iterator must agree with the weight iterator and never index a weight.

    Indexing the fused 3-D weight is what dispatches a redistribute collective per expert under
    FSDP2, so callers that only read quantizer state go through the quantizer-only path.
    """

    @staticmethod
    def _convert(model):
        expert_type = type(model.moe.experts)
        TestQuantFusedExperts._cleanup_registry(expert_type)
        register_fused_experts_on_the_fly(model)
        converted = QuantModuleRegistry.convert(model.moe.experts)
        TestQuantFusedExperts._cleanup_registry(expert_type)
        return converted

    @pytest.mark.parametrize(
        "model_cls", [_TinyMoEModel, _TinyNonGatedMoEModel], ids=["gated", "non_gated"]
    )
    def test_yields_the_same_quantizers_in_the_same_order(self, model_cls):
        experts = self._convert(model_cls())

        from_weights = [q for _, q in experts.iter_weights_for_calibration()]
        quantizers_only = list(experts.iter_weight_quantizers_for_calibration())

        assert quantizers_only, "expected per-expert weight quantizers"
        assert [id(q) for q in quantizers_only] == [id(q) for q in from_weights]

    @pytest.mark.parametrize(
        "model_cls", [_TinyMoEModel, _TinyNonGatedMoEModel], ids=["gated", "non_gated"]
    )
    def test_does_not_index_the_fused_weight(self, model_cls):
        """The point of the override: no ``weight[idx]``, which is the per-expert collective."""
        experts = self._convert(model_cls())

        class _NoIndexing(torch.Tensor):
            @staticmethod
            def __new__(cls, data):
                return torch.Tensor._make_subclass(cls, data, False)

            def __getitem__(self, item):
                raise AssertionError("iter_weight_quantizers_for_calibration indexed the weight")

        for name in (experts._first_proj_attr, "down_proj"):
            weight = getattr(experts, name)
            setattr(experts, name, nn.Parameter(_NoIndexing(weight.data), requires_grad=False))

        assert list(experts.iter_weight_quantizers_for_calibration())
        # The weight iterator is the expensive one; it must still slice, or the guard above is
        # not actually testing anything.
        with pytest.raises(AssertionError, match="indexed the weight"):
            list(experts.iter_weights_for_calibration())


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
            _route_once_to_each_expert(m)

        mtq.quantize(model, quant_cfg, forward_loop=forward_loop)
        converted = model.moe.experts

        with _release_exported_tensors(converted):
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

        # Leaving the block releases the holders; nothing else can free them.
        for idx in range(NUM_EXPERTS):
            assert not hasattr(converted, str(idx)), f"Expert submodule {idx} survived release"

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

            def _spy_export(wrapper, dtype, **_kwargs):
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

            def _spy_export(wrapper, dtype, **_kwargs):
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

    @pytest.mark.parametrize("amax_state", ["calibrated", "zero", "missing"])
    def test_w4a8_sequential_weight_quantizer_exports_split_block_amax(
        self, monkeypatch, amax_state
    ):
        """W4A8 AWQ weights use a SequentialQuantizer: an INT4 block stage (flat per-block
        ``_amax`` of shape ``(rows * blocks, 1)``, as block-quant calibration records it)
        followed by a per-tensor FP8 stage. Export must treat each stage on its own: slice the
        block amax into the gate/up halves (updating the stale ``_amax_shape_for_export``), and
        for experts that saw no calibration tokens derive a per-block fallback instead of
        forcing a scalar into the block stage (``Changing shape when setting amax``).
        """
        block = 16
        experts = _SyntheticFusedExperts()
        expert_type = type(experts)
        if QuantModuleRegistry.get(expert_type) is None:
            QuantModuleRegistry.register({expert_type: "test.SyntheticFusedExperts"})(
                _QuantFusedExperts
            )
        try:
            converted = QuantModuleRegistry.convert(experts)

            def _w4a8_quantizer(rows, fused):
                int4 = TensorQuantizer(
                    QuantizerAttributeConfig(num_bits=4, block_sizes={-1: block, "type": "static"})
                )
                fp8 = TensorQuantizer(QuantizerAttributeConfig(num_bits=(4, 3)))
                # gate_up_proj reduces over HIDDEN_DIM, down_proj over INTERMEDIATE_DIM.
                blocks = (HIDDEN_DIM if fused else INTERMEDIATE_DIM) // block
                int4._amax_shape_for_export = (rows, -1)
                if amax_state != "missing":
                    # Row r of gate_up carries the marker r + 1, so a wrong slice is visible.
                    marker = torch.arange(1, rows + 1).float().repeat_interleave(blocks)[:, None]
                    int4.amax = (
                        marker.clone() if amax_state == "calibrated" else torch.zeros_like(marker)
                    )
                    fp8.amax = torch.tensor(2.0 if amax_state == "calibrated" else 0.0)
                return SequentialQuantizer(int4, fp8)

            for idx in range(NUM_EXPERTS):
                converted.gate_up_proj_weight_quantizers[idx] = _w4a8_quantizer(
                    2 * INTERMEDIATE_DIM, fused=True
                )
                converted.down_proj_weight_quantizers[idx] = _w4a8_quantizer(
                    HIDDEN_DIM, fused=False
                )

            seen = []

            def _spy_export(wrapper, dtype, **_kwargs):
                int4, fp8 = wrapper.weight_quantizer
                seen.append(
                    (
                        wrapper.weight.shape[0],
                        int4.export_amax().detach().clone(),
                        fp8._amax.detach().clone(),
                    )
                )

            monkeypatch.setattr(
                "modelopt.torch.export.unified_export_hf._export_quantized_weight",
                _spy_export,
            )

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                _export_fused_experts(converted, torch.float16)
            fallback_warned = any("was not calibrated" in str(w.message) for w in caught)
            assert fallback_warned == (amax_state != "calibrated")

            assert len(seen) == 3 * NUM_EXPERTS
            for rows, int4_amax, fp8_amax in seen:
                assert int4_amax.shape[0] == rows  # sliced to the projection, not the fused rows
                assert torch.all(int4_amax > 0)
                assert torch.all(fp8_amax > 0)
            # Export order per expert is gate, up, down; gate and up must share the FP8 stage.
            for gate, up in zip(seen[0::3], seen[1::3]):
                assert torch.equal(gate[2], up[2])
            if amax_state == "calibrated":
                gate_up_rows = {int4_amax[0, 0].item() for rows, int4_amax, _ in seen if rows == 16}
                # gate starts at row marker 1, up at INTERMEDIATE_DIM + 1.
                assert gate_up_rows == {1.0, float(INTERMEDIATE_DIM + 1)}
        finally:
            if QuantModuleRegistry.get(expert_type) is not None:
                QuantModuleRegistry.unregister(expert_type)

    @pytest.mark.parametrize(
        ("amax_layout", "expected_gate_shape", "expected_down_shape"),
        [
            ("per_tensor", (1,), (1,)),
            ("per_channel", (INTERMEDIATE_DIM, 1), (HIDDEN_DIM, 1)),
            ("per_row_1d", (INTERMEDIATE_DIM,), (HIDDEN_DIM,)),
            ("per_block", (INTERMEDIATE_DIM, 2), (HIDDEN_DIM, 2)),
            # What block-quant calibration records: (rows * blocks, 1).
            ("flat_block", (INTERMEDIATE_DIM, 2), (HIDDEN_DIM, 2)),
        ],
    )
    def test_uncalibrated_zero_amax_fallback_keeps_layout(
        self, monkeypatch, amax_layout, expected_gate_shape, expected_down_shape
    ):
        """The weight-derived fallback must reproduce the layout of the quantizer's zero amax."""
        experts = _SyntheticFusedExperts()
        expert_type = type(experts)
        if QuantModuleRegistry.get(expert_type) is None:
            QuantModuleRegistry.register({expert_type: "test.SyntheticFusedExperts"})(
                _QuantFusedExperts
            )
        try:
            converted = QuantModuleRegistry.convert(experts)
            layouts = {
                "per_tensor": lambda rows: (1,),
                "per_channel": lambda rows: (rows, 1),
                "per_row_1d": lambda rows: (rows,),
                "per_block": lambda rows: (rows, 2),
                "flat_block": lambda rows: (rows * 2, 1),
            }
            for idx in range(NUM_EXPERTS):
                for wq, rows in (
                    (converted.gate_up_proj_weight_quantizers[idx], 2 * INTERMEDIATE_DIM),
                    (converted.down_proj_weight_quantizers[idx], HIDDEN_DIM),
                ):
                    wq._disabled = False
                    wq.amax = torch.zeros(*layouts[amax_layout](rows))

            seen = []

            def _spy_export(wrapper, dtype, **_kwargs):
                seen.append(wrapper.weight_quantizer._amax.detach().clone())

            monkeypatch.setattr(
                "modelopt.torch.export.unified_export_hf._export_quantized_weight",
                _spy_export,
            )
            with pytest.warns(UserWarning, match="was not calibrated"):
                _export_fused_experts(converted, torch.float16)

            assert all(torch.all(a > 0) for a in seen)
            # Export order per expert is gate, up, down.
            for gate, up, down in zip(seen[0::3], seen[1::3], seen[2::3]):
                assert gate.shape == up.shape == expected_gate_shape
                assert down.shape == expected_down_shape
        finally:
            if QuantModuleRegistry.get(expert_type) is not None:
                QuantModuleRegistry.unregister(expert_type)

    def test_fallback_amax_refuses_layouts_it_cannot_group(self):
        quantizer = TensorQuantizer(QuantizerAttributeConfig(num_bits=8, axis=1))
        quantizer.amax = torch.zeros(1, HIDDEN_DIM)  # per input channel: not a (rows, blocks) grid
        with pytest.raises(ValueError, match="Cannot derive amax"):
            _fallback_amax(torch.randn(INTERMEDIATE_DIM, HIDDEN_DIM), quantizer)


# ---------------------------------------------------------------------------
# Tests for tied-experts dedup in _export_fused_experts
# ---------------------------------------------------------------------------
def _build_two_moe_blocks(tie: bool) -> nn.Module:
    """Build a parent with two _SyntheticSparseMoeBlock children, optionally with tied 3-D params.

    When ``tie`` is set the parent both shares the 3-D expert Parameters and declares the tie via
    ``_tied_weights_keys``, so the name-based map resolves it (object sharing alone is not enough).
    """
    parent = nn.Module()
    parent.encoder = _SyntheticSparseMoeBlock()
    parent.decoder = _SyntheticSparseMoeBlock()
    if tie:
        tie_fused_experts_3d_params(parent.encoder.experts, parent.decoder.experts)
        parent._tied_weights_keys = {
            r"^encoder\.experts\.gate_up_proj$": "decoder.experts.gate_up_proj",
            r"^encoder\.experts\.down_proj$": "decoder.experts.down_proj",
        }
    return parent


def _moe_fp8_quant_cfg():
    """Custom inline FP8 cfg targeting the MoE-specific quantizer names."""
    return {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": "*gate_up_proj_input_quantizer",
                "cfg": {"num_bits": 8, "axis": None},
            },
            {"quantizer_name": "*down_proj_input_quantizer", "cfg": {"num_bits": 8, "axis": None}},
            {"quantizer_name": "*gate_up_proj_weight_quantizer", "cfg": {"num_bits": 8, "axis": 0}},
            {"quantizer_name": "*down_proj_weight_quantizer", "cfg": {"num_bits": 8, "axis": 0}},
        ],
        "algorithm": "max",
    }


def _calibrate_two_moe_blocks(parent):
    """Fire one calibration batch through both encoder.experts and decoder.experts."""

    def forward_loop(m):
        torch.manual_seed(0)
        x = torch.randn(1, 4, HIDDEN_DIM)
        m.encoder(x)
        m.decoder(x)

    mtq.quantize(parent, _moe_fp8_quant_cfg(), forward_loop=forward_loop)


class TestExportFusedExpertsTiedDedup:
    @staticmethod
    def _cleanup_registry(mod_type):
        if QuantModuleRegistry.get(mod_type) is not None:
            QuantModuleRegistry.unregister(mod_type)

    def test_tied_fused_experts_pack_independently_to_equal_values(self):
        """Tied FusedExperts pack independently: distinct data_ptrs, equal bytes (dropped by name later)."""
        parent = _build_two_moe_blocks(tie=True)
        expert_type = type(parent.encoder.experts)
        self._cleanup_registry(expert_type)
        try:
            _calibrate_two_moe_blocks(parent)

            _export_fused_experts(parent.encoder.experts, torch.float16)
            _export_fused_experts(parent.decoder.experts, torch.float16)

            for idx in range(NUM_EXPERTS):
                enc_expert = getattr(parent.encoder.experts, str(idx))
                dec_expert = getattr(parent.decoder.experts, str(idx))
                for proj_name in ("gate_proj", "up_proj", "down_proj"):
                    enc_proj = getattr(enc_expert, proj_name)
                    dec_proj = getattr(dec_expert, proj_name)
                    # independent storage (no aliasing) ...
                    assert enc_proj.weight.data_ptr() != dec_proj.weight.data_ptr()
                    # ... but byte-identical, so postprocess drops one by name
                    assert torch.equal(enc_proj.weight, dec_proj.weight)
        finally:
            self._cleanup_registry(expert_type)

    def test_untied_fused_experts_have_independent_buffers(self):
        """Untied FusedExperts stay fully independent — no aliasing, distinct values."""
        parent = _build_two_moe_blocks(tie=False)
        expert_type = type(parent.encoder.experts)
        self._cleanup_registry(expert_type)
        try:
            _calibrate_two_moe_blocks(parent)

            _export_fused_experts(parent.encoder.experts, torch.float16)
            _export_fused_experts(parent.decoder.experts, torch.float16)

            for idx in range(NUM_EXPERTS):
                enc_expert = getattr(parent.encoder.experts, str(idx))
                dec_expert = getattr(parent.decoder.experts, str(idx))
                for proj_name in ("gate_proj", "up_proj", "down_proj"):
                    enc_proj = getattr(enc_expert, proj_name)
                    dec_proj = getattr(dec_expert, proj_name)
                    assert enc_proj.weight.data_ptr() != dec_proj.weight.data_ptr()
        finally:
            self._cleanup_registry(expert_type)


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
        model.moe.experts.config = _StubConfig(impl="batched_mm")
        normalize_experts_impl_on_the_fly(model)
        assert text_cfg._experts_implementation == "grouped_mm"
        assert vision_cfg._experts_implementation == "grouped_mm"
        # A module's own (deep-copied) config is redirected too, as Nemotron-H MTP needs.
        assert model.moe.experts.config._experts_implementation == "grouped_mm"

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

    @pytest.mark.parametrize("impl", ["eager", "grouped_mm", None])
    def test_leaves_a_supported_value_unchanged(self, impl):
        """``eager``, ``grouped_mm`` and an explicit None (the model's eager ``F.linear`` forward) are
        supported, so they are left as they are."""
        cfg = _StubConfig()
        cfg._experts_implementation = impl
        model = _TinyMoEModelWithConfig(cfg)
        normalize_experts_impl_on_the_fly(model)
        assert cfg._experts_implementation == impl


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
            _route_once_to_each_expert(m)

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


def _local_hessian_grouped_capture_job(rank, size, impl):
    """Local-Hessian capture under a grouped dispatcher; torch_a2a gets a one-rank EP group."""
    ep_group = dist.new_subgroups(1)[0] if impl == "torch_a2a" else None
    try:
        torch.manual_seed(0)
        model = _GroupedMoEModel(impl)
        model.config.ep_group = ep_group

        weight_quant = {"num_bits": 8, "axis": 0}
        quant_cfg = {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {"quantizer_name": "*gate_up_proj_weight_quantizer", "cfg": weight_quant},
                {"quantizer_name": "*down_proj_weight_quantizer", "cfg": weight_quant},
            ],
            "algorithm": "max",
        }
        batches = [torch.randn(1, 8, HIDDEN_DIM) for _ in range(3)]

        def forward_loop(m):
            for batch in batches:
                m(batch)

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

        # Each expert's Hessian, in both projections, holds exactly the rows routed to it.
        routed = torch.cat([model.moe.gate(b.view(-1, HIDDEN_DIM))[2] for b in batches])
        rows = torch.bincount(routed.flatten(), minlength=NUM_EXPERTS).tolist()
        accumulators = model._local_hessian_accumulators
        for quantizers in (
            experts.gate_up_proj_weight_quantizers,
            experts.down_proj_weight_quantizers,
        ):
            captured = [
                accumulators[id(q)].num_samples if id(q) in accumulators else 0 for q in quantizers
            ]
            assert captured == rows, (captured, rows)
        for qid, acc in accumulators.items():
            if acc.num_samples:
                assert (acc.cout, acc.cin) == expected_shape[qid]
        assert all(q.amax is not None and torch.isfinite(q.amax).all() for q in expert_quantizers)
        assert any(
            id(q) in max_amax and not torch.allclose(q.amax, max_amax[id(q)])
            for q in expert_quantizers
        )
    finally:
        if QuantModuleRegistry.get(_SyntheticGroupedExperts) is not None:
            QuantModuleRegistry.unregister(_SyntheticGroupedExperts)
        if ep_group is not None:
            dist.destroy_process_group(ep_group)


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
                # "*up_proj_..." also matches the gated layout's gate_up_proj quantizers.
                {
                    "quantizer_name": "*up_proj_input_quantizer",
                    "cfg": {"num_bits": 8, "axis": None},
                },
                {
                    "quantizer_name": "*down_proj_input_quantizer",
                    "cfg": {"num_bits": 8, "axis": None},
                },
                {"quantizer_name": "*up_proj_weight_quantizer", "cfg": {"num_bits": 8, "axis": 0}},
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

    @pytest.mark.parametrize("gated", [True, False], ids=["gated", "ungated"])
    def test_forward_passthrough_matches(self, gated):
        """Converted grouped experts with their quantizers disabled match the unconverted forward,
        in the gated layout and in NemotronH's ungated one."""
        self._cleanup_registry(_SyntheticGroupedExperts)
        torch.manual_seed(0)
        model = _GroupedMoEModel("grouped_mm", gated=gated)
        ref = _SyntheticGroupedExperts(_ExpertsConfig("grouped_mm"), gated=gated)
        ref.load_state_dict(model.moe.experts.state_dict())

        register_fused_experts_on_the_fly(model)
        converted = QuantModuleRegistry.convert(model.moe.experts)
        for q in converted.modules():
            if isinstance(q, TensorQuantizer):
                q.disable()

        seq = 8
        hidden = torch.randn(seq, HIDDEN_DIM)
        top_k_index = torch.randint(0, NUM_EXPERTS, (seq, TOP_K))
        top_k_weights = torch.softmax(torch.randn(seq, TOP_K), dim=1)
        with torch.no_grad():
            out_ref = ref(hidden, top_k_index, top_k_weights)
            out_test = converted(hidden, top_k_index, top_k_weights)
        assert torch.allclose(out_ref, out_test, atol=1e-6), (
            f"Max diff: {(out_ref - out_test).abs().max().item()}"
        )
        self._cleanup_registry(_SyntheticGroupedExperts)

    @pytest.mark.parametrize("gated", [True, False], ids=["gated", "ungated"])
    def test_calibration_matches_eager(self, gated):
        """grouped_mm PTQ calibration yields the same quantizer state as eager PTQ, in the gated
        layout and in NemotronH's ungated one, which _QuantNemotronHExperts wraps.

        Identical weights + identical routing ⇒ the shared input quantizers see the
        same activation set and each per-expert weight quantizer sees the same slice,
        so amax must match (max abs is order-free; only the down-proj input flows
        through a GEMM, hence the small tolerance there).
        """
        self._cleanup_registry(_SyntheticGroupedExperts)
        torch.manual_seed(0)
        eager_model = _GroupedMoEModel("eager", gated=gated)
        grouped_model = _GroupedMoEModel("grouped_mm", gated=gated)
        grouped_model.load_state_dict(eager_model.state_dict())

        mtq.quantize(eager_model, self._quant_cfg(), forward_loop=self._forward_loop_all_experts)
        mtq.quantize(grouped_model, self._quant_cfg(), forward_loop=self._forward_loop_all_experts)

        e = eager_model.moe.experts
        g = grouped_model.moe.experts
        if not gated:
            assert isinstance(e, _QuantNemotronHExperts) and isinstance(g, _QuantNemotronHExperts)
        assert g._running_grouped_mm is True
        assert e._running_grouped_mm is False
        first = "gate_up_proj" if gated else "up_proj"

        # Shared input quantizers (down flows through a GEMM -> small tolerance).
        for q_name in (f"{first}_input_quantizer", "down_proj_input_quantizer"):
            ea = getattr(e, q_name).amax
            ga = getattr(g, q_name).amax
            assert ea is not None and ga is not None, f"{q_name} not calibrated"
            assert torch.allclose(ea, ga, atol=1e-3, rtol=1e-2), (
                f"{q_name} amax mismatch: eager={ea} grouped={ga}"
            )

        # Per-expert weight amax (weights identical -> tight match), all experts calibrated.
        for idx in range(NUM_EXPERTS):
            for q_list in (f"{first}_weight_quantizers", "down_proj_weight_quantizers"):
                eq = getattr(e, q_list)[idx].amax
                gq = getattr(g, q_list)[idx].amax
                assert eq is not None and gq is not None, f"{q_list}[{idx}] not calibrated"
                assert torch.allclose(eq, gq, atol=1e-5), f"{q_list}[{idx}] weight amax mismatch"
        self._cleanup_registry(_SyntheticGroupedExperts)

    @pytest.mark.parametrize("gated", [True, False], ids=["gated", "ungated"])
    def test_gptq_hessians_match_eager(self, gated, monkeypatch):
        """GPTQ gives each expert the Hessian of only its own tokens under grouped_mm, as eager.

        grouped_mm quantizes one expert-sorted batch per projection rather than one call per
        expert, so GPTQ's per-expert hook splits it back; keyed by the eager loop's expert index
        alone, the first expert took every token and the others none.
        """
        hessians = {}
        free = GPTQHelper.free

        def recording(helper):
            hessians[helper.name] = (helper.n_samples, helper.hessian.clone())
            free(helper)

        monkeypatch.setattr(GPTQHelper, "free", recording)
        cfg = {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {
                    "quantizer_name": "*weight_quantizer",
                    "cfg": {"num_bits": 4, "block_sizes": {-1: 16}},
                },
            ],
            "algorithm": {"method": "gptq", "block_size": 16},
        }
        self._cleanup_registry(_SyntheticGroupedExperts)
        torch.manual_seed(0)
        eager_model = _GroupedMoEModel("eager", gated=gated)
        grouped_model = _GroupedMoEModel("grouped_mm", gated=gated)
        grouped_model.load_state_dict(eager_model.state_dict())
        found = []
        for model in (eager_model, grouped_model):
            hessians.clear()
            mtq.quantize(model, copy.deepcopy(cfg), forward_loop=self._forward_loop_all_experts)
            found.append(dict(hessians))
        self._cleanup_registry(_SyntheticGroupedExperts)

        eager, grouped = found
        assert eager.keys() == grouped.keys() and len(eager) == 2 * NUM_EXPERTS
        for name, (n_samples, hessian) in eager.items():
            assert n_samples > 0 and grouped[name][0] == n_samples, name
            # The down-proj input flows through a GEMM, hence the tolerance.
            torch.testing.assert_close(grouped[name][1], hessian, rtol=1e-3, atol=1e-4)

    @pytest.mark.parametrize("impl", ["grouped_mm", "torch_a2a"])
    def test_local_hessian_captures_per_expert_in_grouped_mode(self, cpu_dist_workers, impl):
        """local_hessian splits the expert-sorted grouped batch by offsets and refines
        each expert's weight amax — the grouped analogue of the eager per-expert hook.
        The torch_a2a forward hands ``_grouped_linear`` the same batch, so it splits the same."""
        cpu_dist_workers.run(_local_hessian_grouped_capture_job, impl=impl)


class TestNonGatedFusedExperts:
    @staticmethod
    def _cleanup_registry(mod_type):
        if QuantModuleRegistry.get(mod_type) is not None:
            QuantModuleRegistry.unregister(mod_type)

    def test_detected_and_picks_nongated_wrapper(self):
        module = _SyntheticNonGatedFusedExperts()
        assert _is_fused_experts_module(module) is True
        assert _fused_experts_wrapper(module) is _QuantNemotronHExperts

    def test_gated_still_picks_base_wrapper(self):
        assert _fused_experts_wrapper(_SyntheticFusedExperts()) is _QuantFusedExperts

    def test_register_uses_nongated_wrapper(self):
        model = _TinyNonGatedMoEModel()
        expert_type = type(model.moe.experts)
        self._cleanup_registry(expert_type)
        register_fused_experts_on_the_fly(model)
        try:
            converted = QuantModuleRegistry.convert(model.moe.experts)
            assert isinstance(converted, _QuantNemotronHExperts)
            assert converted._first_proj_attr == "up_proj"
            assert converted._is_gated is False
            assert hasattr(converted, "up_proj_input_quantizer")
            assert hasattr(converted, "up_proj_weight_quantizers")
            assert not hasattr(converted, "gate_up_proj_input_quantizer")
            assert not hasattr(converted, "gate_up_proj_weight_quantizers")
            assert len(converted.up_proj_weight_quantizers) == NUM_EXPERTS
            assert len(converted.down_proj_weight_quantizers) == NUM_EXPERTS
        finally:
            self._cleanup_registry(expert_type)

    def test_forward_passthrough_matches(self):
        model = _TinyNonGatedMoEModel()
        expert_type = type(model.moe.experts)
        self._cleanup_registry(expert_type)

        ref_experts = _SyntheticNonGatedFusedExperts()
        ref_experts.load_state_dict(model.moe.experts.state_dict())

        register_fused_experts_on_the_fly(model)
        try:
            converted = QuantModuleRegistry.convert(model.moe.experts)
            # Disable quantizers to isolate the wrapper's structural forward
            # (the F.linear interception / per-expert index routing) from
            # dynamic-quant noise — this is a passthrough equivalence check.
            for q in converted.modules():
                if isinstance(q, TensorQuantizer):
                    q.disable()
            seq_len = 8
            hidden_states = torch.randn(seq_len, HIDDEN_DIM)
            top_k_index = torch.randint(0, NUM_EXPERTS, (seq_len, TOP_K))
            top_k_weights = torch.softmax(torch.randn(seq_len, TOP_K), dim=-1)
            with torch.no_grad():
                out_ref = ref_experts(hidden_states, top_k_index, top_k_weights)
                out_test = converted(hidden_states, top_k_index, top_k_weights)
            assert torch.allclose(out_ref, out_test, atol=1e-5), (
                f"Max diff: {(out_ref - out_test).abs().max().item()}"
            )
        finally:
            self._cleanup_registry(expert_type)

    def test_expert_index_recovery(self):
        experts = _SyntheticNonGatedFusedExperts()
        expert_type = type(experts)
        self._cleanup_registry(expert_type)
        register_fused_experts_on_the_fly(_TinyNonGatedMoEModel())
        try:
            converted = QuantModuleRegistry.convert(experts)
            for idx in range(NUM_EXPERTS):
                weight_slice = converted.up_proj[idx]
                assert converted._get_expert_idx_from_first_proj(weight_slice) == idx
        finally:
            self._cleanup_registry(expert_type)

    def _nongated_fp8_cfg(self):
        return {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {
                    "quantizer_name": "*up_proj_input_quantizer",
                    "cfg": {"num_bits": 8, "axis": None},
                },
                {
                    "quantizer_name": "*down_proj_input_quantizer",
                    "cfg": {"num_bits": 8, "axis": None},
                },
                {
                    "quantizer_name": "*up_proj_weight_quantizer",
                    "cfg": {"num_bits": 8, "axis": 0},
                },
                {
                    "quantizer_name": "*down_proj_weight_quantizer",
                    "cfg": {"num_bits": 8, "axis": 0},
                },
            ],
            "algorithm": "max",
        }

    def test_calibration_populates_all_expert_quantizers(self):
        model = _TinyNonGatedMoEModel()
        expert_type = type(model.moe.experts)
        self._cleanup_registry(expert_type)

        def forward_loop(m):
            torch.manual_seed(0)
            _route_once_to_each_expert(m)

        try:
            mtq.quantize(model, self._nongated_fp8_cfg(), forward_loop=forward_loop)
            experts = model.moe.experts
            assert experts.up_proj_input_quantizer.amax is not None
            assert experts.down_proj_input_quantizer.amax is not None
            for idx in range(NUM_EXPERTS):
                assert experts.up_proj_weight_quantizers[idx].amax is not None
                assert experts.down_proj_weight_quantizers[idx].amax is not None
        finally:
            self._cleanup_registry(expert_type)

    def test_export_creates_per_expert_up_down_only(self):
        model = _TinyNonGatedMoEModel()
        expert_type = type(model.moe.experts)
        self._cleanup_registry(expert_type)

        def forward_loop(m):
            torch.manual_seed(0)
            for _ in range(2):
                m(torch.randn(1, 4, HIDDEN_DIM))

        try:
            mtq.quantize(model, self._nongated_fp8_cfg(), forward_loop=forward_loop)
            converted = model.moe.experts
            expected_up_input_scale = (
                converted.up_proj_input_quantizer.amax / converted.up_proj_input_quantizer.maxbound
            ).squeeze()
            expected_down_input_scale = (
                converted.down_proj_input_quantizer.amax
                / converted.down_proj_input_quantizer.maxbound
            ).squeeze()
            _export_fused_experts(converted, torch.float16)

            for idx in range(NUM_EXPERTS):
                expert_mod = getattr(converted, str(idx), None)
                assert expert_mod is not None, f"Missing expert submodule {idx}"
                # Non-gated: up_proj + down_proj, but NO gate_proj.
                assert hasattr(expert_mod, "up_proj"), f"Expert {idx} missing up_proj"
                assert hasattr(expert_mod, "down_proj"), f"Expert {idx} missing down_proj"
                assert not hasattr(expert_mod, "gate_proj"), (
                    f"Expert {idx} should NOT have gate_proj (non-gated MLP)"
                )
                assert expert_mod.up_proj.weight.shape == (INTERMEDIATE_DIM, HIDDEN_DIM)
                assert expert_mod.down_proj.weight.shape == (HIDDEN_DIM, INTERMEDIATE_DIM)
                torch.testing.assert_close(expert_mod.up_proj.input_scale, expected_up_input_scale)
                torch.testing.assert_close(
                    expert_mod.down_proj.input_scale, expected_down_input_scale
                )

            # Fused params and per-expert quantizer lists are removed.
            assert not hasattr(converted, "up_proj")
            assert not hasattr(converted, "down_proj")
            assert not hasattr(converted, "up_proj_weight_quantizers")
            assert not hasattr(converted, "down_proj_weight_quantizers")
        finally:
            self._cleanup_registry(expert_type)

    def test_enumeration_yields_up_and_down_proj(self):
        """weight_attr_names must yield up_proj and down_proj for non-gated experts."""
        model = _TinyNonGatedMoEModel()
        expert_type = type(model.moe.experts)
        self._cleanup_registry(expert_type)
        register_fused_experts_on_the_fly(model)
        try:
            converted = QuantModuleRegistry.convert(model.moe.experts)
            assert set(weight_attr_names(converted)) == {"up_proj", "down_proj"}
        finally:
            self._cleanup_registry(expert_type)

    def test_split_gated_layout_not_claimed_as_nongated(self):
        """A fused container with a separate 3-D gate_proj (split-gated: three
        F.linear calls per expert) must NOT be claimed by the non-gated wrapper,
        whose two-call toggle and up_proj-storage index recovery assume exactly
        two projections. It is left unsupported (None) rather than mis-quantized."""

        class _SplitGatedExperts(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_experts = NUM_EXPERTS
                self.gate_proj = nn.Parameter(
                    torch.randn(NUM_EXPERTS, INTERMEDIATE_DIM, HIDDEN_DIM) * 0.02
                )
                self.up_proj = nn.Parameter(
                    torch.randn(NUM_EXPERTS, INTERMEDIATE_DIM, HIDDEN_DIM) * 0.02
                )
                self.down_proj = nn.Parameter(
                    torch.randn(NUM_EXPERTS, HIDDEN_DIM, INTERMEDIATE_DIM) * 0.02
                )
                self.act_fn = nn.SiLU()

        module = _SplitGatedExperts()
        assert _fused_experts_wrapper(module) is None
        assert _is_fused_experts_module(module) is False

    def test_get_quant_config_resolves_nongated_experts(self):
        """get_quant_config must detect the non-gated experts as quantized."""
        model = _TinyNonGatedMoEModel()
        expert_type = type(model.moe.experts)
        self._cleanup_registry(expert_type)

        def forward_loop(m):
            torch.manual_seed(0)
            for _ in range(2):
                m(torch.randn(1, 4, HIDDEN_DIM))

        try:
            mtq.quantize(model, self._nongated_fp8_cfg(), forward_loop=forward_loop)
            # Format resolves (via down_proj) instead of QUANTIZATION_NONE (None).
            assert get_quantization_format(model.moe.experts) is not None
            # The non-gated experts are reflected in the produced quant config.
            quant = get_quant_config(model)["quantization"]
            assert quant.get("quant_algo") is not None
        finally:
            self._cleanup_registry(expert_type)


# ---------------------------------------------------------------------------
# Regression tests for the real transformers Qwen3-VL MoE text experts
# ---------------------------------------------------------------------------
QWEN_HIDDEN_DIM = 32
QWEN_INTERMEDIATE_DIM = 12
QWEN_NUM_EXPERTS = 4
QWEN_TOP_K = 2


def _make_qwen3_vl_moe_experts():
    """Build a tiny real ``Qwen3VLMoeTextExperts`` with initialized weights."""
    from transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe import Qwen3VLMoeTextConfig
    from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeTextExperts

    config = Qwen3VLMoeTextConfig(
        hidden_size=QWEN_HIDDEN_DIM,
        intermediate_size=QWEN_HIDDEN_DIM,
        moe_intermediate_size=QWEN_INTERMEDIATE_DIM,
        num_experts=QWEN_NUM_EXPERTS,
        num_experts_per_tok=QWEN_TOP_K,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=128,
    )
    # The fused forward dispatches on this; ``eager`` is the only backend that routes
    # through ``F.linear`` and therefore through the quantizer hooks.
    config._experts_implementation = "eager"
    experts = Qwen3VLMoeTextExperts(config)
    # Weights are created with ``torch.empty``; fill them so comparisons are meaningful.
    torch.manual_seed(0)
    with torch.no_grad():
        for param in experts.parameters():
            param.normal_(std=0.02)
    return experts


def _qwen3_vl_moe_forward_args():
    """Routing inputs sized so every expert is hit."""
    seq_len = QWEN_NUM_EXPERTS // QWEN_TOP_K
    torch.manual_seed(0)
    hidden_states = torch.randn(seq_len, QWEN_HIDDEN_DIM)
    router_indices = torch.arange(QWEN_NUM_EXPERTS, dtype=torch.long).reshape(seq_len, QWEN_TOP_K)
    top_k_weights = torch.softmax(torch.randn(seq_len, QWEN_TOP_K), dim=-1)
    return hidden_states, router_indices, top_k_weights


class TestQwen3VLMoeTextExperts:
    """Qwen3-VL MoE text experts must quantize through the generic fused-experts wrapper."""

    @staticmethod
    def _experts_type():
        from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeTextExperts

        return Qwen3VLMoeTextExperts

    def test_left_to_generic_fused_experts_wrapper(self):
        """The experts must be left to the generic fused-experts wrapper."""
        # A static registration would shadow on-the-fly detection.
        assert QuantModuleRegistry.get(self._experts_type()) is None
        assert _fused_experts_wrapper(_make_qwen3_vl_moe_experts()) is _QuantFusedExperts

    def test_convert_and_forward_matches_reference(self):
        """Conversion must succeed and stay numerically transparent before calibration."""
        experts = _make_qwen3_vl_moe_experts()
        experts_type = self._experts_type()
        registered_before = QuantModuleRegistry.get(experts_type)

        reference = _make_qwen3_vl_moe_experts()
        args = _qwen3_vl_moe_forward_args()

        model = nn.Module()
        model.experts = experts
        register_fused_experts_on_the_fly(model)
        try:
            converted = QuantModuleRegistry.convert(experts)
            with torch.no_grad():
                out_ref = reference(*args)
                out_test = converted(*args)
            assert torch.allclose(out_ref, out_test, atol=1e-4), (
                f"Max diff: {(out_ref - out_test).abs().max().item()}"
            )
        finally:
            if registered_before is None and QuantModuleRegistry.get(experts_type) is not None:
                QuantModuleRegistry.unregister(experts_type)

    def test_quantize_collects_amax(self):
        """Calibration must actually reach the experts rather than silently no-op."""
        experts_type = self._experts_type()
        registered_before = QuantModuleRegistry.get(experts_type)

        model = nn.Module()
        model.experts = _make_qwen3_vl_moe_experts()
        args = _qwen3_vl_moe_forward_args()
        model.forward = lambda: model.experts(*args)

        try:
            mtq.quantize(model, mtq.FP8_DEFAULT_CFG, forward_loop=lambda m: m())
            amaxes = [
                m.amax
                for m in model.experts.modules()
                if isinstance(m, TensorQuantizer) and getattr(m, "amax", None) is not None
            ]
            assert amaxes, "No amax collected: the experts were not quantized"
            assert all(a.abs().sum() > 0 for a in amaxes)
        finally:
            if registered_before is None and QuantModuleRegistry.get(experts_type) is not None:
                QuantModuleRegistry.unregister(experts_type)


@pytest.mark.parametrize("activation_scheme", ["dynamic", "static"])
@pytest.mark.parametrize("fold", [True, False])
@pytest.mark.parametrize("has_gate", [True, False])
@pytest.mark.parametrize("block_size", [(128, 128), (64, 128), None])
def test_native_fp8_experts_scales(has_gate, block_size, fold, activation_scheme):
    torch.manual_seed(42)
    config = PretrainedConfig(
        hidden_size=256, intermediate_size=256, num_experts=2, hidden_act="silu"
    )
    experts = FP8Experts(
        config, block_size=block_size, has_gate=has_gate, activation_scheme=activation_scheme
    )
    first = "gate_up_proj" if has_gate else "up_proj"
    expected = {}
    for name in (first, "down_proj"):
        weight = getattr(experts, name)
        scales = getattr(experts, f"{name}_scale_inv")
        with torch.no_grad():
            weight.copy_(torch.randn(weight.shape).to(weight.dtype))
            scales.copy_(torch.linspace(1e-8, 0.08, scales.numel()).reshape(scales.shape))
        rows, cols = block_size or (weight.shape[1] // scales.shape[1], weight.shape[2])
        expanded = scales.float().repeat_interleave(rows, 1).repeat_interleave(cols, 2)
        expected[name] = weight.float() * expanded[:, : weight.shape[1], : weight.shape[2]]

    reference = copy.deepcopy(experts)
    for name, weight in expected.items():
        setattr(reference, name, nn.Parameter(weight, requires_grad=False))
    mtq.quantize(
        experts, {"quant_cfg": [{"quantizer_name": "*", "enable": False}], "algorithm": None}
    )
    for (weight, _), expected_weight in zip(
        experts.iter_weights_for_calibration(),
        [w for name in (first, "down_proj") for w in expected[name]],
        strict=True,
    ):
        torch.testing.assert_close(weight.float(), expected_weight)

    for name in (first, "down_proj"):
        for idx in range(2):
            for dtype in (torch.float16, torch.bfloat16):
                torch.testing.assert_close(
                    experts.get_expert_weight(name, idx, dtype),
                    expected[name][idx].to(dtype),
                    rtol=0,
                    atol=0,
                )

    inputs = torch.randn(2, 256)
    indices = torch.tensor([[0], [1]])
    routing = torch.ones(2, 1)
    torch.testing.assert_close(
        experts(inputs, indices, routing), reference(inputs, indices, routing)
    )
    for _, quantizer in experts.iter_weights_for_calibration():
        quantizer.enable()
    mtq.calibrate(experts, algorithm="max", forward_loop=None)
    for weight, quantizer in experts.iter_weights_for_calibration():
        torch.testing.assert_close(quantizer.amax, weight.abs().amax())
        quantizer.disable()
    if fold:
        experts.unpack_weight()
        experts.fold_weight()
        torch.testing.assert_close(
            experts(inputs.bfloat16(), indices, routing),
            copy.deepcopy(reference).bfloat16()(inputs.bfloat16(), indices, routing),
        )
    torch.testing.assert_close(
        experts(inputs, indices, routing), reference(inputs, indices, routing)
    )
    _export_fused_experts(experts, torch.float32)
    for idx in range(2):
        exported = getattr(experts, str(idx))
        if has_gate:
            torch.testing.assert_close(exported.gate_proj.weight, expected[first][idx, :256])
            torch.testing.assert_close(exported.up_proj.weight, expected[first][idx, 256:])
        else:
            torch.testing.assert_close(exported.up_proj.weight, expected[first][idx])
        torch.testing.assert_close(exported.down_proj.weight, expected["down_proj"][idx])
    assert not any("scale_inv" in key or "activation_scale" in key for key in experts.state_dict())


@pytest.mark.parametrize("has_gate", [True, False])
@pytest.mark.parametrize("unpacked_projection", [None, "first", "down_proj"])
@pytest.mark.parametrize("fold_api", ["direct", "permanent", "temporary"])
def test_native_fp8_experts_reject_packed_fold(has_gate, unpacked_projection, fold_api):
    torch.manual_seed(42)
    experts = FP8Experts(
        PretrainedConfig(hidden_size=32, intermediate_size=32, num_experts=2, hidden_act="silu"),
        block_size=None,
        has_gate=has_gate,
    )
    first = "gate_up_proj" if has_gate else "up_proj"
    for name in (first, "down_proj"):
        weight = getattr(experts, name)
        with torch.no_grad():
            weight.copy_(torch.randn(weight.shape).to(weight.dtype))
            getattr(experts, f"{name}_scale_inv").fill_(0.1)
    # Put a foldable module first to check model-wide validation precedes all mutation.
    model = nn.Sequential(nn.Linear(32, 32), experts)
    mtq.quantize(model, mtq.INT8_DEFAULT_CFG, forward_loop=None)
    if unpacked_projection is not None:
        name = first if unpacked_projection == "first" else unpacked_projection
        weight = torch.stack([experts.get_expert_weight(name, idx) for idx in range(2)])
        setattr(experts, name, nn.Parameter(weight, requires_grad=False))
    params = dict(model.named_parameters())
    state = copy.deepcopy(model.state_dict())
    quantizers = [q for q in model.modules() if isinstance(q, TensorQuantizer)]
    enabled = [q.is_enabled for q in quantizers]
    inputs = torch.randn(2, 32)
    indices = torch.tensor([[0], [1]])
    routing = torch.ones(2, 1)
    expected = experts(inputs, indices, routing)

    with pytest.raises(ValueError, match=r"Call unpack_weight\(\) before folding"):
        if fold_api == "direct":
            experts.fold_weight()
        elif fold_api == "permanent":
            mtq.fold_weight(model)
        else:
            with mtq.temporarily_fold_weights(model):
                pytest.fail("Packed FP8 weights must be rejected before entering the context")

    assert [q.is_enabled for q in quantizers] == enabled
    for name, param in model.named_parameters():
        assert param is params[name]
        assert param.dtype == state[name].dtype
    # Compare FP8 payloads in FP32 because CPU close/equality kernels do not support FP8.
    torch.testing.assert_close(
        {name: value.float() for name, value in model.state_dict().items()},
        {name: value.float() for name, value in state.items()},
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(experts(inputs, indices, routing), expected)


@pytest.mark.parametrize("has_gate", [True, False])
@pytest.mark.parametrize("exception", [True, False])
@pytest.mark.parametrize("snapshot_device", [None, "cpu"])
def test_native_fp8_experts_unpacked_temporary_fold(has_gate, exception, snapshot_device):
    torch.manual_seed(42)
    config = PretrainedConfig(
        hidden_size=32, intermediate_size=32, num_experts=2, hidden_act="silu"
    )
    experts = FP8Experts(config, block_size=None, has_gate=has_gate)
    first = "gate_up_proj" if has_gate else "up_proj"
    for name in (first, "down_proj"):
        weight = getattr(experts, name)
        scales = getattr(experts, f"{name}_scale_inv")
        with torch.no_grad():
            weight.copy_(torch.randn(weight.shape).to(weight.dtype))
            scales.copy_(torch.linspace(0.01, 0.08, scales.numel()).reshape(scales.shape))
    mtq.quantize(
        experts,
        {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 4, "axis": None}},
            ],
            "algorithm": "max",
        },
    )
    experts.unpack_weight()
    params = {name: getattr(experts, name) for name in (first, "down_proj")}
    state = copy.deepcopy(experts.state_dict())
    quantizers = [q for _, q in experts.iter_weights_for_calibration()]
    inputs = torch.randn(2, 32)
    indices = torch.tensor([[0], [1]])
    routing = torch.ones(2, 1)
    expected = experts(inputs, indices, routing)
    expected_exit = pytest.raises(RuntimeError, match="fold exit") if exception else nullcontext()
    with expected_exit, mtq.temporarily_fold_weights(experts, snapshot_device=snapshot_device):
        assert all(not q.is_enabled for q in quantizers)
        for name, param in params.items():
            assert getattr(experts, name) is param
            assert param.dtype == torch.get_default_dtype()
            assert not torch.equal(param, state[name])
        torch.testing.assert_close(experts(inputs, indices, routing), expected)
        if exception:
            raise RuntimeError("fold exit")
    for name, param in params.items():
        assert getattr(experts, name) is param
        assert param.dtype == torch.get_default_dtype()
    assert all(q.is_enabled for q in quantizers)
    torch.testing.assert_close(experts.state_dict(), state, rtol=0, atol=0)
    torch.testing.assert_close(experts(inputs, indices, routing), expected)


@pytest.mark.parametrize(
    ("cfg", "algorithm"),
    [
        ({"num_bits": 4, "block_sizes": {-1: 128}}, {"method": "gptq", "block_size": 128}),
        ({"num_bits": "iq2_xxs", "block_sizes": {-1: 256}, "backend": "ggml"}, "imatrix"),
    ],
    ids=["gptq", "imatrix"],
)
def test_native_fp8_experts_reject_packed_weight_updates(cfg, algorithm):
    # GPTQ and imatrix write the experts' weights in place; packed FP8 experts only hand them
    # dequantized copies, so the update would be lost.
    torch.manual_seed(42)
    config = PretrainedConfig(
        hidden_size=256, intermediate_size=256, num_experts=2, hidden_act="silu"
    )
    experts = FP8Experts(config, block_size=None, has_gate=True)
    for name in ("gate_up_proj", "down_proj"):
        weight = getattr(experts, name)
        with torch.no_grad():
            weight.copy_(torch.randn(weight.shape).to(weight.dtype))
            getattr(experts, f"{name}_scale_inv").fill_(0.02)
    quant_cfg = [
        {"quantizer_name": "*", "enable": False},
        {"quantizer_name": "*weight_quantizer", "cfg": cfg},
    ]
    mtq.quantize(experts, {"quant_cfg": quant_cfg, "algorithm": "max"}, forward_loop=None)
    inputs = torch.randn(64, 256)
    indices = (torch.arange(64) % 2).unsqueeze(1)
    routing = torch.ones(64, 1)
    state = copy.deepcopy(experts.state_dict())

    with pytest.raises(ValueError, match=r"Call unpack_weight\(\)"):
        mtq.calibrate(experts, algorithm, forward_loop=lambda m: m(inputs, indices, routing))
    # Compare FP8 payloads in FP32 because CPU close/equality kernels do not support FP8.
    torch.testing.assert_close(
        {name: value.float() for name, value in experts.state_dict().items()},
        {name: value.float() for name, value in state.items()},
        rtol=0,
        atol=0,
    )

    experts.unpack_weight()
    unpacked = {
        name: getattr(experts, name).detach().clone() for name in ("gate_up_proj", "down_proj")
    }
    mtq.calibrate(experts, algorithm, forward_loop=lambda m: m(inputs, indices, routing))
    for name, weight in unpacked.items():
        assert not torch.equal(getattr(experts, name), weight)


# ---------------------------------------------------------------------------
# A class with its own wrapper keeps eager; the interception check spares unreached experts
# ---------------------------------------------------------------------------
def test_gpt_oss_calibrates_down_proj_input_on_the_eager_forward():
    """GPT-OSS's own wrapper quantizes down_proj's input by intercepting the eager forward's
    bmm/matmul, which the grouped path never calls: the model keeps main's forced eager."""
    transformers = pytest.importorskip("transformers")
    config = transformers.GptOssConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_local_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=64,
        layer_types=["full_attention"],
    )
    torch.manual_seed(0)
    model = transformers.GptOssForCausalLM(config).eval()

    mtq.quantize(model, mtq.FP8_DEFAULT_CFG, lambda m: m(torch.randint(0, 64, (2, 8))))

    assert model.config._experts_implementation == "eager"
    assert model.model.layers[0].mlp.experts.down_proj_input_quantizer.amax is not None


_INT8_EXPERTS_CFG = {
    "quant_cfg": [
        {"quantizer_name": "*", "enable": False},
        {"quantizer_name": "*proj_input_quantizer", "cfg": {"num_bits": 8, "axis": None}},
        {"quantizer_name": "*proj_weight_quantizer", "cfg": {"num_bits": 8, "axis": 0}},
    ],
    "algorithm": "max",
}


class _TextAndVisionMoEModel(nn.Module):
    """Two MoE blocks, as Ernie-4.5-VL has: text-only calibration never reaches the vision one."""

    def __init__(self):
        super().__init__()
        self.text_moe = _SyntheticSparseMoeBlock()
        self.vision_moe = _SyntheticSparseMoeBlock()

    def forward(self, x):
        return self.text_moe(x)


class _BypassingExperts(_SyntheticFusedExperts):
    """A hand-written forward that reaches neither ``F.linear`` nor ``_grouped_linear``."""

    def forward(self, hidden_states, top_k_index, top_k_weights):
        return torch.zeros_like(hidden_states)


class TestFusedExpertsInterceptionCheck:
    def test_experts_no_calibration_token_reached_are_not_an_error(self):
        model = _TextAndVisionMoEModel()
        TestFusedExpertsCalibration._cleanup_registry(_SyntheticFusedExperts)
        try:
            mtq.quantize(model, _INT8_EXPERTS_CFG, lambda m: m(torch.randn(2, 4, HIDDEN_DIM)))
        finally:
            TestFusedExpertsCalibration._cleanup_registry(_SyntheticFusedExperts)

        assert model.text_moe.experts.gate_up_proj_input_quantizer.amax is not None
        assert model.vision_moe.experts.gate_up_proj_input_quantizer.amax is None

    def test_experts_whose_forward_ran_past_both_hooks_still_fail_loudly(self):
        model = _TinyMoEModel()
        model.moe.experts = _BypassingExperts()
        try:
            with pytest.raises(RuntimeError, match="collected nothing"):
                mtq.quantize(model, _INT8_EXPERTS_CFG, lambda m: m(torch.randn(2, 4, HIDDEN_DIM)))
        finally:
            TestFusedExpertsCalibration._cleanup_registry(_BypassingExperts)


# ---------------------------------------------------------------------------
# Fused experts under expert parallelism
# ---------------------------------------------------------------------------
_EP_EXPERTS, _EP_HIDDEN = 4, 8


class _ExpertsHolder(nn.Module):
    def __init__(self, experts):
        super().__init__()
        self.experts = experts

    def forward(self, *args):
        return self.experts(*args)


_WEIGHT_ONLY_INT8_EXPERTS_CFG = {
    "quant_cfg": [
        {"quantizer_name": "*", "enable": False},
        {"quantizer_name": "*proj_weight_quantizer", "cfg": {"num_bits": 8, "axis": 0}},
    ],
    "algorithm": "max",
}


def _ep_experts_config():
    return Qwen3MoeConfig(
        hidden_size=_EP_HIDDEN,
        moe_intermediate_size=2 * _EP_HIDDEN,
        intermediate_size=4 * _EP_HIDDEN,
        num_experts=_EP_EXPERTS,
        num_experts_per_tok=TOP_K,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=32,
    )


def _experts_with_distinct_scales(config):
    """Whole fused experts with a distinct weight scale per expert, so a mix-up cannot pass."""
    experts = Qwen3MoeExperts(config)
    with torch.no_grad():
        for e in range(_EP_EXPERTS):
            experts.gate_up_proj[e].normal_(std=0.1 * (e + 1))
            experts.down_proj[e].normal_(std=0.1 * (e + 1))
    return experts


def _sharded_like_transformers(whole, config, mesh, size):
    """``whole``'s experts as transformers' expert parallelism shards them (``MoEParamShard``): a
    DTensor ``Shard(0)`` over the experts, ``num_experts`` set to the local count."""
    experts = Qwen3MoeExperts(config)
    experts.num_experts = _EP_EXPERTS // size
    for name in ("gate_up_proj", "down_proj"):
        full = getattr(whole, name).detach().clone()
        experts._parameters[name] = nn.Parameter(distribute_tensor(full, mesh, [Shard(0)]))
    return experts


def _ep_sharded_experts_use_their_own_weights_job(rank, size, layout):
    """Each rank's per-expert quantizers must calibrate on, and fold into, the rank's own experts, as
    in the whole module -- not on the global experts their local index names. ``fsdp2-dense`` is the
    dense-sharded expert-parallel layout: the decoder layer FSDP2-sharded over ``dp`` with its
    experts left out (``ignored_params``), so weight calibration opens FSDP2's weight-access window
    on the layer, which must leave the expert DTensors alone -- gathering them would hand every rank
    every expert, indexed by local position."""
    config = _ep_experts_config()
    whole = _ExpertsHolder(_experts_with_distinct_scales(config))
    n_local = _EP_EXPERTS // size
    owned = slice(rank * n_local, (rank + 1) * n_local)
    ep_mesh = init_device_mesh("cpu", (size,), mesh_dim_names=("ep",))
    experts = _sharded_like_transformers(whole.experts, config, ep_mesh, size)
    if layout == "fsdp2-dense":
        layer = nn.Module()
        layer.dense = nn.Linear(_EP_HIDDEN, _EP_HIDDEN)
        layer.experts = experts
        dp_mesh = init_device_mesh("cpu", (size,), mesh_dim_names=("dp",))
        fully_shard(layer, mesh=dp_mesh, ignored_params=set(experts.parameters()))
        model = nn.Module()
        model.layer = layer
    else:
        model = _ExpertsHolder(experts)

    mtq.quantize(whole, _WEIGHT_ONLY_INT8_EXPERTS_CFG)
    mtq.quantize(model, _WEIGHT_ONLY_INT8_EXPERTS_CFG)
    sharded = model.layer.experts if layout == "fsdp2-dense" else model.experts
    for name in ("gate_up_proj_weight_quantizers", "down_proj_weight_quantizers"):
        local_quantizers = getattr(sharded, name)
        assert len(local_quantizers) == n_local
        for i, quantizer in enumerate(local_quantizers):
            reference = getattr(whole.experts, name)[owned.start + i]
            torch.testing.assert_close(
                quantizer.amax, reference.amax, atol=0, rtol=0, msg=f"{name}[{i}]"
            )
    # The expert weights stay sharded: nothing gathered them for good.
    assert sharded.gate_up_proj.to_local().shape[0] == n_local

    if layout == "plain":
        whole.experts.fold_weight()
        sharded.fold_weight()
        for name in ("gate_up_proj", "down_proj"):
            torch.testing.assert_close(
                getattr(sharded, name).to_local(),
                getattr(whole.experts, name)[owned],
                atol=0,
                rtol=0,
            )


@pytest.mark.parametrize("layout", ["plain", "fsdp2-dense"])
def test_ep_sharded_experts_calibrate_and_fold_their_own_weights(cpu_dist_workers, layout):
    """Weight calibration and folding run outside the forward, where expert-parallel weights are still
    DTensors sharded over the experts: each rank must use its own experts, not the global ones. With
    the dense part FSDP2-sharded, the calibration runs inside FSDP2's weight-access window, which
    must not gather the expert-parallel weights FSDP2 does not own."""
    cpu_dist_workers.run(_ep_sharded_experts_use_their_own_weights_job, layout=layout)


def _ep_weight_input_hooks_read_local_experts_job(rank, size):
    """Registering the weight-input hooks (local Hessian, imatrix) hands ``on_weight`` each
    projection's first LOCAL expert, the one its first quantizer quantizes, and runs no collective:
    indexing the expert-parallel DTensor would name a global expert and all-gather every expert."""
    config = _ep_experts_config()
    torch.manual_seed(0)
    whole = _experts_with_distinct_scales(config)
    mesh = init_device_mesh("cpu", (size,))
    model = _ExpertsHolder(_sharded_like_transformers(whole, config, mesh, size))
    n_local = _EP_EXPERTS // size
    mtq.quantize(model, _WEIGHT_ONLY_INT8_EXPERTS_CFG)

    seen = {}
    with CommDebugMode() as comms:
        handles = _register_weight_input_hooks(
            model,
            module_name_maps(model),
            lambda *args: None,
            on_weight=lambda name, weight, quantizer: seen.__setitem__(name, (weight, quantizer)),
        )
    for handle in handles:
        handle.remove()
    assert comms.get_total_counts() == 0, dict(comms.get_comm_counts())
    for name in ("gate_up_proj", "down_proj"):
        weight, quantizer = seen[f"experts.{name}"]
        assert quantizer is getattr(model.experts, f"{name}_weight_quantizers")[0]
        local_first = getattr(whole, name)[rank * n_local].detach()
        torch.testing.assert_close(weight, local_first, atol=0, rtol=0, msg=name)


def test_ep_weight_input_hooks_read_this_ranks_experts_without_a_collective(cpu_dist_workers):
    """Hook registration only needs one expert's shape, so it must not gather other ranks' experts."""
    cpu_dist_workers.run(_ep_weight_input_hooks_read_local_experts_job)


def _ep_gptq_helpers_take_local_experts_job(rank, size):
    """GPTQ builds each expert's helper from this rank's expert, the one its quantizer quantizes:
    indexing the expert-parallel DTensor would name a global expert and all-gather every expert."""
    config = _ep_experts_config()
    torch.manual_seed(0)
    whole = _experts_with_distinct_scales(config)
    mesh = init_device_mesh("cpu", (size,))
    model = _ExpertsHolder(_sharded_like_transformers(whole, config, mesh, size))
    n_local = _EP_EXPERTS // size
    built = {}
    init = GPTQHelper.__init__

    def recording(helper, module, name, *args, weight=None, **kwargs):
        if weight is not None:
            built[name] = weight if isinstance(weight, DTensor) else weight.detach().clone()
        init(helper, module, name, *args, weight=weight, **kwargs)

    cfg = {**_WEIGHT_ONLY_INT8_EXPERTS_CFG, "algorithm": {"method": "gptq", "block_size": 8}}
    with mock.patch.object(GPTQHelper, "__init__", recording):
        mtq.quantize(model, cfg, forward_loop=lambda m: None)

    assert len(built) == 2 * n_local, sorted(built)
    for name, weight in built.items():
        _, proj, idx = name.rsplit(".", 2)
        assert not isinstance(weight, DTensor), name
        expected = getattr(whole, proj)[rank * n_local + int(idx)].detach()
        torch.testing.assert_close(weight, expected, atol=0, rtol=0, msg=name)


def test_ep_gptq_builds_its_expert_helpers_from_this_ranks_experts(cpu_dist_workers):
    """A GPTQ helper only needs its expert's shape, so it must not gather other ranks' experts."""
    cpu_dist_workers.run(_ep_gptq_helpers_take_local_experts_job)
