# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Tests for the ``_QuantDeepseekV4Experts`` plugin.

Covers registration, setup, forward correctness, and activation-quantizer
dispatch (first linear = gate_up, second = down)."""

import pytest
import torch

# The DeepSeek-V4 modeling code only exists on a specific PR branch of transformers,
# so skip the whole module if that import is unavailable in this env.
pytest.importorskip("transformers")
pytest.importorskip("transformers.models.deepseek_v4")

from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config  # noqa: E402
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Experts  # noqa: E402

# Trigger plugin registration
import modelopt.torch.quantization.plugins.huggingface  # noqa: F401, E402
from modelopt.torch.quantization.nn import QuantModuleRegistry, TensorQuantizer  # noqa: E402


def _disable_all_quantizers(module):
    for m in module.modules():
        if isinstance(m, TensorQuantizer):
            m.disable()


def _tiny_config():
    """A small config that keeps DeepseekV4Experts's tensors tiny and cheap."""
    return DeepseekV4Config(
        vocab_size=128,
        hidden_size=32,
        moe_intermediate_size=64,
        num_hidden_layers=2,
        num_nextn_predict_layers=0,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        hidden_act="silu",
        swiglu_limit=10.0,
        # Fields needed by the rest of the model we don't instantiate but config parses:
        index_n_heads=1,
        index_head_dim=8,
        index_topk=4,
        compress_ratios=[0, 0],
        compress_rope_theta=160000.0,
    )


def _make_experts():
    cfg = _tiny_config()
    experts = DeepseekV4Experts(cfg)
    # Populate weights with deterministic small values
    torch.manual_seed(0)
    torch.nn.init.uniform_(experts.gate_up_proj, -0.1, 0.1)
    torch.nn.init.uniform_(experts.down_proj, -0.1, 0.1)
    experts.eval()
    return experts


class TestRegistration:
    def test_deepseek_v4_experts_is_registered(self):
        """Importing the plugin module must register a wrapper for ``DeepseekV4Experts``."""
        assert QuantModuleRegistry.get(DeepseekV4Experts) is not None


class TestSetupAndForward:
    def test_converted_module_has_four_quantizers(self):
        """Conversion installs the input/weight quantizer pair for each projection."""
        experts = _make_experts()
        quant_experts = QuantModuleRegistry.convert(experts)
        for name in [
            "gate_up_proj_input_quantizer",
            "gate_up_proj_weight_quantizer",
            "down_proj_input_quantizer",
            "down_proj_weight_quantizer",
        ]:
            assert hasattr(quant_experts, name), f"missing {name}"

    def test_forward_preserves_shape_and_finiteness(self):
        """Forward must produce a finite output with the correct shape."""
        torch.manual_seed(42)
        quant_experts = QuantModuleRegistry.convert(_make_experts())

        num_tokens = 6
        top_k = 2
        hidden_states = torch.randn(num_tokens, quant_experts.hidden_dim)
        top_k_index = torch.randint(0, quant_experts.num_experts, (num_tokens, top_k))
        top_k_weights = torch.randn(num_tokens, top_k).softmax(dim=-1)

        with torch.no_grad():
            got = quant_experts.forward(hidden_states, top_k_index, top_k_weights)

        assert got.shape == hidden_states.shape
        assert torch.isfinite(got).all()

    def test_forward_is_identity_when_quantizers_disabled(self):
        """With quantizers explicitly disabled, the quantized module must be
        bit-identical to the un-quantized reference."""
        torch.manual_seed(42)
        experts = _make_experts()
        num_tokens = 6
        top_k = 2
        hidden_states = torch.randn(num_tokens, experts.hidden_dim)
        top_k_index = torch.randint(0, experts.num_experts, (num_tokens, top_k))
        top_k_weights = torch.randn(num_tokens, top_k).softmax(dim=-1)

        with torch.no_grad():
            ref = experts.forward(hidden_states, top_k_index, top_k_weights)

        quant_experts = QuantModuleRegistry.convert(_make_experts())
        _disable_all_quantizers(quant_experts)
        with torch.no_grad():
            got = quant_experts.forward(hidden_states, top_k_index, top_k_weights)

        assert got.shape == ref.shape == hidden_states.shape
        assert torch.equal(got, ref), (got - ref).abs().max().item()


class _Counter(torch.nn.Module):
    """Identity module that records the shape of each call."""

    def __init__(self):
        super().__init__()
        self.shapes = []

    def forward(self, x):
        self.shapes.append(tuple(x.shape))
        return x


class TestActivationQuantizerDispatch:
    """The hook should:
      * pre-quantize hidden_states once at forward entry via gate_up_proj_input_quantizer;
      * quantize every down_proj input via down_proj_input_quantizer;
      * leave any other F.linear calls under the forward's scope untouched.
    """

    def test_single_hit_expert(self):
        experts = QuantModuleRegistry.convert(_make_experts())
        _disable_all_quantizers(experts)

        gate_up_counter = _Counter()
        down_counter = _Counter()
        experts.gate_up_proj_input_quantizer = gate_up_counter
        experts.down_proj_input_quantizer = down_counter
        # Hardening: make sure the attribute swap actually took effect on the wrapper.
        assert experts.gate_up_proj_input_quantizer is gate_up_counter
        assert experts.down_proj_input_quantizer is down_counter

        num_tokens, top_k = 4, 1
        hidden_states = torch.randn(num_tokens, experts.hidden_dim)
        top_k_index = torch.zeros(num_tokens, top_k, dtype=torch.long)  # all route to expert 0
        top_k_weights = torch.ones(num_tokens, top_k)

        with torch.no_grad():
            experts.forward(hidden_states, top_k_index, top_k_weights)

        # gate_up is called exactly once at forward entry on the full hidden state.
        assert gate_up_counter.shapes == [(num_tokens, experts.hidden_dim)]
        # down is called once — the single hit expert sees all tokens.
        assert down_counter.shapes == [(num_tokens, experts.intermediate_dim)]

    def test_multiple_hit_experts_dispatched_by_weight_shape(self):
        """Two distinct experts hit -> two down_proj calls, one gate_up pre-quant."""
        experts = QuantModuleRegistry.convert(_make_experts())
        _disable_all_quantizers(experts)

        gate_up_counter = _Counter()
        down_counter = _Counter()
        experts.gate_up_proj_input_quantizer = gate_up_counter
        experts.down_proj_input_quantizer = down_counter

        num_tokens, top_k = 6, 1
        hidden_states = torch.randn(num_tokens, experts.hidden_dim)
        # First half of tokens -> expert 0, second half -> expert 1.
        top_k_index = torch.tensor(
            [[0]] * (num_tokens // 2) + [[1]] * (num_tokens - num_tokens // 2), dtype=torch.long
        )
        top_k_weights = torch.ones(num_tokens, top_k)

        with torch.no_grad():
            experts.forward(hidden_states, top_k_index, top_k_weights)

        # gate_up pre-quant happens exactly once on the full hidden state.
        assert gate_up_counter.shapes == [(num_tokens, experts.hidden_dim)]
        # down is called once per hit expert; each expert sees num_tokens/2 rows.
        assert len(down_counter.shapes) == 2
        assert all(s == (num_tokens // 2, experts.intermediate_dim) for s in down_counter.shapes), (
            down_counter.shapes
        )
