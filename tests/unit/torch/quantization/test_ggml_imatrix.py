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

"""imatrix calibration for the GGML formats."""

import copy
import dataclasses

import pytest
import torch
import torch.nn.functional as F
from _test_utils.torch.transformers_models import get_tiny_llama
from torch import nn

import modelopt.torch.quantization as mtq
from modelopt.recipe import load_recipe
from modelopt.torch.quantization import model_calib
from modelopt.torch.quantization.ggml import GGML_FORMAT_REGISTRY
from modelopt.torch.quantization.ggml.common import pinned_packed_weight
from modelopt.torch.quantization.nn import QuantModuleRegistry

IQ2_XXS = GGML_FORMAT_REGISTRY["iq2_xxs"]
IMATRIX = {"method": "imatrix"}
GPTQ = {"method": "gptq", "block_size": 256, "perc_damp": 0.3}


def _config(algorithm, pattern="*weight_quantizer"):
    return {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": pattern,
                "cfg": {"num_bits": "iq2_xxs", "block_sizes": {-1: 256}, "backend": "ggml"},
                "enable": True,
            },
        ],
        "algorithm": algorithm,
    }


def _inputs(rows=1024, columns=512):
    # Columns of very different scale, so weighting the search by them matters.
    generator = torch.Generator().manual_seed(0)
    scale = torch.randn(columns, generator=generator).mul(1.5).exp()
    return torch.randn(rows, columns, generator=generator) * scale


def _importance(second_moments):
    return second_moments.clamp_min(1e-4 * second_moments.mean()).sqrt()


def _record_importance(monkeypatch):
    calls = []

    def recording_quantize(weight, **kwargs):
        calls.append(kwargs.get("importance"))
        return IQ2_XXS.quantize(weight, **kwargs)

    monkeypatch.setitem(
        GGML_FORMAT_REGISTRY, "iq2_xxs", dataclasses.replace(IQ2_XXS, quantize=recording_quantize)
    )
    return calls


def _decoded(quantizer, shape):
    pinned = pinned_packed_weight(quantizer, "iq2_xxs")
    assert pinned is not None
    return IQ2_XXS.dequantize(pinned, torch.tensor(shape), dtype=torch.float32)


def test_imatrix_weights_the_search_by_the_root_of_the_inputs_second_moments(monkeypatch):
    calls = _record_importance(monkeypatch)
    inputs = _inputs(rows=2500)  # more rows than the hook squares at a time
    mtq.quantize(nn.Linear(512, 8, bias=False), _config(IMATRIX), forward_loop=lambda m: m(inputs))

    torch.testing.assert_close(calls[-1], _importance(inputs.square().mean(dim=0)))


def test_imatrix_pins_the_codes_it_chose():
    torch.manual_seed(0)
    model = nn.Linear(512, 8, bias=False)
    original = model.weight.detach().clone()
    plain = copy.deepcopy(model)
    inputs = _inputs()
    mtq.quantize(model, _config(IMATRIX), forward_loop=lambda m: m(inputs))
    mtq.quantize(plain, _config("max"), forward_loop=lambda m: m(inputs))

    decoded = _decoded(model.weight_quantizer, model.weight.shape)
    assert torch.equal(model.weight.detach(), decoded)
    torch.testing.assert_close(model(inputs), inputs @ decoded.T)

    def output_error(weight):
        return ((weight - original) @ inputs.T).square().sum()

    assert output_error(decoded) < output_error(plain.weight_quantizer(plain.weight))


def test_imatrix_leaves_codes_gptq_chose():
    torch.manual_seed(0)
    gptq_only = nn.Linear(512, 8, bias=False)
    both = copy.deepcopy(gptq_only)
    inputs = _inputs()
    mtq.quantize(gptq_only, _config(GPTQ), forward_loop=lambda m: m(inputs))
    mtq.quantize(both, _config([GPTQ, IMATRIX]), forward_loop=lambda m: m(inputs))

    assert torch.equal(
        pinned_packed_weight(both.weight_quantizer, "iq2_xxs"),
        pinned_packed_weight(gptq_only.weight_quantizer, "iq2_xxs"),
    )
    assert torch.equal(both.weight, gptq_only.weight)


@pytest.mark.parametrize("calibrate_inputs", [False, True])
def test_imatrix_refuses_a_width_it_cannot_encode_before_calibrating(calibrate_inputs):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(512, 640, bias=False), nn.Linear(640, 8, bias=False))
    original = copy.deepcopy(model)
    config = _config(IMATRIX)
    if calibrate_inputs:
        # max_calibrate then runs a forward of its own, which must not run either.
        config["quant_cfg"].append(
            {"quantizer_name": "*input_quantizer", "cfg": {"num_bits": (4, 3)}, "enable": True}
        )
    inputs = _inputs(rows=64)
    forwards = []

    def forward_loop(model):
        forwards.append(model)
        model(inputs)

    with pytest.raises(
        ValueError, match=r"imatrix cannot encode a weight of 1: .*divisible by 256"
    ):
        mtq.quantize(model, config, forward_loop=forward_loop)

    assert not forwards
    for module, before in zip(model, original):
        assert torch.equal(module.weight, before.weight)
        assert module.input_quantizer.amax is None


def test_imatrix_warns_that_ranks_are_not_synced(monkeypatch):
    monkeypatch.setattr(model_calib, "dist_is_initialized", lambda: True)
    monkeypatch.setattr(model_calib, "dist_size", lambda: 2)
    inputs = _inputs()
    with pytest.warns(UserWarning, match="not synced across ranks"):
        mtq.quantize(nn.Linear(512, 8, bias=False), _config(IMATRIX), lambda m: m(inputs))


def test_a_module_calibration_never_reached_gets_the_plain_search():
    torch.manual_seed(0)
    model = nn.ModuleDict(
        {"used": nn.Linear(512, 8, bias=False), "unused": nn.Linear(512, 8, bias=False)}
    )
    original = model["unused"].weight.detach().clone()
    inputs = _inputs()
    mtq.quantize(model, _config(IMATRIX), forward_loop=lambda m: m["used"](inputs))

    packed, shape = IQ2_XXS.quantize(original)
    plain = IQ2_XXS.dequantize(packed, shape, dtype=torch.float32)
    assert torch.equal(model["unused"].weight.detach(), plain)
    assert torch.equal(_decoded(model["unused"].weight_quantizer, original.shape), plain)


NUM_EXPERTS = 4
HIDDEN = 256
INTERMEDIATE = 256


class _FusedExperts(nn.Module):
    """HF transformers 5 fused-experts layout: one ``F.linear`` pair per routed expert."""

    def __init__(self):
        super().__init__()
        self.num_experts = NUM_EXPERTS
        self.gate_up_proj = nn.Parameter(torch.randn(NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN) * 0.02)
        self.down_proj = nn.Parameter(torch.randn(NUM_EXPERTS, HIDDEN, INTERMEDIATE) * 0.02)

    def forward(self, hidden_states, expert_of_token):
        output = torch.zeros_like(hidden_states)
        for idx in range(self.num_experts):
            tokens = (expert_of_token == idx).nonzero().squeeze(-1)
            if len(tokens) == 0:
                continue
            gate, up = F.linear(hidden_states[tokens], self.gate_up_proj[idx]).chunk(2, dim=-1)
            output[tokens] = F.linear(F.silu(gate) * up, self.down_proj[idx])
        return output


class _MoEBlock(nn.Module):
    """Routes token ``t`` to expert ``t % 3``, so the last expert receives no token."""

    def __init__(self):
        super().__init__()
        self.experts = _FusedExperts()
        self.shared_expert = nn.Linear(HIDDEN, HIDDEN, bias=False)

    def forward(self, hidden_states):
        routed = self.experts(hidden_states, torch.arange(len(hidden_states)) % 3)
        return routed + self.shared_expert(hidden_states)


class _MoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _MoEBlock()

    def forward(self, hidden_states):
        return self.mlp(hidden_states)


@pytest.fixture
def moe():
    torch.manual_seed(0)
    model = _MoE()
    yield model
    if QuantModuleRegistry.get(_FusedExperts) is not None:
        QuantModuleRegistry.unregister(_FusedExperts)


def test_imatrix_weights_each_fused_expert_by_its_own_tokens(monkeypatch, moe):
    original = copy.deepcopy(moe.mlp)
    inputs = _inputs(rows=96, columns=HIDDEN)
    calls = _record_importance(monkeypatch)
    # The shipped experts-only recipe, which calibrates in a single pass.
    recipe = load_recipe("general/ptq/iq2_xxs_experts_only").quantize.model_dump()
    assert recipe["algorithm"]["method"] == "imatrix"
    mtq.quantize(moe, recipe, lambda m: m(inputs))

    # The shared expert stays in BF16.
    assert not moe.mlp.shared_expert.weight_quantizer.is_enabled
    assert torch.equal(moe.mlp.shared_expert.weight, original.shared_expert.weight)

    # One encode per expert and projection: gate_up for experts 0..3, then down for 0..3.
    gate_up_calls, down_calls = calls[-8:-4], calls[-4:]
    experts = moe.mlp.experts
    for idx in range(NUM_EXPERTS):
        tokens = inputs[idx::3] if idx < 3 else inputs[:0]
        if len(tokens) == 0:
            # The idle expert keeps the plain search, and its codes are still pinned.
            assert gate_up_calls[idx] is None
            assert down_calls[idx] is None
        else:
            gate, up = F.linear(tokens, original.experts.gate_up_proj[idx]).chunk(2, dim=-1)
            torch.testing.assert_close(gate_up_calls[idx], _importance(tokens.square().mean(dim=0)))
            torch.testing.assert_close(
                down_calls[idx], _importance((F.silu(gate) * up).square().mean(dim=0))
            )
        for weight, quantizers in (
            (experts.gate_up_proj, experts.gate_up_proj_weight_quantizers),
            (experts.down_proj, experts.down_proj_weight_quantizers),
        ):
            assert torch.equal(weight[idx].detach(), _decoded(quantizers[idx], weight[idx].shape))


@pytest.mark.parametrize("layerwise", [False, True])
def test_layerwise_imatrix_sees_the_earlier_layers_quantization(monkeypatch, layerwise):
    calls = _record_importance(monkeypatch)

    def run(layerwise):
        model = get_tiny_llama(
            dtype=torch.float32, hidden_size=256, intermediate_size=256, num_attention_heads=4
        )
        input_ids = torch.randint(0, 32, (4, 16), generator=torch.Generator().manual_seed(0))
        algorithm = {"method": "imatrix", "layerwise": {"enable": layerwise}}
        start = len(calls)
        mtq.quantize(model, _config(algorithm, "*mlp*weight_quantizer"), lambda m: m(input_ids))
        weighted = [c for c in calls[start:] if c is not None]
        assert len(weighted) == 6  # gate, up and down of both layers
        return weighted

    single_pass, other = run(False), run(layerwise)
    # Layer 0's MLP inputs come only from unquantized weights either way.
    for a, b in zip(single_pass[:3], other[:3]):
        torch.testing.assert_close(a, b)
    # Layer 1's do not when layerwise: they pass through layer 0's quantized MLP first.
    changed = any(not torch.allclose(a, b) for a, b in zip(single_pass[3:], other[3:]))
    assert changed == layerwise
