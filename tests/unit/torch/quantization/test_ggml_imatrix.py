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

"""imatrix and local_hessian calibration for the GGML formats."""

import copy
import dataclasses
from collections import defaultdict

import pytest
import torch
import torch.nn.functional as F
from _test_utils.torch.quantization.models import TOY_MOE_EXPERTS as NUM_EXPERTS
from _test_utils.torch.quantization.models import TOY_MOE_HIDDEN as HIDDEN
from _test_utils.torch.transformers_models import get_tiny_llama
from torch import nn

import modelopt.torch.quantization as mtq
from modelopt.recipe import load_recipe
from modelopt.torch.export.moe_utils import _export_fused_experts
from modelopt.torch.quantization import model_calib
from modelopt.torch.quantization.ggml import GGML_FORMAT_REGISTRY
from modelopt.torch.quantization.ggml.choice_search import search_choices
from modelopt.torch.quantization.ggml.common import pinned_importance, pinned_packed_weight

IQ2_XXS = GGML_FORMAT_REGISTRY["iq2_xxs"]
IMATRIX = {"method": "imatrix"}
LOCAL_HESSIAN = {"method": "local_hessian"}
# Both encode each weight once from a statistic of its inputs and pin the codes.
ENCODERS = pytest.mark.parametrize("algorithm", [IMATRIX, LOCAL_HESSIAN], ids=lambda a: a["method"])
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


def _floored(second_moments):
    return second_moments.clamp_min(1e-4 * second_moments.mean())


def _importance(second_moments):
    return _floored(second_moments).sqrt()


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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_imatrix_weights_the_search_by_the_root_of_the_inputs_second_moments(monkeypatch, dtype):
    calls = _record_importance(monkeypatch)
    inputs = _inputs(rows=2500).to(dtype)  # more rows than the hook squares at a time
    model = nn.Linear(512, 8, bias=False, dtype=dtype)
    mtq.quantize(model, _config(IMATRIX), forward_loop=lambda m: m(inputs))

    torch.testing.assert_close(calls[-1], _importance(inputs.float().square().mean(dim=0)))


@ENCODERS
def test_pins_the_codes_it_chose(algorithm):
    torch.manual_seed(0)
    model = nn.Linear(512, 8, bias=False)
    original = model.weight.detach().clone()
    plain = copy.deepcopy(model)
    inputs = _inputs()
    mtq.quantize(model, _config(algorithm), forward_loop=lambda m: m(inputs))
    mtq.quantize(plain, _config("max"), forward_loop=lambda m: m(inputs))

    decoded = _decoded(model.weight_quantizer, model.weight.shape)
    torch.testing.assert_close(model(inputs), inputs @ decoded.T)
    assert torch.equal(model.weight.detach(), original)

    def output_error(weight):
        return ((weight - original) @ inputs.T).square().sum()

    assert output_error(decoded) < output_error(plain.weight_quantizer(plain.weight))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float64])
def test_imatrix_matches_each_blocks_energy_and_encodes_again_to_its_codes(dtype):
    # The stored scales are raised after the search to keep each block's energy. Encoding the
    # weight again, as QAD does once it changes, searches at the scales the codes were searched at,
    # so it gives back the pinned codes rather than the smaller ones the raised scales would pick.
    torch.manual_seed(0)
    model = nn.Linear(512, 16, bias=False, dtype=dtype)
    original = model.weight.detach().clone()
    inputs = _inputs().to(dtype)
    mtq.quantize(model, _config(IMATRIX), forward_loop=lambda m: m(inputs))
    quantizer = model.weight_quantizer
    assert torch.equal(model.weight.detach(), original)

    second_moments = _floored(inputs.float().square().mean(dim=0)).reshape(2, 256)
    decoded = _decoded(quantizer, original.shape).reshape(16, 2, 256)
    energy = (second_moments * decoded.square()).sum(-1)
    target = (second_moments * original.float().reshape(16, 2, 256).square()).sum(-1)
    torch.testing.assert_close(energy, target, rtol=1e-2, atol=0)
    pinned = pinned_packed_weight(quantizer, "iq2_xxs")
    assert torch.equal(IQ2_XXS.pack(model.weight.detach(), quantizer), pinned)
    # Export packs the decoding pin_current_codes returns, whole as well as in rows, and the weight
    # it was searched for still packs to the same codes afterwards.
    decoding = IQ2_XXS.pin_current_codes(model.weight.detach(), quantizer)
    assert torch.equal(IQ2_XXS.pack(decoding, quantizer), pinned)
    assert torch.equal(IQ2_XXS.pack(model.weight.detach(), quantizer), pinned)


def test_imatrix_encodes_alike_under_a_statistic_near_the_float32_maximum():
    # Scaling every calibration input by a power of two scales the statistic, and so every weight
    # of the scale fit, the search and the energy match, alike, which changes none of them, even
    # where the statistic (2**118 here, about the largest its floor's mean accepts) is near the
    # float32 maximum.
    signs = torch.randn(8, 512, generator=torch.Generator().manual_seed(0)).sign()
    pins = []
    for scale in (1.0, 2.0**59):
        torch.manual_seed(0)
        model = nn.Linear(512, 16, bias=False)
        mtq.quantize(model, _config(IMATRIX), lambda m, scale=scale: m(signs * scale))
        pins.append(pinned_packed_weight(model.weight_quantizer, "iq2_xxs"))
    assert torch.equal(*pins)


@ENCODERS
def test_weight_only_calibration_runs_one_forward(algorithm):
    inputs = _inputs()
    forwards = []

    def forward_loop(model):
        forwards.append(model)
        model(inputs)

    mtq.quantize(nn.Linear(512, 8, bias=False), _config(algorithm), forward_loop=forward_loop)
    assert len(forwards) == 1


@ENCODERS
def test_leaves_codes_gptq_chose(algorithm):
    torch.manual_seed(0)
    gptq_only = nn.Linear(512, 8, bias=False)
    both = copy.deepcopy(gptq_only)
    inputs = _inputs()
    mtq.quantize(gptq_only, _config(GPTQ), forward_loop=lambda m: m(inputs))
    mtq.quantize(both, _config([GPTQ, algorithm]), forward_loop=lambda m: m(inputs))

    assert torch.equal(
        pinned_packed_weight(both.weight_quantizer, "iq2_xxs"),
        pinned_packed_weight(gptq_only.weight_quantizer, "iq2_xxs"),
    )
    assert torch.equal(both.weight, gptq_only.weight)


@ENCODERS
@pytest.mark.parametrize("calibrate_inputs", [False, True])
def test_refuses_a_width_it_cannot_encode_before_calibrating(algorithm, calibrate_inputs):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(512, 640, bias=False), nn.Linear(640, 8, bias=False))
    original = copy.deepcopy(model)
    config = _config(algorithm)
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
        ValueError, match=rf"{algorithm['method']} cannot encode a weight of 1: .*divisible by 256"
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


@ENCODERS
def test_a_module_calibration_never_reached_gets_the_plain_search(algorithm):
    torch.manual_seed(0)
    model = nn.ModuleDict(
        {"used": nn.Linear(512, 8, bias=False), "unused": nn.Linear(512, 8, bias=False)}
    )
    original = model["unused"].weight.detach().clone()
    inputs = _inputs()
    mtq.quantize(model, _config(algorithm), forward_loop=lambda m: m["used"](inputs))

    packed, shape = IQ2_XXS.plain_quantize(original)
    plain = IQ2_XXS.dequantize(packed, shape, dtype=torch.float32)
    assert torch.equal(_decoded(model["unused"].weight_quantizer, original.shape), plain)
    assert torch.equal(model["unused"].weight, original)


def test_imatrix_weights_each_fused_expert_by_its_own_tokens(monkeypatch, moe):
    original = copy.deepcopy(moe.mlp)
    inputs = _inputs(rows=96, columns=HIDDEN)
    searches = defaultdict(list)  # the importance of every search, by the weight it searched

    def recording_quantize(weight, **kwargs):
        searches[weight.data_ptr()].append(kwargs.get("importance"))
        return IQ2_XXS.quantize(weight, **kwargs)

    monkeypatch.setitem(
        GGML_FORMAT_REGISTRY, "iq2_xxs", dataclasses.replace(IQ2_XXS, quantize=recording_quantize)
    )
    # The shipped experts-only recipe, which calibrates in a single pass.
    recipe = load_recipe("general/ptq/iq2_xxs_experts_only").quantize.model_dump()
    assert recipe["algorithm"]["method"] == "imatrix"
    mtq.quantize(moe, recipe, lambda m: m(inputs))

    # The shared expert stays in BF16.
    assert not moe.mlp.shared_expert.weight_quantizer.is_enabled
    assert torch.equal(moe.mlp.shared_expert.weight, original.shared_expert.weight)

    experts = moe.mlp.experts
    for idx in range(NUM_EXPERTS):
        tokens = inputs[idx::3] if idx < 3 else inputs[:0]
        gate_up_importance = pinned_importance(
            experts.gate_up_proj_weight_quantizers[idx], "iq2_xxs"
        )
        down_importance = pinned_importance(experts.down_proj_weight_quantizers[idx], "iq2_xxs")
        if len(tokens) == 0:
            # The idle expert keeps the plain search, and its codes are still pinned.
            expected_gate_up = expected_down = None
        else:
            gate, up = F.linear(tokens, original.experts.gate_up_proj[idx]).chunk(2, dim=-1)
            expected_gate_up = _importance(tokens.square().mean(dim=0))
            expected_down = _importance((F.silu(gate) * up).square().mean(dim=0))
        # Every search of an expert's weight used its own tokens, as did the importance pinned.
        for weight, importance, expected in (
            (experts.gate_up_proj[idx], gate_up_importance, expected_gate_up),
            (experts.down_proj[idx], down_importance, expected_down),
        ):
            used = searches[weight.data_ptr()]
            assert used
            for found in (*used, importance):
                if expected is None:
                    assert found is None
                else:
                    torch.testing.assert_close(found, expected)
        for weight, original_weight, quantizers in (
            (
                experts.gate_up_proj,
                original.experts.gate_up_proj,
                experts.gate_up_proj_weight_quantizers,
            ),
            (experts.down_proj, original.experts.down_proj, experts.down_proj_weight_quantizers),
        ):
            pinned = pinned_packed_weight(quantizers[idx], "iq2_xxs")
            assert torch.equal(weight[idx], original_weight[idx])
            assert torch.equal(IQ2_XXS.pack(weight[idx].detach().clone(), quantizers[idx]), pinned)


def test_hf_export_writes_each_fused_expert_s_pinned_rows(moe):
    # imatrix leaves the experts' weights unrounded, and export splits gate_up into gate and up.
    recipe = load_recipe("general/ptq/iq2_xxs_experts_only").quantize.model_dump()
    mtq.quantize(moe, recipe, lambda m: m(_inputs(rows=96, columns=HIDDEN)))
    experts = moe.mlp.experts
    pins = [
        tuple(
            pinned_packed_weight(quantizers[idx], "iq2_xxs").clone()
            for quantizers in (
                experts.gate_up_proj_weight_quantizers,
                experts.down_proj_weight_quantizers,
            )
        )
        for idx in range(NUM_EXPERTS)
    ]

    _export_fused_experts(experts, torch.bfloat16)

    for idx, (gate_up, down) in enumerate(pins):
        expert = getattr(experts, str(idx))
        gate, up = gate_up.chunk(2)
        assert torch.equal(expert.gate_proj.weight, gate)
        assert torch.equal(expert.up_proj.weight, up)
        assert torch.equal(expert.down_proj.weight, down)


def test_local_hessian_codes_hold_until_qad_changes_a_block():
    # Encoding again would re-pick the searched local scales, so the weight they were picked for
    # keeps its codes; a block QAD changed is re-encoded under the pinned scale, as after GPTQ.
    torch.manual_seed(0)
    model = nn.Linear(512, 16, bias=False)
    original = model.weight.detach().clone()
    inputs = _inputs() @ torch.randn(512, 512, generator=torch.Generator().manual_seed(1))
    mtq.quantize(model, _config(LOCAL_HESSIAN), forward_loop=lambda m: m(inputs))
    quantizer = model.weight_quantizer
    pinned = pinned_packed_weight(quantizer, "iq2_xxs").clone()
    assert torch.equal(model.weight, original)
    assert torch.equal(IQ2_XXS.pack(model.weight.detach().clone(), quantizer), pinned)

    with torch.no_grad():
        model.weight[3] += 1e-2
    packed = IQ2_XXS.pack(model.weight.detach().clone(), quantizer)
    rescanned, _ = IQ2_XXS.quantize(
        model.weight.detach(),
        scales=IQ2_XXS.payload_scales(pinned.reshape(-1, IQ2_XXS.block_bytes)),
        importance=pinned_importance(quantizer, "iq2_xxs"),
    )
    others = [row for row in range(16) if row != 3]
    assert torch.equal(packed[others], pinned[others])
    assert torch.equal(packed[3], rescanned[3])


def test_local_hessian_lowers_the_output_error_of_correlated_inputs():
    torch.manual_seed(0)
    imatrix = nn.Linear(512, 16, bias=False)
    original = imatrix.weight.detach().clone()
    local_hessian = copy.deepcopy(imatrix)
    # Mixed columns: imatrix's per-column statistic cannot see how they move together.
    inputs = _inputs() @ torch.randn(512, 512, generator=torch.Generator().manual_seed(1))
    mtq.quantize(imatrix, _config(IMATRIX), forward_loop=lambda m: m(inputs))
    mtq.quantize(local_hessian, _config(LOCAL_HESSIAN), forward_loop=lambda m: m(inputs))

    def output_error(model):
        quantized = model.weight_quantizer(model.weight).detach()
        return ((quantized - original) @ inputs.T).square().sum()

    assert output_error(local_hessian) < 0.95 * output_error(imatrix)


@pytest.mark.parametrize("num_bits", ["iq2_xxs", "iq1_s"])
def test_search_under_the_identity_hessian_keeps_the_encoders_own_choices(num_bits):
    # Under H = I, the limit heavy damping tends to, the output error is the squared error the
    # encoder already minimizes group by group.
    ggml_format = GGML_FORMAT_REGISTRY[num_bits]
    block = torch.randn(16, 256, generator=torch.Generator().manual_seed(0))
    choices = search_choices(ggml_format, block, torch.eye(256)[None])
    assert torch.equal(
        ggml_format.quantize(block, choices=choices)[0], ggml_format.quantize(block)[0]
    )


@pytest.mark.parametrize("layerwise", [False, True])
def test_layerwise_imatrix_sees_the_earlier_layers_quantization(layerwise):
    def run(layerwise):
        model = get_tiny_llama(
            dtype=torch.float32, hidden_size=256, intermediate_size=256, num_attention_heads=4
        )
        input_ids = torch.randint(0, 32, (4, 16), generator=torch.Generator().manual_seed(0))
        algorithm = {"method": "imatrix", "layerwise": {"enable": layerwise}}
        mtq.quantize(model, _config(algorithm, "*mlp*weight_quantizer"), lambda m: m(input_ids))
        # gate, up and down of both layers, in module order
        weighted = [
            pinned_importance(module.weight_quantizer, "iq2_xxs")
            for name, module in model.named_modules()
            if "mlp" in name and hasattr(module, "weight_quantizer")
        ]
        assert len(weighted) == 6 and all(w is not None for w in weighted)
        return weighted

    single_pass, other = run(False), run(layerwise)
    # Layer 0's MLP inputs come only from unquantized weights either way.
    for a, b in zip(single_pass[:3], other[:3]):
        torch.testing.assert_close(a, b)
    # Layer 1's do not when layerwise: they pass through layer 0's quantized MLP first.
    changed = any(not torch.allclose(a, b) for a, b in zip(single_pass[3:], other[3:]))
    assert changed == layerwise
