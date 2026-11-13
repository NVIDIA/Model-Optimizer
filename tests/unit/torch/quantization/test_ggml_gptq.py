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
"""GPTQ for GGML block formats: the group update, the helper, and the payload pin."""

import contextlib
import copy
import dataclasses
import io
import weakref
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from _test_utils.torch.quantization.models import TOY_MOE_EXPERTS, TOY_MOE_HIDDEN, ToyMoE

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.export.unified_export_megatron import GPTModelExporter
from modelopt.torch.opt.utils import named_hparams
from modelopt.torch.quantization import model_calib
from modelopt.torch.quantization.algorithms import QuantRecipeHparam, _AutoQuantizeBaseSearcher
from modelopt.torch.quantization.ggml import GGML_FORMAT_REGISTRY
from modelopt.torch.quantization.ggml.common import (
    PIN_OVERRIDES_WEIGHT,
    _row_keys,
    pin_packed_weight,
    pinned_importance,
    pinned_packed_weight,
)
from modelopt.torch.quantization.ggml.gptq import compensated_block_hessian, gptq_group_update
from modelopt.torch.quantization.nn.modules.tensor_quantizer import GroupedQuantizer
from modelopt.torch.quantization.utils.calib_utils import (
    compute_hessian_inverse,
    gptq_blockwise_update,
)
from modelopt.torch.quantization.utils.layerwise_calib import LayerActivationCollector


def _problem(rows=6, cols=16, seed=0):
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(rows, cols, generator=generator)
    inputs = torch.randn(64, cols, generator=generator) @ torch.randn(
        cols, cols, generator=generator
    )
    hessian = inputs.T @ inputs / inputs.shape[0]
    return weight, hessian, compute_hessian_inverse(hessian, weight, 0.01)


def _round_to_tenths(weight, columns=None):
    return torch.round(weight * 10) / 10


def _round_in_groups_of_4(group, columns=None):
    scale = group.abs().amax(dim=-1, keepdim=True) / 2
    return torch.round(group / scale) * scale


def test_single_column_groups_match_columnwise_gptq():
    weight, _, h_inv = _problem()
    expected = weight.clone()
    gptq_blockwise_update(expected, h_inv, 8, _round_to_tenths)

    gptq_group_update(weight, h_inv, 8, 1, _round_to_tenths)

    torch.testing.assert_close(weight, expected)


def test_group_update_folds_each_group_error_through_its_inverse_hessian_block():
    weight, _, h_inv = _problem()
    expected = weight.clone()
    for start in range(0, expected.shape[1], 4):
        group, rest = slice(start, start + 4), slice(start + 4, None)
        qdq = _round_in_groups_of_4(expected[:, group])
        err = (expected[:, group] - qdq) @ torch.linalg.inv(h_inv[group, group])
        expected[:, group] = qdq
        expected[:, rest] -= err @ h_inv[group, rest]

    gptq_group_update(weight, h_inv, 8, 4, _round_in_groups_of_4)

    torch.testing.assert_close(weight, expected, rtol=1e-4, atol=1e-5)


def test_compensated_block_hessian_is_the_schur_complement_of_the_remaining_columns():
    # Fixing a group while only the later columns compensate costs inv(inv(H_FF)_gg), with F the
    # group and every column after it; the plain damped H_gg would match only the last group.
    _, hessian, h_inv = _problem(cols=16)
    damped = hessian + 0.01 * hessian.diagonal().mean() * torch.eye(16)
    for start in (0, 4, 8, 12):
        remaining = slice(start, None)
        expected = torch.linalg.inv(torch.linalg.inv(damped[remaining, remaining])[:4, :4])
        torch.testing.assert_close(
            compensated_block_hessian(h_inv, slice(start, start + 4)),
            expected,
            rtol=1e-4,
            atol=1e-4,
        )


def test_group_update_rejects_blocks_that_split_a_group():
    weight, _, h_inv = _problem()

    with pytest.raises(ValueError, match="multiples of the quantization group size"):
        gptq_group_update(weight, h_inv, 6, 4, _round_in_groups_of_4)


IQ1_S = GGML_FORMAT_REGISTRY["iq1_s"]
IQ2_XXS = GGML_FORMAT_REGISTRY["iq2_xxs"]
GPTQ = {"method": "gptq", "block_size": 256, "perc_damp": 0.3}


def _iq_config(algorithm, num_bits="iq1_s"):
    return {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": "*weight_quantizer",
                "cfg": {
                    "num_bits": num_bits,
                    "block_sizes": {-1: GGML_FORMAT_REGISTRY[num_bits].block_size},
                    "backend": "ggml",
                },
                "enable": True,
            },
        ],
        "algorithm": algorithm,
    }


def _inputs():
    generator = torch.Generator().manual_seed(0)
    return torch.randn(128, 512, generator=generator) @ torch.randn(512, 512, generator=generator)


def _gptq_model(rows=8, num_bits="iq1_s", algorithm=GPTQ, dtype=torch.float32):
    torch.manual_seed(0)
    model = torch.nn.Linear(512, rows, bias=False, dtype=dtype)
    inputs = _inputs().to(dtype)
    mtq.quantize(model, _iq_config(algorithm, num_bits), forward_loop=lambda m: m(inputs))
    return model, inputs


def _pin(model):
    return pinned_packed_weight(model.weight_quantizer, "iq1_s")


def _decoded(packed, weight):
    return IQ1_S.dequantize(packed, torch.tensor(weight.shape), dtype=torch.float32)


def _q8_0_decoded(model):
    pinned = pinned_packed_weight(model.weight_quantizer, "q8_0")
    shape = torch.tensor(model.weight.shape)
    return GGML_FORMAT_REGISTRY["q8_0"].dequantize(pinned, shape, dtype=model.weight.dtype)


def test_gptq_on_a_ggml_format_pins_the_payload_it_chose():
    torch.manual_seed(0)
    original = torch.nn.Linear(512, 8, bias=False)
    plain = copy.deepcopy(original)
    inputs = _inputs()
    mtq.quantize(plain, _iq_config("max"), forward_loop=lambda m: m(inputs))
    model, _ = _gptq_model()

    pinned = _pin(model)
    decoded = _decoded(pinned, model.weight)
    assert pinned.shape == (8, 2, IQ1_S.block_bytes)
    torch.testing.assert_close(model(inputs), inputs @ decoded.T)
    assert torch.equal(IQ1_S.pack(model.weight, model.weight_quantizer), pinned)

    def weighted_error(weight):
        return ((weight - original.weight) @ inputs.T).square().sum()

    assert weighted_error(decoded) < weighted_error(plain.weight_quantizer(plain.weight))


@pytest.mark.parametrize(
    ("num_bits", "importance_weighted", "weighted"),
    [
        ("iq2_xxs", True, True),
        ("iq1_s", True, True),
        ("iq2_xxs", False, False),
        ("q8_0", True, False),
    ],
)
def test_gptq_weights_the_search_by_the_root_of_the_hessian_diagonal(
    monkeypatch, num_bits, importance_weighted, weighted
):
    ggml_format = GGML_FORMAT_REGISTRY[num_bits]
    calls = []

    def recording_quantize(weight, **kwargs):
        calls.append(kwargs.get("importance"))
        return ggml_format.quantize(weight, **kwargs)

    monkeypatch.setitem(
        GGML_FORMAT_REGISTRY,
        num_bits,
        dataclasses.replace(ggml_format, quantize=recording_quantize),
    )
    model = torch.nn.Linear(512, 8, bias=False)
    inputs = _inputs()
    algorithm = {**GPTQ, "importance_weighted": importance_weighted}
    mtq.quantize(model, _iq_config(algorithm, num_bits), forward_loop=lambda m: m(inputs))

    gptq_calls = calls[-(512 // ggml_format.block_size) :]  # one call per GGML column group
    if not weighted:
        assert gptq_calls == [None] * len(gptq_calls)
        return
    root = inputs.square().sum(dim=0).sqrt()
    importance = torch.cat(gptq_calls)
    torch.testing.assert_close(importance / importance.mean(), root / root.mean())


def test_local_hessian_scales_lower_the_output_error_of_correlated_inputs():
    torch.manual_seed(0)
    plain = torch.nn.Linear(512, 16, bias=False)
    original = plain.weight.detach().clone()
    searched = copy.deepcopy(plain)
    inputs = _inputs()
    algorithm = {**GPTQ, "importance_weighted": True}
    for model, local_hessian_scales in ((plain, False), (searched, True)):
        config = _iq_config({**algorithm, "local_hessian_scales": local_hessian_scales}, "iq2_xxs")
        mtq.quantize(model, config, forward_loop=lambda m: m(inputs))

    def output_error(model):
        return ((model.weight_quantizer(model.weight) - original) @ inputs.T).square().sum()

    assert output_error(searched) < 0.95 * output_error(plain)


def test_a_module_calibration_never_reached_gets_the_plain_search():
    # An MoE expert that no calibration token reaches keeps a zero Hessian. Weighting by its
    # diagonal would make every code score zero error, so GPTQ must fall back to the plain search.
    torch.manual_seed(0)
    model = torch.nn.ModuleDict(
        {"used": torch.nn.Linear(512, 8, bias=False), "unused": torch.nn.Linear(512, 8, bias=False)}
    )
    original = model["unused"].weight.detach().clone()
    inputs = _inputs()
    algorithm = {**GPTQ, "importance_weighted": True}
    mtq.quantize(model, _iq_config(algorithm, "iq2_xxs"), forward_loop=lambda m: m["used"](inputs))

    packed, shape = IQ2_XXS.quantize(original)
    plain = IQ2_XXS.dequantize(packed, shape, dtype=original.dtype)
    pinned = pinned_packed_weight(model["unused"].weight_quantizer, "iq2_xxs")
    decoded = IQ2_XXS.dequantize(pinned, shape, dtype=original.dtype)
    torch.testing.assert_close(decoded, plain, rtol=0, atol=0)


def _int4_config(algorithm):
    return {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": "*weight_quantizer",
                "cfg": {"num_bits": 4, "block_sizes": {-1: 128}},
                "enable": True,
            },
        ],
        "algorithm": algorithm,
    }


# iq2_xxs runs GGMLGPTQHelper and INT4 the base GPTQHelper. At this width a GGML weight is one
# 256-column group, which GGML GPTQ encodes without the Hessian, so only INT4 checks the routing.
@pytest.mark.parametrize("num_bits", ["iq2_xxs", 4])
def test_gptq_quantizes_each_fused_expert_as_a_linear_fed_only_its_tokens(moe, num_bits):
    # The expert no token reaches is compared against a linear calibration never reaches.
    experts = moe.mlp.experts
    original = copy.deepcopy(experts)
    inputs = _inputs()[:96, :TOY_MOE_HIDDEN]
    config = _iq_config(GPTQ, num_bits) if num_bits == "iq2_xxs" else _int4_config(GPTQ)
    mtq.quantize(moe, config, forward_loop=lambda m: m(inputs))

    for idx in range(TOY_MOE_EXPERTS):
        tokens = inputs[idx::3] if idx < 3 else inputs[:0]
        gate, up = F.linear(tokens, original.gate_up_proj[idx]).chunk(2, dim=-1)
        for name, expert_inputs in (("gate_up_proj", tokens), ("down_proj", F.silu(gate) * up)):
            reference = torch.nn.Linear(*getattr(original, name)[idx].shape[::-1], bias=False)
            reference.weight.data.copy_(getattr(original, name)[idx])
            mtq.quantize(reference, config, forward_loop=lambda m, x=expert_inputs: m(x))

            assert torch.equal(getattr(experts, name)[idx], reference.weight)
            if num_bits == "iq2_xxs":
                quantizer = getattr(experts, f"{name}_weight_quantizers")[idx]
                assert torch.equal(
                    pinned_packed_weight(quantizer, "iq2_xxs"),
                    pinned_packed_weight(reference.weight_quantizer, "iq2_xxs"),
                )


class _MoEStack(torch.nn.Module):
    """Toy MoE decoder layers under ``layers``."""

    def __init__(self, *layers):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


def test_layerwise_gptq_calibrates_fused_experts_on_the_earlier_layers_output(monkeypatch, moe):
    monkeypatch.setattr(
        LayerActivationCollector,
        "_decoder_layer_support",
        [(lambda m: isinstance(m, _MoEStack), lambda m: m.layers)],
    )
    stack = _MoEStack(moe, ToyMoE())
    inputs = _inputs()[:96, :TOY_MOE_HIDDEN]

    def run(layerwise):
        model = copy.deepcopy(stack)
        algorithm = {**GPTQ, "layerwise": {"enable": layerwise}}
        # INT4: one GGML group per weight here would ignore the Hessian, and with it the inputs.
        mtq.quantize(model, _int4_config(algorithm), forward_loop=lambda m: m(inputs))
        return [layer.mlp.experts for layer in model.layers]

    def same(a, b):
        return all(torch.equal(getattr(a, n), getattr(b, n)) for n in ("gate_up_proj", "down_proj"))

    single_pass, layerwise = run(False), run(True)
    for before, after in zip(stack.layers, layerwise):
        assert not same(before.mlp.experts, after)  # GPTQ wrote every layer's experts
    # Layer 0's experts see unquantized inputs either way.
    assert same(single_pass[0], layerwise[0])
    # Layer 1's see layer 0's quantized experts only when layerwise.
    assert not same(single_pass[1], layerwise[1])


def test_gptq_drops_each_fused_experts_weights_before_the_next_window(monkeypatch, moe):
    """No expert weight GPTQ updates outlives its module's weight window: under offload it would
    keep that module's onloaded weights on the GPU while the next module's are onloaded."""
    updated = []
    update_expert_weights = model_calib.update_expert_weights
    window = model_calib.enable_weight_access_and_writeback

    def recording_update_expert_weights(work, *args):
        # Fused experts' slices; a linear's weight is its module's own parameter.
        updated.extend(weakref.ref(w) for _, w in work if not isinstance(w, torch.nn.Parameter))
        update_expert_weights(work, *args)

    @contextlib.contextmanager
    def checked_window(*args, **kwargs):
        assert all(ref() is None for ref in updated)
        with window(*args, **kwargs):
            yield

    monkeypatch.setattr(model_calib, "update_expert_weights", recording_update_expert_weights)
    monkeypatch.setattr(model_calib, "enable_weight_access_and_writeback", checked_window)
    inputs = _inputs()[:96, :TOY_MOE_HIDDEN]
    mtq.quantize(_MoEStack(moe, ToyMoE()), _int4_config(GPTQ), forward_loop=lambda m: m(inputs))

    assert updated


@pytest.mark.parametrize("num_bits", ["iq1_s", "iq1_m", "iq2_xxs", "iq2_xs", "iq2_s"])
def test_a_format_that_keeps_its_scales_encodes_gptqs_weight_to_the_pin(num_bits):
    ggml_format = GGML_FORMAT_REGISTRY[num_bits]
    model, inputs = _gptq_model(num_bits=num_bits, algorithm={**GPTQ, "importance_weighted": True})
    quantizer = model.weight_quantizer
    pinned = pinned_packed_weight(quantizer, num_bits)
    decoded = ggml_format.dequantize(pinned, torch.tensor(model.weight.shape), dtype=torch.float32)

    # The weight is left before GPTQ's rounding, yet encodes to GPTQ's codes, as its decoding does.
    assert not torch.equal(model.weight.detach(), decoded)
    assert torch.equal(ggml_format.pack(model.weight, quantizer), pinned)
    assert torch.equal(ggml_format.pack(decoded, quantizer), pinned)
    torch.testing.assert_close(model(inputs), inputs @ decoded.T)

    # A QAD update re-encodes under the pinned scales and importance.
    with torch.no_grad():
        model.weight.add_(torch.randn_like(model.weight) * 0.01)
    scales = ggml_format.payload_scales(pinned.reshape(-1, ggml_format.block_bytes))
    expected, _ = ggml_format.quantize(
        model.weight.detach(),
        scales=scales,
        importance=pinned_importance(quantizer, num_bits),
    )
    packed = ggml_format.pack(model.weight, quantizer)
    assert torch.equal(packed, expected)
    assert torch.equal(
        ggml_format.payload_scales(packed.reshape(-1, ggml_format.block_bytes)), scales
    )


@pytest.mark.parametrize("num_bits", ["iq1_s", "iq1_m", "iq2_xxs", "iq2_xs", "iq2_s"])
def test_casting_the_model_keeps_the_importance_its_weight_encodes_to_the_pin_with(num_bits):
    # Enough rows that a bfloat16 importance changes some blocks' codes.
    model, _ = _gptq_model(
        rows=128,
        num_bits=num_bits,
        algorithm={**GPTQ, "importance_weighted": True},
        dtype=torch.bfloat16,
    )
    pinned = pinned_packed_weight(model.weight_quantizer, num_bits)

    # Megatron's Float16Module calls bfloat16() on the model, which casts its float buffers.
    model.bfloat16()

    assert torch.equal(
        GGML_FORMAT_REGISTRY[num_bits].pack(model.weight, model.weight_quantizer), pinned
    )


@pytest.mark.parametrize("cast", ["bfloat16", "half"])
def test_casting_the_model_keeps_the_local_scales_gptq_searched(cast):
    # GPTQ ran on a float32 weight; a cast after it rounds nearly every block of that weight, and
    # encoding the rounded weight afresh would re-pick the searched local scales.
    algorithm = {**GPTQ, "importance_weighted": True, "local_hessian_scales": True}
    model, _ = _gptq_model(rows=16, num_bits="iq2_xxs", algorithm=algorithm)
    pinned = pinned_packed_weight(model.weight_quantizer, "iq2_xxs")

    getattr(model, cast)()

    assert torch.equal(
        GGML_FORMAT_REGISTRY["iq2_xxs"].pack(model.weight, model.weight_quantizer), pinned
    )


@pytest.mark.parametrize("num_bits", ["iq1_s", "iq1_m", "iq2_xxs", "iq2_xs", "iq2_s"])
def test_a_checkpoint_from_before_kept_scales_still_encodes_to_its_pin(num_bits):
    # Earlier GPTQ left the weight as its decoding, rounded to the model's dtype, and no importance.
    ggml_format = GGML_FORMAT_REGISTRY[num_bits]
    model, _ = _gptq_model(rows=128, num_bits=num_bits, dtype=torch.bfloat16)
    quantizer = model.weight_quantizer
    pinned = pinned_packed_weight(quantizer, num_bits)
    pin_packed_weight(quantizer, num_bits, pinned)
    shape = torch.tensor(model.weight.shape)
    with torch.no_grad():
        model.weight.copy_(ggml_format.dequantize(pinned, shape, dtype=torch.bfloat16))

    assert pinned_importance(quantizer, num_bits) is None
    assert torch.equal(ggml_format.pack(model.weight, quantizer), pinned)


def test_pin_outlives_the_weight_tensor():
    model, _ = _gptq_model(algorithm={**GPTQ, "importance_weighted": True})
    # Same values in a new tensor, as an offload round trip or a layerwise resume leaves them.
    model.weight.data = model.weight.data.clone()

    assert torch.equal(IQ1_S.pack(model.weight, model.weight_quantizer), _pin(model))
    # Encoding the GPTQ'd weight afresh, without GPTQ's importance, would not return its codes.
    assert not torch.equal(IQ1_S.quantize(model.weight.detach())[0], _pin(model))


def test_folding_a_bf16_weight_decodes_the_pinned_payload():
    """Folding fake-quantizes a float32 copy of the weight; it must still decode GPTQ's codes.

    GPTQ leaves a BF16 model its unrounded weight, and ``mtq.fold_weight`` and the vLLM fake-quant
    export hand the quantizer a float32 copy of it, which must encode to GPTQ's codes.
    """
    torch.manual_seed(0)
    model = torch.nn.Linear(512, 8, bias=False, dtype=torch.bfloat16)
    inputs = _inputs().to(torch.bfloat16)
    mtq.quantize(model, _iq_config(GPTQ), forward_loop=lambda m: m(inputs))
    decoded = _decoded(_pin(model), model.weight).to(torch.bfloat16)
    assert not torch.equal(model.weight, decoded)

    # The vLLM fake-quant export folds this way.
    assert torch.equal(model.weight_quantizer(model.weight.float()).to(torch.bfloat16), decoded)
    mtq.fold_weight(model)
    assert torch.equal(model.weight, decoded)
    # The folded weight is the decoding, so the payload is dropped with the other quantizer state.
    assert _pin(model) is None


# Q8_0 keeps no scales: its pin holds while the weight is the decoding. IQ2_XS keeps its pinned
# scales: it re-encodes the unrounded weight, so needs the importance too, local_hessian's
# searched choices need the source fingerprint, and imatrix's the scales its codes were searched at.
@pytest.mark.parametrize(
    ("num_bits", "algorithm"),
    [
        ("q8_0", {**GPTQ, "importance_weighted": True}),
        ("iq2_xs", {**GPTQ, "importance_weighted": True}),
        ("iq2_xxs", {**GPTQ, "importance_weighted": True}),
        ("iq2_xs", {"method": "local_hessian"}),
        ("iq2_xs", {"method": "imatrix"}),
    ],
    ids=["q8_0-gptq", "iq2_xs-gptq", "iq2_xxs-gptq", "iq2_xs-local_hessian", "iq2_xs-imatrix"],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_pin_survives_save_and_restore(num_bits, algorithm, dtype):
    model, inputs = _gptq_model(num_bits=num_bits, algorithm=algorithm, dtype=dtype)
    buffer = io.BytesIO()
    mto.save(model, buffer)
    buffer.seek(0)

    restored = mto.restore(torch.nn.Linear(512, 8, bias=False, dtype=dtype), buffer)

    ggml_format = GGML_FORMAT_REGISTRY[num_bits]
    pinned = pinned_packed_weight(model.weight_quantizer, num_bits)
    assert torch.equal(ggml_format.pack(restored.weight, restored.weight_quantizer), pinned)
    torch.testing.assert_close(restored(inputs), model(inputs))

    # Initialization uses the loaded PTQ result; no calibration data is supplied.
    expected = ggml_format.dequantize(pinned, torch.tensor(restored.weight.shape), dtype=dtype)
    output_before = restored(inputs).detach()
    mtq.initialize_qdq_weights(restored)
    torch.testing.assert_close(restored(inputs), output_before, rtol=0, atol=0)
    torch.testing.assert_close(restored.weight, expected, rtol=0, atol=0)
    assert restored.weight.dtype == dtype and restored.weight_quantizer.is_enabled
    assert torch.equal(pinned_packed_weight(restored.weight_quantizer, num_bits), pinned)


def test_pin_is_dropped_once_the_weight_changes():
    # Q8_0 keeps no scales, so GPTQ leaves its decoding and the pin holds only while it lasts.
    q8_0 = GGML_FORMAT_REGISTRY["q8_0"]
    model, _ = _gptq_model(num_bits="q8_0")
    assert torch.equal(model.weight.detach(), _q8_0_decoded(model))
    with torch.no_grad():
        model.weight.add_(0.01)

    assert torch.equal(
        q8_0.pack(model.weight, model.weight_quantizer), q8_0.quantize(model.weight.detach())[0]
    )


@pytest.mark.parametrize("restore", [False, True])
def test_pin_is_ignored_once_the_quantizer_changes_format(restore):
    model, inputs = _gptq_model()
    if restore:
        buffer = io.BytesIO()
        mto.save(model, buffer)
        buffer.seek(0)
        model = mto.restore(torch.nn.Linear(512, 8, bias=False), buffer)

    mtq.set_quantizer_by_cfg(model, _iq_config(None, "iq2_xxs")["quant_cfg"])

    weight = model.weight.detach()
    expected = IQ2_XXS.dequantize(*IQ2_XXS.plain_quantize(weight), dtype=torch.float32)
    torch.testing.assert_close(model(inputs), inputs @ expected.T)
    assert torch.equal(
        IQ2_XXS.pack(model.weight, model.weight_quantizer), IQ2_XXS.plain_quantize(weight)[0]
    )


def _megatron_exporter(dtype, num_bits):
    """A GPTModelExporter to ``dtype`` whose modules are quantized to ``num_bits``."""
    exporter = object.__new__(GPTModelExporter)
    exporter.dtype = dtype
    exporter._state_dict = {}
    exporter.exclude_modules = []
    exporter.layer_config_dict = {}
    exporter._get_quantization_format = lambda module: num_bits
    return exporter


def _pinned_export(rows, num_bits):
    """A GPTQ'd float32 weight, its quantizer, and its pinned payload."""
    model, _ = _gptq_model(rows, num_bits)
    pinned = pinned_packed_weight(model.weight_quantizer, num_bits)
    return model.weight.detach(), model.weight_quantizer, pinned


# Export dtypes of the float32 model. IQ2_XS keeps its pinned scales, so its weight is not the
# pinned decoding and rounding it to BF16 before encoding could change its codes; Q8_0 keeps none.
EXPORT_DTYPES = pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
EXPORT_FORMATS = pytest.mark.parametrize("num_bits", ["q8_0", "iq2_xs"])


@EXPORT_DTYPES
@EXPORT_FORMATS
def test_megatron_gated_mlp_export_writes_the_pinned_rows(dtype, num_bits):
    weight, quantizer, pinned = _pinned_export(8, num_bits)
    module = SimpleNamespace(
        config=SimpleNamespace(ffn_hidden_size=4), weight=weight, weight_quantizer=quantizer
    )
    exporter = _megatron_exporter(dtype, num_bits)

    exporter._gated_mlp_slicing(module, "mlp.")

    assert torch.equal(exporter._state_dict["mlp.gate_proj.weight"], pinned[:4])
    assert torch.equal(exporter._state_dict["mlp.up_proj.weight"], pinned[4:])


@EXPORT_DTYPES
@EXPORT_FORMATS
def test_megatron_grouped_mlp_export_writes_the_pinned_rows(dtype, num_bits):
    weight, quantizer, pinned = _pinned_export(8, num_bits)
    module = SimpleNamespace(
        num_gemms=1,
        weight0=weight,
        local_expert_indices=[0],
        state_dict=lambda: {"weight0": weight},
        weight_quantizer=GroupedQuantizer(quantizer),
    )
    exporter = _megatron_exporter(dtype, num_bits)

    exporter._grouped_mlp_slicing(
        module, "mlp.experts.{}", gate_proj_name="gate_proj", up_proj_name="up_proj"
    )

    assert torch.equal(exporter._state_dict["mlp.experts.0.gate_proj.weight"], pinned[:4])
    assert torch.equal(exporter._state_dict["mlp.experts.0.up_proj.weight"], pinned[4:])


@EXPORT_DTYPES
@EXPORT_FORMATS
def test_megatron_qkv_export_writes_the_pinned_rows(dtype, num_bits):
    # 4 query groups of [2 q heads, k, v], 4 rows per head: K and V are gathered from scattered
    # rows, which is where a lookup keyed on rounded projections went wrong.
    weight, quantizer, pinned = _pinned_export(64, num_bits)
    config = SimpleNamespace(
        hidden_size=512,
        num_query_groups=4,
        num_attention_heads=8,
        kv_channels=4,
        attention_output_gate=False,
    )
    exporter = _megatron_exporter(dtype, num_bits)

    module = SimpleNamespace(config=config, weight=weight, weight_quantizer=quantizer)
    exporter._qkv_slicing(module, "attn.")

    heads = pinned.view(4, 4, 4, *pinned.shape[-2:])  # [group, q q k v, row, block, byte]
    assert torch.equal(exporter._state_dict["attn.q_proj.weight"], heads[:, :2].flatten(0, 2))
    assert torch.equal(exporter._state_dict["attn.k_proj.weight"], heads[:, 2].flatten(0, 1))
    assert torch.equal(exporter._state_dict["attn.v_proj.weight"], heads[:, 3].flatten(0, 1))


@EXPORT_DTYPES
@EXPORT_FORMATS
def test_megatron_gated_delta_net_export_writes_the_pinned_rows(dtype, num_bits):
    weight, quantizer, pinned = _pinned_export(12, num_bits)
    module = SimpleNamespace(
        in_proj=SimpleNamespace(weight=weight, weight_quantizer=quantizer),
        in_proj_split_names=("query", "key", "value", "z", "beta", "alpha"),
        in_proj_split_sections=(2, 2, 2, 2, 2, 2),
    )
    exporter = _megatron_exporter(dtype, num_bits)

    exporter._gated_delta_net_slicing(module, "mixer.")

    assert torch.equal(exporter._state_dict["mixer.in_proj_qkv.weight"], pinned[:6])
    assert torch.equal(exporter._state_dict["mixer.in_proj_z.weight"], pinned[6:8])


@EXPORT_DTYPES
def test_row_keys_do_not_depend_on_the_batch_a_row_is_hashed_in(dtype):
    # Wide-range values make any floating-point key sensitive to how its reduction is ordered;
    # the key must still be identical for a row hashed in a full table, a gathered subset, or alone.
    generator = torch.Generator().manual_seed(0)
    table = (
        torch.randn(64, 512, generator=generator)
        * torch.randn(64, 512, generator=generator).mul(8).exp()
    ).to(dtype)
    index = torch.tensor([56, 3, 52, 17])

    keys = _row_keys(table)

    assert torch.equal(_row_keys(table[index]), keys[index])
    assert all(torch.equal(_row_keys(table[i : i + 1]), keys[i : i + 1]) for i in index.tolist())
    assert len(keys.unique()) == len(table)


def _recipe(algorithm, num_bits):
    """An AutoQuantize format: ``num_bits`` GGML weights calibrated with ``algorithm``."""
    config = _iq_config(algorithm, num_bits)
    config["quant_cfg"][1]["cfg"]["effective_bits"] = GGML_FORMAT_REGISTRY[num_bits].effective_bits
    return config, num_bits


def _search(model, inputs, formats, effective_bits):
    _, state = mtq.auto_quantize(
        model,
        constraints={"effective_bits": effective_bits},
        quantization_formats=formats,
        data_loader=[inputs],
        forward_step=lambda m, x: m(x),
        loss_func=lambda output, x: output.square().sum(),
        num_calib_steps=1,
        num_score_steps=1,
    )
    return state


def _two_layers():
    torch.manual_seed(0)
    return torch.nn.Sequential(
        torch.nn.Linear(512, 512, bias=False), torch.nn.Linear(512, 256, bias=False)
    )


def _raw_scores(state, recipe_name):
    return [
        stats["raw_scores"][[str(f).split("(")[0] for f in stats["formats"]].index(recipe_name)]
        for stats in state["candidate_stats"].values()
    ]


def test_auto_quantize_scores_other_candidates_on_the_source_weights_gptq_rewrote():
    inputs = _inputs()
    source = _two_layers()
    alone = _search(copy.deepcopy(source), inputs, [_recipe("max", "iq2_s")], 16.0)
    model = copy.deepcopy(source)
    state = _search(model, inputs, [_recipe(GPTQ, "iq2_xxs"), _recipe("max", "iq2_s")], 16.0)

    assert _raw_scores(state, "iq2_s") == pytest.approx(_raw_scores(alone, "iq2_s"))
    # 16 bits selects BF16 everywhere, so the source weights come back untouched.
    for selected, original in zip(model.parameters(), source.parameters()):
        assert torch.equal(selected, original)


@pytest.mark.parametrize(
    ("algorithm", "num_bits", "pinned", "replaces_int4"),
    [
        (GPTQ, "iq2_xxs", True, False),
        # The GGML entry replaces an earlier INT4 entry for the same quantizers.
        (GPTQ, "iq2_xxs", True, True),
        ({"method": "imatrix"}, "q8_0", False, False),  # imatrix leaves Q8_0 to plain PTQ
        # imatrix keeps the scales its codes were searched at, which the decoding must not use.
        ({"method": "imatrix"}, "iq2_xxs", True, False),
    ],
)
def test_auto_quantize_selects_the_codes_calibration_chose(
    algorithm, num_bits, pinned, replaces_int4
):
    inputs = _inputs()
    source = _two_layers()
    model = copy.deepcopy(source)
    config, name = _recipe(algorithm, num_bits)
    if replaces_int4:
        config["quant_cfg"].insert(1, _int4_config(algorithm)["quant_cfg"][1])
    # Just enough bits to quantize the larger first layer and keep the second in BF16.
    ggml_format = GGML_FORMAT_REGISTRY[num_bits]
    sizes = [layer.weight.numel() for layer in source]
    bits = (sizes[0] * ggml_format.effective_bits + sizes[1] * 16) / sum(sizes)
    _search(model, inputs, [(config, name)], bits)

    first, second = model
    assert first.weight_quantizer.num_bits == num_bits
    assert not second.weight_quantizer.is_enabled
    payload = pinned_packed_weight(first.weight_quantizer, num_bits)
    if pinned:
        decoded = ggml_format.dequantize(
            payload, torch.tensor(first.weight.shape), dtype=torch.float32
        )
        assert torch.equal(first.weight, decoded)
        assert not torch.equal(payload, ggml_format.quantize(source[0].weight.detach())[0])
        # The returned model runs, and exports, the codes the search scored.
        assert torch.equal(first.weight_quantizer(first.weight), decoded)
        assert torch.equal(ggml_format.pack(first.weight, first.weight_quantizer), payload)
    else:
        assert payload is None
        assert torch.equal(first.weight, source[0].weight)
    assert torch.equal(second.weight, source[1].weight)
    assert not any(
        getattr(module, PIN_OVERRIDES_WEIGHT, False)
        for hparam in _quant_recipe_hparams(model)
        for quantizers in hparam._all_quantizer_choices.values()
        for by_attr in quantizers.values()
        for module in by_attr.values()
    )


def _quant_recipe_hparams(model):
    return [h for _, h in named_hparams(model, unique=True) if isinstance(h, QuantRecipeHparam)]


@pytest.mark.parametrize(
    "q8_0",
    [None, "*down_proj_weight_quantizer", "*gate_up_proj_weight_quantizers.0"],
    ids=["down_proj_disabled", "down_proj_q8_0", "one_gate_up_expert_q8_0"],
)
def test_auto_quantize_snapshots_only_the_weights_calibration_rewrites(moe, monkeypatch, q8_0):
    # imatrix rewrites the IQ2_XXS gate_up_proj experts and leaves Q8_0 or disabled ones alone;
    # gate_up_proj is one parameter, so any IQ2_XXS expert puts all of it in the snapshot.
    snapshots = []
    snapshot = _AutoQuantizeBaseSearcher._snapshot_source_weights

    def record(self, *args):
        snapshots.append(snapshot(self, *args))
        return snapshots[-1]

    monkeypatch.setattr(_AutoQuantizeBaseSearcher, "_snapshot_source_weights", record)
    source = {n: weight.detach().clone() for n, weight in moe.named_parameters()}
    config, name = _recipe({"method": "imatrix"}, "iq2_xxs")
    config["quant_cfg"][1]["quantizer_name"] = "*gate_up_proj_weight_quantizer"
    if q8_0 is not None:
        q8_0_entry = _recipe({"method": "imatrix"}, "q8_0")[0]["quant_cfg"][1]
        config["quant_cfg"].append({**q8_0_entry, "quantizer_name": q8_0})
    _search(moe, _inputs()[:96, :TOY_MOE_HIDDEN], [(config, name)], 16.0)

    [source_weights] = snapshots
    assert [id(p) for p in source_weights or ()] == [id(moe.mlp.experts.gate_up_proj)]
    # BF16 wins at 16 bits, so the search hands back the source weights.
    weights = dict(moe.named_parameters())
    for weight_name, weight in source.items():
        assert torch.equal(weights[weight_name], weight), weight_name


@pytest.mark.parametrize(
    "case",
    [
        "non_ggml",
        "tied",
        "aliased",
        "fused_moe_configs",
        "rotated_gptq",
        "rotated_imatrix",
        "awq_gptq",
        "layerwise_checkpoint",
    ],
)
def test_auto_quantize_refusal_hands_back_the_unconverted_model(case, request):
    inputs = _inputs()
    source = _two_layers()
    # An INT4 weight quantizer cannot hold the codes GPTQ would choose.
    formats = [(_int4_config(GPTQ), "INT4_GPTQ")]
    if case == "tied":
        # GPTQ would rewrite the weight under both layers.
        source[1] = torch.nn.Linear(512, 512, bias=False)
        source[1].weight = source[0].weight
        formats = [_recipe(GPTQ, "iq2_xxs")]
    elif case == "fused_moe_configs":
        # Conversion moves these fused experts onto a backend their quantizers intercept, on the
        # root, a nested and the experts' own config.
        source = request.getfixturevalue("moe")
        inputs = inputs[:96, :TOY_MOE_HIDDEN]
        text_config = SimpleNamespace(_experts_implementation="batched_mm")
        source.config = SimpleNamespace(_experts_implementation="kernels", text_config=text_config)
        source.mlp.experts.config = SimpleNamespace(_experts_implementation="batched_mm")
    elif case.startswith("rotated"):
        # The quantizer would quantize the rotated weight, but the codes GPTQ or the imatrix
        # encoder pins are the weight's own, so the selection would not be what scoring measured.
        config, name = _recipe(GPTQ if case == "rotated_gptq" else {"method": "imatrix"}, "iq2_xxs")
        config["quant_cfg"][1]["cfg"]["rotate"] = True
        formats = [(config, name)]
    elif case == "awq_gptq":
        # AWQ leaves its scale on the weight quantizer, which GPTQ's codes do not account for, and
        # the selected model would fold the scale into the codes scoring measured.
        formats = [_recipe([{"method": "awq_lite"}, GPTQ], "iq2_xxs")]
    elif case == "layerwise_checkpoint":
        # A rerun would resume from per-layer checkpoints holding the rewritten weights, which
        # load into new tensors that restoring the source weights does not reach.
        checkpoint_dir = str(request.getfixturevalue("tmp_path"))
        layerwise = {"enable": True, "checkpoint_dir": checkpoint_dir}
        formats = [_recipe({**GPTQ, "layerwise": layerwise}, "iq2_xxs")]
    model = copy.deepcopy(source)
    if case == "aliased":
        # Distinct Parameters over one storage, which deepcopy would split: GPTQ on the first layer
        # would rewrite the second's weight, also when the second selects BF16.
        for layers in (source, model):
            layers[1].weight = torch.nn.Parameter(layers[0].weight[:256])
        formats = [_recipe(GPTQ, "iq2_xxs")]
    configs = []
    if case == "fused_moe_configs":
        configs = [model.config, model.config.text_config, model.mlp.experts.config]
    held = [dict(vars(config)) for config in configs]

    with pytest.raises((ValueError, NotImplementedError)):
        _search(model, inputs, formats, 16.0)

    assert not mto.ModeloptStateManager.is_converted(model)
    assert [type(m) for m in model.modules()] == [type(m) for m in source.modules()]
    for weight, original in zip(model.parameters(), source.parameters()):
        assert torch.equal(weight, original)
    if configs:
        restored = [model.config, model.config.text_config, model.mlp.experts.config]
        assert list(map(id, restored)) == list(map(id, configs))
        assert [vars(config) for config in configs] == held
    # The same model can search again: here Q8_0, which imatrix leaves alone even when tied.
    _search(model, inputs, [_recipe({"method": "imatrix"}, "q8_0")], 16.0)
    if configs:
        # A search that converts moves them, so the refusal did hand them back.
        assert [vars(config) for config in configs] != held


@pytest.mark.parametrize("case", ["rewrite_then_fail", "algorithm_list_fails_midway"])
def test_auto_quantize_calibration_failure_hands_back_the_source_weights(case):
    inputs = _inputs()
    if case == "rewrite_then_fail":
        # GPTQ rewrites the first weight, then cannot encode the second's 640 columns.
        source = torch.nn.Sequential(
            torch.nn.Linear(512, 640, bias=False), torch.nn.Linear(640, 256, bias=False)
        )
        algorithm = GPTQ
    else:
        # GPTQ completes and adds its mode, then layerwise imatrix finds no decoder layers.
        source = _two_layers()
        algorithm = [GPTQ, {"method": "imatrix", "layerwise": {"enable": True}}]
    model = copy.deepcopy(source)

    with pytest.raises((ValueError, RuntimeError)):
        _search(model, inputs, [_recipe(algorithm, "iq2_xxs")], 16.0)

    for weight, original in zip(model.parameters(), source.parameters()):
        assert torch.equal(weight, original)
    assert model_calib._ENABLE_FOLDING_PQS_TO_WEIGHTS
    # The model stays converted, without a mode of the calibration that failed.
    assert [mode for mode, _ in mto.ModeloptStateManager(model).state_dict()] == ["auto_quantize"]


def test_a_replayed_pin_fake_quantizes_to_exactly_its_decoding_in_bf16():
    torch.manual_seed(0)
    layer = torch.nn.Linear(512, 64, bias=False).to(torch.bfloat16)
    mtq.quantize(layer, _iq_config("max", "iq2_xxs"), forward_loop=lambda m: m(layer.weight))
    iq2_xxs = GGML_FORMAT_REGISTRY["iq2_xxs"]
    # Codes far smaller than the source weight they replay over.
    other = torch.randn(layer.weight.shape) * 1e-3
    pin_packed_weight(layer.weight_quantizer, "iq2_xxs", iq2_xxs.quantize(other)[0])
    setattr(layer.weight_quantizer, PIN_OVERRIDES_WEIGHT, True)

    replayed = layer.weight_quantizer(layer.weight)

    assert torch.equal(replayed, iq2_xxs.decode_pinned(layer.weight_quantizer, layer.weight))
