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

"""Tests for translating a transformers-targeted quant_cfg to Megatron-Core names."""

from types import SimpleNamespace

import pytest
from torch import nn

from modelopt.recipe import load_recipe
from modelopt.torch.quantization.config import normalize_quant_cfg_list
from modelopt.torch.quantization.conversion import set_quantizer_by_cfg
from modelopt.torch.quantization.hf_to_megatron import (
    ModuleMapping,
    module_mappings_from_registry,
    translate_quant_cfg_to_megatron,
)
from modelopt.torch.quantization.nn import TensorQuantizer

FP8 = {"num_bits": (4, 3), "axis": None}
DISABLE_ALL = {"quantizer_name": "*", "enable": False}
LINEAR = ("weight_quantizer", "input_quantizer", "output_quantizer")
KV = ("q_bmm_quantizer", "k_bmm_quantizer", "v_bmm_quantizer")
LAYERS = (0, 1)
EXPERTS = 3


def _mapping(megatron_param, hf_param):
    return SimpleNamespace(megatron_param=megatron_param, hf_param=hf_param)


# A stand-in shaped like a Megatron-Bridge registry of a Qwen3-MoE-like model, plus mappings to be
# skipped. The real registries are read in tests/gpu_megatron/torch/utils/plugins/test_mbridge.py.
REGISTRY = SimpleNamespace(
    mappings=[
        _mapping("output_layer.weight", "lm_head.weight"),
        _mapping(
            "decoder.layers.*.self_attention.linear_proj.weight",
            "model.layers.*.self_attn.o_proj.weight",
        ),
        _mapping(
            "decoder.layers.*.self_attention.linear_qkv.weight",
            {f"{p}": f"model.layers.*.self_attn.{p}_proj.weight" for p in "qkv"},
        ),
        _mapping(
            "decoder.layers.*.mlp.experts.local_experts.*.linear_fc1.weight",
            {p: f"model.layers.*.mlp.experts.*.{p}_proj.weight" for p in ("gate", "up")},
        ),
        _mapping(
            "decoder.layers.*.mlp.experts.local_experts.*.linear_fc2.weight",
            "model.layers.*.mlp.experts.*.down_proj.weight",
        ),
        _mapping(
            "decoder.layers.*.mlp.router.expert_bias",
            "model.layers.*.mlp.gate.e_score_correction_bias",
        ),
        _mapping(
            "mtp.layers.*.mtp_model_layer.mlp.linear_fc2.weight",
            "mtp.layers.*.mlp.down_proj.weight",
        ),
        _mapping("vision_tower.**", "vision_tower.**"),
    ]
)
MAPPINGS = module_mappings_from_registry(REGISTRY)

# Grouped experts of a VLM whose transformers experts are one fused parameter.
FUSED_EXPERTS_REGISTRY = SimpleNamespace(
    mappings=[
        _mapping(
            "language_model.decoder.layers.*.mlp.experts.linear_fc1.weight*",
            "model.language_model.layers.*.mlp.experts.gate_up_proj",
        ),
        _mapping(
            "language_model.decoder.layers.*.mlp.experts.linear_fc2.weight*",
            "model.language_model.layers.*.mlp.experts.down_proj",
        ),
    ]
)

# Experts whose transformers parameters are packed, in both Megatron layouts: the grouped module
# and the sequential ``local_experts.*`` ones map from the same parameters (as GPT-OSS's do).
PACKED_EXPERTS_REGISTRY = SimpleNamespace(
    mappings=[
        *FUSED_EXPERTS_REGISTRY.mappings,
        _mapping(
            "language_model.decoder.layers.*.mlp.experts.local_experts.*.linear_fc1.weight",
            "model.language_model.layers.*.mlp.experts.gate_up_proj",
        ),
        _mapping(
            "language_model.decoder.layers.*.mlp.experts.local_experts.*.linear_fc2.weight",
            "model.language_model.layers.*.mlp.experts.down_proj",
        ),
    ]
)

# The language-model rows of Megatron-Bridge's Qwen3.5-MoE registry: GatedDeltaNet and gated
# attention layers with MoE MLPs, where Megatron fuses four transformers modules into ``in_proj``.
_HF_LAYER = "model.language_model.layers.*"
_MEGATRON_LAYER = "language_model.decoder.layers.*"
QWEN3_5_MOE_REGISTRY = SimpleNamespace(
    mappings=[
        _mapping("language_model.output_layer.weight", "lm_head.weight"),
        _mapping(f"{_MEGATRON_LAYER}.mlp.router.weight", f"{_HF_LAYER}.mlp.gate.weight"),
        _mapping(
            f"{_MEGATRON_LAYER}.self_attention.linear_proj.weight",
            f"{_HF_LAYER}.self_attn.o_proj.weight",
        ),
        _mapping(
            f"{_MEGATRON_LAYER}.self_attention.linear_qkv.weight",
            {p: f"{_HF_LAYER}.self_attn.{p}_proj.weight" for p in "qkv"},
        ),
        _mapping(
            f"{_MEGATRON_LAYER}.self_attention.in_proj.weight",
            {p: f"{_HF_LAYER}.linear_attn.in_proj_{p}.weight" for p in ("qkv", "z", "b", "a")},
        ),
        _mapping(
            f"{_MEGATRON_LAYER}.self_attention.out_proj.weight",
            f"{_HF_LAYER}.linear_attn.out_proj.weight",
        ),
        _mapping(
            f"{_MEGATRON_LAYER}.self_attention.conv1d.weight",
            f"{_HF_LAYER}.linear_attn.conv1d.weight",
        ),
        _mapping(
            f"{_MEGATRON_LAYER}.mlp.shared_experts.linear_fc1.weight",
            {p: f"{_HF_LAYER}.mlp.shared_expert.{p}_proj.weight" for p in ("gate", "up")},
        ),
        _mapping(
            f"{_MEGATRON_LAYER}.mlp.shared_experts.linear_fc2.weight",
            f"{_HF_LAYER}.mlp.shared_expert.down_proj.weight",
        ),
        *FUSED_EXPERTS_REGISTRY.mappings,
    ]
)


def _names(template, suffixes, layers=LAYERS):
    return {
        f"{template.format(l=layer, e=expert)}.{suffix}"
        for layer in layers
        for expert in range(EXPERTS)
        for suffix in suffixes
    }


QKV = "decoder.layers.{l}.self_attention.linear_qkv"
PROJ = "decoder.layers.{l}.self_attention.linear_proj"
ATTN = "decoder.layers.{l}.self_attention.core_attention"
FC1 = "decoder.layers.{l}.mlp.experts.local_experts.{e}.linear_fc1"
FC2 = "decoder.layers.{l}.mlp.experts.local_experts.{e}.linear_fc2"
ALL_NAMES = (
    _names("output_layer", LINEAR)
    | _names(QKV, LINEAR)
    | _names(PROJ, LINEAR)
    | _names(ATTN, KV)
    | _names(FC1, LINEAR)
    | _names(FC2, LINEAR)
)


# Qwen3.5-MoE as Megatron-Core builds it: three GatedDeltaNet layers, one gated-attention layer, a
# MoE MLP in each, and an MTP layer.
_GATED_DELTA_NET, _ATTENTION, _ALL_LAYERS = (0, 1, 2), (3,), (0, 1, 2, 3)
_QWEN3_5_MOE_MODULES = {
    "self_attention.in_proj": _GATED_DELTA_NET,
    "self_attention.out_proj": _GATED_DELTA_NET,
    "self_attention.conv1d": _GATED_DELTA_NET,
    "self_attention.linear_qkv": _ATTENTION,
    "self_attention.linear_proj": _ATTENTION,
    "mlp.router": _ALL_LAYERS,
    "mlp.experts.linear_fc1": _ALL_LAYERS,
    "mlp.experts.linear_fc2": _ALL_LAYERS,
    "mlp.shared_experts.linear_fc1": _ALL_LAYERS,
    "mlp.shared_experts.linear_fc2": _ALL_LAYERS,
}
QWEN3_5_MOE_NAMES = (
    _names("language_model.output_layer", LINEAR)
    | _names("language_model.decoder.layers.{l}.self_attention.core_attention", KV, _ATTENTION)
    | _names("language_model.mtp.layers.0.mtp_model_layer.mlp.experts.linear_fc1", LINEAR)
    | {
        name
        for module, layers in _QWEN3_5_MOE_MODULES.items()
        for name in _names(f"language_model.decoder.layers.{{l}}.{module}", LINEAR, layers)
    }
)


def _formats(cfg, names):
    """The format of every quantizer ``cfg`` enables on a module tree with one quantizer per name."""
    model = nn.Module()
    for name in names:
        *path, quantizer = name.split(".")
        parent = model
        for part in path:
            if not hasattr(parent, part):
                setattr(parent, part, nn.Module())
            parent = getattr(parent, part)
        setattr(parent, quantizer, TensorQuantizer())
    set_quantizer_by_cfg(model, cfg)
    return {
        name: module.num_bits
        for name, module in model.named_modules()
        if isinstance(module, TensorQuantizer) and module.is_enabled
    }


def _apply(cfg, mappings, names, layer_indices=LAYERS):
    """Translate ``cfg`` and return the names of the quantizers it enables."""
    translated = translate_quant_cfg_to_megatron([DISABLE_ALL, *cfg], mappings, layer_indices)
    return set(_formats(translated, names))


def _entry(glob, **kwargs):
    return {"quantizer_name": glob, "cfg": FP8, **kwargs}


@pytest.mark.parametrize(
    ("entries", "expected"),
    [
        pytest.param([_entry("*lm_head*")], _names("output_layer", LINEAR), id="rename"),
        pytest.param(
            [_entry("*[qkvo]_proj*weight_quantizer")],
            _names(QKV, LINEAR[:1]) | _names(PROJ, LINEAR[:1]),
            id="fused_qkv_and_glob_class",
        ),
        pytest.param(
            [_entry("*layers.1.self_attn*weight_quantizer")],
            _names(QKV, LINEAR[:1], (1,)) | _names(PROJ, LINEAR[:1], (1,)),
            id="layer_index",
        ),
        pytest.param(
            [_entry("*self_attn*[kv]_bmm_quantizer")], _names(ATTN, KV[1:]), id="kv_cache"
        ),
        pytest.param(
            [_entry("*experts*gate_proj*"), _entry("*experts*up_proj*")],
            _names(FC1, LINEAR),
            id="fused_pieces_selected_by_separate_entries",
        ),
        pytest.param(
            [_entry("*mlp.experts*weight_quantizer"), _entry("*linear_proj*weight_quantizer")],
            _names(FC1, LINEAR[:1]) | _names(FC2, LINEAR[:1]) | _names(PROJ, LINEAR[:1]),
            id="already_matching_or_megatron_spelled",
        ),
        # A piece configured alone is fine when a later entry settles the fused module as a whole.
        pytest.param([_entry("*q_proj*"), DISABLE_ALL], set(), id="piece_then_disable_all"),
        pytest.param(
            [_entry("*q_proj*"), {"quantizer_name": "*linear_qkv*", "enable": False}],
            set(),
            id="piece_then_disable_the_megatron_module",
        ),
        pytest.param(
            [_entry("*q_proj*"), _entry("*linear_qkv*")],
            _names(QKV, LINEAR),
            id="piece_then_configure_the_megatron_module",
        ),
    ],
)
def test_translated_entries_select_the_megatron_quantizers(entries, expected):
    assert _apply(entries, MAPPINGS, ALL_NAMES) == expected


def test_entries_that_already_match_are_kept_unchanged():
    cfg = [
        DISABLE_ALL,
        _entry("*weight_quantizer"),
        _entry("*linear_proj*input_quantizer"),
        # Selects by class, not by name, so its transformers-spelled glob must not be translated.
        {"parent_class": "nn.Linear", "quantizer_name": "*lm_head*", "enable": False},
    ]
    assert translate_quant_cfg_to_megatron(cfg, MAPPINGS, LAYERS) == normalize_quant_cfg_list(cfg)


def test_layer_independent_entries_add_one_glob_per_module_not_per_layer():
    translated = translate_quant_cfg_to_megatron([_entry("*[qkv]_proj*")], MAPPINGS, range(48))
    assert len(translated) == 1 + len(LINEAR)


def test_translation_follows_the_pipeline_stage_layer_offset():
    entries = [
        _entry("*layers.3.self_attn*weight_quantizer"),
        _entry("*layers.3.*input_quantizer"),
    ]
    # This rank holds global layers 2 and 3 as its local layers 0 and 1.
    names = _apply(entries, MAPPINGS, ALL_NAMES, layer_indices=(2, 3))
    inputs = LINEAR[1:2]
    assert names == (
        _names(QKV, LINEAR[:1] + inputs, (1,))
        | _names(PROJ, LINEAR[:1] + inputs, (1,))
        | _names(FC1, inputs, (1,))
        | _names(FC2, inputs, (1,))
    )


@pytest.mark.parametrize(
    "fc1",
    [
        "language_model.decoder.layers.{l}.mlp.experts.linear_fc1",
        "language_model.decoder.layers.{l}.mlp.experts.local_experts.{e}.linear_fc1",
    ],
    ids=["grouped", "sequential"],
)
def test_packed_experts_parameter_maps_onto_either_experts_layout(fc1):
    mappings = module_mappings_from_registry(PACKED_EXPERTS_REGISTRY)
    names = _names(fc1, LINEAR) | _names(fc1.replace("fc1", "fc2"), LINEAR)
    assert _apply([_entry("*gate_up_proj_weight_quantizer")], mappings, names) == _names(
        fc1, LINEAR[:1]
    )


@pytest.mark.parametrize(
    ("entries", "megatron_module"),
    [
        ([_entry("*experts*gate_proj*")], "linear_fc1"),
        ([_entry("*q_proj*"), _entry("*k_proj*")], "linear_qkv"),
        ([_entry("*"), {"quantizer_name": "*v_proj*", "enable": False}], "linear_qkv"),
        ([_entry("*experts.0.down_proj*")], "linear_fc2"),
        ([_entry("*experts.2.down_proj*")], "linear_fc2"),
        ([_entry("*experts.[01].down_proj*")], "linear_fc2"),
        ([_entry("*experts.37.down_proj*")], "linear_fc2"),
    ],
    ids=[
        "gate_without_up",
        "q_k_without_v",
        "v_disabled",
        "first_expert",
        "third_expert",
        "first_two_experts",
        "expert_beyond_the_enumerated_ones",
    ],
)
def test_entries_a_megatron_module_cannot_express_raise(entries, megatron_module):
    with pytest.raises(ValueError, match=megatron_module):
        translate_quant_cfg_to_megatron([DISABLE_ALL, *entries], MAPPINGS, LAYERS)


def test_gated_delta_net_gates_do_not_decide_the_fused_in_proj():
    mappings = module_mappings_from_registry(QWEN3_5_MOE_REGISTRY)
    in_proj = _names(
        "language_model.decoder.layers.{l}.self_attention.in_proj", LINEAR[:1], range(3)
    )

    def quantized(*entries):
        return _apply(entries, mappings, QWEN3_5_MOE_NAMES, range(4))

    # The usual recipe: quantize everything, exclude the gates (Megatron fuses them into ``in_proj``).
    exclude_gates = {"quantizer_name": "*linear_attn.in_proj_[ab]*", "enable": False}
    assert in_proj <= quantized(_entry("*weight_quantizer"), exclude_gates)
    assert quantized(_entry("*linear_attn.in_proj_[ab]*")) == set()


def test_module_mappings_skip_what_is_not_a_decoder_module_weight():
    assert [m.megatron for m in MAPPINGS] == [
        "output_layer",
        "decoder.layers.*.self_attention.linear_proj",
        "decoder.layers.*.self_attention.linear_qkv",
        "decoder.layers.*.self_attention.core_attention",
        "decoder.layers.*.mlp.experts.local_experts.*.linear_fc1",
        "decoder.layers.*.mlp.experts.local_experts.*.linear_fc2",
    ]
    assert (
        ModuleMapping(
            "decoder.layers.*.mlp.experts.local_experts.*.linear_fc1",
            ("model.layers.*.mlp.experts.*.gate_proj", "model.layers.*.mlp.experts.*.up_proj"),
        )
        in MAPPINGS
    )


# The W4A4 NVFP4 / FP8-attention Qwen3.6-MoE recipe written with transformers module names only.
# Megatron fuses the four GatedDeltaNet ``in_proj_*`` modules, so it treats them alike.
QWEN3_6_MOE_TRANSFORMERS_RECIPE = """\
# modelopt-schema: modelopt.recipe.config.ModelOptPTQRecipe
imports:
  base_disable_all: configs/ptq/units/base_disable_all
  fp8: configs/numerics/fp8
  kv_fp8_cast: configs/ptq/units/kv_fp8_cast
  nvfp4: configs/numerics/nvfp4
metadata:
  description: Qwen3.6-MoE W4A4 NVFP4 with FP8 attention, written with transformers module names.
quantize:
  algorithm: max
  quant_cfg:
    - $import: base_disable_all
    - {quantizer_name: '*mlp.experts*weight_quantizer', cfg: {$import: nvfp4}}
    - {quantizer_name: '*mlp.experts*input_quantizer', cfg: {$import: nvfp4}}
    - {quantizer_name: '*mlp.shared_expert.*weight_quantizer', cfg: {$import: nvfp4}}
    - {quantizer_name: '*mlp.shared_expert.*input_quantizer', cfg: {$import: nvfp4}}
    - {quantizer_name: '*self_attn*weight_quantizer', cfg: {$import: fp8}}
    - {quantizer_name: '*self_attn*input_quantizer', cfg: {$import: fp8}}
    - {quantizer_name: '*linear_attn.in_proj*weight_quantizer', cfg: {$import: fp8}}
    - {quantizer_name: '*linear_attn.in_proj*input_quantizer', cfg: {$import: fp8}}
    - {quantizer_name: '*linear_attn.out_proj*weight_quantizer', cfg: {$import: fp8}}
    - {quantizer_name: '*linear_attn.out_proj*input_quantizer', cfg: {$import: fp8}}
    - $import: kv_fp8_cast
    - {quantizer_name: '*lm_head*', enable: false}
    - {quantizer_name: '*mlp.gate.*', enable: false}
    - {quantizer_name: '*mlp.shared_expert_gate.*', enable: false}
    - {quantizer_name: '*linear_attn.conv1d*', enable: false}
    - {quantizer_name: '*mtp*', enable: false}
    - {quantizer_name: '*lm_head.weight_quantizer', cfg: {$import: nvfp4}}
    - {quantizer_name: '*lm_head.input_quantizer', cfg: {$import: nvfp4}}
"""


def test_transformers_form_of_the_mcore_recipe_quantizes_the_same_modules(tmp_path):
    native = load_recipe("model_type/qwen3_6_moe/ptq/w4a4_nvfp4-fp8_attn-kv_fp8_cast_mcore")
    recipe_path = tmp_path / "w4a4_nvfp4-fp8_attn-kv_fp8_cast.yaml"
    recipe_path.write_text(QWEN3_6_MOE_TRANSFORMERS_RECIPE)
    mappings = module_mappings_from_registry(QWEN3_5_MOE_REGISTRY)

    def translate(recipe):
        return translate_quant_cfg_to_megatron(recipe.quantize.quant_cfg, mappings, range(4))

    expected = _formats(native.quantize.quant_cfg, QWEN3_5_MOE_NAMES)
    assert _formats(translate(load_recipe(recipe_path)), QWEN3_5_MOE_NAMES) == expected
    # Translating the Megatron-spelled recipe must not change it: the transformers-named
    # exclusions it imports (``in_proj_a``/``in_proj_b``) must not reach its fused ``in_proj``.
    assert _formats(translate(native), QWEN3_5_MOE_NAMES) == expected

    # The comparison is not vacuous: NVFP4 on experts and the output layer, FP8 on both attentions.
    nvfp4, fp8, layer = (2, 1), (4, 3), "language_model.decoder.layers"
    assert expected["language_model.output_layer.weight_quantizer"] == nvfp4
    assert expected[f"{layer}.3.mlp.experts.linear_fc1.weight_quantizer"] == nvfp4
    assert expected[f"{layer}.0.self_attention.in_proj.weight_quantizer"] == fp8
    assert expected[f"{layer}.3.self_attention.linear_qkv.weight_quantizer"] == fp8
