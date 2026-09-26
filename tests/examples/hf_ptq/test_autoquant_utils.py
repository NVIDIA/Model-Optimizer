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

"""Offline coverage for the Hugging Face AutoQuantize example helpers."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from modelopt.recipe import load_recipe
from modelopt.recipe.config import AutoQuantizeConfig, AutoQuantizeConstraints
from modelopt.recipe.presets import QUANT_CFG_CHOICES
from modelopt.torch.quantization.config import QuantizeConfig


@pytest.fixture
def autoquant_utils(monkeypatch):
    examples_dir = Path(__file__).resolve().parents[3] / "examples" / "hf_ptq"
    monkeypatch.syspath_prepend(str(examples_dir))
    return importlib.import_module("autoquant_utils")


def test_autoquant_recipe_builds_mtq_inputs(autoquant_utils):
    """The recipe path maps an AutoQuantizeConfig to the expected mtq.auto_quantize inputs."""
    args = SimpleNamespace(kv_cache_qformat="none")
    aq = load_recipe("general/auto_quantize/nvfp4_fp8_at_5p4bits").auto_quantize
    inputs = autoquant_utils._mtq_inputs_from_auto_quantize_config(aq, args)

    # The shared base cost-excluded unit is spliced into every general AutoQuantize recipe, so it
    # reaches mtq under constraints.cost (VL vision tower / MTP out of the bit-budget denominator).
    assert inputs["constraints"] == {
        "effective_bits": 5.4,
        "cost_model": "weight",
        "cost": {"excluded_module_name_patterns": ["*visual*", "*mtp*", "*vision_tower*"]},
    }
    assert inputs["kv_cache_quant_cfg"] is None
    assert inputs["method"] == "gradient"
    assert inputs["score_size"] == 128
    assert inputs["fixed_quantization_config"] is None
    assert inputs["module_search_spaces"] == []
    # disabled_layers come straight from the recipe (no model introspection).
    assert inputs["disabled_layers"] == aq.disabled_layers
    assert "*output_layer*" in inputs["disabled_layers"]
    # Candidates resolve to the exact preset dicts mtq expects (preset identity preserved).
    assert inputs["quantization_formats"][0] == QUANT_CFG_CHOICES["nvfp4"]
    assert inputs["quantization_formats"][1] == QUANT_CFG_CHOICES["fp8"]


def test_kv_autoquant_recipe_builds_kv_search_inputs(autoquant_utils):
    args = SimpleNamespace(kv_cache_qformat="fp8_cast")
    aq = load_recipe("general/auto_quantize/kv_fp8_nvfp4_cast_kl_div_at_5p4bits").auto_quantize
    inputs = autoquant_utils._mtq_inputs_from_auto_quantize_config(aq, args)

    assert inputs["search_domain"] == "kv_cache"
    assert inputs["constraints"] == {"effective_bits": 5.4, "cost_model": "kv_cache"}
    assert inputs["method"] == "kl_div"
    assert [config["effective_bits"] for config in inputs["quantization_formats"]] == [8.0, 4.5]
    assert aq.cost_excluded_layers == []
    assert "*mtp*" in inputs["disabled_layers"]
    assert "kv_cache_quant_cfg" not in inputs


def test_followup_kv_autoquant_suppresses_uniform_kv_fallback(autoquant_utils):
    args = SimpleNamespace(kv_cache_qformat="fp8_cast")
    aq = load_recipe("general/auto_quantize/nvfp4_fp8_at_5p4bits").auto_quantize

    inputs = autoquant_utils._mtq_inputs_from_auto_quantize_config(aq, args, allow_uniform_kv=False)

    assert inputs["kv_cache_quant_cfg"] is None


def test_fixed_ptq_kv_precheck_does_not_widen_scoped_gemm_rule(autoquant_utils):
    fixed = QuantizeConfig(
        quant_cfg=[
            {
                "quantizer_name": "model.layers.*.mlp.*",
                "cfg": {"num_bits": (4, 3), "constant_amax": 1.0},
            }
        ],
        algorithm="max",
    )

    assert not autoquant_utils._quantize_config_explicitly_enables_kv(fixed.model_dump())


@pytest.mark.parametrize(
    "kv_pattern",
    [
        "model.layers.*.self_attn.*[kv]_bmm_quantizer",
        "*self_attn*k_bmm_quantizer",
        "*.language_model.*.attention.*_bmm_quantizer",
        "*k_bmm*",
        "*self_attn.*",
        "*[kv]_bmm*",
    ],
)
def test_fixed_ptq_kv_precheck_detects_scoped_kv_rules(autoquant_utils, kv_pattern):
    fixed = QuantizeConfig(
        quant_cfg=[
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": kv_pattern,
                "cfg": {"num_bits": (4, 3), "constant_amax": 1.0},
            },
            {
                "parent_class": "nn.Embedding",
                "quantizer_name": "*",
                "enable": False,
            },
        ],
        algorithm="max",
    )

    assert autoquant_utils._quantize_config_explicitly_enables_kv(fixed.model_dump())


def test_fixed_ptq_kv_precheck_detects_parent_scoped_kv_rule(autoquant_utils):
    fixed = QuantizeConfig(
        quant_cfg=[
            {"quantizer_name": "*", "enable": False},
            {
                "parent_class": "LlamaAttention",
                "quantizer_name": "*_bmm_quantizer",
                "cfg": {"num_bits": (4, 3), "constant_amax": 1.0},
            },
        ],
        algorithm="max",
    )

    assert autoquant_utils._quantize_config_explicitly_enables_kv(fixed.model_dump())


def test_kv_autoquant_kl_excludes_padding_positions(autoquant_utils):
    logits = torch.arange(2 * 4 * 3).reshape(2, 4, 3)
    attention_mask = torch.tensor([[1, 1, 0, 0], [0, 1, 1, 0]])

    selected = autoquant_utils._select_unpadded_logits(logits, {"attention_mask": attention_mask})

    assert torch.equal(selected, logits[attention_mask.bool()])


def test_kv_autoquant_kl_rejects_misaligned_attention_mask(autoquant_utils):

    with pytest.raises(ValueError, match="matching token dimensions"):
        autoquant_utils._select_unpadded_logits(
            torch.zeros(2, 4, 3), {"attention_mask": torch.ones(2, 3)}
        )


@pytest.mark.parametrize(
    ("search_domain", "expected_shape"),
    [("weight", (2, 4, 3)), ("kv_cache", (4, 3))],
)
def test_kl_padding_exclusion_is_scoped_to_kv_autoquant(
    autoquant_utils, monkeypatch, search_domain, expected_shape
):
    inputs = {
        "search_domain": search_domain,
        "constraints": {"effective_bits": 8.0},
        "quantization_formats": [],
        "fixed_quantization_config": None,
        "module_search_spaces": [],
        "disabled_layers": [],
        "kv_cache_quant_cfg": None,
        "method": "kl_div",
        "score_size": 1,
    }
    monkeypatch.setattr(
        autoquant_utils, "_mtq_inputs_from_auto_quantize_config", lambda *_args, **_kwargs: inputs
    )
    logits = torch.arange(2 * 4 * 3).reshape(2, 4, 3).float()
    batch = {
        "input_ids": torch.ones(2, 4, dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 0, 0], [0, 1, 1, 0]]),
    }

    class Model(torch.nn.Module):
        def forward(self, **_kwargs):
            return SimpleNamespace(logits=logits, loss=torch.tensor(0.0))

    observed = {}

    def fake_auto_quantize(search_model, **kwargs):
        observed["shape"] = tuple(kwargs["forward_step"](search_model, batch).shape)
        return search_model, {}

    monkeypatch.setattr(autoquant_utils.mtq, "auto_quantize", fake_auto_quantize)
    args = SimpleNamespace(
        calib_with_images=False,
        inference_pipeline_parallel=1,
        use_fsdp2=False,
        batch_size=1,
        auto_quantize_checkpoint=None,
    )
    model = Model()

    autoquant_utils.auto_quantize(args, model, [batch], SimpleNamespace(), full_model=model)

    assert observed["shape"] == expected_shape


def test_kv_autoquant_rejects_fsdp2(autoquant_utils, monkeypatch):
    monkeypatch.setattr(
        autoquant_utils,
        "_mtq_inputs_from_auto_quantize_config",
        lambda *_args, **_kwargs: {"search_domain": "kv_cache"},
    )
    args = SimpleNamespace(
        calib_with_images=False,
        inference_pipeline_parallel=1,
        use_fsdp2=True,
    )

    with pytest.raises(NotImplementedError, match="KV-cache AutoQuantize does not support"):
        autoquant_utils.auto_quantize(args, torch.nn.Module(), [], SimpleNamespace())


def test_weight_autoquant_retains_fsdp2_warning(autoquant_utils, monkeypatch):
    model = torch.nn.Module()
    inputs = {
        "search_domain": "weight",
        "constraints": {"effective_bits": 8.0},
        "quantization_formats": [],
        "fixed_quantization_config": None,
        "module_search_spaces": [],
        "disabled_layers": [],
        "kv_cache_quant_cfg": None,
        "method": "gradient",
        "score_size": 1,
    }
    monkeypatch.setattr(
        autoquant_utils, "_mtq_inputs_from_auto_quantize_config", lambda *_args, **_kwargs: inputs
    )
    monkeypatch.setattr(
        autoquant_utils.mtq, "auto_quantize", lambda search_model, **_kwargs: (search_model, {})
    )
    args = SimpleNamespace(
        calib_with_images=False,
        inference_pipeline_parallel=1,
        use_fsdp2=True,
        batch_size=1,
        auto_quantize_checkpoint=None,
    )

    with pytest.warns(UserWarning, match="use at your own risk"):
        assert autoquant_utils.auto_quantize(args, model, [], SimpleNamespace()) is model


def test_fsdp2_preload_guard_distinguishes_weight_and_kv_autoquant(autoquant_utils):

    assert autoquant_utils._recipe_is_kv_auto_quantize(
        "general/auto_quantize/kv_fp8_nvfp4_cast_kl_div_at_5p4bits"
    )
    assert autoquant_utils._recipe_is_kv_auto_quantize(
        "general/auto_quantize/nvfp4_fp8_gradient_then_kv_fp8_nvfp4_cast_kl_div_at_5p4bits"
    )
    assert not autoquant_utils._recipe_is_kv_auto_quantize(
        "general/auto_quantize/nvfp4_fp8_at_5p4bits"
    )


def test_autoquant_recipe_cost_excluded_layers_map_into_cost(autoquant_utils):
    """Top-level cost_excluded_layers maps to the mtq constraints.cost.excluded_module_name_patterns
    key (distinct from disabled_layers), so a cost-exclusion recipe matches the nested mtq dict."""
    args = SimpleNamespace(kv_cache_qformat="none")
    aq = load_recipe(
        "model_type/qwen3_6_moe/auto_quantize/w4a16_nvfp4_fp8_at_6p0bits-active_moe"
    ).auto_quantize
    inputs = autoquant_utils._mtq_inputs_from_auto_quantize_config(aq, args)

    # cost-exclusion is hoisted to a sibling of disabled_layers but still reaches the mtq cost dict.
    assert aq.cost_excluded_layers == ["*visual*", "*mtp*", "*vision_tower*"]
    assert inputs["constraints"]["cost"] == {
        "active_moe_expert_ratio": 0.03125,
        "excluded_module_name_patterns": ["*visual*", "*mtp*", "*vision_tower*"],
    }
    # The two exclusions are independent: cost-excluded patterns are also disabled here, but the
    # roles (cost-accounting vs search) are tracked separately.
    assert "*visual*" in inputs["disabled_layers"]


def test_autoquant_recipe_maps_module_search_spaces(autoquant_utils):
    """Fixed PTQ baseline and explicit recipe candidates map to mtq inputs."""
    args = SimpleNamespace(kv_cache_qformat="none")
    recipe = load_recipe(
        "model_type/qwen3_6_moe/auto_quantize/w4a16_nvfp4_fp8_module_spaces_at_6p0bits-active_moe"
    )
    inputs = autoquant_utils._mtq_inputs_from_auto_quantize_config(
        recipe.auto_quantize, args, fixed_quantize_config=recipe.quantize
    )
    model_ptq = load_recipe("model_type/qwen3_5_moe/ptq/w4a16_nvfp4-fp8_attn-kv_fp8_cast")

    assert inputs["quantization_formats"] == []
    assert inputs["fixed_quantization_config"] == model_ptq.quantize.model_dump()
    (searched,) = inputs["module_search_spaces"]
    assert searched["module_name_patterns"] == [
        "*mlp.shared_expert*",
        "*linear_attn*",
        "*self_attn*",
        "*lm_head*",
    ]
    assert searched["quantization_formats"] == [
        QUANT_CFG_CHOICES["w4a16_nvfp4"],
        QUANT_CFG_CHOICES["fp8"],
    ]
    assert searched["allow_no_quant"] is False


def test_autoquant_rejects_non_export_safe_candidate(autoquant_utils):
    """A candidate that resolves to a preset outside the export-safe set is rejected before search."""
    args = SimpleNamespace(kv_cache_qformat="none")
    non_safe = next(
        k for k in QUANT_CFG_CHOICES if k not in autoquant_utils._AUTO_QUANTIZE_QFORMATS
    )
    aq = AutoQuantizeConfig(
        constraints=AutoQuantizeConstraints(effective_bits=4.8),
        candidate_formats=[
            QuantizeConfig(**QUANT_CFG_CHOICES["fp8"]),
            QuantizeConfig(**QUANT_CFG_CHOICES[non_safe]),
        ],
    )
    with pytest.raises(ValueError, match="not supported for unified checkpoint export"):
        autoquant_utils._mtq_inputs_from_auto_quantize_config(aq, args)


def test_autoquant_warns_on_custom_candidate(autoquant_utils):
    """A candidate matching no shipped preset can't be export-verified, so it warns (not blocks)."""
    args = SimpleNamespace(kv_cache_qformat="none")
    custom = QuantizeConfig(quant_cfg=[{"quantizer_name": "*", "enable": False}])
    aq = AutoQuantizeConfig(
        constraints=AutoQuantizeConstraints(effective_bits=4.8),
        candidate_formats=[QuantizeConfig(**QUANT_CFG_CHOICES["fp8"]), custom],
    )
    with pytest.warns(UserWarning, match="export compatibility cannot be verified"):
        autoquant_utils._mtq_inputs_from_auto_quantize_config(aq, args)


def test_autoquant_export_guard_not_bypassed_by_effective_bits(autoquant_utils):
    """A non-export-safe preset can't dodge the guard by adding a cost-only effective_bits override."""
    args = SimpleNamespace(kv_cache_qformat="none")
    non_safe = next(
        k for k in QUANT_CFG_CHOICES if k not in autoquant_utils._AUTO_QUANTIZE_QFORMATS
    )
    tampered = QuantizeConfig(**{**QUANT_CFG_CHOICES[non_safe], "effective_bits": 4.5})
    aq = AutoQuantizeConfig(
        constraints=AutoQuantizeConstraints(effective_bits=5.4),
        candidate_formats=[QuantizeConfig(**QUANT_CFG_CHOICES["fp8"]), tampered],
    )
    with pytest.raises(ValueError, match="not supported for unified checkpoint export"):
        autoquant_utils._mtq_inputs_from_auto_quantize_config(aq, args)
