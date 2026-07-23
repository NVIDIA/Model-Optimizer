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

import importlib
import sys
from pathlib import Path

import pytest

from modelopt.recipe import load_recipe
from modelopt.recipe.config import AutoQuantizeConfig, AutoQuantizeConstraints
from modelopt.recipe.presets import QUANT_CFG_CHOICES
from modelopt.torch.quantization.config import QuantizeConfig

_EXAMPLES_DIR = Path(__file__).resolve().parents[3] / "examples" / "hf_ptq"


def _import_hf_ptq(monkeypatch):
    monkeypatch.syspath_prepend(str(_EXAMPLES_DIR))
    return importlib.import_module("hf_ptq")


def _parse_hf_ptq_args(monkeypatch, *args):
    hf_ptq = _import_hf_ptq(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["hf_ptq.py", *args])
    parsed_args = hf_ptq.parse_args()
    parsed_args.dataset = (
        parsed_args.dataset.split(",")
        if isinstance(parsed_args.dataset, str)
        else parsed_args.dataset
    )
    parsed_args.calib_size = [int(num_sample) for num_sample in parsed_args.calib_size.split(",")]
    return hf_ptq, parsed_args


def test_autoquant_recipe_builds_mtq_inputs(monkeypatch):
    """The recipe path maps an AutoQuantizeConfig to the expected mtq.auto_quantize inputs."""
    hf_ptq, args = _parse_hf_ptq_args(
        monkeypatch, "--pyt_ckpt_path", "dummy", "--kv_cache_qformat", "none"
    )
    aq = load_recipe("general/auto_quantize/nvfp4_fp8_at_5p4bits").auto_quantize
    inputs = hf_ptq._mtq_inputs_from_auto_quantize_config(aq, args)

    assert inputs["constraints"] == {"effective_bits": 5.4, "cost_model": "weight"}
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


def test_autoquant_recipe_cost_excluded_layers_map_into_cost(monkeypatch):
    """Top-level cost_excluded_layers maps to the mtq constraints.cost.excluded_module_name_patterns
    key (distinct from disabled_layers), so a cost-exclusion recipe matches the nested mtq dict."""
    hf_ptq, args = _parse_hf_ptq_args(
        monkeypatch, "--pyt_ckpt_path", "dummy", "--kv_cache_qformat", "none"
    )
    aq = load_recipe(
        "huggingface/qwen3_6_moe/auto_quantize/w4a16_nvfp4_fp8_at_6p0bits-active_moe"
    ).auto_quantize
    inputs = hf_ptq._mtq_inputs_from_auto_quantize_config(aq, args)

    # cost-exclusion is hoisted to a sibling of disabled_layers but still reaches the mtq cost dict.
    assert aq.cost_excluded_layers == ["*visual*", "*mtp*", "*vision_tower*"]
    assert inputs["constraints"]["cost"] == {
        "active_moe_expert_ratio": 0.03125,
        "excluded_module_name_patterns": ["*visual*", "*mtp*", "*vision_tower*"],
    }
    # The two exclusions are independent: cost-excluded patterns are also disabled here, but the
    # roles (cost-accounting vs search) are tracked separately.
    assert "*visual*" in inputs["disabled_layers"]


def test_autoquant_recipe_maps_module_search_spaces(monkeypatch):
    """Fixed PTQ baseline and explicit recipe candidates map to mtq inputs."""
    hf_ptq, args = _parse_hf_ptq_args(
        monkeypatch, "--pyt_ckpt_path", "dummy", "--kv_cache_qformat", "none"
    )
    recipe = load_recipe(
        "huggingface/qwen3_6_moe/auto_quantize/w4a16_nvfp4_fp8_module_spaces_at_6p0bits-active_moe"
    )
    inputs = hf_ptq._mtq_inputs_from_auto_quantize_config(
        recipe.auto_quantize, args, fixed_quantize_config=recipe.quantize
    )
    model_ptq = load_recipe("huggingface/qwen3_5_moe/ptq/w4a16_nvfp4-fp8_attn-kv_fp8_cast")

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


def test_autoquant_rejects_non_export_safe_candidate(monkeypatch):
    """A candidate that resolves to a preset outside the export-safe set is rejected before search."""
    hf_ptq, args = _parse_hf_ptq_args(
        monkeypatch, "--pyt_ckpt_path", "dummy", "--kv_cache_qformat", "none"
    )
    non_safe = next(k for k in QUANT_CFG_CHOICES if k not in hf_ptq._AUTO_QUANTIZE_QFORMATS)
    aq = AutoQuantizeConfig(
        constraints=AutoQuantizeConstraints(effective_bits=4.8),
        candidate_formats=[
            QuantizeConfig(**QUANT_CFG_CHOICES["fp8"]),
            QuantizeConfig(**QUANT_CFG_CHOICES[non_safe]),
        ],
    )
    with pytest.raises(ValueError, match="not supported for unified checkpoint export"):
        hf_ptq._mtq_inputs_from_auto_quantize_config(aq, args)


def test_autoquant_warns_on_custom_candidate(monkeypatch):
    """A candidate matching no shipped preset can't be export-verified, so it warns (not blocks)."""
    hf_ptq, args = _parse_hf_ptq_args(
        monkeypatch, "--pyt_ckpt_path", "dummy", "--kv_cache_qformat", "none"
    )
    custom = QuantizeConfig(quant_cfg=[{"quantizer_name": "*", "enable": False}])
    aq = AutoQuantizeConfig(
        constraints=AutoQuantizeConstraints(effective_bits=4.8),
        candidate_formats=[QuantizeConfig(**QUANT_CFG_CHOICES["fp8"]), custom],
    )
    with pytest.warns(UserWarning, match="export compatibility cannot be verified"):
        hf_ptq._mtq_inputs_from_auto_quantize_config(aq, args)


def test_autoquant_export_guard_not_bypassed_by_effective_bits(monkeypatch):
    """A non-export-safe preset can't dodge the guard by adding a cost-only effective_bits override."""
    hf_ptq, args = _parse_hf_ptq_args(
        monkeypatch, "--pyt_ckpt_path", "dummy", "--kv_cache_qformat", "none"
    )
    non_safe = next(k for k in QUANT_CFG_CHOICES if k not in hf_ptq._AUTO_QUANTIZE_QFORMATS)
    tampered = QuantizeConfig(**{**QUANT_CFG_CHOICES[non_safe], "effective_bits": 4.5})
    aq = AutoQuantizeConfig(
        constraints=AutoQuantizeConstraints(effective_bits=5.4),
        candidate_formats=[QuantizeConfig(**QUANT_CFG_CHOICES["fp8"]), tampered],
    )
    with pytest.raises(ValueError, match="not supported for unified checkpoint export"):
        hf_ptq._mtq_inputs_from_auto_quantize_config(aq, args)


def test_autoquant_config_from_deprecated_cli_flags(monkeypatch):
    """The deprecated --auto_quantize_* flags convert to an AutoQuantizeConfig with the shared
    base disabled + cost-excluded patterns appended (no new flags, no model introspection)."""
    hf_ptq, args = _parse_hf_ptq_args(
        monkeypatch,
        "--pyt_ckpt_path",
        "dummy",
        "--qformat",
        "fp8,nvfp4",
        "--auto_quantize_bits",
        "5.4",
        "--auto_quantize_cost_model",
        "active_moe",
        "--auto_quantize_active_moe_expert_ratio",
        "0.03125",
        "--kv_cache_qformat",
        "none",
    )
    aq = hf_ptq._auto_quantize_config_from_cli(args)

    assert aq.constraints.effective_bits == 5.4
    assert aq.constraints.cost_model == "active_moe"
    assert aq.constraints.cost.active_moe_expert_ratio == 0.03125
    assert aq.auto_quantize_method == "gradient"
    assert aq.score_size == 128
    # candidates come from --qformat and resolve to their shipped presets.
    assert [hf_ptq._match_candidate_to_preset(f)[0] for f in aq.candidate_formats] == [
        "fp8",
        "nvfp4",
    ]
    # base disabled + base cost-excluded appended from the shared units (no introspection).
    assert "*output_layer*" in aq.disabled_layers
    assert aq.cost_excluded_layers == ["*visual*", "*mtp*", "*vision_tower*"]
