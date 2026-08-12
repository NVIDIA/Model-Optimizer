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
"""Smoke tests for ``modelopt.recipe.presets`` preset discovery.

Guards the eager import-time load shared by the PTQ example scripts: every preset
under the model/KV dirs must load into a usable ``quant_cfg`` dict, and the KV
``none`` sentinel must not collide with a discovered preset. A single malformed
preset YAML would otherwise break ``import modelopt.recipe.presets`` (and every
PTQ example).
"""

import pytest

import modelopt.torch.quantization as mtq
from modelopt.recipe import load_recipe, presets
from modelopt.torch.opt.config_loader import BUILTIN_CONFIG_ROOT
from modelopt.torch.quantization.config import QuantizeConfig


def _yaml_basenames(subdir: str) -> set[str]:
    return {
        entry.name.rsplit(".", 1)[0]
        for entry in BUILTIN_CONFIG_ROOT.joinpath(subdir).iterdir()
        if entry.name.endswith((".yaml", ".yml"))
    }


@pytest.mark.parametrize(
    ("choices", "preset_dir"),
    [
        (presets.QUANT_CFG_CHOICES, presets.MODEL_QUANT_PRESET_DIR),
        (presets.KV_QUANT_CFG_CHOICES, presets.KV_QUANT_PRESET_DIR),
    ],
    ids=["model", "kv"],
)
def test_every_discovered_preset_loads(choices, preset_dir):
    # Configs load eagerly at import, so a malformed preset would already have raised.
    # Assert discovery is non-empty, covers every YAML on disk, and that each resolved
    # entry is a usable quant_cfg dict.
    basenames = _yaml_basenames(preset_dir)
    assert basenames, f"no preset YAMLs discovered under {preset_dir}"
    assert basenames <= set(choices), "a preset YAML is missing from the discovered choices"
    for name, cfg in choices.items():
        assert isinstance(cfg, dict), f"{name} did not resolve to a dict"
        assert "quant_cfg" in cfg, f"{name} is missing the 'quant_cfg' key"


def test_kv_none_sentinel_is_not_a_discovered_preset():
    # The scripts branch on ``kv_cache_qformat != KV_CACHE_NONE``; a real preset named
    # "none" would make that branch ambiguous.
    assert presets.KV_CACHE_NONE not in presets.KV_QUANT_CFG_CHOICES


def test_w4a16_nvfp4_preset_disables_vllm_marlin_incompatible_projections():
    disabled_quantizers = {
        entry["quantizer_name"]
        for entry in presets.QUANT_CFG_CHOICES["w4a16_nvfp4"]["quant_cfg"]
        if entry.get("enable") is False
    }

    assert {
        "*linear_attn.in_proj_a*",
        "*linear_attn.in_proj_b*",
        "*visual*",
        "*vision_tower*",
    } <= disabled_quantizers


@pytest.mark.parametrize(
    ("recipe_name", "cfg_name"),
    [
        ("general/ptq/mxfp4_mlp_weight_only", "MXFP4_MLP_WEIGHT_ONLY_CFG"),
        ("general/ptq/nvfp4_mlp_weight_only", "NVFP4_MLP_WEIGHT_ONLY_CFG"),
    ],
)
def test_mlp_weight_only_recipe_matches_its_mtq_cfg(recipe_name, cfg_name):
    # examples/gpt-oss migrated from --quant_cfg <CFG> to --recipe <recipe>; pin the
    # equality so the recipe and the mtq constant cannot drift apart silently.
    recipe_cfg = load_recipe(recipe_name).quantize.model_dump(exclude_unset=True)
    mtq_cfg = QuantizeConfig(**getattr(mtq, cfg_name)).model_dump(exclude_unset=True)
    assert recipe_cfg == mtq_cfg
