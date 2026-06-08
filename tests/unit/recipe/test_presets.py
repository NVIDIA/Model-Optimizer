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
under the model/KV dirs must load into a usable ``quant_cfg`` dict, deprecation
aliases must resolve to their canonical preset, and the KV ``none`` sentinel must
not collide with a discovered preset. A single malformed preset YAML would
otherwise break ``import modelopt.recipe.presets`` (and every PTQ example).
"""

import pytest

from modelopt.recipe import presets
from modelopt.torch.opt.config_loader import BUILTIN_CONFIG_ROOT


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


def test_aliases_resolve_to_their_canonical_preset():
    for alias, target in presets.QFORMAT_ALIASES.items():
        assert alias in presets.QUANT_CFG_CHOICES, f"alias {alias!r} not exposed"
        assert target in presets.QUANT_CFG_CHOICES, f"alias target {target!r} missing"
        assert presets.QUANT_CFG_CHOICES[alias] == presets.QUANT_CFG_CHOICES[target]


def test_kv_none_sentinel_is_not_a_discovered_preset():
    # The scripts branch on ``kv_cache_qformat != KV_CACHE_NONE``; a real preset named
    # "none" would make that branch ambiguous.
    assert presets.KV_CACHE_NONE not in presets.KV_QUANT_CFG_CHOICES


def test_load_quant_cfg_choices_rejects_stale_alias():
    with pytest.raises(ValueError, match="does-not-exist"):
        presets.load_quant_cfg_choices(
            presets.MODEL_QUANT_PRESET_DIR, {"bad_alias": "does-not-exist"}
        )
