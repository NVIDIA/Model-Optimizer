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

"""Tests for preset registry (schema/presets.py)."""

from collections import Counter
from pathlib import Path

import pytest
import yaml

from modelopt.torch.recipes.schema._bundled_presets import BUNDLED_PRESETS
from modelopt.torch.recipes.schema.presets import (
    _PRESET_YAML_MAP,
    _deep_merge,
    _load_recipe_from_yaml,
    _load_yaml_with_bases,
    _normalize_yaml_config,
    get_preset,
    get_preset_source,
    list_presets,
)

# ── Helpers ──


def _write_yaml(path: Path, data: dict):
    """Write a YAML file, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=True)


# Common base fragment shared by all presets.
_COMMON_BASE = {
    "quant_cfg": {
        "default": {"enable": False},
        "*lm_head*": {"enable": False},
        "*output_layer*": {"enable": False},
        "*router*": {"enable": False},
        "*block_sparse_moe.gate*": {"enable": False},
        "*mlp.gate.*": {"enable": False},
        "*mlp.shared_expert_gate.*": {"enable": False},
        "*proj_out.*": {"enable": False},
        "*linear_attn.conv1d*": {"enable": False},
        "*mixer.conv1d*": {"enable": False},
        "output.*": {"enable": False},
        "nn.BatchNorm1d": {"*": {"enable": False}},
        "nn.BatchNorm2d": {"*": {"enable": False}},
        "nn.BatchNorm3d": {"*": {"enable": False}},
        "nn.LeakyReLU": {"*": {"enable": False}},
    }
}

_FP8_KV_FRAGMENT = {
    "quant_cfg": {
        "*k_proj*input_quantizer": {"axis": None, "num_bits": [4, 3]},
        "*v_proj*input_quantizer": {"axis": None, "num_bits": [4, 3]},
    }
}


def _setup_base_fragments(recipes_root: Path):
    """Create the shared base and KV cache fragments."""
    _write_yaml(recipes_root / "fragments" / "base.yml", _COMMON_BASE)
    _write_yaml(recipes_root / "fragments" / "fp8_kv.yml", _FP8_KV_FRAGMENT)
    _write_yaml(recipes_root / "fragments" / "algo_max.yml", {"algorithm": "max"})
    _write_yaml(
        recipes_root / "fragments" / "algo_awq_lite.yml",
        {"algorithm": {"method": "awq_lite", "alpha_step": 0.1}},
    )
    _write_yaml(
        recipes_root / "fragments" / "algo_smoothquant.yml",
        {"algorithm": "smoothquant"},
    )


def _setup_preset(
    recipes_root: Path,
    recipe_dir: str,
    model_quant_bases: list[str],
    model_quant_override: dict,
    kv_quant_bases: list[str] | None = None,
    kv_quant_override: dict | None = None,
):
    """Create a composed recipe directory with model_quant.yml and optional kv_quant.yml."""
    recipe_path = recipes_root / recipe_dir
    recipe_path.mkdir(parents=True, exist_ok=True)

    model_data = {"__base__": model_quant_bases}
    model_data.update(model_quant_override)
    _write_yaml(recipe_path / "model_quant.yml", model_data)

    if kv_quant_bases or kv_quant_override:
        kv_data = {}
        if kv_quant_bases:
            kv_data["__base__"] = kv_quant_bases
        if kv_quant_override:
            kv_data.update(kv_quant_override)
        _write_yaml(recipe_path / "kv_quant.yml", kv_data)


@pytest.fixture
def recipes_root(tmp_path):
    """Set up a fake modelopt_recipes/ directory tree with 5 presets."""
    root = tmp_path
    _setup_base_fragments(root)

    # fp8
    _write_yaml(
        root / "fragments" / "fp8_quantizer.yml",
        {
            "quant_cfg": {
                "*weight_quantizer": {"axis": None, "num_bits": [4, 3]},
                "*input_quantizer": {"axis": None, "num_bits": [4, 3]},
            }
        },
    )
    _setup_preset(
        root,
        "general/ptq/fp8_default-fp8_kv",
        model_quant_bases=["fragments/base", "fragments/fp8_quantizer", "fragments/algo_max"],
        model_quant_override={},
        kv_quant_bases=["fragments/fp8_kv"],
    )

    # int8
    _write_yaml(
        root / "fragments" / "int8_quantizer.yml",
        {
            "quant_cfg": {
                "*weight_quantizer": {"axis": 0, "num_bits": 8},
                "*input_quantizer": {"axis": None, "num_bits": 8},
            }
        },
    )
    _setup_preset(
        root,
        "general/ptq/int8_default-fp8_kv",
        model_quant_bases=["fragments/base", "fragments/int8_quantizer", "fragments/algo_max"],
        model_quant_override={},
        kv_quant_bases=["fragments/fp8_kv"],
    )

    # int4_awq
    _write_yaml(
        root / "fragments" / "int4_wo_quantizer.yml",
        {
            "quant_cfg": {
                "*weight_quantizer": {
                    "enable": True,
                    "num_bits": 4,
                    "block_sizes": {"-1": 128, "type": "static"},
                },
                "*input_quantizer": {"enable": False},
            }
        },
    )
    _setup_preset(
        root,
        "general/ptq/int4_awq-fp8_kv",
        model_quant_bases=[
            "fragments/base",
            "fragments/int4_wo_quantizer",
            "fragments/algo_awq_lite",
        ],
        model_quant_override={},
        kv_quant_bases=["fragments/fp8_kv"],
    )

    # nvfp4
    _write_yaml(
        root / "fragments" / "nvfp4_quantizer.yml",
        {
            "quant_cfg": {
                "*weight_quantizer": {
                    "axis": None,
                    "enable": True,
                    "num_bits": [2, 1],
                    "block_sizes": {"-1": 16, "scale_bits": [4, 3], "type": "dynamic"},
                },
                "*input_quantizer": {
                    "axis": None,
                    "enable": True,
                    "num_bits": [2, 1],
                    "block_sizes": {"-1": 16, "scale_bits": [4, 3], "type": "dynamic"},
                },
            }
        },
    )
    _setup_preset(
        root,
        "general/ptq/nvfp4_default-fp8_kv",
        model_quant_bases=["fragments/base", "fragments/nvfp4_quantizer", "fragments/algo_max"],
        model_quant_override={},
        kv_quant_bases=["fragments/fp8_kv"],
    )

    # int8_sq
    _setup_preset(
        root,
        "general/ptq/int8_smoothquant-fp8_kv",
        model_quant_bases=[
            "fragments/base",
            "fragments/int8_quantizer",
            "fragments/algo_smoothquant",
        ],
        model_quant_override={},
        kv_quant_bases=["fragments/fp8_kv"],
    )

    return root


# ── Unit tests: _deep_merge ──


class TestDeepMerge:
    def test_simple_merge(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        base = {"quant_cfg": {"default": {"enable": False}, "*weight*": {"num_bits": 8}}}
        override = {"quant_cfg": {"*weight*": {"axis": 0}}}
        result = _deep_merge(base, override)
        assert result["quant_cfg"]["default"] == {"enable": False}
        assert result["quant_cfg"]["*weight*"] == {"num_bits": 8, "axis": 0}

    def test_override_replaces_non_dict(self):
        base = {"algorithm": "max"}
        override = {"algorithm": "awq_lite"}
        result = _deep_merge(base, override)
        assert result["algorithm"] == "awq_lite"

    def test_no_mutation(self):
        base = {"a": {"b": 1}}
        override = {"a": {"c": 2}}
        result = _deep_merge(base, override)
        assert "c" not in base["a"]
        assert result["a"] == {"b": 1, "c": 2}


# ── Unit tests: _normalize_yaml_config ──


class TestNormalizeConfig:
    def test_lists_to_tuples(self):
        config = {"quant_cfg": {"*weight*": {"num_bits": [4, 3]}}}
        result = _normalize_yaml_config(config)
        assert result["quant_cfg"]["*weight*"]["num_bits"] == (4, 3)

    def test_nested_lists(self):
        config = {"block_sizes": {"scale_bits": [4, 3]}}
        result = _normalize_yaml_config(config)
        assert result["block_sizes"]["scale_bits"] == (4, 3)

    def test_scalars_unchanged(self):
        config = {"algorithm": "max", "enable": True, "axis": 0}
        result = _normalize_yaml_config(config)
        assert result == config


# ── Unit tests: __base__ inheritance ──


class TestYamlBaseInheritance:
    def test_base_resolution(self, tmp_path):
        """Test __base__ inheritance with temp YAML files."""
        base_yml = tmp_path / "configs" / "base.yml"
        base_yml.parent.mkdir(parents=True)
        base_yml.write_text(
            yaml.dump({"quant_cfg": {"default": {"enable": False}, "*lm_head*": {"enable": False}}})
        )
        quant_yml = tmp_path / "configs" / "fp8.yml"
        quant_yml.write_text(
            yaml.dump(
                {
                    "quant_cfg": {
                        "*weight_quantizer": {"num_bits": [4, 3], "axis": None},
                        "*input_quantizer": {"num_bits": [4, 3], "axis": None},
                    }
                }
            )
        )
        algo_yml = tmp_path / "configs" / "algo_max.yml"
        algo_yml.write_text(yaml.dump({"algorithm": "max"}))

        recipe_yml = tmp_path / "recipe.yml"
        recipe_yml.write_text(
            yaml.dump({"__base__": ["configs/base", "configs/fp8", "configs/algo_max"]})
        )

        result = _load_yaml_with_bases(recipe_yml, tmp_path)
        assert result["algorithm"] == "max"
        assert result["quant_cfg"]["default"] == {"enable": False}
        assert result["quant_cfg"]["*weight_quantizer"]["num_bits"] == [4, 3]

    def test_base_override_order(self, tmp_path):
        """Later bases override earlier bases."""
        _write_yaml(tmp_path / "a.yml", {"algorithm": "max", "x": 1})
        _write_yaml(tmp_path / "b.yml", {"algorithm": "awq_lite", "y": 2})
        _write_yaml(tmp_path / "c.yml", {"__base__": ["a", "b"]})
        result = _load_yaml_with_bases(tmp_path / "c.yml", tmp_path)
        assert result["algorithm"] == "awq_lite"
        assert result["x"] == 1
        assert result["y"] == 2

    def test_leaf_overrides_bases(self, tmp_path):
        """Leaf file values override all bases."""
        _write_yaml(tmp_path / "base.yml", {"algorithm": "max", "extra": "keep"})
        _write_yaml(
            tmp_path / "leaf.yml",
            {"__base__": ["base"], "algorithm": "awq_lite"},
        )
        result = _load_yaml_with_bases(tmp_path / "leaf.yml", tmp_path)
        assert result["algorithm"] == "awq_lite"
        assert result["extra"] == "keep"

    def test_deep_merge_preserves_nested(self, tmp_path):
        """Deep merge combines nested quant_cfg entries."""
        _write_yaml(
            tmp_path / "base.yml",
            {"quant_cfg": {"*weight_quantizer": {"num_bits": 8}}},
        )
        _write_yaml(
            tmp_path / "extra.yml",
            {"quant_cfg": {"*weight_quantizer": {"axis": 0}}},
        )
        _write_yaml(tmp_path / "composed.yml", {"__base__": ["base", "extra"]})
        result = _load_yaml_with_bases(tmp_path / "composed.yml", tmp_path)
        wq = result["quant_cfg"]["*weight_quantizer"]
        assert wq["num_bits"] == 8
        assert wq["axis"] == 0

    def test_missing_base_raises(self, tmp_path):
        recipe_yml = tmp_path / "recipe.yml"
        recipe_yml.write_text(yaml.dump({"__base__": ["nonexistent"]}))
        with pytest.raises(FileNotFoundError, match="nonexistent"):
            _load_yaml_with_bases(recipe_yml, tmp_path)


# ── E2E tests: fake modelopt_recipes/ → _load_recipe_from_yaml ──


class TestYamlPresetE2E:
    def test_fp8_loads_correctly(self, recipes_root):
        """FP8 preset resolves __base__ chain and produces expected config."""
        config = _load_recipe_from_yaml("general/ptq/fp8_default-fp8_kv", recipes_root)
        assert config["algorithm"] == "max"
        qcfg = config["quant_cfg"]
        assert qcfg["default"] == {"enable": False}
        assert qcfg["*lm_head*"] == {"enable": False}
        assert qcfg["*weight_quantizer"]["num_bits"] == (4, 3)
        assert qcfg["*input_quantizer"]["num_bits"] == (4, 3)
        assert "*k_proj*input_quantizer" in qcfg

    def test_int8_loads_correctly(self, recipes_root):
        """INT8 preset: integer num_bits, per-axis weights."""
        config = _load_recipe_from_yaml("general/ptq/int8_default-fp8_kv", recipes_root)
        assert config["algorithm"] == "max"
        qcfg = config["quant_cfg"]
        assert qcfg["*weight_quantizer"]["num_bits"] == 8
        assert qcfg["*weight_quantizer"]["axis"] == 0
        assert qcfg["*input_quantizer"]["num_bits"] == 8

    def test_int4_awq_loads_correctly(self, recipes_root):
        """INT4 AWQ: dict algorithm, weight-only, block_sizes."""
        config = _load_recipe_from_yaml("general/ptq/int4_awq-fp8_kv", recipes_root)
        assert config["algorithm"]["method"] == "awq_lite"
        qcfg = config["quant_cfg"]
        assert qcfg["*weight_quantizer"]["num_bits"] == 4
        assert qcfg["*input_quantizer"] == {"enable": False}

    def test_nvfp4_loads_correctly(self, recipes_root):
        """NVFP4: tuple num_bits (2,1), block_sizes with scale_bits tuple."""
        config = _load_recipe_from_yaml("general/ptq/nvfp4_default-fp8_kv", recipes_root)
        assert config["algorithm"] == "max"
        wq = config["quant_cfg"]["*weight_quantizer"]
        assert wq["num_bits"] == (2, 1)
        assert wq["block_sizes"]["scale_bits"] == (4, 3)
        assert wq["block_sizes"]["type"] == "dynamic"

    def test_int8_sq_loads_correctly(self, recipes_root):
        """INT8 SmoothQuant: same quantizers as INT8 but different algorithm."""
        config = _load_recipe_from_yaml("general/ptq/int8_smoothquant-fp8_kv", recipes_root)
        assert config["algorithm"] == "smoothquant"
        assert config["quant_cfg"]["*weight_quantizer"]["num_bits"] == 8

    def test_kv_cache_merging(self, recipes_root):
        """KV cache entries are merged into model config."""
        config = _load_recipe_from_yaml("general/ptq/fp8_default-fp8_kv", recipes_root)
        qcfg = config["quant_cfg"]
        assert "*k_proj*input_quantizer" in qcfg
        assert "*v_proj*input_quantizer" in qcfg

    def test_normalization_converts_lists_to_tuples(self):
        """_normalize_yaml_config converts all lists to tuples recursively."""
        raw = {"quant_cfg": {"*wq": {"num_bits": [4, 3], "block_sizes": {"scale_bits": [4, 3]}}}}
        result = _normalize_yaml_config(raw)
        assert result["quant_cfg"]["*wq"]["num_bits"] == (4, 3)
        assert result["quant_cfg"]["*wq"]["block_sizes"]["scale_bits"] == (4, 3)


# ── Preset map consistency ──


class TestPresetMapConsistency:
    def test_yaml_map_covers_all_bundled_presets(self):
        """Every bundled preset should have a corresponding YAML map entry."""
        missing = set(BUNDLED_PRESETS.keys()) - set(_PRESET_YAML_MAP.keys())
        assert not missing, f"Bundled presets missing from _PRESET_YAML_MAP: {sorted(missing)}"

    def test_yaml_map_has_no_extra_presets(self):
        """YAML map should not have presets missing from bundled snapshot."""
        extra = set(_PRESET_YAML_MAP.keys()) - set(BUNDLED_PRESETS.keys())
        assert not extra, f"_PRESET_YAML_MAP has presets not in BUNDLED_PRESETS: {sorted(extra)}"

    def test_yaml_map_paths_follow_convention(self):
        """All YAML map paths should follow general/ptq/<name>-fp8_kv pattern."""
        for name, path in _PRESET_YAML_MAP.items():
            assert path.startswith("general/ptq/"), f"'{name}' has non-standard path: {path}"
            assert path.endswith("-fp8_kv"), f"'{name}' path doesn't end with -fp8_kv: {path}"

    def test_no_duplicate_paths(self):
        """No two presets should map to the same directory (except known aliases)."""
        counts = Counter(_PRESET_YAML_MAP.values())
        allowed_aliases = {"general/ptq/nvfp4_awq_lite-fp8_kv": {"nvfp4_awq", "nvfp4_awq_lite"}}
        for path, count in counts.items():
            if count > 1:
                names = {n for n, p in _PRESET_YAML_MAP.items() if p == path}
                if path in allowed_aliases:
                    assert names == allowed_aliases[path], f"Unexpected aliases for {path}: {names}"
                else:
                    pytest.fail(f"Unexpected duplicate path {path}: {names}")


# ── Public API ──


class TestPresetAPI:
    def test_source_is_valid(self):
        source = get_preset_source()
        assert source in ("yaml", "live", "bundled")

    def test_list_presets_nonempty(self):
        presets = list_presets()
        assert len(presets) > 0
        assert "fp8" in presets

    def test_get_preset_returns_deep_copy(self):
        p1 = get_preset("fp8")
        p2 = get_preset("fp8")
        assert p1 == p2
        assert p1 is not p2

    def test_unknown_preset_raises(self):
        with pytest.raises(KeyError, match="Unknown preset"):
            get_preset("nonexistent_preset_xyz")
