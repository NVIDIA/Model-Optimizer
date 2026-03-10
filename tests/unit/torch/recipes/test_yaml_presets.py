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

"""Tests for YAML-based preset loading (Tier 1)."""

import pytest
import yaml

from modelopt.torch.recipes.schema.presets import (
    _deep_merge,
    _load_yaml_with_bases,
    _normalize_yaml_config,
)


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
        assert "c" not in base["a"]  # base not mutated
        assert result["a"] == {"b": 1, "c": 2}


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


class TestYamlBaseInheritance:
    def test_base_resolution(self, tmp_path):
        """Test __base__ inheritance with temp YAML files."""
        # Create base fragment
        base_yml = tmp_path / "configs" / "base.yml"
        base_yml.parent.mkdir(parents=True)
        base_yml.write_text(
            yaml.dump(
                {
                    "quant_cfg": {
                        "default": {"enable": False},
                        "*lm_head*": {"enable": False},
                    }
                }
            )
        )

        # Create quantizer fragment
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

        # Create algorithm fragment
        algo_yml = tmp_path / "configs" / "algo_max.yml"
        algo_yml.write_text(yaml.dump({"algorithm": "max"}))

        # Create composed recipe with __base__
        recipe_yml = tmp_path / "recipe.yml"
        recipe_yml.write_text(
            yaml.dump(
                {
                    "__base__": [
                        "configs/base",
                        "configs/fp8",
                        "configs/algo_max",
                    ]
                }
            )
        )

        result = _load_yaml_with_bases(recipe_yml, tmp_path)

        # Should have merged all three fragments
        assert result["algorithm"] == "max"
        assert result["quant_cfg"]["default"] == {"enable": False}
        assert result["quant_cfg"]["*lm_head*"] == {"enable": False}
        assert result["quant_cfg"]["*weight_quantizer"]["num_bits"] == [4, 3]
        assert result["quant_cfg"]["*input_quantizer"]["num_bits"] == [4, 3]

    def test_override_in_leaf(self, tmp_path):
        """Test that leaf file values override base values."""
        base_yml = tmp_path / "base.yml"
        base_yml.write_text(yaml.dump({"algorithm": "max", "extra": "keep"}))

        leaf_yml = tmp_path / "leaf.yml"
        leaf_yml.write_text(yaml.dump({"__base__": ["base"], "algorithm": "awq_lite"}))

        result = _load_yaml_with_bases(leaf_yml, tmp_path)
        assert result["algorithm"] == "awq_lite"
        assert result["extra"] == "keep"

    def test_missing_base_raises(self, tmp_path):
        recipe_yml = tmp_path / "recipe.yml"
        recipe_yml.write_text(yaml.dump({"__base__": ["nonexistent"]}))

        with pytest.raises(FileNotFoundError, match="nonexistent"):
            _load_yaml_with_bases(recipe_yml, tmp_path)


class TestPresetSource:
    def test_source_is_valid(self):
        from modelopt.torch.recipes.schema.presets import get_preset_source

        source = get_preset_source()
        assert source in ("yaml", "live", "bundled")

    def test_list_presets_nonempty(self):
        from modelopt.torch.recipes.schema.presets import list_presets

        presets = list_presets()
        assert len(presets) > 0
        assert "fp8" in presets

    def test_get_preset_returns_deep_copy(self):
        from modelopt.torch.recipes.schema.presets import get_preset

        p1 = get_preset("fp8")
        p2 = get_preset("fp8")
        assert p1 == p2
        assert p1 is not p2  # deep copy

    def test_unknown_preset_raises(self):
        from modelopt.torch.recipes.schema.presets import get_preset

        with pytest.raises(KeyError, match="Unknown preset"):
            get_preset("nonexistent_preset_xyz")
