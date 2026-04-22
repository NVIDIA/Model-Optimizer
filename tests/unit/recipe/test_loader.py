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

"""Unit tests for modelopt.recipe.loader and modelopt.recipe.loader.load_config."""

import re

import pytest
import yaml

from modelopt.recipe.config import ModelOptPTQRecipe, RecipeType
from modelopt.recipe.loader import load_config, load_recipe

# ---------------------------------------------------------------------------
# Static YAML fixtures
# ---------------------------------------------------------------------------

CFG_AB = """\
a: 1
b: 2
"""

CFG_KEY_VAL = """\
key: val
"""

CFG_RECIPE_MISSING_TYPE = """\
metadata:
  description: Missing recipe_type.
quantize: {}
"""

CFG_RECIPE_MISSING_quantize = """\
metadata:
  recipe_type: ptq
"""

CFG_RECIPE_UNSUPPORTED_TYPE = """\
metadata:
  recipe_type: unknown_type
"""

# ---------------------------------------------------------------------------
# Directory-format YAML fixtures
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# load_config — basic behaviour
# ---------------------------------------------------------------------------


def test_load_config_plain(tmp_path):
    """A plain config is returned as-is."""
    (tmp_path / "cfg.yml").write_text(CFG_AB)
    assert load_config(tmp_path / "cfg.yml") == {"a": 1, "b": 2}


def test_load_config_suffix_probe(tmp_path):
    """load_config finds a .yml file when suffix is omitted from a string path."""
    (tmp_path / "mycfg.yml").write_text(CFG_KEY_VAL)
    assert load_config(str(tmp_path / "mycfg")) == {"key": "val"}


def test_load_config_missing_file_raises(tmp_path):
    """load_config raises ValueError for a path that does not exist."""
    with pytest.raises(ValueError, match="Cannot find config file"):
        load_config(str(tmp_path / "nonexistent"))


# ---------------------------------------------------------------------------
# load_recipe — built-in PTQ recipes
# ---------------------------------------------------------------------------


def test_load_recipe_builtin_with_suffix():
    """load_recipe loads a built-in PTQ recipe given the full YAML path."""
    recipe = load_recipe("general/ptq/fp8_default-fp8_kv.yaml")
    assert recipe.recipe_type == RecipeType.PTQ
    assert isinstance(recipe, ModelOptPTQRecipe)
    assert recipe.quantize


def test_load_recipe_builtin_without_suffix():
    """load_recipe resolves the .yaml suffix automatically."""
    recipe = load_recipe("general/ptq/fp8_default-fp8_kv")
    assert recipe.recipe_type == RecipeType.PTQ


def test_load_recipe_builtin_description():
    """The description field is loaded from the YAML metadata."""
    recipe = load_recipe("general/ptq/fp8_default-fp8_kv.yaml")
    assert isinstance(recipe.description, str)
    assert len(recipe.description) > 0


_BUILTIN_PTQ_RECIPES = [
    "general/ptq/fp8_default-fp8_kv",
    "general/ptq/nvfp4_default-fp8_kv",
    "general/ptq/nvfp4_mlp_only-fp8_kv",
    "general/ptq/nvfp4_omlp_only-fp8_kv",
]


@pytest.mark.parametrize("recipe_path", _BUILTIN_PTQ_RECIPES)
def test_load_recipe_all_builtins(recipe_path):
    """Smoke-test: every built-in PTQ recipe loads without error and has quantize."""
    recipe = load_recipe(recipe_path)
    assert recipe.recipe_type == RecipeType.PTQ
    assert isinstance(recipe, ModelOptPTQRecipe)
    assert recipe.quantize


# ---------------------------------------------------------------------------
# load_recipe — error cases
# ---------------------------------------------------------------------------


def test_load_recipe_missing_raises(tmp_path):
    """load_recipe raises ValueError for a path that doesn't exist."""
    with pytest.raises(ValueError):
        load_recipe(str(tmp_path / "does_not_exist.yml"))


def test_load_recipe_missing_recipe_type_raises(tmp_path):
    """load_recipe raises ValueError when metadata.recipe_type is absent."""
    bad = tmp_path / "bad.yml"
    bad.write_text(CFG_RECIPE_MISSING_TYPE)
    with pytest.raises(ValueError, match="recipe_type"):
        load_recipe(bad)


def test_load_recipe_missing_quantize_raises(tmp_path):
    """load_recipe raises ValueError when quantize is absent for a PTQ recipe."""
    bad = tmp_path / "bad.yml"
    bad.write_text(CFG_RECIPE_MISSING_quantize)
    with pytest.raises(ValueError, match="quantize"):
        load_recipe(bad)


def test_load_recipe_unsupported_type_raises(tmp_path):
    """load_recipe raises ValueError for an unknown recipe_type."""
    bad = tmp_path / "bad.yml"
    bad.write_text(CFG_RECIPE_UNSUPPORTED_TYPE)
    with pytest.raises(ValueError, match="Unsupported recipe type"):
        load_recipe(bad)


# ---------------------------------------------------------------------------
# load_config — !include / !concat custom tags
# ---------------------------------------------------------------------------


def test_include_relative_path(tmp_path):
    """!include resolves a path relative to the including file."""
    (tmp_path / "child.yml").write_text("a: 1\nb: 2\n")
    (tmp_path / "parent.yml").write_text("payload: !include child.yml\n")
    assert load_config(tmp_path / "parent.yml") == {"payload": {"a": 1, "b": 2}}


def test_include_suffix_optional(tmp_path):
    """!include probes .yml / .yaml when the suffix is omitted."""
    (tmp_path / "child.yaml").write_text("k: v\n")
    (tmp_path / "parent.yml").write_text("payload: !include child\n")
    assert load_config(tmp_path / "parent.yml") == {"payload": {"k": "v"}}


def test_include_builtin_recipe_fragment(tmp_path):
    """!include resolves built-in fragments under modelopt_recipes/."""
    (tmp_path / "parent.yml").write_text("disable_all: !include _base/disable_all\n")
    out = load_config(tmp_path / "parent.yml")
    assert out == {"disable_all": [{"quantizer_name": "*", "enable": False}]}


def test_include_inside_concat(tmp_path):
    """!include nested under !concat splices its sequence; inline mappings are appended."""
    (tmp_path / "head.yml").write_text("- {quantizer_name: '*', enable: false}\n")
    (tmp_path / "tail.yml").write_text("- {quantizer_name: '*lm_head*', enable: false}\n")
    (tmp_path / "parent.yml").write_text(
        "quant_cfg: !concat\n"
        "  - !include head.yml\n"
        "  - {quantizer_name: '*weight_quantizer', cfg: {num_bits: 8}}\n"
        "  - !include tail.yml\n"
    )
    out = load_config(tmp_path / "parent.yml")
    assert out == {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 8}},
            {"quantizer_name": "*lm_head*", "enable": False},
        ]
    }


def test_concat_passes_through_non_sequence_items(tmp_path):
    """!concat splices sequences and passes mappings/scalars through as-is."""
    (tmp_path / "parent.yml").write_text(
        "values: !concat\n  - [1, 2]\n  - 3\n  - {a: 1}\n  - [4, 5]\n"
    )
    out = load_config(tmp_path / "parent.yml")
    assert out == {"values": [1, 2, 3, {"a": 1}, 4, 5]}


def test_concat_requires_sequence_node(tmp_path):
    """!concat must be applied to a YAML sequence, not a scalar/mapping."""
    (tmp_path / "parent.yml").write_text("values: !concat 42\n")
    with pytest.raises(yaml.YAMLError, match="!concat expects a sequence"):
        load_config(tmp_path / "parent.yml")


def test_include_cycle_detected(tmp_path):
    """A self-including chain raises ValueError instead of recursing forever."""
    (tmp_path / "a.yml").write_text("payload: !include b.yml\n")
    (tmp_path / "b.yml").write_text("payload: !include a.yml\n")
    with pytest.raises(ValueError, match="Cycle detected"):
        load_config(tmp_path / "a.yml")


def test_include_missing_target_raises(tmp_path):
    """!include with an unresolvable target raises ValueError."""
    (tmp_path / "parent.yml").write_text("payload: !include nope\n")
    with pytest.raises(ValueError, match="Cannot find config file"):
        load_config(tmp_path / "parent.yml")


def test_include_preserves_exmy_parsing(tmp_path):
    """ExMy strings inside an included fragment are still converted to tuples."""
    (tmp_path / "child.yml").write_text("cfg:\n  num_bits: e4m3\n")
    (tmp_path / "parent.yml").write_text("entry: !include child.yml\n")
    out = load_config(tmp_path / "parent.yml")
    assert out == {"entry": {"cfg": {"num_bits": (4, 3)}}}


# ---------------------------------------------------------------------------
# load_recipe — directory format
# ---------------------------------------------------------------------------


def test_load_recipe_dir(tmp_path):
    """load_recipe loads a recipe from a directory with recipe.yml + quantize.yml."""
    (tmp_path / "recipe.yml").write_text(
        "metadata:\n  recipe_type: ptq\n  description: Dir test.\n"
    )
    (tmp_path / "quantize.yml").write_text("algorithm: max\nquant_cfg: []\n")
    recipe = load_recipe(tmp_path)
    assert recipe.recipe_type == RecipeType.PTQ
    assert recipe.description == "Dir test."
    assert recipe.quantize.algorithm == "max"
    assert recipe.quantize.quant_cfg == []


def test_load_recipe_dir_missing_recipe_raises(tmp_path):
    """load_recipe raises ValueError when recipe.yml is absent from the directory."""
    (tmp_path / "quantize.yml").write_text("algorithm: max\nquant_cfg: {}\n")
    with pytest.raises(ValueError, match="recipe descriptor"):
        load_recipe(tmp_path)


def test_load_recipe_dir_missing_quantize_raises(tmp_path):
    """load_recipe raises ValueError when quantize.yml is absent from the directory."""
    (tmp_path / "recipe.yml").write_text("metadata:\n  recipe_type: ptq\n")
    with pytest.raises(ValueError, match="quantize"):
        load_recipe(tmp_path)


# ---------------------------------------------------------------------------
# YAML recipe consistency — built-in general/ptq files match config.py dicts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("yaml_path", "model_cfg_name", "kv_cfg_name"),
    [
        ("general/ptq/fp8_default-fp8_kv.yaml", "FP8_DEFAULT_CFG", "FP8_KV_CFG"),
        ("general/ptq/nvfp4_default-fp8_kv.yaml", "NVFP4_DEFAULT_CFG", "FP8_KV_CFG"),
        ("general/ptq/nvfp4_mlp_only-fp8_kv.yaml", "NVFP4_MLP_ONLY_CFG", "FP8_KV_CFG"),
        ("general/ptq/nvfp4_omlp_only-fp8_kv.yaml", "NVFP4_OMLP_ONLY_CFG", "FP8_KV_CFG"),
    ],
)
def test_general_ptq_yaml_matches_config_dicts(yaml_path, model_cfg_name, kv_cfg_name):
    """Each general/ptq YAML's quant_cfg list matches the merged Python config dicts."""
    import json

    import modelopt.torch.quantization.config as qcfg
    from modelopt.torch.quantization.config import normalize_quant_cfg_list

    model_cfg = getattr(qcfg, model_cfg_name)
    kv_cfg = getattr(qcfg, kv_cfg_name)
    yaml_data = load_config(yaml_path)

    def _normalize_fpx(val):
        """Normalize FPx representations to a canonical ``[E, M]`` list.

        Python configs may use tuple form ``(E, M)`` or string alias ``"eEmM"``;
        YAML always uses the string form.  Both are converted to ``[E, M]`` so the
        comparison is representation-agnostic.
        """
        if isinstance(val, str):
            m = re.fullmatch(r"e(\d+)m(\d+)", val)
            if m:
                return [int(m.group(1)), int(m.group(2))]
        if isinstance(val, tuple) and len(val) == 2 and all(isinstance(x, int) for x in val):
            return list(val)
        if isinstance(val, dict):
            return {str(k): _normalize_fpx(v) for k, v in val.items()}
        return val

    def _normalize_entries(raw_entries):
        """Normalize a raw quant_cfg list to a canonical, JSON-serialisable form."""
        entries = normalize_quant_cfg_list(list(raw_entries))
        result = []
        for entry in entries:
            e = {k: v for k, v in entry.items() if v is not None}
            if "cfg" in e and e["cfg"] is not None:
                e["cfg"] = _normalize_fpx(e["cfg"])
            result.append(e)
        return result

    def _sort_key(entry):
        return json.dumps(entry, sort_keys=True, default=str)

    python_entries = _normalize_entries(model_cfg["quant_cfg"] + kv_cfg["quant_cfg"])
    yaml_entries = _normalize_entries(yaml_data["quantize"]["quant_cfg"])

    assert sorted(python_entries, key=_sort_key) == sorted(yaml_entries, key=_sort_key)
    assert model_cfg["algorithm"] == yaml_data["quantize"]["algorithm"]
