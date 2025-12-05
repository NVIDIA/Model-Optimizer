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

import pytest

from modelopt.recipe.config import ModelOptPTQRecipe, RecipeType
from modelopt.recipe.loader import load_config, load_recipe

# ---------------------------------------------------------------------------
# Static YAML fixtures
# ---------------------------------------------------------------------------

CFG_AB = """\
a: 1
b: 2
"""

CFG_XY = """\
x: 1
y: 2
"""

CFG_YZ = """\
y: 20
z: 30
"""

CFG_KEY_VAL = """\
key: val
"""

CFG_ALGO_MAX = """\
algorithm: max
"""

CFG_ALGO_MAX_EXTRA = """\
algorithm: max
extra: original
"""

CFG_QUANT_CFG_EMPTY = """\
quant_cfg: {}
"""

CFG_B1 = """\
a: 1
b: 2
"""

CFG_B2 = """\
b: 20
c: 30
"""

CFG_LEAF_FROM_BASE = """\
val: from_base
"""

CFG_LEAF_AB = """\
a: 1
b: 2
"""

CFG_LEAF_DEEP = """\
val: deep
"""

# ---------------------------------------------------------------------------
# Error-case YAML fixtures
# ---------------------------------------------------------------------------

CFG_INVALID_BASE_SCALAR = """\
__base__:
  model_quant: not_a_list
"""

CFG_RECIPE_MISSING_TYPE = """\
metadata:
  description: Missing recipe_type.
model_quant: {}
kv_quant: {}
"""

CFG_RECIPE_MISSING_MODEL_QUANT = """\
metadata:
  recipe_type: ptq
kv_quant: {}
"""

CFG_RECIPE_MISSING_KV_QUANT = """\
metadata:
  recipe_type: ptq
model_quant: {}
"""

CFG_RECIPE_UNSUPPORTED_TYPE = """\
metadata:
  recipe_type: unknown_type
"""

# ---------------------------------------------------------------------------
# Directory-format YAML fixtures
# ---------------------------------------------------------------------------

DIR_RECIPE_DESCRIPTOR = """\
recipe_type: ptq
description: Dir format test.
"""

DIR_MODEL_QUANT = """\
algorithm: max
quant_cfg: {}
"""

DIR_KV_QUANT = """\
quant_cfg: {}
"""

# ---------------------------------------------------------------------------
# load_config — __base__ inheritance
# ---------------------------------------------------------------------------


def test_load_config_no_base(tmp_path):
    """A config without __base__ is returned as-is."""
    (tmp_path / "cfg.yml").write_text(CFG_AB)
    assert load_config(tmp_path / "cfg.yml") == {"a": 1, "b": 2}


def test_load_config_flat_base_list(tmp_path):
    """Flat-list __base__ merges base configs into current level."""
    base = tmp_path / "base.yml"
    base.write_text(CFG_AB)
    child = tmp_path / "child.yml"
    child.write_text(f"""\
__base__:
  - {base}
b: 99
""")
    assert load_config(child) == {"a": 1, "b": 99}


def test_load_config_flat_base_multiple(tmp_path):
    """Multiple flat-list bases are merged left-to-right; later entries win."""
    b1 = tmp_path / "b1.yml"
    b1.write_text(CFG_XY)
    b2 = tmp_path / "b2.yml"
    b2.write_text(CFG_YZ)
    child = tmp_path / "child.yml"
    child.write_text(f"""\
__base__:
  - {b1}
  - {b2}
""")
    assert load_config(child) == {"x": 1, "y": 20, "z": 30}


def test_load_config_flat_base_chained(tmp_path):
    """Transitive flat-list inheritance: child → middle → grandparent."""
    grandparent = tmp_path / "grandparent.yml"
    grandparent.write_text(CFG_XY)
    middle = tmp_path / "middle.yml"
    middle.write_text(f"""\
__base__:
  - {grandparent}
y: 20
z: 30
""")
    child = tmp_path / "child.yml"
    child.write_text(f"""\
__base__:
  - {middle}
z: 300
""")
    assert load_config(child) == {"x": 1, "y": 20, "z": 300}


def test_load_config_dict_base(tmp_path):
    """Dict-style __base__ merges bases into each named section."""
    mq_base = tmp_path / "mq_base.yml"
    mq_base.write_text(CFG_ALGO_MAX)
    kv_base = tmp_path / "kv_base.yml"
    kv_base.write_text(CFG_QUANT_CFG_EMPTY)
    recipe = tmp_path / "recipe.yml"
    recipe.write_text(f"""\
__base__:
  model_quant:
    - {mq_base}
  kv_quant:
    - {kv_base}
""")
    result = load_config(recipe)
    assert result["model_quant"] == {"algorithm": "max"}
    assert result["kv_quant"] == {"quant_cfg": {}}


def test_load_config_dict_base_with_override(tmp_path):
    """Inline overrides are merged on top of dict-style __base__ bases."""
    mq_base = tmp_path / "mq_base.yml"
    mq_base.write_text(CFG_ALGO_MAX_EXTRA)
    recipe = tmp_path / "recipe.yml"
    recipe.write_text(f"""\
__base__:
  model_quant:
    - {mq_base}
model_quant:
  extra: overridden
""")
    result = load_config(recipe)
    assert result["model_quant"]["algorithm"] == "max"
    assert result["model_quant"]["extra"] == "overridden"


def test_load_config_dict_base_multiple_per_section(tmp_path):
    """Dict-style __base__ with multiple bases per section; later entries win."""
    b1 = tmp_path / "b1.yml"
    b1.write_text(CFG_B1)
    b2 = tmp_path / "b2.yml"
    b2.write_text(CFG_B2)
    recipe = tmp_path / "recipe.yml"
    recipe.write_text(f"""\
__base__:
  section:
    - {b1}
    - {b2}
""")
    assert load_config(recipe)["section"] == {"a": 1, "b": 20, "c": 30}


def test_load_config_dict_base_nested(tmp_path):
    """Dict-style __base__ supports arbitrary nesting depth."""
    leaf = tmp_path / "leaf.yml"
    leaf.write_text(CFG_LEAF_FROM_BASE)
    recipe = tmp_path / "recipe.yml"
    recipe.write_text(f"""\
__base__:
  section:
    subsection:
      - {leaf}
""")
    assert load_config(recipe)["section"]["subsection"] == {"val": "from_base"}


def test_load_config_dict_base_nested_with_override(tmp_path):
    """Nested dict-style __base__ applies inline overrides at the leaf level."""
    leaf = tmp_path / "leaf.yml"
    leaf.write_text(CFG_LEAF_AB)
    recipe = tmp_path / "recipe.yml"
    recipe.write_text(f"""\
__base__:
  section:
    subsection:
      - {leaf}
section:
  subsection:
    b: 99
""")
    assert load_config(recipe)["section"]["subsection"] == {"a": 1, "b": 99}


def test_load_config_dict_base_three_levels(tmp_path):
    """Dict-style __base__ resolves correctly at three levels of nesting."""
    leaf = tmp_path / "leaf.yml"
    leaf.write_text(CFG_LEAF_DEEP)
    recipe = tmp_path / "recipe.yml"
    recipe.write_text(f"""\
__base__:
  a:
    b:
      c:
        - {leaf}
""")
    assert load_config(recipe)["a"]["b"]["c"] == {"val": "deep"}


def test_load_config_suffix_probe(tmp_path):
    """load_config finds a .yml file when suffix is omitted from a string path."""
    (tmp_path / "mycfg.yml").write_text(CFG_KEY_VAL)
    assert load_config(str(tmp_path / "mycfg")) == {"key": "val"}


def test_load_config_missing_file_raises(tmp_path):
    """load_config raises ValueError for a path that does not exist."""
    with pytest.raises(ValueError, match="Cannot find config file"):
        load_config(str(tmp_path / "nonexistent"))


def test_load_config_dict_base_non_list_raises(tmp_path):
    """Dict-style __base__ with a scalar (non-list, non-dict) value raises ValueError."""
    recipe = tmp_path / "recipe.yml"
    recipe.write_text(CFG_INVALID_BASE_SCALAR)
    with pytest.raises(ValueError, match="must be lists"):
        load_config(recipe)


# ---------------------------------------------------------------------------
# load_recipe — built-in PTQ recipes
# ---------------------------------------------------------------------------


def test_load_recipe_builtin_with_suffix():
    """load_recipe loads a built-in PTQ recipe given the full YAML path."""
    recipe = load_recipe("general/ptq/fp8_default-fp8_kv.yml")
    assert recipe.recipe_type == RecipeType.PTQ
    assert isinstance(recipe, ModelOptPTQRecipe)
    assert recipe.model_quant is not None
    assert recipe.kv_quant is not None


def test_load_recipe_builtin_without_suffix():
    """load_recipe resolves the .yml suffix automatically."""
    recipe = load_recipe("general/ptq/fp8_default-fp8_kv")
    assert recipe.recipe_type == RecipeType.PTQ


def test_load_recipe_builtin_description():
    """The description field is loaded from the YAML metadata."""
    recipe = load_recipe("general/ptq/fp8_default-fp8_kv.yml")
    assert isinstance(recipe.description, str)
    assert len(recipe.description) > 0


_BUILTIN_PTQ_RECIPES = [
    "general/ptq/fp8_default-fp8_kv",
    "general/ptq/fp8_per_channel_per_token-fp8_kv",
    "general/ptq/fp8_2d_blockwise_weight_only-fp8_kv",
    "general/ptq/int4_awq-fp8_kv",
    "general/ptq/int4_blockwise_weight_only-fp8_kv",
    "general/ptq/int8_default-fp8_kv",
    "general/ptq/int8_smoothquant-fp8_kv",
    "general/ptq/int8_weight_only-fp8_kv",
    "general/ptq/mamba_moe_fp8_aggressive-fp8_kv",
    "general/ptq/mamba_moe_fp8_conservative-fp8_kv",
    "general/ptq/mamba_moe_nvfp4_aggressive-fp8_kv",
    "general/ptq/mamba_moe_nvfp4_conservative-fp8_kv",
    "general/ptq/mxfp4_default-fp8_kv",
    "general/ptq/mxfp4_mlp_weight_only-fp8_kv",
    "general/ptq/mxfp6_default-fp8_kv",
    "general/ptq/mxfp8_default-fp8_kv",
    "general/ptq/mxint8_default-fp8_kv",
    "general/ptq/nvfp4_awq_clip-fp8_kv",
    "general/ptq/nvfp4_awq_full-fp8_kv",
    "general/ptq/nvfp4_awq_lite-fp8_kv",
    "general/ptq/nvfp4_default-fp8_kv",
    "general/ptq/nvfp4_fp8_mha-fp8_kv",
    "general/ptq/nvfp4_mlp_only-fp8_kv",
    "general/ptq/nvfp4_mlp_weight_only-fp8_kv",
    "general/ptq/nvfp4_omlp_only-fp8_kv",
    "general/ptq/nvfp4_svdquant_default-fp8_kv",
    "general/ptq/nvfp4_w4a4_weight_local_hessian-fp8_kv",
    "general/ptq/nvfp4_w4a4_weight_mse_fp8_sweep-fp8_kv",
    "general/ptq/w4a8_awq_beta-fp8_kv",
    "general/ptq/w4a8_mxfp4_fp8-fp8_kv",
    "general/ptq/w4a8_nvfp4_fp8-fp8_kv",
]


@pytest.mark.parametrize("recipe_path", _BUILTIN_PTQ_RECIPES)
def test_load_recipe_all_builtins(recipe_path):
    """Smoke-test: every built-in PTQ recipe loads without error and has model_quant."""
    recipe = load_recipe(recipe_path)
    assert recipe.recipe_type == RecipeType.PTQ
    assert isinstance(recipe, ModelOptPTQRecipe)
    assert recipe.model_quant is not None


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


def test_load_recipe_missing_model_quant_raises(tmp_path):
    """load_recipe raises ValueError when model_quant is absent for a PTQ recipe."""
    bad = tmp_path / "bad.yml"
    bad.write_text(CFG_RECIPE_MISSING_MODEL_QUANT)
    with pytest.raises(ValueError, match="model_quant"):
        load_recipe(bad)


def test_load_recipe_missing_kv_quant_raises(tmp_path):
    """load_recipe raises ValueError when kv_quant is absent for a PTQ recipe."""
    bad = tmp_path / "bad.yml"
    bad.write_text(CFG_RECIPE_MISSING_KV_QUANT)
    with pytest.raises(ValueError, match="kv_quant"):
        load_recipe(bad)


def test_load_recipe_unsupported_type_raises(tmp_path):
    """load_recipe raises ValueError for an unknown recipe_type."""
    bad = tmp_path / "bad.yml"
    bad.write_text(CFG_RECIPE_UNSUPPORTED_TYPE)
    with pytest.raises(ValueError, match="Unsupported recipe type"):
        load_recipe(bad)


# ---------------------------------------------------------------------------
# load_recipe — directory format
# ---------------------------------------------------------------------------


def test_load_recipe_dir(tmp_path):
    """load_recipe loads a recipe from a directory containing separate YAML files."""
    (tmp_path / "recipe.yml").write_text(DIR_RECIPE_DESCRIPTOR)
    (tmp_path / "model_quant.yml").write_text(DIR_MODEL_QUANT)
    (tmp_path / "kv_quant.yml").write_text(DIR_KV_QUANT)
    recipe = load_recipe(tmp_path)
    assert recipe.recipe_type == RecipeType.PTQ
    assert recipe.description == "Dir format test."


def test_load_recipe_dir_missing_descriptor_raises(tmp_path):
    """load_recipe raises ValueError when recipe.yml is absent from the directory."""
    with pytest.raises(ValueError, match="recipe descriptor"):
        load_recipe(tmp_path)
