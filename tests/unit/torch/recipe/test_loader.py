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

CFG_KEY_VAL = """\
key: val
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
