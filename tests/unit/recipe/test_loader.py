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
    "general/ptq/nvfp4_default-none_kv_gptq",
    "general/ptq/nvfp4_experts_only-fp8_kv",
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


# ---------------------------------------------------------------------------
# imports — named config snippet resolution
# ---------------------------------------------------------------------------


def test_import_resolves_cfg_reference(tmp_path):
    """$import in cfg is replaced with the imported config dict."""
    (tmp_path / "fp8.yml").write_text("num_bits: e4m3\naxis:\n")
    recipe_file = tmp_path / "recipe.yml"
    recipe_file.write_text(
        f"imports:\n"
        f"  fp8: {tmp_path / 'fp8.yml'}\n"
        f"metadata:\n"
        f"  recipe_type: ptq\n"
        f"quantize:\n"
        f"  algorithm: max\n"
        f"  quant_cfg:\n"
        f"    - quantizer_name: '*weight_quantizer'\n"
        f"      cfg:\n"
        f"        $import: fp8\n"
    )
    recipe = load_recipe(recipe_file)
    entry = recipe.quantize["quant_cfg"][0]
    assert entry["cfg"] == {"num_bits": (4, 3), "axis": None}


def test_import_same_name_used_twice(tmp_path):
    """The same import can be referenced in multiple quant_cfg entries."""
    (tmp_path / "fp8.yml").write_text("num_bits: e4m3\naxis:\n")
    recipe_file = tmp_path / "recipe.yml"
    recipe_file.write_text(
        f"imports:\n"
        f"  fp8: {tmp_path / 'fp8.yml'}\n"
        f"metadata:\n"
        f"  recipe_type: ptq\n"
        f"quantize:\n"
        f"  algorithm: max\n"
        f"  quant_cfg:\n"
        f"    - quantizer_name: '*weight_quantizer'\n"
        f"      cfg:\n"
        f"        $import: fp8\n"
        f"    - quantizer_name: '*input_quantizer'\n"
        f"      cfg:\n"
        f"        $import: fp8\n"
    )
    recipe = load_recipe(recipe_file)
    assert recipe.quantize["quant_cfg"][0]["cfg"] == recipe.quantize["quant_cfg"][1]["cfg"]


def test_import_multiple_snippets(tmp_path):
    """Multiple imports with different names resolve independently."""
    (tmp_path / "fp8.yml").write_text("num_bits: e4m3\naxis:\n")
    (tmp_path / "nvfp4.yml").write_text("num_bits: e2m1\nblock_sizes:\n  -1: 16\n")
    recipe_file = tmp_path / "recipe.yml"
    recipe_file.write_text(
        f"imports:\n"
        f"  fp8: {tmp_path / 'fp8.yml'}\n"
        f"  nvfp4: {tmp_path / 'nvfp4.yml'}\n"
        f"metadata:\n"
        f"  recipe_type: ptq\n"
        f"quantize:\n"
        f"  algorithm: max\n"
        f"  quant_cfg:\n"
        f"    - quantizer_name: '*weight_quantizer'\n"
        f"      cfg:\n"
        f"        $import: nvfp4\n"
        f"    - quantizer_name: '*[kv]_bmm_quantizer'\n"
        f"      cfg:\n"
        f"        $import: fp8\n"
    )
    recipe = load_recipe(recipe_file)
    assert recipe.quantize["quant_cfg"][0]["cfg"]["num_bits"] == (2, 1)
    assert recipe.quantize["quant_cfg"][1]["cfg"]["num_bits"] == (4, 3)


def test_import_inline_cfg_not_affected(tmp_path):
    """Inline dict cfg entries without $import are not touched."""
    (tmp_path / "fp8.yml").write_text("num_bits: e4m3\naxis:\n")
    recipe_file = tmp_path / "recipe.yml"
    recipe_file.write_text(
        f"imports:\n"
        f"  fp8: {tmp_path / 'fp8.yml'}\n"
        f"metadata:\n"
        f"  recipe_type: ptq\n"
        f"quantize:\n"
        f"  algorithm: max\n"
        f"  quant_cfg:\n"
        f"    - quantizer_name: '*weight_quantizer'\n"
        f"      cfg:\n"
        f"        $import: fp8\n"
        f"    - quantizer_name: '*input_quantizer'\n"
        f"      cfg:\n"
        f"        num_bits: 8\n"
        f"        axis: 0\n"
    )
    recipe = load_recipe(recipe_file)
    assert recipe.quantize["quant_cfg"][1]["cfg"] == {"num_bits": 8, "axis": 0}


def test_import_unknown_reference_raises(tmp_path):
    """Referencing an undefined import name raises ValueError."""
    recipe_file = tmp_path / "recipe.yml"
    recipe_file.write_text(
        "imports:\n"
        "  fp8: configs/numerics/fp8\n"
        "metadata:\n"
        "  recipe_type: ptq\n"
        "quantize:\n"
        "  algorithm: max\n"
        "  quant_cfg:\n"
        "    - quantizer_name: '*weight_quantizer'\n"
        "      cfg:\n"
        "        $import: nonexistent\n"
    )
    with pytest.raises(ValueError, match=r"Unknown \$import reference"):
        load_recipe(recipe_file)


def test_import_empty_path_raises(tmp_path):
    """Import with empty config path raises ValueError."""
    recipe_file = tmp_path / "recipe.yml"
    recipe_file.write_text(
        "imports:\n"
        "  fp8:\n"
        "metadata:\n"
        "  recipe_type: ptq\n"
        "quantize:\n"
        "  algorithm: max\n"
        "  quant_cfg: []\n"
    )
    with pytest.raises(ValueError, match="empty config path"):
        load_recipe(recipe_file)


def test_import_not_a_dict_raises(tmp_path):
    """Import section that is not a dict raises ValueError."""
    recipe_file = tmp_path / "recipe.yml"
    recipe_file.write_text(
        "imports:\n"
        "  - configs/numerics/fp8\n"
        "metadata:\n"
        "  recipe_type: ptq\n"
        "quantize:\n"
        "  algorithm: max\n"
        "  quant_cfg: []\n"
    )
    with pytest.raises(ValueError, match="must be a dict"):
        load_recipe(recipe_file)


def test_import_no_imports_section(tmp_path):
    """Recipes without imports load normally."""
    recipe_file = tmp_path / "recipe.yml"
    recipe_file.write_text(
        "metadata:\n"
        "  recipe_type: ptq\n"
        "quantize:\n"
        "  algorithm: max\n"
        "  quant_cfg:\n"
        "    - quantizer_name: '*'\n"
        "      enable: false\n"
    )
    recipe = load_recipe(recipe_file)
    assert recipe.quantize["quant_cfg"][0]["enable"] is False


def test_import_builtin_recipe_with_imports():
    """Built-in recipes using $import load and resolve correctly."""
    recipe = load_recipe("general/ptq/fp8_default-fp8_kv")
    assert recipe.quantize
    # Verify $import was resolved — cfg should be a dict, not a {$import: ...} marker
    for entry in recipe.quantize["quant_cfg"]:
        if "cfg" in entry and entry["cfg"] is not None:
            assert "$import" not in entry["cfg"], f"Unresolved $import in {entry}"


def test_import_entry_single_element_list(tmp_path):
    """$import splices a single-element list snippet into quant_cfg."""
    (tmp_path / "disable.yml").write_text("- quantizer_name: '*'\n  enable: false\n")
    recipe_file = tmp_path / "recipe.yml"
    recipe_file.write_text(
        f"imports:\n"
        f"  disable_all: {tmp_path / 'disable.yml'}\n"
        f"metadata:\n"
        f"  recipe_type: ptq\n"
        f"quantize:\n"
        f"  algorithm: max\n"
        f"  quant_cfg:\n"
        f"    - $import: disable_all\n"
    )
    recipe = load_recipe(recipe_file)
    assert len(recipe.quantize["quant_cfg"]) == 1
    assert recipe.quantize["quant_cfg"][0] == {"quantizer_name": "*", "enable": False}


def test_import_entry_non_list_raises(tmp_path):
    """$import in quant_cfg list position raises if snippet is not a list."""
    (tmp_path / "disable.yml").write_text("quantizer_name: '*'\nenable: false\n")
    recipe_file = tmp_path / "recipe.yml"
    recipe_file.write_text(
        f"imports:\n"
        f"  disable_all: {tmp_path / 'disable.yml'}\n"
        f"metadata:\n"
        f"  recipe_type: ptq\n"
        f"quantize:\n"
        f"  algorithm: max\n"
        f"  quant_cfg:\n"
        f"    - $import: disable_all\n"
    )
    with pytest.raises(ValueError, match="must resolve to a list"):
        load_recipe(recipe_file)


def test_import_entry_list_splice(tmp_path):
    """$import as a quant_cfg list entry splices a list-valued snippet."""
    (tmp_path / "disables.yml").write_text(
        "- quantizer_name: '*lm_head*'\n  enable: false\n"
        "- quantizer_name: '*router*'\n  enable: false\n"
    )
    recipe_file = tmp_path / "recipe.yml"
    recipe_file.write_text(
        f"imports:\n"
        f"  disables: {tmp_path / 'disables.yml'}\n"
        f"metadata:\n"
        f"  recipe_type: ptq\n"
        f"quantize:\n"
        f"  algorithm: max\n"
        f"  quant_cfg:\n"
        f"    - quantizer_name: '*'\n"
        f"      enable: false\n"
        f"    - $import: disables\n"
    )
    recipe = load_recipe(recipe_file)
    assert len(recipe.quantize["quant_cfg"]) == 3
    assert recipe.quantize["quant_cfg"][1]["quantizer_name"] == "*lm_head*"
    assert recipe.quantize["quant_cfg"][2]["quantizer_name"] == "*router*"


def test_import_entry_sibling_keys_raises(tmp_path):
    """$import as a list entry with sibling keys raises ValueError."""
    (tmp_path / "disable.yml").write_text("- quantizer_name: '*'\n  enable: false\n")
    recipe_file = tmp_path / "recipe.yml"
    recipe_file.write_text(
        f"imports:\n"
        f"  disable_all: {tmp_path / 'disable.yml'}\n"
        f"metadata:\n"
        f"  recipe_type: ptq\n"
        f"quantize:\n"
        f"  algorithm: max\n"
        f"  quant_cfg:\n"
        f"    - $import: disable_all\n"
        f"      quantizer_name: '*extra*'\n"
    )
    with pytest.raises(ValueError, match="must be the only key"):
        load_recipe(recipe_file)


def test_import_cfg_extend(tmp_path):
    """$import in cfg with extra non-conflicting keys extends the snippet."""
    (tmp_path / "fp8.yml").write_text("num_bits: e4m3\n")
    recipe_file = tmp_path / "recipe.yml"
    recipe_file.write_text(
        f"imports:\n"
        f"  fp8: {tmp_path / 'fp8.yml'}\n"
        f"metadata:\n"
        f"  recipe_type: ptq\n"
        f"quantize:\n"
        f"  algorithm: max\n"
        f"  quant_cfg:\n"
        f"    - quantizer_name: '*weight_quantizer'\n"
        f"      cfg:\n"
        f"        $import: fp8\n"
        f"        axis: 0\n"
    )
    recipe = load_recipe(recipe_file)
    cfg = recipe.quantize["quant_cfg"][0]["cfg"]
    assert cfg == {"num_bits": (4, 3), "axis": 0}


def test_import_cfg_inline_overrides_import(tmp_path):
    """Inline keys override imported values (highest precedence)."""
    (tmp_path / "fp8.yml").write_text("num_bits: e4m3\naxis:\n")
    recipe_file = tmp_path / "recipe.yml"
    recipe_file.write_text(
        f"imports:\n"
        f"  fp8: {tmp_path / 'fp8.yml'}\n"
        f"metadata:\n"
        f"  recipe_type: ptq\n"
        f"quantize:\n"
        f"  algorithm: max\n"
        f"  quant_cfg:\n"
        f"    - quantizer_name: '*weight_quantizer'\n"
        f"      cfg:\n"
        f"        $import: fp8\n"
        f"        num_bits: 8\n"
    )
    recipe = load_recipe(recipe_file)
    cfg = recipe.quantize["quant_cfg"][0]["cfg"]
    # inline num_bits: 8 overrides imported num_bits: e4m3 → (4,3)
    assert cfg["num_bits"] == 8
    # imported axis: None is preserved (no inline override)
    assert cfg["axis"] is None


def test_import_cfg_multi_import(tmp_path):
    """$import with a list of names merges non-overlapping snippets."""
    (tmp_path / "bits.yml").write_text("num_bits: e4m3\n")
    (tmp_path / "axis.yml").write_text("axis: 0\n")
    recipe_file = tmp_path / "recipe.yml"
    recipe_file.write_text(
        f"imports:\n"
        f"  bits: {tmp_path / 'bits.yml'}\n"
        f"  axis: {tmp_path / 'axis.yml'}\n"
        f"metadata:\n"
        f"  recipe_type: ptq\n"
        f"quantize:\n"
        f"  algorithm: max\n"
        f"  quant_cfg:\n"
        f"    - quantizer_name: '*weight_quantizer'\n"
        f"      cfg:\n"
        f"        $import: [bits, axis]\n"
    )
    recipe = load_recipe(recipe_file)
    cfg = recipe.quantize["quant_cfg"][0]["cfg"]
    assert cfg == {"num_bits": (4, 3), "axis": 0}


def test_import_cfg_multi_import_later_overrides_earlier(tmp_path):
    """In $import list, later snippets override earlier ones on key conflicts."""
    (tmp_path / "a.yml").write_text("num_bits: e4m3\naxis: 0\n")
    (tmp_path / "b.yml").write_text("num_bits: 8\n")
    recipe_file = tmp_path / "recipe.yml"
    recipe_file.write_text(
        f"imports:\n"
        f"  a: {tmp_path / 'a.yml'}\n"
        f"  b: {tmp_path / 'b.yml'}\n"
        f"metadata:\n"
        f"  recipe_type: ptq\n"
        f"quantize:\n"
        f"  algorithm: max\n"
        f"  quant_cfg:\n"
        f"    - quantizer_name: '*weight_quantizer'\n"
        f"      cfg:\n"
        f"        $import: [a, b]\n"
    )
    recipe = load_recipe(recipe_file)
    cfg = recipe.quantize["quant_cfg"][0]["cfg"]
    # b overrides a's num_bits; a's axis is preserved
    assert cfg["num_bits"] == 8
    assert cfg["axis"] == 0


def test_import_cfg_multi_import_with_extend(tmp_path):
    """$import list + inline keys all merge without conflicts."""
    (tmp_path / "bits.yml").write_text("num_bits: e4m3\n")
    (tmp_path / "scale.yml").write_text("scale_bits: e8m0\n")
    recipe_file = tmp_path / "recipe.yml"
    recipe_file.write_text(
        f"imports:\n"
        f"  bits: {tmp_path / 'bits.yml'}\n"
        f"  scale: {tmp_path / 'scale.yml'}\n"
        f"metadata:\n"
        f"  recipe_type: ptq\n"
        f"quantize:\n"
        f"  algorithm: max\n"
        f"  quant_cfg:\n"
        f"    - quantizer_name: '*weight_quantizer'\n"
        f"      cfg:\n"
        f"        $import: [bits, scale]\n"
        f"        axis: 0\n"
    )
    recipe = load_recipe(recipe_file)
    cfg = recipe.quantize["quant_cfg"][0]["cfg"]
    assert cfg == {"num_bits": (4, 3), "scale_bits": (8, 0), "axis": 0}


def test_import_dir_format(tmp_path):
    """Imports in recipe.yml work with the directory recipe format."""
    (tmp_path / "fp8.yml").write_text("num_bits: e4m3\naxis:\n")
    (tmp_path / "recipe.yml").write_text(
        f"imports:\n"
        f"  fp8: {tmp_path / 'fp8.yml'}\n"
        f"metadata:\n"
        f"  recipe_type: ptq\n"
        f"  description: Dir with imports.\n"
    )
    (tmp_path / "quantize.yml").write_text(
        "algorithm: max\n"
        "quant_cfg:\n"
        "  - quantizer_name: '*weight_quantizer'\n"
        "    cfg:\n"
        "      $import: fp8\n"
    )
    recipe = load_recipe(tmp_path)
    assert recipe.quantize["quant_cfg"][0]["cfg"] == {"num_bits": (4, 3), "axis": None}


# ---------------------------------------------------------------------------
# imports — multi-document snippets
# ---------------------------------------------------------------------------


def test_import_multi_document_list_snippet(tmp_path):
    """List snippet using multi-document YAML (imports --- content) resolves $import."""
    (tmp_path / "fp8.yml").write_text("num_bits: e4m3\n")
    (tmp_path / "kv.yaml").write_text(
        f"imports:\n"
        f"  fp8: {tmp_path / 'fp8.yml'}\n"
        f"---\n"
        f"- quantizer_name: '*[kv]_bmm_quantizer'\n"
        f"  cfg:\n"
        f"    $import: fp8\n"
    )
    recipe_file = tmp_path / "recipe.yml"
    recipe_file.write_text(
        f"imports:\n"
        f"  kv: {tmp_path / 'kv.yaml'}\n"
        f"metadata:\n"
        f"  recipe_type: ptq\n"
        f"quantize:\n"
        f"  algorithm: max\n"
        f"  quant_cfg:\n"
        f"    - $import: kv\n"
    )
    recipe = load_recipe(recipe_file)
    assert len(recipe.quantize["quant_cfg"]) == 1
    assert recipe.quantize["quant_cfg"][0]["quantizer_name"] == "*[kv]_bmm_quantizer"
    assert recipe.quantize["quant_cfg"][0]["cfg"] == {"num_bits": (4, 3)}


def test_import_builtin_fp8_kv_snippet():
    """Built-in fp8_kv snippet uses multi-document format and resolves correctly."""
    recipe = load_recipe("general/ptq/fp8_default-fp8_kv")
    kv_entries = [
        e for e in recipe.quantize["quant_cfg"] if e.get("quantizer_name") == "*[kv]_bmm_quantizer"
    ]
    assert len(kv_entries) == 1
    assert kv_entries[0]["cfg"]["num_bits"] == (4, 3)


# ---------------------------------------------------------------------------
# imports — recursive resolution and cycle detection
# ---------------------------------------------------------------------------


def test_import_recursive(tmp_path):
    """A snippet can itself import other snippets."""
    (tmp_path / "base.yml").write_text("num_bits: e4m3\n")
    (tmp_path / "mid.yml").write_text(
        f"imports:\n  base: {tmp_path / 'base.yml'}\nnum_bits:\n  $import: base\n"
    )
    recipe_file = tmp_path / "recipe.yml"
    recipe_file.write_text(
        f"imports:\n"
        f"  mid: {tmp_path / 'mid.yml'}\n"
        f"metadata:\n"
        f"  recipe_type: ptq\n"
        f"quantize:\n"
        f"  algorithm: max\n"
        f"  quant_cfg:\n"
        f"    - quantizer_name: '*weight_quantizer'\n"
        f"      cfg:\n"
        f"        $import: mid\n"
    )
    recipe = load_recipe(recipe_file)
    cfg = recipe.quantize["quant_cfg"][0]["cfg"]
    # mid.yml resolved "num_bits: {$import: base}" → base.yml content
    assert cfg["num_bits"] == {"num_bits": (4, 3)}


def test_import_circular_raises(tmp_path):
    """Circular imports are detected and raise ValueError."""
    (tmp_path / "a.yml").write_text(f"imports:\n  b: {tmp_path / 'b.yml'}\nnum_bits: 8\n")
    (tmp_path / "b.yml").write_text(f"imports:\n  a: {tmp_path / 'a.yml'}\nnum_bits: 4\n")
    recipe_file = tmp_path / "recipe.yml"
    recipe_file.write_text(
        f"imports:\n"
        f"  a: {tmp_path / 'a.yml'}\n"
        f"metadata:\n"
        f"  recipe_type: ptq\n"
        f"quantize:\n"
        f"  algorithm: max\n"
        f"  quant_cfg: []\n"
    )
    with pytest.raises(ValueError, match="Circular import"):
        load_recipe(recipe_file)


def test_import_cross_file_same_name_no_conflict(tmp_path):
    """Same import name in parent and child resolve independently (scoped)."""
    (tmp_path / "fp8.yml").write_text("num_bits: e4m3\n")
    (tmp_path / "nvfp4.yml").write_text("num_bits: e2m1\nblock_sizes:\n  -1: 16\n")
    (tmp_path / "child.yml").write_text(
        f"imports:\n  fmt: {tmp_path / 'nvfp4.yml'}\nweight_format: fmt\n"
    )
    recipe_file = tmp_path / "recipe.yml"
    recipe_file.write_text(
        f"imports:\n"
        f"  fmt: {tmp_path / 'fp8.yml'}\n"
        f"  child: {tmp_path / 'child.yml'}\n"
        f"metadata:\n"
        f"  recipe_type: ptq\n"
        f"quantize:\n"
        f"  algorithm: max\n"
        f"  quant_cfg:\n"
        f"    - quantizer_name: '*weight_quantizer'\n"
        f"      cfg:\n"
        f"        $import: fmt\n"
    )
    recipe = load_recipe(recipe_file)
    # Parent's "fmt" resolves to fp8 (e4m3), not child's nvfp4
    cfg = recipe.quantize["quant_cfg"][0]["cfg"]
    assert cfg == {"num_bits": (4, 3)}
