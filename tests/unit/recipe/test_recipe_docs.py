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

"""Consistency checks between the shipped recipe YAML files and modelopt_recipes/ptq.md.

These tests force recipe additions, removals, and renames to be reflected in the
PTQ recipe guide so the doc never drifts from the files on disk.
"""

import re
from importlib.resources import files
from pathlib import Path

RECIPES_DIR = Path(str(files("modelopt_recipes")))
GENERAL_PTQ_DIR = RECIPES_DIR / "general" / "ptq"
PTQ_MD = RECIPES_DIR / "ptq.md"


def _ptq_md_text() -> str:
    return PTQ_MD.read_text(encoding="utf-8")


def _general_ptq_stems() -> list[str]:
    return sorted(p.stem for p in GENERAL_PTQ_DIR.glob("*.yaml"))


def test_every_general_ptq_recipe_is_documented():
    """Every general/ptq/*.yaml recipe must be mentioned (backticked) in ptq.md."""
    doc = _ptq_md_text()
    missing = [stem for stem in _general_ptq_stems() if f"`{stem}`" not in doc]
    assert not missing, (
        f"Recipes under modelopt_recipes/general/ptq/ are missing from "
        f"modelopt_recipes/ptq.md: {missing}. When adding a recipe, add a row to "
        "the 'shipped recipes' table in ptq.md (and describe any new scheme, "
        "KV mode, or calibration variant in the matching section)."
    )


def test_documented_general_ptq_recipes_exist_on_disk():
    """Every recipe row in the ptq.md shipped-recipes table must exist on disk.

    Catches renames/removals that leave stale rows behind. Rows are identified
    by a first cell that is a single backticked token, which only occurs in the
    shipped-recipes table.
    """
    doc = _ptq_md_text()
    documented = re.findall(r"^\| `([^`]+)` \|", doc, flags=re.MULTILINE)
    assert documented, "ptq.md shipped-recipes table not found — was it reformatted?"
    stale = [name for name in documented if not (GENERAL_PTQ_DIR / f"{name}.yaml").is_file()]
    assert not stale, (
        f"modelopt_recipes/ptq.md documents general/ptq recipes that do not exist "
        f"on disk: {stale}. Update the 'shipped recipes' table after renaming or "
        "removing a recipe."
    )


def test_general_ptq_recipe_count_in_ptq_md():
    """The 'All N general/ptq/ recipes' summary line must match the file count."""
    doc = _ptq_md_text()
    match = re.search(r"All (\d+) <code>general/ptq/</code> recipes", doc)
    assert match, (
        "Could not find the 'All N <code>general/ptq/</code> recipes' summary "
        "line in modelopt_recipes/ptq.md — keep that phrasing so this check can "
        "verify the recipe count."
    )
    documented_count = int(match.group(1))
    actual_count = len(_general_ptq_stems())
    assert documented_count == actual_count, (
        f"modelopt_recipes/ptq.md says 'All {documented_count} general/ptq/ "
        f"recipes' but modelopt_recipes/general/ptq/ contains {actual_count} "
        "recipes. Update the count and the table in ptq.md."
    )


def test_every_model_specific_ptq_dir_is_mentioned():
    """Every model dir under huggingface/ with PTQ recipes must appear in ptq.md.

    The identifier checked is the directory containing the ptq/ folder — the
    HF model_type (e.g. ``gemma4``), a nested checkpoint dir (e.g.
    ``Step3.5-Flash``), or a models/<org>/<checkpoint> leaf (e.g.
    ``Nemotron-3-Nano-4B``).
    """
    doc = _ptq_md_text()
    hf_dir = RECIPES_DIR / "huggingface"
    model_dirs = sorted(
        {yaml_path.parent.parent.name for yaml_path in hf_dir.glob("**/ptq/*.yaml")}
    )
    assert model_dirs, "No model-specific PTQ recipes found under huggingface/"
    missing = [name for name in model_dirs if name not in doc]
    assert not missing, (
        f"Model-specific PTQ recipe folders are missing from "
        f"modelopt_recipes/ptq.md: {missing}. Add them to the model-specific "
        "recipes section (kinds table and/or the matching subsection)."
    )
