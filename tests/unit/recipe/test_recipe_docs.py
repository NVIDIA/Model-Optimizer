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

import pytest

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
    """Every model-specific PTQ recipe must be identifiable in ptq.md.

    ``huggingface/<model_type>/ptq/`` recipes are checked by their ``model_type``
    (e.g. ``gemma4``); ``models/<org>/<model_id>/ptq/`` recipes are checked by their
    full ``<org>/<model_id>`` hub path (e.g. ``nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16``), so
    the org — the whole point of the top-level tier — is verified too and an org
    re-key (e.g. ``step3p5`` → ``stepfun-ai``) can't silently drift from the doc.
    """
    doc = _ptq_md_text()
    # model_type recipes: huggingface/<model_type>/ptq/<recipe>.yaml -> <model_type>
    hf_ids = {p.parent.parent.name for p in (RECIPES_DIR / "huggingface").glob("**/ptq/*.yaml")}
    # checkpoint recipes: models/<org>/<model_id>/ptq/<recipe>.yaml -> <org>/<model_id>
    model_ids = {
        f"{p.parent.parent.parent.name}/{p.parent.parent.name}"
        for p in (RECIPES_DIR / "models").glob("**/ptq/*.yaml")
    }
    identifiers = sorted(hf_ids | model_ids)
    assert identifiers, "No model-specific PTQ recipes found under huggingface/ or models/"
    missing = [name for name in identifiers if name not in doc]
    assert not missing, (
        f"Model-specific PTQ recipe folders are missing from "
        f"modelopt_recipes/ptq.md: {missing}. Add them to the model-specific "
        "recipes section (kinds table and/or the matching subsection)."
    )


def test_checkpoint_recipes_live_in_the_top_level_models_tier():
    """Lock in the model_type-vs-checkpoint split.

    Checkpoint-mirror recipes belong at ``models/<org>/<model_id>/``; ``huggingface/``
    holds only per-``model_type`` recipes. ``huggingface/models`` is kept as a
    backward-compatibility **symlink** to the top-level ``models/`` tier, so the old
    ``--recipe huggingface/models/<org>/<model_id>/...`` paths still resolve; it must
    stay a symlink that points at ``../models`` and never become a real directory that
    holds recipes. A checkpoint recipe nested under a ``model_type`` (e.g.
    ``huggingface/<model_type>/<checkpoint>/<task>/``) still fails loudly here instead
    of silently shipping both tiers — e.g. on a bad merge that re-adds the old layout.
    """
    hf = RECIPES_DIR / "huggingface"
    models = RECIPES_DIR / "models"
    hf_models = hf / "models"
    assert hf_models.is_symlink(), (
        "huggingface/models must be a symlink to the top-level modelopt_recipes/models/ "
        "tier (a backward-compat alias for the old --recipe paths), not a real directory."
    )
    assert hf_models.resolve() == models.resolve(), (
        f"huggingface/models must resolve to the top-level models/ tier; resolves to "
        f"{hf_models.resolve()} instead of {models.resolve()}."
    )
    # Every recipe under huggingface/ must be <model_type>/<task>/<file> (3 parts);
    # anything deeper is a checkpoint nested under a model_type and belongs in models/.
    # Skip the huggingface/models symlink so the models/ recipes it aliases (4 parts)
    # aren't miscounted as nested here.
    nested = sorted(
        str(p.relative_to(RECIPES_DIR))
        for ext in ("*.yaml", "*.yml")
        for p in hf.glob(f"**/{ext}")
        if hf_models not in p.parents and len(p.relative_to(hf).parts) != 3
    )
    assert not nested, (
        f"Recipes under huggingface/ must be <model_type>/<task>/<file>; found nested "
        f"paths (a checkpoint recipe belongs under models/<org>/<model_id>/): {nested}"
    )
    # Every recipe under models/ must be <org>/<model_id>/<task>/<file> (4 parts) so the
    # path is exactly the model-hub path; a different depth breaks that convention.
    misplaced = sorted(
        str(p.relative_to(RECIPES_DIR))
        for ext in ("*.yaml", "*.yml")
        for p in models.glob(f"**/{ext}")
        if len(p.relative_to(models).parts) != 4
    )
    assert not misplaced, (
        f"Recipes under models/ must be <org>/<model_id>/<task>/<file>; found: {misplaced}"
    )


def test_launcher_yaml_recipe_paths_resolve():
    """Every modelopt_recipes recipe path a launcher example selects must resolve on disk.

    Guards against a recipe rename — e.g. keying ``models/nvidia/<id>`` by the canonical Hub id,
    which carries the ``NVIDIA-`` prefix — drifting from the launcher YAML that loads it. The
    depth/doc tests can't catch a launcher pointing at a recipe path that no longer exists.
    """
    repo_root = Path(__file__).resolve().parents[3]
    launcher_dir = repo_root / "tools" / "launcher" / "examples"
    if not launcher_dir.is_dir():
        pytest.skip("tools/launcher/examples not available in this checkout")

    def _resolves(rel: str) -> bool:
        return any((RECIPES_DIR / f"{rel}{suffix}").exists() for suffix in ("", ".yaml", ".yml"))

    # ``--recipe <p>`` / ``QUANT_CFG: <p>`` are modelopt_recipes-relative — only tier-prefixed
    # values are recipe paths; bare names like ``auto`` or ``FP8_DEFAULT_CFG`` are not. The
    # ``modelopt_recipes/<p>.yaml`` form (e.g. ``--config``) embeds the path directly.
    tier = r"(?:general|huggingface|models|configs)/[A-Za-z0-9._/-]+"
    rel_re = re.compile(rf"(?:--recipe\s+|QUANT_CFG:\s*)({tier})")
    abs_re = re.compile(rf"modelopt_recipes/({tier}\.ya?ml)")

    missing = []
    for yaml_path in sorted(launcher_dir.rglob("*.yaml")):
        text = yaml_path.read_text(encoding="utf-8")
        candidates = set(rel_re.findall(text)) | {
            re.sub(r"\.ya?ml$", "", m) for m in abs_re.findall(text)
        }
        missing.extend(
            f"{yaml_path.relative_to(repo_root)} -> {rel}"
            for rel in sorted(candidates)
            if not _resolves(rel)
        )
    assert not missing, "Launcher YAMLs reference recipe paths that do not resolve:\n" + "\n".join(
        missing
    )
