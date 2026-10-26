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

"""Pre-commit hook: validate ModelOpt recipes.

Pre-commit passes changed file paths as arguments. This script resolves each
file to its parent recipe (single-file or directory format), deduplicates, and
validates each recipe exactly once.

Checks performed:

1. ``quant_cfg`` must use the list-of-dicts format with explicit
   ``quantizer_name`` keys (legacy dict format is rejected). PTQ recipes only.
2. PTQ recipes must use ``quantize`` as the top-level key
   (not ``ptq_cfg`` or other variants).
3. Each recipe (PTQ, EAGLE, DFlash, Medusa) is loaded via ``load_recipe()``
   to catch structural and Pydantic-validation errors (skipped if modelopt is
   not installed).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

_YAML_PARSE_ERROR = object()

# Recipe types reached through the LEGACY metadata.recipe_type path only. A recipe that
# declares a ``# modelopt-schema:`` comment, or that delegates with ``$import``, is
# validated whether or not its kind appears here -- _is_recipe_file returns True on those
# branches before this set is consulted. So a new kind that declares a schema (the
# recommended form) needs no change here; only a new kind still using the deprecated
# metadata.recipe_type would. Mirrors RecipeType in modelopt.recipe.config; kept as a
# literal set so the hook can run without importing modelopt (which is also why
# _try_load_recipe gates on ImportError).
_SUPPORTED_RECIPE_TYPES = frozenset(
    {"ptq", "speculative_eagle", "speculative_dflash", "speculative_medusa"}
)

# A recipe usually declares its kind with a ``# modelopt-schema:`` comment naming its
# schema class rather than with ``metadata.recipe_type`` (see modelopt/recipe/loader.py).
# Matched here by name so the hook keeps working without importing modelopt.
_SCHEMA_COMMENT_RE = re.compile(
    r"^\s*#\s*modelopt-schema:\s*modelopt\.recipe\.config\.ModelOpt\w+Recipe\s*$",
    re.MULTILINE,
)

# Any ``# modelopt-schema:`` declaration, recipe or not. The reusable snippets under
# ``modelopt_recipes/configs/`` declare non-recipe schemas -- QuantizerAttributeConfig,
# LayerPatternList and friends -- and a snippet is allowed a top-level ``$import`` of its
# own, which would otherwise make it indistinguishable here from a delegating alias.
_ANY_SCHEMA_COMMENT_RE = re.compile(
    r"^\s*#\s*modelopt-schema:\s*\S+\s*$",
    re.MULTILINE,
)


def _declares_recipe_schema(path: Path) -> bool:
    """Whether *path* names one of the recipe schema classes in a ``# modelopt-schema:`` comment.

    Searched over the whole file, deliberately laxer than the loader's
    ``_parse_modelopt_schema``, which stops at the first non-comment line. A file carrying
    the comment *below* its YAML body is therefore a recipe to this hook and not to the
    loader -- which is the outcome we want: the hook hands it to ``load_recipe``, which
    rejects it with "does not say what kind of recipe it is" rather than the file being
    skipped silently. ``test_shipped_modelopt_schema_comments_are_in_the_preamble`` keeps
    the shipped tree free of that shape.
    """
    try:
        return bool(_SCHEMA_COMMENT_RE.search(path.read_text(encoding="utf-8")))
    except OSError:
        return False


def _declares_non_recipe_schema(path: Path) -> bool:
    """Whether *path* declares a ``# modelopt-schema:`` that is not a recipe schema.

    That is the signature of a reusable snippet (a quantizer attribute, a layer-pattern
    list), which ``load_recipe`` cannot load and should never be handed.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False
    return bool(_ANY_SCHEMA_COMMENT_RE.search(text)) and not bool(_SCHEMA_COMMENT_RE.search(text))


def _check_quant_cfg(quant_cfg, label: str) -> list[str]:
    """Validate quant_cfg format. *label* is used in error messages."""
    errors: list[str] = []
    if isinstance(quant_cfg, dict):
        errors.append(
            f"{label}: quant_cfg uses the legacy dict format. "
            "Use the list-of-dicts format with explicit 'quantizer_name' keys instead. "
            "See https://nvidia.github.io/Model-Optimizer/guides/_quant_cfg.html for the format specification."
        )
    elif isinstance(quant_cfg, list):
        for i, entry in enumerate(quant_cfg):
            if not isinstance(entry, dict):
                errors.append(
                    f"{label}: quant_cfg[{i}] must be a dict with "
                    f"'quantizer_name' or '$import', got {type(entry).__name__}. "
                    "See https://nvidia.github.io/Model-Optimizer/guides/_quant_cfg.html"
                )
                continue
            # {$import: name} entries are resolved at load time
            if "$import" in entry:
                ref = entry["$import"]
                if not isinstance(ref, (str, list)) or (
                    isinstance(ref, list) and not all(isinstance(r, str) for r in ref)
                ):
                    errors.append(
                        f"{label}: quant_cfg[{i}] '$import' must be a string or list of strings, "
                        f"got {type(ref).__name__}: {ref!r}"
                    )
                continue
            if "quantizer_name" not in entry:
                errors.append(
                    f"{label}: quant_cfg[{i}] is missing 'quantizer_name'. "
                    "Each entry must have an explicit 'quantizer_name' or '$import' key. "
                    "See https://nvidia.github.io/Model-Optimizer/guides/_quant_cfg.html"
                )
    return errors


def _load_yaml(path: Path):
    """Load the first YAML document, returning _YAML_PARSE_ERROR on parse failure."""
    try:
        docs = list(yaml.safe_load_all(path.read_text(encoding="utf-8")))
    except Exception:
        return _YAML_PARSE_ERROR
    return docs[0] if docs else None


def _check_single_file_recipe(path: Path) -> list[str]:
    """Check a single-file recipe (metadata + quantize in one file)."""
    errors: list[str] = []
    label = str(path)
    data = _load_yaml(path)
    if data is _YAML_PARSE_ERROR:
        return [f"{label}: failed to parse YAML"]
    if not isinstance(data, dict):
        return []  # not a recipe file

    metadata = data.get("metadata")
    if not isinstance(metadata, dict) and not _declares_recipe_schema(path):
        return []  # not a recipe file

    if "ptq_cfg" in data:
        errors.append(
            f"{label}: uses 'ptq_cfg' as the top-level key. "
            "PTQ recipes must use 'quantize' instead."
        )
    if "quantize" in data:
        quant_section = data["quantize"]
    elif "ptq_cfg" in data:
        quant_section = data["ptq_cfg"]
    else:
        return errors

    if isinstance(quant_section, dict):
        quant_cfg = quant_section.get("quant_cfg")
        if quant_cfg is not None:
            errors.extend(_check_quant_cfg(quant_cfg, label))

    return errors


def _check_dir_recipe(dir_path: Path) -> list[str]:
    """Check a directory-format recipe (metadata.yml + quantize.yml)."""
    errors: list[str] = []

    for name in ("quantize.yml", "quantize.yaml"):
        quantize_file = dir_path / name
        if quantize_file.is_file():
            data = _load_yaml(quantize_file)
            if data is _YAML_PARSE_ERROR:
                errors.append(f"{quantize_file}: failed to parse YAML")
            elif isinstance(data, dict):
                quant_cfg = data.get("quant_cfg")
                if quant_cfg is not None:
                    errors.extend(_check_quant_cfg(quant_cfg, str(quantize_file)))
            break

    return errors


def _try_load_recipe(path: str) -> list[str]:
    """Try loading a recipe via modelopt; return errors or []."""
    try:
        from modelopt.recipe.loader import load_recipe
    except ImportError:
        return []  # modelopt not installed, skip

    try:
        load_recipe(path)
    except Exception as exc:
        return [f"{path}: recipe failed to load: {exc}"]
    return []


def _is_dir_recipe(dir_path: Path) -> bool:
    """Return True if *dir_path* is a directory-format recipe."""
    return any((dir_path / n).is_file() for n in ("metadata.yml", "metadata.yaml"))


def _is_recipe_file(path: Path) -> bool:
    """Return True if *path* looks like a recipe file that should be validated.

    Three ways in, checked in this order: a ``# modelopt-schema:`` comment, a top-level
    ``$import`` (a delegating alias, whose kind comes from what it imports), and finally
    the deprecated ``metadata.recipe_type`` gated on ``_SUPPORTED_RECIPE_TYPES``. Only
    that last branch consults the set, so a recipe of any kind that declares a schema is
    validated here -- including kinds deliberately absent from the set, such as
    ``auto_quantize``. ``load_recipe`` handles those, so this is intended.

    Malformed or unparseable files return True so that ``load_recipe()`` can
    report the actual error.
    """
    data = _load_yaml(path)
    if data is _YAML_PARSE_ERROR:
        return True  # let load_recipe report the parse error
    if not isinstance(data, dict):
        return False  # not a recipe file at all
    if _declares_recipe_schema(path):
        return True
    if _declares_non_recipe_schema(path):
        # A snippet, not a recipe -- and snippets may carry a top-level ``$import`` of
        # their own (see ``test_import_cross_file_same_name_no_conflict``). Without this
        # the next branch would claim it and ``load_recipe`` would reject it with "does
        # not say what kind of recipe it is", which is a confusing way to learn that a
        # fragment was never meant to be loaded as a recipe.
        return False
    if "$import" in data:
        # A delegating alias declares neither a schema comment nor a recipe_type: its
        # kind comes from the recipe it imports. Validate it so a typo in ``imports:``
        # or a ``$import`` naming an undeclared import fails here rather than at use.
        return True
    metadata = data.get("metadata")
    if not isinstance(metadata, dict) or "recipe_type" not in metadata:
        return False  # not a recipe file at all
    return metadata["recipe_type"] in _SUPPORTED_RECIPE_TYPES


def _is_metadata_file(path: Path) -> bool:
    """Return True if *path* looks like a directory recipe metadata file.

    Directory-format recipes are PTQ-only (speculative-decoding recipes are
    always single YAML files), so the check is limited to ``recipe_type: ptq``.
    """
    data = _load_yaml(path)
    if data is _YAML_PARSE_ERROR:
        return True  # let load_recipe report the parse error
    if not isinstance(data, dict):
        return False
    return data.get("recipe_type") == "ptq"


def _resolve_recipes(changed_files: list[str]) -> dict[Path, str]:
    """Resolve changed files to recipes. Returns {recipe_path: kind} mapping.

    Non-recipe YAML files are silently skipped.
    kind is "file" for single-file recipes or "dir" for directory-format recipes.
    """
    recipes: dict[Path, str] = {}
    for f in changed_files:
        path = Path(f)

        # Check if this file is inside a directory-format recipe.
        if _is_dir_recipe(path.parent):
            # Directory recipes have a metadata.yml with top-level metadata fields.
            for name in ("metadata.yml", "metadata.yaml"):
                candidate = path.parent / name
                if candidate.is_file() and _is_metadata_file(candidate):
                    recipes.setdefault(path.parent, "dir")
                    break
        elif path.is_file() and path.suffix in (".yml", ".yaml"):
            if _is_recipe_file(path):
                recipes.setdefault(path, "file")

    return recipes


def main() -> int:
    """Validate changed recipes passed as CLI args, exit 1 on errors."""
    recipes = _resolve_recipes(sys.argv[1:])
    errors: list[str] = []

    for recipe_path, kind in recipes.items():
        if kind == "dir":
            recipe_errors = _check_dir_recipe(recipe_path)
        else:
            recipe_errors = _check_single_file_recipe(recipe_path)

        errors.extend(recipe_errors)
        if not recipe_errors:
            errors.extend(_try_load_recipe(str(recipe_path)))

    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
