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
   ``quantizer_name`` keys (legacy dict format is rejected).
2. PTQ recipes must use ``quantize`` as the top-level key
   (not ``ptq_cfg`` or other variants).
3. Each recipe is loaded via ``load_recipe()`` to catch structural and
   validation errors (skipped if modelopt is not installed).
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml


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
                    f"'quantizer_name', got {type(entry).__name__}. "
                    "See https://nvidia.github.io/Model-Optimizer/guides/_quant_cfg.html"
                )
                continue
            if "quantizer_name" not in entry:
                errors.append(
                    f"{label}: quant_cfg[{i}] is missing 'quantizer_name'. "
                    "Each entry must have an explicit 'quantizer_name' key. "
                    "See https://nvidia.github.io/Model-Optimizer/guides/_quant_cfg.html"
                )
    return errors


def _load_yaml(path: Path) -> dict | None:
    """Load a YAML file, returning None on parse failure."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _check_single_file_recipe(path: Path) -> list[str]:
    """Check a single-file recipe (metadata + quantize in one file)."""
    errors: list[str] = []
    label = str(path)
    data = _load_yaml(path)
    if data is None:
        return [f"{label}: failed to parse YAML"]

    metadata = data.get("metadata")
    if not isinstance(metadata, dict) or "recipe_type" not in metadata:
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
    """Check a directory-format recipe (recipe.yml + quantize.yml)."""
    errors: list[str] = []

    for name in ("quantize.yml", "quantize.yaml"):
        quantize_file = dir_path / name
        if quantize_file.is_file():
            data = _load_yaml(quantize_file)
            if data is not None:
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
    return any((dir_path / n).is_file() for n in ("recipe.yml", "recipe.yaml"))


def _is_recipe_file(path: Path) -> bool:
    """Return True if *path* looks like a recipe file that should be validated.

    Currently only PTQ recipes are checked; other recipe types (e.g. QAT) can
    be added here in the future.

    Malformed or unparseable files return True so that ``load_recipe()`` can
    report the actual error.
    """
    data = _load_yaml(path)
    if data is None:
        return True  # let load_recipe report the parse error
    metadata = data.get("metadata")
    if not isinstance(metadata, dict) or "recipe_type" not in metadata:
        return False  # not a recipe file at all
    return metadata["recipe_type"] == "ptq"


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
            # Directory recipes have a recipe.yml with metadata; check it.
            for name in ("recipe.yml", "recipe.yaml"):
                candidate = path.parent / name
                if candidate.is_file() and _is_recipe_file(candidate):
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
