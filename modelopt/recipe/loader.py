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

"""Recipe loading utilities."""

try:
    from importlib.resources.abc import Traversable
except ImportError:  # Python < 3.11
    from importlib.abc import Traversable
from pathlib import Path

from ._config_loader import BUILTIN_RECIPES_LIB, load_config
from .config import ModelOptPTQRecipe, ModelOptRecipeBase, RecipeType

__all__ = ["load_config", "load_recipe"]


def _resolve_recipe_path(recipe_path: str | Path | Traversable) -> Path | Traversable:
    """Resolve a recipe path, checking the built-in library first then the filesystem.

    Returns the resolved path (file or directory).
    """
    if isinstance(recipe_path, (str, Path)) and not (
        isinstance(recipe_path, Path) and recipe_path.is_absolute()
    ):
        rp_str = str(recipe_path)
        suffixes = [""] if rp_str.endswith((".yml", ".yaml")) else ["", ".yml", ".yaml"]
        for suffix in suffixes:
            candidate = BUILTIN_RECIPES_LIB.joinpath(rp_str + suffix)
            if candidate.is_file() or candidate.is_dir():
                return candidate
        for suffix in suffixes:
            fs_candidate = Path(rp_str + suffix)
            if fs_candidate.is_file() or fs_candidate.is_dir():
                return fs_candidate
        return Path(rp_str)
    return recipe_path


def load_recipe(recipe_path: str | Path | Traversable) -> ModelOptRecipeBase:
    """Load a recipe from a YAML file or directory.

    ``recipe_path`` can be:

    * A ``.yml`` / ``.yaml`` file with ``metadata`` and ``quantize`` sections.
      The suffix may be omitted and will be probed automatically.
    * A directory containing ``recipe.yml`` (metadata) and ``quantize.yml``.

    The path may be relative to the built-in recipes library or an absolute /
    relative filesystem path.
    """
    resolved = _resolve_recipe_path(recipe_path)

    _builtin_prefix = str(BUILTIN_RECIPES_LIB)
    _resolved_str = str(resolved)
    if _resolved_str.startswith(_builtin_prefix):
        _display = "<builtin>/" + _resolved_str[len(_builtin_prefix) :].lstrip("/\\")
    else:
        _display = _resolved_str
    print(f"[load_recipe] loading: {_display}")

    if resolved.is_file():
        return _load_recipe_from_file(resolved)

    if resolved.is_dir():
        return _load_recipe_from_dir(resolved)

    raise ValueError(f"Recipe path {recipe_path!r} is not a valid YAML file or directory.")


def _load_recipe_from_file(recipe_file: Path | Traversable) -> ModelOptRecipeBase:
    """Load a recipe from a YAML file.

    The file must contain a ``metadata`` section with at least ``recipe_type``,
    plus a ``quant_cfg`` mapping and an optional ``algorithm`` for PTQ recipes.
    """
    data = load_config(recipe_file)

    metadata = data.get("metadata", {})
    recipe_type = metadata.get("recipe_type")
    if recipe_type is None:
        raise ValueError(f"Recipe file {recipe_file} must contain a 'metadata.recipe_type' field.")

    if recipe_type == RecipeType.PTQ:
        if "quantize" not in data:
            raise ValueError(f"PTQ recipe file {recipe_file} must contain 'quantize'.")
        return ModelOptPTQRecipe(
            recipe_type=RecipeType.PTQ,
            description=metadata.get("description", "PTQ recipe."),
            quantize=data["quantize"],
        )
    raise ValueError(f"Unsupported recipe type: {recipe_type!r}")


def _load_recipe_from_dir(recipe_dir: Path | Traversable) -> ModelOptRecipeBase:
    """Load a recipe from a directory containing ``recipe.yml`` and ``quantize.yml``."""
    recipe_file = None
    for name in ("recipe.yml", "recipe.yaml"):
        candidate = recipe_dir.joinpath(name)
        if candidate.is_file():
            recipe_file = candidate
            break
    if recipe_file is None:
        raise ValueError(
            f"Cannot find a recipe descriptor in {recipe_dir}. Looked for: recipe.yml, recipe.yaml"
        )

    metadata = load_config(recipe_file).get("metadata", {})
    recipe_type = metadata.get("recipe_type")
    if recipe_type is None:
        raise ValueError(f"Recipe file {recipe_file} must contain a 'metadata.recipe_type' field.")

    if recipe_type == RecipeType.PTQ:
        quantize_file = None
        for name in ("quantize.yml", "quantize.yaml"):
            candidate = recipe_dir.joinpath(name)
            if candidate.is_file():
                quantize_file = candidate
                break
        if quantize_file is None:
            raise ValueError(
                f"Cannot find quantize in {recipe_dir}. Looked for: quantize.yml, quantize.yaml"
            )
        return ModelOptPTQRecipe(
            recipe_type=RecipeType.PTQ,
            description=metadata.get("description", "PTQ recipe."),
            quantize=load_config(quantize_file),
        )
    raise ValueError(f"Unsupported recipe type: {recipe_type!r}")
