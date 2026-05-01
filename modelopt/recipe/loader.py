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

from omegaconf import OmegaConf

from ._config_loader import BUILTIN_RECIPES_LIB, load_config
from .config import (
    ModelOptDFlashRecipe,
    ModelOptEagleRecipe,
    ModelOptMedusaRecipe,
    ModelOptPTQRecipe,
    ModelOptRecipeBase,
    RecipeType,
)

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


def load_recipe(
    recipe_path: str | Path | Traversable,
    overrides: list[str] | None = None,
) -> ModelOptRecipeBase:
    """Load a recipe from a YAML file or directory, with optional CLI-style overrides.

    ``recipe_path`` can be:

    * A ``.yml`` / ``.yaml`` file with ``metadata`` and one of ``quantize`` (PTQ),
      ``eagle`` (EAGLE speculative decoding), ``dflash`` (DFlash speculative
      decoding) or ``medusa`` (Medusa speculative decoding) sections. The suffix
      may be omitted and will be probed automatically.
    * A directory containing ``recipe.yml`` (metadata) plus ``quantize.yml`` —
      **PTQ recipes only**. Speculative-decoding recipes are always single YAML files.

    The path may be relative to the built-in recipes library or an absolute /
    relative filesystem path.

    ``overrides`` is an optional list of ``key.path=value`` dotlist entries applied
    on top of the YAML before Pydantic validation. Values are parsed with
    ``yaml.safe_load`` so they get proper types (``foo.bar=true`` → bool, ``foo=1``
    → int, ``foo=[1,2]`` → list, etc.). Only supported when *recipe_path* is a
    single YAML file.
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
        return _load_recipe_from_file(resolved, overrides=overrides)

    if resolved.is_dir():
        if overrides:
            raise ValueError(
                "overrides are not supported for directory-format recipes; "
                "use the single-YAML-file form instead."
            )
        return _load_recipe_from_dir(resolved)

    raise ValueError(f"Recipe path {recipe_path!r} is not a valid YAML file or directory.")


def _apply_dotlist(data: dict, overrides: list[str]) -> dict:
    """Merge ``a.b.c=value`` command line overrides on top of ``data`` via OmegaConf."""
    for entry in overrides:
        if "=" not in entry:
            raise ValueError(f"Invalid override (missing '='): {entry!r}")
    merged = OmegaConf.merge(
        OmegaConf.create(data),
        OmegaConf.from_dotlist(list(overrides)),
    )
    return OmegaConf.to_container(merged, resolve=True)


def _load_recipe_from_file(
    recipe_file: Path | Traversable,
    overrides: list[str] | None = None,
) -> ModelOptRecipeBase:
    """Load a recipe from a YAML file, optionally applying dotlist overrides.

    The file must contain a ``metadata`` section with at least ``recipe_type``,
    plus the algorithm-specific section (``quantize`` / ``eagle`` / ``dflash`` / ``medusa``).
    """
    data = load_config(recipe_file)
    if overrides:
        data = _apply_dotlist(data, overrides)

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
    if recipe_type == RecipeType.SPECULATIVE_EAGLE:
        if "eagle" not in data:
            raise ValueError(f"EAGLE recipe file {recipe_file} must contain 'eagle'.")
        return ModelOptEagleRecipe(
            description=metadata.get("description", "EAGLE speculative decoding recipe."),
            model=data.get("model") or {},
            data=data.get("data") or {},
            training=data.get("training") or {},
            eagle=data["eagle"],
        )
    if recipe_type == RecipeType.SPECULATIVE_DFLASH:
        if "dflash" not in data:
            raise ValueError(f"DFlash recipe file {recipe_file} must contain 'dflash'.")
        return ModelOptDFlashRecipe(
            description=metadata.get("description", "DFlash speculative decoding recipe."),
            model=data.get("model") or {},
            data=data.get("data") or {},
            training=data.get("training") or {},
            dflash=data["dflash"],
        )
    if recipe_type == RecipeType.SPECULATIVE_MEDUSA:
        if "medusa" not in data:
            raise ValueError(f"Medusa recipe file {recipe_file} must contain 'medusa'.")
        return ModelOptMedusaRecipe(
            description=metadata.get("description", "Medusa speculative decoding recipe."),
            model=data.get("model") or {},
            data=data.get("data") or {},
            training=data.get("training") or {},
            medusa=data["medusa"],
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
