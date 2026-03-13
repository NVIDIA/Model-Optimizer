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
from typing import Any

from ._config_loader import BUILTIN_RECIPES_LIB, load_config
from .config import ModelOptPTQRecipe, ModelOptRecipeBase, RecipeType

__all__ = ["load_config", "load_recipe"]


def load_recipe(recipe_path: str | Path | Traversable) -> ModelOptRecipeBase:
    """Load a recipe from a YAML file or a recipe directory.

    ``recipe_path`` can be:

    * A single YAML recipe file (``.yml`` / ``.yaml``) where ``metadata``,
      ``model_quant``, and ``kv_quant`` live together.  The ``.yml`` / ``.yaml``
      suffix may be omitted and will be probed automatically.
    * A directory containing ``recipe.yml``, ``model_quant.yml``, and
      ``kv_quant.yml`` as separate files.

    In both cases the path may be relative to the built-in recipes library or
    an absolute / relative filesystem path.
    """
    resolved: Path | Traversable

    if isinstance(recipe_path, str):
        # Prefer the built-in library, fall back to the filesystem.
        # If no suffix, probe .yml then .yaml automatically.
        suffixes = [""] if recipe_path.endswith((".yml", ".yaml")) else ["", ".yml", ".yaml"]
        resolved = Path(recipe_path)  # filesystem fallback default
        for suffix in suffixes:
            candidate = BUILTIN_RECIPES_LIB.joinpath(recipe_path + suffix)
            if candidate.is_file() or candidate.is_dir():
                resolved = candidate
                break
        else:
            # Not found in built-in library; probe filesystem with suffixes.
            for suffix in suffixes:
                fs_candidate = Path(recipe_path + suffix)
                if fs_candidate.is_file() or fs_candidate.is_dir():
                    resolved = fs_candidate
                    break
    elif isinstance(recipe_path, Path) and not recipe_path.is_absolute():
        # Relative Path: mirror the same BUILTIN_RECIPES_LIB lookup as for str inputs.
        rp_str = str(recipe_path)
        suffixes = [""] if rp_str.endswith((".yml", ".yaml")) else ["", ".yml", ".yaml"]
        resolved = recipe_path  # filesystem fallback
        for suffix in suffixes:
            candidate = BUILTIN_RECIPES_LIB.joinpath(rp_str + suffix)
            if candidate.is_file() or candidate.is_dir():
                resolved = candidate
                break
        else:
            for suffix in suffixes:
                fs_candidate = Path(rp_str + suffix)
                if fs_candidate.is_file() or fs_candidate.is_dir():
                    resolved = fs_candidate
                    break
    else:
        resolved = recipe_path

    if resolved.is_file():
        return _load_recipe_from_file(resolved)

    if resolved.is_dir():
        return _load_recipe_from_dir(resolved)

    raise ValueError(f"Recipe path {recipe_path!r} is neither a valid YAML file nor a directory.")


def _load_recipe_from_file(recipe_file: Path | Traversable) -> ModelOptRecipeBase:
    """Load a recipe from a single merged YAML file.

    The file must contain a ``metadata`` section with at least ``recipe_type``,
    plus the type-specific sections (``model_quant`` and ``kv_quant`` for PTQ).
    ``__base__`` inheritance inside any section is resolved by :func:`load_config`.
    """
    data = load_config(recipe_file)

    metadata = data.get("metadata", {})
    recipe_type = metadata.get("recipe_type")
    if recipe_type is None:
        raise ValueError(f"Recipe file {recipe_file} must contain a 'metadata.recipe_type' field.")

    if recipe_type == RecipeType.PTQ:
        from modelopt.torch.quantization.config import QuantizeConfig, QuantizeQuantCfgType

        if "model_quant" not in data:
            raise ValueError(f"PTQ recipe file {recipe_file} must contain 'model_quant'.")
        if "kv_quant" not in data:
            raise ValueError(f"PTQ recipe file {recipe_file} must contain 'kv_quant'.")
        return ModelOptPTQRecipe(
            recipe_type=RecipeType.PTQ,
            description=metadata.get("description", "PTQ recipe."),
            model_quant=QuantizeConfig(**data["model_quant"]),
            kv_quant=QuantizeQuantCfgType(**data["kv_quant"]),
        )
    raise ValueError(f"Unsupported recipe type: {recipe_type!r}")


def _load_recipe_from_dir(recipe_dir: Path | Traversable) -> ModelOptRecipeBase:
    """Load a recipe from a directory containing separate YAML files."""
    descriptor_path = None
    for descriptor_file in ["recipe.yml", "recipe.yaml"]:
        descriptor_path = recipe_dir.joinpath(descriptor_file)
        if descriptor_path.is_file():
            break
    if descriptor_path is None or not descriptor_path.is_file():
        raise ValueError(
            f"Cannot find a valid recipe descriptor file in {recipe_dir}. "
            "Looked for: recipe.yml, recipe.yaml"
        )

    recipe_descriptor = load_config(descriptor_path)
    if not isinstance(recipe_descriptor, dict):
        raise ValueError("Recipe descriptor should be a dictionary.")
    if "recipe_type" not in recipe_descriptor:
        raise ValueError("Recipe descriptor should contain 'recipe_type' field.")

    if recipe_descriptor["recipe_type"] == RecipeType.PTQ:
        return _load_ptq_recipe_from_dir(recipe_dir, recipe_descriptor)
    raise ValueError(f"Unsupported recipe type: {recipe_descriptor['recipe_type']!r}")


def _load_ptq_recipe_from_dir(
    recipe_path: Path | Traversable, recipe_descriptor: dict[str, Any]
) -> ModelOptPTQRecipe:
    """Load a PTQ recipe from a directory with separate model_quant / kv_quant files."""
    model_quant_path = None
    for name in ["model_quant.yml", "model_quant.yaml"]:
        model_quant_path = recipe_path.joinpath(name)
        if model_quant_path.is_file():
            break

    kv_quant_path = None
    for name in ["kv_quant.yml", "kv_quant.yaml"]:
        kv_quant_path = recipe_path.joinpath(name)
        if kv_quant_path.is_file():
            break

    if model_quant_path is None or not model_quant_path.is_file():
        raise ValueError(
            f"Cannot find model_quant config in {recipe_path}. "
            "Looked for: model_quant.yml, model_quant.yaml"
        )
    if kv_quant_path is None or not kv_quant_path.is_file():
        raise ValueError(
            f"Cannot find kv_quant config in {recipe_path}. Looked for: kv_quant.yml, kv_quant.yaml"
        )

    from modelopt.torch.quantization.config import QuantizeConfig, QuantizeQuantCfgType

    return ModelOptPTQRecipe(
        recipe_type=RecipeType.PTQ,
        description=recipe_descriptor.get("description", "PTQ recipe."),
        model_quant=QuantizeConfig(**load_config(model_quant_path)),
        kv_quant=QuantizeQuantCfgType(**load_config(kv_quant_path)),
    )
