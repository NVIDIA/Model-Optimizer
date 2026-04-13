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


def _resolve_imports(
    data: dict[str, Any], _loading: frozenset[str] | None = None
) -> dict[str, Any]:
    """Resolve the ``imports`` section in a recipe and substitute named references.

    An ``imports`` block is a dict mapping short names to config file paths::

        imports:
          fp8: configs/numerics/fp8
          nvfp4: configs/numerics/nvfp4_dynamic

    ``cfg`` values in ``quant_cfg`` entries that are plain strings are looked up
    against the imported names and replaced with the loaded config dict.

    Resolution is **recursive**: an imported snippet may itself contain an
    ``imports`` section.  Circular imports are detected and raise ``ValueError``.
    """
    imports_dict = data.pop("imports", None)
    if not imports_dict:
        return data

    if not isinstance(imports_dict, dict):
        raise ValueError(
            f"'imports' must be a dict mapping names to config paths, got: {type(imports_dict).__name__}"
        )

    if _loading is None:
        _loading = frozenset()

    # Build name → config mapping (recursively resolve nested imports)
    import_map: dict[str, Any] = {}
    for name, config_path in imports_dict.items():
        if not config_path:
            raise ValueError(f"Import {name!r} has an empty config path.")
        if config_path in _loading:
            raise ValueError(
                f"Circular import detected: {config_path!r} is already being loaded. "
                f"Import chain: {sorted(_loading)}"
            )
        snippet = load_config(config_path)
        if isinstance(snippet, dict) and "imports" in snippet:
            snippet = _resolve_imports(snippet, _loading | {config_path})
        import_map[name] = snippet

    # Resolve string references in quant_cfg entries
    quantize = data.get("quantize")
    if isinstance(quantize, dict):
        quant_cfg = quantize.get("quant_cfg")
        if isinstance(quant_cfg, list):
            resolved_cfg: list[Any] = []
            for entry in quant_cfg:
                if isinstance(entry, str):
                    # Entire entry is a string → replace with the imported value
                    if entry not in import_map:
                        raise ValueError(
                            f"Unknown import reference {entry!r} in quant_cfg list. "
                            f"Available imports: {list(import_map.keys())}"
                        )
                    imported = import_map[entry]
                    if isinstance(imported, list):
                        # List import → splice all entries in place
                        resolved_cfg.extend(imported)
                    else:
                        resolved_cfg.append(imported)
                elif isinstance(entry, dict) and isinstance(entry.get("cfg"), str):
                    # cfg field is a string → replace cfg value
                    ref_name = entry["cfg"]
                    if ref_name not in import_map:
                        raise ValueError(
                            f"Unknown import reference {ref_name!r} in quant_cfg entry "
                            f"{entry!r}. Available imports: {list(import_map.keys())}"
                        )
                    entry["cfg"] = import_map[ref_name]
                    resolved_cfg.append(entry)
                else:
                    resolved_cfg.append(entry)
            quantize["quant_cfg"] = resolved_cfg

    return data


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
    raw = load_config(recipe_file)
    assert isinstance(raw, dict), f"Recipe file {recipe_file} must be a YAML mapping."
    data = _resolve_imports(raw)

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

    recipe_data = load_config(recipe_file)
    assert isinstance(recipe_data, dict), f"Recipe file {recipe_file} must be a YAML mapping."
    metadata = recipe_data.get("metadata", {})
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
        # Resolve imports: imports are in recipe.yml, quantize data is separate
        quantize_data = load_config(quantize_file)
        assert isinstance(quantize_data, dict), f"{quantize_file} must be a YAML mapping."
        combined: dict[str, Any] = {"quantize": quantize_data}
        imports = recipe_data.get("imports")
        if imports:
            combined["imports"] = imports
        combined = _resolve_imports(combined)
        return ModelOptPTQRecipe(
            recipe_type=RecipeType.PTQ,
            description=metadata.get("description", "PTQ recipe."),
            quantize=combined["quantize"],
        )
    raise ValueError(f"Unsupported recipe type: {recipe_type!r}")
