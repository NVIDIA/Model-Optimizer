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

"""YAML config loading with ``__base__`` inheritance.

This module is intentionally free of ``modelopt.torch`` imports so that
``modelopt.torch.quantization.config`` can import :func:`load_config` without
triggering a circular import through ``modelopt.recipe.loader``.
"""

from importlib.resources import files

try:
    from importlib.resources.abc import Traversable
except ImportError:  # Python < 3.11
    from importlib.abc import Traversable
from pathlib import Path
from typing import Any, cast

import yaml
from omegaconf import OmegaConf

# Root to all built-in recipes. Users can create own recipes.
BUILTIN_RECIPES_LIB = files("modelopt_recipes")


def _merge_base_list(base_list: list, override: dict[str, Any] | None) -> dict[str, Any]:
    """Load and merge a list of base config paths, then overlay ``override``."""
    sub_bases: list[dict[str, Any]] = [load_config(b) for b in base_list]
    if override:
        sub_bases.append(override)
    if not sub_bases:
        return {}
    if len(sub_bases) == 1:
        return sub_bases[0]
    return cast(
        "dict[str, Any]",
        OmegaConf.to_container(OmegaConf.merge(*sub_bases), resolve=True),
    )


def _apply_base_dict(base: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    """Apply dict-style ``__base__`` to ``current`` recursively.

    Each key in ``base`` identifies a section of ``current`` to populate:

    - If the value is a **list**, it is treated as a list of config paths that are loaded
      and merged into ``current[key]``, with any existing ``current[key]`` as the final
      override (highest priority).
    - If the value is a **dict**, the same logic is applied one level deeper, allowing
      arbitrarily nested ``__base__`` path specifications.
    """
    for section, section_base in base.items():
        if isinstance(section_base, list):
            override = current.get(section)
            current[section] = _merge_base_list(
                section_base, override if isinstance(override, dict) else None
            )
        elif isinstance(section_base, dict):
            sub_current = current.get(section)
            current[section] = _apply_base_dict(
                section_base, dict(sub_current) if isinstance(sub_current, dict) else {}
            )
        else:
            raise ValueError(
                f"__base__ dict values must be lists of config paths or nested dicts, "
                f"but '{section}' has type {type(section_base).__name__!r}."
            )
    return current


def _resolve_bases(data: dict[str, Any]) -> dict[str, Any]:
    """Resolve ``__base__`` inheritance at this dict level and recursively in nested dicts."""
    current: dict[str, Any] = {}
    for key, value in data.items():
        current[key] = _resolve_bases(value) if isinstance(value, dict) else value

    base = current.pop("__base__", None)
    if base is None:
        return current

    if isinstance(base, list):
        base_configs = [load_config(b) for b in base]
        base_configs.append(current)
        if len(base_configs) == 1:
            return base_configs[0]
        return cast(
            "dict[str, Any]",
            OmegaConf.to_container(OmegaConf.merge(*base_configs), resolve=True),
        )

    return _apply_base_dict(base, current)


def load_config(config_file: str | Path | Traversable) -> dict[str, Any]:
    """Load a config yaml.

    config_file: Path to a config yaml file. The path suffix can be omitted.

    ``__base__`` inheritance is resolved at every nesting level, not just the top level.  This
    means a sub-section such as ``model_quant`` or ``kv_quant`` can declare its own ``__base__``
    list and the referenced configs will be merged into that sub-section.
    """
    paths_to_check: list[Path | Traversable] = []
    if isinstance(config_file, str):
        if not config_file.endswith(".yml") and not config_file.endswith(".yaml"):
            paths_to_check.append(Path(f"{config_file}.yml"))
            paths_to_check.append(Path(f"{config_file}.yaml"))
            paths_to_check.append(BUILTIN_RECIPES_LIB.joinpath(f"{config_file}.yml"))
            paths_to_check.append(BUILTIN_RECIPES_LIB.joinpath(f"{config_file}.yaml"))
        else:
            paths_to_check.append(Path(config_file))
            paths_to_check.append(BUILTIN_RECIPES_LIB.joinpath(config_file))
    elif isinstance(config_file, Path):
        if config_file.suffix in (".yml", ".yaml"):
            paths_to_check.append(config_file)
            if not config_file.is_absolute():
                paths_to_check.append(BUILTIN_RECIPES_LIB.joinpath(str(config_file)))
        else:
            paths_to_check.append(Path(f"{config_file}.yml"))
            paths_to_check.append(Path(f"{config_file}.yaml"))
            if not config_file.is_absolute():
                paths_to_check.append(BUILTIN_RECIPES_LIB.joinpath(f"{config_file}.yml"))
                paths_to_check.append(BUILTIN_RECIPES_LIB.joinpath(f"{config_file}.yaml"))
    elif isinstance(config_file, Traversable):
        paths_to_check.append(config_file)
    else:
        raise ValueError(f"Invalid config file of {config_file}")

    config_path = None
    for path in paths_to_check:
        if path.is_file():
            config_path = path
            break
    if not config_path:
        raise ValueError(
            f"Cannot find config file of {config_file}, paths checked: {paths_to_check}"
        )

    _raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if _raw is None:
        config_data: dict[str, Any] = {}
    elif not isinstance(_raw, dict):
        raise ValueError(
            f"Config file {config_path} must contain a YAML mapping, got {type(_raw).__name__}"
        )
    else:
        config_data = _raw

    return _resolve_bases(config_data)
