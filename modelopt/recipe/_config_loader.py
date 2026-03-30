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

"""YAML config loading utilities.

This module is intentionally free of ``modelopt.torch`` imports so that
``modelopt.torch.quantization.config`` can import :func:`load_config` without
triggering a circular import through ``modelopt.recipe.loader``.
"""

from importlib.resources import files

try:
    from importlib.resources.abc import Traversable
except ImportError:  # Python < 3.11
    from importlib.abc import Traversable
import re
from pathlib import Path
from typing import Any

import yaml

# Root to all built-in recipes. Users can create own recipes.
BUILTIN_RECIPES_LIB = files("modelopt_recipes")

_EXMY_RE = re.compile(r"^[Ee](\d+)[Mm](\d+)$")
_EXMY_KEYS = frozenset({"num_bits", "scale_bits"})


def _parse_exmy_num_bits(obj: Any) -> Any:
    """Recursively convert ``ExMy`` strings in ``num_bits`` / ``scale_bits`` to ``(x, y)`` tuples."""
    if isinstance(obj, dict):
        return {
            k: (
                _parse_exmy(v)
                if k in _EXMY_KEYS and isinstance(v, str)
                else _parse_exmy_num_bits(v)
            )
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_parse_exmy_num_bits(item) for item in obj]
    return obj


def _parse_exmy(s: str) -> tuple[int, int] | str:
    m = _EXMY_RE.match(s)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    return s


def load_config(config_file: str | Path | Traversable) -> dict[str, Any]:
    """Load a config yaml.

    config_file: Path to a config yaml file. The path suffix can be omitted.
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
        return {}
    if not isinstance(_raw, dict):
        raise ValueError(
            f"Config file {config_path} must contain a YAML mapping, got {type(_raw).__name__}"
        )
    return _parse_exmy_num_bits(_raw)
