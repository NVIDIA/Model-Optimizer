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

"""Lightweight experiment-config composition for the Puzzletron controller."""

from __future__ import annotations

import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml

__all__ = ["load_experiment_config"]

_INTERPOLATION = re.compile(r"\$\{([^${}]*)\}")
_SCIENTIFIC_FLOAT = re.compile(r"^[+-]?[0-9][0-9_]*[eE][+-]?[0-9]+$")


class _HydraSafeLoader(yaml.SafeLoader):
    """Parse plain scientific notation with Hydra-compatible numeric semantics."""


_HydraSafeLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    _SCIENTIFIC_FLOAT,
    list("-+0123456789"),
)


def _load_yaml(value: str) -> Any:
    loader = _HydraSafeLoader(value)
    try:
        return loader.get_single_data()
    finally:
        loader.dispose()


def _mapping(value: Any, *, source: Path) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"YAML root must be a mapping: {source}")
    return dict(value)


def _merge(base: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge(dict(merged[key]), value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _config_root(path: Path) -> Path:
    for parent in (path.parent, *path.parents):
        if (parent / "base.yaml").is_file():
            return parent
    return path.parent


def _default_path(item: str, *, current: Path, root: Path) -> Path:
    reference, _, package = item.partition("@")
    if package and package != "_global_":
        raise ValueError(f"Unsupported Hydra package {package!r} in {current}")
    relative = Path(reference.lstrip("/"))
    base = root if reference.startswith("/") else current.parent
    candidate = base / relative
    if candidate.suffix not in {".yaml", ".yml"}:
        candidate = candidate.with_suffix(".yaml")
    return candidate


def _compose(path: Path, *, root: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
    path = path.resolve()
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise ValueError(f"Config defaults cycle: {chain}")
    payload = _mapping(_load_yaml(path.read_text()), source=path)
    defaults = payload.pop("defaults", [])
    if not isinstance(defaults, list):
        raise ValueError(f"defaults must be a list: {path}")

    result: dict[str, Any] = {}
    merged_self = False
    for item in defaults:
        if item == "_self_":
            result = _merge(result, payload)
            merged_self = True
            continue
        if not isinstance(item, str):
            raise ValueError(f"Unsupported defaults entry {item!r} in {path}")
        dependency = _default_path(item, current=path, root=root)
        result = _merge(
            result,
            _compose(dependency, root=root, stack=(*stack, path)),
        )
    if not merged_self:
        result = _merge(result, payload)
    return result


def _lookup(config: Mapping[str, Any], dotted_path: str) -> Any:
    value: Any = config
    for key in dotted_path.split("."):
        if not isinstance(value, Mapping) or key not in value:
            raise KeyError(dotted_path)
        value = value[key]
    return value


def _resolve_expression(expression: str, config: Mapping[str, Any]) -> Any:
    if expression.startswith("oc.env:"):
        name, separator, default = expression.removeprefix("oc.env:").partition(",")
        if name in os.environ:
            return os.environ[name]
        if separator:
            return default
        raise ValueError(f"Required environment variable {name!r} is not set")
    if expression.startswith("to_path:"):
        return expression.removeprefix("to_path:")
    if expression.startswith("get_object:"):
        return {"__type__": expression.removeprefix("get_object:")}
    try:
        return deepcopy(_lookup(config, expression))
    except KeyError:
        return "${" + expression + "}"


def _resolve_string(value: str, config: Mapping[str, Any]) -> Any:
    previous = None
    resolved: Any = value
    for _ in range(50):
        if not isinstance(resolved, str) or resolved == previous:
            return resolved
        previous = resolved
        matches = list(_INTERPOLATION.finditer(resolved))
        if not matches:
            return resolved
        if len(matches) == 1 and matches[0].span() == (0, len(resolved)):
            replacement = _resolve_expression(matches[0].group(1), config)
            if replacement == resolved:
                return resolved
            resolved = replacement
            continue
        for match in reversed(matches):
            replacement = _resolve_expression(match.group(1), config)
            resolved = resolved[: match.start()] + str(replacement) + resolved[match.end() :]
    raise ValueError(f"Interpolation did not converge: {value!r}")


def _resolve(value: Any, config: Mapping[str, Any]) -> Any:
    if isinstance(value, str):
        return _resolve_string(value, config)
    if isinstance(value, list):
        return [_resolve(item, config) for item in value]
    if isinstance(value, Mapping):
        return {key: _resolve(item, config) for key, item in value.items()}
    return value


def _apply_override(config: dict[str, Any], override: str) -> None:
    key, separator, raw_value = override.partition("=")
    if not separator:
        raise ValueError(f"Override must have KEY=VALUE form: {override!r}")
    keys = key.lstrip("+").split(".")
    target = config
    for part in keys[:-1]:
        child = target.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(f"Override path crosses a scalar: {override!r}")
        target = child
    target[keys[-1]] = _load_yaml(raw_value)


def load_experiment_config(
    path: str | Path,
    *,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    """Compose a Puzzletron experiment without importing Hydra or PyTorch.

    The controller needs only global defaults, mapping merges, dotted overrides,
    environment references, and ordinary config interpolation. GPU jobs still
    load the original config through Puzzletron's full Hydra runtime.
    """

    config_path = Path(path).resolve()
    config = _compose(config_path, root=_config_root(config_path), stack=())
    for override in overrides or ():
        _apply_override(config, override)

    for _ in range(50):
        resolved = _resolve(config, config)
        if resolved == config:
            break
        config = resolved
    else:
        raise ValueError(f"Config interpolation did not converge: {config_path}")

    config["_runtime"] = {
        "config_path": str(config_path),
        "overrides": list(overrides or ()),
        "num_nodes": 1,
        "node_index": 0,
    }
    return config
