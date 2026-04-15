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

"""General-purpose YAML config loading with ``$import`` resolution.

This module provides the config loading infrastructure used by both
``modelopt.recipe`` and ``modelopt.torch.quantization.config``.  It lives
in ``modelopt.torch.opt`` (the lowest dependency layer) to avoid circular
imports.
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

# Root to all built-in configs and recipes.
BUILTIN_CONFIG_ROOT = files("modelopt_recipes")

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


def _load_raw_config(config_file: str | Path | Traversable) -> dict[str, Any] | list[Any]:
    """Load a config YAML without resolving ``$import`` references.

    config_file: Path to a config yaml file. The path suffix can be omitted.
    """
    paths_to_check: list[Path | Traversable] = []
    if isinstance(config_file, str):
        if not config_file.endswith(".yml") and not config_file.endswith(".yaml"):
            paths_to_check.append(Path(f"{config_file}.yml"))
            paths_to_check.append(Path(f"{config_file}.yaml"))
            paths_to_check.append(BUILTIN_CONFIG_ROOT.joinpath(f"{config_file}.yml"))
            paths_to_check.append(BUILTIN_CONFIG_ROOT.joinpath(f"{config_file}.yaml"))
        else:
            paths_to_check.append(Path(config_file))
            paths_to_check.append(BUILTIN_CONFIG_ROOT.joinpath(config_file))
    elif isinstance(config_file, Path):
        if config_file.suffix in (".yml", ".yaml"):
            paths_to_check.append(config_file)
            if not config_file.is_absolute():
                paths_to_check.append(BUILTIN_CONFIG_ROOT.joinpath(str(config_file)))
        else:
            paths_to_check.append(Path(f"{config_file}.yml"))
            paths_to_check.append(Path(f"{config_file}.yaml"))
            if not config_file.is_absolute():
                paths_to_check.append(BUILTIN_CONFIG_ROOT.joinpath(f"{config_file}.yml"))
                paths_to_check.append(BUILTIN_CONFIG_ROOT.joinpath(f"{config_file}.yaml"))
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

    text = config_path.read_text(encoding="utf-8")
    docs = list(yaml.safe_load_all(text))

    if len(docs) == 0 or docs[0] is None:
        return {}
    if len(docs) == 1:
        _raw = docs[0]
    elif len(docs) == 2:
        # Multi-document: first doc is imports/metadata, second is content.
        # Merge the imports into the content for downstream resolution.
        header, content = docs[0], docs[1]
        if not isinstance(header, dict):
            raise ValueError(
                f"Config file {config_path}: first YAML document must be a mapping, "
                f"got {type(header).__name__}"
            )
        if content is None:
            content = {}
        if isinstance(content, dict):
            _raw = {**header, **content}
        elif isinstance(content, list):
            # List content with a header dict — attach imports via wrapper
            _raw = {**header, "_list_content": content}
        else:
            raise ValueError(
                f"Config file {config_path}: second YAML document must be a mapping or list, "
                f"got {type(content).__name__}"
            )
    else:
        raise ValueError(
            f"Config file {config_path}: expected 1 or 2 YAML documents, got {len(docs)}"
        )

    if not isinstance(_raw, (dict, list)):
        raise ValueError(
            f"Config file {config_path} must contain a YAML mapping or list, got {type(_raw).__name__}"
        )
    return _parse_exmy_num_bits(_raw)


# ---------------------------------------------------------------------------
# $import resolution
# ---------------------------------------------------------------------------

_IMPORT_KEY = "$import"


def _resolve_imports(
    data: dict[str, Any], _loading: frozenset[str] | None = None
) -> dict[str, Any]:
    """Resolve the ``imports`` section and ``$import`` references.

    See ``modelopt.recipe.loader`` module docstring for the full specification.
    This function lives in ``_config_loader`` (not ``loader``) so that it can be
    used from ``modelopt.torch.quantization.config`` without circular imports.
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
        snippet = _load_raw_config(config_path)
        if isinstance(snippet, dict) and "imports" in snippet:
            snippet = _resolve_imports(snippet, _loading | {config_path})
        # Unwrap _list_content (multi-document YAML: imports + list content)
        if isinstance(snippet, dict) and "_list_content" in snippet:
            snippet = snippet["_list_content"]
        import_map[name] = snippet

    def _lookup(ref_name: str, context: str) -> Any:
        if ref_name not in import_map:
            raise ValueError(
                f"Unknown $import reference {ref_name!r} in {context}. "
                f"Available imports: {list(import_map.keys())}"
            )
        return import_map[ref_name]

    def _resolve_list(entries: list[Any]) -> list[Any]:
        """Resolve $import markers in a list of entries."""
        resolved: list[Any] = []
        for entry in entries:
            if isinstance(entry, dict) and _IMPORT_KEY in entry:
                if len(entry) > 1:
                    raise ValueError(
                        f"$import must be the only key in the dict, got extra keys: "
                        f"{sorted(k for k in entry if k != _IMPORT_KEY)}"
                    )
                imported = _lookup(entry[_IMPORT_KEY], "list entry")
                if not isinstance(imported, list):
                    raise ValueError(
                        f"$import {entry[_IMPORT_KEY]!r} in list must resolve to a "
                        f"list, got {type(imported).__name__}."
                    )
                resolved.extend(imported)
            elif (
                isinstance(entry, dict)
                and isinstance(entry.get("cfg"), dict)
                and _IMPORT_KEY in entry["cfg"]
            ):
                ref = entry["cfg"].pop(_IMPORT_KEY)
                inline_keys = dict(entry["cfg"])
                ref_names = ref if isinstance(ref, list) else [ref]

                merged: dict[str, Any] = {}
                for rname in ref_names:
                    snippet = _lookup(rname, f"cfg of {entry}")
                    if not isinstance(snippet, dict):
                        raise ValueError(
                            f"$import {rname!r} in cfg must resolve to a dict, "
                            f"got {type(snippet).__name__}."
                        )
                    merged.update(snippet)

                merged.update(inline_keys)
                entry["cfg"] = merged
                resolved.append(entry)
            else:
                resolved.append(entry)
        return resolved

    # Resolve in quant_cfg (top-level or nested under quantize)
    for container in [data, data.get("quantize", {})]:
        if isinstance(container, dict):
            quant_cfg = container.get("quant_cfg")
            if isinstance(quant_cfg, list):
                container["quant_cfg"] = _resolve_list(quant_cfg)

    # Resolve in _list_content (multi-document snippets)
    if "_list_content" in data:
        data["_list_content"] = _resolve_list(data["_list_content"])

    return data


def load_config(config_path: str | Path | Traversable) -> dict[str, Any] | list[Any]:
    """Load a YAML config and resolve all ``$import`` references.

    This is the primary config loading entry point.  It loads the YAML file,
    resolves any ``imports`` / ``$import`` directives, and returns the final
    config dict or list.
    """
    data = _load_raw_config(config_path)
    if isinstance(data, dict) and "imports" in data:
        data = _resolve_imports(data)
    return data
