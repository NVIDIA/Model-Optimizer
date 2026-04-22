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

from dataclasses import dataclass, field
from importlib.resources import files

try:
    from importlib.resources.abc import Traversable
except ImportError:  # Python < 3.11
    from importlib.abc import Traversable
import re
from pathlib import Path
from typing import Any

import yaml


@dataclass
class _ListSnippet:
    """Multi-document YAML: a header dict (with optional ``imports:``) + a list body.

    YAML requires one root node per document, so a file that is "a list with an
    ``imports`` section" has to use two documents separated by ``---``. This
    wrapper is the internal transport carrying both pieces from
    :func:`_load_raw_config` to :func:`_resolve_imports` without smuggling them
    through a sentinel dict key (which would collide if a user happened to
    choose the same key name).
    """

    imports: dict[str, Any] = field(default_factory=dict)
    content: list[Any] = field(default_factory=list)


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


def _resolve_config_path(config_file: str | Path | Traversable) -> Path | Traversable:
    """Probe the filesystem and built-in library to locate a config file.

    Return type mirrors the input family: filesystem paths return ``Path``;
    built-in package resources return a ``Traversable``. Raises ``ValueError``
    if no candidate exists.

    Factored out of :func:`_load_raw_config` so :func:`_resolve_imports` can
    compute a canonical cycle-detection key without reading the file twice.
    """
    # Probe order: filesystem first, then built-in library.
    # This lets users override built-in configs by placing a file locally.
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

    for path in paths_to_check:
        if path.is_file():
            return path
    raise ValueError(f"Cannot find config file of {config_file}, paths checked: {paths_to_check}")


def _canonical_key(path: Path | Traversable) -> str:
    """Stable cycle-detection key for :func:`_resolve_imports`.

    Filesystem paths are resolved (``Path.resolve()``) so that aliases like
    ``foo/bar``, ``./foo/bar``, and their absolute form produce the same key.
    Built-in ``Traversable`` resources are already canonical — their ``str()``
    points into the installed package.
    """
    if isinstance(path, Path):
        try:
            return str(path.resolve())
        except OSError:
            return str(path)
    return str(path)


def _load_raw_config(
    config_file: str | Path | Traversable,
) -> dict[str, Any] | list[Any] | _ListSnippet:
    """Load a config YAML without resolving ``$import`` references.

    config_file: Path to a config yaml file. The path suffix can be omitted.

    Return type:
        * ``dict`` — single-document or two-document-dict YAML.
        * ``list`` — single-document list YAML.
        * :class:`_ListSnippet` — two-document YAML with a list body;
          carries the header's ``imports`` alongside the list content.
    """
    config_path = _resolve_config_path(config_file)
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
            # List body with a header dict (for declaring ``imports:``).
            # Only ``imports`` from the header is carried forward; any other
            # header keys are meaningless alongside a list body.
            imports = header.get("imports", {}) or {}
            return _ListSnippet(
                imports=imports,
                content=_parse_exmy_num_bits(content),
            )
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
    data: dict[str, Any] | _ListSnippet, _loading: frozenset[str] | None = None
) -> dict[str, Any] | list[Any]:
    """Resolve the ``imports`` section and ``$import`` references.

    Accepts either a raw dict (with optional top-level ``imports:``) or a
    :class:`_ListSnippet` (a list body carrying its own ``imports``). Returns
    a dict for the former and a list for the latter — the imports section is
    consumed.

    See ``modelopt.recipe.loader`` module docstring for the full specification.
    This function lives in ``_config_loader`` (not ``loader``) so that it can be
    used from ``modelopt.torch.quantization.config`` without circular imports.
    """
    if isinstance(data, _ListSnippet):
        imports_dict = data.imports
        body: dict[str, Any] | list[Any] = data.content
    else:
        imports_dict = data.get("imports")
        body = {k: v for k, v in data.items() if k != "imports"}

    if not imports_dict:
        return body

    if not isinstance(imports_dict, dict):
        raise ValueError(
            f"'imports' must be a dict mapping names to config paths, got: {type(imports_dict).__name__}"
        )

    if _loading is None:
        _loading = frozenset()

    # Build name → config mapping (recursively resolve nested imports).
    # Cycle detection uses the *resolved* file path as the key so that aliases
    # such as ``foo/bar``, ``./foo/bar``, and its absolute form all map to the
    # same cycle entry.
    import_map: dict[str, Any] = {}
    for name, config_path in imports_dict.items():
        if not config_path:
            raise ValueError(f"Import {name!r} has an empty config path.")
        resolved_path = _resolve_config_path(config_path)
        cycle_key = _canonical_key(resolved_path)
        if cycle_key in _loading:
            raise ValueError(
                f"Circular import detected: {config_path!r} (resolves to "
                f"{cycle_key!r}) is already being loaded. "
                f"Import chain: {sorted(_loading)}"
            )
        snippet = _load_raw_config(config_path)
        if isinstance(snippet, _ListSnippet) or (
            isinstance(snippet, dict) and "imports" in snippet
        ):
            snippet = _resolve_imports(snippet, _loading | {cycle_key})
        import_map[name] = snippet

    def _lookup(ref_name: str, context: str) -> Any:
        if ref_name not in import_map:
            raise ValueError(
                f"Unknown $import reference {ref_name!r} in {context}. "
                f"Available imports: {list(import_map.keys())}"
            )
        return import_map[ref_name]

    def _resolve_value(obj: Any) -> Any:
        """Recursively resolve ``$import`` markers anywhere in the config tree.

        - Dict with ``$import`` as only key and list value → splice (in list context)
        - Dict with ``$import`` key → replace/merge (import + override with inline keys)
        - List → resolve each element (with list-splice for ``$import`` entries)
        - Other → return as-is
        """
        if isinstance(obj, dict):
            if _IMPORT_KEY in obj:
                # {$import: name, ...inline} → import, merge, override.
                # Read without mutating ``obj`` so _resolve_value stays pure and
                # idempotent — double resolution must be a no-op on the first
                # result, not silently corrupt it.
                ref = obj[_IMPORT_KEY]
                inline_keys = {k: v for k, v in obj.items() if k != _IMPORT_KEY}
                ref_names = ref if isinstance(ref, list) else [ref]

                merged: dict[str, Any] = {}
                for rname in ref_names:
                    snippet = _lookup(rname, "dict value")
                    if not isinstance(snippet, dict):
                        raise ValueError(
                            f"$import {rname!r} in dict must resolve to a dict, "
                            f"got {type(snippet).__name__}."
                        )
                    merged.update(snippet)

                merged.update(inline_keys)
                return _resolve_value(merged)  # resolve any nested $import in result
            else:
                return {k: _resolve_value(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            resolved: list[Any] = []
            for entry in obj:
                if isinstance(entry, dict) and _IMPORT_KEY in entry and len(entry) == 1:
                    # {$import: name} as sole key in list → splice
                    imported = _lookup(entry[_IMPORT_KEY], "list entry")
                    if not isinstance(imported, list):
                        raise ValueError(
                            f"$import {entry[_IMPORT_KEY]!r} in list must resolve to a "
                            f"list, got {type(imported).__name__}."
                        )
                    resolved.extend(_resolve_value(imported))
                else:
                    resolved.append(_resolve_value(entry))
            return resolved
        return obj

    return _resolve_value(body)


def load_config(config_path: str | Path | Traversable) -> dict[str, Any] | list[Any]:
    """Load a YAML config and resolve all ``$import`` references.

    This is the primary config loading entry point.  It loads the YAML file,
    resolves any ``imports`` / ``$import`` directives, and returns the final
    config dict or list.
    """
    data = _load_raw_config(config_path)
    if isinstance(data, _ListSnippet) or (isinstance(data, dict) and "imports" in data):
        data = _resolve_imports(data)
    return data
