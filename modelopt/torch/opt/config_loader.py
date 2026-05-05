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
from importlib import import_module
from importlib.resources import files
from types import NoneType, UnionType

try:
    from importlib.resources.abc import Traversable
except ImportError:  # Python < 3.11
    from importlib.abc import Traversable
import re
import sys
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints

import yaml
from pydantic import TypeAdapter
from typing_extensions import NotRequired, Required, is_typeddict


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


@dataclass
class _RawConfig:
    """Raw YAML content plus optional ModelOpt schema metadata."""

    data: dict[str, Any] | list[Any] | _ListSnippet
    schema: str | None = None
    path: Path | Traversable | None = None


@dataclass
class _ResolvedImport:
    """Resolved imported payload plus the required schema declared by that payload."""

    data: Any
    schema: str
    schema_type: Any
    path: Path | Traversable | None


# Root to all built-in configs and recipes.
BUILTIN_CONFIG_ROOT = files("modelopt_recipes")

_EXMY_RE = re.compile(r"^[Ee](\d+)[Mm](\d+)$")
_EXMY_KEYS = frozenset({"num_bits", "scale_bits"})
_MODELOPT_SCHEMA_RE = re.compile(r"^\s*#\s*modelopt-schema:\s*(\S+)\s*$")


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


def _parse_modelopt_schema(text: str, config_path: Path | Traversable) -> str | None:
    """Parse a ``# modelopt-schema: ...`` preamble comment, if present."""
    schema: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if not stripped.startswith("#"):
            break
        match = _MODELOPT_SCHEMA_RE.match(line)
        if not match:
            continue
        if schema is not None:
            raise ValueError(f"Config file {config_path}: multiple modelopt-schema comments found.")
        schema = match.group(1)
    return schema


def _load_raw_config_with_schema(config_file: str | Path | Traversable) -> _RawConfig:
    """Load a config YAML without resolving ``$import`` references."""
    config_path = _resolve_config_path(config_file)
    text = config_path.read_text(encoding="utf-8")
    schema = _parse_modelopt_schema(text, config_path)
    docs = list(yaml.safe_load_all(text))

    if len(docs) == 0 or docs[0] is None:
        return _RawConfig({}, schema=schema, path=config_path)
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
            return _RawConfig(
                _ListSnippet(
                    imports=imports,
                    content=_parse_exmy_num_bits(content),
                ),
                schema=schema,
                path=config_path,
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
            f"Config file {config_path} must contain a YAML mapping or list, "
            f"got {type(_raw).__name__}"
        )
    return _RawConfig(
        _parse_exmy_num_bits(_raw),
        schema=schema,
        path=config_path,
    )


def _load_raw_config(
    config_file: str | Path | Traversable,
) -> dict[str, Any] | list[Any] | _ListSnippet:
    """Load a config YAML without resolving ``$import`` references."""
    return _load_raw_config_with_schema(config_file).data


# ---------------------------------------------------------------------------
# $import resolution
# ---------------------------------------------------------------------------

_IMPORT_KEY = "$import"


def _schema_type(schema_path: str) -> Any:
    """Resolve a schema path to a Python type.

    ``modelopt-schema`` comments are intentionally limited to import paths under
    ``modelopt.*`` so config files cannot trigger arbitrary third-party imports.
    The resolved object is expected to be a Pydantic-validatable type annotation,
    such as a BaseModel class, TypedDict, list[TypedDict], or union/type alias.

    If the target module is still being initialized and the requested schema has
    not been defined yet, raise an error that points to the likely circular import.
    """
    if not schema_path.startswith("modelopt."):
        raise ValueError(
            f"Unsupported modelopt-schema {schema_path!r}; schemas must live under 'modelopt.'."
        )

    module_name, _, attr_name = schema_path.rpartition(".")
    if not module_name or not attr_name:
        raise ValueError(f"Invalid modelopt-schema path: {schema_path!r}.")

    module = sys.modules.get(module_name) or import_module(module_name)
    try:
        schema_type: Any = module
        for part in attr_name.split("."):
            schema_type = getattr(schema_type, part)
        return schema_type
    except AttributeError as exc:
        is_initializing = getattr(getattr(module, "__spec__", None), "_initializing", False)
        if is_initializing:
            raise ValueError(
                f"Cannot resolve modelopt-schema {schema_path!r}: module {module_name!r} is "
                "still being initialized. This likely indicates a circular import or a schema "
                "defined after config loading."
            ) from exc
        raise ValueError(f"Cannot resolve modelopt-schema {schema_path!r}.") from exc


def _schema_label(schema_type: Any | None, schema_path: str | None = None) -> str:
    """Return a compact human-readable schema name for diagnostics."""
    if schema_path:
        return schema_path
    if schema_type is None:
        return "<untyped>"
    return getattr(schema_type, "__qualname__", repr(schema_type))


def _unwrap_schema_type(schema_type: Any | None) -> Any | None:
    """Unwrap typing wrappers that do not change the value shape."""
    if schema_type is None:
        return None
    origin = get_origin(schema_type)
    if origin in (Required, NotRequired):
        return _unwrap_schema_type(get_args(schema_type)[0])
    if origin in (UnionType, Union):
        args = tuple(arg for arg in get_args(schema_type) if arg is not NoneType)
        if len(args) == 1:
            return _unwrap_schema_type(args[0])
    return schema_type


def _schema_equal(left: Any | None, right: Any | None) -> bool:
    """Compare schema annotations structurally enough for import splice decisions."""
    left = _unwrap_schema_type(left)
    right = _unwrap_schema_type(right)
    if left == right:
        return True

    left_origin, right_origin = get_origin(left), get_origin(right)
    if left_origin is None or right_origin is None or left_origin != right_origin:
        return False

    left_args = get_args(left)
    right_args = get_args(right)
    return len(left_args) == len(right_args) and all(
        _schema_equal(l_arg, r_arg) for l_arg, r_arg in zip(left_args, right_args)
    )


def _list_element_schema(schema_type: Any | None) -> Any | None:
    """Return the element schema for a typed ``list[T]`` annotation."""
    schema_type = _unwrap_schema_type(schema_type)
    if get_origin(schema_type) is not list:
        return None
    args = get_args(schema_type)
    if len(args) != 1 or args[0] is Any:
        return None
    return _unwrap_schema_type(args[0])


def _child_schema(schema_type: Any | None, key: Any) -> Any | None:
    """Return the schema for ``key`` under a Pydantic model, TypedDict, or dict annotation."""
    schema_type = _unwrap_schema_type(schema_type)
    if schema_type is None:
        return None

    model_fields = getattr(schema_type, "model_fields", None)
    if isinstance(key, str) and model_fields and key in model_fields:
        return _unwrap_schema_type(model_fields[key].annotation)

    if isinstance(key, str) and is_typeddict(schema_type):
        try:
            annotations = get_type_hints(schema_type, include_extras=True)
        except Exception:
            annotations = getattr(schema_type, "__annotations__", {})
        return _unwrap_schema_type(annotations.get(key))

    origin = get_origin(schema_type)
    if origin is dict:
        args = get_args(schema_type)
        if len(args) == 2:
            return _unwrap_schema_type(args[1])

    return None


def _validate_modelopt_schema(
    schema_path: str | None,
    data: Any,
    config_path: Any,
    schema_type: Any | None = None,
) -> None:
    """Validate resolved config content against the requested schema without mutating it."""
    if schema_type is None and not schema_path:
        return
    if schema_type is None:
        assert schema_path is not None
        schema_type = _schema_type(schema_path)
    try:
        # TypeAdapter validates the schema types we allow here: BaseModel classes
        # plus regular typing constructs such as TypedDict, list[TypedDict], unions,
        # and aliases. Schema comments are not treated as arbitrary validators.
        TypeAdapter(schema_type).validate_python(data)
    except Exception as exc:
        raise ValueError(
            f"Config file {config_path} does not match modelopt-schema "
            f"{_schema_label(schema_type, schema_path)!r}: {exc}"
        ) from exc


def _resolve_imports(
    data: dict[str, Any] | _ListSnippet,
    _loading: frozenset[str] | None = None,
    schema_type: Any | None = None,
) -> dict[str, Any] | list[Any]:
    """Resolve the ``imports`` section and ``$import`` references.

    Accepts either a raw dict (with optional top-level ``imports:``) or a
    :class:`_ListSnippet` (a list body carrying its own ``imports``). Returns
    a dict for the former and a list for the latter — the imports section is
    consumed.  Bare ``$import`` entries inside lists require ``schema_type`` so
    the resolver can distinguish list splicing from element appending.

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
        unresolved = _find_import_marker(body)
        if unresolved is not None:
            ref_name, context = unresolved
            raise ValueError(
                f"Unknown $import reference {ref_name!r} in {context}. No imports are declared."
            )
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
    import_map: dict[str, _ResolvedImport] = {}
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
        raw_snippet = _load_raw_config_with_schema(config_path)
        # Every path listed under ``imports`` is a reusable snippet dependency.
        # Require an explicit schema before exposing it to either dict-valued
        # imports or typed list append/splice decisions.
        if raw_snippet.schema is None:
            raise ValueError(
                f"Import {name!r} ({raw_snippet.path}) must reference a snippet with "
                "a modelopt-schema comment."
            )
        snippet_schema = raw_snippet.schema
        snippet = raw_snippet.data
        snippet_schema_type = _schema_type(snippet_schema)
        if isinstance(snippet, _ListSnippet) or (
            isinstance(snippet, dict) and "imports" in snippet
        ):
            snippet = _resolve_imports(
                snippet, _loading | {cycle_key}, schema_type=snippet_schema_type
            )
        _validate_modelopt_schema(
            snippet_schema, snippet, raw_snippet.path, schema_type=snippet_schema_type
        )
        import_map[name] = _ResolvedImport(
            data=snippet,
            schema=snippet_schema,
            schema_type=snippet_schema_type,
            path=raw_snippet.path,
        )

    def _lookup(ref_name: str, context: str) -> _ResolvedImport:
        if ref_name not in import_map:
            raise ValueError(
                f"Unknown $import reference {ref_name!r} in {context}. "
                f"Available imports: {list(import_map.keys())}"
            )
        return import_map[ref_name]

    def _resolve_list_import(
        imported: _ResolvedImport, list_schema: Any | None, ref_name: str, context: str
    ) -> list[Any]:
        """Resolve a bare list-entry import using the containing list's schema."""
        element_schema = _list_element_schema(list_schema)
        if element_schema is None:
            raise ValueError(
                f"$import {ref_name!r} in list at {context} requires a typed list schema "
                "(expected list[ElementType])."
            )

        if _schema_equal(imported.schema_type, list_schema):
            if not isinstance(imported.data, list):
                raise ValueError(
                    f"$import {ref_name!r} in list at {context} declared schema "
                    f"{_schema_label(imported.schema_type, imported.schema)!r} but resolved to "
                    f"{type(imported.data).__name__}, expected list."
                )
            return list(imported.data)

        if _schema_equal(imported.schema_type, element_schema):
            return [imported.data]

        raise ValueError(
            f"$import {ref_name!r} in list at {context} has schema "
            f"{_schema_label(imported.schema_type, imported.schema)!r}; expected either "
            f"the list schema {_schema_label(list_schema)!r} for splicing or the element "
            f"schema {_schema_label(element_schema)!r} for appending."
        )

    def _resolve_value(obj: Any, value_schema: Any | None = None, context: str = "root") -> Any:
        """Recursively resolve ``$import`` markers anywhere in the config tree.

        - Dict with ``$import`` as only key in list context → splice or append by schema
        - Dict with ``$import`` key → replace/merge (import + override with inline keys)
        - List → resolve each element with the list element schema
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
                    imported = _lookup(rname, f"dict value at {context}")
                    snippet = imported.data
                    if not isinstance(snippet, dict):
                        raise ValueError(
                            f"$import {rname!r} in dict must resolve to a dict, "
                            f"got {type(snippet).__name__}."
                        )
                    merged.update(snippet)

                merged.update(inline_keys)
                return _resolve_value(
                    merged, value_schema, context
                )  # resolve any nested $import in result
            else:
                return {
                    k: _resolve_value(v, _child_schema(value_schema, k), f"{context}.{k}")
                    for k, v in obj.items()
                }
        elif isinstance(obj, list):
            resolved: list[Any] = []
            element_schema = _list_element_schema(value_schema)
            for index, entry in enumerate(obj):
                entry_context = f"{context}[{index}]"
                if isinstance(entry, dict) and _IMPORT_KEY in entry and len(entry) == 1:
                    # {$import: name} as sole key in a typed list splices list[T] snippets
                    # and appends T snippets. Untyped list imports are intentionally rejected.
                    imported = _lookup(entry[_IMPORT_KEY], f"list entry at {entry_context}")
                    resolved.extend(
                        _resolve_list_import(
                            imported, value_schema, entry[_IMPORT_KEY], entry_context
                        )
                    )
                else:
                    resolved.append(_resolve_value(entry, element_schema, entry_context))
            return resolved
        return obj

    return _resolve_value(body, schema_type)


def _find_import_marker(obj: Any, context: str = "root") -> tuple[Any, str] | None:
    """Return the first unresolved ``$import`` marker in ``obj``, if any."""
    if isinstance(obj, dict):
        if _IMPORT_KEY in obj:
            return obj[_IMPORT_KEY], context
        for key, value in obj.items():
            found = _find_import_marker(value, f"{context}.{key}")
            if found is not None:
                return found
    elif isinstance(obj, list):
        for index, entry in enumerate(obj):
            found = _find_import_marker(entry, f"{context}[{index}]")
            if found is not None:
                return found
    return None


def load_config(
    config_path: str | Path | Traversable,
    *,
    schema_type: Any | None = None,
) -> dict[str, Any] | list[Any]:
    """Load a YAML config and resolve all ``$import`` references.

    This is the primary config loading entry point.  It loads the YAML file,
    resolves any ``imports`` / ``$import`` directives, and returns the final
    config dict or list.

    ``schema_type`` supplies a typing context for import resolution when the
    file itself has no ``modelopt-schema`` comment. It is intentionally not a
    request to validate the top-level file. Top-level files are validated only
    when they declare ``modelopt-schema``; imported snippets are stricter and
    must always declare ``modelopt-schema``.
    """
    raw = _load_raw_config_with_schema(config_path)
    data = raw.data
    declared_schema_type = _schema_type(raw.schema) if raw.schema else None
    resolver_schema_type = declared_schema_type or schema_type

    if isinstance(data, (_ListSnippet, dict)):
        data = _resolve_imports(data, schema_type=resolver_schema_type)
    _validate_modelopt_schema(raw.schema, data, raw.path, schema_type=declared_schema_type)
    return data
