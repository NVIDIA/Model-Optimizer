# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit defaults-file loading and value provenance for setup v2."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import yaml

from puzzletron_setup import SetupError

__all__ = ["DefaultsResolver", "ResolvedDefault", "load_defaults"]


class _AnyMapping:
    pass


_ANY_MAPPING = _AnyMapping()
_SCHEMA = {
    "schema_version": None,
    "campaign": {
        "result_root": None,
        "generate_smoke": None,
        "generate_production": None,
    },
    "model": {
        "source": None,
        "revision": None,
        "trust_remote_code": None,
        "force_hf": None,
    },
    "data": {
        "source": None,
        "modality": None,
        "layout": None,
        "sequence_length": None,
    },
    "infrastructure": {
        "gpus_per_node": None,
        "execution_contract": {
            "repository": None,
            "venv": None,
            "container": None,
            "container_mounts": None,
            "prerun_commands": None,
            "postrun_commands": None,
        },
        "runner": {
            "kind": None,
            "slurm": {
                "account": None,
                "partition_interactive": None,
                "partition_batch": None,
                "partition_cpu": None,
                "interactive_max_nodes": None,
                "max_nodes": None,
                "time_limit": None,
                "qos": None,
            },
            "inventory": _ANY_MAPPING,
        },
    },
    "pruning": _ANY_MAPPING,
    "stages": _ANY_MAPPING,
    "profiles": _ANY_MAPPING,
    "vllm": _ANY_MAPPING,
    "mip": _ANY_MAPPING,
    "post_mip": _ANY_MAPPING,
    "output": _ANY_MAPPING,
}


@dataclass(frozen=True)
class ResolvedDefault:
    """One effective default and its highest-precedence source."""

    value: Any
    source: str


def _validate_mapping(value: Any, schema: Any, path: str) -> None:
    if schema is _ANY_MAPPING:
        if not isinstance(value, Mapping):
            raise SetupError(f"Defaults field {path} must be a mapping.")
        return
    if schema is None:
        return
    if not isinstance(value, Mapping):
        raise SetupError(f"Defaults field {path} must be a mapping.")
    for key, item in value.items():
        child_path = f"{path}.{key}" if path else str(key)
        if key not in schema:
            raise SetupError(f"Unknown defaults field: {child_path}")
        _validate_mapping(item, schema[key], child_path)


def load_defaults(path: Optional[Path]) -> dict[str, Any]:
    """Load an explicitly selected versioned defaults file."""

    if path is None:
        return {}
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise SetupError(f"Defaults file does not exist: {resolved}")
    try:
        payload = yaml.safe_load(resolved.read_text()) or {}
    except (OSError, yaml.YAMLError) as error:
        raise SetupError(f"Cannot read defaults file {resolved}: {error}") from error
    if not isinstance(payload, Mapping):
        raise SetupError(f"Defaults file must contain a YAML mapping: {resolved}")
    if payload.get("schema_version") != 1:
        raise SetupError(
            f"Unsupported defaults schema {payload.get('schema_version')!r}; expected 1."
        )
    _validate_mapping(payload, _SCHEMA, "")
    return deepcopy(dict(payload))


def _lookup(mapping: Mapping[str, Any], dotted_path: str) -> Tuple[bool, Any]:
    value: Any = mapping
    for key in dotted_path.split("."):
        if not isinstance(value, Mapping) or key not in value:
            return False, None
        value = value[key]
    return True, value


class DefaultsResolver:
    """Resolve defaults from ordered sources while preserving provenance."""

    def __init__(
        self,
        *,
        builtins: Optional[Mapping[str, Any]] = None,
        model_derived: Optional[Mapping[str, Any]] = None,
        file_defaults: Optional[Mapping[str, Any]] = None,
        preserved: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self._default_layers = (
            ("builtin", dict(builtins or {})),
            ("model", dict(model_derived or {})),
            ("defaults_file", dict(file_defaults or {})),
        )
        self._file_defaults = dict(file_defaults or {})
        self._layers = (
            *self._default_layers,
            ("preserved", dict(preserved or {})),
        )

    @staticmethod
    def _resolve_layers(
        layers: tuple[tuple[str, Mapping[str, Any]], ...],
        path: str,
        fallback: Any,
    ) -> ResolvedDefault:
        resolved = ResolvedDefault(deepcopy(fallback), "fallback")
        for source, layer in layers:
            found, value = _lookup(layer, path)
            if found:
                resolved = ResolvedDefault(deepcopy(value), source)
        return resolved

    def resolve(self, path: str, fallback: Any = None) -> ResolvedDefault:
        """Return the suggested value, including preserved wizard answers."""

        return self._resolve_layers(self._layers, path, fallback)

    def resolve_default(self, path: str, fallback: Any = None) -> ResolvedDefault:
        """Return built-in, model-derived, or explicit-file defaults."""

        return self._resolve_layers(self._default_layers, path, fallback)

    def file_default(self, path: str) -> Optional[ResolvedDefault]:
        """Return an explicitly supplied file default, if present."""

        found, value = _lookup(self._file_defaults, path)
        if not found:
            return None
        return ResolvedDefault(deepcopy(value), "defaults_file")
