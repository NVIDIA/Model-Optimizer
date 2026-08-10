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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit defaults-file loading and value provenance for setup v2."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from puzzletron_setup import SetupError

__all__ = ["DefaultsResolver", "ResolvedDefault", "load_defaults", "validate_defaults"]


class _AnyMapping:
    pass


_ANY_MAPPING = _AnyMapping()
_INTEGER_MINIMUMS = {
    "data.sequence_length": 1,
    "data.acquisition.seed": 0,
    "data.acquisition.train_samples": 1,
    "data.acquisition.validation_samples": 1,
    "data.acquisition.num_samples": 1,
    "data.acquisition.max_shards_per_subset": 1,
    "infrastructure.gpus_per_node": 1,
    "infrastructure.runner.slurm.interactive_max_nodes": 1,
    "infrastructure.runner.slurm.max_nodes": 1,
    "pruning.depth_remove": 0,
    "pruning.depth_importance_samples": 1,
    "pruning.width_importance_samples": 1,
    "pruning.sort_sanity_samples": 1,
    "pruning.width_sanity_samples": 1,
    "pruning.width_sanity_layer_count": 1,
    "pruning.width_sanity_targets_per_axis": 1,
    "pruning.replacement_samples": 1,
    "pruning.bypass.samples": 1,
    "pruning.bypass.sequence_length": 1,
    "pruning.bypass.batch_size": 1,
    "pruning.bypass.grad_accumulation_steps": 1,
    "vllm.prefill_seq_len": 1,
    "vllm.generation_seq_len": 1,
    "vllm.batch_size": 1,
    "vllm.max_num_seqs": 1,
    "vllm.topology.tensor_parallel_size": 1,
    "vllm.topology.pipeline_parallel_size": 1,
    "vllm.topology.data_parallel_size": 1,
    "vllm.topology.prefill_context_parallel_size": 1,
    "vllm.topology.decode_context_parallel_size": 1,
    "mip.num_solutions": 1,
}
_BOOLEAN_PATHS = {
    "campaign.generate_smoke",
    "campaign.generate_production",
    "model.trust_remote_code",
    "model.force_hf",
    "pruning.sort_sanity",
    "pruning.width_sanity",
    "pruning.slicing_sanity",
    "pruning.bypass.enabled",
    "vllm.enabled",
    "vllm.topology.enable_expert_parallel",
}
_PROFILE_INTEGER_FIELDS = {"tp", "cp", "pp", "dp", "dp_shard", "dp_replicate", "ep"}
_PROFILE_BOOLEAN_FIELDS = {"sequence_parallel"}
_STAGE_INTEGER_FIELDS = {"batch", "instances"}
_SEQUENCE_PATHS = {
    "infrastructure.execution_contract.prerun_commands",
    "infrastructure.execution_contract.postrun_commands",
}
_STRING_OR_SEQUENCE_PATHS = {"data.subsets", "data.acquisition.subsets"}
_SCHEMA = {
    "schema_version": None,
    "campaign": {
        "result_root": None,
        "generate_smoke": None,
        "generate_production": None,
    },
    "model": {
        "source": None,
        "trust_remote_code": None,
        "force_hf": None,
    },
    "data": {
        "source": None,
        "modality": None,
        "layout": None,
        "sequence_length": None,
        "subsets": None,
        "acquisition": {
            "output": None,
            "seed": None,
            "train_samples": None,
            "validation_samples": None,
            "subsets": None,
            "num_samples": None,
            "max_shards_per_subset": None,
        },
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


def _validate_leaf(value: Any, path: str) -> None:
    parts = path.split(".")
    minimum = _INTEGER_MINIMUMS.get(path)
    if len(parts) == 3 and parts[0] == "profiles" and parts[2] in _PROFILE_INTEGER_FIELDS:
        minimum = 1
    if len(parts) == 3 and parts[0] == "stages" and parts[2] in _STAGE_INTEGER_FIELDS:
        minimum = 1
    if minimum is not None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise SetupError(f"Defaults field {path} must be an integer.")
        if value < minimum:
            raise SetupError(f"Defaults field {path} must be at least {minimum}.")
    is_profile_boolean = (
        len(parts) == 3 and parts[0] == "profiles" and parts[2] in _PROFILE_BOOLEAN_FIELDS
    )
    if (path in _BOOLEAN_PATHS or is_profile_boolean) and not isinstance(value, bool):
        raise SetupError(f"Defaults field {path} must be a boolean.")
    if len(parts) == 4 and parts[:2] == ["pruning", "axes"] and parts[3] == "values":
        if (
            isinstance(value, (str, bytes))
            or not isinstance(value, Sequence)
            or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        ):
            raise SetupError(f"Defaults field {path} must be a sequence of integers.")
    if path in _SEQUENCE_PATHS and value is not None:
        if (
            isinstance(value, (str, bytes))
            or not isinstance(value, Sequence)
            or any(not isinstance(item, str) for item in value)
        ):
            raise SetupError(f"Defaults field {path} must be a sequence of strings.")
    is_profile_consumers = len(parts) == 3 and parts[:1] == ["profiles"] and parts[2] == "consumers"
    if is_profile_consumers:
        if (
            isinstance(value, (str, bytes))
            or not isinstance(value, Sequence)
            or any(not isinstance(item, str) for item in value)
        ):
            raise SetupError(f"Defaults field {path} must be a sequence of strings.")
    if path in _STRING_OR_SEQUENCE_PATHS and value is not None and not isinstance(value, str):
        if (
            isinstance(value, bytes)
            or not isinstance(value, Sequence)
            or any(not isinstance(item, str) for item in value)
        ):
            raise SetupError(f"Defaults field {path} must be a string or a sequence of strings.")


def _requires_mapping(path: str) -> bool:
    parts = path.split(".")
    return (
        (len(parts) == 2 and parts[0] in {"profiles", "stages"})
        or parts in (["pruning", "axes"], ["pruning", "bypass"], ["vllm", "topology"])
        or (len(parts) == 3 and parts[:2] == ["pruning", "axes"])
    )


def _validate_mapping(value: Any, schema: Any, path: str) -> None:
    if schema is _ANY_MAPPING:
        if not isinstance(value, Mapping):
            raise SetupError(f"Defaults field {path} must be a mapping.")
        for key, item in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            _validate_leaf(item, child_path)
            if _requires_mapping(child_path) and not isinstance(item, Mapping):
                raise SetupError(f"Defaults field {child_path} must be a mapping.")
            if isinstance(item, Mapping):
                _validate_mapping(item, _ANY_MAPPING, child_path)
        return
    if schema is None:
        _validate_leaf(value, path)
        return
    if not isinstance(value, Mapping):
        raise SetupError(f"Defaults field {path} must be a mapping.")
    for key, item in value.items():
        child_path = f"{path}.{key}" if path else str(key)
        if key not in schema:
            raise SetupError(f"Unknown defaults field: {child_path}")
        _validate_mapping(item, schema[key], child_path)


def validate_defaults(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and isolate one versioned setup-defaults mapping."""
    if payload.get("schema_version") != 1:
        raise SetupError(
            f"Unsupported defaults schema {payload.get('schema_version')!r}; expected 1."
        )
    _validate_mapping(payload, _SCHEMA, "")
    return deepcopy(dict(payload))


def load_defaults(path: Path | None) -> dict[str, Any]:
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
    return validate_defaults(payload)


def _lookup(mapping: Mapping[str, Any], dotted_path: str) -> tuple[bool, Any]:
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
        builtins: Mapping[str, Any] | None = None,
        model_derived: Mapping[str, Any] | None = None,
        preset_defaults: Mapping[str, Any] | None = None,
        model_profile_defaults: Mapping[str, Any] | None = None,
        file_defaults: Mapping[str, Any] | None = None,
        preserved: Mapping[str, Any] | None = None,
    ) -> None:
        """Build the ordered builtin, model, profile, file, and preserved layers."""
        self._resolutions: dict[str, ResolvedDefault] = {}
        self._default_layers = (
            ("builtin", dict(builtins or {})),
            ("model", dict(model_derived or {})),
            ("preset", dict(preset_defaults or {})),
            ("model_profile", dict(model_profile_defaults or {})),
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
        resolved = self._resolve_layers(self._layers, path, fallback)
        self._resolutions[path] = deepcopy(resolved)
        return resolved

    def resolve_default(self, path: str, fallback: Any = None) -> ResolvedDefault:
        """Return built-in, model-derived, or explicit-file defaults."""
        resolved = self._resolve_layers(self._default_layers, path, fallback)
        self._resolutions[path] = deepcopy(resolved)
        return resolved

    def file_default(self, path: str) -> ResolvedDefault | None:
        """Return an explicitly supplied file default, if present."""
        found, value = _lookup(self._file_defaults, path)
        if not found:
            return None
        resolved = ResolvedDefault(deepcopy(value), "defaults_file")
        self._resolutions[path] = deepcopy(resolved)
        return resolved

    def resolutions(self) -> Mapping[str, ResolvedDefault]:
        """Return every default decision resolved during this wizard run."""
        return deepcopy(self._resolutions)
