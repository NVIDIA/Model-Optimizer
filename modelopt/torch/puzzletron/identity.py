# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "Identity",
    "canonicalize",
    "canonical_json",
    "stable_hash",
    "model_identity",
    "config_identity",
    "block_config_identity",
    "candidate_identity",
    "stage_identity",
    "score_identity",
    "mip_identity",
    "mip_execution_identity",
    "vllm_settings_identity",
    "solution_identity",
    "cache_key",
]


@dataclass(frozen=True)
class Identity:
    """Stable short hash plus the canonical payload that produced it."""

    kind: str
    value: str
    payload: Any

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "value": self.value, "payload": canonicalize(self.payload)}


def _qualified_name(obj: Any) -> str:
    module = getattr(obj, "__module__", None)
    qualname = getattr(obj, "__qualname__", None) or getattr(obj, "__name__", None)
    if module and qualname:
        return f"{module}.{qualname}"
    if qualname:
        return str(qualname)
    return repr(obj)


def canonicalize(obj: Any) -> Any:
    """Convert arbitrary simple objects to a deterministic JSON-compatible shape."""
    if isinstance(obj, Identity):
        return obj.to_dict()
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if dataclasses.is_dataclass(obj):
        if hasattr(obj, "to_dict"):
            return canonicalize(obj.to_dict())
        return canonicalize(dataclasses.asdict(obj))
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, type):
        return {"__type__": _qualified_name(obj)}
    if isinstance(obj, dict):
        return {
            str(key): canonicalize(obj[key])
            for key in sorted(obj, key=lambda key: (type(key).__name__, repr(key)))
        }
    if isinstance(obj, (list, tuple)):
        return [canonicalize(value) for value in obj]
    if isinstance(obj, set):
        return [canonicalize(value) for value in sorted(obj, key=repr)]
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        try:
            return canonicalize(obj.to_dict())
        except TypeError:
            return {"__repr__": repr(obj)}
    if callable(obj):
        return {"__callable__": _qualified_name(obj)}
    try:
        json.dumps(obj)
    except TypeError:
        return {"__repr__": repr(obj)}
    return obj


def canonical_json(obj: Any) -> str:
    return json.dumps(canonicalize(obj), sort_keys=True, separators=(",", ":"), default=str)


def stable_hash(obj: Any, *, prefix: str = "id", length: int = 16) -> str:
    digest = hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"


def _identity(kind: str, payload: Any, *, prefix: str | None = None) -> Identity:
    return Identity(kind=kind, value=stable_hash(payload, prefix=prefix or kind), payload=payload)


def model_identity(config: Any, tokenizer: Any = None, base_checkpoint_manifest: Any = None) -> Identity:
    payload = {
        "config": config,
        "tokenizer": tokenizer,
        "base_checkpoint_manifest": base_checkpoint_manifest,
    }
    return _identity("model", payload)


def config_identity(config: Any) -> Identity:
    return _identity("config", config)


def block_config_identity(block_config: Any) -> Identity:
    return _identity("block_config", block_config)


def candidate_identity(layer_idx: int, block_config: Any, source: Any) -> Identity:
    return _identity(
        "candidate",
        {"layer_idx": layer_idx, "block_config": block_config, "source": source},
    )


def stage_identity(stage: str, inputs: Any, settings: Any, code_version: Any = None) -> Identity:
    return _identity(
        "stage",
        {
            "stage": stage,
            "inputs": inputs,
            "settings": settings,
            "code_version": code_version,
        },
    )


def score_identity(candidate: Any, data_identity: Any, scorer_settings: Any) -> Identity:
    return _identity(
        "score",
        {
            "candidate": candidate,
            "data_identity": data_identity,
            "scorer_settings": scorer_settings,
        },
    )


def mip_identity(library: Any, constraints: Any, objective: Any, solver_settings: Any) -> Identity:
    return _identity(
        "mip",
        {
            "library": library,
            "constraints": constraints,
            "objective": objective,
            "solver_settings": solver_settings,
        },
    )


def mip_execution_identity(
    mip_config: Any,
    *,
    widths: Any,
    max_depth: int,
    depth_trajectory: Any,
    solve_only: bool,
    input_artifact_identity: str,
) -> str:
    """Identify the complete concrete MIP expansion and its materialization mode."""

    return stable_hash(
        {
            "mip_config": mip_config,
            "widths": widths,
            "max_depth": int(max_depth),
            "depth_trajectory": depth_trajectory,
            "solve_only": bool(solve_only),
            "input_artifact_identity": input_artifact_identity,
        },
        prefix="mip_execution",
    )


def vllm_settings_identity(settings: Any) -> Identity:
    return _identity("vllm_settings", settings)


def solution_identity(layer_to_candidate: Any, constraints: Any, objective: Any) -> Identity:
    return _identity(
        "solution",
        {
            "layer_to_candidate": layer_to_candidate,
            "constraints": constraints,
            "objective": objective,
        },
    )


def cache_key(stage: str, inputs: Any, settings: Any) -> Identity:
    return _identity("cache", {"stage": stage, "inputs": inputs, "settings": settings})
