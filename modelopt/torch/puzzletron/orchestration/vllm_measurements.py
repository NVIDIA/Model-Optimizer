# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-light view of named vLLM measurement contracts."""

from __future__ import annotations

import hashlib
import json
import re
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .mesh import normalize_vllm_topology

__all__ = ["VllmMeasurement", "normalize_vllm_measurements"]

_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def _merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


@dataclass(frozen=True)
class VllmMeasurement:
    measurement_id: str
    prefill_seq_len: int
    generation_seq_len: int
    batch_size: int
    max_num_seqs: int
    granularity: str
    runtime_stats: Mapping[str, Any]
    model_hidden_sizes: tuple[int, ...] = ()
    description: str = ""
    legacy: bool = False

    def __post_init__(self) -> None:
        if not _ID.fullmatch(self.measurement_id):
            raise ValueError(f"invalid vLLM measurement ID {self.measurement_id!r}")
        values = (
            self.prefill_seq_len,
            self.generation_seq_len,
            self.batch_size,
            self.max_num_seqs,
        )
        if any(int(value) < 1 for value in values):
            raise ValueError("vLLM workload values must be positive")
        if self.max_num_seqs < self.batch_size:
            raise ValueError("vLLM max_num_seqs must be at least batch_size")
        if self.granularity not in {"block", "subblock"}:
            raise ValueError("vLLM granularity must be block or subblock")
        normalize_vllm_topology(self.topology)

    @property
    def topology(self) -> Mapping[str, Any]:
        return dict(self.runtime_stats.get("topology") or {})

    @property
    def gpu_group_size(self) -> int:
        return int(normalize_vllm_topology(self.topology)["gpu_count"])

    @property
    def identity(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    @property
    def relative_stats_path(self) -> Path:
        if self.legacy:
            return Path("subblock_stats.json")
        return (
            Path("artifacts")
            / "vllm_stats"
            / "measurements"
            / self.measurement_id
            / "subblock_stats.json"
        )

    def to_dict(self) -> dict[str, Any]:
        result = {
            "prefill_seq_len": self.prefill_seq_len,
            "generation_seq_len": self.generation_seq_len,
            "batch_size": self.batch_size,
            "max_num_seqs": self.max_num_seqs,
            "granularity": self.granularity,
            "runtime_stats": deepcopy(dict(self.runtime_stats)),
        }
        if self.model_hidden_sizes:
            result["model_hidden_sizes"] = list(self.model_hidden_sizes)
        if self.description:
            result["description"] = self.description
        return result


def normalize_vllm_measurements(
    config: Mapping[str, Any],
) -> OrderedDict[str, VllmMeasurement]:
    """Normalize named settings while preserving legacy output behavior."""

    vllm = config.get("vllm_stats", config)
    if not isinstance(vllm, Mapping):
        raise ValueError("vllm_stats must be a mapping")
    runtime = deepcopy(dict(vllm.get("runtime_stats") or {}))
    batches = tuple(int(value) for value in (vllm.get("batch_sizes") or (1,)))
    common = {
        "prefill_seq_len": int(vllm.get("prefill_seq_len", 4096)),
        "generation_seq_len": int(vllm.get("generation_seq_len", 1024)),
        "batch_size": batches[0],
        "max_num_seqs": int(runtime.get("max_num_seqs", batches[0])),
        "granularity": str(runtime.get("granularity", "block")),
        "runtime_stats": runtime,
        "model_hidden_sizes": tuple(int(v) for v in vllm.get("model_hidden_sizes", ())),
    }
    named = vllm.get("measurements") or {}
    if not named:
        return OrderedDict(
            (("default", VllmMeasurement("default", legacy=True, **common)),)
        )
    if not isinstance(named, Mapping):
        raise ValueError("vllm_stats.measurements must be a mapping")
    result: OrderedDict[str, VllmMeasurement] = OrderedDict()
    identities = {}
    for measurement_id, raw in named.items():
        if not isinstance(raw, Mapping):
            raise ValueError(f"vLLM measurement {measurement_id!r} must be a mapping")
        selected_runtime = _merge(runtime, raw.get("runtime_stats") or {})
        batch = int(raw.get("batch_size", common["batch_size"]))
        measurement = VllmMeasurement(
            str(measurement_id),
            prefill_seq_len=int(raw.get("prefill_seq_len", common["prefill_seq_len"])),
            generation_seq_len=int(
                raw.get("generation_seq_len", common["generation_seq_len"])
            ),
            batch_size=batch,
            max_num_seqs=int(
                raw.get("max_num_seqs", selected_runtime.get("max_num_seqs", batch))
            ),
            granularity=str(
                raw.get("granularity", selected_runtime.get("granularity", "block"))
            ),
            runtime_stats=selected_runtime,
            model_hidden_sizes=tuple(
                int(value)
                for value in raw.get("model_hidden_sizes", common["model_hidden_sizes"])
            ),
            description=str(raw.get("description", "")),
        )
        fingerprint = json.dumps(measurement.to_dict(), sort_keys=True)
        if fingerprint in identities:
            raise ValueError(
                f"vLLM measurement {measurement_id!r} duplicates {identities[fingerprint]!r}"
            )
        identities[fingerprint] = str(measurement_id)
        result[str(measurement_id)] = measurement
    return result
