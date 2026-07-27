# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lossless per-data-lane observations for bypass distillation."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

__all__ = [
    "BypassObservation",
    "CandidateCatalog",
    "ObservationWriter",
    "merge_rank_observations",
    "normalized_parameter_ratio",
]


def normalized_parameter_ratio(active_params: int, teacher_params: int) -> float:
    """Normalize one candidate's active parameters by its teacher counterpart."""

    active_params = int(active_params)
    teacher_params = int(teacher_params)
    if teacher_params <= 0:
        raise ValueError(f"teacher parameter count must be positive, got {teacher_params}")
    if active_params < 0 or active_params > teacher_params:
        raise ValueError(
            f"active parameter count must be in [0, {teacher_params}], got {active_params}"
        )
    return active_params / teacher_params


@dataclass(frozen=True)
class BypassObservation:
    """One loss point for one logical DP lane and trained unit."""

    step: int
    micro_step: int
    dp_lane: int
    granularity: str
    layer_idx: int
    loss: float
    candidate_id: str
    active_params: int
    teacher_params: int
    parameter_ratio: float
    subblock_kind: str | None = None
    subblock_name: str | None = None
    hidden_width: int | None = None
    ple_width: int | None = None
    learning_rate: float | None = None
    grad_norm: float | None = None
    elapsed_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CandidateCatalog:
    """Map stable candidate IDs to complete resolved configurations."""

    def __init__(self, entries: dict[str, Any] | None = None):
        self._entries = dict(entries or {})

    def __len__(self) -> int:
        return len(self._entries)

    def register(self, candidate_id: str, config: Any) -> None:
        candidate_id = str(candidate_id)
        normalized = json.loads(json.dumps(config, sort_keys=True, default=str))
        previous = self._entries.get(candidate_id)
        if previous is not None and previous != normalized:
            raise RuntimeError(
                f"candidate {candidate_id!r} has conflicting complete configurations"
            )
        self._entries[candidate_id] = normalized

    def to_dict(self) -> dict[str, Any]:
        return dict(sorted(self._entries.items()))

    def merge(self, other: "CandidateCatalog") -> None:
        for candidate_id, config in other.to_dict().items():
            self.register(candidate_id, config)

    def write(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)


def _lane_selections(rank_payloads) -> dict[int, dict[str, Any]]:
    selections: dict[int, dict[str, Any]] = {}
    for payload in rank_payloads:
        lane = int(payload["dp_lane"])
        selection = payload["selection"]
        previous = selections.get(lane)
        if previous is not None and previous != selection:
            raise RuntimeError(f"logical DP lane {lane} has conflicting selections")
        selections[lane] = selection
    return selections


def merge_rank_observations(
    rank_payloads,
    *,
    step: int,
    micro_step: int = 0,
    granularity: str,
    learning_rate: float | None,
    grad_norm: float | None,
    elapsed_seconds: float | None,
    candidate_metadata: dict[int, dict[int, dict[str, dict[str, Any]]]] | None = None,
) -> tuple[list[BypassObservation], CandidateCatalog]:
    """Merge model-parallel copies while preserving separate DP-lane points."""

    if granularity not in {"block", "subblock"}:
        raise ValueError(f"unsupported bypass observation granularity {granularity!r}")
    selections = _lane_selections(rank_payloads)
    layer_losses: dict[tuple[int, int], list[float]] = defaultdict(list)
    subblock_losses: dict[tuple[int, int, str, str], list[float]] = defaultdict(list)
    for payload in rank_payloads:
        lane = int(payload["dp_lane"])
        for layer_idx, loss in (payload.get("per_layer_loss") or {}).items():
            layer_losses[(lane, int(layer_idx))].append(float(loss))
        for key, loss in (payload.get("per_subblock_loss") or {}).items():
            layer_idx, kind, name = str(key).split(":", 2)
            subblock_losses[(lane, int(layer_idx), kind, name)].append(float(loss))

    catalog = CandidateCatalog()
    points: list[BypassObservation] = []

    def resolved_layer(selection, layer_idx):
        layer = next(
            item for item in selection["layers"] if int(item["layer_idx"]) == layer_idx
        )
        if "block_config" in layer:
            return layer
        width = int(selection["hidden_width"])
        metadata = (candidate_metadata or {})[width][layer_idx][str(layer["candidate_id"])]
        return {**metadata, **layer}

    if granularity == "block":
        for (lane, layer_idx), values in sorted(layer_losses.items()):
            selection = selections[lane]
            layer = resolved_layer(selection, layer_idx)
            candidate_id = str(layer["candidate_id"])
            catalog.register(candidate_id, layer["block_config"])
            active = int(layer["parameter_count"])
            teacher = int(layer["teacher_parameter_count"])
            points.append(
                BypassObservation(
                    step=int(step),
                    micro_step=int(micro_step),
                    dp_lane=lane,
                    granularity=granularity,
                    layer_idx=layer_idx,
                    loss=sum(values) / len(values),
                    candidate_id=candidate_id,
                    active_params=active,
                    teacher_params=teacher,
                    parameter_ratio=normalized_parameter_ratio(active, teacher),
                    hidden_width=selection.get("hidden_width"),
                    ple_width=selection.get("ple_width"),
                    learning_rate=learning_rate,
                    grad_norm=grad_norm,
                    elapsed_seconds=elapsed_seconds,
                )
            )
    else:
        for (lane, layer_idx, kind, name), values in sorted(subblock_losses.items()):
            selection = selections[lane]
            layer = resolved_layer(selection, layer_idx)
            subblock = next(
                item
                for item in layer["subblocks"]
                if str(item["kind"]) == kind and str(item["name"]) == name
            )
            candidate_id = f"{layer['candidate_id']}:{kind}:{name}"
            catalog.register(candidate_id, subblock["config"])
            active = int(subblock["parameter_count"])
            teacher = int(subblock["teacher_parameter_count"])
            points.append(
                BypassObservation(
                    step=int(step),
                    micro_step=int(micro_step),
                    dp_lane=lane,
                    granularity=granularity,
                    layer_idx=layer_idx,
                    subblock_kind=kind,
                    subblock_name=name,
                    loss=sum(values) / len(values),
                    candidate_id=candidate_id,
                    active_params=active,
                    teacher_params=teacher,
                    parameter_ratio=normalized_parameter_ratio(active, teacher),
                    hidden_width=selection.get("hidden_width"),
                    ple_width=selection.get("ple_width"),
                    learning_rate=learning_rate,
                    grad_norm=grad_norm,
                    elapsed_seconds=elapsed_seconds,
                )
            )
    return points, catalog


class ObservationWriter:
    """Recoverable append-only JSONL writer with checkpoint-step truncation."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._recover()

    def _records(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        records = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
        return records

    def _write_records(self, records: Iterable[dict[str, Any]]) -> None:
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        temporary.replace(self.path)

    def _recover(self) -> None:
        if not self.path.is_file():
            return
        records = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                break
        self._write_records(records)

    def append_step(self, step: int, observations: Iterable[BypassObservation]) -> None:
        payload = {
            "step": int(step),
            "observations": [
                observation.to_dict()
                if isinstance(observation, BypassObservation)
                else dict(observation)
                for observation in observations
            ],
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def truncate_after_step(self, step: int) -> None:
        self._write_records(
            record for record in self._records() if int(record["step"]) <= int(step)
        )

    def steps(self) -> list[int]:
        return [int(record["step"]) for record in self._records()]
