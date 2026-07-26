# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Durable candidate lineage and observations for configurable post-MIP flows."""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..identity import canonicalize, stable_hash

__all__ = [
    "ArchitectureCandidate",
    "ArtifactKind",
    "CandidateLedger",
    "CandidateRevision",
    "CandidateSet",
    "NodeObservation",
]


class ArtifactKind(str, Enum):
    CONFIG = "config"
    CHECKPOINT = "checkpoint"


@dataclass
class ArchitectureCandidate:
    architecture_id: str
    block_configs: list[Any]
    mip_metrics: dict[str, float] = field(default_factory=dict)
    origins: list[dict[str, Any]] = field(default_factory=list)
    origin_revision_id: str = ""


@dataclass
class CandidateRevision:
    revision_id: str
    architecture_id: str
    artifact_kind: ArtifactKind
    artifact: dict[str, Any]
    parent_revision_id: str | None = None
    producer_node: str = "mip"


@dataclass
class NodeObservation:
    node_id: str
    input_revision_id: str
    source_revision_id: str
    output_revision_id: str | None
    status: str
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass(frozen=True)
class CandidateSet:
    flow_id: str
    node_id: str
    revision_ids: tuple[str, ...]
    identity: str
    producer_execution_identity: str

    @classmethod
    def create(
        cls,
        flow_id: str,
        node_id: str,
        revision_ids: Iterable[str],
        *,
        producer_execution_identity: str,
    ) -> CandidateSet:
        values = tuple(dict.fromkeys(str(value) for value in revision_ids))
        return cls(
            flow_id=flow_id,
            node_id=node_id,
            revision_ids=values,
            identity=stable_hash(
                {
                    "flow_id": flow_id,
                    "node_id": node_id,
                    "revision_ids": values,
                    "producer_execution_identity": producer_execution_identity,
                },
                prefix="candidate_set",
            ),
            producer_execution_identity=producer_execution_identity,
        )


class CandidateLedger:
    """Single-writer transactional registry backed by immutable node result files."""

    VERSION = 2

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.registry_path = self.root / "candidate_registry.json"
        self.architectures: dict[str, ArchitectureCandidate] = {}
        self.revisions: dict[str, CandidateRevision] = {}
        self.observations: dict[str, dict[str, NodeObservation]] = {}
        self.active_mip_execution_identity = ""
        self.active_profile_ids: set[str] = set()
        self._load()

    def _load(self) -> None:
        if not self.registry_path.is_file():
            return
        payload = json.loads(self.registry_path.read_text())
        if int(payload.get("version", -1)) != self.VERSION:
            raise ValueError(f"unsupported candidate ledger version: {payload.get('version')}")
        self.architectures = {
            key: ArchitectureCandidate(**value)
            for key, value in dict(payload.get("architectures") or {}).items()
        }
        self.revisions = {
            key: CandidateRevision(
                **{**value, "artifact_kind": ArtifactKind(value["artifact_kind"])}
            )
            for key, value in dict(payload.get("revisions") or {}).items()
        }
        self.active_mip_execution_identity = str(
            payload.get("active_mip_execution_identity") or ""
        )
        self.active_profile_ids = {
            str(value) for value in payload.get("active_profile_ids") or ()
        }
        for current_path in sorted((self.root / "nodes").glob("*/current.json")):
            node_id = current_path.parent.name
            current = json.loads(current_path.read_text())
            execution_identity = str(current["execution_identity"])
            path = current_path.parent / "executions" / execution_identity / "observations.json"
            rows = json.loads(path.read_text())
            self.observations[node_id] = {
                row["input_revision_id"]: NodeObservation(**row) for row in rows
            }

    def publish(self) -> Path:
        payload = {
            "version": self.VERSION,
            "architectures": {
                key: canonicalize(asdict(value)) for key, value in self.architectures.items()
            },
            "revisions": {
                key: canonicalize(asdict(value)) for key, value in self.revisions.items()
            },
            "active_mip_execution_identity": self.active_mip_execution_identity,
            "active_profile_ids": sorted(self.active_profile_ids),
        }
        return self._atomic_json(self.registry_path, payload)

    def publish_node(
        self,
        node_id: str,
        observations: Iterable[NodeObservation],
        candidate_set: CandidateSet,
        execution_identity: str,
    ) -> tuple[Path, Path]:
        node_root = self.root / "nodes" / node_id
        execution_root = node_root / "executions" / execution_identity
        rows = list(observations)
        self.observations[node_id] = {row.input_revision_id: row for row in rows}
        observations_path = self._immutable_json(
            execution_root / "observations.json",
            [canonicalize(asdict(row)) for row in rows],
        )
        candidate_set_path = self._immutable_json(
            execution_root / "candidate_set.json", canonicalize(asdict(candidate_set))
        )
        index_path = node_root / "index.json"
        index = json.loads(index_path.read_text()) if index_path.is_file() else {"executions": []}
        if execution_identity not in index["executions"]:
            index["executions"].append(execution_identity)
        index["current"] = execution_identity
        self._atomic_json(index_path, index)
        self._atomic_json(node_root / "current.json", {"execution_identity": execution_identity})
        self.publish()
        return observations_path, candidate_set_path

    def load_candidate_set(self, node_id: str) -> CandidateSet:
        node_root = self.root / "nodes" / node_id
        current = json.loads((node_root / "current.json").read_text())
        payload = json.loads(
            (
                node_root
                / "executions"
                / str(current["execution_identity"])
                / "candidate_set.json"
            ).read_text()
        )
        return CandidateSet(
            flow_id=str(payload["flow_id"]),
            node_id=str(payload["node_id"]),
            revision_ids=tuple(payload["revision_ids"]),
            identity=str(payload["identity"]),
            producer_execution_identity=str(payload["producer_execution_identity"]),
        )

    def resolve_metric(self, revision_id: str, reference: str) -> float | None:
        revision = self.revisions[revision_id]
        architecture = self.architectures[revision.architecture_id]
        owner, separator, metric = reference.partition(".")
        if not separator:
            raise ValueError(
                "metric reference must be mip.<metric> or <node>.<metric>: "
                f"{reference}"
            )
        if owner == "mip":
            root_revision = revision
            while root_revision.parent_revision_id is not None:
                root_revision = self.revisions[root_revision.parent_revision_id]
            value = dict(root_revision.artifact.get("mip_metrics") or {}).get(
                metric, architecture.mip_metrics.get(metric)
            )
        else:
            observation = self._observation_for_revision(owner, revision_id)
            value = observation.metrics.get(metric) if observation else None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        return None

    def resolve_concurrency_metrics(
        self,
        revision_id: str,
        reference: str,
    ) -> dict[int, float]:
        """Resolve finite ``concurrency_<N>.<metric>`` values from one node observation."""

        owner, separator, metric = reference.partition(".")
        if not separator or not owner or not metric or owner == "mip":
            raise ValueError(
                f"concurrency sweep metric must be node-qualified as <node>.<metric>: {reference}"
            )
        observation = self._observation_for_revision(owner, revision_id)
        if observation is None:
            return {}
        pattern = re.compile(rf"^concurrency_([1-9][0-9]*)\.{re.escape(metric)}$")
        values = {}
        for name, value in observation.metrics.items():
            match = pattern.fullmatch(name)
            if (
                match is not None
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
            ):
                values[int(match.group(1))] = float(value)
        return values

    def source_revision(self, input_revision_id: str, source: str) -> CandidateRevision:
        current = self.revisions[input_revision_id]
        if source == "latest":
            return current
        if source == "origin":
            while current.parent_revision_id is not None:
                current = self.revisions[current.parent_revision_id]
            return current
        observation = self._observation_for_revision(source, input_revision_id)
        if observation is None or observation.output_revision_id is None:
            raise ValueError(
                f"node {source!r} produced no revision for architecture {current.architecture_id}"
            )
        return self.revisions[observation.output_revision_id]

    def _observation_for_revision(self, node_id: str, revision_id: str) -> NodeObservation | None:
        """Find a node observation on this revision's lineage.

        Transform nodes create a new revision, but all earlier measurements remain
        valid metadata for that architecture. Walking parents makes those metrics
        available to later filters without flattening or copying observations.
        """

        rows = self.observations.get(node_id, {})
        current_id: str | None = revision_id
        visited: set[str] = set()
        while current_id and current_id not in visited:
            visited.add(current_id)
            observation = rows.get(current_id)
            if observation is None:
                observation = next(
                    (row for row in rows.values() if row.output_revision_id == current_id),
                    None,
                )
            if observation is not None:
                return observation
            revision = self.revisions.get(current_id)
            current_id = revision.parent_revision_id if revision is not None else None
        architecture_id = self.revisions[revision_id].architecture_id
        return next(
            (
                row
                for row in rows.values()
                if (
                    row.output_revision_id in self.revisions
                    and self.revisions[row.output_revision_id].architecture_id == architecture_id
                )
                or (
                    row.input_revision_id in self.revisions
                    and self.revisions[row.input_revision_id].architecture_id == architecture_id
                )
            ),
            None,
        )

    def add_revision(
        self,
        *,
        architecture_id: str,
        artifact_kind: ArtifactKind,
        artifact: Mapping[str, Any],
        parent_revision_id: str | None,
        producer_node: str,
    ) -> CandidateRevision:
        payload = {
            "architecture_id": architecture_id,
            "artifact_kind": artifact_kind.value,
            "artifact": dict(artifact),
            "parent_revision_id": parent_revision_id,
            "producer_node": producer_node,
        }
        revision = CandidateRevision(
            revision_id=stable_hash(payload, prefix="revision"),
            architecture_id=architecture_id,
            artifact_kind=artifact_kind,
            artifact=dict(artifact),
            parent_revision_id=parent_revision_id,
            producer_node=producer_node,
        )
        self.revisions.setdefault(revision.revision_id, revision)
        return self.revisions[revision.revision_id]

    def candidate_metadata(self, revision_id: str) -> dict[str, Any]:
        """Return architecture, revision lineage, and accumulated observations."""

        revision = self.revisions[revision_id]
        lineage = []
        current: CandidateRevision | None = revision
        while current is not None:
            lineage.append(canonicalize(asdict(current)))
            current = (
                self.revisions.get(current.parent_revision_id)
                if current.parent_revision_id is not None
                else None
            )
        observations = {}
        for node_id in self.observations:
            observation = self._observation_for_revision(node_id, revision_id)
            if observation is not None:
                observations[node_id] = canonicalize(asdict(observation))
        return {
            "revision_id": revision_id,
            "architecture": canonicalize(
                asdict(self.architectures[revision.architecture_id])
            ),
            "lineage": lineage,
            "observations": observations,
        }

    def ingest_mip(self, puzzle_dir: str | Path) -> None:
        puzzle_dir = Path(puzzle_dir)
        active_path = puzzle_dir / "mip" / "active_profiles.json"
        try:
            active = json.loads(active_path.read_text())
        except (OSError, ValueError) as error:
            raise RuntimeError(
                f"current MIP profile manifest is unavailable: {active_path}"
            ) from error
        if active.get("status") != "success":
            raise RuntimeError(f"current MIP profile manifest is not complete: {active_path}")
        self.active_mip_execution_identity = str(active["execution_identity"])
        self.active_profile_ids = {str(value) for value in active.get("profile_ids") or ()}
        if not self.active_profile_ids:
            raise RuntimeError(f"current MIP profile manifest contains no profiles: {active_path}")
        for grid_path in sorted((puzzle_dir / "mip" / "profiles").glob("*/mip_grid.json")):
            grid = json.loads(grid_path.read_text())
            profile = dict(grid.get("profile") or {})
            profile["_execution_identity"] = grid.get("execution_identity")
            for scenario in grid.get("scenarios") or ():
                raw_solutions = self._solution_rows(scenario)
                result_rows = list(scenario.get("solutions") or ())
                if not result_rows and scenario.get("status") == "feasible":
                    result_rows = [scenario]
                for index, (raw, result) in enumerate(zip(raw_solutions, result_rows)):
                    self._ingest_solution(
                        profile,
                        scenario,
                        raw,
                        result,
                        index,
                        "heterogeneous",
                        str(scenario.get("solution_path")),
                    )
                homogeneous_path = scenario.get("homogeneous_solution_path")
                if homogeneous_path and Path(homogeneous_path).is_file():
                    homogeneous_raw = json.loads(Path(homogeneous_path).read_text())
                    for index, (raw, result) in enumerate(
                        zip(homogeneous_raw, scenario.get("homogeneous_solutions") or ())
                    ):
                        self._ingest_solution(
                            profile,
                            scenario,
                            raw,
                            result,
                            index,
                            "homogeneous",
                            str(homogeneous_path),
                        )
        self.publish()

    def root_set(self, flow_id: str, source: Mapping[str, Any]) -> CandidateSet:
        run = str(source.get("run") or "")
        if not run:
            raise ValueError(f"post-MIP flow {flow_id!r} source must name one MIP run")
        variants = source.get("variants", "all")
        objectives = source.get("objectives", "all")
        if isinstance(variants, str) and variants != "all":
            variants = [variants]
        if isinstance(objectives, str) and objectives != "all":
            objectives = [objectives]
        revision_ids = []
        for architecture in self.architectures.values():
            matching_origins = [
                origin
                for origin in architecture.origins
                if (
                origin.get("profile_id") in self.active_profile_ids
                and origin.get("mip_execution_identity")
                == self.active_mip_execution_identity
                and origin.get("run_id") == run
                and (variants == "all" or origin.get("variant_id") in variants)
                and (
                    objectives == "all"
                    or (origin.get("objective") or {}).get("metric") in objectives
                )
                )
            ]
            if matching_origins:
                matching_origins.sort(
                    key=lambda origin: (
                        str(origin.get("profile_id")),
                        str(origin.get("kind")),
                        int(origin.get("rank", 0)),
                    )
                )
                revision_ids.append(matching_origins[0]["revision_id"])
        return CandidateSet.create(
            flow_id,
            "source",
            sorted(revision_ids),
            producer_execution_identity=self.active_mip_execution_identity,
        )

    def _ingest_solution(
        self,
        profile: Mapping[str, Any],
        scenario: Mapping[str, Any],
        raw: Mapping[str, Any],
        result: Mapping[str, Any],
        rank: int,
        kind: str,
        solution_path: str | None = None,
    ) -> None:
        block_configs = list(raw.get("chosen_block_configs") or ())
        if not block_configs:
            block_configs = [
                row.get("block_config") or row.get("child_block_configs")
                for row in raw.get("chosen_replacements") or ()
            ]
        architecture_id = stable_hash(
            {
                "hidden_width": scenario.get("hidden_width"),
                "block_configs": block_configs,
            },
            prefix="architecture",
        )
        costs = {}
        for key, value in dict(
            result.get("total_costs") or raw.get("total_costs") or {}
        ).items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            key = str(key)
            costs[key] = float(value)
            if key.startswith("stats."):
                costs[key.removeprefix("stats.")] = float(value)
        score = result.get("score")
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            costs["score"] = float(score)
        origin = {
            "profile_id": profile.get("id"),
            "run_id": profile.get("run_id") or profile.get("base_profile_id"),
            "variant_id": profile.get("variant_id"),
            "objective": profile.get("objective"),
            "constraints": profile.get("constraints"),
            "hidden_width": scenario.get("hidden_width"),
            "depth_selection": scenario.get("depth_selection"),
            "rank": rank,
            "kind": kind,
            "mip_execution_identity": profile.get("_execution_identity"),
        }
        architecture = self.architectures.get(architecture_id)
        if architecture is None:
            architecture = ArchitectureCandidate(architecture_id, block_configs, costs)
            self.architectures[architecture_id] = architecture
        else:
            architecture.mip_metrics.update(
                {key: value for key, value in costs.items() if key not in architecture.mip_metrics}
            )
        artifact = {
            "solution_path": str(solution_path or scenario.get("solution_path")),
            "solution_index": rank,
            "kind": kind,
            "hidden_width": scenario.get("hidden_width"),
            "mip_metrics": costs,
            "mip_execution_identity": profile.get("_execution_identity"),
            "solve_identity": scenario.get("solve_identity"),
        }
        checkpoint = result.get("checkpoint")
        artifact_kind = ArtifactKind.CHECKPOINT if checkpoint else ArtifactKind.CONFIG
        if checkpoint:
            artifact["checkpoint"] = str(checkpoint)
        revision = self.add_revision(
            architecture_id=architecture_id,
            artifact_kind=artifact_kind,
            artifact=artifact,
            parent_revision_id=None,
            producer_node="mip",
        )
        origin["revision_id"] = revision.revision_id
        if origin not in architecture.origins:
            architecture.origins.append(origin)
        if not architecture.origin_revision_id or artifact_kind is ArtifactKind.CHECKPOINT:
            architecture.origin_revision_id = revision.revision_id

    @staticmethod
    def _solution_rows(scenario: Mapping[str, Any]) -> list[dict[str, Any]]:
        path = scenario.get("solution_path")
        if not path or not Path(path).is_file():
            return []
        payload = json.loads(Path(path).read_text())
        return list(payload) if isinstance(payload, list) else []

    @staticmethod
    def _atomic_json(path: Path, payload: Any) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(canonicalize(payload), indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
        return path

    @staticmethod
    def _immutable_json(path: Path, payload: Any) -> Path:
        canonical_payload = canonicalize(payload)
        if path.is_file():
            if json.loads(path.read_text()) != canonical_payload:
                raise RuntimeError(f"immutable post-MIP execution artifact changed: {path}")
            return path
        return CandidateLedger._atomic_json(path, canonical_payload)
