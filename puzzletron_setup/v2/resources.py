# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable parallel profiles, batches, and stage allocation summaries."""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any, Optional

from puzzletron_orchestrator import ParallelMesh, pack_gpu_allocation
from puzzletron_orchestrator.mesh import gpus_per_instance, validate_mesh

from puzzletron_setup import SetupError

__all__ = [
    "AllocationSummary",
    "BatchResolution",
    "ParallelProfile",
    "ResourceProfileRegistry",
    "StageResources",
    "allocation_summary",
    "resolve_batch",
    "validate_parallel_profile",
]


@dataclass(frozen=True)
class ParallelProfile:
    """One named AutoModel parallel configuration."""

    name: str
    tp: int = 1
    cp: int = 1
    pp: int = 1
    dp_shard: int = 1
    dp_replicate: int = 1
    ep: int = 1
    sequence_parallel: bool = False

    @property
    def batch_unit(self) -> int:
        return self.pp * self.dp_shard * self.dp_replicate

    @property
    def gpu_count(self) -> int:
        return gpus_per_instance(self.mesh)

    @property
    def mesh(self) -> ParallelMesh:
        return ParallelMesh(
            tp=self.tp,
            cp=self.cp,
            pp=self.pp,
            ep=self.ep,
            dp_shard=self.dp_shard,
            dp_replicate=self.dp_replicate,
        )

    def to_parallel(self) -> dict[str, Any]:
        return {
            **self.mesh.as_dict(),
            "sequence_parallel": self.sequence_parallel,
        }

    @classmethod
    def from_mapping(cls, name: str, payload: Mapping[str, Any]) -> "ParallelProfile":
        return cls(
            name=name,
            tp=int(payload.get("tp", 1)),
            cp=int(payload.get("cp", 1)),
            pp=int(payload.get("pp", 1)),
            dp_shard=int(payload.get("dp_shard", 1)),
            dp_replicate=int(payload.get("dp_replicate", payload.get("dp", 1))),
            ep=int(payload.get("ep", 1)),
            sequence_parallel=bool(payload.get("sequence_parallel", False)),
        )


@dataclass(frozen=True)
class BatchResolution:
    """Requested and scheduling-compatible effective batch values."""

    requested: int
    effective: int
    unit: int
    adjusted: bool


@dataclass(frozen=True)
class StageResources:
    """Authoring-time execution settings for one stage."""

    stage_id: str
    strategy: str
    instances: int
    profile: Optional[ParallelProfile] = None
    resource: str = "gpu"
    partition: Optional[str] = None
    gpus_per_node: Optional[int] = None
    batches: Mapping[str, BatchResolution] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "batches", dict(self.batches or {}))
        if self.strategy not in {"single", "sharded", "persistent_pool"}:
            raise SetupError(f"Unsupported execution strategy: {self.strategy}")
        if self.instances < 1:
            raise SetupError("Stage instances/workers must be positive.")
        if self.resource not in {"cpu", "gpu"}:
            raise SetupError("Stage resource must be cpu or gpu.")
        if self.resource == "cpu" and self.instances != 1:
            raise SetupError("CPU stages support one controller task.")

    def to_execution(self, default_gpus_per_node: int) -> dict[str, Any]:
        rendered: dict[str, Any] = {
            "strategy": self.strategy,
            "instances": self.instances,
            "resource": self.resource,
            "gpus_per_node": int(self.gpus_per_node or default_gpus_per_node),
        }
        if self.partition:
            rendered["partition"] = self.partition
        if self.profile is not None and self.resource == "gpu":
            rendered["parallel"] = self.profile.to_parallel()
        return rendered


@dataclass(frozen=True)
class AllocationSummary:
    """Derived allocation and task topology shown by the wizard."""

    instances: int
    gpus_per_instance: int
    nodes: int
    total_gpus: int
    task_count: int
    gpus_per_task: int
    tasks_per_group: int
    unused_node_capacity: int
    partition: Optional[str]


def resolve_batch(requested: int, profile: ParallelProfile) -> BatchResolution:
    """Round a requested model batch upward to its scheduling unit."""

    requested = max(1, int(requested))
    unit = profile.batch_unit
    effective = max(unit, ((requested + unit - 1) // unit) * unit)
    return BatchResolution(requested, effective, unit, effective != requested)


def validate_parallel_profile(
    profile: ParallelProfile,
    inventory: Optional[Any] = None,
) -> None:
    """Validate generic mesh rules plus inspected model capabilities."""

    try:
        validate_mesh(profile.mesh)
    except ValueError as error:
        raise SetupError(str(error)) from error
    facts = getattr(inventory, "facts", {}) if inventory is not None else {}
    experts = facts.get("num_experts") if isinstance(facts, Mapping) else None
    if experts is not None and int(experts) % profile.ep:
        raise SetupError(
            f"EP={profile.ep} does not divide the model's {int(experts)} experts."
        )
    if profile.sequence_parallel and profile.tp == 1:
        raise SetupError("Sequence parallelism requires tensor parallelism greater than one.")


def allocation_summary(
    resources: StageResources,
    *,
    gpus_per_node: int,
) -> AllocationSummary:
    """Derive nodes and scheduler tasks from instances and the model mesh."""

    if resources.resource == "cpu":
        return AllocationSummary(
            instances=1,
            gpus_per_instance=0,
            nodes=1,
            total_gpus=0,
            task_count=1,
            gpus_per_task=0,
            tasks_per_group=1,
            unused_node_capacity=0,
            partition=resources.partition,
        )
    profile = resources.profile or ParallelProfile(name=resources.stage_id)
    validate_parallel_profile(profile)
    node_gpus = int(resources.gpus_per_node or gpus_per_node)
    allocation = pack_gpu_allocation(
        mesh=profile.mesh,
        instances=resources.instances,
        gpus_per_node=node_gpus,
    )
    tasks_per_instance = max(1, math.ceil(allocation.gpus_per_instance / node_gpus))
    if allocation.gpus_per_instance % tasks_per_instance:
        raise SetupError(
            f"{resources.stage_id} cannot divide {allocation.gpus_per_instance} GPUs "
            f"across {tasks_per_instance} task(s) per model instance."
        )
    task_count = resources.instances * tasks_per_instance
    return AllocationSummary(
        instances=resources.instances,
        gpus_per_instance=allocation.gpus_per_instance,
        nodes=allocation.nodes,
        total_gpus=allocation.total_gpus,
        task_count=task_count,
        gpus_per_task=allocation.gpus_per_instance // tasks_per_instance,
        tasks_per_group=tasks_per_instance,
        unused_node_capacity=allocation.nodes * node_gpus - allocation.total_gpus,
        partition=resources.partition,
    )


class ResourceProfileRegistry:
    """Own named profiles and track every stage that reuses one."""

    def __init__(
        self,
        profiles: Optional[Mapping[str, ParallelProfile]] = None,
    ) -> None:
        self._profiles: OrderedDict[str, ParallelProfile] = OrderedDict(profiles or {})
        self._consumers: dict[str, set[str]] = {}

    def names(self) -> tuple[str, ...]:
        return tuple(self._profiles)

    def get(self, name: str) -> ParallelProfile:
        return self._profiles[name]

    def create(
        self,
        profile: ParallelProfile,
        *,
        consumer: Optional[str] = None,
    ) -> ParallelProfile:
        if profile.name in self._profiles:
            raise SetupError(f"Parallel profile already exists: {profile.name}")
        validate_parallel_profile(profile)
        self._profiles[profile.name] = profile
        if consumer:
            self._consumers.setdefault(profile.name, set()).add(consumer)
        return profile

    def reuse(self, name: str, *, consumer: str) -> ParallelProfile:
        profile = self.get(name)
        self._consumers.setdefault(name, set()).add(consumer)
        return profile

    def copy(
        self,
        source_name: str,
        target_name: str,
        *,
        consumer: Optional[str] = None,
        **changes: Any,
    ) -> ParallelProfile:
        copied = replace(self.get(source_name), name=target_name, **changes)
        return self.create(copied, consumer=consumer)

    def consumers(self, name: str) -> tuple[str, ...]:
        return tuple(sorted(self._consumers.get(name, ())))

    def update(self, profile: ParallelProfile) -> tuple[str, ...]:
        if profile.name not in self._profiles:
            raise KeyError(profile.name)
        validate_parallel_profile(profile)
        self._profiles[profile.name] = profile
        return self.consumers(profile.name)

    def to_dict(self) -> dict[str, Any]:
        return {
            name: {
                **asdict(profile),
                "consumers": list(self.consumers(name)),
            }
            for name, profile in self._profiles.items()
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ResourceProfileRegistry":
        registry = cls()
        for name, value in payload.items():
            item = dict(value)
            consumers = tuple(item.pop("consumers", ()))
            item.pop("name", None)
            registry.create(ParallelProfile.from_mapping(name, item))
            for consumer in consumers:
                registry.reuse(name, consumer=str(consumer))
        return registry
