# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validation and derived values for scheduler-neutral task topology."""

from __future__ import annotations

from dataclasses import dataclass

from .schema import AttemptSpec, TaskLauncher

__all__ = ["ResolvedTaskTopology", "resolve_task_topology"]


@dataclass(frozen=True)
class ResolvedTaskTopology:
    """Validated task topology with all allocation-derived values resolved."""

    nodes: int
    gpus_per_node: int
    task_count: int
    gpus_per_task: int
    tasks_per_group: int
    tasks_per_node: int
    task_capacity: int
    group_count: int
    unused_gpus: int
    launcher: TaskLauncher
    placement: str


def resolve_task_topology(attempt: AttemptSpec) -> ResolvedTaskTopology:
    """Validate and resolve the explicit task topology for an attempt."""

    declared = attempt.task_topology
    nodes = int(attempt.allocation_nodes)
    if nodes < 1:
        raise ValueError(f"task topology requires positive nodes; got N={nodes}")
    default_gpus_per_node = attempt.allocation_gpus // nodes
    gpus_per_node = int(attempt.metadata.get("gpus_per_node", default_gpus_per_node))
    if gpus_per_node < 0:
        raise ValueError(
            f"task topology requires non-negative GPUs per node; got K={gpus_per_node}"
        )
    gpus_per_task = int(
        declared.gpus_per_task
        if declared.gpus_per_task is not None
        else min(attempt.allocation_gpus, gpus_per_node)
    )
    if not 0 <= gpus_per_task <= gpus_per_node:
        raise ValueError(
            f"gpus_per_task={gpus_per_task} exceeds gpus_per_node={gpus_per_node}"
        )
    task_count = int(declared.task_count)
    tasks_per_group = int(declared.tasks_per_group)
    if gpus_per_task == 0:
        if gpus_per_node != 0:
            raise ValueError("CPU tasks require gpus_per_node=0")
        if declared.launcher is not TaskLauncher.DIRECT:
            raise ValueError("CPU tasks require the direct launcher")
        tasks_per_node = task_count
    else:
        tasks_per_node = gpus_per_node // gpus_per_task
    task_capacity = nodes * tasks_per_node
    if not 1 <= task_count <= task_capacity:
        raise ValueError(f"task_count={task_count} exceeds capacity={task_capacity}")
    if not 1 <= tasks_per_group <= task_count:
        raise ValueError(
            f"tasks_per_group={tasks_per_group} must be between 1 and task_count={task_count}"
        )
    if task_count % tasks_per_group:
        raise ValueError(
            f"task_count={task_count} must be divisible by tasks_per_group={tasks_per_group}"
        )
    if declared.placement != "block":
        raise ValueError(
            f"unsupported task placement {declared.placement!r}; expected 'block'"
        )
    launcher = TaskLauncher(declared.launcher)
    return ResolvedTaskTopology(
        nodes=nodes,
        gpus_per_node=gpus_per_node,
        task_count=task_count,
        gpus_per_task=gpus_per_task,
        tasks_per_group=tasks_per_group,
        tasks_per_node=tasks_per_node,
        task_capacity=task_capacity,
        group_count=task_count // tasks_per_group,
        unused_gpus=nodes * gpus_per_node - task_count * gpus_per_task,
        launcher=launcher,
        placement=declared.placement,
    )
