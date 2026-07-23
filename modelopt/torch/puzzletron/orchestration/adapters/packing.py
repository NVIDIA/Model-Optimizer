# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared allocation helpers for packing independent stage instances."""

from __future__ import annotations

import math

from ..schema import StagePlanNode, TaskLauncher, TaskTopology

__all__ = ["packed_allocation"]


def packed_allocation(
    node: StagePlanNode,
    *,
    instances: int | None = None,
    launcher: TaskLauncher = TaskLauncher.DIRECT,
) -> tuple[int, int, TaskTopology]:
    """Return nodes, GPUs, and task topology for independent model instances."""

    instance_count = int(node.instances if instances is None else instances)
    if instance_count < 1:
        raise ValueError(f"packed instance count must be positive, got {instance_count}")
    if node.gpus_per_instance == 0:
        if instance_count != 1:
            raise ValueError(
                f"CPU stage {node.stage_id!r} supports one task; got instances={instance_count}"
            )
        return (
            1,
            0,
            TaskTopology(
                task_count=1,
                gpus_per_task=0,
                tasks_per_group=1,
                launcher=TaskLauncher.DIRECT,
            ),
        )
    tasks_per_instance = max(1, math.ceil(node.gpus_per_instance / node.gpus_per_node))
    if node.gpus_per_instance % tasks_per_instance:
        raise ValueError(
            f"{node.stage_id} cannot divide {node.gpus_per_instance} GPUs across "
            f"{tasks_per_instance} task(s) per instance"
        )
    total_gpus = instance_count * node.gpus_per_instance
    nodes = max(1, math.ceil(total_gpus / node.gpus_per_node))
    return (
        nodes,
        total_gpus,
        TaskTopology(
            task_count=instance_count * tasks_per_instance,
            gpus_per_task=node.gpus_per_instance // tasks_per_instance,
            tasks_per_group=tasks_per_instance,
            launcher=launcher,
        ),
    )
