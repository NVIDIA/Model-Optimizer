# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for explicit orchestration task topology."""

import pytest

from puzzletron_orchestrator import task_launcher
from puzzletron_orchestrator.schema import AttemptSpec, CommandSpec, TaskLauncher, TaskTopology
from puzzletron_orchestrator.task_topology import resolve_task_topology


def _attempt(
    *,
    nodes: int = 1,
    total_gpus: int = 8,
    gpus_per_node: int = 8,
    topology: TaskTopology = TaskTopology(),
) -> AttemptSpec:
    return AttemptSpec(
        attempt_id="attempt-a",
        work_id="stage:0",
        stage_id="stage",
        command=CommandSpec(argv=("python", "worker.py")),
        allocation_nodes=nodes,
        allocation_gpus=total_gpus,
        metadata={"gpus_per_node": gpus_per_node},
        task_topology=topology,
    )


@pytest.mark.parametrize(
    ("nodes", "gpus_per_node", "tasks", "gpus_per_task", "group", "capacity"),
    [
        (1, 8, 1, 8, 1, 1),
        (2, 8, 2, 8, 2, 2),
        (1, 8, 8, 1, 1, 8),
        (4, 8, 4, 8, 2, 4),
        (1, 8, 4, 2, 1, 4),
    ],
)
def test_resolve_task_topology_accepts_valid_layouts(
    nodes: int,
    gpus_per_node: int,
    tasks: int,
    gpus_per_task: int,
    group: int,
    capacity: int,
) -> None:
    attempt = _attempt(
        nodes=nodes,
        total_gpus=nodes * gpus_per_node,
        gpus_per_node=gpus_per_node,
        topology=TaskTopology(
            task_count=tasks,
            gpus_per_task=gpus_per_task,
            tasks_per_group=group,
            launcher=TaskLauncher.TORCHRUN if group > 1 else TaskLauncher.DIRECT,
        ),
    )

    resolved = resolve_task_topology(attempt)

    assert resolved.task_capacity == capacity
    assert resolved.group_count == tasks // group
    assert resolved.unused_gpus == nodes * gpus_per_node - tasks * gpus_per_task


@pytest.mark.parametrize(
    ("topology", "message"),
    [
        (TaskTopology(task_count=1, gpus_per_task=9), "gpus_per_task=9 exceeds"),
        (TaskTopology(task_count=5, gpus_per_task=2), "task_count=5 exceeds capacity=4"),
        (TaskTopology(task_count=3, gpus_per_task=1, tasks_per_group=2), "divisible"),
    ],
)
def test_resolve_task_topology_rejects_invalid_layouts(
    topology: TaskTopology, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_task_topology(_attempt(topology=topology))


def test_legacy_topology_defaults_to_one_direct_task_using_allocated_gpu_slice() -> None:
    resolved = resolve_task_topology(_attempt(total_gpus=4, gpus_per_node=8))

    assert resolved.task_count == 1
    assert resolved.gpus_per_task == 4
    assert resolved.tasks_per_group == 1
    assert resolved.launcher is TaskLauncher.DIRECT


def test_resolve_task_topology_accepts_one_cpu_task() -> None:
    resolved = resolve_task_topology(
        _attempt(
            total_gpus=0,
            gpus_per_node=0,
            topology=TaskTopology(task_count=1, gpus_per_task=0),
        )
    )

    assert resolved.gpus_per_node == 0
    assert resolved.gpus_per_task == 0
    assert resolved.task_count == 1
    assert resolved.task_capacity == 1
    assert resolved.unused_gpus == 0


@pytest.mark.parametrize(
    ("task_count", "local_task_index", "gpus_per_task", "expected"),
    [
        (8, 3, 1, "3"),
        (4, 2, 2, "4,5"),
    ],
)
def test_task_launcher_slices_full_node_visibility_for_packed_container_tasks(
    monkeypatch,
    task_count: int,
    local_task_index: int,
    gpus_per_task: int,
    expected: str,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    monkeypatch.setenv("PUZZLETRON_TASK_INDEX", str(local_task_index))
    monkeypatch.setenv("PUZZLETRON_LOCAL_TASK_INDEX", str(local_task_index))
    monkeypatch.setenv("PUZZLETRON_TASK_HOSTS", "node-a")

    def fake_execvpe(executable, command, env) -> None:
        captured.update(executable=executable, command=command, env=env)

    monkeypatch.setattr(task_launcher.os, "execvpe", fake_execvpe)

    result = task_launcher.main(
        [
            "--attempt-id",
            "attempt-a",
            "--nodes",
            "1",
            "--gpus-per-node",
            "8",
            "--task-count",
            str(task_count),
            "--gpus-per-task",
            str(gpus_per_task),
            "--tasks-per-group",
            "1",
            "--launcher",
            "direct",
            "--",
            "python",
            "worker.py",
        ]
    )

    assert result == 0
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == expected
