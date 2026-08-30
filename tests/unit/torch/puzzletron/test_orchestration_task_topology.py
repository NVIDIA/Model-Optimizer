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

"""Tests for explicit orchestration task topology."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from puzzletron_orchestrator import task_launcher
from puzzletron_orchestrator.schema import AttemptSpec, CommandSpec, TaskLauncher, TaskTopology
from puzzletron_orchestrator.task_topology import resolve_task_topology


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
    (
        "task_count",
        "local_task_index",
        "gpus_per_task",
        "expected_gpus",
        "wait_status",
        "expected_exit_code",
    ),
    [
        (8, 3, 1, "3", 0, 0),
        (4, 2, 2, "4,5", 7 << 8, 7),
    ],
)
def test_task_launcher_slices_visibility_and_propagates_payload_exit_status(
    monkeypatch,
    task_count: int,
    local_task_index: int,
    gpus_per_task: int,
    expected_gpus: str,
    wait_status: int,
    expected_exit_code: int,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    monkeypatch.setenv("PUZZLETRON_TASK_INDEX", str(local_task_index))
    monkeypatch.setenv("PUZZLETRON_LOCAL_TASK_INDEX", str(local_task_index))
    monkeypatch.setenv("PUZZLETRON_TASK_HOSTS", "node-a")

    def fake_posix_spawnp(executable, command, env) -> int:
        captured.update(executable=executable, command=command, env=env)
        return 123

    monkeypatch.setattr(task_launcher.os, "posix_spawnp", fake_posix_spawnp)
    monkeypatch.setattr(task_launcher.os, "waitpid", lambda pid, _options: (pid, wait_status))

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

    assert result == expected_exit_code
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == expected_gpus
    assert captured["env"]["PUZZLETRON_TASK_LAUNCHER"] == "direct"
    assert captured["env"]["PUZZLETRON_RENDEZVOUS_ENDPOINT"] == "localhost:0"
    assert {
        key: captured["env"][key]
        for key in ("LOCAL_RANK", "LOCAL_WORLD_SIZE", "RANK", "WORLD_SIZE")
    } == {
        "LOCAL_RANK": "0",
        "LOCAL_WORLD_SIZE": "1",
        "RANK": "0",
        "WORLD_SIZE": "1",
    }
    assert captured["env"]["MASTER_ADDR"] == "node-a"
    assert int(captured["env"]["MASTER_PORT"]) > 0


def test_task_launcher_exports_shared_multi_node_rendezvous(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    monkeypatch.setenv("RANK", "99")
    monkeypatch.setenv("WORLD_SIZE", "99")
    monkeypatch.setenv("PUZZLETRON_TASK_INDEX", "1")
    monkeypatch.setenv("PUZZLETRON_LOCAL_TASK_INDEX", "0")
    monkeypatch.setenv("PUZZLETRON_TASK_HOSTS", "node-a,node-b")

    def fake_posix_spawnp(executable, command, env) -> int:
        captured.update(executable=executable, command=command, env=env)
        return 123

    monkeypatch.setattr(task_launcher.os, "posix_spawnp", fake_posix_spawnp)
    monkeypatch.setattr(task_launcher.os, "waitpid", lambda pid, _options: (pid, 0))

    assert (
        task_launcher.main(
            [
                "--attempt-id",
                "attempt-a",
                "--nodes",
                "2",
                "--gpus-per-node",
                "8",
                "--task-count",
                "2",
                "--gpus-per-task",
                "8",
                "--tasks-per-group",
                "2",
                "--launcher",
                "direct",
                "--",
                "python",
                "worker.py",
            ]
        )
        == 0
    )

    env = captured["env"]
    assert isinstance(env, dict)
    assert env["PUZZLETRON_GROUP_SIZE"] == "2"
    assert env["PUZZLETRON_GROUP_RANK"] == "1"
    assert env["PUZZLETRON_RENDEZVOUS_ENDPOINT"].startswith("node-a:")
    assert env["PUZZLETRON_RENDEZVOUS_ID"] == "attempt-a-group-0"
    assert "RANK" not in env
    assert "WORLD_SIZE" not in env


def test_run_worker_consumes_multi_node_task_launcher_identity(tmp_path: Path) -> None:
    script = Path(__file__).parents[4] / "examples/puzzletron/distributed_eval/run_worker.sh"
    env = {
        **os.environ,
        "CAMPAIGN_DIR": str(tmp_path / "campaign"),
        "CONFIG_PATH": str(tmp_path / "experiment.yaml"),
        "TORCHRUN": "/bin/echo",
        "NPROC_PER_NODE": "4",
        "PUZZLETRON_GROUP_SIZE": "2",
        "PUZZLETRON_GROUP_RANK": "1",
        "PUZZLETRON_RENDEZVOUS_ENDPOINT": "node-a:23456",
        "PUZZLETRON_RENDEZVOUS_ID": "attempt-a-group-0",
    }

    result = subprocess.run(
        ["bash", str(script)],
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert "--nnodes 2" in result.stdout
    assert "--node-rank 1" in result.stdout
    assert "--rdzv-endpoint node-a:23456" in result.stdout
    assert "--rdzv-id attempt-a-group-0" in result.stdout


@pytest.mark.parametrize("script_name", ["run_replacement_pool.sh", "run_depth_pool.sh"])
def test_nonzero_group_rank_does_not_own_pool_control_path(
    tmp_path: Path, script_name: str
) -> None:
    true_bin = shutil.which("true")
    false_bin = shutil.which("false")
    assert true_bin is not None
    assert false_bin is not None
    campaign_dir = tmp_path / "campaign"
    campaign_dir.mkdir()
    (campaign_dir / "manifest.json").write_text("{}\n")
    script = Path(__file__).parents[4] / "examples/puzzletron/distributed_eval" / script_name
    env = {
        **os.environ,
        "CAMPAIGN_DIR": str(campaign_dir),
        "CONFIG_PATH": str(tmp_path / "experiment.yaml"),
        "WORLD_SIZE": "2",
        "WORKER_COUNT": "1",
        "NPROC_PER_NODE": "1",
        "TORCHRUN": true_bin,
        "PYTHON_BIN": false_bin,
        "PUZZLETRON_GROUP_INDEX": "0",
        "PUZZLETRON_GROUP_RANK": "1",
        "PUZZLETRON_GROUP_SIZE": "2",
        "PUZZLETRON_RENDEZVOUS_ENDPOINT": "node-a:23456",
        "PUZZLETRON_RENDEZVOUS_ID": "attempt-a-group-0",
    }

    subprocess.run(["bash", str(script)], env=env, check=True, timeout=10)


def test_single_node_torchrun_lets_c10d_choose_a_free_local_port() -> None:
    command = task_launcher.build_task_command(
        payload=("python", "worker.py"),
        launcher=TaskLauncher.TORCHRUN,
        binding=_task_binding(group_size=1),
        gpus_per_task=4,
    )

    assert "--rdzv-endpoint=localhost:0" in command


def test_multi_node_torchrun_uses_master_hostname_for_rendezvous() -> None:
    command = task_launcher.build_task_command(
        payload=("python", "worker.py"),
        launcher=TaskLauncher.TORCHRUN,
        binding=_task_binding(group_size=2),
        gpus_per_task=4,
    )

    assert "--rdzv-endpoint=node-a:23456" in command


def test_direct_launcher_does_not_wrap_payload() -> None:
    command = task_launcher.build_task_command(
        payload=("python", "worker.py"),
        launcher=TaskLauncher.DIRECT,
        binding=_task_binding(group_size=1),
        gpus_per_task=4,
    )

    assert command == ("python", "worker.py")


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


def _task_binding(*, group_size: int) -> task_launcher.TaskBinding:
    return task_launcher.TaskBinding(
        task_index=0,
        local_task_index=0,
        hostname="node-a",
        group_index=0,
        group_rank=0,
        group_size=group_size,
        master_addr="node-a",
        master_port=23456,
        rendezvous_id="attempt-a-group-0",
    )
