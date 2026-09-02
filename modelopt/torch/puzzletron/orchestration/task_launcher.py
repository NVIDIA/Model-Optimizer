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

"""Dependency-light launcher for one task in an orchestration attempt."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .schema import TaskLauncher

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = [
    "TASK_IDENTITY_ENV_KEYS",
    "TaskBinding",
    "build_task_command",
    "main",
    "rendezvous_endpoint",
    "rendezvous_port",
    "resolve_task_binding",
]

TORCH_DISTRIBUTED_ENV_KEYS = frozenset(
    {
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "RANK",
        "WORLD_SIZE",
    }
)

TASK_IDENTITY_ENV_KEYS = TORCH_DISTRIBUTED_ENV_KEYS | frozenset(
    {
        "CUDA_VISIBLE_DEVICES",
        "SLURM_LOCALID",
        "SLURM_NTASKS",
        "SLURM_PROCID",
        "PUZZLETRON_GROUP_INDEX",
        "PUZZLETRON_GROUP_RANK",
        "PUZZLETRON_GROUP_SIZE",
        "PUZZLETRON_LOCAL_TASK_INDEX",
        "PUZZLETRON_MASTER_ADDR",
        "PUZZLETRON_MASTER_PORT",
        "PUZZLETRON_RENDEZVOUS_ENDPOINT",
        "PUZZLETRON_RENDEZVOUS_ID",
        "PUZZLETRON_TASK_HOSTS",
        "PUZZLETRON_TASK_INDEX",
        "PUZZLETRON_TASK_LAUNCHER",
    }
)


@dataclass(frozen=True)
class TaskBinding:
    """Placement and distributed-group identity for one scheduler task."""

    task_index: int
    local_task_index: int
    hostname: str
    group_index: int
    group_rank: int
    group_size: int
    master_addr: str
    master_port: int
    rendezvous_id: str


def rendezvous_port(attempt_id: str, group_index: int, group_count: int) -> int:
    """Return a stable, group-specific port in the unprivileged range."""

    port_start = 20000
    port_span = 30000
    if not 0 <= group_index < group_count:
        raise ValueError(
            f"group_index={group_index} must be between 0 and group_count={group_count}"
        )
    if group_count > port_span:
        raise ValueError(f"group_count={group_count} exceeds rendezvous port span={port_span}")
    seed = int(hashlib.sha256(attempt_id.encode()).hexdigest()[:8], 16)
    return port_start + (seed % (port_span - group_count + 1)) + group_index


def rendezvous_endpoint(binding: TaskBinding) -> str:
    """Return a collision-free local endpoint or the shared multi-node endpoint."""

    if binding.group_size == 1:
        return "localhost:0"
    return f"{binding.master_addr}:{binding.master_port}"


def resolve_task_binding(
    *,
    attempt_id: str,
    hosts: tuple[str, ...],
    task_index: int,
    local_task_index: int,
    tasks_per_node: int,
    tasks_per_group: int,
    group_count: int,
) -> TaskBinding:
    """Resolve deterministic block placement and group identity for one task."""

    task_count = tasks_per_group * group_count
    if not 0 <= task_index < task_count:
        raise ValueError(f"task_index={task_index} is outside task_count={task_count}")
    if tasks_per_node < 1:
        raise ValueError(f"tasks_per_node must be positive, got {tasks_per_node}")
    group_index = task_index // tasks_per_group
    group_rank = task_index % tasks_per_group
    task_host_index = task_index // tasks_per_node
    master_task_index = group_index * tasks_per_group
    master_host_index = master_task_index // tasks_per_node
    if not 0 <= task_host_index < len(hosts):
        raise ValueError(
            f"task host index {task_host_index} is outside host list of size {len(hosts)}"
        )
    if not 0 <= master_host_index < len(hosts):
        raise ValueError(
            f"master host index {master_host_index} is outside host list of size {len(hosts)}"
        )
    return TaskBinding(
        task_index=task_index,
        local_task_index=local_task_index,
        hostname=hosts[task_host_index],
        group_index=group_index,
        group_rank=group_rank,
        group_size=tasks_per_group,
        master_addr=hosts[master_host_index],
        master_port=rendezvous_port(attempt_id, group_index, group_count),
        rendezvous_id=f"{attempt_id}-group-{group_index}",
    )


def build_task_command(
    *,
    payload: Sequence[str],
    launcher: TaskLauncher,
    binding: TaskBinding,
    gpus_per_task: int,
) -> tuple[str, ...]:
    """Wrap an application payload in torchrun when the topology requests it."""

    command = tuple(str(part) for part in payload)
    if launcher is TaskLauncher.DIRECT:
        return command
    endpoint = rendezvous_endpoint(binding)
    return (
        "python",
        "-m",
        "torch.distributed.run",
        f"--nnodes={binding.group_size}",
        f"--nproc-per-node={gpus_per_task}",
        "--rdzv-backend=c10d",
        f"--rdzv-endpoint={endpoint}",
        f"--rdzv-id={binding.rendezvous_id}",
        "--no-python",
        *command,
    )


def _required_index(env: Mapping[str, str], primary: str, fallback: str) -> int:
    value = env.get(primary, env.get(fallback))
    if value is None:
        raise RuntimeError(f"missing task identity: set {primary} or {fallback}")
    return int(value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--gpus-per-node", type=int, required=True)
    parser.add_argument("--task-count", type=int, required=True)
    parser.add_argument("--gpus-per-task", type=int, required=True)
    parser.add_argument("--tasks-per-group", type=int, required=True)
    parser.add_argument(
        "--launcher", choices=tuple(item.value for item in TaskLauncher), required=True
    )
    parser.add_argument("payload", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Resolve this task's binding and replace the launcher with its payload."""

    args = _parser().parse_args(argv)
    payload = tuple(args.payload[1:] if args.payload[:1] == ["--"] else args.payload)
    if not payload:
        raise RuntimeError("task launcher requires a payload after --")
    if args.gpus_per_task < 0 or args.gpus_per_node < args.gpus_per_task:
        raise RuntimeError(f"invalid GPU layout K={args.gpus_per_node}, k={args.gpus_per_task}")
    if args.gpus_per_task == 0 and (
        args.gpus_per_node != 0 or args.launcher != TaskLauncher.DIRECT.value
    ):
        raise RuntimeError("CPU tasks require K=0, k=0, and the direct launcher")
    if args.task_count < 1 or args.tasks_per_group < 1 or args.task_count % args.tasks_per_group:
        raise RuntimeError(
            f"task_count={args.task_count} must be divisible by "
            f"tasks_per_group={args.tasks_per_group}"
        )
    env = os.environ.copy()
    task_index = _required_index(env, "PUZZLETRON_TASK_INDEX", "SLURM_PROCID")
    local_task_index = _required_index(env, "PUZZLETRON_LOCAL_TASK_INDEX", "SLURM_LOCALID")
    visible_gpus = tuple(gpu for gpu in env.get("CUDA_VISIBLE_DEVICES", "").split(",") if gpu)
    if args.gpus_per_task == 0:
        visible_gpus = ()
        env["CUDA_VISIBLE_DEVICES"] = ""
    elif len(visible_gpus) == args.gpus_per_node and len(visible_gpus) != args.gpus_per_task:
        slice_start = local_task_index * args.gpus_per_task
        slice_end = slice_start + args.gpus_per_task
        visible_gpus = visible_gpus[slice_start:slice_end]
        env["CUDA_VISIBLE_DEVICES"] = ",".join(visible_gpus)
    if len(visible_gpus) != args.gpus_per_task:
        raise RuntimeError(
            f"task {task_index} expected {args.gpus_per_task} visible GPUs, got {visible_gpus}"
        )
    tasks_per_node = (
        args.task_count if args.gpus_per_task == 0 else args.gpus_per_node // args.gpus_per_task
    )
    expected_hosts = math.ceil(args.task_count / tasks_per_node)
    if expected_hosts > args.nodes:
        raise RuntimeError(
            f"task layout needs {expected_hosts} hosts but allocation contains {args.nodes}"
        )
    hosts = tuple(host for host in env.get("PUZZLETRON_TASK_HOSTS", "").split(",") if host)
    if len(hosts) != expected_hosts:
        raise RuntimeError(f"expected {expected_hosts} task hosts, got {hosts}")
    group_count = args.task_count // args.tasks_per_group
    binding = resolve_task_binding(
        attempt_id=args.attempt_id,
        hosts=hosts,
        task_index=task_index,
        local_task_index=local_task_index,
        tasks_per_node=tasks_per_node,
        tasks_per_group=args.tasks_per_group,
        group_count=group_count,
    )
    env.update(
        PUZZLETRON_TASK_INDEX=str(binding.task_index),
        PUZZLETRON_TASK_LAUNCHER=str(args.launcher),
        PUZZLETRON_LOCAL_TASK_INDEX=str(binding.local_task_index),
        PUZZLETRON_GROUP_INDEX=str(binding.group_index),
        PUZZLETRON_GROUP_RANK=str(binding.group_rank),
        PUZZLETRON_GROUP_SIZE=str(binding.group_size),
        PUZZLETRON_MASTER_ADDR=binding.master_addr,
        PUZZLETRON_MASTER_PORT=str(binding.master_port),
        PUZZLETRON_RENDEZVOUS_ENDPOINT=rendezvous_endpoint(binding),
        PUZZLETRON_RENDEZVOUS_ID=binding.rendezvous_id,
    )
    for key in TORCH_DISTRIBUTED_ENV_KEYS:
        env.pop(key, None)
    if args.launcher == TaskLauncher.DIRECT.value and binding.group_size == 1:
        env.update(
            LOCAL_RANK="0",
            LOCAL_WORLD_SIZE="1",
            MASTER_ADDR=binding.master_addr,
            MASTER_PORT=str(binding.master_port),
            RANK="0",
            WORLD_SIZE="1",
        )
    print(
        "puzzletron binding "
        f"host={binding.hostname} task={binding.task_index} "
        f"local={binding.local_task_index} gpus={','.join(visible_gpus)} "
        f"group={binding.group_index} rank={binding.group_rank}/{binding.group_size} "
        f"endpoint={rendezvous_endpoint(binding)} "
        f"rdzv_id={binding.rendezvous_id}",
        flush=True,
    )
    command = build_task_command(
        payload=payload,
        launcher=TaskLauncher(args.launcher),
        binding=binding,
        gpus_per_task=args.gpus_per_task,
    )
    # The command is explicit argv from the compiled campaign plan, and no shell
    # is involved. The payload inherits this task's process group so scheduler
    # group signals reach both launcher and payload.
    pid = os.posix_spawnp(command[0], command, env)
    _, status = os.waitpid(pid, 0)
    exit_code = os.waitstatus_to_exitcode(status)
    return exit_code if exit_code >= 0 else 128 - exit_code


if __name__ == "__main__":
    raise SystemExit(main())
