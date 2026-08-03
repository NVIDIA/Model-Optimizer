# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bare-metal SSH executor with atomic multi-host GPU leases."""

from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from ..schema import AttemptSpec, BareMetalHost, JobHandle, JobState, JobStatus, RunnerEnvironment
from ..task_launcher import TASK_IDENTITY_ENV_KEYS
from ..task_topology import ResolvedTaskTopology, resolve_task_topology
from .base import Executor

__all__ = ["BareMetalSSHExecutor", "GpuLeaseManager"]


def _run_command(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(argv), capture_output=True, text=True, check=False)


@dataclass(frozen=True)
class _TaskLease:
    task_index: int
    hostname: str
    gpu_ids: tuple[int, ...]


class GpuLeaseManager:
    """Atomically assign task-sized GPU slices across a bare-metal inventory."""

    def __init__(self, hosts: tuple[BareMetalHost, ...], state_path: Path) -> None:
        self.hosts = {host.hostname: host.gpus for host in hosts}
        self.state_path = state_path
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self._leases: dict[str, tuple[_TaskLease, ...]] = {}
        self._load()

    def _load(self) -> None:
        if not self.state_path.is_file():
            return
        payload = json.loads(self.state_path.read_text())
        for attempt_id, raw_leases in payload.items():
            items = raw_leases if isinstance(raw_leases, list) else [raw_leases]
            self._leases[attempt_id] = tuple(
                _TaskLease(
                    task_index=int(item.get("task_index", index)),
                    hostname=str(item["hostname"]),
                    gpu_ids=tuple(int(gpu) for gpu in item["gpu_ids"]),
                )
                for index, item in enumerate(items)
            )

    def _persist(self) -> None:
        payload = {
            attempt_id: [
                {
                    "task_index": lease.task_index,
                    "hostname": lease.hostname,
                    "gpu_ids": list(lease.gpu_ids),
                }
                for lease in leases
            ]
            for attempt_id, leases in self._leases.items()
        }
        self.state_path.write_text(json.dumps(payload, indent=2))

    def acquire_topology(
        self, attempt_id: str, topology: ResolvedTaskTopology
    ) -> tuple[_TaskLease, ...]:
        """Lease all task GPU slices or leave lease state unchanged."""

        existing = self._leases.get(attempt_id)
        if existing is not None:
            return existing
        used = {hostname: set() for hostname in self.hosts}
        for leases in self._leases.values():
            for lease in leases:
                used.setdefault(lease.hostname, set()).update(lease.gpu_ids)
        candidate: list[_TaskLease] = []
        selected_hosts = 0
        for hostname, inventory_gpus in self.hosts.items():
            usable_gpus = min(inventory_gpus, topology.gpus_per_node)
            free = [gpu for gpu in range(usable_gpus) if gpu not in used[hostname]]
            slices = [
                tuple(free[index : index + topology.gpus_per_task])
                for index in range(0, len(free), topology.gpus_per_task)
                if len(free[index : index + topology.gpus_per_task])
                == topology.gpus_per_task
            ]
            if not slices:
                continue
            selected_hosts += 1
            for gpu_ids in slices[: topology.tasks_per_node]:
                candidate.append(
                    _TaskLease(
                        task_index=len(candidate),
                        hostname=hostname,
                        gpu_ids=gpu_ids,
                    )
                )
                if len(candidate) == topology.task_count:
                    break
            if len(candidate) == topology.task_count or selected_hosts == topology.nodes:
                break
        if len(candidate) != topology.task_count:
            raise RuntimeError(
                f"unable to atomically lease M={topology.task_count} tasks with "
                f"k={topology.gpus_per_task} GPUs across N={topology.nodes} hosts"
            )
        leases = tuple(candidate)
        self._leases[attempt_id] = leases
        self._persist()
        return leases

    def release(self, attempt_id: str) -> None:
        self._leases.pop(attempt_id, None)
        self._persist()


class BareMetalSSHExecutor(Executor):
    """Launch and aggregate explicit task sets over SSH-managed hosts."""

    backend = "baremetal"

    def __init__(self, runner: RunnerEnvironment, *, state_dir: Path | None = None) -> None:
        if runner.baremetal is None:
            raise ValueError("BareMetalSSHExecutor requires runner.baremetal")
        self.runner = runner
        root = state_dir or (
            Path(runner.contract.repository) / "puzzle_runs" / "orchestration" / "baremetal"
        )
        root.mkdir(parents=True, exist_ok=True)
        self.state_dir = root
        self.leases = GpuLeaseManager(runner.baremetal.hosts, root / "gpu_leases.json")

    def preflight(self) -> None:
        repository = self.runner.contract.repository
        venv = self.runner.contract.venv
        for host in self.runner.baremetal.hosts if self.runner.baremetal else ():
            result = _run_command(
                [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    host.hostname,
                    f"hostname && test -d {shlex.quote(repository)} && "
                    f"cd {shlex.quote(repository)} && "
                    f"test -f {shlex.quote(venv)}/bin/activate && nvidia-smi -L",
                ]
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"bare-metal preflight failed for {host.hostname}: {result.stderr.strip()}"
                )

    @staticmethod
    def _task_path(base: str, task_index: int, task_count: int) -> str:
        if task_count == 1:
            return base
        path = Path(base)
        return str(path.with_name(f"{path.stem}.task-{task_index:04d}{path.suffix}"))

    @staticmethod
    def _launcher_argv(
        attempt: AttemptSpec, topology: ResolvedTaskTopology
    ) -> tuple[str, ...]:
        payload = attempt.command.argv
        if attempt.command.shell:
            payload = ("bash", "-lc", " ".join(shlex.quote(part) for part in payload))
        return (
            "python",
            "-m",
            "puzzletron_orchestrator.task_launcher",
            "--attempt-id",
            attempt.attempt_id,
            "--nodes",
            str(topology.nodes),
            "--gpus-per-node",
            str(topology.gpus_per_node),
            "--task-count",
            str(topology.task_count),
            "--gpus-per-task",
            str(topology.gpus_per_task),
            "--tasks-per-group",
            str(topology.tasks_per_group),
            "--launcher",
            topology.launcher.value,
            "--",
            *payload,
        )

    def _kill_records(self, records: Sequence[dict[str, Any]]) -> None:
        for record in records:
            pid_path = shlex.quote(str(record["pid_path"]))
            _run_command(
                [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    str(record["hostname"]),
                    f"if [[ -f {pid_path} ]]; then kill -- -$(cat {pid_path}) || true; fi",
                ]
            )

    def submit(self, attempt: AttemptSpec) -> JobHandle:
        topology = resolve_task_topology(attempt)
        leases = self.leases.acquire_topology(attempt.attempt_id, topology)
        task_hosts = tuple(dict.fromkeys(lease.hostname for lease in leases))
        launcher = " ".join(
            shlex.quote(part) for part in self._launcher_argv(attempt, topology)
        )
        contract = self.runner.contract
        hooks: list[str] = []
        if contract.setup_env:
            hooks.append(f"source {shlex.quote(contract.setup_env)}")
        hooks.extend(str(command) for command in contract.prerun_commands)
        hook_prefix = "; ".join(hooks)
        if hook_prefix:
            hook_prefix += "; "
        postrun = "; ".join(str(command) for command in contract.postrun_commands)
        postrun_trap = f"trap {shlex.quote(postrun)} EXIT; " if postrun else ""
        working_directory = attempt.command.cwd or contract.repository
        base_log_path = attempt.command.log_path or str(
            self.state_dir / f"{attempt.stage_id}_{attempt.attempt_id}.log"
        )
        records: list[dict[str, Any]] = []
        try:
            for lease in leases:
                task_index = lease.task_index
                log_path = self._task_path(base_log_path, task_index, topology.task_count)
                pid_path = str(self.state_dir / f"{attempt.attempt_id}.{task_index}.pid")
                exit_path = str(self.state_dir / f"{attempt.attempt_id}.{task_index}.exit")
                env = {
                    key: str(value)
                    for key, value in attempt.command.env.items()
                    if key not in TASK_IDENTITY_ENV_KEYS
                }
                env.update(
                    CUDA_VISIBLE_DEVICES=",".join(str(gpu) for gpu in lease.gpu_ids),
                    PUZZLETRON_TASK_INDEX=str(task_index),
                    PUZZLETRON_LOCAL_TASK_INDEX=str(task_index % topology.tasks_per_node),
                    PUZZLETRON_TASK_HOSTS=",".join(task_hosts),
                )
                env_exports = " ".join(
                    f"export {key}={shlex.quote(value)};" for key, value in env.items()
                )
                payload = (
                    f"{postrun_trap}set +e; {launcher}; status=$?; "
                    f"printf '%s\\n' \"$status\" > {shlex.quote(exit_path)}; exit \"$status\""
                )
                remote_command = (
                    f"set -Eeuo pipefail; cd {shlex.quote(working_directory)}; "
                    f"{hook_prefix}source {shlex.quote(contract.venv)}/bin/activate; "
                    f"export PYTHONPATH={shlex.quote(contract.repository)}:${{PYTHONPATH:-}}; "
                    f"{env_exports} rm -f {shlex.quote(exit_path)}; "
                    f"nohup setsid bash -lc {shlex.quote(payload)} > "
                    f"{shlex.quote(log_path)} 2>&1 & "
                    f"echo $! > {shlex.quote(pid_path)}"
                )
                result = _run_command(
                    ["ssh", "-o", "BatchMode=yes", lease.hostname, remote_command]
                )
                if result.returncode != 0:
                    raise RuntimeError(result.stderr.strip() or "ssh launch failed")
                records.append(
                    {
                        "task_index": task_index,
                        "hostname": lease.hostname,
                        "gpu_ids": lease.gpu_ids,
                        "pid_path": pid_path,
                        "exit_path": exit_path,
                        "log_path": log_path,
                    }
                )
        except BaseException:
            self._kill_records(records)
            self.leases.release(attempt.attempt_id)
            raise
        return JobHandle(
            backend=self.backend,
            handle_id=f"bare-{attempt.attempt_id}",
            attempt_id=attempt.attempt_id,
            metadata={
                "tasks": tuple(records),
                "log_paths": tuple(record["log_path"] for record in records),
            },
        )

    def poll(self, handles: Sequence[JobHandle]) -> list[JobStatus]:
        return [self.recover(handle) for handle in handles]

    def cancel(self, handles: Sequence[JobHandle]) -> None:
        for handle in handles:
            records = tuple(handle.metadata.get("tasks", ()))
            self._kill_records(records)
            self.leases.release(handle.attempt_id)

    def recover(self, handle: JobHandle) -> JobStatus:
        records = tuple(handle.metadata.get("tasks", ()))
        log_paths = self.fetch_logs(handle)
        if not records:
            return JobStatus(
                handle=handle,
                state=JobState.UNKNOWN,
                reason="missing remote task metadata",
                log_paths=log_paths,
            )
        running = False
        failure_reason = None
        failure_code = None
        for record in records:
            pid_path = shlex.quote(str(record["pid_path"]))
            exit_path = shlex.quote(str(record["exit_path"]))
            probe_command = (
                f"if [[ -f {exit_path} ]]; then printf 'DONE '; cat {exit_path}; "
                f"elif [[ -f {pid_path} ]] && kill -0 \"$(cat {pid_path})\" "
                ">/dev/null 2>&1; then echo RUNNING; else echo LOST; fi"
            )
            probe = _run_command(
                [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    str(record["hostname"]),
                    probe_command,
                ]
            )
            if probe.returncode != 0:
                failure_reason = probe.stderr.strip() or "SSH task probe failed"
                failure_code = 1
                break
            state = probe.stdout.strip()
            if state == "RUNNING":
                running = True
            elif state.startswith("DONE "):
                try:
                    exit_code = int(state.split(maxsplit=1)[1])
                except ValueError:
                    exit_code = 1
                if exit_code:
                    failure_reason = f"task {record['task_index']} exited with {exit_code}"
                    failure_code = exit_code
                    break
            else:
                failure_reason = f"task {record['task_index']} has no live PID or exit record"
                failure_code = 1
                break
        if failure_reason is not None:
            self._kill_records(records)
            self.leases.release(handle.attempt_id)
            return JobStatus(
                handle=handle,
                state=JobState.FAILED,
                exit_code=failure_code,
                reason=failure_reason,
                log_paths=log_paths,
            )
        if running:
            return JobStatus(handle=handle, state=JobState.RUNNING, log_paths=log_paths)
        self.leases.release(handle.attempt_id)
        return JobStatus(
            handle=handle,
            state=JobState.COMPLETED,
            exit_code=0,
            log_paths=log_paths,
        )
