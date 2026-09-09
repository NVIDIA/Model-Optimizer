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

"""Local subprocess executor for tests and CPU stages."""

from __future__ import annotations

import os
import shlex
import signal
import socket

# Required to supervise concurrent worker process groups; every launch uses an
# argument sequence with ``shell=False`` and never interpolates a shell command.
import subprocess  # nosec B404 - required for process-group supervision; shell=False.
import sys
import time
import uuid
from pathlib import Path
from typing import Sequence

from ..schema import AttemptSpec, JobHandle, JobState, JobStatus, RunnerEnvironment
from ..task_launcher import TASK_IDENTITY_ENV_KEYS
from ..task_topology import ResolvedTaskTopology, resolve_task_topology
from .base import Executor

__all__ = ["LocalExecutor"]


class LocalExecutor(Executor):
    backend = "local"

    def __init__(
        self,
        runner: RunnerEnvironment | None = None,
        *,
        gpu_capacity: int | None = None,
        session_id: str | None = None,
        environment_prepared: bool = False,
    ) -> None:
        self.runner = runner
        self._environment_prepared = environment_prepared
        self._processes: dict[str, tuple[subprocess.Popen[str], ...]] = {}
        self._gpu_capacity = gpu_capacity
        self._session_id = session_id or str(uuid.uuid4())
        self._gpu_leases: dict[str, tuple[str, ...]] = {}
        visible = tuple(gpu for gpu in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if gpu)
        if gpu_capacity is not None:
            if gpu_capacity <= 0:
                raise ValueError("local executor gpu_capacity must be positive")
            if visible and len(visible) < gpu_capacity:
                raise RuntimeError(
                    f"local executor capacity is {gpu_capacity} GPUs but only {visible} are visible"
                )
            self._managed_gpus = visible[:gpu_capacity] or tuple(
                str(gpu) for gpu in range(gpu_capacity)
            )
        else:
            self._managed_gpus = ()

    def can_submit(self, attempt: AttemptSpec) -> bool:
        if self._gpu_capacity is None:
            return True
        topology = resolve_task_topology(attempt)
        if topology.gpus_per_task == 0:
            return True
        required = topology.task_count * topology.gpus_per_task
        leased = {gpu for lease in self._gpu_leases.values() for gpu in lease}
        return len([gpu for gpu in self._managed_gpus if gpu not in leased]) >= required

    def can_submit_all(self, attempts: Sequence[AttemptSpec]) -> bool:
        if self._gpu_capacity is None:
            return True
        required = sum(
            topology.task_count * topology.gpus_per_task
            for topology in (resolve_task_topology(attempt) for attempt in attempts)
        )
        leased = {gpu for lease in self._gpu_leases.values() for gpu in lease}
        return len([gpu for gpu in self._managed_gpus if gpu not in leased]) >= required

    def _acquire_gpus(
        self, attempt: AttemptSpec, topology: ResolvedTaskTopology
    ) -> tuple[str, ...]:
        if self._gpu_capacity is None:
            visible = tuple(
                gpu for gpu in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if gpu
            ) or tuple(str(gpu) for gpu in range(topology.gpus_per_node))
            if len(visible) < topology.gpus_per_node:
                raise RuntimeError(
                    f"local executor needs K={topology.gpus_per_node} visible GPUs, got {visible}"
                )
            return visible[: topology.gpus_per_node]
        if topology.gpus_per_task == 0:
            return ()
        required = topology.task_count * topology.gpus_per_task
        leased = {gpu for lease in self._gpu_leases.values() for gpu in lease}
        available = tuple(gpu for gpu in self._managed_gpus if gpu not in leased)
        if len(available) < required:
            raise RuntimeError(
                f"local executor has no capacity for {required} GPUs; available={available}"
            )
        allocation = available[:required]
        self._gpu_leases[attempt.attempt_id] = allocation
        return allocation

    def _release_gpus(self, attempt_id: str) -> None:
        self._gpu_leases.pop(attempt_id, None)

    def _launcher_argv(
        self, attempt: AttemptSpec, topology: ResolvedTaskTopology
    ) -> tuple[str, ...]:
        payload = attempt.command.argv
        if attempt.command.shell:
            payload = ("bash", "-lc", " ".join(shlex.quote(part) for part in payload))
        return (
            "python" if self.runner is not None else sys.executable,
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

    def _wrapped_argv(self, argv: tuple[str, ...]) -> list[str]:
        if self.runner is None:
            return list(argv)
        contract = self.runner.contract
        if self._environment_prepared:
            if not contract.source_guard:
                return list(argv)
            return [
                "bash",
                "-c",
                "; ".join(
                    (
                        "set -Eeuo pipefail",
                        contract.source_guard,
                        " ".join(shlex.quote(part) for part in argv),
                    )
                ),
            ]
        hooks: list[str] = []
        if contract.setup_env:
            hooks.append(f"source {shlex.quote(contract.setup_env)}")
        hooks.extend(str(command) for command in contract.prerun_commands)
        parts = ["set -Eeuo pipefail", *hooks]
        parts.extend(
            (
                f"source {shlex.quote(contract.venv)}/bin/activate",
                f"export PYTHONPATH={shlex.quote(contract.repository)}:${{PYTHONPATH:-}}",
            )
        )
        if contract.source_guard:
            parts.append(contract.source_guard)
        if contract.postrun_commands:
            postrun = "; ".join(str(command) for command in contract.postrun_commands)
            parts.append(f"trap {shlex.quote(postrun)} EXIT")
        parts.append(" ".join(shlex.quote(part) for part in argv))
        return ["bash", "-lc", "; ".join(parts)]

    @staticmethod
    def _task_log_path(base: str | None, task_index: int, task_count: int) -> str | None:
        if base is None or task_count == 1:
            return base
        path = Path(base)
        return str(path.with_name(f"{path.stem}.task-{task_index:04d}{path.suffix}"))

    @staticmethod
    def _terminate_processes(processes: Sequence[subprocess.Popen[str]]) -> None:
        processes = tuple(processes)
        if all(process.poll() is not None for process in processes):
            return
        process_groups = {process.pid for process in processes}
        for process in processes:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                process_groups.discard(process.pid)

        deadline = time.monotonic() + 30
        while process_groups and time.monotonic() < deadline:
            for process in processes:
                process.poll()
            for process_group in tuple(process_groups):
                try:
                    os.killpg(process_group, 0)
                except ProcessLookupError:
                    process_groups.remove(process_group)
                except PermissionError:
                    # No signalable process remains in the group. This can be
                    # reported for an already-terminated group on macOS.
                    process_groups.remove(process_group)
            if process_groups:
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))

        for process_group in process_groups:
            try:
                os.killpg(process_group, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for process in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    raise RuntimeError(
                        f"local task process {process.pid} survived process-group termination"
                    ) from None

        kill_deadline = time.monotonic() + 5
        while process_groups and time.monotonic() < kill_deadline:
            for process_group in tuple(process_groups):
                try:
                    os.killpg(process_group, 0)
                except ProcessLookupError:
                    process_groups.remove(process_group)
                except PermissionError:
                    process_groups.remove(process_group)
            if process_groups:
                time.sleep(min(0.05, max(0.0, kill_deadline - time.monotonic())))
        if process_groups:
            raise RuntimeError(
                "local task process group(s) survived termination: "
                + ", ".join(str(process_group) for process_group in sorted(process_groups))
            )

    @staticmethod
    def _close_log_files(processes: Sequence[subprocess.Popen[str]]) -> None:
        for process in processes:
            log_file = getattr(process, "_puzzletron_log_file", None)
            if log_file is not None:
                log_file.close()
                process._puzzletron_log_file = None  # type: ignore[attr-defined]

    def submit(self, attempt: AttemptSpec) -> JobHandle:
        topology = resolve_task_topology(attempt)
        if topology.nodes != 1:
            raise ValueError(
                f"local executor requires N=1, got N={topology.nodes} for {attempt.attempt_id}"
            )
        visible = self._acquire_gpus(attempt, topology)
        base_env = os.environ.copy()
        if self.runner is None:
            repository = str(Path(__file__).resolve().parents[5])
            base_env["PYTHONPATH"] = (
                f"{repository}:{base_env['PYTHONPATH']}"
                if base_env.get("PYTHONPATH")
                else repository
            )
        base_env.update(
            (key, str(value))
            for key, value in attempt.command.env.items()
            if key not in TASK_IDENTITY_ENV_KEYS
        )
        host = socket.gethostname()
        launcher_argv = self._launcher_argv(attempt, topology)
        processes: list[subprocess.Popen[str]] = []
        log_paths: list[str] = []
        try:
            for task_index in range(topology.task_count):
                start = task_index * topology.gpus_per_task
                task_gpus = visible[start : start + topology.gpus_per_task]
                env = base_env.copy()
                env.update(
                    CUDA_VISIBLE_DEVICES=",".join(task_gpus),
                    PUZZLETRON_TASK_INDEX=str(task_index),
                    PUZZLETRON_LOCAL_TASK_INDEX=str(task_index),
                    PUZZLETRON_TASK_HOSTS=host,
                )
                log_path = self._task_log_path(
                    attempt.command.log_path, task_index, topology.task_count
                )
                log_file = None
                stdout = None
                if log_path:
                    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
                    log_file = open(log_path, "w", encoding="utf-8")  # noqa: SIM115
                    stdout = log_file
                    log_paths.append(log_path)
                try:
                    # Popen is required so the executor can poll and terminate the
                    # whole worker process group before releasing its GPU lease.
                    process = subprocess.Popen(  # nosec B603
                        self._wrapped_argv(launcher_argv),
                        cwd=attempt.command.cwd,
                        env=env,
                        stdout=stdout,
                        stderr=subprocess.STDOUT,
                        text=True,
                        shell=False,
                        start_new_session=True,
                    )
                except BaseException:
                    if log_file is not None:
                        log_file.close()
                    raise
                if log_file is not None:
                    process._puzzletron_log_file = log_file  # type: ignore[attr-defined]
                processes.append(process)
        except BaseException:
            self._terminate_processes(processes)
            self._close_log_files(processes)
            self._release_gpus(attempt.attempt_id)
            raise
        handle_id = f"local-{attempt.attempt_id}"
        self._processes[handle_id] = tuple(processes)
        return JobHandle(
            backend=self.backend,
            handle_id=handle_id,
            attempt_id=attempt.attempt_id,
            metadata={
                "pids": tuple(process.pid for process in processes),
                "log_paths": tuple(log_paths),
                "local_session_id": self._session_id,
                "allocated_gpus": visible,
            },
        )

    def poll(self, handles: Sequence[JobHandle]) -> list[JobStatus]:
        statuses: list[JobStatus] = []
        for handle in handles:
            statuses.append(self.recover(handle))
        return statuses

    def cancel(self, handles: Sequence[JobHandle]) -> None:
        for handle in handles:
            processes = self._processes.get(handle.handle_id)
            if processes is not None:
                self._terminate_processes(processes)
                self._close_log_files(processes)
            self._release_gpus(handle.attempt_id)

    def recover(self, handle: JobHandle) -> JobStatus:
        processes = self._processes.get(handle.handle_id)
        if processes is None:
            if (
                self._gpu_capacity is not None
                and handle.metadata.get("local_session_id") != self._session_id
            ):
                return JobStatus(
                    handle=handle,
                    state=JobState.CANCELLED,
                    reason="prior reusable allocation ended",
                    log_paths=tuple(handle.metadata.get("log_paths", ())),
                )
            return JobStatus(
                handle=handle, state=JobState.UNKNOWN, reason="missing local processes"
            )
        return_codes = tuple(process.poll() for process in processes)
        log_paths = tuple(handle.metadata.get("log_paths", ()))
        first_failure = next(
            (return_code for return_code in return_codes if return_code not in (None, 0)),
            None,
        )
        if first_failure is not None:
            self._terminate_processes(processes)
            self._close_log_files(processes)
            self._release_gpus(handle.attempt_id)
            return JobStatus(
                handle=handle,
                state=JobState.FAILED,
                exit_code=first_failure,
                log_paths=log_paths,
            )
        if any(return_code is None for return_code in return_codes):
            return JobStatus(handle=handle, state=JobState.RUNNING, log_paths=log_paths)
        self._close_log_files(processes)
        self._release_gpus(handle.attempt_id)
        return JobStatus(
            handle=handle,
            state=JobState.COMPLETED,
            exit_code=0,
            log_paths=log_paths,
        )
