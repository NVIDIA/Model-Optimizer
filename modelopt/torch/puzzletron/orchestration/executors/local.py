# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local subprocess executor for tests and CPU stages."""

from __future__ import annotations

import os
import shlex
import signal
import socket
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from ..schema import AttemptSpec, JobHandle, JobState, JobStatus, RunnerEnvironment
from ..task_launcher import TASK_IDENTITY_ENV_KEYS
from ..task_topology import ResolvedTaskTopology, resolve_task_topology
from .base import Executor

__all__ = ["LocalExecutor"]


class LocalExecutor(Executor):
    backend = "local"

    def __init__(self, runner: RunnerEnvironment | None = None) -> None:
        self.runner = runner
        self._processes: dict[str, tuple[subprocess.Popen[str], ...]] = {}

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
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for process in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()

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
        visible = tuple(
            gpu for gpu in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if gpu
        ) or tuple(str(gpu) for gpu in range(topology.gpus_per_node))
        if len(visible) < topology.gpus_per_node:
            raise RuntimeError(
                f"local executor needs K={topology.gpus_per_node} visible GPUs, got {visible}"
            )
        visible = visible[: topology.gpus_per_node]
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
                    process = subprocess.Popen(
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

    def recover(self, handle: JobHandle) -> JobStatus:
        processes = self._processes.get(handle.handle_id)
        if processes is None:
            return JobStatus(handle=handle, state=JobState.UNKNOWN, reason="missing local processes")
        return_codes = tuple(process.poll() for process in processes)
        log_paths = tuple(handle.metadata.get("log_paths", ()))
        first_failure = next(
            (return_code for return_code in return_codes if return_code not in (None, 0)),
            None,
        )
        if first_failure is not None:
            self._terminate_processes(processes)
            self._close_log_files(processes)
            return JobStatus(
                handle=handle,
                state=JobState.FAILED,
                exit_code=first_failure,
                log_paths=log_paths,
            )
        if any(return_code is None for return_code in return_codes):
            return JobStatus(handle=handle, state=JobState.RUNNING, log_paths=log_paths)
        self._close_log_files(processes)
        return JobStatus(
            handle=handle,
            state=JobState.COMPLETED,
            exit_code=0,
            log_paths=log_paths,
        )
