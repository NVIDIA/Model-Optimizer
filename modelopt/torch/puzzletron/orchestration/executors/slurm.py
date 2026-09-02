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

"""Slurm executor using stdlib subprocess only."""

from __future__ import annotations

import math
import os
import shlex
import time
from pathlib import Path
from typing import Protocol, Sequence

from ..schema import AttemptSpec, JobHandle, JobState, JobStatus, RunnerEnvironment
from ..task_launcher import TASK_IDENTITY_ENV_KEYS
from ..task_topology import resolve_task_topology
from .baremetal import _run_command
from .base import Executor

__all__ = ["SlurmExecutor", "render_hook_lines", "render_sbatch_script"]


class _CapturedStreams(Protocol):
    stdout: str
    stderr: str


def _slurm_job_id(handle: JobHandle) -> str | None:
    """Resolve a Slurm job id from handle metadata or ``slurm-<id>`` handle ids."""

    job_id = handle.metadata.get("job_id")
    if job_id:
        return str(job_id)
    handle_id = str(handle.handle_id)
    if handle_id.startswith("slurm-"):
        return handle_id.removeprefix("slurm-") or None
    return None


_TRANSIENT_SUBMIT_ERRORS = (
    "socket timed out",
    "unable to contact slurm controller",
    "communication connection failure",
    "connection refused",
    "slurm_receive_msg",
    "slurm_send_recv",
)
_CANCEL_POLL_ATTEMPTS = 60
_CANCEL_POLL_SECONDS = 1.0


def _is_transient_submit_error(result: _CapturedStreams) -> bool:
    detail = f"{result.stderr}\n{result.stdout}".lower()
    return any(marker in detail for marker in _TRANSIENT_SUBMIT_ERRORS)


def _find_queued_job(job_name: str) -> str | None:
    """Find a uniquely named job after an ambiguous ``sbatch`` response."""

    argv = ["squeue", "-h", "--name", job_name, "-o", "%A"]
    user = os.environ.get("USER")
    if user:
        argv[2:2] = ["--user", user]
    result = _run_command(argv)
    if result.returncode != 0:
        return None
    job_ids = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    if not job_ids:
        return None
    if len(job_ids) > 1:
        raise RuntimeError(
            f"multiple Slurm jobs found for idempotent submission {job_name!r}: "
            + ", ".join(sorted(job_ids))
        )
    return job_ids.pop()


def render_hook_lines(commands: Sequence[str]) -> str:
    """Render shell commands that must run before or after the stage payload."""

    lines: list[str] = []
    for command in commands:
        command = str(command).strip()
        if not command:
            continue
        if command.startswith(("source ", ".")):
            lines.append(command)
        else:
            lines.append(command)
    return "\n".join(lines)


def render_sbatch_script(
    *,
    attempt: AttemptSpec,
    runner: RunnerEnvironment,
    partition: str | None,
    account: str,
    time_limit: str,
    qos: str | None,
    job_name: str,
) -> str:
    """Render one sbatch script for an attempt."""

    contract = runner.contract
    topology = resolve_task_topology(attempt)
    log_path = attempt.command.log_path or (
        f"puzzle_runs/logs/{attempt.stage_id}_{attempt.attempt_id}.out"
    )
    # ``topology.gpus_per_node`` is node capacity, not necessarily this
    # attempt's request. Partial-node jobs must ask Slurm for only their active
    # GPUs or container runtimes can expose the full allocation to each task.
    step_nodes = math.ceil(topology.task_count / topology.tasks_per_node)
    step_tasks_per_node = math.ceil(topology.task_count / step_nodes)
    step_gpus_per_node = step_tasks_per_node * topology.gpus_per_task
    kill_on_bad_exit = topology.task_count > 1 or bool(attempt.metadata.get("kill_on_bad_exit"))
    env_lines: list[str] = []
    for key, value in attempt.command.env.items():
        if key in TASK_IDENTITY_ENV_KEYS:
            continue
        env_lines.append(f"export {key}={shlex.quote(str(value))}")
    launcher_argv = (
        "python",
        "-m",
        "puzzletron_orchestrator.task_launcher",
        "--attempt-id",
        attempt.attempt_id,
        "--nodes",
        str(step_nodes),
        "--gpus-per-node",
        str(step_gpus_per_node),
        "--task-count",
        str(topology.task_count),
        "--gpus-per-task",
        str(topology.gpus_per_task),
        "--tasks-per-group",
        str(topology.tasks_per_group),
        "--launcher",
        topology.launcher.value,
        "--",
        *attempt.command.argv,
    )
    argv = " ".join(shlex.quote(part) for part in launcher_argv)
    container_image = contract.container or ""
    container_mounts = contract.container_mounts or ""
    setup_env = contract.setup_env or ""
    prerun = list(contract.prerun_commands)
    if setup_env and f"source {setup_env}" not in prerun and setup_env not in prerun:
        prerun.insert(0, f"source {setup_env}")
    prerun_block = render_hook_lines(prerun)
    postrun_block = render_hook_lines(contract.postrun_commands)
    postrun_trap = f"trap {shlex.quote(postrun_block)} EXIT" if postrun_block else ""
    venv = contract.venv
    repository = contract.repository
    working_directory = attempt.command.cwd or repository
    # Build the header as discrete lines. Do not interpolate optional
    # `#SBATCH --qos` fragments into a textwrap.dedent() template: a trailing
    # newline in those fragments collapses common indent to zero and leaves
    # leading spaces before `#!/bin/bash`, which sbatch rejects ("This does not
    # look like a batch script").
    header_lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --account={account}",
        f"#SBATCH --time={time_limit}",
    ]
    if partition:
        header_lines.insert(2, f"#SBATCH --partition={partition}")
    if qos:
        header_lines.append(f"#SBATCH --qos={qos}")
    header_lines.extend(
        [
            "#SBATCH --no-requeue",
            f"#SBATCH --nodes={step_nodes}",
            f"#SBATCH --ntasks={topology.task_count}",
            f"#SBATCH --ntasks-per-node={step_tasks_per_node}",
        ]
    )
    if step_gpus_per_node:
        header_lines.append(f"#SBATCH --gpus-per-node={step_gpus_per_node}")
    header_lines.append(f"#SBATCH --output={log_path}")
    script = "\n".join(header_lines) + "\n"
    prologue_parts = [
        "set -Eeuo pipefail",
        postrun_trap,
        f"ROOT={shlex.quote(repository)}",
        f"cd {shlex.quote(working_directory)}",
        prerun_block,
        f"source {shlex.quote(venv)}/bin/activate",
        'export PYTHONPATH="$ROOT:${PYTHONPATH:-}"',
        "export PYTHONUNBUFFERED=1",
        *env_lines,
    ]
    prologue = "\n".join(part for part in prologue_parts if part) + "\n"
    # Slurm already directs stdout and stderr to ``#SBATCH --output``. Appending
    # through ``tee`` to that same path duplicates every single-node log line.
    inner = prologue + f"{argv}\n"
    script += (
        "mapfile -t PUZZLETRON_ALLOCATED_HOSTS "
        '< <(scontrol show hostnames "$SLURM_JOB_NODELIST")\n'
        "PUZZLETRON_TASK_HOSTS=$(IFS=,; echo "
        f'"${{PUZZLETRON_ALLOCATED_HOSTS[*]:0:{step_nodes}}}")\n'
        "export PUZZLETRON_TASK_HOSTS\n"
    )
    srun_parts = [
        "srun",
        f"--account={shlex.quote(account)}",
        f"--job-name={shlex.quote(job_name)}",
        f"--nodes={step_nodes}",
        f"--ntasks={topology.task_count}",
        f"--ntasks-per-node={step_tasks_per_node}",
        "--distribution=block:block",
        '--nodelist="$PUZZLETRON_TASK_HOSTS"',
    ]
    if topology.gpus_per_task:
        srun_parts.extend(
            (
                f"--gpus-per-task={topology.gpus_per_task}",
                "--gpu-bind=closest",
            )
        )
    if kill_on_bad_exit:
        srun_parts.append("--kill-on-bad-exit=1")
    if container_image:
        srun_parts.extend(
            [
                f"--container-image={shlex.quote(container_image)}",
                f"--container-mounts={shlex.quote(container_mounts)}",
                f"--container-workdir={shlex.quote(repository)}",
                "--mpi=pmix",
            ]
        )
    srun_parts.extend(("/bin/bash", "-lc", shlex.quote(inner)))
    script += " ".join(srun_parts) + "\n"
    return script


class SlurmExecutor(Executor):
    backend = "slurm"

    def __init__(self, runner: RunnerEnvironment, *, scripts_dir: Path | None = None) -> None:
        if runner.slurm is None:
            raise ValueError("SlurmExecutor requires runner.slurm")
        self.runner = runner
        self.scripts_dir = (
            scripts_dir
            or Path(runner.contract.repository) / "puzzle_runs" / "orchestration" / "sbatch"
        )
        self.scripts_dir.mkdir(parents=True, exist_ok=True)

    def submit(self, attempt: AttemptSpec) -> JobHandle:
        slurm = self.runner.slurm
        if slurm is None:
            raise RuntimeError("Slurm executor lost its runner configuration")
        partition = attempt.metadata.get("partition") or slurm.partition_for_nodes(
            attempt.allocation_nodes
        )
        if partition is not None:
            partition = str(partition)
        job_name = f"{slurm.job_name_prefix}-{attempt.stage_id[:18]}-{attempt.attempt_id[:8]}"
        script_path = self.scripts_dir / f"{attempt.stage_id}_{attempt.attempt_id}.sh"
        script_path.write_text(
            render_sbatch_script(
                attempt=attempt,
                runner=self.runner,
                partition=partition,
                account=slurm.account,
                time_limit=slurm.time_limit,
                qos=slurm.qos,
                job_name=job_name,
            )
        )
        script_path.chmod(0o755)
        job_id = None
        errors: list[str] = []
        for submit_index in range(3):
            if submit_index:
                job_id = _find_queued_job(job_name)
                if job_id:
                    break
            result = _run_command(["sbatch", "--parsable", str(script_path)])
            if result.returncode == 0:
                job_id = result.stdout.strip().split(";")[0]
                break
            detail = (result.stderr or result.stdout or "sbatch failed").strip()
            errors.append(detail)
            if not _is_transient_submit_error(result):
                raise RuntimeError(detail)
            # The controller may have accepted the job even though its response
            # timed out. Reconcile by the attempt-unique job name before retrying.
            for _ in range(3):
                job_id = _find_queued_job(job_name)
                if job_id:
                    break
                time.sleep(0.5)
            if job_id:
                break
            time.sleep(submit_index + 1)
        if not job_id:
            raise RuntimeError("; ".join(errors) or "sbatch failed")
        log_path = attempt.command.log_path
        return JobHandle(
            backend=self.backend,
            handle_id=f"slurm-{job_id}",
            attempt_id=attempt.attempt_id,
            metadata={
                "job_id": job_id,
                "script_path": str(script_path),
                "partition": partition,
                "log_paths": (log_path,) if log_path else (),
            },
        )

    def poll(self, handles: Sequence[JobHandle]) -> list[JobStatus]:
        return [self.recover(handle) for handle in handles]

    def cancel(self, handles: Sequence[JobHandle]) -> None:
        job_ids: list[str] = []
        seen: set[str] = set()
        for handle in handles:
            job_id = _slurm_job_id(handle)
            if not job_id or job_id in seen:
                continue
            seen.add(job_id)
            job_ids.append(job_id)
        if not job_ids:
            return

        def _scancel(flags: Sequence[str], ids: Sequence[str]) -> None:
            result = _run_command(["scancel", *flags, *ids])
            if result.returncode == 0:
                return
            detail = (result.stderr or result.stdout or "scancel failed").strip()
            failures: list[str] = [detail]
            for job_id in ids:
                single = _run_command(["scancel", *flags, job_id])
                if single.returncode != 0:
                    failures.append(
                        f"{job_id}: {(single.stderr or single.stdout or 'failed').strip()}"
                    )
            raise RuntimeError("; ".join(failures))

        # Soft-cancel the allocation, then force-kill remaining steps (srun /
        # container children) so Ctrl-C frees GPUs even if soft cancel is ignored.
        _scancel((), job_ids)
        try:
            _scancel(("--full", "-s", "SIGKILL"), job_ids)
        except RuntimeError:
            # Jobs may already be gone after the soft cancel; that is success.
            pass

        remaining = set(job_ids)
        for _ in range(_CANCEL_POLL_ATTEMPTS):
            # Query the queue globally and intersect locally. ``squeue -j`` returns
            # an error once every requested job has disappeared, which would turn
            # successful cancellation into a false failure.
            queue = _run_command(["squeue", "-h", "-o", "%A"])
            if queue.returncode != 0:
                detail = (queue.stderr or queue.stdout or "squeue failed").strip()
                raise RuntimeError(f"unable to verify cancellation: {detail}")
            queued = {line.strip() for line in queue.stdout.splitlines() if line.strip()}
            remaining.intersection_update(queued)
            if not remaining:
                return
            time.sleep(_CANCEL_POLL_SECONDS)
        raise RuntimeError(
            "Slurm jobs remained queued after cancellation: " + ", ".join(sorted(remaining))
        )

    def recover(self, handle: JobHandle) -> JobStatus:
        job_id = str(handle.metadata.get("job_id", ""))
        if not job_id:
            return JobStatus(handle=handle, state=JobState.UNKNOWN, reason="missing job_id")
        queue = _run_command(["squeue", "-h", "-j", job_id, "-o", "%T"])
        if queue.returncode == 0 and queue.stdout.strip():
            state_name = queue.stdout.strip().splitlines()[0].strip().upper()
            if state_name in {"PENDING", "CONFIGURING", "SUSPENDED", "REQUEUED"}:
                return JobStatus(
                    handle=handle, state=JobState.PENDING, log_paths=self.fetch_logs(handle)
                )
            if state_name in {"RUNNING", "COMPLETING"}:
                return JobStatus(
                    handle=handle, state=JobState.RUNNING, log_paths=self.fetch_logs(handle)
                )
        # squeue can briefly miss a live job (controller blips). Fall back to
        # sacct, but keep non-terminal accounting states non-terminal — otherwise
        # RUNNING is misclassified as FAILED and the orchestrator aborts.
        sacct = _run_command(["sacct", "-j", job_id, "-n", "-X", "--format=State,ExitCode", "-P"])
        if sacct.returncode != 0 or not sacct.stdout.strip():
            return JobStatus(handle=handle, state=JobState.UNKNOWN, reason="sacct unavailable")
        line = sacct.stdout.strip().splitlines()[0]
        state_part, _, exit_part = line.partition("|")
        state_name = state_part.strip().upper()
        exit_code = None
        if exit_part:
            try:
                exit_code = int(exit_part.split(":")[0])
            except ValueError:
                exit_code = None
        if state_name.startswith(("PENDING", "CONFIGURING", "SUSPENDED", "REQUEUED")):
            return JobStatus(
                handle=handle, state=JobState.PENDING, log_paths=self.fetch_logs(handle)
            )
        if state_name.startswith(("RUNNING", "COMPLETING")):
            return JobStatus(
                handle=handle, state=JobState.RUNNING, log_paths=self.fetch_logs(handle)
            )
        if state_name.startswith("COMPLETED") and (exit_code in (None, 0)):
            return JobStatus(
                handle=handle,
                state=JobState.COMPLETED,
                exit_code=exit_code or 0,
                log_paths=self.fetch_logs(handle),
            )
        if state_name.startswith("CANCEL"):
            return JobStatus(
                handle=handle,
                state=JobState.CANCELLED,
                exit_code=exit_code,
                reason=state_name,
                log_paths=self.fetch_logs(handle),
            )
        if state_name:
            return JobStatus(
                handle=handle,
                state=JobState.FAILED,
                exit_code=exit_code,
                reason=state_name,
                log_paths=self.fetch_logs(handle),
            )
        return JobStatus(handle=handle, state=JobState.UNKNOWN, reason="unknown slurm state")
