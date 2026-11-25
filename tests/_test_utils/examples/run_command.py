# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Utility functions for running example commands reused in multiple example tests."""

import contextlib
import os
import signal
import subprocess
import threading
import time
import warnings
from pathlib import Path
from typing import TextIO

from _test_utils.transient_errors import TransientError, find_transient_marker

MODELOPT_ROOT = Path(__file__).parents[3]


def extend_cmd_parts(cmd_parts: list[str], **kwargs):
    for key, value in kwargs.items():
        if value is not None:
            cmd_parts.extend([f"--{key}", str(value)])
    if kwargs.get("trust_remote_code", False):
        cmd_parts.append("--trust_remote_code")
    return cmd_parts


# Grace period for descendants to flush and close the inherited output pipe after the command exits,
# then a short bound on the post-SIGKILL drain (the fds close as the kernel tears the group down).
_ORPHAN_PIPE_TIMEOUT_S = 30
_KILLED_PIPE_TIMEOUT_S = 5


class ProcessGroupCleanupError(RuntimeError):
    """A deployment's workers could not be confirmed stopped before the next case."""


def _wait_for_process_group_exit(pgid: int) -> None:
    """Wait for this unreaped child's Linux process group to have no live members."""
    if os.name != "posix":
        return  # /proc exists only on POSIX; there is no process group to confirm elsewhere
    deadline = time.monotonic() + _KILLED_PIPE_TIMEOUT_S
    while True:
        live = []
        try:
            for entry in Path("/proc").iterdir():
                if not entry.name.isdecimal():
                    continue
                try:
                    if os.getpgid(int(entry.name)) != pgid:
                        continue
                    fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
                except (FileNotFoundError, ProcessLookupError):
                    continue
                if int(fields[2]) == pgid and fields[0] not in {"Z", "X"}:
                    live.append(int(entry.name))
        except OSError as exc:
            raise ProcessGroupCleanupError(
                f"Cannot inspect deployment process group {pgid}"
            ) from exc
        if not live:
            return
        if time.monotonic() >= deadline:
            raise ProcessGroupCleanupError(
                f"Deployment process group {pgid} still has live workers after SIGKILL: {live}"
            )
        time.sleep(0.02)


def _wait_without_reaping(
    process: subprocess.Popen, timeout: float | None, capture_failed: threading.Event
) -> None:
    # Keep the group leader's PID reserved until its workers have been cleaned up.
    deadline = None if timeout is None else time.monotonic() + timeout
    while not capture_failed.is_set():
        if os.name == "posix":
            if os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOWAIT | os.WNOHANG) is not None:
                return
        elif process.poll() is not None:
            return
        delay = 0.1
        if deadline is not None:
            delay = min(delay, deadline - time.monotonic())
            if delay <= 0:
                raise subprocess.TimeoutExpired(process.args, timeout)
        capture_failed.wait(delay)


def _run_capturing(
    cmd_parts: list[str],
    cwd: Path,
    env: dict[str, str],
    *,
    cleanup_process_group: bool = False,
    timeout: float | None = None,
    log: TextIO | None = None,
) -> tuple[int, str]:
    """Run a command, streaming and capturing combined stdout/stderr to catch transient errors.

    Drained by a reader thread rather than ``subprocess.run``, which waits for EOF: a descendant
    outliving the command keeps the pipe open, hanging the read and discarding every log line.
    When a log is supplied, capture to that file instead of accumulating output in memory.
    """
    chunks: list[str] = []
    process = subprocess.Popen(
        cmd_parts,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8" if log is not None else None,
        errors="replace",
        start_new_session=True,
    )
    # start_new_session makes the child its own group leader, so the group id is its pid (read now:
    # os.getpgid() stops resolving it once wait() reaps the child).
    pgid = process.pid

    def _kill_process_group():
        with contextlib.suppress(ProcessLookupError, PermissionError):
            if os.name == "posix":
                os.killpg(pgid, signal.SIGKILL)
            else:  # no process groups; the command itself is the best we can reach
                process.kill()

    capture_failed = threading.Event()
    capture_error: Exception | None = None

    def _drain():
        nonlocal capture_error
        assert process.stdout is not None
        try:
            with process.stdout as stream:
                for line in stream:
                    print(line, end="", flush=log is not None)
                    if log is None:
                        chunks.append(line)
                    else:
                        log.write(line)
                        log.flush()
        except Exception as exc:
            capture_error = exc
            capture_failed.set()

    reader = threading.Thread(target=_drain, daemon=True)
    reader.start()
    try:
        _wait_without_reaping(process, timeout, capture_failed)
        if cleanup_process_group or capture_failed.is_set():
            _kill_process_group()
            if cleanup_process_group:
                _wait_for_process_group_exit(pgid)
        else:
            reader.join(_ORPHAN_PIPE_TIMEOUT_S)
            if reader.is_alive():
                warnings.warn(
                    f"{cmd_parts[0]} left descendants holding its output pipe open; killing the "
                    "process group. Output may be truncated."
                )
                _kill_process_group()
        returncode = process.wait()
    except ProcessGroupCleanupError:
        with contextlib.suppress(Exception):
            process.wait(timeout=_KILLED_PIPE_TIMEOUT_S)
        raise
    except BaseException:
        # Interrupted (most likely pytest-timeout) while the command runs: its own process group is
        # not swept up with pytest's, so take the descendants down rather than leak them.
        try:
            if process.returncode is None:
                _kill_process_group()
                if cleanup_process_group:
                    _wait_for_process_group_exit(pgid)
        finally:
            with contextlib.suppress(Exception):
                process.wait(timeout=_KILLED_PIPE_TIMEOUT_S)  # reap: WNOWAIT left it waitable
        raise
    finally:
        reader.join(_KILLED_PIPE_TIMEOUT_S)
    if reader.is_alive():
        raise OSError("Timed out draining subprocess output")
    if capture_error is not None:
        raise OSError("Failed to capture subprocess output") from capture_error
    return returncode, "".join(chunks)


# Set by a suite's conftest to run a command without spawning a subprocess (see
# ``_test_utils.examples.megatron_example_runner``). Unset means always use a subprocess.
_in_process_runner = None


def set_in_process_runner(runner) -> None:
    """Register ``runner(cmd_parts, example_path) -> str``; it must not fall back to a subprocess."""
    global _in_process_runner
    _in_process_runner = runner


def _exception_chain_text(exc: BaseException | None) -> str:
    """``Type: message`` of ``exc`` and of each exception it was raised from or while handling."""
    lines, seen = [], set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        lines.append(f"{type(exc).__name__}: {exc}")
        exc = exc.__cause__ or (None if exc.__suppress_context__ else exc.__context__)
    return "\n".join(lines)


def run_example_command(
    cmd_parts: list[str],
    example_path: str,
    setup_free_port: bool = False,
    env: dict[str, str] | None = None,
) -> str:
    """Run an example command, raising ``TransientError`` on a transient failure so the test reruns."""
    print(f"[{example_path}] Running command: {cmd_parts}")
    in_process = _in_process_runner if env is None else None
    if _in_process_runner is not None and env is not None:
        # The in-process runner uses the ambient environment, so a caller-supplied env would be
        # silently dropped. Fall back to a subprocess rather than run with the wrong environment.
        warnings.warn(f"[{example_path}] env= given; running this step as a subprocess")
    env = os.environ.copy() if env is None else env
    cwd = MODELOPT_ROOT / "examples" / example_path

    if in_process is not None:
        try:
            result = in_process(cmd_parts, example_path)
        except Exception as e:
            # Match the messages of the exception chain (a Hub ConnectionError typically surfaces
            # wrapped in a DatasetGenerationError) and what the step printed, for launcher-level
            # failures. Not a formatted traceback: its source lines can contain a marker.
            marker = find_transient_marker(
                f"{_exception_chain_text(e)}\n{getattr(e, 'captured_output', '')}"
            )
            if marker is None:
                raise  # a real failure keeps its own traceback
            raise TransientError(f"[{example_path}] transient error: {marker!r}") from e
        assert result is not None, "an in-process runner must return output, not fall back"
        return result

    if setup_free_port:
        # The port helper imports torch.distributed, which standalone runners do not need.
        from _test_utils.torch.distributed.utils import get_free_port

        # Subprocess steps only: an in-process runner picks its own free port, since env is a
        # copy it never sees.
        env["MASTER_PORT"] = str(get_free_port())
    returncode, output = _run_capturing(cmd_parts, cwd, env)
    if returncode == 0:
        return output
    error = subprocess.CalledProcessError(returncode, cmd_parts, output=output)
    marker = find_transient_marker(output)
    if marker is None:
        raise error
    raise TransientError(f"[{example_path}] transient error: {marker!r}") from error


def run_hf_ptq_command(*, model: str, quant: str | None = None, vlm: bool = False, **kwargs):
    kwargs.update({"model": model, "quant": quant})
    kwargs.setdefault("tasks", "quant")
    kwargs.setdefault("calib", 16)

    cmd_parts = ["scripts/huggingface_example.sh", "--no-verbose"]
    if vlm:
        # VLM PTQ shares the hf_ptq entry point; --vlm runs the multimodal deploy smoke test.
        cmd_parts.append("--vlm")
    cmd_parts = extend_cmd_parts(cmd_parts, **kwargs)
    # Run a single-GPU TRT-LLM deploy's executor in the script's process. By default it spawns an
    # MPI worker that imports TRT-LLM again, adding ~20s to every deploy.
    env = {**os.environ, "TLLM_WORKER_USE_SINGLE_PROCESS": "1"}
    run_example_command(cmd_parts, "hf_ptq", env=env)
