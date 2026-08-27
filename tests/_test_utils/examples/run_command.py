# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from _test_utils.torch.distributed.utils import get_free_port

MODELOPT_ROOT = Path(__file__).parents[3]

# Substrings that identify a *transient* HuggingFace Hub / dataset access failure
_HF_TRANSIENT_MARKERS = (
    "HfHubHTTPError",
    "Too Many Requests",
    "500 Server Error",
    "502 Bad Gateway",
    "503 Server Error",
    "504 Server Error",
    "Bad Gateway",
    "Service Unavailable",
    "Gateway Time-out",
    "ConnectionError",
    "ReadTimeout",
    "ConnectTimeout",
    "Max retries exceeded",
    "NewConnectionError",
    "Connection reset by peer",
    "Connection aborted",
    "Consistency check failed",  # partial / interrupted HF download
    "couldn't connect to 'https://huggingface.co'",
    "Couldn't reach",  # datasets: "Couldn't reach <repo> on the Hub"
    "Temporary failure in name resolution",  # transient DNS
    "Name or service not known",  # transient DNS
)
_HF_MAX_RETRIES = 1
_HF_RETRY_DELAY_S = 10


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


def _run_capturing(cmd_parts: list[str], cwd: Path, env: dict[str, str]) -> tuple[int, str]:
    """Run a command, streaming and capturing combined stdout/stderr to catch transient HF errors.

    Drained by a reader thread rather than ``subprocess.run``, which waits for EOF: a descendant
    outliving the command keeps the pipe open, hanging the read and discarding every log line.
    """
    chunks: list[str] = []
    process = subprocess.Popen(
        cmd_parts,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
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

    def _drain():
        with contextlib.suppress(ValueError):  # the stream is closed under us on the escape path
            for line in process.stdout:  # type: ignore[union-attr]
                print(line, end="")
                chunks.append(line)

    reader = threading.Thread(target=_drain, daemon=True)
    reader.start()
    try:
        if os.name == "posix":
            # Wait *without* reaping: a reaped pid can be recycled, and pgid == the child's pid, so
            # reaping before the kill below would risk signalling an unrelated group.
            os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOWAIT)
        else:  # no process groups, so nothing to protect the pid for
            process.wait()
    except BaseException:
        # Interrupted (most likely pytest-timeout) while the command runs: its own process group is
        # not swept up with pytest's, so take the descendants down rather than leak them.
        _kill_process_group()
        with contextlib.suppress(Exception):
            process.wait(timeout=_KILLED_PIPE_TIMEOUT_S)  # reap: WNOWAIT left it waitable
        raise
    reader.join(_ORPHAN_PIPE_TIMEOUT_S)

    if reader.is_alive():
        warnings.warn(
            f"{cmd_parts[0]} left descendants holding its output pipe open; killing the process "
            "group. Output may be truncated."
        )
        _kill_process_group()
        reader.join(_KILLED_PIPE_TIMEOUT_S)

    returncode = process.wait()

    with contextlib.suppress(Exception):
        process.stdout.close()  # type: ignore[union-attr]
    return returncode, "".join(chunks)


def run_example_command(
    cmd_parts: list[str],
    example_path: str,
    setup_free_port: bool = False,
    env: dict[str, str] | None = None,
    hf_max_retries: int = _HF_MAX_RETRIES,
    hf_retry_delay_s: int = _HF_RETRY_DELAY_S,
) -> str | None:
    """Run an example command, retrying transient HuggingFace access errors."""
    print(f"[{example_path}] Running command: {cmd_parts}")
    env = env or os.environ.copy()
    cwd = MODELOPT_ROOT / "examples" / example_path

    for attempt in range(hf_max_retries + 1):
        if setup_free_port:
            env["MASTER_PORT"] = str(get_free_port())  # fresh port per attempt
        returncode, output = _run_capturing(cmd_parts, cwd, env)
        if returncode == 0:
            return output
        transient = any(marker in output for marker in _HF_TRANSIENT_MARKERS)
        if not transient or attempt == hf_max_retries:
            raise subprocess.CalledProcessError(returncode, cmd_parts)
        warnings.warn(
            f"[{example_path}] transient HuggingFace access error; retrying in "
            f"{hf_retry_delay_s}s (attempt {attempt + 1}/{hf_max_retries})"
        )
        time.sleep(hf_retry_delay_s)


def run_hf_ptq_command(*, model: str, quant: str | None = None, vlm: bool = False, **kwargs):
    kwargs.update({"model": model, "quant": quant})
    kwargs.setdefault("tasks", "quant")
    kwargs.setdefault("calib", 16)

    cmd_parts = ["scripts/huggingface_example.sh", "--no-verbose"]
    if vlm:
        # VLM PTQ shares the hf_ptq entry point; --vlm runs the multimodal deploy smoke test.
        cmd_parts.append("--vlm")
    cmd_parts = extend_cmd_parts(cmd_parts, **kwargs)
    run_example_command(cmd_parts, "hf_ptq")
