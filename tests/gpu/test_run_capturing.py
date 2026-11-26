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

import contextlib
import ctypes
import os
import signal
import sys
import time

import pytest
from _test_utils.examples import run_command

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Requires Linux process groups")


def test_run_capturing_does_not_block_on_a_survivor_holding_the_pipe(monkeypatch, tmp_path):
    """A killed command whose descendant inherited the output pipe must not hang the caller."""
    # Adopt the orphaned grandchild so the liveness check can reap it; kill(0) alone would
    # still see a zombie in a container whose PID 1 does not reap.
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    assert libc.prctl(37, ctypes.byref(previous), 0, 0, 0) == 0  # PR_GET_CHILD_SUBREAPER
    assert libc.prctl(36, 1, 0, 0, 0) == 0  # PR_SET_CHILD_SUBREAPER
    monkeypatch.setattr(run_command, "_ORPHAN_PIPE_TIMEOUT_S", 1)
    pid_file = tmp_path / "survivor.pid"

    started = time.monotonic()
    try:
        with pytest.warns(UserWarning, match="descendants holding its output pipe"):
            returncode, output = run_command._run_capturing(
                ["bash", "-c", f"sleep 60 & echo $! > {pid_file}; echo out; sleep 0.3; kill -9 $$"],
                tmp_path,
                os.environ.copy(),
            )

        survivor = int(pid_file.read_text())
        assert returncode == -9
        assert "out" in output  # captured despite the survivor
        assert time.monotonic() - started < 10  # the patched grace period is 1s

        for _ in range(50):  # the group kill is asynchronous
            try:
                reaped, _ = os.waitpid(survivor, os.WNOHANG)
            except ChildProcessError:
                reaped = 0  # adopted by another reaper; probe it below instead
            if reaped == survivor:
                break
            try:
                os.kill(survivor, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        else:
            pytest.fail(f"survivor {survivor} outlived _run_capturing")
    finally:
        try:
            if pid_file.exists():
                survivor = int(pid_file.read_text())
                with contextlib.suppress(ProcessLookupError):
                    os.kill(survivor, signal.SIGKILL)
                with contextlib.suppress(ChildProcessError):
                    os.waitpid(survivor, 0)
        finally:
            assert libc.prctl(36, previous.value, 0, 0, 0) == 0


@pytest.mark.parametrize("escaped", ["hold", "close"], ids=["holds-pipe", "closes-pipe"])
@pytest.mark.parametrize("timeout", [None, 0.5], ids=["normal-exit", "timeout"])
@pytest.mark.timeout(15)
def test_run_capturing_escaped_worker(monkeypatch, tmp_path, timeout, escaped):
    # The supervisor normally reaps the worker. Adopt it here if the supervisor fails first.
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    assert libc.prctl(37, ctypes.byref(previous), 0, 0, 0) == 0  # PR_GET_CHILD_SUBREAPER
    assert libc.prctl(36, 1, 0, 0, 0) == 0  # PR_SET_CHILD_SUBREAPER
    monkeypatch.setattr(run_command, "_KILLED_PIPE_TIMEOUT_S", 0.5)
    pid_file = tmp_path / "escaped.pid"
    script = """
import os
import pathlib
import sys
import time
pid = os.fork()
if pid == 0:
    os.setsid()
    if sys.argv[3] == "close":
        os.close(1)
        os.close(2)
    pathlib.Path(sys.argv[1]).write_text(str(os.getpid()))
    time.sleep(60)
    os._exit(0)
while not pathlib.Path(sys.argv[1]).exists():
    time.sleep(0.01)
if sys.argv[2] == "timeout":
    time.sleep(60)
"""
    try:
        with pytest.raises(run_command.ProcessGroupCleanupError):
            run_command._run_capturing(
                [
                    sys.executable,
                    "-c",
                    script,
                    str(pid_file),
                    "timeout" if timeout else "exit",
                    "close" if escaped == "close" else "hold",
                ],
                tmp_path,
                os.environ.copy(),
                cleanup_process_group=True,
                timeout=timeout,
            )
    finally:
        try:
            if pid_file.exists():
                pid = int(pid_file.read_text())
                with contextlib.suppress(ChildProcessError):
                    if os.waitpid(pid, os.WNOHANG) == (0, 0):
                        os.kill(pid, signal.SIGKILL)
                        os.waitpid(pid, 0)
        finally:
            assert libc.prctl(36, previous.value, 0, 0, 0) == 0


def test_run_capturing_missing_executable(tmp_path):
    executable = str(tmp_path / "missing-command")
    with pytest.raises(FileNotFoundError) as error:
        run_command._run_capturing(
            [executable], tmp_path, os.environ.copy(), cleanup_process_group=True
        )
    assert error.value.filename == executable
