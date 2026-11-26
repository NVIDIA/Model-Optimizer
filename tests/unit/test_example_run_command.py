# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Tests for the example-command runner shared by the example tests."""

import contextlib
import os
import signal
import sys
import time

import pytest
from _test_utils.examples import run_command
from _test_utils.transient_errors import TransientError


@pytest.mark.parametrize(
    ("error", "transient"),
    [
        ("HfHubHTTPError: 503 Server Error: Service Unavailable", True),
        ("DistNetworkError: port: 38789, name: EADDRINUSE, message: address already in use", True),
        ("AssertionError: the quantized model diverged", False),
    ],
)
def test_run_example_command_flags_only_transient_errors(monkeypatch, error, transient):
    def failing_step(cmd_parts, example_path):
        raise RuntimeError(error)  # a marker on a source line must not count: ConnectionError

    monkeypatch.setattr(run_command, "_in_process_runner", failing_step)
    with pytest.raises(RuntimeError) as excinfo:
        run_command.run_example_command(["torchrun", "step.py"], "hf_ptq")
    # TransientError is what the rerun marker in tests/conftest.py matches
    assert isinstance(excinfo.value, TransientError) == transient
    original = excinfo.value.__cause__ if transient else excinfo.value
    assert str(original) == error


def test_run_capturing_does_not_block_on_a_survivor_holding_the_pipe(
    skip_on_windows, monkeypatch, tmp_path
):
    """A killed command whose descendant inherited the output pipe must not hang the caller."""
    monkeypatch.setattr(run_command, "_ORPHAN_PIPE_TIMEOUT_S", 1)
    pid_file = tmp_path / "survivor.pid"

    started = time.monotonic()
    with pytest.warns(UserWarning, match="descendants holding its output pipe"):
        returncode, output = run_command._run_capturing(
            ["bash", "-c", f"sleep 60 & echo $! > {pid_file}; echo out; sleep 0.3; kill -9 $$"],
            tmp_path,
            os.environ.copy(),
        )

    survivor = int(pid_file.read_text())
    try:
        assert returncode == -9
        assert "out" in output  # captured despite the survivor
        assert time.monotonic() - started < 10  # the patched grace period is 1s

        for _ in range(50):  # the group kill is asynchronous
            try:
                os.kill(survivor, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        else:
            pytest.fail(f"survivor {survivor} outlived _run_capturing")
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.kill(survivor, signal.SIGKILL)


def _detacher_script(tmp_path):
    """A descendant in its own session escapes the process-group scan but keeps the pipe."""
    script = tmp_path / "detach.py"
    script.write_text(
        "import os, sys, time\n"
        "os.setsid()\n"
        "open(sys.argv[1], 'w').write(str(os.getpid()))\n"
        "time.sleep(60)\n"
    )
    return script


def _kill_detached_survivor(pid_file):
    for _ in range(50):  # the pid lands after the detached child starts
        if pid_file.exists():
            break
        time.sleep(0.1)
    if pid_file.exists():
        with contextlib.suppress(ProcessLookupError, ValueError):
            os.kill(int(pid_file.read_text()), signal.SIGKILL)


def test_run_capturing_flags_a_detached_group_survivor(skip_on_windows, monkeypatch, tmp_path):
    """``cleanup_process_group`` fails loudly when only the pipe proves a survivor exists."""
    monkeypatch.setattr(run_command, "_KILLED_PIPE_TIMEOUT_S", 1)
    pid_file = tmp_path / "survivor.pid"
    script = _detacher_script(tmp_path)

    try:
        with pytest.raises(run_command.ProcessGroupCleanupError, match="Output pipe"):
            run_command._run_capturing(
                ["bash", "-c", f"{sys.executable} {script} {pid_file} & echo out; sleep 0.3"],
                tmp_path,
                os.environ.copy(),
                cleanup_process_group=True,
            )
    finally:
        _kill_detached_survivor(pid_file)


def test_run_capturing_timeout_flags_a_detached_group_survivor(
    skip_on_windows, monkeypatch, tmp_path
):
    """The timeout path reports the same unconfirmed cleanup instead of a bare timeout."""
    monkeypatch.setattr(run_command, "_KILLED_PIPE_TIMEOUT_S", 1)
    pid_file = tmp_path / "survivor.pid"
    script = _detacher_script(tmp_path)

    try:
        with pytest.raises(run_command.ProcessGroupCleanupError, match="Output pipe"):
            run_command._run_capturing(
                ["bash", "-c", f"{sys.executable} {script} {pid_file} & sleep 60"],
                tmp_path,
                os.environ.copy(),
                cleanup_process_group=True,
                timeout=1,
            )
    finally:
        _kill_detached_survivor(pid_file)
