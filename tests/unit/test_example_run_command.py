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
import time

import pytest
from _test_utils.examples import run_command


@pytest.mark.parametrize(
    ("error", "attempts"),
    [
        ("HfHubHTTPError: 503 Server Error: Service Unavailable", 2),
        ("DistNetworkError: port: 38789, name: EADDRINUSE, message: address already in use", 2),
        ("AssertionError: the quantized model diverged", 1),
    ],
)
def test_run_example_command_retries_only_transient_errors(monkeypatch, error, attempts):
    calls = []

    def step_failing_once(cmd_parts, example_path):
        calls.append(cmd_parts)
        if len(calls) == 1:
            raise RuntimeError(error)
        return "output"

    def run():
        return run_command.run_example_command(
            ["torchrun", "step.py"], "hf_ptq", hf_retry_delay_s=0
        )

    monkeypatch.setattr(run_command, "_in_process_runner", step_failing_once)
    if attempts == 1:
        with pytest.raises(RuntimeError, match="diverged"):
            run()
    else:
        assert run() == "output"
    assert len(calls) == attempts


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
