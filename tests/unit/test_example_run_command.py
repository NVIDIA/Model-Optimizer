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
