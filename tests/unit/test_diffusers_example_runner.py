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
"""Tests for the in-process runner of the diffusers example tests."""

import logging
import os
import sys
import textwrap
import types

import pytest
import torch
from _test_utils.examples import diffusers_example_runner

# Does what the diffusers example scripts do to process-wide state, and records what it saw.
_SCRIPT = textwrap.dedent(
    """
    import logging
    import os
    import sys

    import torch
    from utils import GREETING  # bare sibling import, like quantize.py's `from utils import ...`

    def main():
        torch.nn.RMSNorm = torch.nn.LayerNorm
        torch.nn.modules.normalization.RMSNorm = torch.nn.LayerNorm
        os.environ["_RUNNER_TEST_VAR"] = "set"
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        logger.addHandler(logging.StreamHandler(sys.stdout))
        logging.getLogger("_runner_test_lib").setLevel(logging.ERROR)
        logger.info(f"{GREETING} argv={sys.argv[1:]} cwd={os.path.basename(os.getcwd())}")
        if "--fail" in sys.argv:
            sys.exit(3)

    if __name__ == "__main__":
        main()
    """
)


@pytest.fixture
def example(tmp_path, monkeypatch):
    example_dir = tmp_path / "examples" / "fake_example"
    example_dir.mkdir(parents=True)
    (example_dir / "script.py").write_text(_SCRIPT)
    (example_dir / "utils.py").write_text('GREETING = "hello"\n')
    monkeypatch.setattr(diffusers_example_runner, "MODELOPT_ROOT", tmp_path)
    # Another module already imported as ``utils`` must neither shadow the script's sibling nor
    # be lost afterwards.
    other_utils = types.ModuleType("utils")
    monkeypatch.setitem(sys.modules, "utils", other_utils)
    monkeypatch.setattr(torch.nn, "RMSNorm", torch.nn.RMSNorm)
    monkeypatch.setattr(torch.nn.modules.normalization, "RMSNorm", torch.nn.RMSNorm)
    return other_utils


def test_runs_script_as_main_and_restores_process_state(example):
    rmsnorm = torch.nn.RMSNorm
    main_logger = logging.getLogger("__main__")
    handlers, level = list(main_logger.handlers), main_logger.level
    lib_level = logging.getLogger("_runner_test_lib").level
    cwd, path = os.getcwd(), list(sys.path)

    output = diffusers_example_runner.run_example_in_process(
        ["python", "script.py", "--flag", "1"], "fake_example"
    )

    assert "hello argv=['--flag', '1'] cwd=fake_example" in output
    assert sys.modules["utils"] is example
    assert torch.nn.RMSNorm is rmsnorm
    assert torch.nn.modules.normalization.RMSNorm is rmsnorm
    assert "_RUNNER_TEST_VAR" not in os.environ
    assert main_logger.handlers == handlers
    assert main_logger.level == level
    assert logging.getLogger("_runner_test_lib").level == lib_level
    assert (os.getcwd(), sys.path) == (cwd, path)


def test_nonzero_exit_raises_with_output(example):
    with pytest.raises(RuntimeError, match="exited with code 3") as excinfo:
        diffusers_example_runner.run_example_in_process(
            ["python", "script.py", "--fail"], "fake_example"
        )
    assert "hello" in excinfo.value.captured_output
    assert sys.modules["utils"] is example
