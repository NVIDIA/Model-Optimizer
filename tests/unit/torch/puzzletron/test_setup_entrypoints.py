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

"""Compatibility checks for setup entry points."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from examples.puzzletron import puzzletron as public_cli
from puzzletron_setup.v2 import cli as setup_cli

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]


def test_public_cli_uses_the_existing_setup_wizard(monkeypatch):
    calls = []
    monkeypatch.setattr(setup_cli, "main", lambda argv: calls.append(argv) or 12)

    assert public_cli.main(["setup", "--resume", "campaign"]) == 12
    assert calls == [["--resume", "campaign"]]


def test_public_help_advertises_one_setup_command(capsys):
    with pytest.raises(SystemExit) as error:
        public_cli.main(["--help"])
    assert error.value.code == 0

    output = capsys.readouterr().out
    assert "setup" in output
    assert "setup custom" not in output
    assert "init" not in output
    assert "site init" not in output


def test_existing_setup_scripts_keep_their_wizard_help():
    for script_name, expected_option in (
        ("puzzletron_setup.py", "--detailed"),
        ("puzzletron_setup_v2.py", "--campaign-dir"),
    ):
        completed = subprocess.run(
            [sys.executable, str(REPOSITORY_ROOT / "examples/puzzletron" / script_name), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )

        assert completed.returncode == 0, completed.stderr
        assert expected_option in completed.stdout
