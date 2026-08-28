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

"""Tests for Puzzletron's unified text-evaluation entry point."""

import os
import subprocess
import sys
from pathlib import Path

from examples.puzzletron.evaluation.text import cli

REPOSITORY_ROOT = Path(__file__).parents[5]


def test_default_backend_forwards_arguments_and_exit_code(monkeypatch):
    received = None

    def fake_main(arguments):
        nonlocal received
        received = arguments
        return 7

    monkeypatch.setattr(cli, "_load_lmms_main", lambda: fake_main)

    assert cli.main(["--checkpoint", "/checkpoint", "--limit", "2"]) == 7
    assert received == ["--checkpoint", "/checkpoint", "--limit", "2"]


def test_nemo_backend_removes_selector_and_forwards_arguments(monkeypatch):
    received = None

    def fake_main(arguments):
        nonlocal received
        received = arguments

    monkeypatch.setattr(cli, "_load_nemo_main", lambda: fake_main)

    assert cli.main(["--backend", "nemo", "--base-config", "base.yaml"]) == 0
    assert received == ["--base-config", "base.yaml"]


def test_module_help_does_not_import_evaluator_dependencies(tmp_path):
    (tmp_path / "torch.py").write_text("raise RuntimeError('evaluator imported')\n")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(tmp_path), str(REPOSITORY_ROOT), environment.get("PYTHONPATH", ""))
    )

    result = subprocess.run(
        [sys.executable, "-m", "examples.puzzletron.evaluation.text", "--help"],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert "--backend {lmms,nemo}" in result.stdout
