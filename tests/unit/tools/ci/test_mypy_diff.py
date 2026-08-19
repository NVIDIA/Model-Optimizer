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

"""Tests for the changed-file mypy quality ratchet."""

from __future__ import annotations

import shlex
import subprocess
from contextlib import nullcontext
from pathlib import Path

import pytest
import yaml

from noxfile import (
    _ChangedPythonFile,
    _new_mypy_diagnostics,
    _parse_changed_python_files,
    _run_mypy,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def test_parse_changed_python_files_tracks_paths_across_statuses():
    output = "\n".join(
        [
            "A\tadded.py",
            "M\tmodified.py",
            "R100\told.py\trenamed.py",
            "C095\tsource.py\tcopied.py",
        ]
    )

    changes = _parse_changed_python_files(output)

    assert changes == (
        _ChangedPythonFile(base_path=None, head_path="added.py"),
        _ChangedPythonFile(base_path="modified.py", head_path="modified.py"),
        _ChangedPythonFile(base_path="old.py", head_path="renamed.py"),
        _ChangedPythonFile(base_path=None, head_path="copied.py"),
    )


def test_parse_changed_python_files_rejects_unsupported_status():
    with pytest.raises(ValueError, match="Unsupported git name-status line"):
        _parse_changed_python_files("D\tdeleted.py")


def test_new_mypy_diagnostics_preserves_inherited_errors_after_line_shift():
    base_output = "old.py:10:5: error: Incompatible return value type  [return-value]"
    head_output = "\n".join(
        [
            "renamed.py:14:5: error: Incompatible return value type  [return-value]",
            "renamed.py:20:1: error: Missing return statement  [return]",
        ]
    )

    diagnostics = _new_mypy_diagnostics(base_output, head_output, {"old.py": "renamed.py"})

    assert [diagnostic.raw for diagnostic in diagnostics] == [
        "renamed.py:20:1: error: Missing return statement  [return]"
    ]


def test_new_mypy_diagnostics_preserves_error_multiplicity():
    base_output = "module.py:10:5: error: Missing return statement  [return]"
    head_output = "\n".join(
        [
            "module.py:12:5: error: Missing return statement  [return]",
            "module.py:20:5: error: Missing return statement  [return]",
        ]
    )

    diagnostics = _new_mypy_diagnostics(base_output, head_output, {"module.py": "module.py"})

    assert [diagnostic.raw for diagnostic in diagnostics] == [
        "module.py:20:5: error: Missing return statement  [return]"
    ]


def test_run_mypy_separates_paths_from_options():
    class RecordingSession:
        def __init__(self):
            self.arguments = ()

        def chdir(self, _checkout):
            return nullcontext()

        def run(self, *arguments, **_kwargs):
            self.arguments = arguments
            return ""

    session = RecordingSession()

    _run_mypy(session, Path("checkout"), ["-strict.py"], Path("cache"))

    assert session.arguments[-2:] == ("--", "-strict.py")
    assert "--follow-imports=silent" in session.arguments
    assert "--follow-imports=skip" not in session.arguments
    assert "--scripts-are-modules" in session.arguments


def test_merge_parent_avoids_target_only_files_selected_by_stale_event_base(tmp_path):
    _git(tmp_path, "init", "--initial-branch=main")
    _git(tmp_path, "config", "user.name", "ModelOpt CI")
    _git(tmp_path, "config", "user.email", "modelopt-ci@example.com")
    _git(tmp_path, "config", "commit.gpgsign", "false")
    _git(tmp_path, "config", "core.hooksPath", ".git/no-hooks")

    (tmp_path / "base.py").write_text("value = 1\n", encoding="utf-8")
    _git(tmp_path, "add", "base.py")
    _git(tmp_path, "commit", "-m", "base")
    stale_base = _git(tmp_path, "rev-parse", "HEAD")

    _git(tmp_path, "switch", "-c", "topic")
    (tmp_path / "README.md").write_text("documentation\n", encoding="utf-8")
    _git(tmp_path, "add", "README.md")
    _git(tmp_path, "commit", "-m", "docs")

    _git(tmp_path, "switch", "main")
    (tmp_path / "target_only.py").write_text("target = True\n", encoding="utf-8")
    _git(tmp_path, "add", "target_only.py")
    _git(tmp_path, "commit", "-m", "advance target")
    _git(tmp_path, "merge", "--no-ff", "topic", "-m", "synthetic pull request merge")
    merge_head = _git(tmp_path, "rev-parse", "HEAD")
    current_base = _git(tmp_path, "rev-parse", f"{merge_head}^1")

    stale_output = _git(
        tmp_path,
        "diff",
        "--name-status",
        "--diff-filter=ACMR",
        f"{stale_base}...{merge_head}",
        "--",
        "*.py",
    )
    exact_output = _git(
        tmp_path,
        "diff",
        "--name-status",
        "--diff-filter=ACMR",
        f"{current_base}...{merge_head}",
        "--",
        "*.py",
    )

    assert _parse_changed_python_files(stale_output) == (
        _ChangedPythonFile(base_path=None, head_path="target_only.py"),
    )
    assert _parse_changed_python_files(exact_output) == ()


def test_code_quality_uses_checked_out_pull_request_merge_parent():
    workflow = yaml.safe_load(
        (REPOSITORY_ROOT / ".github/workflows/code_quality.yml").read_text(encoding="utf-8")
    )
    changed_file_step = next(
        step
        for step in workflow["jobs"]["code-quality"]["steps"]
        if step.get("name") == "Run changed-file code quality checks"
    )
    command = shlex.split(changed_file_step["run"])

    assert command[-6:] == ["nox", "-s", "pre_commit_diff", "--", "HEAD^1", "HEAD"]
    assert "github.event.pull_request.base.sha" not in changed_file_step["run"]


def test_mypy_hook_uses_modelopt_environment():
    config = yaml.safe_load(
        (REPOSITORY_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    )
    repository, hook = next(
        (repository, hook)
        for repository in config["repos"]
        for hook in repository["hooks"]
        if hook["id"] == "mypy"
    )

    assert repository["repo"] == "local"
    assert hook["entry"] == "python -m mypy"
    assert hook["language"] == "system"
    assert hook["types_or"] == ["python", "pyi"]
    assert hook["require_serial"] is True
    assert hook["args"] == [
        "--no-install-types",
        "--interactive",
        "--ignore-missing-imports",
        "--follow-imports=silent",
        "--scripts-are-modules",
    ]
