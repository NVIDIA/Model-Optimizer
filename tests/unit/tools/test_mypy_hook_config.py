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

"""Tests for the Mypy pre-commit hook environment."""

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _mypy_hook():
    config = yaml.safe_load(
        (REPOSITORY_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    )
    return next(
        (repository, hook)
        for repository in config["repos"]
        for hook in repository["hooks"]
        if hook["id"] == "mypy"
    )


def test_mypy_hook_uses_modelopt_environment_and_bounded_imports():
    repository, hook = _mypy_hook()

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


def test_mypy_hook_preserves_imported_type_information():
    if importlib.util.find_spec("mypy") is None:
        if __name__ == "__main__":
            raise RuntimeError("The ModelOpt lint environment must provide Mypy")
        return

    _, hook = _mypy_hook()
    with tempfile.TemporaryDirectory(prefix="modelopt-mypy-hook-test-") as directory:
        root = Path(directory)
        (root / "provider.py").write_text(
            "def takes_int(value: int) -> int:\n    return value\n", encoding="utf-8"
        )
        consumer = root / "consumer.py"
        consumer.write_text(
            'from provider import takes_int\n\ntakes_int("not an int")\n', encoding="utf-8"
        )

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "mypy",
                *hook["args"],
                "--no-color-output",
                "--show-error-codes",
                "--cache-dir",
                str(root / ".mypy_cache"),
                str(consumer),
            ],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )

    assert result.returncode == 1, result.stdout + result.stderr
    assert (
        'Argument 1 to "takes_int" has incompatible type "str"; expected "int"  [arg-type]'
        in result.stdout
    )


if __name__ == "__main__":
    test_mypy_hook_uses_modelopt_environment_and_bounded_imports()
    test_mypy_hook_preserves_imported_type_information()
