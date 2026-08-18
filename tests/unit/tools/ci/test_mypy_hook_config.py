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

"""Behavioral check for the Mypy pre-commit hook configuration.

Run with ``python tests/unit/tools/ci/test_mypy_hook_config.py``.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]


def main() -> None:
    config = yaml.safe_load(
        (REPOSITORY_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    )
    hook = next(
        hook
        for repository in config["repos"]
        for hook in repository["hooks"]
        if hook["id"] == "mypy"
    )

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

    if result.returncode != 1:
        raise RuntimeError(result.stdout + result.stderr)
    expected = 'Argument 1 to "takes_int" has incompatible type "str"; expected "int"  [arg-type]'
    if expected not in result.stdout:
        raise RuntimeError(result.stdout + result.stderr)


if __name__ == "__main__":
    main()
