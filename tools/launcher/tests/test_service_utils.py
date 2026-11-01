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

"""latest_checkpoint in common/service_utils.sh."""

import subprocess
from pathlib import Path

import pytest

_SERVICE_UTILS = Path(__file__).resolve().parents[1] / "common" / "service_utils.sh"


def _latest(root, prefix="exported-checkpoint-"):
    # Sourcing appends to ./modules/Model-Optimizer/modelopt/__init__.py; from the launcher
    # dir that is this checkout, so run in the temp dir.
    out = subprocess.run(
        [
            "bash",
            "-c",
            'source "$0" >/dev/null 2>&1; latest_checkpoint "$1" "$2"',
            _SERVICE_UTILS,
            root,
            prefix,
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


@pytest.mark.parametrize(
    ("names", "expected"),
    [
        # Numeric order, not string order; non-numeric suffixes are ignored.
        (["900", "1000", "tmp"], "1000"),
        # final wins over any step.
        (["900", "1000", "final"], "final"),
        ([], None),
    ],
)
def test_latest_checkpoint(tmp_path, names, expected):
    """Final wins, else the highest step; a dash in a parent dir once broke the parsing."""
    root = tmp_path / "my-drafts" / "run-1"
    root.mkdir(parents=True)
    for name in names:
        (root / f"exported-checkpoint-{name}").mkdir()
    (root / "exported-checkpoint-5000.txt").touch()  # files are not checkpoints

    want = str(root / f"exported-checkpoint-{expected}") if expected else ""
    assert _latest(root) == want
