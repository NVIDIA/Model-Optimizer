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
"""Unit tests for ``tools/precommit/check_env_knobs.py``."""

import sys
from pathlib import Path

import pytest

_PRECOMMIT_DIR = Path(__file__).resolve().parents[4] / "tools" / "precommit"
if str(_PRECOMMIT_DIR) not in sys.path:
    sys.path.insert(0, str(_PRECOMMIT_DIR))

import check_env_knobs


@pytest.mark.parametrize(
    ("statement", "is_read"),
    [
        ('os.environ.get("MODELOPT_X")', True),
        ('os.getenv("MODELOPT_X")', True),
        ('os.environ["MODELOPT_X"]', True),
        ('"MODELOPT_X" in os.environ', True),
        ('os.environ["MODELOPT_X"] += ",more"', True),
        ('os.environ["MODELOPT_X"] = "1"', False),
        ('del os.environ["MODELOPT_X"]', False),
        ('os.putenv("MODELOPT_X", "1")', False),
        ('os.unsetenv("MODELOPT_X")', False),
    ],
)
def test_library_file_fails_on_a_read_but_not_on_a_write(tmp_path, monkeypatch, statement, is_read):
    """Setting or deleting a variable passes; a read fails, including the read inside ``+=``."""
    monkeypatch.setattr(check_env_knobs, "_REPO_ROOT", tmp_path)
    path = tmp_path / "modelopt" / "module.py"
    path.parent.mkdir()
    path.write_text(f"import os\n{statement}\n")
    assert bool(check_env_knobs._violations(path, set())) is is_read
