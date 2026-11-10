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
"""Unit tests for ``tools/precommit/insert_license.py``."""

import re
import sys
from pathlib import Path

import pytest

_PRECOMMIT_DIR = Path(__file__).resolve().parents[4] / "tools" / "precommit"
if str(_PRECOMMIT_DIR) not in sys.path:
    sys.path.insert(0, str(_PRECOMMIT_DIR))

import insert_license

NVIDIA = "\n".join(insert_license.license_header())
# Older header of a file with code copied from an MIT source
NVIDIA_OLD_MIT = re.sub(r"\d{4}", "2023-2024", NVIDIA, count=1).replace(
    "Apache-2.0", "Apache-2.0 AND MIT"
)
THIRD_PARTY = (
    "# Adapted from https://github.com/example/project/blob/0123abc/module.py\n"
    "# Copyright (c) 2023 Example Authors\n"
    "#\n"
    "# Permission is hereby granted, free of charge, to any person obtaining a copy\n"
)


@pytest.mark.parametrize(
    ("before", "after", "exit_code"),
    [
        # No header: inserted at the top, below the shebang
        ("#!/bin/bash\nset -e\n", f"#!/bin/bash\n{NVIDIA}\n\nset -e\n", 1),
        # Third-party header without the NVIDIA header: inserted below it
        (f'{THIRD_PARTY}\n"""Doc."""\n', f'{THIRD_PARTY}\n{NVIDIA}\n\n"""Doc."""\n', 1),
        # Past year and a mixed SPDX expression, below the third-party header: accepted
        (f"{THIRD_PARTY}\n{NVIDIA_OLD_MIT}\n\nimport os\n", None, 0),
        # NVIDIA header above the third-party header: flagged, not changed
        (f"{NVIDIA}\n\n{THIRD_PARTY}\nimport os\n", None, 1),
    ],
)
def test_insert_license(tmp_path, before, after, exit_code):
    path = tmp_path / "file"
    path.write_text(before)
    assert insert_license.main([str(path)]) == exit_code
    assert path.read_text() == (after or before)
    if after:
        assert insert_license.main([str(path)]) == 0
