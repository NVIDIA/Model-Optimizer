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

"""Fixtures for launcher unit tests.

Run from the launcher directory:
    cd Model-Optimizer/tools/launcher
    uv pip install pytest
    uv run python3 -m pytest tests/ -v

Or via nox from Model-Optimizer root:
    nox -s "unit-3.12(torch_211, tf_latest)"
"""

import os
import sys

import pytest

# Make the launcher dir importable so test modules can `import core`, `import slurm_config`,
# etc. at module-load time. conftest.py is imported by pytest before any test module, so
# this mutation is in effect before the first test-module import resolves.
_LAUNCHER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _LAUNCHER_DIR not in sys.path:
    sys.path.insert(0, _LAUNCHER_DIR)


@pytest.fixture
def tmp_yaml(tmp_path):
    """Helper to write a YAML file and return its path."""

    def _write(content, name="test.yaml"):
        p = tmp_path / name
        p.write_text(content)
        return str(p)

    return _write
