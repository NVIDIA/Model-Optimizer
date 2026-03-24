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

Or via tox from Model-Optimizer root:
    tox -e py312-launcher
"""

import os
import sys

import pytest


@pytest.fixture(autouse=True)
def add_launcher_to_path():
    """Add the launcher directory to sys.path so core.py and slurm_config.py can be imported."""
    launcher_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if launcher_dir not in sys.path:
        sys.path.insert(0, launcher_dir)
    yield
    if launcher_dir in sys.path:
        sys.path.remove(launcher_dir)


@pytest.fixture
def tmp_yaml(tmp_path):
    """Helper to write a YAML file and return its path."""

    def _write(content, name="test.yaml"):
        p = tmp_path / name
        p.write_text(content)
        return str(p)

    return _write
