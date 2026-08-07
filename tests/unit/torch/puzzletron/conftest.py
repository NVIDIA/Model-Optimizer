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

"""Shared fixtures for Puzzletron CPU unit tests."""

import json
import platform
from pathlib import Path

import pytest


@pytest.fixture
def write_terminal_manifest():
    """Return a dependency-light terminal-manifest writer with a success default."""

    def write(root: Path, stage: str, **extra: object) -> None:
        path = root / "manifests" / f"{stage}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"stage": stage, "status": "success", **extra}) + "\n")

    return write


# `import fcntl` fails on Windows
def pytest_ignore_collect(collection_path, config):
    return platform.system() == "Windows"
