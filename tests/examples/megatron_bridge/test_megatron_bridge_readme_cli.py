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

"""Check README Python flag names without executing example scenarios."""

from pathlib import Path

import pytest
from _test_utils.doc_tests.cli import check_cli_options, python_commands

REPO = Path(__file__).resolve().parents[3]
README = REPO / "examples/megatron_bridge/README.md"


@pytest.mark.timeout(90)
@pytest.mark.parametrize(
    "command",
    [
        pytest.param(
            command,
            id=f"{command.snippet}:{command.line}",
            marks=[pytest.mark.manual] if command.snippet == "bridge-generation" else [],
        )
        for command in python_commands(README, cwd=README.parent)
    ],
)
def test_readme_cli_options(command):
    check_cli_options(README, command)
