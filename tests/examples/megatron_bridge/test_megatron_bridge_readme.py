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

"""Execute README fences through the shared documentation runner."""

from pathlib import Path

import pytest
from _test_utils.doc_tests.pytest_utils import execute_scenario, scenario_parameters

REPO = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "scenario", scenario_parameters(REPO / "examples/megatron_bridge/README.md")
)
def test_readme(scenario, tmp_path, monkeypatch):
    execute_scenario(scenario, REPO, tmp_path, monkeypatch)
