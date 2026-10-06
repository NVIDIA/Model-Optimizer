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

"""Execute explicitly annotated README scenarios as part of the example CI lane."""

import os
from pathlib import Path

import pytest
from _test_utils.doc_tests.parser import parse_markdown
from _test_utils.doc_tests.runner import run_scenario

REPO = Path(__file__).resolve().parents[3]
SCENARIOS = parse_markdown(REPO / "examples/llm_qat/README.md")


@pytest.mark.parametrize(
    "scenario",
    [
        pytest.param(
            scenario, id=scenario.id, marks=pytest.mark.timeout(scenario.timeout_seconds + 30)
        )
        for scenario in SCENARIOS
    ],
)
def test_readme(scenario, tmp_path, monkeypatch):
    if scenario.profile == "gpu":
        # Keep CPU scenario collection independent of the ML runtime.
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA is required for the QAT README scenario")
    for key in list(os.environ):
        if key in {
            "RANK",
            "LOCAL_RANK",
            "WORLD_SIZE",
            "LOCAL_WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
        }:
            monkeypatch.delenv(key)
    print(run_scenario(scenario, REPO, tmp_path))
