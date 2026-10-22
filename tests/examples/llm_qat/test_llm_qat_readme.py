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

"""Execute named README snippets through the shared documentation runner."""

from pathlib import Path

import pytest
from _test_utils.doc_tests.parser import Scenario
from _test_utils.doc_tests.pytest_utils import execute_scenario, scenario_parameters

REPO = Path(__file__).resolve().parents[3]
README = REPO / "examples/llm_qat/README.md"

SCENARIOS = [
    Scenario(
        "llm-qat-install",
        ["llm-qat-install"],
        timeout_seconds=7200,
        handler="_test_utils.doc_tests.fixtures.llm_qat_readme:llm_qat_install",
        manual=True,
    ),
    Scenario("llm-qat-recipes", ["llm-qat-recipes"], timeout_seconds=30),
    Scenario(
        "llm-qat-quickstart",
        ["llm-qat-quickstart"],
        profile="gpu",
        timeout_seconds=900,
        handler="_test_utils.doc_tests.fixtures.llm_qat_readme:llm_qat_quickstart",
    ),
    Scenario(
        "llm-qad-quickstart",
        ["llm-qad-quickstart"],
        profile="gpu",
        timeout_seconds=900,
        handler="_test_utils.doc_tests.fixtures.llm_qat_readme:llm_qad_quickstart",
    ),
    Scenario(
        "llm-qat-simple",
        ["llm-qat-simple"],
        profile="gpu",
        timeout_seconds=900,
        handler="_test_utils.doc_tests.fixtures.llm_qat_readme:llm_qat_simple",
    ),
    Scenario(
        "llm-qlora-quickstart",
        ["llm-qlora-quickstart"],
        profile="gpu",
        timeout_seconds=900,
        handler="_test_utils.doc_tests.fixtures.llm_qat_readme:llm_qlora_quickstart",
    ),
    Scenario(
        "llm-qlora-serve",
        ["llm-qlora-serve"],
        profile="gpu",
        timeout_seconds=900,
        handler="_test_utils.doc_tests.fixtures.llm_qat_readme:llm_qlora_serve",
        manual=True,
        requires=["vllm"],
        background=True,
    ),
    Scenario(
        "llm-qat-python",
        ["llm-qat-python"],
        profile="gpu",
        timeout_seconds=900,
        handler="_test_utils.doc_tests.fixtures.llm_qat_readme:llm_qat_python",
    ),
    Scenario(
        "llm-qat-trainer",
        ["llm-qat-trainer"],
        profile="gpu",
        timeout_seconds=900,
        handler="_test_utils.doc_tests.fixtures.llm_qat_readme:llm_qat_trainer",
    ),
    Scenario(
        "llm-qad-trainer",
        ["llm-qad-trainer"],
        profile="gpu",
        timeout_seconds=900,
        handler="_test_utils.doc_tests.fixtures.llm_qat_readme:llm_qad_trainer",
    ),
    Scenario(
        "llm-qat-llama-wrap",
        ["llm-qat-llama-wrap"],
        profile="gpu",
        timeout_seconds=900,
        handler="_test_utils.doc_tests.fixtures.llm_qat_readme:llm_qat_llama_wrap",
    ),
    Scenario(
        "llm-qat-copy-config",
        ["llm-qat-copy-config"],
        timeout_seconds=900,
        handler="_test_utils.doc_tests.fixtures.llm_qat_readme:llm_qat_copy_config",
    ),
    Scenario(
        "llm-qat-cli-override",
        ["llm-qat-cli-override"],
        profile="gpu",
        timeout_seconds=900,
        handler="_test_utils.doc_tests.fixtures.llm_qat_readme:llm_qat_cli_override",
    ),
    Scenario(
        "llm-qat-dataset",
        ["llm-qat-dataset"],
        timeout_seconds=900,
        handler="_test_utils.doc_tests.fixtures.llm_qat_readme:llm_qat_dataset",
    ),
    Scenario(
        "llm-qat-evaluation",
        ["llm-qat-evaluation"],
        profile="gpu",
        timeout_seconds=7200,
        handler="_test_utils.doc_tests.fixtures.llm_qat_readme:llm_qat_evaluation",
        manual=True,
        requires=["lm_eval"],
    ),
]

EXCLUDED = {}


@pytest.mark.parametrize("scenario", scenario_parameters(README, SCENARIOS, excluded=EXCLUDED))
def test_readme(scenario, tmp_path, monkeypatch):
    execute_scenario(scenario, REPO, tmp_path, monkeypatch)
