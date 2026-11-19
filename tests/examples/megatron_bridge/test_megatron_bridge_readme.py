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
README = REPO / "examples/megatron_bridge/README.md"

SCENARIOS = [
    Scenario(
        "bridge-login",
        ["bridge-login"],
        timeout_seconds=120,
        handler="_test_utils.doc_tests.fixtures.megatron_bridge_readme:bridge_login",
        manual=True,
        requires=["huggingface_hub"],
    ),
    Scenario(
        "bridge-import",
        ["bridge-import"],
        profile="gpu",
        timeout_seconds=7200,
        handler="_test_utils.doc_tests.fixtures.megatron_bridge_readme:bridge_import",
        manual=True,
        min_gpus=8,
        requires=["megatron.bridge"],
    ),
    Scenario(
        "bridge-mlflow",
        ["bridge-mlflow"],
        profile="gpu",
        timeout_seconds=7200,
        handler="_test_utils.doc_tests.fixtures.megatron_bridge_readme:bridge_mlflow",
        manual=True,
        min_gpus=2,
        requires=["megatron.bridge", "mlflow"],
    ),
    Scenario(
        "bridge-real-data",
        ["bridge-real-data"],
        profile="gpu",
        timeout_seconds=86400,
        handler="_test_utils.doc_tests.fixtures.megatron_bridge_readme:bridge_real_data",
        manual=True,
        min_gpus=8,
        requires=["megatron.bridge"],
    ),
    Scenario(
        "bridge-qad",
        ["bridge-qad"],
        profile="gpu",
        timeout_seconds=86400,
        handler="_test_utils.doc_tests.fixtures.megatron_bridge_readme:bridge_qad",
        manual=True,
        min_gpus=8,
        requires=["megatron.bridge"],
    ),
    Scenario(
        "bridge-prune-active",
        ["bridge-prune-active"],
        profile="gpu",
        timeout_seconds=7200,
        handler="_test_utils.doc_tests.fixtures.megatron_bridge_readme:bridge_prune_active",
        manual=True,
        min_gpus=2,
        requires=["megatron.bridge"],
    ),
    Scenario(
        "bridge-prune-memory",
        ["bridge-prune-memory"],
        profile="gpu",
        timeout_seconds=7200,
        handler="_test_utils.doc_tests.fixtures.megatron_bridge_readme:bridge_prune_memory",
        manual=True,
        min_gpus=2,
        requires=["megatron.bridge"],
    ),
    Scenario(
        "bridge-prune-vlm",
        ["bridge-prune-vlm"],
        profile="gpu",
        timeout_seconds=7200,
        handler="_test_utils.doc_tests.fixtures.megatron_bridge_readme:bridge_prune_vlm",
        manual=True,
        min_gpus=2,
        requires=["megatron.bridge"],
    ),
    Scenario(
        "bridge-generation",
        ["bridge-generation"],
        profile="gpu",
        timeout_seconds=7200,
        handler="_test_utils.doc_tests.fixtures.megatron_bridge_readme:bridge_generation",
        manual=True,
        min_gpus=1,
        requires=["vllm"],
    ),
]

EXCLUDED = {
    "bridge-quantize": "Executed by test_quantize_export.py::test_quantize_and_export[readme].",
    "bridge-export": "Executed by test_quantize_export.py::test_quantize_and_export[readme].",
    "bridge-distill": "Executed by test_distill.py::test_distill_llm_hf_export.",
    "bridge-distill-export": "Executed by test_distill.py::test_distill_llm_hf_export.",
    "bridge-prune-params": "Executed by test_prune_minitron.py::test_prune_minitron[qwen3].",
    "bridge-prune-manual": "Executed by test_prune_minitron.py::test_prune_minitron[readme-manual].",
    "bridge-distill-help": "Validated by the README CLI preflight.",
    "bridge-prune-minitron-help": "Validated by the README CLI preflight.",
    "bridge-lora-ptq": "Separate LoRA PTQ integration; full-model adapter calibration is outside this pilot.",
    "bridge-docker": "Interactive container provisioning runs outside pytest; CI supplies the container.",
    "bridge-vlm-template": "Incomplete template; concrete training/export snippets are tested.",
    "bridge-inline-export-template": "Incomplete template; concrete training/export snippets are tested.",
    "bridge-single-export-template": "User-selected model/iteration; multi-iteration conversion is tested.",
    "bridge-output-tree": "Illustrative output tree, not executable code.",
}


@pytest.mark.parametrize("scenario", scenario_parameters(README, SCENARIOS, excluded=EXCLUDED))
def test_readme(scenario, tmp_path, monkeypatch):
    execute_scenario(scenario, REPO, tmp_path, monkeypatch)
