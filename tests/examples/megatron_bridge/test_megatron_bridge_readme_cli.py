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
from _test_utils.doc_tests.parser import resolve_scenarios

REPO = Path(__file__).resolve().parents[3]
README = REPO / "examples/megatron_bridge/README.md"

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
    "bridge-login": "Interactive authentication is supplied by the user, outside CI.",
    "bridge-import": "Full-checkpoint conversion needs a separate model environment; outside this pilot.",
    "bridge-mlflow": "Requires a live tracking service and full checkpoint; outside this pilot.",
    "bridge-real-data": "Full-scale training with external data is outside this pilot; mock data is tested.",
    "bridge-qad": "Full-scale QAD needs prepared checkpoints and external data; outside this pilot.",
    "bridge-prune-active": "Full-model active-parameter search is outside this pilot; other pruning targets run in CI.",
    "bridge-prune-memory": "Full-model memory-target search is outside this pilot; parameter/manual pruning is tested.",
    "bridge-prune-vlm": "Full-model VLM pruning is outside this pilot; text-model pruning is tested.",
    "bridge-generation": "Requires a prepared export and vLLM environment; outside this pilot.",
}

resolve_scenarios(README, [], excluded=EXCLUDED)


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
