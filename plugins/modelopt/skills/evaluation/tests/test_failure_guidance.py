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

"""Check guidance/config consistency; these tests do not run agents or scoring."""

import json
import re
from pathlib import Path

import yaml

_SKILL = Path(__file__).resolve().parents[1]
_ROOT = _SKILL.parents[3]


def test_terminal_bench_concurrency_matches_recipe():
    template = yaml.safe_load((_SKILL / "recipes/examples/example_eval_next.yaml").read_text())
    recipe = (_SKILL / "recipes/tasks/aa_next/terminal_bench_2_1.md").read_text()
    yaml_block = re.search(r"```yaml\n(.*?)```", recipe, re.DOTALL)
    assert yaml_block is not None
    fragment = yaml.safe_load(yaml_block.group(1))
    for config in (template, fragment):
        benchmark = config["benchmarks"][0]
        assert benchmark["max_concurrent"] == benchmark["sandbox"]["concurrency"] == 8
        assert benchmark["repeats"] == 8


def test_failure_policy_is_reachable_from_parent_and_evaluators():
    policy_path = "evaluation/references/run-validation.md"
    policy = (_SKILL / "references/run-validation.md").read_text()
    title = "Bounded Evaluation-Failure Policy"
    assert f"### {title}" in policy
    assert "**Response gate" in policy and "**Trial gate" in policy
    assert "Both applicable gates must pass" in policy
    assert policy_path in (_ROOT / "AGENTS.md").read_text()
    for path in (
        _SKILL / "SKILL.md",
        _ROOT / ".codex/agents/modelopt_model_evaluator.toml",
        _ROOT / "plugins/modelopt/agents/modelopt-model-evaluator.md",
    ):
        assert title in path.read_text()


def test_evaluation_scenarios_reference_existing_skills():
    scenarios = json.loads((_SKILL / "tests/evals.json").read_text())
    names = [scenario["name"] for scenario in scenarios]
    assert len(names) == len(set(names))
    for scenario in scenarios:
        assert scenario["query"] and scenario["expected_behavior"]
        for skill in scenario["skills"]:
            assert (_SKILL.parent / skill / "SKILL.md").is_file()
