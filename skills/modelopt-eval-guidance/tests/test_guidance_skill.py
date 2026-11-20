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

"""Deterministic checks for the tool-neutral evaluation guidance and the skills that point at it.

The behavior cases in each skill's ``evals/evals.json`` are judged by a model; these tests pin the
structure they depend on: the guidance stays tool-neutral, the skills that must reach it still do,
and the NEL files keep the rules NEL runs rely on.
"""

import json
import re
from pathlib import Path

import pytest

_SKILLS = Path(__file__).resolve().parents[2]
_ROOT = _SKILLS.parent
_GUIDANCE = _SKILLS / "modelopt-eval-guidance"
_GUIDANCE_FILES = sorted(p for p in _GUIDANCE.rglob("*.md") if "tests" not in p.parts)


def _text(path):
    return Path(path).read_text()


def test_guidance_has_the_expected_files():
    names = {p.name for p in (_GUIDANCE / "references").glob("*.md")}
    assert names >= {
        "doom-loops.md",
        "external-baseline-check.md",
        "nvfp4-modelcard-sampling.md",
        "quantization-benchmarks.md",
        "run-validity.md",
        "vendor-reference-and-sampling.md",
    }
    frontmatter = _text(_GUIDANCE / "SKILL.md").split("---", 2)[1]
    assert re.findall(r"^name: (.+)$", frontmatter, re.MULTILINE) == ["modelopt-eval-guidance"]


@pytest.mark.parametrize("path", _GUIDANCE_FILES, ids=lambda p: str(p.relative_to(_GUIDANCE)))
def test_guidance_relative_links_resolve(path):
    for target in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", _text(path)):
        if target.startswith(("http", "mailto")):
            continue
        assert (path.parent / target).resolve().exists(), f"{path.name}: broken link {target}"


# The guidance is read by runs of every tool, so it must not instruct tool-specific commands.
# NEL appears only as a named exception that points at the NEL skill.
# Internal tool names are spelled in pieces so that this public file does not name them itself.
_INTERNAL_NAME = r"\b" + "E" + "FB" + r"\b"
_INTERNAL_SKILLS = r"fron" + r"tier-?eval"
_TOOL_SPECIFIC = [
    _INTERNAL_NAME,
    _INTERNAL_SKILLS,
    r"`nel (run|status|info|ls)",
    r"\bnemo-evaluator-launcher (run|status)",
    r"NEMO_EVALUATOR_TRUST_PRE_CMD",
]


@pytest.mark.parametrize("path", _GUIDANCE_FILES, ids=lambda p: str(p.relative_to(_GUIDANCE)))
def test_guidance_is_tool_neutral(path):
    text = _text(path)
    for pattern in _TOOL_SPECIFIC:
        assert not re.search(pattern, text, re.IGNORECASE), f"{path.name}: matches {pattern}"


def test_guidance_does_not_override_the_tools_output_budget():
    text = _text(_GUIDANCE / "references/vendor-reference-and-sampling.md")
    assert "Record, do not override" in text
    # the NEL numeric fallback lives with NEL, not in the neutral text as an instruction
    assert "modelopt-evaluation/references/model-card-research.md" in text


def test_run_validity_leaves_the_fault_rule_to_the_tool():
    text = _text(_GUIDANCE / "references/run-validity.md")
    assert "What counts as an evaluation fault is the tool's rule" in text
    assert "do not add a response-level invalidation of your own" in text
    assert "modelopt-evaluation/references/run-validation.md" in text  # NEL still gets its policy


# Skills that must reach the guidance, so a later edit cannot silently cut the link.
@pytest.mark.parametrize(
    "path",
    [
        _ROOT / ".claude/agents/modelopt-model-evaluator.md",
        _ROOT / ".codex/agents/modelopt_model_evaluator.toml",
        _SKILLS / "modelopt-compare-results/SKILL.md",
        _SKILLS / "modelopt-day0-release/SKILL.md",
        _SKILLS / "modelopt-quant-recipe-search/SKILL.md",
        _SKILLS / "modelopt-evaluation/SKILL.md",
    ],
    ids=lambda p: str(p.relative_to(_ROOT)),
)
def test_skills_point_at_the_guidance(path):
    assert "modelopt-eval-guidance" in _text(path)


def test_nel_files_keep_the_rules_nel_runs_rely_on():
    nel = _SKILLS / "modelopt-evaluation"
    validation = _text(nel / "references/run-validation.md")
    for needle in (
        "Bounded Evaluation-Failure Policy",
        "NEL Timeout and Resume Behavior",
        "External Baseline Sanity Check",
        "Timeout and Output-Limit Accounting",
        "NEL Diagnostics",
    ):
        assert needle in validation
    assert "run-validation.md" in _text(nel / "SKILL.md")
    assert "65536" in _text(nel / "references/model-card-research.md")  # NEL numeric fallback
    for name in ("doom_loops.md", "quantization-benchmarks.md"):
        assert (nel / "references" / name).is_file()


def test_evaluator_role_still_routes_nel_to_its_own_steps():
    role = _text(_ROOT / ".claude/agents/modelopt-model-evaluator.md")
    assert "for NEL, follow those steps" in role


def test_monitor_knows_both_tools():
    monitor = _text(_SKILLS / "modelopt-monitor/SKILL.md")
    assert "`type: gym`" in monitor and "type: nel" in monitor
    assert "only when the task names NEL" in monitor
    assert "run-and-babysit-eval" in monitor


def test_recipe_search_names_both_tools_evaluation_skills():
    text = _text(_SKILLS / "modelopt-quant-recipe-search/SKILL.md")
    assert "Use `modelopt-evaluation` to create NEL configs and submit evals (NEL)." in text
    assert "For NeMo Gym, use its evaluation skills" in text
    assert "modelopt-eval-guidance" in text


# General skills are used outside the Day 0 program, so they must reach the evaluation skills
# directly and must not depend on the Day 0 subagent definitions.
@pytest.mark.parametrize(
    "name",
    [
        "modelopt-quant-recipe-search",
        "modelopt-compare-results",
        "modelopt-monitor",
        "modelopt-eval-guidance",
        "modelopt-day0-release",
    ],
)
def test_general_skills_do_not_depend_on_the_day0_evaluator_role(name):
    for path in sorted((_SKILLS / name).rglob("*.md")) + sorted((_SKILLS / name).rglob("*.json")):
        if "tests" in path.parts:
            continue
        assert "evaluator role" not in _text(path).lower(), str(path.relative_to(_SKILLS))


@pytest.mark.parametrize(
    "path",
    [_SKILLS / "common/workspace-management.md", _SKILLS / "common/remote-execution.md"],
    ids=lambda p: p.name,
)
def test_common_skills_do_not_describe_evaluation_as_nel_only(path):
    text = _text(path)
    assert "Evaluation: NEL artifacts" not in text
    assert "A wedged NEL job is" not in text


# Only the skills whose cases were written for this change; the other skills keep their own formats.
_CASE_FILES = [
    _SKILLS / name / "evals/evals.json"
    for name in (
        "modelopt-eval-guidance",
        "modelopt-monitor",
        "modelopt-quant-recipe-search",
        "modelopt-compare-results",
        "modelopt-day0-release",
    )
]


@pytest.mark.parametrize("path", _CASE_FILES, ids=lambda p: p.parent.parent.name)
def test_eval_cases_are_well_formed(path):
    cases = json.loads(path.read_text())
    names = [c["name"] for c in cases]
    assert len(names) == len(set(names)), "duplicate eval case names"
    for c in cases:
        assert {"name", "skills", "query", "expected_behavior"} <= set(c), c.get("name")
        assert c["skills"] and c["query"] and c["expected_behavior"], c["name"]
