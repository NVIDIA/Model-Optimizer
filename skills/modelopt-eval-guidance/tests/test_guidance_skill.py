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
_GENERIC_FILES = [p for p in _GUIDANCE_FILES if "tools" not in p.relative_to(_GUIDANCE).parts]
_TOOL_FILES = [p for p in _GUIDANCE_FILES if p not in _GENERIC_FILES]


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


# Generic guidance names no tool at all: every tool-specific fact is in references/tools/<tool>.md, so
# retiring a tool is deleting its file (and its adapter), not editing the generic text.
_NAMED_TOOLS = [
    r"\bNEL\b",
    r"\bnemo[- ]?gym\b",
    r"nemo[-_]evaluator",
    r"nemo-skills",
    r"nemo_evaluator_config",
]


@pytest.mark.parametrize("path", _GENERIC_FILES, ids=lambda p: str(p.relative_to(_GUIDANCE)))
def test_generic_guidance_names_no_tool(path):
    text = _text(path)
    for pattern in _NAMED_TOOLS:
        assert not re.search(pattern, text, re.IGNORECASE), f"{path.name}: names a tool ({pattern})"


# The skills that carry tool-specific facts keep them under references/tools/ (and the gate adapters).
_GENERIC_SKILLS = (
    "modelopt-eval-guidance",
    "modelopt-compare-results",
    "modelopt-monitor",
    "modelopt-day0-release",
    "modelopt-quant-recipe-search",
)


def _generic_skill_files(name):
    root = _SKILLS / name
    files = [
        p
        for p in list(root.rglob("*.md")) + list(root.glob("scripts/*.py"))
        if "tests" not in p.parts
        and "tools" not in p.relative_to(root).parts
        and not p.name.endswith("_adapter.py")
    ]
    return sorted(files)


@pytest.mark.parametrize("name", _GENERIC_SKILLS)
def test_generic_skill_files_name_no_tool(name):
    for path in _generic_skill_files(name):
        text = _text(path)
        for pattern in _NAMED_TOOLS:
            assert not re.search(pattern, text, re.IGNORECASE), f"{path}: names a tool ({pattern})"


@pytest.mark.parametrize(
    "name", ["modelopt-compare-results", "modelopt-monitor", "modelopt-day0-release"]
)
def test_skills_with_tool_facts_keep_one_file_per_tool(name):
    names = {p.name for p in (_SKILLS / name / "references/tools").glob("*.md")}
    assert {"nel.md", "gym.md"} <= names


def test_nel_deletion_is_removing_files():
    """The files NEL retirement removes are exactly the tool references and the adapter."""
    adapters = {p.name for p in (_SKILLS / "modelopt-day0-release/scripts").glob("*_adapter.py")}
    assert adapters == {"nel_adapter.py", "gym_adapter.py"}


def test_guidance_makes_the_tool_reference_the_first_read():
    # round-5 NEL trials loaded the guidance but never opened their tool's file; step 0 must say to read it first
    text = _text(_GUIDANCE / "SKILL.md")
    assert re.search(
        r"^0\. \*\*First, the reference file of the tool you are using\*\*", text, re.MULTILINE
    )
    assert "references/tools/<tool>.md" in text


def test_each_tool_has_a_reference_and_the_index_lists_it():
    names = {p.name for p in _TOOL_FILES}
    assert names >= {"index.md", "nel.md", "gym.md"}
    index = _text(_GUIDANCE / "references/tools/index.md")
    assert "nel.md" in index and "gym.md" in index
    assert "default tool" in index.lower() and "NEL" in index


def test_nel_reference_keeps_the_nel_rules():
    nel = _text(_GUIDANCE / "references/tools/nel.md")
    for needle in (
        "65536",  # numeric output-budget fallback
        "modelopt-evaluation/references/run-validation.md",  # bounded-failure policy
        "scancel",  # never cancel a wedged job
        "nemo_evaluator_config",
        "answer_tokens_statistics_avg",
        "one matched correction",  # the supported recovery when the baseline fails the gate at the output cap
    ):
        assert needle in nel


def test_gym_reference_keeps_the_gym_rules():
    gym = _text(_GUIDANCE / "references/tools/gym.md")
    for needle in (
        "add-model-eval-config",
        "run-and-babysit-eval",
        "nemo-gym-score-triage",
        "<agent>/mean/reward",
        "UNVERIFIED",
    ):
        assert needle in gym


def test_guidance_does_not_override_the_tools_output_budget():
    text = _text(_GUIDANCE / "references/vendor-reference-and-sampling.md")
    assert "Record, do not override" in text
    assert "never invent one" in text


def test_run_validity_leaves_the_fault_rule_to_the_tool():
    text = _text(_GUIDANCE / "references/run-validity.md")
    assert "What counts as an evaluation fault is the tool's rule" in text
    assert "Do not add a response-level invalidation of your own" in text
    assert "modelopt-evaluation/references/run-validation.md" in _text(
        _GUIDANCE / "references/tools/nel.md"
    )


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


def test_evaluator_role_defaults_to_nel_and_points_at_the_skills():
    role = _text(_ROOT / ".claude/agents/modelopt-model-evaluator.md")
    assert "NEL, the default" in role
    for pointer in (
        "modelopt-eval-guidance/SKILL.md",
        "modelopt-compare-results/SKILL.md",
        "common/workspace-management.md",
    ):
        assert pointer in role


# Logic belongs in the skills, so any user of the skills gets it; the subagent files carry only the
# scope, the pointers and the handoff contract. These phrases moved from the evaluator role into the guidance.
_MOVED_FROM_THE_ROLE = [
    "retained PTQ smoke server",
    "Never report scores from an incomplete or invalid run",
    "export of the results to MLflow",
    "serving record",
    "tool-managed deployments",
]


@pytest.mark.parametrize(
    "path",
    [
        _ROOT / ".claude/agents/modelopt-model-evaluator.md",
        _ROOT / ".codex/agents/modelopt_model_evaluator.toml",
    ],
    ids=lambda p: p.name,
)
def test_evaluator_role_holds_no_evaluation_logic(path):
    role = _text(path).lower()
    for phrase in _MOVED_FROM_THE_ROLE:
        assert phrase.lower() not in role, f"{path.name} still carries: {phrase}"
    assert len(role.split()) < 220, "the evaluator role is growing again; put the logic in a skill"


def test_the_moved_rules_are_in_the_guidance():
    guidance = _text(_GUIDANCE / "SKILL.md").lower()
    for phrase in _MOVED_FROM_THE_ROLE:
        assert phrase.lower() in guidance, phrase


@pytest.mark.parametrize(
    "path", sorted((_ROOT / ".claude/agents").glob("*.md")), ids=lambda p: p.name
)
def test_no_subagent_file_is_a_workflow(path):
    body = _text(path).split("---", 2)[2]
    assert len(body.split()) < 260, f"{path.name}: move workflow text into a skill"


def test_monitor_keeps_each_tools_check_in_its_reference():
    monitor = _text(_SKILLS / "modelopt-monitor/SKILL.md")
    assert "references/tools/" in monitor
    nel = _text(_SKILLS / "modelopt-monitor/references/tools/nel.md")
    assert (
        "type: nel" in nel
        and "nel status" in nel
        and "Use this only when the task names NEL" in nel
    )
    gym = _text(_SKILLS / "modelopt-monitor/references/tools/gym.md")
    assert "type: gym" in gym and "run-and-babysit-eval" in gym


def test_compare_results_keeps_each_tools_rules_in_its_reference():
    nel = _text(_SKILLS / "modelopt-compare-results/references/tools/nel.md")
    assert "Score Extraction" in nel and "run-validation.md" in nel
    gym = _text(_SKILLS / "modelopt-compare-results/references/tools/gym.md")
    assert "<agent>/mean/reward" in gym and "nemo-gym-score-triage" in gym


def test_recipe_search_points_at_the_tool_index():
    text = _text(_SKILLS / "modelopt-quant-recipe-search/SKILL.md")
    assert "modelopt-eval-guidance/references/tools/index.md" in text


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
    # A file keeps one of the two case formats in use under skills/: id/question/expected_skill or
    # name/skills/query. Either way a case needs a unique id, a prompt and behaviors to judge.
    cases = json.loads(path.read_text())
    ids = [c.get("id") or c.get("name") for c in cases]
    assert all(ids) and len(ids) == len(set(ids)), "missing or duplicate eval case ids"
    for c in cases:
        prompt = c.get("question") or c.get("query")
        # a negative case ("must not trigger") has expected_skill null, so only its presence is required
        assert "expected_skill" in c or c.get("skills"), c.get("id") or c.get("name")
        assert prompt and c["expected_behavior"], c.get("id") or c.get("name")
