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

"""Extract named Markdown snippets and resolve Python-owned test scenarios."""

import re
import textwrap
from dataclasses import dataclass, field, replace
from pathlib import Path

__all__ = ["DocTestError", "Scenario", "Snippet", "parse_markdown", "resolve_scenarios"]

_ID = r"[a-z0-9][a-z0-9-]*"


class DocTestError(ValueError):
    """A Markdown snippet or scenario is malformed or failed to execute."""


@dataclass
class Snippet:
    """A named code fence with its original Markdown location."""

    id: str
    kind: str
    code: str
    line: int


@dataclass
class Scenario:
    """An ordered snippet selection with worker-side orchestration and resource limits."""

    id: str
    snippets: list[str]
    profile: str = "cpu"
    timeout_seconds: int = 60
    handler: str | None = None
    manual: bool = False
    min_gpus: int = 0
    requires: list[str] = field(default_factory=list)
    background: bool = False
    path: Path | None = None
    steps: list[Snippet] = field(default_factory=list)

    @property
    def line(self):
        return self.steps[0].line


def parse_markdown(path: Path, *, require_ids: bool = False) -> dict[str, Snippet]:
    """Extract ID-marked fences without executing code; optionally require IDs on every fence."""
    path = Path(path).resolve()
    lines = path.read_text().splitlines(keepends=True)
    snippets = {}
    pending = None
    i = 0

    def fail(message, line):
        raise DocTestError(f"{path}:{line}: {message}")

    def unquote(text):
        return re.sub(r"^ {0,3}> ?", "", text)

    while i < len(lines):
        line = i + 1
        text = unquote(lines[i]).strip()
        fence = re.fullmatch(r"(`{3,}|~{3,})([^`~]*)", text)
        if fence:
            delimiter, language = fence.groups()
            quoted = lines[i].lstrip().startswith(">")
            start = i + 1
            i += 1
            while i < len(lines) and not re.fullmatch(
                re.escape(delimiter[0]) + "{" + str(len(delimiter)) + r",}\s*",
                (unquote(lines[i]) if quoted else lines[i]).strip(),
            ):
                i += 1
            if i == len(lines) and (pending or require_ids):
                fail("unclosed snippet fence", line)
            if pending:
                code = textwrap.dedent(
                    "".join(unquote(part) if quoted else part for part in lines[start:i])
                )
                language = language.strip()
                kind = {"bash": "run", "sh": "run", "python": "python"}.get(
                    language, "unsupported:" + language
                )
                snippets[pending] = Snippet(pending, kind, code, start + 1)
                pending = None
            elif require_ids:
                fail("unaccounted fence: add a snippet ID and a test or exclusion", line)
            i += 1
            continue
        if pending and text:
            fail("snippet ID must be immediately followed by a fence (blank lines allowed)", line)
        if "modelopt-doc-test:" in text:
            fail(
                "legacy doc-test directives are unsupported; use snippet IDs and Python tests", line
            )
        if "modelopt-snippet:" in text:
            match = re.fullmatch(r"<!-- modelopt-snippet: (" + _ID + r") -->", text)
            if not match:
                fail("invalid snippet ID marker", line)
            pending = match.group(1)
            if pending in snippets:
                fail(f"duplicate snippet ID: {pending}", line)
        i += 1
    if pending:
        fail(f"snippet ID {pending} missing its fence", len(lines))
    return snippets


def resolve_scenarios(path, scenarios, *, excluded=None):
    """Validate references and account for every fence before any scenario can execute."""
    path = Path(path).resolve()
    snippets = parse_markdown(path, require_ids=True)
    excluded = {} if excluded is None else excluded
    unknown = excluded.keys() - snippets.keys()
    if unknown:
        raise DocTestError(f"{path}: unknown excluded snippets: {sorted(unknown)}")
    result, covered = [], set()
    for scenario in scenarios:

        def fail(message):
            raise DocTestError(f"{path}: scenario {scenario.id}: {message}")

        missing = set(scenario.snippets) - snippets.keys()
        if missing:
            fail(f"unknown snippets: {sorted(missing)}")
        overlap = set(scenario.snippets) & excluded.keys()
        if overlap:
            fail(f"snippets are both tested and excluded: {sorted(overlap)}")
        steps = [snippets[key] for key in scenario.snippets]
        kinds = {step.kind for step in steps}
        if not kinds <= {"run", "python"} or len(kinds) != 1:
            fail("select only shell fences or only Python fences in each scenario")
        if scenario.background:
            if kinds != {"run"} or scenario.handler is None:
                fail("background requires shell snippets and a verification handler")
            steps = [*steps[:-1], replace(steps[-1], kind="background")]
        for step in steps:
            if step.kind == "python":
                try:
                    compile("\n" * (step.line - 1) + step.code, str(path), "exec")
                except SyntaxError as error:
                    fail(f"invalid Python in {step.id} at line {error.lineno}: {error.msg}")
        covered.update(scenario.snippets)
        result.append(replace(scenario, path=path, steps=steps))
    unaccounted = snippets.keys() - covered - excluded.keys()
    if unaccounted:
        raise DocTestError(f"{path}: unaccounted snippets: {sorted(unaccounted)}")
    return result
