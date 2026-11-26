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

from sybil import Document
from sybil.parsers.markdown.lexers import FencedCodeBlockLexer

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
    path: Path | None = None
    steps: list[Snippet] = field(default_factory=list)

    @property
    def line(self):
        return self.steps[0].line


def parse_markdown(path: Path) -> dict[str, Snippet]:
    """Extract ID-marked fences without executing code; ignore untagged fences."""
    path = Path(path).resolve()
    source = path.read_text()
    document = Document(source, str(path))
    snippets = {}
    position = 0

    def fail(message, line):
        raise DocTestError(f"{path}:{line}: {message}")

    def marker_before(end):
        pending = None
        start_line = source.count("\n", 0, position) + 1
        for line, text in enumerate(source[position:end].splitlines(), start_line):
            text = re.sub(r"^ {0,3}(?:> ?)*", "", text).strip()
            if pending and text:
                fail(
                    "snippet ID must be immediately followed by a fence (blank lines allowed)", line
                )
            if "modelopt-doc-test:" in text:
                fail(
                    "legacy doc-test directives are unsupported; use snippet IDs and Python tests",
                    line,
                )
            if "modelopt-snippet:" in text:
                match = re.fullmatch(r"<!-- modelopt-snippet: (" + _ID + r") -->", text)
                if not match:
                    fail("invalid snippet ID marker", line)
                pending = match.group(1)
                if pending in snippets:
                    fail(f"duplicate snippet ID: {pending}", line)
        return pending

    fences = sorted(FencedCodeBlockLexer(language=".*")(document), key=lambda fence: fence.start)
    for fence in fences:
        # Nested fences are literal contents of their enclosing snippet.
        if fence.start < position:
            continue
        pending = marker_before(fence.start)
        if pending:
            code = fence.lexemes["source"]
            language = fence.lexemes["language"].strip()
            kind = {"bash": "run", "sh": "run", "python": "python"}.get(
                language, "unsupported:" + language
            )
            line = source.count("\n", 0, fence.start + code.offset) + 1
            snippets[pending] = Snippet(pending, kind, textwrap.dedent(code), line)
        position = fence.end
    pending = marker_before(len(source))
    if pending:
        fail(f"snippet ID {pending} missing its fence", len(source.splitlines()))
    return snippets


def resolve_scenarios(path, scenarios, *, excluded=None):
    """Validate references and account for every named fence before scenarios execute."""
    path = Path(path).resolve()
    snippets = parse_markdown(path)
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
