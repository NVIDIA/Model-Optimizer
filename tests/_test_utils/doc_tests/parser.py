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

"""Parse explicit, trusted Markdown test scenarios without importing ML dependencies."""

import re
from dataclasses import dataclass, field
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

__all__ = ["DocTestError", "Scenario", "Step", "parse_markdown"]


class DocTestError(ValueError):
    """A Markdown scenario is malformed or failed to execute."""


@dataclass
class Step:
    """Source code with its original Markdown location."""

    kind: str
    code: str
    line: int


@dataclass
class Scenario:
    """An explicitly opted-in sequence of setup, shell fences and verification."""

    path: Path
    line: int
    id: str
    profile: str
    timeout_seconds: int
    steps: list[Step] = field(default_factory=list)


def parse_markdown(path: Path) -> list[Scenario]:
    """Validate all directives and return scenarios; unmarked fences are never executed."""
    path = Path(path).resolve()
    lines = path.read_text().splitlines(keepends=True)
    scenarios = []
    current = None
    pending = False
    phase = "setup"
    i = 0

    def fail(message, line):
        raise DocTestError(f"{path}:{line}: {message}")

    while i < len(lines):
        line = i + 1
        text = lines[i].strip()
        fence = re.fullmatch(r"(`{3,}|~{3,})([^`~]*)", text)
        if fence:
            delimiter, language = fence.groups()
            start = i + 1
            i += 1
            while i < len(lines) and not re.fullmatch(
                re.escape(delimiter[0]) + "{" + str(len(delimiter)) + r",}\s*", lines[i].strip()
            ):
                i += 1
            if pending:
                if i == len(lines):
                    fail("unclosed executable fence", line)
                if language.strip() not in {"bash", "sh"}:
                    fail("run requires a bash or sh fence", line)
                current.steps.append(Step("run", "".join(lines[start:i]), start + 1))
                pending = False
                phase = "run"
            i += 1
            continue
        if pending and text:
            fail("run must be immediately followed by a fence (blank lines allowed)", line)
        if "modelopt-doc-test:" not in text:
            i += 1
            continue
        match = re.fullmatch(
            r"<!-- modelopt-doc-test:(begin|setup python|verify python|run|end)(.*)", text
        )
        if not match:
            fail("invalid doc-test directive", line)
        kind, suffix = match.groups()
        body = ""
        if kind in {"run", "end"}:
            if suffix != " -->":
                fail("run/end directives must occupy one line", line)
        else:
            if suffix:
                fail("multiline directive header must end after its kind", line)
            start = i + 1
            i += 1
            while i < len(lines) and lines[i].strip() != "-->":
                i += 1
            if i == len(lines):
                fail("unclosed doc-test directive", line)
            body = "".join(lines[start:i])
        if kind == "begin":
            if current:
                fail("nested scenarios are not allowed", line)
            try:
                metadata = tomllib.loads(body)
            except tomllib.TOMLDecodeError as error:
                fail(f"invalid scenario metadata: {error}", line)
            if set(metadata) != {"id", "profile", "timeout_seconds"}:
                fail("begin requires exactly id, profile, timeout_seconds", line)
            if not isinstance(metadata["id"], str) or not re.fullmatch(
                r"[a-z0-9][a-z0-9-]*", metadata["id"]
            ):
                fail("id must contain lowercase letters, digits or hyphens", line)
            if any(s.id == metadata["id"] for s in scenarios):
                fail("duplicate scenario id", line)
            if metadata["profile"] not in ("cpu", "gpu"):
                fail("profile must be cpu or gpu", line)
            if type(metadata["timeout_seconds"]) is not int or metadata["timeout_seconds"] <= 0:
                fail("timeout_seconds must be a positive integer", line)
            current = Scenario(path, line, **metadata)
            phase = "setup"
        elif current is None:
            fail("directive outside a scenario", line)
        elif kind == "end":
            if not any(step.kind == "run" for step in current.steps):
                fail("scenario needs at least one run fence", line)
            scenarios.append(current)
            current = None
        elif kind == "run":
            if phase == "verify":
                fail("run cannot follow verification", line)
            pending = True
        else:
            kind = kind.split()[0]
            if kind == "setup" and phase != "setup":
                fail("setup must precede all run fences", line)
            if kind == "verify" and phase == "setup":
                fail("verification must follow run fences", line)
            try:
                compile("\n" * line + body, str(path), "exec")
            except SyntaxError as error:
                fail(f"invalid Python: {error.msg}", error.lineno)
            current.steps.append(Step(kind, body, line + 1))
            phase = kind
        i += 1
    if current:
        fail("scenario missing end", current.line)
    return scenarios
