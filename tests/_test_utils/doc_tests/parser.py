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
import textwrap
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
    """An explicitly opted-in sequence of setup, executable fences and verification."""

    path: Path
    line: int
    id: str
    profile: str
    timeout_seconds: int
    steps: list[Step] = field(default_factory=list)
    manual: bool = False
    min_gpus: int = 0
    requires: list[str] = field(default_factory=list)


def parse_markdown(path: Path, *, require_coverage: bool = False) -> list[Scenario]:
    """Validate all directives and return scenarios; unmarked fences are never executed."""
    path = Path(path).resolve()
    lines = path.read_text().splitlines(keepends=True)
    scenarios = []
    current = None
    pending = False
    excluded = False
    phase = "setup"
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
            if i == len(lines) and (pending or require_coverage):
                fail("unclosed executable fence", line)
            if pending:
                language = language.strip()
                if language not in {"bash", "sh", "python"}:
                    fail("run requires a bash, sh or python fence", line)
                if pending == "background" and language == "python":
                    fail("background requires a shell fence", line)
                kind = "python" if language == "python" else pending
                code = textwrap.dedent(
                    "".join(unquote(part) if quoted else part for part in lines[start:i])
                )
                if kind == "python":
                    try:
                        compile("\n" * start + code, str(path), "exec")
                    except SyntaxError as error:
                        fail(f"invalid Python: {error.msg}", error.lineno)
                current.steps.append(Step(kind, code, start + 1))
                pending = False
                phase = "run"
            elif require_coverage and not excluded:
                fail("unaccounted fence: add run or skip with a reason", line)
            excluded = False
            i += 1
            continue
        if (pending or excluded) and text:
            fail("run must be immediately followed by a fence (blank lines allowed)", line)
        if "modelopt-doc-test:" not in text:
            i += 1
            continue
        skip = re.fullmatch(r"<!-- modelopt-doc-test:skip (.+) -->", text)
        if skip:
            if current or not skip.group(1).strip():
                fail("skip requires a reason outside a scenario", line)
            excluded = True
            i += 1
            continue
        match = re.fullmatch(
            r"<!-- modelopt-doc-test:(begin|setup python|verify python|run background|run|end)(.*)",
            text,
        )
        if not match:
            fail("invalid doc-test directive", line)
        kind, suffix = match.groups()
        body = ""
        if kind in {"run", "run background", "end"}:
            if suffix != " -->":
                fail("run/end directives must occupy one line", line)
        else:
            if suffix:
                fail("multiline directive header must end after its kind", line)
            start = i + 1
            i += 1
            while i < len(lines) and unquote(lines[i]).strip() != "-->":
                i += 1
            if i == len(lines):
                fail("unclosed doc-test directive", line)
            body = textwrap.dedent("".join(unquote(part) for part in lines[start:i]))
        if kind == "begin":
            if current:
                fail("nested scenarios are not allowed", line)
            try:
                metadata = tomllib.loads(body)
            except tomllib.TOMLDecodeError as error:
                fail(f"invalid scenario metadata: {error}", line)
            required = {"id", "profile", "timeout_seconds"}
            if not required <= metadata.keys() or metadata.keys() - required - {
                "manual",
                "min_gpus",
                "requires",
            }:
                fail(
                    "begin requires id, profile, timeout_seconds; optional manual, min_gpus, requires",
                    line,
                )
            if type(metadata.get("manual", False)) is not bool:
                fail("manual must be a boolean", line)
            if type(metadata.get("min_gpus", 0)) is not int or metadata.get("min_gpus", 0) < 0:
                fail("min_gpus must be a nonnegative integer", line)
            dependencies = metadata.get("requires", [])
            if not isinstance(dependencies, list) or not all(
                isinstance(name, str) and re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_.]*", name)
                for name in dependencies
            ):
                fail("requires must be a list of Python module names", line)
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
            runs = [
                step.kind for step in current.steps if step.kind in {"run", "python", "background"}
            ]
            if not runs:
                fail("scenario needs at least one run fence", line)
            if "python" in runs and any(kind != "python" for kind in runs):
                fail("use separate scenarios for Python and shell fences", line)
            if "background" in runs and (
                runs.count("background") != 1
                or runs[-1] != "background"
                or not any(step.kind == "verify" for step in current.steps)
            ):
                fail("background must be the last run and have verification", line)
            scenarios.append(current)
            current = None
        elif kind in {"run", "run background"}:
            if phase == "verify":
                fail("run cannot follow verification", line)
            pending = "background" if kind == "run background" else "run"
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
    if excluded:
        fail("skip missing its fence", len(lines))
    if current:
        fail("scenario missing end", current.line)
    return scenarios
