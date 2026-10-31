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

"""Check README Python CLI option names against each script's public help."""

import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from .parser import DocTestError, parse_markdown

__all__ = ["PythonCommand", "check_cli_options", "python_commands"]


@dataclass(frozen=True)
class PythonCommand:
    """A README invocation, without executing its arguments or launcher."""

    snippet: str
    line: int
    script: Path
    flags: tuple[str, ...]
    argv: tuple[str, ...] = ()


def python_commands(path, *, cwd, cwd_overrides=None):
    """Extract Python script invocations from simple shell commands and continuations."""
    commands = []
    for snippet in parse_markdown(path).values():
        if snippet.kind != "run":
            continue
        directory = Path((cwd_overrides or {}).get(snippet.id, cwd))
        logical, start = "", snippet.line
        for offset, line in enumerate(snippet.code.splitlines()):
            if not logical:
                start = snippet.line + offset
            logical += line.rstrip().removesuffix("\\") + " "
            if line.rstrip().endswith("\\"):
                continue
            lexer = shlex.shlex(logical, posix=True, punctuation_chars=";&|")
            lexer.whitespace_split = True
            tokens = list(lexer)
            logical = ""
            if not tokens:
                continue
            if tokens[0] == "cd" and len(tokens) == 2:
                directory = (directory / tokens[1]).resolve()
                continue
            launcher = Path(tokens[0]).name
            if not (
                re.fullmatch(r"python(?:[23](?:\.\d+)?)?", launcher)
                or launcher == "torchrun"
                or tokens[:2] == ["accelerate", "launch"]
            ):
                continue
            index = next(
                (i for i, token in enumerate(tokens[1:], 1) if token.endswith(".py")), None
            )
            if index is None:
                continue
            if any(token in {";", "&&", "||", "|", "&"} for token in tokens):
                raise DocTestError(f"{path}:{start}: {snippet.id}: compound commands unsupported")
            flags = []
            for token in tokens[index + 1 :]:
                if token == "--":
                    break
                if re.fullmatch(r"--?[a-zA-Z][\w-]*(?:=.*)?", token):
                    flags.append(token.split("=", 1)[0])
            commands.append(
                PythonCommand(
                    snippet.id,
                    start,
                    (directory / tokens[index]).resolve(),
                    tuple(flags),
                    tuple(tokens),
                )
            )
    return commands


@cache
def _help_options(script, timeout):
    if not script.is_file():
        raise DocTestError(f"Python script does not exist: {script}")
    try:
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=script.parent,
            env=dict(
                os.environ,
                CUDA_VISIBLE_DEVICES="",
                NO_COLOR="1",
                HF_HUB_OFFLINE="1",
                HF_DATASETS_OFFLINE="1",
            ),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise DocTestError(f"{script.name} --help timed out after {timeout}s") from error
    if result.returncode:
        raise DocTestError(
            f"{script.name} --help exited with {result.returncode}:\n{result.stdout}{result.stderr}"
        )
    options = set()
    for line in result.stdout.splitlines():
        if re.match(r"^  -[a-zA-Z-]", line):
            definition = re.split(r"\s{2,}", line.strip(), maxsplit=1)[0]
            options.update(re.findall(r"(?<![\w-])--?[a-zA-Z][\w-]*", definition))
    if not options:
        raise DocTestError(f"{script.name} --help did not list any options")
    return options


def check_cli_options(path, command, *, timeout=60):
    """Reject documented flags absent from help; never pass README values to the script."""
    location = f"{path}:{command.line}: {command.snippet}"
    try:
        supported = _help_options(command.script, timeout)
    except DocTestError as error:
        raise DocTestError(f"{location}: {error}") from error
    unknown = set(command.flags) - supported
    if unknown:
        raise DocTestError(f"{location}: {command.script.name}: unknown options: {sorted(unknown)}")
