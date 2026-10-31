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

"""Read documented Python commands for the existing example test runners."""

import re
from pathlib import Path

from _test_utils.doc_tests.cli import python_commands
from _test_utils.doc_tests.parser import DocTestError

__all__ = ["readme_command"]


def readme_command(path, snippet, script, *, overrides=None, cwd_overrides=None):
    """Extract one invocation, replacing explicit CI inputs without executing shell text."""
    path = Path(path)
    commands = [
        command
        for command in python_commands(path, cwd=path.parent, cwd_overrides=cwd_overrides)
        if command.snippet == snippet and command.script.name == script
    ]
    if len(commands) != 1:
        raise DocTestError(
            f"{path}: {snippet}: expected one {script} invocation, found {len(commands)}"
        )
    command = commands[0]
    options = {
        key if key.startswith("-") else "--" + key: value
        for key, value in (overrides or {}).items()
    }
    result = []
    tokens = list(command.argv)
    index = 0
    while index < len(tokens):
        token = tokens[index]
        option = token.split("=", 1)[0]
        if option not in options:
            result.append(token)
            index += 1
            continue
        value = options.pop(option)
        result.extend([option, str(value)])
        index += 1
        if "=" not in token:
            if index < len(tokens) and not re.match(r"--?[A-Za-z]", tokens[index]):
                index += 1
    for option, value in options.items():
        result.extend([option, str(value)])

    def expand_default(match):
        default = match.group(1)
        if default is None:
            raise DocTestError(
                f"{path}:{command.line}: {snippet}: unresolved shell variable {match.group(0)}"
            )
        return default

    expanded = [
        re.sub(r"\$\{\w+(?::-([^{}]*))?\}|\$[A-Za-z_]\w*", expand_default, token)
        for token in result
    ]
    if any("$" in token for token in expanded):
        raise DocTestError(f"{path}:{command.line}: {snippet}: unresolved shell expression")
    return expanded
