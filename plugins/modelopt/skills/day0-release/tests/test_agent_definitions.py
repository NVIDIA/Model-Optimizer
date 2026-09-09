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

import re
from pathlib import Path

import tomllib

_ROOT = Path(__file__).resolve().parents[5]
_CODEX_AGENTS = _ROOT / ".codex" / "agents"
_CLAUDE_AGENTS = _ROOT / "plugins" / "modelopt" / "agents"
_CLAUDE_LINKS = _ROOT / ".claude" / "agents"
_REFERENCED_DOC = re.compile(r"`((?:\.agents/skills/|references/)[^`]+\.md)`")


def _load_claude_agent(path: Path) -> tuple[str, str]:
    text = path.read_text()
    assert text.startswith("---\n"), f"{path} has no YAML frontmatter"
    frontmatter, body = text.removeprefix("---\n").split("\n---\n", 1)
    names = [
        line.removeprefix("name: ")
        for line in frontmatter.splitlines()
        if line.startswith("name: ")
    ]
    assert len(names) == 1, f"{path} must declare exactly one name"
    return names[0], body.strip()


def _assert_references_exist(instructions: str, source: Path) -> None:
    skill_dir = None
    for match in _REFERENCED_DOC.finditer(instructions):
        reference = match.group(1)
        if reference.startswith(".agents/"):
            target = _ROOT / reference
            if target.name == "SKILL.md":
                skill_dir = target.parent
        else:
            assert skill_dir is not None, f"{source}: {reference} has no owning skill"
            target = skill_dir / reference
        assert target.is_file(), f"{source} references missing file {target.relative_to(_ROOT)}"


def test_agent_definitions_are_synchronized():
    codex = {}
    for path in _CODEX_AGENTS.glob("*.toml"):
        with path.open("rb") as file:
            agent = tomllib.load(file)
        for field in ("name", "description", "developer_instructions"):
            assert agent.get(field), f"{path} is missing {field}"
        name = agent["name"].replace("_", "-")
        assert path.stem.replace("_", "-") == name, f"{path} does not match agent name {name}"
        codex[name] = (path, agent["developer_instructions"].strip())

    claude = {}
    for path in _CLAUDE_AGENTS.glob("*.md"):
        name, instructions = _load_claude_agent(path)
        assert path.stem == name, f"{path} does not match agent name {name}"
        claude[name] = (path, instructions)

    links = {path.stem: path for path in _CLAUDE_LINKS.glob("*.md")}
    assert codex.keys() == claude.keys() == links.keys()

    for name, (codex_path, codex_instructions) in codex.items():
        claude_path, claude_instructions = claude[name]
        assert codex_instructions == claude_instructions, (
            f"Core instructions differ between {codex_path} and {claude_path}"
        )
        _assert_references_exist(codex_instructions, codex_path)

        link = links[name]
        assert link.is_symlink(), f"{link} must be a symlink"
        assert link.resolve() == claude_path.resolve(), f"{link} must target {claude_path}"
