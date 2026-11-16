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

"""Check checkout discovery and compatibility paths shared by plugin packages."""

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_SKILLS = _ROOT / "skills"


@pytest.mark.parametrize(
    "canonical", sorted(_SKILLS.glob("modelopt-*/SKILL.md")), ids=lambda p: p.parent.name
)
def test_compatibility_entrypoints(canonical):
    """Each old name reaches its canonical sibling through both discovery roots."""
    name = canonical.parent.name
    old_name = name.removeprefix("modelopt-")
    stub = _SKILLS / old_name / "SKILL.md"
    for path in (canonical, stub):
        frontmatter = path.read_text().split("---", 2)[1]
        assert re.findall(r"^name: (.+)$", frontmatter, re.MULTILINE) == [path.parent.name]
        assert re.search(r"^description: \S", frontmatter, re.MULTILINE)
        for discovery in (_ROOT / ".agents/skills", _ROOT / ".claude/skills"):
            assert (discovery / path.parent.name / "SKILL.md").resolve(strict=True) == path

    # The forwarding target must be packaged beside the stub, never back to itself.
    body = stub.read_text().split("---", 2)[2]
    targets = re.findall(r"\]\(([^)]+/SKILL\.md)\)", body)
    assert len(targets) == 1
    assert (stub.parent / targets[0]).resolve(strict=True) == canonical

    for resource in canonical.parent.iterdir():
        if resource.name in ("SKILL.md", "skill-card.md") or resource.name.startswith((".", "__")):
            continue
        legacy_path = stub.parent / resource.name
        assert legacy_path.is_symlink()
        assert legacy_path.resolve(strict=True) == resource
