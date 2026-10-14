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

"""Stdlib-only reader for the gym example templates (the CI skill job has no PyYAML)."""

from pathlib import Path

EXAMPLES = Path(__file__).parents[1] / "recipes" / "examples" / "gym"


def block(template, key):
    """Return (header, nested lines) for the first ``key:`` in ``template``."""
    lines = template.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip().startswith(f"{key}:"))
    indent = len(lines[start]) - len(lines[start].lstrip())
    nested = []
    for line in lines[start + 1 :]:
        if line.strip() and len(line) - len(line.lstrip()) <= indent:
            break
        nested.append(line)
    return lines[start], "\n".join(nested)
