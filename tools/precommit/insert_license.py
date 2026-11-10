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

"""Pre-commit hook: insert the NVIDIA license header into python and shell files.

A file with code copied from a third party keeps the original copyright header on top and the NVIDIA
header below it (CONTRIBUTING.md, "Copying code from other sources"). Lucas-C's ``insert-license``
looks for the header only in the top five lines, so this hook replaces it for python and shell files:

* It accepts the NVIDIA header anywhere in the file's leading comment block, with any past year and
  any ``SPDX-License-Identifier`` expression that starts with ``Apache-2.0`` (e.g. ``AND MIT``).
* If the header is missing, it inserts ``LICENSE_HEADER`` with the current year: after the leading
  comment block if that block has a third-party copyright notice, else at the top of the file, after
  any shebang and encoding lines, as ``insert-license`` does.
* It fails without changing the file if a third-party copyright notice follows the NVIDIA header.

Exits 1 if it changed or flagged any file.
"""

import re
import sys
from datetime import date
from pathlib import Path

_LICENSE_HEADER = Path(__file__).resolve().parents[2] / "LICENSE_HEADER"
# A year, list or range ("2024", "2023-2025", "2021, 2023-2024"), ignored like --allow-past-years does
_YEARS = re.compile(r"\b\d{4}([ ,-]+\d{2,4})*\b")
_SPDX_APACHE = re.compile(r"(SPDX-License-Identifier: Apache-2\.0)\b.*")
_THIRD_PARTY_COPYRIGHT = re.compile(r"^(?!.*NVIDIA).*\bcopyright\b", re.IGNORECASE)


def _normalize(line: str) -> str:
    return _SPDX_APACHE.sub(r"\1", _YEARS.sub("", line.strip()))


def _leading_comment_block(lines: list[str]) -> tuple[int, int]:
    """Return where ``insert-license`` inserts a header and the end of the comment block from there."""
    start = 0
    for i, line in enumerate(lines):
        if line.startswith(("#!", "# -*- coding")):
            start = i + 1
        elif line.strip():
            start = i
            break
    end = start
    for i in range(start, len(lines)):
        if lines[i].lstrip().startswith("#"):
            end = i + 1
        elif lines[i].strip():
            break
    return start, end


def fix_header(path: Path, header: list[str]) -> str | None:
    """Insert ``header`` into ``path`` if missing; return a message if the file was changed or flagged."""
    lines = path.read_text(encoding="utf-8").split("\n")
    start, end = _leading_comment_block(lines)
    expected = [_normalize(line) for line in header]
    block = [_normalize(line) for line in lines[start:end]]
    found = next(
        (start + i for i in range(len(block)) if block[i : i + len(expected)] == expected), None
    )
    third_party = [i for i in range(start, end) if _THIRD_PARTY_COPYRIGHT.search(lines[i])]
    if found is not None:
        below = [i for i in third_party if i > found]
        if below:
            return (
                f"{path}:{below[0] + 1}: third-party copyright notice below the NVIDIA license header."
                " Move the NVIDIA header below the third-party header, see CONTRIBUTING.md."
            )
        return None
    pos = end if third_party else start
    before = [""] if third_party else []
    after = [""] if pos < len(lines) and lines[pos].strip() else []
    lines[pos:pos] = before + header + after
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    note = (
        ", below the third-party header. Unless that code is Apache-2.0, add its license to the"
        " SPDX-License-Identifier, see CONTRIBUTING.md."
        if third_party
        else ""
    )
    return f"{path}: inserted the NVIDIA license header at line {pos + len(before) + 1}{note}"


def license_header() -> list[str]:
    """Return ``LICENSE_HEADER`` as comment lines, with the current year."""
    text = _YEARS.sub(str(date.today().year), _LICENSE_HEADER.read_text(encoding="utf-8"), count=1)
    return [f"# {line}" if line else "#" for line in text.splitlines()]


def main(argv: list[str]) -> int:
    """Fix or flag the NVIDIA license header of each file in ``argv``."""
    header = license_header()
    messages = [msg for name in argv if (msg := fix_header(Path(name), header))]
    for msg in messages:
        print(msg)
    return 1 if messages else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
