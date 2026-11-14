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
cannot check that layout, even with a larger ``--detect-license-in-X-top-lines``: it needs the
SPDX line to read exactly ``Apache-2.0``, and it inserts a missing header at the top of the file.
This hook replaces it for python and shell files:

* It accepts the NVIDIA header anywhere in the file's leading comment block, with any past year and
  any ``SPDX-License-Identifier`` expression that starts with ``Apache-2.0`` (e.g. ``AND MIT``).
* If the header is missing, it inserts ``LICENSE_HEADER`` with the current year: after the leading
  comment block if that block has a third-party copyright notice or source link, else at the top of
  the file, after any shebang and encoding lines, as ``insert-license`` does.
* It fails without changing the file if a third-party copyright notice or source link follows the
  NVIDIA header.
* With ``--third-party-list FILE``, it keeps ``FILE`` listing the files whose leading comment block
  has a third-party copyright notice or source link: the files with code copied from other sources,
  which the Open Source Review Board asks for.

Exits 1 if it changed or flagged any file.
"""

import argparse
import re
import sys
from datetime import date
from pathlib import Path

_LICENSE_HEADER = Path(__file__).resolve().parents[2] / "LICENSE_HEADER"
# A year, list or range ("2024", "2023-2025", "2021, 2023-2024"), ignored like --allow-past-years does
_YEARS = re.compile(r"\b\d{4}([ ,-]+\d{2,4})*\b")
_SPDX_APACHE = re.compile(r"(SPDX-License-Identifier: Apache-2\.0)\b.*")
# A copyright notice (also "SPDX-FileCopyrightText: <year> <holder>"), or a source link such as
# "Adapted from <url>", that doesn't name NVIDIA. A link without such a word, e.g. to documentation,
# doesn't count.
_THIRD_PARTY = re.compile(
    r"^(?!.*NVIDIA).*(copyright"
    r"|\b(adapted|copied|derived|modified|ported|borrowed|taken|inspired)\b.*https?://)",
    re.IGNORECASE,
)
_LIST_NOTE = (
    "# Python and shell files with code copied from other sources: their leading comments have a\n"
    "# third-party copyright notice or source link (CONTRIBUTING.md). The insert-license pre-commit\n"
    "# hook keeps this list up to date.\n"
)


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
    third_party = [i for i in range(start, end) if _THIRD_PARTY.search(lines[i])]
    if found is not None:
        below = [i for i in third_party if i > found]
        if below:
            return (
                f"{path}:{below[0] + 1}: third-party copyright notice or source link below the NVIDIA"
                " license header. Move the NVIDIA header below the third-party header, see"
                " CONTRIBUTING.md."
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


def has_third_party_header(path: Path) -> bool:
    """Whether the leading comment block of ``path`` has a third-party copyright or source link."""
    lines = path.read_text(encoding="utf-8").split("\n")
    start, end = _leading_comment_block(lines)
    return any(_THIRD_PARTY.search(line) for line in lines[start:end])


def update_third_party_list(list_path: Path, paths: list[Path]) -> str | None:
    """Update ``paths`` in the list at ``list_path`` and drop deleted files; say if it changed."""
    old = list_path.read_text(encoding="utf-8") if list_path.exists() else ""
    checked = {path.as_posix() for path in paths}
    kept = {
        name
        for name in old.splitlines()
        if name and not name.startswith("#") and name not in checked and Path(name).exists()
    }
    listed = kept | {path.as_posix() for path in paths if has_third_party_header(path)}
    new = _LIST_NOTE + "".join(f"{name}\n" for name in sorted(listed))
    if new == old:
        return None
    list_path.write_text(new, encoding="utf-8", newline="\n")
    return f"{list_path}: updated the list of files with third-party code"


def license_header() -> list[str]:
    """Return ``LICENSE_HEADER`` as comment lines, with the current year."""
    text = _YEARS.sub(str(date.today().year), _LICENSE_HEADER.read_text(encoding="utf-8"), count=1)
    return [f"# {line}" if line else "#" for line in text.splitlines()]


def main(argv: list[str]) -> int:
    """Fix or flag the NVIDIA license header of each file, and update the third-party file list."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--third-party-list", type=Path, help="list of files with third-party code")
    parser.add_argument("files", nargs="*", type=Path)
    args = parser.parse_args(argv)
    header = license_header()
    messages = [msg for path in args.files if (msg := fix_header(path, header))]
    if args.third_party_list and (
        msg := update_third_party_list(args.third_party_list, args.files)
    ):
        messages.append(msg)
    for msg in messages:
        print(msg)
    return 1 if messages else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
