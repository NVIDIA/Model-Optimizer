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
"""Compare Sphinx warnings with an exact normalized baseline."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

__all__ = [
    "compare_warning_counts",
    "load_warning_baseline",
    "normalize_warning",
    "read_warning_counts",
]


_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_SOURCE_LINE = re.compile(r":\d+(?=:(?:<| (?:WARNING|ERROR):))")


def normalize_warning(line: str, repo_root: Path) -> str | None:
    """Return a stable Sphinx diagnostic fingerprint, or ``None`` for other lines."""
    if "WARNING:" not in line and "ERROR:" not in line:
        return None

    warning = _ANSI_ESCAPE.sub("", line.strip())
    warning = warning.replace(f"{repo_root.resolve()}/", "")
    return _SOURCE_LINE.sub("", warning)


def read_warning_counts(warning_log: Path, repo_root: Path) -> Counter[str]:
    """Read and normalize every warning in a Sphinx warning log."""
    warnings = (
        normalize_warning(line, repo_root)
        for line in warning_log.read_text(encoding="utf-8").splitlines()
    )
    return Counter(warning for warning in warnings if warning is not None)


def load_warning_baseline(baseline_path: Path) -> Counter[str]:
    """Load and validate the expected warning counts."""
    data = json.loads(baseline_path.read_text(encoding="utf-8"))
    warning_counts = data.get("warnings")
    if not isinstance(warning_counts, dict):
        raise ValueError("warning baseline must contain a 'warnings' object")

    invalid = {
        warning: count
        for warning, count in warning_counts.items()
        if not isinstance(warning, str) or not isinstance(count, int) or count < 1
    }
    if invalid:
        raise ValueError(f"warning baseline contains invalid entries: {invalid!r}")

    return Counter(warning_counts)


def compare_warning_counts(
    actual: Counter[str], expected: Counter[str]
) -> tuple[Counter[str], Counter[str]]:
    """Return unexpected and resolved warning counts."""
    return actual - expected, expected - actual


def _format_counts(heading: str, counts: Counter[str]) -> list[str]:
    lines = [heading]
    lines.extend(f"  {count}x {warning}" for warning, count in sorted(counts.items()))
    return lines


def main(argv: list[str] | None = None) -> int:
    """Check a Sphinx warning log against its baseline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("warning_log", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)

    actual = read_warning_counts(args.warning_log, args.repo_root)
    expected = load_warning_baseline(args.baseline)
    unexpected, resolved = compare_warning_counts(actual, expected)

    if unexpected or resolved:
        messages = ["Sphinx warning baseline changed."]
        if unexpected:
            messages.extend(_format_counts("Unexpected warnings:", unexpected))
        if resolved:
            messages.extend(
                _format_counts("Resolved warnings to remove from the baseline:", resolved)
            )
        print("\n".join(messages))
        return 1

    print(f"Sphinx warning baseline matched {sum(actual.values())} warning(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
