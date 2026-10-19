# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Pre-commit hook: normalize .yml to .yaml in modelopt_recipes/.

Standardizes YAML file extensions to ``.yaml`` for consistency.  When a
``.yml`` file is detected, it is renamed to ``.yaml`` and the hook exits
with code 1 so the user can re-stage and commit.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    """Rename .yml files to .yaml, exit 1 if any were renamed."""
    renamed: list[tuple[Path, Path]] = []
    collisions: list[tuple[Path, Path]] = []
    for f in sys.argv[1:]:
        path = Path(f)
        if path.suffix == ".yml" and path.is_file():
            new_path = path.with_suffix(".yaml")
            if new_path.exists():
                collisions.append((path, new_path))
                continue
            os.rename(path, new_path)
            renamed.append((path, new_path))

    if collisions:
        for old, new in collisions:
            print(f"ERROR: Cannot rename {old} -> {new} (destination already exists)")
        return 1

    if renamed:
        for old, new in renamed:
            print(f"Renamed: {old} -> {new}")
        print(
            f"\n{len(renamed)} file(s) renamed from .yml to .yaml. "
            "Please re-stage the changes and commit again."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
