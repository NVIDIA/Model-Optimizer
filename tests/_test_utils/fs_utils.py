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

"""Filesystem helpers for tests."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


def _manifest(root: Path) -> dict[str, tuple[int, int]]:
    """``relative path -> (size, mtime_ns)`` for every file below ``root``."""
    return {
        str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in root.rglob("*")
        if p.is_file() and not p.is_symlink()
    }


@contextmanager
def assert_unmodified_tree(path: Path | str) -> Iterator[Path]:
    """Fail if anything under ``path`` is added, removed, or rewritten inside the ``with``.

    For session/module-scoped model-directory fixtures: a test that writes into a shared
    directory silently changes what every later test sees. Comparing a file manifest on
    teardown catches that. ``chmod``-ing the tree read-only would report at the write rather
    than at teardown, but it only works for an unprivileged user -- root has
    ``CAP_DAC_OVERRIDE`` and writes straight through the permission bits, and the CI
    containers run as root.
    """
    path = Path(path)
    before = _manifest(path)
    yield path
    if not path.exists():
        raise AssertionError(f"shared fixture directory {path} was deleted by a test")
    after = _manifest(path)
    added = sorted(after.keys() - before.keys())
    removed = sorted(before.keys() - after.keys())
    changed = sorted(k for k in before.keys() & after.keys() if before[k] != after[k])
    if added or removed or changed:
        raise AssertionError(
            f"shared fixture directory {path} was modified by a test "
            f"(added={added}, removed={removed}, changed={changed}); "
            "copy it into the test's own tmp_path instead of writing into the shared tree"
        )
