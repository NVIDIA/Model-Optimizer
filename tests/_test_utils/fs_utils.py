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

import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

# Read + traverse for everyone, write for no one.
_RO_DIR = 0o555
_RO_FILE = 0o444


def _set_read_only(root: Path, read_only: bool) -> None:
    for path in [root, *root.rglob("*")]:
        if path.is_symlink():
            continue
        if read_only:
            path.chmod(_RO_DIR if path.is_dir() else _RO_FILE)
        else:
            path.chmod(path.stat().st_mode | stat.S_IWUSR)


@contextmanager
def read_only_tree(path: Path | str) -> Iterator[Path]:
    """Make ``path`` read-only for the body, restoring write permission on exit.

    For session/module-scoped model-directory fixtures: a test that writes into a shared
    directory silently changes what every later test sees, so make that fail loudly at the
    write instead. Write permission is restored on teardown so pytest's ``tmp_path``
    retention cleanup can still remove the tree.
    """
    path = Path(path)
    _set_read_only(path, read_only=True)
    try:
        yield path
    finally:
        if path.exists():
            _set_read_only(path, read_only=False)
