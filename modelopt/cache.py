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

"""Per-user cache directory shared by all ModelOpt packages."""

import functools
import os
import tempfile
from pathlib import Path

__all__ = ["get_cache_dir"]


@functools.cache
def _fallback_cache_dir() -> Path:
    """One private temp directory per process, for when there is no usable cache directory."""
    return Path(tempfile.mkdtemp(prefix="modelopt_cache_"))


def _mkdir(path: Path) -> Path:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return path


def get_cache_dir(*subdirs: str) -> Path:
    """Return ModelOpt's private per-user cache directory (plus ``subdirs``), creating it if needed.

    That is ``$XDG_CACHE_HOME/modelopt`` (``~/.cache/modelopt`` if the variable is unset, empty or
    relative), created with mode 0700. A directory that another user owns or that group or others
    can write to is not used, since they could pre-plant or symlink files in it. Neither is an
    unusable one (no home directory, a directory this user cannot write to, read-only filesystem):
    a private temp directory, created once per process, takes its place, so callers never fail
    because of the cache. Store only data that can be regenerated.
    """
    try:
        xdg_cache_home = Path(os.environ.get("XDG_CACHE_HOME", ""))
        cache_root = xdg_cache_home if xdg_cache_home.is_absolute() else Path.home() / ".cache"
        cache_dir = _mkdir(cache_root / "modelopt")
        if os.name == "posix":
            stat_result = cache_dir.stat()
            if (
                stat_result.st_uid != os.getuid()
                or stat_result.st_mode & 0o022
                or not os.access(cache_dir, os.W_OK | os.X_OK)
            ):
                raise OSError(f"{cache_dir} is not a private, writable directory of this user")
        return _mkdir(cache_dir.joinpath(*subdirs))
    except (OSError, RuntimeError):  # no usable cache directory
        return _mkdir(_fallback_cache_dir().joinpath(*subdirs))
