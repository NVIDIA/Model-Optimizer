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

"""Tests for the shared per-user cache directory in modelopt/cache.py."""

import os
import stat
import sys
import tempfile
from types import SimpleNamespace

import pytest

from modelopt import cache
from modelopt.cache import get_cache_dir

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="requires POSIX file modes")
posix_not_root = pytest.mark.skipif(
    sys.platform == "win32" or os.geteuid() == 0, reason="needs POSIX modes and a non-root user"
)


@pytest.fixture
def env(tmp_path, monkeypatch):
    home = tmp_path / "home"
    shared_tmp = tmp_path / "shared_tmp"
    shared_tmp.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))  # Path.home() on Windows
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setattr(tempfile, "tempdir", str(shared_tmp))
    cache._fallback_cache_dir.cache_clear()
    yield SimpleNamespace(home=home, xdg=tmp_path / "xdg", shared_tmp=shared_tmp)
    cache._fallback_cache_dir.cache_clear()


@pytest.mark.parametrize(
    "scenario",
    [
        "home",
        "xdg",
        "xdg_relative",
        "home_unusable",
        pytest.param("writable_by_others", marks=posix_only),
        pytest.param("owned_by_someone_else", marks=posix_only),
        pytest.param("not_writable", marks=posix_not_root),
    ],
)
def test_cache_dir_is_private(env, monkeypatch, scenario):
    if scenario == "home_unusable":
        env.home.write_text("")  # a file, so ~/.cache/modelopt cannot be created
    else:
        env.home.mkdir()
    if scenario == "xdg":
        monkeypatch.setenv("XDG_CACHE_HOME", str(env.xdg))
    elif scenario == "xdg_relative":
        monkeypatch.setenv("XDG_CACHE_HOME", "relative/cache")  # the XDG spec says to ignore it
    elif scenario == "writable_by_others":
        (env.home / ".cache" / "modelopt").mkdir(parents=True)
        (env.home / ".cache" / "modelopt").chmod(0o777)
    elif scenario == "owned_by_someone_else":
        # Safe permissions, so that only the ownership is wrong.
        (env.home / ".cache" / "modelopt").mkdir(mode=0o700, parents=True)
        monkeypatch.setattr(os, "getuid", lambda: os.stat(env.home).st_uid + 1)
    elif scenario == "not_writable":
        (env.home / ".cache" / "modelopt").mkdir(mode=0o500, parents=True)

    path = get_cache_dir()

    if scenario in ("home", "xdg_relative"):
        assert path == env.home / ".cache" / "modelopt"
    elif scenario == "xdg":
        assert path == env.xdg / "modelopt"
    else:
        assert path.parent == env.shared_tmp  # a fresh private directory in the temp dir
    if sys.platform != "win32":  # POSIX file modes
        assert stat.S_IMODE(path.stat().st_mode) == 0o700
    if scenario == "not_writable":
        (env.home / ".cache" / "modelopt").chmod(0o700)  # so that pytest can remove it


def test_subdirs_are_created_under_the_cache_dir(env):
    env.home.mkdir()
    assert get_cache_dir("a", "b") == env.home / ".cache" / "modelopt" / "a" / "b"
    assert (env.home / ".cache" / "modelopt" / "a" / "b").is_dir()


def test_fallback_is_one_directory_per_process(env):
    env.home.write_text("")  # a file, so ~/.cache/modelopt cannot be created
    assert get_cache_dir("x").parent == get_cache_dir("y").parent
    assert len(list(env.shared_tmp.iterdir())) == 1
