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

"""Tests for the default timing-cache location in autotune/benchmark.py."""

import stat
import sys
import tempfile
from pathlib import Path

import pytest

from modelopt.onnx.quantization.autotune.benchmark import Benchmark


class _NoOpBenchmark(Benchmark):
    def run(self, path_or_bytes, log_file=None):
        return 0.0


@pytest.mark.parametrize("home_is_usable", [True, False])
def test_default_timing_cache_is_private(tmp_path, monkeypatch, home_is_usable):
    home = tmp_path / "home"
    if home_is_usable:
        home.mkdir()
    else:
        home.write_text("")  # a file, so ~/.cache/modelopt cannot be created
    shared_tmp = tmp_path / "shared_tmp"
    shared_tmp.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))  # Path.home() on Windows
    monkeypatch.setattr(tempfile, "tempdir", str(shared_tmp))

    cache_file = Path(_NoOpBenchmark().timing_cache_file)

    # A fixed file in the shared temp dir could be pre-planted or symlinked by another local user.
    if home_is_usable:
        assert cache_file.parent == home / ".cache" / "modelopt"
    else:
        assert cache_file.parent.parent == shared_tmp  # a fresh directory inside the temp dir
    if sys.platform != "win32":  # POSIX file modes
        assert stat.S_IMODE(cache_file.parent.stat().st_mode) == 0o700
