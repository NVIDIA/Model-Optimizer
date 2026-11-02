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

"""Tests for the default timing-cache file in autotune/benchmark.py."""

import stat
import sys
import tempfile
from pathlib import Path

from modelopt.onnx.quantization.autotune.benchmark import Benchmark, default_timing_cache_file


class _NoOpBenchmark(Benchmark):
    def run(self, path_or_bytes, log_file=None):
        return 0.0


def test_default_timing_cache_is_private_and_shared_within_a_process(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    default_timing_cache_file.cache_clear()
    first = Path(_NoOpBenchmark().timing_cache_file)
    second = Path(_NoOpBenchmark().timing_cache_file)
    default_timing_cache_file.cache_clear()

    # A fixed shared path would let another local user pre-create the file or plant a symlink.
    assert first == second
    assert first.name == "trtexec_timing.cache"
    assert first.parent.parent == tmp_path
    if sys.platform != "win32":  # POSIX file modes
        assert stat.S_IMODE(first.parent.stat().st_mode) == 0o700
