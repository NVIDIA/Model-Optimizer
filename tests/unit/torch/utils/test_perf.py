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

"""Counter and threshold logic of maybe_clear_cuda_cache, without needing a GPU."""

import pytest

from modelopt.torch.utils import perf


@pytest.fixture
def cuda_stub(monkeypatch):
    """Stand in for the CUDA allocator: records empty_cache calls, lets tests set the slack."""

    class Stub:
        def __init__(self):
            self.reserved = 0
            self.allocated = 0
            self.empty_cache_calls = 0

    stub = Stub()
    monkeypatch.setattr(perf.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(perf.torch.cuda, "memory_reserved", lambda *a, **k: stub.reserved)
    monkeypatch.setattr(perf.torch.cuda, "memory_allocated", lambda *a, **k: stub.allocated)

    def _empty_cache():
        stub.empty_cache_calls += 1

    monkeypatch.setattr(perf.torch.cuda, "empty_cache", _empty_cache)
    # The counter is process-global; start each test from a known phase.
    monkeypatch.setattr(perf, "_empty_cache_calls", 0)
    return stub


def test_checks_once_per_interval(cuda_stub):
    """One check per ``_EMPTY_CACHE_CHECK_EVERY`` calls, and none in between."""
    cuda_stub.reserved = 8 * 1024**3  # 8 GiB of slack, well over the default threshold
    every = perf._EMPTY_CACHE_CHECK_EVERY

    for _ in range(every - 1):
        perf.maybe_clear_cuda_cache()
    assert cuda_stub.empty_cache_calls == 0, "cleared before reaching the interval"

    perf.maybe_clear_cuda_cache()
    assert cuda_stub.empty_cache_calls == 1

    for _ in range(every):
        perf.maybe_clear_cuda_cache()
    assert cuda_stub.empty_cache_calls == 2, "should clear once per interval, not more"


def test_counter_restarts_on_each_check(cuda_stub):
    """The interval is measured from the last check, not from process start.

    Without the reset, a caller inheriting a mid-interval counter would fire early and then
    drift; with it, every caller gets a full interval between checks.
    """
    cuda_stub.reserved = 8 * 1024**3
    every = perf._EMPTY_CACHE_CHECK_EVERY

    # Land mid-interval, as a second export in the same process would.
    for _ in range(every + 5):
        perf.maybe_clear_cuda_cache()
    assert cuda_stub.empty_cache_calls == 1
    assert perf._empty_cache_calls == 5, "counter should restart at the check, not keep climbing"

    for _ in range(every - 5):
        perf.maybe_clear_cuda_cache()
    assert cuda_stub.empty_cache_calls == 2, "a full interval after the previous check"


def test_does_not_clear_below_the_slack_threshold(cuda_stub):
    """A sampled call with too little reclaimable slack leaves the cache alone."""
    cuda_stub.reserved = 4 * 1024**3
    cuda_stub.allocated = 4 * 1024**3  # no slack at all

    for _ in range(perf._EMPTY_CACHE_CHECK_EVERY * 3):
        perf.maybe_clear_cuda_cache()
    assert cuda_stub.empty_cache_calls == 0


def test_slack_threshold_is_configurable(cuda_stub):
    """A caller that is memory-tight can lower the bar for reclaiming."""
    cuda_stub.reserved = 1024**3  # 1 GiB of slack: under the 4 GiB default, over a 1 MiB bar

    for _ in range(perf._EMPTY_CACHE_CHECK_EVERY):
        perf.maybe_clear_cuda_cache()
    assert cuda_stub.empty_cache_calls == 0, "1 GiB should not trip the default 4 GiB threshold"

    for _ in range(perf._EMPTY_CACHE_CHECK_EVERY):
        perf.maybe_clear_cuda_cache(slack_bytes=1024**2)
    assert cuda_stub.empty_cache_calls == 1


def test_no_cuda_is_a_noop(monkeypatch):
    """Without CUDA the sampled call must not touch the allocator."""
    monkeypatch.setattr(perf.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(perf, "_empty_cache_calls", 0)

    def _boom(*a, **k):
        raise AssertionError("queried the allocator with no CUDA available")

    monkeypatch.setattr(perf.torch.cuda, "memory_reserved", _boom)
    monkeypatch.setattr(perf.torch.cuda, "empty_cache", _boom)

    for _ in range(perf._EMPTY_CACHE_CHECK_EVERY * 2):
        perf.maybe_clear_cuda_cache()
