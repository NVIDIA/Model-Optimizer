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

import fcntl
import threading
import time

from _test_utils.examples.gpu_lock import gpu_lock


def _can_lock(path, mode):
    with open(path, "a") as f:
        try:
            fcntl.flock(f, mode | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return True


def test_shared_holders_overlap_but_exclude_an_exclusive_one(tmp_path):
    with gpu_lock(tmp_path, shared=True):
        assert _can_lock(tmp_path / "gpu.lock", fcntl.LOCK_SH)
        assert not _can_lock(tmp_path / "gpu.lock", fcntl.LOCK_EX)
    assert _can_lock(tmp_path / "gpu.lock", fcntl.LOCK_EX)


def test_exclusive_holder_runs_alone(tmp_path):
    with gpu_lock(tmp_path, shared=False):
        assert not _can_lock(tmp_path / "gpu.lock", fcntl.LOCK_SH)


def test_waiting_exclusive_holder_queues_new_shared_holders(tmp_path):
    exclusive_acquired = threading.Event()

    def take_exclusive():
        with gpu_lock(tmp_path, shared=False):
            exclusive_acquired.set()

    with gpu_lock(tmp_path, shared=True):
        waiter = threading.Thread(target=take_exclusive)
        waiter.start()
        # The exclusive waiter takes the turnstile, which a new shared holder must pass.
        deadline = time.monotonic() + 10
        while _can_lock(tmp_path / "gpu.turnstile", fcntl.LOCK_EX):
            assert time.monotonic() < deadline, "the waiting exclusive holder left the turnstile"
        assert not exclusive_acquired.is_set()
    waiter.join(timeout=10)
    assert exclusive_acquired.is_set()
