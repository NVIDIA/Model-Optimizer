# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Single-process contracts only: everything here must hold without an initialized
# torch.distributed process group.

import os
import threading

import pytest
from torch import nn

import modelopt.torch.utils.distributed as dist


def test_local_rank_fallback_warns_and_uses_global_rank(monkeypatch):
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    with pytest.warns(UserWarning, match="LOCAL_RANK"):
        assert dist.local_rank() == 0


def test_master_only_runs_and_returns_result():
    @dist.master_only
    def compute():
        return {"x": 1}

    assert compute() == {"x": 1}


@pytest.mark.parametrize("group", [None, -1])
def test_process_group_rank_is_minus_one_when_uninitialized(group):
    assert dist.DistributedProcessGroup(group).rank() == -1


def test_get_group_returns_none_when_uninitialized():
    assert dist.get_group([0]) is None


def test_is_dtensor_sharded_false_for_plain_model():
    assert not dist.is_dtensor_sharded(nn.Linear(2, 2))


def test_filelock_context_creates_and_removes_lockfile(tmp_path):
    lock_path = str(tmp_path / "my.lock")
    with dist.FileLock(lock_path):
        assert os.path.exists(lock_path)
    assert not os.path.exists(lock_path)


def test_filelock_try_acquire_conflict(tmp_path):
    lock_path = str(tmp_path / "my.lock")
    first = dist.FileLock(lock_path)
    second = dist.FileLock(lock_path)
    assert first.try_acquire()
    assert not second.try_acquire()  # already held
    first.release()
    assert second.try_acquire()
    second.release()
    assert not os.path.exists(lock_path)


def test_filelock_all_acquire_waits_for_release(tmp_path):
    lock_path = str(tmp_path / "my.lock")
    first = dist.FileLock(lock_path)
    assert first.try_acquire()
    releaser = threading.Timer(0.2, first.release)
    releaser.start()
    try:
        # blocks in wait() polling until the timer releases the first lock
        with dist.FileLock(lock_path, all_acquire=True, poll_time=0.02):
            assert os.path.exists(lock_path)
        assert not os.path.exists(lock_path)
    finally:
        releaser.join()
