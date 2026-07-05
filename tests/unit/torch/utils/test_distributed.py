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

import pytest
import torch
from torch import nn

import modelopt.torch.utils.distributed as dist


def test_single_process_defaults():
    assert not dist.is_initialized()
    assert dist.backend() is None
    assert dist.size() == 1
    assert dist.rank() == 0
    assert dist.is_master()
    assert dist.is_last_process()


def test_local_rank_from_env(monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "3")
    assert dist.local_rank() == 3


def test_local_rank_fallback_warns_and_uses_global_rank(monkeypatch):
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    with pytest.warns(UserWarning, match="LOCAL_RANK"):
        assert dist.local_rank() == 0


def test_broadcast_is_identity_in_single_process():
    obj = {"a": [1, 2], "b": "x"}
    assert dist.broadcast(obj) is obj


def test_allgather_returns_singleton_list():
    assert dist.allgather(5) == [5]
    obj = {"k": 1}
    assert dist.allgather(obj) == [obj]


def test_allreduce_sum_and_unsupported_reduction():
    assert dist.allreduce(5) == 5
    with pytest.raises(NotImplementedError):
        dist.allreduce(5, reduction="max")


def test_barrier_is_noop():
    assert dist.barrier() is None


def test_master_only_runs_and_returns_result():
    @dist.master_only
    def compute():
        return {"x": 1}

    assert compute() == {"x": 1}


def test_serialize_deserialize_round_trip():
    obj = {"w": torch.arange(4, dtype=torch.float32), "meta": "hi"}
    tensor = dist._serialize(obj)
    assert tensor.dtype == torch.uint8
    restored = dist._deserialize(tensor)
    assert restored["meta"] == "hi"
    assert torch.equal(restored["w"], obj["w"])
    # The size argument slices the buffer back to the payload; note the
    # deserializer tolerates trailing bytes either way (pickle stops at its
    # own payload), so this pins correct size handling, not strict trimming.
    padded = torch.cat([tensor, torch.zeros(8, dtype=torch.uint8)])
    restored2 = dist._deserialize(padded, size=tensor.numel())
    assert restored2["meta"] == "hi"
    assert torch.equal(restored2["w"], obj["w"])


@pytest.mark.parametrize("group", [None, -1])
def test_process_group_uninitialized_defaults(group):
    pg = dist.DistributedProcessGroup(group)
    assert not pg.is_initialized()
    assert pg.rank() == -1
    assert pg.world_size() == -1
    assert "initialized: False" in repr(pg)


def test_get_dist_syncd_obj_passthrough_when_uninitialized():
    pg = dist.DistributedProcessGroup(None)
    # op must NOT be applied for uninitialized groups
    assert dist.DistributedProcessGroup.get_dist_syncd_obj(7, pg, sum) == 7
    assert dist.DistributedProcessGroup.get_dist_syncd_obj(7, [pg, pg], max) == 7


def test_parallel_state_defaults_and_repr():
    state = dist.ParallelState()
    assert not state.data_parallel_group.is_initialized()
    assert not state.tensor_parallel_group.is_initialized()
    assert not state.expert_model_parallel_group.is_initialized()
    assert state.data_parallel_group.world_size() == -1
    for name in ("data_parallel_group", "tensor_parallel_group", "expert_model_parallel_group"):
        assert name in repr(state)


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
