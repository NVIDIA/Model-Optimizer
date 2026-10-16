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

"""A hung job must not leave a pool that stalls every later job."""

import queue
import time
from types import SimpleNamespace

import pytest
import torch.distributed as dist
from _test_utils.torch.distributed.utils import DistributedWorkerPool, get_live_pool


def _hang(rank, size):
    time.sleep(600)


def _rank_0_fails_and_rank_1_waits_for_it(rank, size):
    if rank == 0:
        raise ValueError("rank 0 failed")
    dist.barrier()  # never completes: its partner left


def test_a_hung_job_kills_the_pool_instead_of_stalling_later_runs(capfd):
    pool = DistributedWorkerPool(world_size=1, backend="gloo", teardown_fn=None)
    try:
        pool.job_timeout = 1
        with pytest.raises(queue.Empty):
            pool.run(_hang)

        assert pool.broken
        assert not any(p.is_alive() for p in pool._processes)
        # Fails at once rather than waiting out another timeout.
        with pytest.raises(RuntimeError, match="killed after an earlier job hung"):
            pool.run(_hang)
        # The stuck worker's stack reached stderr, which is what makes the next CI hang diagnosable.
        assert "_hang" in capfd.readouterr().err
    finally:
        pool.shutdown()


def test_a_failed_rank_does_not_make_its_peers_wait_out_the_job_timeout():
    pool = DistributedWorkerPool(world_size=2, backend="gloo", teardown_fn=None)
    try:
        pool.failure_grace = 2
        start = time.monotonic()
        with pytest.raises(RuntimeError, match="rank 0 failed"):
            pool.run(_rank_0_fails_and_rank_1_waits_for_it)

        assert time.monotonic() - start < 60  # not the 600 s job timeout
        assert pool.broken
    finally:
        pool.shutdown()


def test_a_killed_pool_is_replaced_and_a_live_one_is_reused():
    pools = {}
    made = []

    def make():
        made.append(SimpleNamespace(broken=False))
        return made[-1]

    first = get_live_pool(pools, 2, make)
    assert get_live_pool(pools, 2, make) is first
    first.broken = True
    second = get_live_pool(pools, 2, make)
    assert second is not first and len(made) == 2
