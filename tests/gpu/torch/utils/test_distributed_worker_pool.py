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

"""A hung job must not leave a pool that stalls every later job."""

import queue
import time

import pytest
import torch.distributed as dist
from _test_utils.torch.distributed.utils import DistributedWorkerPool


def _hang(rank, size):
    time.sleep(600)


def _rank_0_fails_and_rank_1_hangs(rank, size):
    if rank == 0:
        raise ValueError("rank 0 failed")
    time.sleep(600)


def _teardown_barrier(rank, size):
    dist.barrier()  # like megatron_worker_teardown: blocks while a peer is still in the job


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


def test_a_failed_rank_is_reported_even_when_its_teardown_blocks_on_a_hung_peer():
    pool = DistributedWorkerPool(world_size=2, backend="gloo", teardown_fn=_teardown_barrier)
    try:
        pool.failure_grace = 2
        pool.job_timeout = 120  # without the early report the parent would wait this long
        start = time.monotonic()
        with pytest.raises(RuntimeError, match="rank 0 failed"):
            pool.run(_rank_0_fails_and_rank_1_hangs)

        assert time.monotonic() - start < 25
        assert pool.broken
    finally:
        pool.shutdown()
