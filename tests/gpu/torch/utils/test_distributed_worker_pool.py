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

import multiprocessing
import os
import queue
import time

import pytest
import torch
import torch.distributed as dist
from _test_utils.torch.distributed.utils import DistributedWorkerPool, reset_worker_state


def _hang(rank, size):
    time.sleep(600)


def _rank_0_fails_and_rank_1_hangs(rank, size):
    if rank == 0:
        raise ValueError("rank 0 failed")
    time.sleep(600)


def _fail(rank, size):
    raise ValueError("job failed")


def _teardown_barrier(rank, size):
    dist.barrier()  # like megatron_worker_teardown: blocks while a peer is still in the job


def _rank_0_fails_and_rank_1_dies_later(rank, size):
    if rank == 0:
        raise ValueError("rank 0 failed")
    time.sleep(2)  # rank 0 has reported by then
    os._exit(13)


def _rank_0_dies_and_rank_1_waits_for_it(rank, size):
    if rank == 0:
        os._exit(13)  # what an illegal memory access or an OOM kill does: no exception, no result
    time.sleep(600)


def _torch_settings():
    return (
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cuda.matmul.allow_tf32,
        torch.are_deterministic_algorithms_enabled(),
    )


def _change_the_settings_then_reset_them(rank, size):
    spawn_time = _torch_settings()
    torch.backends.cudnn.deterministic = not spawn_time[0]
    torch.backends.cudnn.benchmark = not spawn_time[1]
    torch.backends.cudnn.allow_tf32 = not spawn_time[2]
    torch.backends.cuda.matmul.allow_tf32 = not spawn_time[3]
    torch.use_deterministic_algorithms(not spawn_time[4])
    torch.manual_seed(5)
    assert _torch_settings() != spawn_time

    reset_worker_state(rank, size)

    assert _torch_settings() == spawn_time
    assert torch.initial_seed() == 1234


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


def test_a_failed_job_retires_the_pool_even_when_every_rank_replied():
    """A failed rank reports before its teardown, which may still be blocked; never reuse it."""
    pool = DistributedWorkerPool(world_size=1, backend="gloo", teardown_fn=None)
    try:
        with pytest.raises(RuntimeError, match="job failed"):
            pool.run(_fail)
        assert pool.broken
    finally:
        pool.shutdown()


@pytest.mark.parametrize(
    ("job", "message"),
    [
        (_rank_0_dies_and_rank_1_waits_for_it, r"died.*13"),
        # The error a rank reported before the death is kept in the message
        (_rank_0_fails_and_rank_1_dies_later, r"(?s)died.*13.*rank 0 failed"),
    ],
)
def test_a_dead_worker_fails_the_run_at_once_and_kills_its_peers(job, message):
    pool = DistributedWorkerPool(world_size=2, backend="gloo", teardown_fn=None)
    try:
        pool.job_timeout = 30  # a run that misses the death fails here, not at the test's own cap
        start = time.monotonic()
        with pytest.raises(RuntimeError, match=message):
            pool.run(job)

        assert time.monotonic() - start < 20  # not the 600 s job timeout
        assert pool.broken
        assert not any(p.is_alive() for p in pool._processes)  # the peer in the job is killed too
    finally:
        pool.shutdown()


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"backend": "not-a-backend"}, RuntimeError),  # every worker dies in init_process_group
        ({"ready_timeout": 0.01}, queue.Empty),  # the workers are still starting when the wait ends
    ],
)
def test_a_failed_start_raises_and_leaves_no_workers(kwargs, error):
    before = {p.pid for p in multiprocessing.active_children()}
    with pytest.raises(error):
        DistributedWorkerPool(world_size=1, **{"backend": "gloo", "teardown_fn": None, **kwargs})
    time.sleep(1)
    assert {p.pid for p in multiprocessing.active_children()} <= before


def test_reset_worker_state_restores_the_settings_a_worker_starts_with():
    pool = DistributedWorkerPool(world_size=1, backend="gloo", teardown_fn=None)
    try:
        pool.run(_change_the_settings_then_reset_them)
    finally:
        pool.shutdown()
