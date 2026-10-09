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

"""CPU (gloo) tests for the failure handling of ``DistributedWorkerPool``."""

import multiprocessing
import os
import signal
import time
from functools import partial
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from _test_utils.torch.distributed.utils import (
    DistributedWorkerPool,
    destroy_extra_process_groups,
    reset_worker_state,
)

pytestmark = [pytest.mark.usefixtures("skip_on_windows"), pytest.mark.timeout(120)]


def _touch(rank, size, directory, name="ran"):
    (Path(directory) / f"{name}{rank}").write_text(str(os.getpid()))


def _raise_on_rank1(rank, size):
    if rank == 1:
        raise ValueError("boom")


def _hang(rank, size):
    time.sleep(3600)


def _die_on_rank0(rank, size):
    if rank == 0:
        os._exit(13)
    time.sleep(3600)


def _extra_groups_are_destroyed(rank, size, directory):
    from torch.distributed.distributed_c10d import _world

    groups = [dist.new_group([0, 1]) for _ in range(3)]
    dist.all_reduce(torch.ones(1), group=groups[0])
    assert len(_world.pg_map) == 1 + 3

    destroy_extra_process_groups()
    assert len(_world.pg_map) == 1

    # Groups can be created and used again afterwards, and the default group still works
    dist.all_reduce(torch.ones(1), group=dist.new_group([0, 1]))
    destroy_extra_process_groups()
    dist.all_reduce(torch.ones(1))
    (Path(directory) / f"groups{rank}").write_text("ok")


def _flags_are_reset(rank, size, directory):
    baseline = (
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cuda.matmul.allow_tf32,
        torch.are_deterministic_algorithms_enabled(),
    )
    torch.backends.cudnn.deterministic = not baseline[0]
    torch.backends.cudnn.benchmark = not baseline[1]
    torch.backends.cudnn.allow_tf32 = not baseline[2]
    torch.backends.cuda.matmul.allow_tf32 = not baseline[3]
    torch.use_deterministic_algorithms(not baseline[4])
    torch.manual_seed(5)

    reset_worker_state(rank, size)

    assert baseline == (
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cuda.matmul.allow_tf32,
        torch.are_deterministic_algorithms_enabled(),
    )
    assert torch.initial_seed() == 1234
    (Path(directory) / f"flags{rank}").write_text("ok")


class _Interrupt(BaseException):
    """Stands in for pytest-timeout's ``Failed``, which is also a ``BaseException``."""


@pytest.fixture
def pool():
    pool = DistributedWorkerPool(
        world_size=2, backend="gloo", teardown_fn=None, ready_timeout=90, run_timeout=60
    )
    yield pool
    pool.shutdown()


def _pids(pool):
    return [p.pid for p in pool._processes]


def _ran_pids(directory, name="ran"):
    return {int(path.read_text()) for path in Path(directory).glob(f"{name}*")}


def test_job_error_keeps_the_workers(pool, tmp_path):
    pids = _pids(pool)
    pool.run(partial(_touch, directory=tmp_path))
    with pytest.raises(RuntimeError, match="boom"):
        pool.run(_raise_on_rank1)
    pool.run(partial(_touch, directory=tmp_path, name="again"))

    assert _pids(pool) == pids
    assert _ran_pids(tmp_path, "again") == set(pids)


def test_timeout_respawns_the_workers(pool, tmp_path):
    pool._run_timeout = 3
    old = _pids(pool)
    with pytest.raises(TimeoutError):
        pool.run(_hang)
    assert pool._broken

    pool.run(partial(_touch, directory=tmp_path))

    assert not pool._broken
    assert not set(_pids(pool)) & set(old)
    assert _ran_pids(tmp_path) == set(_pids(pool))
    for pid in old:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def test_interrupt_respawns_the_workers(pool, tmp_path):
    def interrupt(signum, frame):
        raise _Interrupt

    old = _pids(pool)
    previous = signal.signal(signal.SIGALRM, interrupt)
    signal.setitimer(signal.ITIMER_REAL, 2)
    try:
        with pytest.raises(_Interrupt):
            pool.run(_hang)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    assert pool._broken

    pool.run(partial(_touch, directory=tmp_path))

    assert not set(_pids(pool)) & set(old)
    assert _ran_pids(tmp_path) == set(_pids(pool))


def test_dead_worker_fails_fast_and_respawns(pool, tmp_path):
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="died"):
        pool.run(_die_on_rank0)
    assert time.monotonic() - started < 20
    assert pool._broken

    pool.run(partial(_touch, directory=tmp_path))

    assert _ran_pids(tmp_path) == set(_pids(pool))


def test_gives_up_after_repeated_start_failures(pool, tmp_path):
    pool._run_timeout = 3
    with pytest.raises(TimeoutError):
        pool.run(_hang)

    pool._backend = "not-a-backend"  # makes the workers of every restart crash on start-up
    for _ in range(DistributedWorkerPool._MAX_START_FAILURES):
        with pytest.raises(RuntimeError, match="died"):
            pool.run(partial(_touch, directory=tmp_path))

    children = len(multiprocessing.active_children())
    with pytest.raises(RuntimeError, match="could not be restarted"):
        pool.run(partial(_touch, directory=tmp_path))
    assert len(multiprocessing.active_children()) == children


def test_extra_process_groups_are_destroyed(pool, tmp_path):
    pool.run(partial(_extra_groups_are_destroyed, directory=tmp_path))
    assert len(list(tmp_path.glob("groups*"))) == 2


def test_reset_worker_state_restores_the_spawn_time_settings(pool, tmp_path):
    pool.run(partial(_flags_are_reset, directory=tmp_path))
    assert len(list(tmp_path.glob("flags*"))) == 2
