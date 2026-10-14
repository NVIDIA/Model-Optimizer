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


"""Unit tests for distributed PTQ: the distribution plan, the collectives that fail together, the
process group's lifetime, launch detection."""

from datetime import timedelta

import pytest
import torch.distributed

import modelopt.torch.utils.distributed as dist
from modelopt.torch.utils.distributed import (
    DistributionPlan,
    all_reduce,
    broadcast_from,
    distributed_run,
    is_distributed_launch,
)

# --------------------------------------------------------------------------------- DistributionPlan


def test_plain_world_replicates_everything():
    plan = DistributionPlan(world_size=8)
    assert plan.edp == 8
    assert not plan.is_expert_parallel
    assert not plan.is_fsdp_dense


def test_expert_parallel_splits_the_expert_replica_pool():
    plan = DistributionPlan(world_size=8, ep_size=2)
    assert plan.edp == 4  # each expert shard is replicated 4x
    assert plan.is_expert_parallel


def test_fsdp_dense_plan():
    plan = DistributionPlan(world_size=8, ep_size=8, fsdp_size=8)
    assert plan.is_fsdp_dense
    assert plan.edp == 1


def test_an_ep_degree_that_does_not_divide_the_world_is_rejected():
    """An unsatisfiable plan must fail here, not as a confusing mesh error thousands of ranks later."""
    with pytest.raises(ValueError, match="must divide world_size"):
        DistributionPlan(world_size=8, ep_size=3)


def test_repr_names_the_degrees():
    assert repr(DistributionPlan(world_size=8, ep_size=2)) == (
        "DistributionPlan(world=8, ep_size=2, edp=4, dense=replicate)"
    )
    assert "fsdp_size=8" in repr(DistributionPlan(world_size=8, ep_size=8, fsdp_size=8))


# ------------------------------------------------------------------------------------ collectives


def _run_with_ranks(monkeypatch, world_size):
    """Make this process look like one rank of a live ``world_size``-rank run."""
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: world_size)


def _as_rank(monkeypatch, rank, received=None):
    """Answer the broadcasts of a two-rank run as ``rank``; returns ``(message, sending rank)`` for each.

    A rank that is not the sending one receives ``received``, as the sending rank would have sent it.
    """
    _run_with_ranks(monkeypatch, 2)
    sent = []

    def broadcast_object_list(message, group=None, group_src=None):
        sent.append((message[0], group_src))
        if rank != group_src:
            message[0] = received

    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: rank)
    monkeypatch.setattr(torch.distributed, "broadcast_object_list", broadcast_object_list)
    return sent


def _gather_as_rank(monkeypatch, rank, others):
    """Answer the gathers as ``rank``; returns what each gather was handed.

    ``others`` are the other ranks' messages in rank order, as they would have sent them.
    """
    _run_with_ranks(monkeypatch, len(others) + 1)
    sent = []

    def all_gather_object(gathered, mine, group=None):
        sent.append(mine)
        gathered[:] = [*others[:rank], mine, *others[rank:]]

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)
    monkeypatch.setattr(
        torch.distributed, "broadcast_object_list", lambda *a, **k: pytest.fail("a rank broadcast")
    )
    return sent


def test_a_single_process_computes_alone(monkeypatch):
    """Nothing to sync with: no collective."""
    for collective in ("broadcast_object_list", "all_gather_object"):
        monkeypatch.setattr(
            torch.distributed, collective, lambda *a, **k: pytest.fail("a collective ran")
        )
    assert broadcast_from("x", lambda: 3) == 3
    assert all_reduce("smallest", lambda: 4, min) == 4


def test_the_sending_rank_sends_its_value(monkeypatch):
    sent = _as_rank(monkeypatch, 0)
    plan = DistributionPlan(world_size=2, ep_size=2)

    assert broadcast_from("plan", lambda: plan) is plan
    assert sent == [(("plan", plan, None), 0)]


def test_other_ranks_take_the_sending_ranks_value_without_running_fn(monkeypatch):
    rank0_plan = DistributionPlan(world_size=2, ep_size=2)
    _as_rank(monkeypatch, 1, received=("plan", rank0_plan, None))

    plan = broadcast_from("plan", lambda: pytest.fail("a rank that does not send ran fn"))

    assert plan.ep_size == 2


def test_any_rank_can_broadcast(monkeypatch):
    """With rank=1, rank 1 runs fn and sends it; rank 0 takes rank 1's value."""
    sent = _as_rank(monkeypatch, 1)
    assert broadcast_from("x", lambda: "rank 1's", rank=1) == "rank 1's"
    assert sent == [(("x", "rank 1's", None), 1)]

    _as_rank(monkeypatch, 0, received=("x", "rank 1's", None))
    assert broadcast_from("x", lambda: pytest.fail("rank 0 ran fn"), rank=1) == "rank 1's"


def test_a_rank_outside_the_run_cannot_broadcast(monkeypatch):
    _run_with_ranks(monkeypatch, 2)
    with pytest.raises(ValueError, match="rank=2 is not a rank of this 2-rank run"):
        broadcast_from("x", lambda: 0, rank=2)


def test_the_sending_rank_failing_still_reaches_the_broadcast(monkeypatch):
    """The sending rank must still send on failure, or every other rank hangs in the broadcast."""
    sent = _as_rank(monkeypatch, 0)

    def fail():
        raise ValueError("bad degrees")

    with pytest.raises(ValueError, match="bad degrees"):
        broadcast_from("x", fail)
    assert sent == [(("x", None, "ValueError: bad degrees"), 0)]


def test_other_ranks_raise_when_the_sending_rank_failed(monkeypatch):
    _as_rank(monkeypatch, 1, received=("x", None, "ValueError: bad degrees"))
    with pytest.raises(RuntimeError, match=r"rank 0 failed to compute x.*bad degrees"):
        broadcast_from("x", lambda: 0)


def test_ranks_broadcasting_out_of_step_fail(monkeypatch):
    """Rank 0 sending one value while rank 1 expects another would cross the two."""
    _as_rank(monkeypatch, 1, received=("other", 7, None))
    with pytest.raises(RuntimeError, match="out of step"):
        broadcast_from("plan", lambda: DistributionPlan(world_size=2))


def test_all_reduce_gives_every_rank_the_combined_value(monkeypatch):
    """Every rank's value counts, and every rank gets the same result."""
    sent = _gather_as_rank(monkeypatch, 1, others=[("smallest", 8, None), ("smallest", 2, None)])

    assert all_reduce("smallest", lambda: 4, min) == 2
    assert sent == [("smallest", 4, None)]


def test_a_rank_failing_to_compute_still_joins_the_gather(monkeypatch):
    """A failing rank must still reach the gather, or every other rank waits in it."""
    sent = _gather_as_rank(monkeypatch, 1, others=[("smallest", 8, None)])

    def fail():
        raise ValueError("no memory reading")

    with pytest.raises(ValueError, match="no memory reading"):
        all_reduce("smallest", fail, min)
    assert sent == [("smallest", None, "ValueError: no memory reading")]


def test_the_other_ranks_raise_when_one_failed_to_compute(monkeypatch):
    _gather_as_rank(monkeypatch, 0, others=[("smallest", None, "ValueError: no memory reading")])
    with pytest.raises(RuntimeError, match=r"rank 1 failed to compute smallest.*memory"):
        all_reduce("smallest", lambda: 4, min)


def test_ranks_reducing_out_of_step_fail(monkeypatch):
    _gather_as_rank(monkeypatch, 0, others=[("other", 8, None)])
    with pytest.raises(RuntimeError, match="out of step"):
        all_reduce("smallest", lambda: 4, min)


# ---------------------------------------------------------------------------------- process group


def test_a_single_process_sets_no_group_up(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setattr(dist, "setup", lambda **kwargs: pytest.fail("set a group up"))
    monkeypatch.setattr(dist, "cleanup", lambda: pytest.fail("tore a group down"))
    with distributed_run():
        pass


def test_a_distributed_launch_owns_the_group_even_when_the_body_raises(monkeypatch, capsys):
    """The group is torn down on the way out, and the body's error is printed first: a rank failing
    alone would otherwise wait in the cleanup barrier while the rest sit in a collective."""
    monkeypatch.setenv("WORLD_SIZE", "2")
    calls = []
    monkeypatch.setattr(dist, "setup", lambda timeout=None: calls.append(("setup", timeout)))
    monkeypatch.setattr(dist, "cleanup", lambda: calls.append(("cleanup", capsys.readouterr().out)))

    with (
        pytest.raises(ValueError, match="calibration failed"),
        distributed_run(timeout=timedelta(hours=1)),
    ):
        raise ValueError("calibration failed")

    assert calls[0] == ("setup", timedelta(hours=1))
    assert calls[1][0] == "cleanup"
    assert "calibration failed" in calls[1][1], "the error must be printed before the cleanup"


# ------------------------------------------------------------------------------------ launch detect


@pytest.mark.parametrize(("world", "expected"), [(None, False), ("1", False), ("8", True)])
def test_launch_detection_reads_the_env_before_any_process_group(monkeypatch, world, expected):
    """Must work at parse time, before distributed_run sets the process group up."""
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    if world is not None:
        monkeypatch.setenv("WORLD_SIZE", world)
    assert is_distributed_launch() is expected
