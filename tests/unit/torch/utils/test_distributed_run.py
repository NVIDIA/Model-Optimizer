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


"""Unit tests for distributed PTQ: the distribution plan and a model's record of it, the collectives
that fail together, the process group's lifetime."""

from contextlib import nullcontext
from datetime import timedelta
from unittest import mock

import pytest
import torch.distributed
from _test_utils.torch.distributed.utils import bounded_collectives
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

import modelopt.torch.utils.distributed as dist
from modelopt.torch.utils.distributed import (
    DistributedState,
    DistributionPlan,
    all_reduce,
    broadcast_from,
    distributed_run,
    distribution_plan,
    set_distribution_plan,
)

# --------------------------------------------------------------------------------- DistributionPlan


@pytest.mark.parametrize(
    ("plan", "edp", "expert_parallel", "fsdp_dense"),
    [
        (DistributionPlan(world_size=8), 8, False, False),
        # each expert shard is replicated 4x
        (DistributionPlan(world_size=8, ep_size=2), 4, True, False),
        (DistributionPlan(world_size=8, ep_size=8, dense="fsdp"), 1, True, True),
    ],
    ids=["plain-world", "expert-parallel", "fsdp-dense"],
)
def test_a_plan_splits_the_world(plan, edp, expert_parallel, fsdp_dense):
    assert plan.edp == edp
    assert plan.is_expert_parallel is expert_parallel
    assert plan.is_fsdp_dense is fsdp_dense


def test_the_default_plan_is_a_single_process():
    assert DistributionPlan() == DistributionPlan(world_size=1, ep_size=1, dense="replicate")


@pytest.mark.parametrize(
    ("degrees", "match"),
    [
        ({"world_size": 8, "ep_size": 3}, "must divide world_size"),
        ({"world_size": 0}, "must be at least 1"),
        ({"world_size": -4}, "must be at least 1"),
        ({"world_size": 4, "ep_size": 0}, "must be at least 1"),
        ({"world_size": 4, "ep_size": -2}, "must be at least 1"),
        ({"world_size": 8, "dense": "sharded"}, "expected 'replicate' or 'fsdp'"),
    ],
    ids=["ep-not-dividing", "world-0", "world-negative", "ep-0", "ep-negative", "unknown-dense"],
)
def test_an_invalid_plan_is_rejected(degrees, match):
    """An unsatisfiable plan must fail here, not as a confusing mesh error thousands of ranks later.
    ep_size=-2 divides a world of 4 and would read as no expert parallelism, replicating an MoE an
    EP knob of -2 asked to shard; ep_size=0 would divide by zero."""
    with pytest.raises(ValueError, match=match):
        DistributionPlan(**degrees)


# ------------------------------------------------------------------- a model's recorded plan


def _model():
    return nn.Sequential(nn.Linear(2, 2), nn.Sequential(nn.Linear(2, 2)))


def test_every_submodule_reads_the_recorded_plan():
    model, plan = _model(), DistributionPlan(world_size=8, ep_size=8)
    set_distribution_plan(model, plan)
    assert all(distribution_plan(module) is plan for module in model.modules())


@pytest.mark.parametrize(
    ("recorded_on", "again"),
    [
        ("model", DistributionPlan(world_size=8, ep_size=8)),
        ("model", DistributionPlan(world_size=8, dense="fsdp")),
        ("submodule", DistributionPlan(world_size=8)),
    ],
    ids=["reload", "redistribute", "submodule"],
)
def test_a_model_records_its_plan_once(recorded_on, again):
    """A model is distributed once, by its load: recording any plan over a recorded one, the same
    or another, means a second writer, and is refused, also when only a submodule carries one. A
    refusal records nothing."""
    model, plan = _model(), DistributionPlan(world_size=8, ep_size=8)
    set_distribution_plan(model if recorded_on == "model" else model[1], plan)
    holders = [module for module in model.modules() if distribution_plan(module) is plan]

    with pytest.raises(ValueError, match="its plan recorded once"):
        set_distribution_plan(model, again)
    for module in model.modules():
        if module in holders:
            assert distribution_plan(module) is plan
        else:
            assert distribution_plan(module) == DistributionPlan()


def test_a_model_with_no_plan_is_a_single_process():
    assert distribution_plan(_model()) == DistributionPlan()


def _outcome(call):
    """``(value, None)``, or ``(None, error)`` if ``call()`` raises.

    For pooled jobs, which cannot use ``pytest.raises``: its failure is a ``BaseException``, which
    ends the worker instead of failing the job.
    """
    try:
        return call(), None
    except Exception as error:
        return None, error


def _no_plan_on_several_ranks_job(rank, size):
    _, error = _outcome(lambda: distribution_plan(_model()))
    assert isinstance(error, RuntimeError), error
    assert f"is held in a run of {size} ranks" in str(error), error
    model = _model()
    fully_shard(model, mesh=init_device_mesh("cpu", (size,)))
    _, error = _outcome(lambda: distribution_plan(model))
    assert isinstance(error, RuntimeError), error
    assert "is FSDP2-sharded" in str(error), error
    plan = DistributionPlan(world_size=size, dense="fsdp")
    set_distribution_plan(model, plan)
    assert distribution_plan(model) is plan
    model = _model()
    set_distribution_plan(model, DistributionPlan(world_size=size))
    model.add_module("added", nn.Linear(2, 2))
    _, error = _outcome(lambda: distribution_plan(model.added))
    assert isinstance(error, RuntimeError), error


def test_a_model_with_no_plan_raises_in_a_run_of_several_ranks(cpu_dist_workers):
    """Held whole or FSDP2-sharded, rather than be taken for a single process's; once a plan is
    recorded, that plan is what it reads. A module added after the plan was recorded has none, and
    raises too, rather than read as a single process's."""
    cpu_dist_workers.run(_no_plan_on_several_ranks_job)


# ------------------------------------------------------------------------------------ collectives


def _run_with_ranks(monkeypatch, world_size):
    """Make this process look like one rank of a live ``world_size``-rank run."""
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: world_size)


def _as_rank(monkeypatch, rank):
    """Record the broadcasts of a two-rank run as ``rank``; returns ``(message, sending rank)`` for each."""
    _run_with_ranks(monkeypatch, 2)
    sent = []

    def broadcast_object_list(message, group=None, group_src=None):
        sent.append((message[0], group_src))

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
    with pytest.raises(ValueError, match="rank=1"):
        broadcast_from("x", lambda: 3, rank=1)


def _broadcast_from_job(rank, size):
    ran = []

    def compute():
        ran.append(rank)
        return f"rank {rank}'s"

    def fail():
        raise ValueError("bad degrees")

    with bounded_collectives():
        for sender in range(size):
            ran.clear()
            assert broadcast_from("x", compute, rank=sender) == f"rank {sender}'s"
            assert ran == ([rank] if rank == sender else []), "a rank that does not send ran fn"
        _, error = _outcome(lambda: broadcast_from("x", compute, rank=size))
        assert isinstance(error, ValueError), error
        assert f"rank={size} is not a rank of this {size}-rank run" in str(error), error
        _, error = _outcome(lambda: broadcast_from("x", fail))
        if rank == 0:
            assert isinstance(error, ValueError), error
        else:
            assert isinstance(error, RuntimeError), error
            assert "rank 0 failed to compute x" in str(error), error
        sent, error = _outcome(lambda: broadcast_from("plan" if rank == 0 else "other", lambda: 7))
        if rank == 0:
            assert (sent, error) == (7, None)
        else:
            assert isinstance(error, RuntimeError), error
            assert "out of step" in str(error), error


def test_broadcast_from_gives_every_rank_the_sending_ranks_value(cpu_dist_workers):
    """Over a real broadcast: any rank can send, and only it runs fn. A rank outside the run is
    refused, the sending rank's failure stops every rank, and ranks out of step fail."""
    cpu_dist_workers.run(_broadcast_from_job)


@pytest.mark.parametrize("error", [ValueError, SystemExit])
def test_the_sending_rank_failing_still_reaches_the_broadcast(monkeypatch, error):
    """The sending rank must still send on failure, or every other rank hangs in the broadcast; a
    SystemExit from an argparse-style check too. A stand-in broadcast catches that at once, where a
    real one would wait out its timeout."""
    sent = _as_rank(monkeypatch, 0)

    def fail():
        raise error("bad degrees")

    with pytest.raises(error, match="bad degrees"):
        broadcast_from("x", fail)
    assert sent == [(("x", None, f"{error.__name__}: bad degrees"), 0)]


def _all_reduce_job(rank, size):
    def fail_on_rank_1():
        if rank == 1:
            raise ValueError("no memory reading")
        return 4

    with bounded_collectives():
        assert all_reduce("smallest", lambda: [8, 2][rank], min) == 2
        _, error = _outcome(lambda: all_reduce("smallest", fail_on_rank_1, min))
        if rank == 1:
            assert isinstance(error, ValueError), error
        else:
            assert isinstance(error, RuntimeError), error
            assert "rank 1 failed to compute smallest" in str(error), error
        _, error = _outcome(
            lambda: all_reduce("smallest" if rank == 0 else "other", lambda: 4, min)
        )
        assert isinstance(error, RuntimeError), error
        assert "out of step" in str(error), error


def test_all_reduce_gives_every_rank_the_combined_value(cpu_dist_workers):
    """Over a real gather: every rank's value counts and every rank gets the same result. One rank's
    failure stops every rank, and ranks out of step fail."""
    cpu_dist_workers.run(_all_reduce_job)


@pytest.mark.parametrize("error", [ValueError, SystemExit])
def test_a_rank_failing_to_compute_still_joins_the_gather(monkeypatch, error):
    """A failing rank must still reach the gather, or every other rank waits in it; a SystemExit
    too. A stand-in gather catches that at once, where a real one would wait out its timeout."""
    sent = _gather_as_rank(monkeypatch, 1, others=[("smallest", 8, None)])

    def fail():
        raise error("no memory reading")

    with pytest.raises(error, match="no memory reading"):
        all_reduce("smallest", fail, min)
    assert sent == [("smallest", None, f"{error.__name__}: no memory reading")]


def _bounded_barrier_job(rank, size):
    groups, own = [], object()
    with (
        mock.patch.object(
            torch.distributed, "barrier", lambda *a, group=None, **k: groups.append(group)
        ),
        bounded_collectives(),
    ):
        torch.distributed.barrier(group=None)  # as modelopt's barrier() calls it
        torch.distributed.barrier(group=own)
    assert len(groups) == 2 and groups[0] not in (None, own) and groups[1] is own, groups


def test_bounded_collectives_bound_a_barrier_called_with_group_none(cpu_dist_workers):
    """modelopt's ``barrier()`` names ``group=None`` explicitly; the bound holds for it too, or a
    rank left waiting in it would wait out the default group's timeout. A collective on a group of
    its own keeps that group."""
    cpu_dist_workers.run(_bounded_barrier_job)


# ---------------------------------------------------------------------------------- process group


def test_a_single_process_is_rank_0_of_one_on_the_given_device(monkeypatch):
    """With no ranks to sync with, the run sets no group up and tears none down."""
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setattr(dist, "setup", lambda **kwargs: pytest.fail("set a group up"))
    monkeypatch.setattr(dist, "cleanup", lambda: pytest.fail("tore a group down"))
    with distributed_run(device="cpu") as state:
        assert state == DistributedState(rank=0, world_size=1, device="cpu")
        assert state.is_main


def test_a_distributed_launch_yields_this_ranks_state_on_its_cuda_device(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    timeouts = []
    monkeypatch.setattr(dist, "setup", lambda timeout=None: timeouts.append(timeout))
    monkeypatch.setattr(dist, "cleanup", lambda: None)
    monkeypatch.setattr(dist, "rank", lambda group=None: 2)
    monkeypatch.setattr(dist, "size", lambda group=None: 4)
    monkeypatch.setattr(dist, "local_rank", lambda: 1)
    with distributed_run(timeout=timedelta(hours=1)) as state:
        assert state == DistributedState(rank=2, world_size=4, device=torch.device("cuda:1"))
        assert not state.is_main
    assert timeouts == [timedelta(hours=1)]


@pytest.mark.parametrize("device", ["cpu", "cuda:3"])
def test_a_distributed_launch_refuses_another_device(monkeypatch, device):
    """Every rank runs on the CUDA device its LOCAL_RANK names, so another device, or another index,
    is refused, before any setup, rather than ignored."""
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(dist, "setup", lambda **kwargs: pytest.fail("set a group up"))
    with pytest.raises(ValueError, match=f"device='{device}'"), distributed_run(device=device):
        pass


class _ExitedError(Exception):
    """Raised by a stand-in for abort(), which exits the process instead of returning."""


@pytest.mark.parametrize("in_except", [False, True], ids=["entered-plainly", "entered-in-except"])
@pytest.mark.parametrize(
    ("launch", "body_raises", "ends_by"),
    [
        (True, None, "cleanup"),
        (True, ValueError, "abort"),
        (True, SystemExit, "cleanup"),
        (False, None, "cleanup"),
        (False, ValueError, "cleanup"),
    ],
    ids=["launch", "launch-raises", "launch-exits", "single-process", "single-process-raises"],
)
def test_the_run_ends_the_group_it_set_up(
    monkeypatch, capsys, launch, body_raises, ends_by, in_except
):
    """The launch's group, or the one a single process's FSDP2 load sets up mid-run, is destroyed on
    the way out. A rank of several whose body raised exits through abort() instead, and torchrun tears
    the job down: the cleanup's barrier and destroy_process_group would wait on peers that may sit in
    a collective this rank never reaches. Every rank reaches an intentional exit. Only the body's own
    exception counts, not one the caller is handling where it entered the run."""
    if launch:
        monkeypatch.setenv("WORLD_SIZE", "2")
    else:
        monkeypatch.delenv("WORLD_SIZE", raising=False)
    live, ended = [], []
    monkeypatch.setattr(dist, "setup", lambda timeout=None: live.append("the launch's group"))
    monkeypatch.setattr(dist, "is_initialized", lambda: bool(live))
    monkeypatch.setattr(dist, "rank", lambda group=None: 0)
    monkeypatch.setattr(dist, "size", lambda group=None: 2 if launch else 1)
    monkeypatch.setattr(dist, "local_rank", lambda: 0)
    monkeypatch.setattr(dist, "cleanup", lambda: ended.append("cleanup"))

    def abort():
        ended.append("abort")
        raise _ExitedError

    monkeypatch.setattr(dist, "abort", abort)
    escapes = _ExitedError if ends_by == "abort" else body_raises
    expected = pytest.raises(escapes) if escapes else nullcontext()

    def run():
        with expected, distributed_run(device="cuda" if launch else "cpu"):
            if not launch:
                live.append("the body's group")
            if body_raises:
                raise body_raises

    if in_except:
        try:
            raise KeyError("the caller's")
        except KeyError:
            run()
    else:
        run()

    assert ended == [ends_by]
    # An intentional exit is not reported as an error.
    assert ("body raised" in capsys.readouterr().out) == (body_raises is ValueError)


@pytest.mark.parametrize("raises", [False, True], ids=["returns", "raises"])
@pytest.mark.parametrize("world_size", [None, "2"], ids=["single-process", "launch"])
def test_the_run_leaves_a_group_it_found_alone(monkeypatch, world_size, raises):
    """Under a launch too, and when the body raises: a caller that set its group up before the run
    keeps it, and gets the error."""
    if world_size is None:
        monkeypatch.delenv("WORLD_SIZE", raising=False)
    else:
        monkeypatch.setenv("WORLD_SIZE", world_size)
        monkeypatch.setattr(dist, "setup", lambda timeout=None: pytest.fail("set it up again"))
        monkeypatch.setattr(dist, "rank", lambda group=None: 0)
        monkeypatch.setattr(dist, "size", lambda group=None: 2)
        monkeypatch.setattr(dist, "local_rank", lambda: 0)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "cleanup", lambda: pytest.fail("destroyed the caller's group"))
    monkeypatch.setattr(dist, "abort", lambda: pytest.fail("exited the caller's process"))
    expected = pytest.raises(ValueError, match="calibration failed") if raises else nullcontext()
    with expected, distributed_run(device="cpu" if world_size is None else "cuda"):
        if raises:
            raise ValueError("calibration failed")
