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

"""Unit tests for the distributed PTQ session: the distribution plan, the session handle, its synced
attributes, launch detection."""

import pytest
import torch.distributed

from modelopt.torch.utils.distributed import (
    DistributionPlan,
    ModelOptSession,
    Synced,
    SyncedAttribute,
    get_active_mo_session,
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


# ------------------------------------------------------------------------------------- session handle


def test_required_session_fails_fast_when_none_was_entered():
    with pytest.raises(RuntimeError, match="no active session"):
        get_active_mo_session()


def test_optional_session_is_readable_without_a_session():
    """Export / calibration read the session defensively; they must not explode with no session."""
    session = get_active_mo_session(required=False)
    assert session is not None
    assert not session.initialized
    assert session.ep_mesh is None


def test_identity_reads_raise_before_initialization():
    """A stale or zero default would silently mislabel every rank; reading early must fail instead."""
    with pytest.raises(RuntimeError, match="before initialization"):
        _ = ModelOptSession().rank


def test_reset_clears_the_identity_and_the_ep_mesh():
    session = ModelOptSession()
    session.initialize(
        distributed=True, rank=1, world_size=2, local_rank=1, device=None, is_main=False
    )
    session.ep_mesh = object()
    assert (session.rank, session.world_size, session.is_main) == (1, 2, False)
    session._reset()
    assert not session.initialized
    assert session.ep_mesh is None


# ------------------------------------------------------------------------------- synced attributes


class _Session(ModelOptSession):
    """A session with a second synced attribute, and a plain one."""

    other: Synced[int]
    note: str


def _session(rank=None):
    """Rank ``rank`` of a two-rank session; ``None``: a single process."""
    session = _Session()
    session.initialize(
        distributed=rank is not None,
        rank=rank or 0,
        world_size=1 if rank is None else 2,
        local_rank=rank or 0,
        device=None,
        is_main=not rank,
    )
    return session


def _as_rank(monkeypatch, rank, rank0_message=None):
    """Answer the session's collectives as ``rank``; returns what each broadcast was handed.

    A non-zero rank receives ``rank0_message``, as the group's first rank would have sent it.
    """
    sent = []

    def broadcast_object_list(message, group=None, group_src=None):
        assert group_src == 0
        sent.append((message[0], group))
        if rank != 0:
            message[0] = rank0_message

    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: rank)
    monkeypatch.setattr(torch.distributed, "broadcast_object_list", broadcast_object_list)
    return sent


def _gather_as_rank(monkeypatch, rank, others):
    """Answer the session's gathers as ``rank``; returns what each gather was handed.

    ``others`` are the other ranks' messages in rank order, as they would have sent them.
    """
    sent = []

    def all_gather_object(gathered, mine, group=None):
        sent.append((mine, group))
        gathered[:] = [*others[:rank], mine, *others[rank:]]

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: len(others) + 1)
    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)
    monkeypatch.setattr(
        torch.distributed, "broadcast_object_list", lambda *a, **k: pytest.fail("rank 0 decided")
    )
    return sent


def test_only_attributes_annotated_synced_are_synced():
    """The annotation carries both the value's type and the synced semantic."""
    assert isinstance(vars(_Session)["other"], SyncedAttribute)
    assert isinstance(vars(ModelOptSession)["plan"], SyncedAttribute)
    assert "note" not in vars(_Session)


def test_a_plain_attribute_assigns_without_a_collective(monkeypatch):
    for collective in ("broadcast_object_list", "all_gather_object"):
        monkeypatch.setattr(
            torch.distributed, collective, lambda *a, **k: pytest.fail("a collective ran")
        )
    session = _session(rank=1)
    session.note = "local"
    assert session.note == "local"


def test_a_synced_attribute_is_plain_in_a_single_process(monkeypatch):
    """Nothing to sync with: no collective, and an uninitialized session behaves the same."""
    for collective in ("broadcast_object_list", "all_gather_object"):
        monkeypatch.setattr(
            torch.distributed, collective, lambda *a, **k: pytest.fail("a collective ran")
        )
    for session in (_session(), _Session()):
        plan = DistributionPlan(world_size=1)
        session.plan = plan
        assert session.plan is plan
        assert session.decide("other", lambda: 3) == 3
        assert session.agree("smallest", lambda: 4, min) == 4
        assert session.agree("versions", lambda: "2.6") == "2.6"


def test_rank0_sends_what_it_assigns_and_keeps_its_own_copy(monkeypatch):
    sent = _as_rank(monkeypatch, 0)
    session, plan = _session(rank=0), DistributionPlan(world_size=2, ep_size=2)

    session.plan = plan

    assert sent == [(("plan", plan, None), None)]
    assert session.plan.ep_size == 2
    assert session.plan is not plan  # mutating the caller's object must not reach the session


def test_other_ranks_hold_what_rank0_assigned(monkeypatch):
    """Whatever a non-zero rank passes, it ends up with rank 0's value, as every rank does."""
    rank0_plan = DistributionPlan(world_size=2, ep_size=2)
    _as_rank(monkeypatch, 1, rank0_message=("plan", rank0_plan, None))
    session = _session(rank=1)

    session.plan = DistributionPlan(world_size=2)

    assert session.plan is rank0_plan


def test_decide_runs_only_on_rank0(monkeypatch):
    _as_rank(monkeypatch, 1, rank0_message=("other", 7, None))
    session = _session(rank=1)

    assert session.decide("other", lambda: pytest.fail("a non-zero rank decided")) == 7
    assert session.other == 7


def test_rank0_failing_to_decide_still_reaches_the_broadcast(monkeypatch):
    """Rank 0 must still reach the broadcast on failure, or every other rank hangs in it."""
    sent = _as_rank(monkeypatch, 0)

    def fail():
        raise ValueError("bad degrees")

    with pytest.raises(ValueError, match="bad degrees"):
        _session(rank=0).decide("other", fail)
    assert sent == [(("other", None, "ValueError: bad degrees"), None)]


def test_other_ranks_raise_when_rank0_failed(monkeypatch):
    _as_rank(monkeypatch, 1, rank0_message=("other", None, "ValueError: bad degrees"))
    with pytest.raises(RuntimeError, match=r"rank 0 failed to decide ModelOptSession.other.*bad"):
        _session(rank=1).decide("other", lambda: 0)


def test_ranks_setting_different_attributes_fail_instead_of_mixing_them(monkeypatch):
    """Rank 0 setting one attribute while rank 1 sets another would cross the two values."""
    _as_rank(monkeypatch, 1, rank0_message=("other", 7, None))
    with pytest.raises(RuntimeError, match="out of step"):
        _session(rank=1).plan = DistributionPlan(world_size=2)


def test_agree_gives_every_rank_the_combined_value(monkeypatch):
    """Every rank's value counts, and every rank gets the same result."""
    sent = _gather_as_rank(monkeypatch, 1, others=[("smallest", 8, None), ("smallest", 2, None)])

    assert _session(rank=1).agree("smallest", lambda: 4, min) == 2
    assert sent == [(("smallest", 4, None), None)]


def test_agree_keeps_nothing_on_the_session(monkeypatch):
    _gather_as_rank(monkeypatch, 1, others=[("smallest", 8, None)])
    session = _session(rank=1)

    session.agree("smallest", lambda: 4, min)

    assert session._synced == {}


def test_without_combine_every_rank_must_pass_the_same_value(monkeypatch):
    _gather_as_rank(monkeypatch, 1, others=[("versions", "2.6", None)])
    assert _session(rank=1).agree("versions", lambda: "2.6") == "2.6"


def test_ranks_that_disagree_all_raise(monkeypatch):
    """Adopting rank 0's value instead would hide the difference until a collective hangs on it."""
    _gather_as_rank(monkeypatch, 0, others=[("versions", "2.5", None)])
    with pytest.raises(
        RuntimeError, match=r"disagree on versions: rank 0 has '2.6', rank 1 has '2.5'"
    ):
        _session(rank=0).agree("versions", lambda: "2.6")


def test_a_rank_failing_to_agree_still_joins_the_gather(monkeypatch):
    """A failing rank must still reach the gather, or every other rank waits in it."""
    sent = _gather_as_rank(monkeypatch, 1, others=[("smallest", 8, None)])

    def fail():
        raise ValueError("no memory reading")

    with pytest.raises(ValueError, match="no memory reading"):
        _session(rank=1).agree("smallest", fail, min)
    assert sent == [(("smallest", None, "ValueError: no memory reading"), None)]


def test_the_other_ranks_raise_when_one_failed_to_agree(monkeypatch):
    _gather_as_rank(monkeypatch, 0, others=[("smallest", None, "ValueError: no memory reading")])
    with pytest.raises(RuntimeError, match=r"rank 1 failed to agree on smallest.*memory"):
        _session(rank=0).agree("smallest", lambda: 4, min)


def test_ranks_agreeing_out_of_step_fail(monkeypatch):
    _gather_as_rank(monkeypatch, 0, others=[("other", 8, None)])
    with pytest.raises(RuntimeError, match="out of step"):
        _session(rank=0).agree("smallest", lambda: 4, min)


def test_only_synced_attributes_can_be_decided():
    with pytest.raises(AttributeError, match="not a synced attribute"):
        _session().decide("ep_mesh", lambda: None)


def test_reset_clears_synced_attributes():
    session = _session()
    session.plan = DistributionPlan(world_size=1)
    session._reset()
    assert session.plan is None


# ------------------------------------------------------------------------------------ launch detect


@pytest.mark.parametrize(("world", "expected"), [(None, False), ("1", False), ("8", True)])
def test_launch_detection_reads_the_env_before_any_process_group(monkeypatch, world, expected):
    """Must work at parse time, before modelopt_session exists."""
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    if world is not None:
        monkeypatch.setenv("WORLD_SIZE", world)
    assert is_distributed_launch() is expected
