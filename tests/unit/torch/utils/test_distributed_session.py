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

"""Unit tests for the distributed PTQ session: the distribution plan, the session handle, launch detection."""

import pytest

from modelopt.torch.utils.distributed import (
    DistributionPlan,
    ModelOptSession,
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


# ------------------------------------------------------------------------------------ launch detect


@pytest.mark.parametrize(("world", "expected"), [(None, False), ("1", False), ("8", True)])
def test_launch_detection_reads_the_env_before_any_process_group(monkeypatch, world, expected):
    """Must work at parse time, before modelopt_session exists."""
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    if world is not None:
        monkeypatch.setenv("WORLD_SIZE", world)
    assert is_distributed_launch() is expected
