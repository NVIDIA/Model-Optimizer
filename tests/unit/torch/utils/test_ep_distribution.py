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

"""Unit tests for EP degree selection and the decided distribution plan."""

from types import SimpleNamespace

import pytest
import torch.distributed as dist

import modelopt.torch.utils.distributed as mto_dist
import modelopt.torch.utils.moe as moe
from modelopt.torch.utils.moe import (
    _resolve_ep_size,
    choose_ep_dp_degrees,
    decide_distribution_plan,
    is_expert_parallel_capable,
    num_experts_from_config,
)


def cfg(**kw):
    return SimpleNamespace(**kw)


def moe_cfg(experts=64, **kw):
    """An EP-capable MoE config: transformers advertises EP support via ``base_model_ep_plan``."""
    return cfg(num_experts=experts, base_model_ep_plan={"experts": "grouped"}, **kw)


# ------------------------------------------------------------------------------- expert counting


def test_dense_config_reports_no_experts():
    assert num_experts_from_config(cfg(hidden_size=128)) == 0


@pytest.mark.parametrize(
    "attr", ["num_experts", "num_local_experts", "n_routed_experts", "num_routed_experts"]
)
def test_every_spelling_of_the_expert_count_is_understood(attr):
    """Architectures disagree on the field name; missing one would silently disable EP."""
    assert num_experts_from_config(cfg(**{attr: 16})) == 16


@pytest.mark.parametrize("nest", ["text_config", "llm_config"])
def test_expert_count_is_found_on_a_vlms_nested_text_model(nest):
    """For a VLM the MoE is the nested LM, so the root config has no expert count of its own."""
    assert num_experts_from_config(cfg(**{nest: cfg(n_routed_experts=32)})) == 32


# ----------------------------------------------------------------------------------- capability


def test_moe_with_an_ep_plan_is_capable():
    assert is_expert_parallel_capable(moe_cfg())


def test_moe_without_an_ep_plan_is_not_capable():
    """No ``base_model_ep_plan`` means transformers cannot shard these experts -> use plain FSDP2."""
    assert not is_expert_parallel_capable(cfg(num_experts=64))


def test_ep_plan_without_experts_is_not_capable():
    assert not is_expert_parallel_capable(cfg(base_model_ep_plan={"experts": "grouped"}))


def test_capability_is_detected_through_a_vlm_nesting():
    assert is_expert_parallel_capable(cfg(text_config=moe_cfg(experts=8)))


# ------------------------------------------------------------------------------- degree selection


def test_largest_valid_ep_degree_is_chosen():
    assert choose_ep_dp_degrees(8, 64) == (8, 1)


def test_ep_must_divide_the_expert_count_so_ranks_hold_whole_experts():
    assert choose_ep_dp_degrees(8, 6) == (2, 4)


def test_ep_falls_back_to_one_when_nothing_divides():
    assert choose_ep_dp_degrees(8, 5) == (1, 8)


@pytest.mark.parametrize(("world", "experts"), [(8, 0), (1, 64)])
def test_no_expert_parallelism_for_dense_or_single_rank(world, experts):
    assert choose_ep_dp_degrees(world, experts) == (1, world)


@pytest.mark.parametrize("world", [2, 4, 8, 16])
def test_degrees_always_factor_the_world(world):
    ep, dp = choose_ep_dp_degrees(world, 64)
    assert ep * dp == world


def test_an_explicit_ep_size_overrides_auto_resolution():
    assert _resolve_ep_size(moe_cfg(), world_size=8, ep_size=2) == 2


def test_auto_ep_is_one_for_a_model_that_cannot_shard_experts():
    assert _resolve_ep_size(cfg(num_experts=64), world_size=8, ep_size="auto") == 1


# ------------------------------------------------------------------------------- decided plan


def test_auto_plan_takes_the_largest_ep_degree_dividing_the_experts():
    plan = decide_distribution_plan(moe_cfg(experts=6), world_size=8)
    assert plan.ep_size == 2
    assert plan.edp == 4


def test_an_ep_degree_that_does_not_divide_the_world_is_rejected():
    with pytest.raises(ValueError, match="must divide world_size"):
        decide_distribution_plan(moe_cfg(), world_size=8, ep_size=3)


@pytest.mark.parametrize(("knob", "expected"), [("world", 8), ("full", 8), (4, 4), (99, 8), (0, 1)])
def test_dense_fsdp_knob_is_honoured_and_clamped_to_the_world(knob, expected):
    plan = decide_distribution_plan(moe_cfg(), world_size=8, fsdp_size=knob)
    assert plan.fsdp_size == expected


def test_unknown_dense_size_defaults_to_replication():
    """With no way to estimate the dense footprint, stay on the cheaper replicate path."""
    assert decide_distribution_plan(moe_cfg(), world_size=8).fsdp_size == 1


# ------------------------------------------------------------------------------- rank-0 decision


def _as_rank(monkeypatch, rank, rank0_decision=None):
    """Pretend to be ``rank`` of a multi-rank run; returns what this rank handed the broadcast.

    The fake broadcast delivers ``rank0_decision`` to a non-zero rank, as rank 0 would have sent it.
    """
    sent = []

    def fake_broadcast(objs, src):
        assert src == 0
        sent.append(objs[0])
        if rank != 0:
            objs[0] = rank0_decision

    monkeypatch.setattr(mto_dist, "is_distributed_run", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda group=None: rank)
    monkeypatch.setattr(dist, "broadcast_object_list", fake_broadcast)
    return sent


def test_rank0_decides_and_broadcasts_its_plan(monkeypatch):
    sent = _as_rank(monkeypatch, 0)
    plan = decide_distribution_plan(moe_cfg(), world_size=8, ep_size=4)
    assert sent == [(plan, None)]
    assert plan.ep_size == 4


def test_other_ranks_adopt_rank0s_plan_without_deciding(monkeypatch):
    """A non-zero rank's own inputs (here an EP knob of 2) must not leak into the plan."""
    rank0_plan = decide_distribution_plan(moe_cfg(), world_size=8, ep_size=4)
    _as_rank(monkeypatch, 1, rank0_decision=(rank0_plan, None))

    def must_not_decide(*args, **kwargs):
        raise AssertionError("a non-zero rank decided its own plan")

    monkeypatch.setattr(moe, "_decide_distribution_plan_on_rank0", must_not_decide)
    assert decide_distribution_plan(moe_cfg(), world_size=8, ep_size=2) is rank0_plan


def test_rank0_failure_is_broadcast_before_it_raises(monkeypatch):
    """Rank 0 must still reach the broadcast on failure, or every other rank hangs in it."""
    sent = _as_rank(monkeypatch, 0)
    with pytest.raises(ValueError, match="must divide world_size"):
        decide_distribution_plan(moe_cfg(), world_size=8, ep_size=3)
    assert sent[0][0] is None
    assert "ValueError" in sent[0][1]


def test_other_ranks_raise_when_rank0_failed(monkeypatch):
    _as_rank(monkeypatch, 1, rank0_decision=(None, "ValueError: bad degrees"))
    with pytest.raises(RuntimeError, match=r"rank 0 failed to decide.*bad degrees"):
        decide_distribution_plan(moe_cfg(), world_size=8)
