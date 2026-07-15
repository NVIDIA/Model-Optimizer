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

"""Parallelism-aware reduction primitives for AutoModel activation scoring.

These run on a 4-rank gloo/CPU process group arranged as a 2x2 (dp x tp) logical
mesh, and assert that a SUM-reduce over the token group composed with a GATHER
over the scored (feature) axis reproduces the single-process reference — i.e. the
final scores are independent of the parallel layout. No GPU required.
"""

import torch
import torch.distributed as dist
from _test_utils.torch.distributed.utils import spawn_multiprocess_job
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

from modelopt.torch.puzzletron.plugins.automodel.hooks.moe import (
    MoEExpertRemovalDiffScorer,
    MoEGroupedExpertChannelScorer,
    MoELatentCalibrationScorer,
    _expert_owner_range,
    _gather_ep_inputs,
)
from modelopt.torch.puzzletron.plugins.automodel.reduction import (
    MeshGroups,
    finalize_additive,
    gather_scored_axis,
    is_writer,
    reduce_token_sum,
    to_local_with_feature_group,
)

# Logical layout for all jobs: world size 4, tp_size 2, dp_size 2.
# global rank r -> dp = r // 2, tp = r % 2  (row-major, matches init_device_mesh)
#   tp groups (fixed dp): {0,1}, {2,3}        -> shard the scored/feature axis
#   token groups (fixed tp): {0,2}, {1,3}     -> shard the token axis (dp)
_TP_SIZE = 2
_N_TOKENS = 8
_N_CHANNELS = 6


def _build_2x2_groups(rank):
    """Create the 2x2 (dp x tp) process groups (collective; all ranks, same order)."""
    dp = rank // _TP_SIZE
    tp = rank % _TP_SIZE
    tp_groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
    token_groups = [dist.new_group([0, 2]), dist.new_group([1, 3])]
    groups = MeshGroups(token_group=token_groups[tp], tp_group=tp_groups[dp])
    return groups, dp, tp


def _reference():
    """Deterministic full activation, identical on every rank (fixed seed)."""
    torch.manual_seed(0)
    return torch.randn(_N_TOKENS, _N_CHANNELS, dtype=torch.float64)


def _job_additive(rank, size):
    assert size == 4
    groups, dp, tp = _build_2x2_groups(rank)

    # reduce_token_sum: each rank contributes (token_rank + 1); the 2-rank token
    # group sums to 1 + 2 = 3 regardless of which rank we are.
    val = torch.tensor([float(groups.token_rank) + 1.0], dtype=torch.float64)
    reduce_token_sum(val, groups.token_group)
    assert torch.allclose(val, torch.tensor([3.0], dtype=torch.float64)), val

    # gather_scored_axis: concat in tp-rank order -> [0, 1].
    gathered = gather_scored_axis(
        torch.tensor([float(groups.tp_rank)], dtype=torch.float64), groups.tp_group
    )
    assert torch.allclose(gathered, torch.tensor([0.0, 1.0], dtype=torch.float64)), gathered

    # finalize_additive: token SUM-reduce + scored GATHER == single-process ref.
    full = _reference()
    local = full[dp * 4 : (dp + 1) * 4, tp * 3 : (tp + 1) * 3].clone()  # [4, 3]
    partial = local.abs().sum(dim=0)  # per-local-channel sum over local tokens -> [3]
    out = finalize_additive(partial, feature_group=groups.tp_group, groups=groups)
    expected = full.abs().sum(dim=0)  # [6]
    assert out.shape == (_N_CHANNELS,), out.shape
    assert torch.allclose(out, expected), (out, expected)

    # single-writer rule: exactly one writer across the world (global rank 0).
    writer_flag = torch.tensor([int(is_writer(groups))], dtype=torch.int64)
    dist.all_reduce(writer_flag, op=dist.ReduceOp.SUM)
    assert writer_flag.item() == 1, writer_flag.item()
    assert is_writer(groups) == (rank == 0)


def _job_dtensor(rank, size):
    assert size == 4
    mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))
    groups = MeshGroups.from_device_mesh(mesh)
    assert groups.token_group is not None, "token group ('dp') should resolve"
    assert groups.tp_group is not None, "tp group should resolve"

    full = _reference()
    # Shard rows over dp (token axis) and cols over tp (feature axis), as the
    # down_proj input would be under TP (feature-sharded) + data parallel.
    dtensor = distribute_tensor(full, mesh, [Shard(0), Shard(1)])
    local, feature_group = to_local_with_feature_group(dtensor, feature_dim=-1)
    assert local.shape == (4, 3), local.shape
    assert feature_group is not None, "feature dim is sharded -> group must be returned"

    partial = local.abs().sum(dim=0)  # [3]
    out = finalize_additive(partial, feature_group=feature_group, groups=groups)
    expected = full.abs().sum(dim=0)  # [6]
    assert torch.allclose(out, expected), (out, expected)


def _job_no_parallel(rank, size):
    # No groups -> reductions/gathers are no-ops; a plain tensor passes through.
    assert size == 1
    groups = MeshGroups()
    full = _reference()
    partial = full.abs().sum(dim=0)
    local, feature_group = to_local_with_feature_group(partial)  # plain tensor
    assert feature_group is None
    out = finalize_additive(local, feature_group=feature_group, groups=groups)
    assert torch.allclose(out, full.abs().sum(dim=0))
    assert is_writer(groups) is True


def _build_ep_shard_groups(rank):
    class _Axis:
        def __init__(self, group):
            self._group = group

        def get_group(self):
            return self._group

    class _Mesh:
        def __init__(self, names, groups):
            self.mesh_dim_names = names
            self._groups = groups

        def __getitem__(self, name):
            return _Axis(self._groups[name])

    root_group = dist.new_group([0, 1, 2, 3])
    token_groups = [dist.new_group([0, 2]), dist.new_group([1, 3])]
    ep_groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
    root_mesh = _Mesh(("dp_shard",), {"dp_shard": root_group})
    moe_mesh = _Mesh(
        ("ep_shard", "ep"),
        {"ep_shard": token_groups[rank % 2], "ep": ep_groups[rank // 2]},
    )
    return MeshGroups.from_device_mesh(root_mesh, moe_mesh=moe_mesh)


def _job_ep_shard_groups(rank, size):
    """EP gathers experts while its orthogonal ep_shard axis reduces samples."""
    assert size == 4
    groups = _build_ep_shard_groups(rank)

    expected_token_ranks = [0, 2] if rank % 2 == 0 else [1, 3]
    expected_ep_ranks = [0, 1] if rank < 2 else [2, 3]
    assert dist.get_process_group_ranks(groups.token_group) == expected_token_ranks
    assert dist.get_process_group_ranks(groups.ep_shard_group) == expected_token_ranks
    assert dist.get_process_group_ranks(groups.ep_group) == expected_ep_ranks
    # Scoring hooks observe GroupedExperts' arguments before its native EP
    # all-gather, so these inputs remain distinct even though the MoE mesh has
    # an orthogonal ep_shard axis.
    assert groups.ep_inputs_replicated is False

    gathered_inputs = _gather_ep_inputs(
        torch.tensor([rank], dtype=torch.long), groups
    )
    torch.testing.assert_close(
        gathered_inputs,
        torch.tensor(expected_ep_ranks, dtype=torch.long),
    )

    token_sum = torch.tensor([float(groups.token_rank + 1)], dtype=torch.float64)
    reduce_token_sum(token_sum, groups.token_group)
    torch.testing.assert_close(token_sum, torch.tensor([3.0], dtype=torch.float64))

    experts = gather_scored_axis(
        torch.tensor([float(groups.ep_rank)], dtype=torch.float64), groups.ep_group
    )
    torch.testing.assert_close(experts, torch.tensor([0.0, 1.0], dtype=torch.float64))

    writer_flag = torch.tensor([int(is_writer(groups))], dtype=torch.int64)
    dist.all_reduce(writer_flag, op=dist.ReduceOp.SUM)
    assert writer_flag.item() == 1


def _job_moe_ep_finalizers(rank, size):
    """MoE scorers reduce samples over ep_shard, then gather the expert axis."""
    assert size == 4
    groups = _build_ep_shard_groups(rank)
    token_value = float(groups.token_rank + 1)
    expert_value = float(groups.ep_rank + 1)

    removal = MoEExpertRemovalDiffScorer.__new__(MoEExpertRemovalDiffScorer)
    removal.groups = groups
    removal._mse = torch.tensor([token_value * expert_value], dtype=torch.float64)
    removal._cosine = 2 * removal._mse
    removal._displaced = torch.tensor([token_value], dtype=torch.float64)
    removal._denom = torch.tensor([token_value], dtype=torch.float64)
    removal_result = removal.finalize()
    torch.testing.assert_close(removal_result["score"], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(
        removal_result["cosine_diffs"], torch.tensor([2.0, 4.0])
    )
    torch.testing.assert_close(
        removal_result["num_tokens_displaced"], torch.tensor([3.0, 3.0])
    )

    grouped = MoEGroupedExpertChannelScorer.__new__(MoEGroupedExpertChannelScorer)
    grouped.groups = groups
    grouped.num_experts = 2
    grouped.local_start = groups.ep_rank
    grouped.local_experts = 1
    grouped.intermediate = 2
    grouped.pruning_iters = 1
    grouped.curr_iter = 0
    grouped.schedule = [2]
    grouped._pruned = torch.zeros(1, 2, dtype=torch.bool)
    grouped._agg = torch.zeros(1, 2, dtype=torch.float32)
    grouped._last_score = torch.full((1, 2), torch.inf)
    grouped._prune_debt = torch.zeros(1, dtype=torch.long)
    local_score = (
        torch.tensor([[1.0, 2.0]])
        if groups.ep_rank == 0
        else torch.tensor([[2.0, 1.0]])
    )
    grouped._pending = local_score.double() * token_value
    grouped._pending_count = torch.tensor([token_value], dtype=torch.float64)
    grouped._orders = [[]]
    grouped.step_iteration()
    grouped_result = grouped.finalize()
    torch.testing.assert_close(
        grouped_result["score"], torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    )

    latent = MoELatentCalibrationScorer.__new__(MoELatentCalibrationScorer)
    latent.groups = groups
    latent._tokens_per_batch = 1
    ep_writer_value = token_value if groups.ep_rank == 0 else 0.0
    latent._latent_cov_in_sum = torch.tensor([[ep_writer_value]], dtype=torch.float64)
    latent._latent_cov_in_n = torch.tensor(ep_writer_value, dtype=torch.float64)
    latent._expert_weights_sum = torch.tensor(
        [token_value * expert_value], dtype=torch.float64
    )
    latent._latent_cov_out_sum = torch.tensor(
        [[token_value * expert_value]], dtype=torch.float64
    )
    latent._latent_cov_out_weight = torch.tensor(
        token_value * expert_value, dtype=torch.float64
    )
    latent_result = latent.finalize()
    torch.testing.assert_close(latent_result["latent_cov_in"], torch.ones(1, 1).double())
    torch.testing.assert_close(
        latent_result["expert_weights_sum"], torch.tensor([3.0, 6.0]).double()
    )
    torch.testing.assert_close(
        latent_result["latent_cov_out"], torch.ones(1, 1).double()
    )


def _job_expert_ownership_ignores_orthogonal_fsdp_mesh_dim(rank, size):
    assert size == 4
    mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("ep_shard", "ep"))
    groups = MeshGroups.from_device_mesh(mesh, moe_mesh=mesh)
    weights = distribute_tensor(
        torch.zeros(4, 4, 2), mesh, placements=(Shard(1), Shard(0))
    )

    class _Experts:
        gate_and_up_projs = weights

    start, end = _expert_owner_range(_Experts(), groups, num_experts=4)
    assert (start, end) == (groups.ep_rank * 2, (groups.ep_rank + 1) * 2)
    assert isinstance(weights.placements[0], Shard)
    assert isinstance(weights.placements[1], Shard)


def test_reduction_additive_primitives():
    spawn_multiprocess_job(size=4, job=_job_additive, backend="gloo")


def test_reduction_with_dtensor():
    spawn_multiprocess_job(size=4, job=_job_dtensor, backend="gloo")


def test_reduction_single_process_noop():
    spawn_multiprocess_job(size=1, job=_job_no_parallel, backend="gloo")


def test_ep_scored_axis_is_orthogonal_to_token_reduction():
    spawn_multiprocess_job(size=4, job=_job_ep_shard_groups, backend="gloo")


def test_moe_finalizers_aggregate_over_ep_shard_then_gather_experts():
    spawn_multiprocess_job(size=4, job=_job_moe_ep_finalizers, backend="gloo")


def test_expert_ownership_uses_ep_group_with_multidimensional_dtensor_mesh():
    spawn_multiprocess_job(
        size=4, job=_job_expert_ownership_ignores_orthogonal_fsdp_mesh_dim, backend="gloo"
    )
