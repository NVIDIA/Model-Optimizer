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


def test_reduction_additive_primitives():
    spawn_multiprocess_job(size=4, job=_job_additive, backend="gloo")


def test_reduction_with_dtensor():
    spawn_multiprocess_job(size=4, job=_job_dtensor, backend="gloo")


def test_reduction_single_process_noop():
    spawn_multiprocess_job(size=1, job=_job_no_parallel, backend="gloo")
