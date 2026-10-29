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

"""The pieces of the no-gather HF checkpoint writer (``modelopt.torch.export.distributed_writer``).

The writer moves every tensor to exactly one owner rank and writes it from there, so what it can
get wrong is structural: a tensor missing, duplicated or mis-assembled from its shards. These tests
drive each piece directly, on CPU.
"""

from unittest import mock

import pytest
import torch
from _test_utils.torch.distributed.utils import bounded_collectives
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from torch.multiprocessing.reductions import StorageWeakRef

pytest.importorskip("safetensors")
pytest.importorskip("transformers")

from modelopt.torch.export.distributed_writer import (
    _is_full_world_shard0,
    _materialize_owned,
    _nbytes,
)

# ------------------------------------------------------------------ pure helpers


def test_nbytes_counts_the_global_size_of_a_dtensor(gloo_world_of_one):
    # This rank holds 2 of 6 rows; every rank must agree on the size, so all 6 count.
    shard = DTensor.from_local(
        torch.zeros(2, 4),
        gloo_world_of_one,
        [Shard(0)],
        run_check=False,
        shape=torch.Size([6, 4]),
        stride=(4, 1),
    )
    assert _nbytes(shard) == 6 * 4 * 4


def test_is_full_world_shard0_answers_from_the_mesh_memo(gloo_world_of_one):
    tensor = distribute_tensor(torch.zeros(4, 3), gloo_world_of_one, [Shard(0)])

    memo = {}
    assert _is_full_world_shard0(tensor, 1, memo) is True
    assert memo == {id(gloo_world_of_one): True}
    # A cached answer is used as is: the mesh is not read back again.
    assert _is_full_world_shard0(tensor, 1, {id(gloo_world_of_one): False}) is False


def test_materialize_owned_leaves_every_tensor_whole_and_plain(gloo_world_of_one):
    mesh = gloo_world_of_one
    full = torch.arange(12.0).reshape(4, 3)
    expert_key, expert = "model.layers.0.mlp.experts.0.up_proj.weight", torch.full((2,), 7.0)
    sd = {
        "plain": torch.ones(3),
        "replicated": distribute_tensor(full, mesh, [Replicate()]),
        "sharded": distribute_tensor(full, mesh, [Shard(0)]),
        "empty": distribute_tensor(torch.zeros(0, 4, dtype=torch.bfloat16), mesh, [Shard(0)]),
        expert_key: expert,
    }

    _materialize_owned(sd, {expert_key}, 0, 1)

    assert set(sd) == {"plain", "replicated", "sharded", "empty", expert_key}
    assert not any(isinstance(v, DTensor) for v in sd.values())
    assert torch.equal(sd["plain"], torch.ones(3))
    assert torch.equal(sd["replicated"], full)
    assert torch.equal(sd["sharded"], full)
    assert sd["empty"].shape == (0, 4) and sd["empty"].dtype == torch.bfloat16
    assert sd[expert_key] is expert  # experts are already disjoint per rank: left alone


def _padded_shard(mesh):
    """This rank's 2 rows of a 3-row tensor: a padded local chunk, which would concatenate wrong."""
    return DTensor.from_local(
        torch.zeros(2, 4),
        mesh,
        [Shard(0)],
        run_check=False,
        shape=torch.Size([3, 4]),
        stride=(4, 1),
    )


@pytest.mark.parametrize(
    ("build", "match"),
    [
        (
            lambda mesh: distribute_tensor(torch.zeros(4, 4), mesh, [Shard(1)]),
            "cannot materialize",
        ),
        # A dim-0 shard over a 2-D mesh; a smaller mesh is a two-rank case below.
        (
            lambda mesh: distribute_tensor(
                torch.zeros(4, 3), init_device_mesh("cpu", (1, 1)), [Shard(0), Replicate()]
            ),
            "cannot materialize",
        ),
        (_padded_shard, "do not tile their global shape"),
        # Empty, so there is nothing to assemble, but in a layout it would refuse a non-empty one in.
        (
            lambda mesh: distribute_tensor(torch.zeros(0, 4), mesh, [Shard(1)]),
            "cannot materialize",
        ),
    ],
    ids=["shard1", "2d-mesh", "padded", "empty-shard1"],
)
def test_materialize_owned_rejects_a_tensor_it_cannot_assemble(gloo_world_of_one, build, match):
    """Refused before anything moves, rather than assembled wrong."""
    with pytest.raises(RuntimeError, match=match):
        _materialize_owned({"w": build(gloo_world_of_one)}, set(), 0, 1)


def _refusing_materialize_job(rank, world_size, case):
    if case == "smaller-mesh":
        # A dim-0 shard over a mesh smaller than the world: each rank's own 1-rank sub-mesh.
        sub = init_device_mesh("cpu", (world_size, 1), mesh_dim_names=("dp", "sub"))["sub"]
        w = distribute_tensor(torch.zeros(4, 3), sub, [Shard(0)])
    elif case == "size":
        # A plain buffer rank 1 holds in another dtype: the ranks would assign owners by different sizes.
        w = torch.zeros(2, dtype=torch.float32 if rank == 0 else torch.bfloat16)
    elif case == "grouping":
        # Only rank 1 keeps w with another key: the ranks would assign owners by different units.
        w = torch.zeros(2)
    else:
        mesh = init_device_mesh("cpu", (world_size,))
        if case == "placement":
            # Only rank 1 holds a placement the export cannot materialize.
            local, placement = (
                (torch.zeros(2, 4), Shard(0)) if rank == 0 else (torch.zeros(4, 2), Shard(1))
            )
            gshape = (4, 4)
        elif case == "dtype":
            # Rank 1 cast its shard: the transfer's byte counts would not match.
            dtype = torch.float32 if rank == 0 else torch.bfloat16
            local, placement, gshape = torch.zeros(2, 4, dtype=dtype), Shard(0), (4, 4)
        elif case == "global-shape":
            # A rewrap that re-inferred the global shape per rank: rank 0's own view tiles, rank 1's not.
            local, placement, gshape = torch.zeros(2, 4), Shard(0), (4 if rank == 0 else 3, 4)
        else:
            # An expert, which each rank writes as it holds it: only rank 1's is still a DTensor.
            local, placement, gshape = torch.zeros(2, 4), Shard(0), (4, 4)
        w = DTensor.from_local(
            local, mesh, [placement], run_check=False, shape=torch.Size(gshape), stride=(4, 1)
        )
        if case == "expert" and rank == 0:
            w = w.to_local()
    local_sd = {"w": w, "b": torch.zeros(1)} if case == "grouping" else {"w": w}
    together = [["w", "b"]] if case == "grouping" and rank == 1 else ()
    try:
        with bounded_collectives():
            _materialize_owned(
                local_sd, {"w"} if case == "expert" else set(), rank, world_size, together
            )
    except RuntimeError as error:
        raised = str(error)
    else:
        raised = "no error"
    match = {
        "global-shape": "do not tile their global shape",
        "dtype": "different dtype",
        "expert": "still DTensors",
        "size": "different size",
        "grouping": "group the keys",
    }.get(case, "cannot materialize")
    assert match in raised, raised


@pytest.mark.parametrize(
    "case", ["smaller-mesh", "placement", "global-shape", "dtype", "expert", "size", "grouping"]
)
def test_materialize_owned_refuses_on_every_rank(cpu_dist_workers, case):
    """Every rank refuses, the ranks whose own view looks fine included, instead of waiting in a
    collective the refusing rank never joins."""
    cpu_dist_workers.run(_refusing_materialize_job, case=case)


def _materialize_two_ranks_job(rank, world_size):
    mesh = init_device_mesh("cpu", (world_size,))
    torch.manual_seed(0)
    full = {
        "a.weight": torch.randn(7, 4),  # uneven: 4 + 3 rows
        "b.weight": torch.randn(8, 3).to(torch.bfloat16),  # a second dtype between the same ranks
        "c.weight": torch.randn(2, 5),
        "d.weight": torch.randn(1, 6),  # rank 1's shard is empty
    }
    # Rank 0 owns the larger and holds all of its rows: it assembles having received nothing, so
    # what it sent is all it could still be holding.
    send_only = {"e.weight": torch.randn(1, 60), "f.weight": torch.randn(7, 4)}
    # The copies this rank sends, and how many of them are still held at each assembled store.
    sent, held = [], []
    batch_isend_irecv = torch.distributed.batch_isend_irecv

    def recording_sends(ops):
        isends = [op for op in ops if op.op is torch.distributed.isend]
        sent.extend(StorageWeakRef(op.tensor.untyped_storage()) for op in isends)
        return batch_isend_irecv(ops)

    class CountingHeldSends(dict):
        def __setitem__(self, key, value):
            held.append(sum(not ref.expired() for ref in sent))
            super().__setitem__(key, value)

    for layout in (full, send_only):
        sd = CountingHeldSends(
            {k: distribute_tensor(v, mesh, [Shard(0)]) for k, v in layout.items()}
        )

        with (
            bounded_collectives(),
            mock.patch.object(torch.distributed, "batch_isend_irecv", recording_sends),
        ):
            _materialize_owned(sd, set(), rank, world_size)

        for k, v in sd.items():
            assert not isinstance(v, DTensor), k
            assert torch.equal(v, layout[k]), k
        owned: list = [None] * world_size
        torch.distributed.all_gather_object(owned, sorted(sd))
        assert sorted(k for keys in owned for k in keys) == sorted(layout)
    assert sent and held and not any(held), held


def test_materialize_owned_moves_each_shard_to_its_owner_over_two_ranks(cpu_dist_workers):
    """Each key ends whole on exactly one rank, its shards packed per peer and dtype, unpacked at
    their offsets and joined in rank order, an uneven and an empty shard included. What a rank
    sent is freed before it assembles what it owns."""
    cpu_dist_workers.run(_materialize_two_ranks_job)
