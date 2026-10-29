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
get wrong is structural: a tensor missing, duplicated, mis-assembled from its shards, renamed, or
left out of the weight index. These tests drive each piece directly, on CPU.
"""

import contextlib
import json
import math
import os
import types
from unittest import mock

import pytest
import torch
from _test_utils.torch.distributed.utils import bounded_collectives
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from torch.distributed.tensor.placement_types import _StridedShard
from torch.multiprocessing.reductions import StorageWeakRef

pytest.importorskip("safetensors")
pytest.importorskip("transformers")
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, LlamaConfig, MixtralConfig
from transformers.core_model_loading import Chunk, WeightConverter

from modelopt.torch.export.distributed_writer import (
    _even_bins,
    _is_full_world_shard0,
    _materialize_owned,
    _merge_groups,
    _nbytes,
    _revert_whole_sd,
    _write_even_shards,
)

_WRITER = "modelopt.torch.export.distributed_writer"


def _raw_bytes(tensor):
    return tensor.detach().contiguous().reshape(-1).view(torch.uint8)


def _assert_byte_identical(got, want):
    assert set(got) == set(want), (sorted(set(want) - set(got)), sorted(set(got) - set(want)))
    for key, ref in want.items():
        assert got[key].dtype == ref.dtype, key
        assert got[key].shape == ref.shape, key
        assert torch.equal(_raw_bytes(got[key]), _raw_bytes(ref.cpu())), key


def _load_checkpoint(export_dir):
    """Every tensor the index names, read from the shard the index names for it."""
    index = json.loads((export_dir / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    shards = sorted(p.name for p in export_dir.glob("*.safetensors"))
    assert set(weight_map.values()) == set(shards), "index and shard files disagree"
    tensors = {}
    for shard in shards:
        loaded = load_file(str(export_dir / shard))
        assert {k for k, f in weight_map.items() if f == shard} == set(loaded), shard
        tensors.update(loaded)
    assert index["metadata"]["total_size"] == sum(_nbytes(t) for t in tensors.values())
    return tensors


def _mixtral_config(**overrides):
    """A tiny Mixtral: fused experts in memory, per-expert ``block_sparse_moe`` names on the hub."""
    kwargs = {
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_local_experts": 4,
        "num_experts_per_tok": 2,
        "max_position_embeddings": 32,
        **overrides,
    }
    return MixtralConfig(**kwargs)


def _llama_config():
    # 63 vocab rows: the embedding and lm_head shard unevenly over two ranks.
    return LlamaConfig(
        vocab_size=63,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )


# ------------------------------------------------------------------ pure helpers


@pytest.mark.parametrize(
    ("sizes", "n", "file_sizes"),
    [
        # max_shard_size is a bound, not a target: two files of 8 rather than 10 + 6.
        ({"a": 4, "b": 3, "c": 3, "d": 2, "e": 2, "f": 2}, 2, [8, 8]),
        # One file per tensor once no two fit together.
        ({"a": 6, "b": 6, "c": 6}, 1, [6, 6, 6]),
        # A tensor over the bound gets a file of its own, and the rest still share one.
        ({"big": 100, "a": 1, "b": 1, "c": 1}, 1, [3, 100]),
        # Never an empty file, asked for more files than tensors or for none.
        ({"a": 1, "b": 1}, 5, [1, 1]),
        ({}, 3, []),
    ],
    ids=["balanced", "grows", "oversized", "more-files-than-tensors", "nothing"],
)
def test_even_bins_balances_the_files_within_the_bound(sizes, n, file_sizes):
    bins = _even_bins(list(sizes), sizes, n, 10)

    assert sorted(sum(sizes[k] for k in b) for b in bins) == file_sizes


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
        # Its shards interleave, so joining them in rank order would permute the rows.
        (
            lambda mesh: DTensor.from_local(
                torch.zeros(4, 3),
                mesh,
                [_StridedShard(0, split_factor=2)],
                run_check=False,
                shape=torch.Size([4, 3]),
                stride=(3, 1),
            ),
            "cannot materialize",
        ),
        # Empty, so there is nothing to assemble, but in a layout it would refuse a non-empty one in.
        (
            lambda mesh: distribute_tensor(torch.zeros(0, 4), mesh, [Shard(1)]),
            "cannot materialize",
        ),
    ],
    ids=["shard1", "2d-mesh", "padded", "strided-shard0", "empty-shard1"],
)
def test_materialize_owned_rejects_a_tensor_it_cannot_assemble(gloo_world_of_one, build, match):
    """Refused before anything moves, rather than assembled wrong."""
    with pytest.raises(RuntimeError, match=match):
        _materialize_owned({"w": build(gloo_world_of_one)}, set(), 0, 1)


# A whole tensor rank 1 holds unlike rank 0: the ranks would assign owners by different sizes, or at
# an equal size write the owner's version of it.
_PLAIN_DISAGREEMENTS = {
    "size": lambda rank: torch.zeros(2, dtype=torch.float32 if rank == 0 else torch.bfloat16),
    "same-size-dtype": lambda rank: torch.zeros(
        2, dtype=torch.float32 if rank == 0 else torch.int32
    ),
    "same-size-shape": lambda rank: torch.zeros((2, 4) if rank == 0 else (4, 2)),
    "empty-shape": lambda rank: torch.zeros((0, 4) if rank == 0 else (0, 2)),
}


def _refusing_materialize_job(rank, world_size, case):
    if case == "smaller-mesh":
        # A dim-0 shard over a mesh smaller than the world: each rank's own 1-rank sub-mesh.
        sub = init_device_mesh("cpu", (world_size, 1), mesh_dim_names=("dp", "sub"))["sub"]
        w = distribute_tensor(torch.zeros(4, 3), sub, [Shard(0)])
    elif case in _PLAIN_DISAGREEMENTS:
        w = _PLAIN_DISAGREEMENTS[case](rank)
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
        elif case in ("device", "devices"):
            # Only rank 1's shard on another device; or both ranks' shards of a second key.
            on_meta = case == "device" and rank == 1
            local, placement, gshape = (
                torch.zeros(2, 4, device="meta" if on_meta else "cpu"),
                Shard(0),
                (4, 4),
            )
        else:
            # An expert, which each rank writes as it holds it: only rank 1's is still a DTensor.
            local, placement, gshape = torch.zeros(2, 4), Shard(0), (4, 4)
        w = DTensor.from_local(
            local, mesh, [placement], run_check=False, shape=torch.Size(gshape), stride=(4, 1)
        )
        if case == "expert" and rank == 0:
            w = w.to_local()
    local_sd = {"w": w, "b": torch.zeros(1)} if case == "grouping" else {"w": w}
    if case == "devices":
        local_sd["m"] = DTensor.from_local(
            torch.zeros(2, 4, device="meta"),
            mesh,
            [Shard(0)],
            run_check=False,
            shape=(4, 4),
            stride=(4, 1),
        )
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
        "dtype": "different dtype or device",
        "device": "different dtype or device",
        "devices": "more than one device type",
        "expert": "still DTensors",
        "grouping": "group the keys",
        **dict.fromkeys(_PLAIN_DISAGREEMENTS, "different dtype or shape"),
    }.get(case, "cannot materialize")
    assert match in raised, raised


@pytest.mark.parametrize(
    "case",
    [
        "smaller-mesh",
        "placement",
        "global-shape",
        "dtype",
        "device",
        "devices",
        "expert",
        *_PLAIN_DISAGREEMENTS,
        "grouping",
    ],
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
    # Two merge groups sharing a key: one owner for all three, where their sizes alone would put x
    # on one rank and y and z on the other.
    grouped = {k: torch.randn(n, 4) for k, n in (("x", 6), ("y", 5), ("z", 4))}
    together = [["x", "y"], ["y", "z"]]
    # The copies this rank sends, and how many of them are still held at each assembled store; the
    # buffers it receives into, and how many of them are still held when the next round starts.
    sent, held, received, stale = [], [], [], []
    batch_isend_irecv = torch.distributed.batch_isend_irecv

    def recording_sends(ops):
        stale.append(sum(not ref.expired() for ref in received))
        for op in ops:
            mine = sent if op.op is torch.distributed.isend else received
            mine.append(StorageWeakRef(op.tensor.untyped_storage()))
        return batch_isend_irecv(ops)

    class CountingHeldSends(dict):
        def __setitem__(self, key, value):
            held.append(sum(not ref.expired() for ref in sent))
            super().__setitem__(key, value)

    for layout, groups in ((full, ()), (send_only, ()), (grouped, together)):
        sd = CountingHeldSends(
            {k: distribute_tensor(v, mesh, [Shard(0)]) for k, v in layout.items()}
        )

        with (
            bounded_collectives(),
            mock.patch.object(torch.distributed, "batch_isend_irecv", recording_sends),
            mock.patch(f"{_WRITER}._KEYS_PER_ROUND", 2),  # rounds of two keys
        ):
            _materialize_owned(sd, set(), rank, world_size, groups)

        for k, v in sd.items():
            assert not isinstance(v, DTensor), k
            assert torch.equal(v, layout[k]), k
        owned: list = [None] * world_size
        torch.distributed.all_gather_object(owned, sorted(sd))
        assert sorted(k for keys in owned for k in keys) == sorted(layout)
        if groups:
            assert sorted(layout) in owned, owned
    assert sent and held and not any(held), held
    assert received and len(stale) > 1 and not any(stale), stale


def test_materialize_owned_moves_each_shard_to_its_owner_over_two_ranks(cpu_dist_workers):
    """Each key ends whole on exactly one rank, its shards packed per peer and dtype, unpacked at
    their offsets and joined in rank order, an uneven and an empty shard included, and merge groups
    that share a key on one rank. What a rank sent is freed before it assembles what it owns, and what
    it received before the next round of transfers starts."""
    cpu_dist_workers.run(_materialize_two_ranks_job)


def test_revert_whole_sd_keeps_a_state_dict_with_nothing_to_revert():
    """A model without a conversion mapping gets its input dict back from the reverse pass."""
    model = AutoModelForCausalLM.from_config(_llama_config())
    sd = dict(model.state_dict())
    before = dict(sd)

    assert _revert_whole_sd(model, sd) is True

    assert sd == before


def test_revert_whole_sd_renames_to_hub_names_in_place():
    model = AutoModelForCausalLM.from_config(_mixtral_config())
    gate, w1, w3, w2 = (torch.full((2, 2), float(i)) for i in range(4))
    sd = {
        "model.layers.0.mlp.gate.weight": gate,
        "model.layers.0.mlp.experts.0.gate_proj.weight": w1,
        "model.layers.0.mlp.experts.0.up_proj.weight": w3,
        "model.layers.0.mlp.experts.0.down_proj.weight": w2,
    }
    same_dict = sd

    assert _revert_whole_sd(model, sd) is True

    assert sd is same_dict
    hub = "model.layers.0.block_sparse_moe"
    assert set(sd) == {
        f"{hub}.gate.weight",
        f"{hub}.experts.0.w1.weight",
        f"{hub}.experts.0.w3.weight",
        f"{hub}.experts.0.w2.weight",
    }
    assert torch.equal(sd[f"{hub}.experts.0.w1.weight"], w1)
    assert torch.equal(sd[f"{hub}.experts.0.w2.weight"], w2)


def _one_rank_cannot_revert_job(rank, world_size):
    # Rank 0's subset reverts; rank 1's fails, as a merge whose group ownership split does.
    model = AutoModelForCausalLM.from_config(_mixtral_config())
    sd = {"model.layers.0.mlp.experts.0.up_proj.weight": torch.ones(2)}
    before = dict(sd)
    failure = mock.patch(
        f"{_WRITER}.revert_weight_conversion_quant_aware",
        side_effect=RuntimeError("incomplete merge"),
    )
    with (
        bounded_collectives(),
        failure if rank == 1 else contextlib.nullcontext(),
        pytest.warns(UserWarning, match="rank 1: incomplete merge"),
    ):
        assert _revert_whole_sd(model, sd) is False

    assert sd == before


def test_revert_whole_sd_keeps_in_memory_names_on_every_rank_when_one_fails(cpu_dist_workers):
    """Rank 0 could revert its own subset, but a checkpoint mixing hub and in-memory names would
    load neither way."""
    cpu_dist_workers.run(_one_rank_cannot_revert_job)


def _merge_group_job(rank, world_size, case):
    # A checkpoint's fused qkv, which the model holds as three projections. Anchored to the weight,
    # the reverse's patterns do not match the projections' scales.
    anchor = "$" if case == "anchored" else ""
    qkv = WeightConverter(
        source_patterns="attn.qkv.weight",
        target_patterns=[f"attn.{part}_proj.weight{anchor}" for part in "qkv"],
        operations=[Chunk(dim=0)],
    )
    model = types.SimpleNamespace(_weight_conversions=[qkv])
    module = "blk.mlp" if case == "unmatched" else "blk.attn"
    # q is the largest key and k, v the smallest: placed one key at a time, the group would split
    # over the ranks (q and v on one, k on the other).
    full = {
        f"{module}.q_proj.weight": torch.full((6, 4), 1.0),
        f"{module}.k_proj.weight": torch.full((2, 4), 2.0),
        f"{module}.v_proj.weight": torch.full((2, 4), 3.0),
        "blk.o_proj.weight": torch.full((5, 4), 4.0),
    }
    if case in ("quantized", "anchored"):
        # Quantized weights the reverse cannot merge: the owner must see their scales to refuse.
        full.update({f"{module}.{part}_proj.weight_scale": torch.tensor(0.5) for part in "qkv"})
    sd = dict(full)
    with bounded_collectives():
        _materialize_owned(sd, set(), rank, world_size, _merge_groups(model, sd))
        if case == "merge":
            assert _revert_whole_sd(model, sd) is True
        elif case in ("quantized", "anchored"):
            with pytest.warns(UserWarning, match="cannot merge quantization state"):
                assert _revert_whole_sd(model, sd) is False
        else:
            with pytest.warns(UserWarning, match="matched no state-dict key"):
                assert _revert_whole_sd(model, sd) is False
    owned: list = [None] * world_size
    torch.distributed.all_gather_object(owned, dict(sd))
    written = {k: v for mine in owned for k, v in mine.items()}
    if case in ("quantized", "anchored"):
        # The rank holding the group, the one that refuses it, also holds the weights' scales.
        group = {
            f"{module}.{part}_proj.{leaf}" for part in "qkv" for leaf in ("weight", "weight_scale")
        }
        assert any(group <= set(mine) for mine in owned), [sorted(mine) for mine in owned]
    if case == "merge":
        assert sorted(written) == ["blk.attn.qkv.weight", "blk.o_proj.weight"]
        want = torch.cat([full[f"blk.attn.{part}_proj.weight"] for part in "qkv"])
        assert torch.equal(written["blk.attn.qkv.weight"], want)
    else:
        assert sorted(written) == sorted(full)


@pytest.mark.parametrize("case", ["merge", "unmatched", "quantized", "anchored"])
def test_revert_whole_sd_merges_a_group_ownership_keeps_on_one_rank(cpu_dist_workers, case):
    """A merge group goes to one owner, which merges it while the other rank skips the rule, as the
    gather export merges it. A rule no rank's keys match is refused, as the gather export refuses
    it, and so is a quantized group, whose scales go to its owner with it, also when the merge's
    patterns are anchored to the weights and do not match the scales."""
    cpu_dist_workers.run(_merge_group_job, case=case)


def test_write_even_shards_writes_even_shards_and_an_index_of_every_tensor(
    gloo_world_of_one, tmp_path
):
    torch.manual_seed(0)
    tensors = {f"model.layers.{i}.weight": torch.randn(16, 8 + 4 * i) for i in range(10)}
    tensors["model.layers.0.weight_scale"] = torch.tensor(0.5)
    tensors["model.layers.1.weight"] = torch.randn(16, 16).to(torch.float8_e4m3fn)
    tensors["model.norm.weight"] = torch.randn(16, dtype=torch.bfloat16)
    tensors["lm_head.weight"] = torch.randn(16, 8).t()  # not contiguous
    max_bytes = 4096
    assert max(_nbytes(t) for t in tensors.values()) <= max_bytes
    local = dict(tensors)

    _write_even_shards(local, tmp_path, "4096", 0, 1)

    assert not local, "a written tensor must be dropped from the rank's set"
    shards = sorted(p.name for p in tmp_path.glob("*.safetensors"))
    n = len(shards)
    assert n == math.ceil(sum(_nbytes(t) for t in tensors.values()) / max_bytes)
    assert shards == [f"model-{i:05d}-of-{n:05d}.safetensors" for i in range(1, n + 1)]
    _assert_byte_identical(_load_checkpoint(tmp_path), tensors)
    for shard in shards:
        assert sum(_nbytes(t) for t in load_file(str(tmp_path / shard)).values()) <= max_bytes


def test_write_even_shards_replaces_a_checkpoint_already_there(gloo_world_of_one, tmp_path):
    """A re-export into the directory replaces the shards and index it finds there."""
    _write_even_shards({"w": torch.zeros(2)}, tmp_path, "1MB", 0, 1)
    tensors = {"w": torch.ones(2)}

    _write_even_shards(dict(tensors), tmp_path, "1MB", 0, 1)

    _assert_byte_identical(_load_checkpoint(tmp_path), tensors)
    assert not list(tmp_path.glob("*.tmp"))


def test_write_even_shards_fails_loudly_when_no_rank_has_anything_to_write(
    gloo_world_of_one, tmp_path
):
    with pytest.raises(RuntimeError, match="produced no shards"):
        _write_even_shards({}, tmp_path, "1MB", 0, 1)


def test_a_failed_write_keeps_its_error_when_the_cleanup_fails_too(gloo_world_of_one, tmp_path):
    """The filesystem that failed the write can fail the cleanup too: the write's error is the one
    raised, not the cleanup's, and the file the cleanup could not remove is named in a warning."""
    with (
        mock.patch(f"{_WRITER}.save_file", side_effect=OSError("disk full")),
        mock.patch("pathlib.Path.unlink", side_effect=OSError("read-only file system")),
        pytest.warns(UserWarning, match="could not remove 1 temporary file"),
        pytest.raises(OSError, match="disk full"),
    ):
        _write_even_shards({"w": torch.ones(2)}, tmp_path, "1MB", 0, 1)


def test_write_even_shards_drops_each_tensor_before_writing_its_file(gloo_world_of_one, tmp_path):
    """A rank drops its own tensor as soon as the host copy exists, so a file's tensors are not
    held on the device while the file is written."""
    local = {f"w{i}": torch.randn(16, 8) for i in range(4)}
    still_held = []

    def spy(tensors, *args, **kwargs):
        still_held.extend(set(tensors) & set(local))
        return save_file(tensors, *args, **kwargs)

    with mock.patch(f"{_WRITER}.save_file", side_effect=spy):
        _write_even_shards(local, tmp_path, "1100", 0, 1)

    assert not still_held and not local


def test_write_even_shards_writes_both_names_of_a_shared_tensor(gloo_world_of_one, tmp_path):
    """Both names of one CPU tensor are written, though ``save_file`` refuses shared memory."""
    shared = torch.arange(4.0)

    _write_even_shards({"a": shared, "b": shared}, tmp_path, "1MB", 0, 1)

    _assert_byte_identical(_load_checkpoint(tmp_path), {"a": shared, "b": shared})


def _write_two_ranks_job(rank, world_size, tmp_dir):
    torch.manual_seed(rank)
    # Two of these 512-byte tensors fit a file, so rank 0 writes one file and rank 1 two: a file
    # number off by a rank's count then lands on another rank's file, or past the total.
    mine = {f"rank{rank}.w{i}": torch.randn(16, 8) for i in range(2 + rank)}
    everyone: list = [None] * world_size
    with bounded_collectives():
        torch.distributed.all_gather_object(everyone, mine)
        _write_even_shards(dict(mine), tmp_dir, "1100", rank, world_size)

    _assert_byte_identical(
        _load_checkpoint(tmp_dir), {k: v for m in everyone for k, v in m.items()}
    )


def test_write_even_shards_numbers_two_ranks_files_into_one_checkpoint(cpu_dist_workers, tmp_path):
    """Each rank's files are numbered after those of the ranks before it, and the index rank 0
    writes names every rank's tensors."""
    cpu_dist_workers.run(_write_two_ranks_job, tmp_dir=tmp_path)

    shards = sorted(p.name for p in tmp_path.glob("*.safetensors"))
    assert shards == [f"model-{i:05d}-of-00003.safetensors" for i in (1, 2, 3)]


def _one_rank_fails_job(rank, world_size, tmp_dir, fails):
    failing_rank, target = {
        "dir": (1, "pathlib.Path.mkdir"),
        "shard": (1, f"{_WRITER}.save_file"),
        "index": (0, "json.dump"),
    }[fails]
    failure = (
        mock.patch(target, side_effect=OSError("disk full"))
        if rank == failing_rank
        else contextlib.nullcontext()
    )
    try:
        with bounded_collectives(), failure:
            _write_even_shards(
                {f"rank{rank}.weight": torch.ones(2)}, tmp_dir, "1MB", rank, world_size
            )
    except (OSError, RuntimeError) as error:
        raised = error
    else:
        raised = None
    # The failing rank raises its own error; its peer gets that error, naming the rank, instead of
    # waiting for it.
    if rank == failing_rank:
        assert isinstance(raised, OSError), raised
    else:
        assert isinstance(raised, RuntimeError), raised
        assert f"rank {failing_rank}" in str(raised) and "disk full" in str(raised), raised


def _existing_checkpoint(export_dir):
    """A checkpoint under the names the two-rank export below writes, holding other tensors."""
    weight_map = {
        "old.a": "model-00001-of-00002.safetensors",
        "old.b": "model-00002-of-00002.safetensors",
    }
    for key, shard in weight_map.items():
        save_file({key: torch.zeros(3)}, str(export_dir / shard))
    index = {"metadata": {"total_size": 24}, "weight_map": weight_map}
    (export_dir / "model.safetensors.index.json").write_text(json.dumps(index))
    return {p.name: p.read_bytes() for p in export_dir.iterdir()}


@pytest.mark.parametrize("existing", [False, True], ids=["empty", "over-a-checkpoint"])
@pytest.mark.parametrize("fails", ["dir", "shard", "index"])
def test_a_write_that_fails_on_one_rank_fails_on_every_rank(
    cpu_dist_workers, tmp_path, fails, existing
):
    """Not just on the rank that failed, while its peers wait in a collective it never joins; and
    the directory is left as it was: no file of the failed export, and a checkpoint already there
    byte for byte, though the export would have written its names."""
    before = _existing_checkpoint(tmp_path) if existing else {}

    cpu_dist_workers.run(_one_rank_fails_job, tmp_dir=tmp_path, fails=fails)

    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def _shard_commit_fails_job(rank, world_size, tmp_dir):
    # Every file is written; rank 1 then fails to rename its shard into place.
    replace = os.replace

    def failing_shard_rename(src, dst):
        if str(dst).endswith(".safetensors"):
            raise OSError("disk full")
        replace(src, dst)

    failure = (
        mock.patch(f"{_WRITER}.os.replace", side_effect=failing_shard_rename)
        if rank == 1
        else contextlib.nullcontext()
    )
    with bounded_collectives(), failure, pytest.raises((OSError, RuntimeError), match="disk full"):
        _write_even_shards({f"rank{rank}.weight": torch.ones(2)}, tmp_dir, "1MB", rank, world_size)


@pytest.mark.parametrize("existing", [False, True], ids=["empty", "over-a-checkpoint"])
def test_the_index_goes_in_place_only_after_every_shard(cpu_dist_workers, tmp_path, existing):
    """A rank that fails to put its shard in place leaves the index as it was, so an index in the
    directory never names shards that are missing."""
    index = tmp_path / "model.safetensors.index.json"
    before = _existing_checkpoint(tmp_path)[index.name] if existing else None

    cpu_dist_workers.run(_shard_commit_fails_job, tmp_dir=tmp_path)

    assert (index.read_bytes() if index.exists() else None) == before


def _cross_rank_collision_job(rank, world_size, tmp_dir):
    try:
        with bounded_collectives():
            _write_even_shards(
                {"model.norm.weight": torch.ones(2)}, tmp_dir, "1MB", rank, world_size
            )
    except RuntimeError as error:
        raised = str(error)
    else:
        raised = "no error"
    assert "name collision across ranks" in raised, raised


def test_a_name_two_ranks_both_write_fails_the_write_on_every_rank(cpu_dist_workers, tmp_path):
    """Before any rank writes a shard, which a retry into the directory would otherwise inherit."""
    cpu_dist_workers.run(_cross_rank_collision_job, tmp_dir=tmp_path)

    assert not list(tmp_path.glob("*.safetensors"))
