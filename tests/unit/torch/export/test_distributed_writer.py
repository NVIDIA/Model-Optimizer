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

"""The no-gather FSDP2 HF checkpoint writer (``modelopt.torch.export.distributed_writer``) on CPU.

The writer moves every tensor to exactly one owner rank and writes it from there, so what it can
get wrong is structural: a tensor missing, duplicated, mis-assembled from its shards, renamed, or
left out of the weight index. The helpers are tested directly; the writer itself runs end to end on
a gloo process group -- one rank in process, two spawned ranks where shards actually travel -- and
the written checkpoint is reloaded and compared with the source checkpoint or with what the
single-process export writes for the same model.
"""

import contextlib
import copy
import json
import math
import os
import types
from unittest import mock

import pytest
import torch
import torch.distributed as dist
from _test_utils.torch.distributed.utils import bounded_collectives
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from torch.distributed.tensor.placement_types import _StridedShard
from torch.multiprocessing.reductions import StorageWeakRef

pytest.importorskip("safetensors")
pytest.importorskip("transformers")
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, LlamaConfig, MixtralConfig, Qwen2MoeConfig
from transformers.core_model_loading import Chunk, WeightConverter

import modelopt.torch.quantization as mtq
from modelopt.torch.export.distributed_writer import (
    _even_bins,
    _is_full_world_shard0,
    _materialize_owned,
    _merge_groups,
    _nbytes,
    _revert_whole_sd,
    _write_even_shards,
    distributed_save_hf_checkpoint,
)
from modelopt.torch.export.quant_aware_conversion import revert_weight_conversion_quant_aware
from modelopt.torch.export.unified_export_hf import (
    _export_transformers_checkpoint,
    export_hf_checkpoint,
)
from modelopt.torch.utils.distributed import DistributionPlan, set_distribution_plan


def _cpu_device_query():
    """The writer asks CUDA which device to all-reduce MoE expert input scales on.

    Nothing written in this file has expert input scales, so on a CPU-only host any answer works.
    """
    return mock.patch.object(torch.cuda, "current_device", return_value=0)


def _write(model, export_dir, **kwargs):
    with _cpu_device_query():
        distributed_save_hf_checkpoint(
            model, export_dir, maxbound=448, kv_cache_format=None, max_shard_size="8KB", **kwargs
        )


def _nogather_export(model, export_dir, **kwargs):
    with _cpu_device_query(), mock.patch.dict(os.environ, {"MODELOPT_FSDP2_EXPORT": "nogather"}):
        export_hf_checkpoint(
            model, dtype=torch.bfloat16, export_dir=export_dir, max_shard_size="16KB", **kwargs
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
    shards = sorted(p.name for p in export_dir.glob("model*.safetensors"))
    assert set(weight_map.values()) == set(shards), "index and shard files disagree"
    tensors = {}
    for shard in shards:
        loaded = load_file(str(export_dir / shard))
        assert {k for k, f in weight_map.items() if f == shard} == set(loaded), shard
        tensors.update(loaded)
    assert index["metadata"]["total_size"] == sum(_nbytes(t) for t in tensors.values())
    return tensors


def _save(config, path):
    torch.manual_seed(0)
    AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16).save_pretrained(path)
    return path


def _load(path):
    return AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).eval()


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


def _fsdp2_shard(model, mesh, wrap_root=True):
    """Shard each decoder layer, and the root too unless ``wrap_root`` is off (as expert parallelism
    leaves it, sharding only the dense layers), and record that the model is FSDP2-sharded."""
    for layer in model.model.layers:
        fully_shard(layer, mesh=mesh)
    if wrap_root:
        fully_shard(model, mesh=mesh)
    set_distribution_plan(model, DistributionPlan(mesh.size(), dense="fsdp"))
    return model


def _quantize_with_reference(model, quant_cfg):
    """Quantize ``model`` and return what the single-process export writes for a copy of it."""
    torch.manual_seed(1)
    batch = torch.randint(0, model.config.vocab_size, (2, 8))
    mtq.quantize(model, quant_cfg, lambda m: m(batch))
    ref = copy.deepcopy(model)
    want, _ = _export_transformers_checkpoint(ref, torch.bfloat16)
    want = revert_weight_conversion_quant_aware(ref, dict(want))
    assert any(v.dtype in (torch.float8_e4m3fn, torch.uint8) for v in want.values())
    return want


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


def _one_rank_cannot_revert_job(rank, world_size, message):
    # Rank 0's subset reverts; rank 1's fails, as a merge whose group ownership split does.
    model = AutoModelForCausalLM.from_config(_mixtral_config())
    sd = {"model.layers.0.mlp.experts.0.up_proj.weight": torch.ones(2)}
    before = dict(sd)
    failure = mock.patch(
        f"{_WRITER}.revert_weight_conversion_quant_aware", side_effect=RuntimeError(message)
    )
    with (
        bounded_collectives(),
        failure if rank == 1 else contextlib.nullcontext(),
        pytest.warns(UserWarning, match=f"rank 1: {message or 'RuntimeError'}"),
    ):
        assert _revert_whole_sd(model, sd) is False

    assert sd == before


@pytest.mark.parametrize("message", ["incomplete merge", ""], ids=["message", "no-message"])
def test_revert_whole_sd_keeps_in_memory_names_on_every_rank_when_one_fails(
    cpu_dist_workers, message
):
    """Rank 0 could revert its own subset, but a checkpoint mixing hub and in-memory names would
    load neither way. An exception with no message is a failure all the same."""
    cpu_dist_workers.run(_one_rank_cannot_revert_job, message=message)


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


def _earlier_checkpoint(export_dir, layout):
    """A checkpoint an earlier export left in ``export_dir``: one shard, three, or one file."""
    export_dir.mkdir(parents=True, exist_ok=True)
    if layout == "single-file":
        save_file({"old": torch.zeros(2)}, str(export_dir / "model.safetensors"))
    else:
        n = 1 if layout == "one-shard" else 3
        names = [f"model-{i:05d}-of-{n:05d}.safetensors" for i in range(1, n + 1)]
        for i, name in enumerate(names):
            save_file({f"old{i}": torch.zeros(2)}, str(export_dir / name))
        weight_map = {f"old{i}": name for i, name in enumerate(names)}
        index = {"metadata": {"total_size": 8 * n}, "weight_map": weight_map}
        (export_dir / "model.safetensors.index.json").write_text(json.dumps(index))
    return {p.name: p.read_bytes() for p in export_dir.iterdir()}


def _refuses_a_checkpoint_job(rank, world_size, tmp_dir, layout):
    # Only rank 1's view of the directory holds the checkpoint: rank 0 must not write either.
    export_dir = tmp_dir / f"rank{rank}"
    before = _earlier_checkpoint(export_dir, layout) if rank == 1 else {}
    with bounded_collectives(), pytest.raises((FileExistsError, RuntimeError)) as raised:
        _write_even_shards(
            {f"rank{rank}.weight": torch.ones(2)}, export_dir, "1MB", rank, world_size
        )

    assert "already holds a checkpoint" in str(raised.value), raised.value
    assert {p.name: p.read_bytes() for p in export_dir.iterdir()} == before


@pytest.mark.parametrize("layout", ["one-shard", "three-shards", "single-file"])
def test_write_even_shards_refuses_a_checkpoint_already_there(cpu_dist_workers, tmp_path, layout):
    """Every rank refuses before anything is written, though only one rank sees the checkpoint, and
    the checkpoint is left byte for byte: a rename failing partway through the commit would leave
    it mixed."""
    cpu_dist_workers.run(_refuses_a_checkpoint_job, tmp_dir=tmp_path, layout=layout)


def test_write_even_shards_writes_beside_other_files(gloo_world_of_one, tmp_path):
    """A config and an auxiliary weights file the index does not cover are no checkpoint: the export
    writes beside them and leaves them as they were."""
    others = {"config.json": b"{}", "mtp.safetensors": b"weights the index does not cover"}
    for name, data in others.items():
        (tmp_path / name).write_bytes(data)
    tensors = {"w": torch.ones(2)}

    _write_even_shards(dict(tensors), tmp_path, "1MB", 0, 1)

    _assert_byte_identical(_load_checkpoint(tmp_path), tensors)
    assert all((tmp_path / name).read_bytes() == data for name, data in others.items())


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


@pytest.mark.parametrize("fails", ["dir", "shard", "index"])
def test_a_write_that_fails_on_one_rank_fails_on_every_rank(cpu_dist_workers, tmp_path, fails):
    """Not just on the rank that failed, while its peers wait in a collective it never joins; and
    the directory is left as it was, with no file of the failed export."""
    cpu_dist_workers.run(_one_rank_fails_job, tmp_dir=tmp_path, fails=fails)

    assert not list(tmp_path.iterdir())


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


def test_the_index_goes_in_place_only_after_every_shard(cpu_dist_workers, tmp_path):
    """A rank that fails to put its shard in place leaves no index behind, so an index in the
    directory never names shards that are missing."""
    cpu_dist_workers.run(_shard_commit_fails_job, tmp_dir=tmp_path)

    assert not (tmp_path / "model.safetensors.index.json").exists()


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


# ------------------------------------------------------------------ the writer, one rank


@pytest.mark.parametrize("case", ["tied", "untied", "inference-mode", "extra-state-dict"])
def test_distributed_save_writes_the_hub_checkpoint_of_a_sharded_moe(
    gloo_world_of_one, tmp_path, case
):
    """Fused experts are split, every name is reverse-converted, and a declared tie is dropped.

    A caller in inference mode hands the writer inference tensors, which its writes reject. Extra
    tensors already carry hub names, so the reverse conversion must not rename them: the first is
    named like an in-memory Mixtral expert weight, which the reverse would turn into
    ``block_sparse_moe.experts.0.w1``.
    """
    src = _save(_mixtral_config(tie_word_embeddings=case == "tied"), tmp_path / "src")
    model = _fsdp2_shard(_load(src), gloo_world_of_one)
    want = load_file(str(src / "model.safetensors"))
    extra = None
    if case == "extra-state-dict":
        extra = {
            "model.layers.9.mlp.experts.0.gate_proj.weight": torch.arange(
                64, dtype=torch.bfloat16
            ).reshape(8, 8),
            "model.mtp.norm.weight": torch.full((8,), 3.0, dtype=torch.bfloat16),
        }
        want.update(extra)
    elif case == "inference-mode":
        want["model.norm.weight_scale"] = torch.tensor(0.5)

    with torch.inference_mode(case == "inference-mode"):
        if case == "inference-mode":
            # Made in inference mode, like the scales an inference-mode pack registers.
            model.model.norm.register_buffer("weight_scale", torch.full((), 0.5))
        _write(model, tmp_path / "export", extra_state_dict=extra)

    assert len(list((tmp_path / "export").glob("*.safetensors"))) > 1
    _assert_byte_identical(_load_checkpoint(tmp_path / "export"), want)


def test_distributed_save_rejects_extra_state_dict_that_overlaps_the_model(
    gloo_world_of_one, tmp_path
):
    model = _fsdp2_shard(_load(_save(_mixtral_config(), tmp_path / "src")), gloo_world_of_one)

    with pytest.raises(RuntimeError, match="overlaps the exported model weights"):
        _write(model, tmp_path / "export", extra_state_dict={"model.norm.weight": torch.ones(32)})


@pytest.mark.parametrize("wrap_root", [True, False], ids=["root-wrapped", "layers-only"])
@pytest.mark.parametrize(
    "quant_cfg", [mtq.FP8_DEFAULT_CFG, mtq.NVFP4_DEFAULT_CFG], ids=["fp8", "nvfp4"]
)
def test_nogather_export_writes_what_the_single_process_export_writes(
    gloo_world_of_one, tmp_path, quant_cfg, wrap_root
):
    src = _save(_llama_config(), tmp_path / "src")
    sidecar = src / "chat_template.jinja"  # a source file only the exporter carries over
    sidecar.write_text("{{ messages }}")
    model = _load(src)
    want = _quantize_with_reference(model, quant_cfg)

    _nogather_export(_fsdp2_shard(model, gloo_world_of_one, wrap_root), tmp_path / "export")

    _assert_byte_identical(_load_checkpoint(tmp_path / "export"), want)
    for name in ("config.json", "generation_config.json", "hf_quant_config.json"):
        assert (tmp_path / "export" / name).is_file(), name
    assert (tmp_path / "export" / sidecar.name).read_text() == sidecar.read_text()


def test_nogather_export_rejects_save_modelopt_state(gloo_world_of_one, tmp_path):
    """The path never runs ``save_pretrained``, which writes that state: refuse, not drop it."""
    model = _fsdp2_shard(_load(_save(_llama_config(), tmp_path / "src")), gloo_world_of_one)

    with pytest.raises(NotImplementedError, match="save_modelopt_state"):
        _nogather_export(model, tmp_path / "export", save_modelopt_state=True)


@pytest.mark.parametrize(
    "quant_cfg", [mtq.FP8_DEFAULT_CFG, mtq.NVFP4_DEFAULT_CFG], ids=["fp8", "nvfp4"]
)
def test_nogather_export_writes_moe_experts_quantized(gloo_world_of_one, tmp_path, quant_cfg):
    """Qwen2-MoE's shared expert follows its experts in the same FSDP2 unit, and packing it opens a
    weight window on that unit. Unless the experts were folded inside a window of their own, which
    hands FSDP2 the packed weights, that window's reshard puts FSDP2's unquantized shards back."""
    config = Qwen2MoeConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=32,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=32,
    )
    model = _load(_save(config, tmp_path / "src"))
    # Weight-only experts: expert input scales would need the writer's CUDA-only all-reduce.
    quant_cfg = copy.deepcopy(quant_cfg)
    quant_cfg["quant_cfg"].append({"quantizer_name": "*experts*input_quantizer", "enable": False})
    want = _quantize_with_reference(model, quant_cfg)

    _nogather_export(_fsdp2_shard(model, gloo_world_of_one), tmp_path / "export")

    _assert_byte_identical(_load_checkpoint(tmp_path / "export"), want)


# ------------------------------------------------------------------ the writer, two ranks


def _mtp_tail():
    """Tensors the model does not own, named like an inlined MTP tail the loader dropped."""
    return {
        "model.layers.99.mtp.weight": torch.arange(64, dtype=torch.bfloat16).reshape(8, 8),
        "model.layers.99.mtp.bias": torch.full((8,), 3.0, dtype=torch.bfloat16),
    }


def _two_rank_job(rank, world_size, tmp_dir):
    mesh = init_device_mesh("cpu", (world_size,))

    # An unquantized MoE whose vocab rows and experts split unevenly, written with tensors it does
    # not own, which every rank is handed; the test compares what the two ranks wrote between them
    # with the source checkpoint and those tensors.
    _write(
        _fsdp2_shard(_load(tmp_dir / "moe"), mesh),
        tmp_dir / "moe_export",
        extra_state_dict=_mtp_tail(),
    )

    # A quantized dense model through the export entry point: fp8 weights and bf16 norms travel
    # in separate per-dtype buckets. Every rank quantizes identically, so rank 0's single-process
    # export of its own copy is the reference.
    model = _load(tmp_dir / "dense")
    want = _quantize_with_reference(model, mtq.FP8_DEFAULT_CFG)
    _nogather_export(_fsdp2_shard(model, mesh), tmp_dir / "dense_export")
    if rank == 0:
        _assert_byte_identical(_load_checkpoint(tmp_dir / "dense_export"), want)

    # Each dense tensor ends up whole and plain on exactly one rank, whatever its layout, and the
    # owners are spread out. A zero-row tensor has no rows to move: its owner writes it as it is.
    full = {f"t{i}": torch.full((4, 2), float(i)) for i in range(9)}
    full["t9"] = torch.zeros(0, 4, dtype=torch.bfloat16)
    sd = {}
    for i, (k, v) in enumerate(full.items()):
        layout = ([Shard(0)], [Replicate()], None)[i % 3]
        sd[k] = v if layout is None else distribute_tensor(v, mesh, layout)
    _materialize_owned(sd, set(), rank, world_size)
    owned = [None] * world_size
    dist.all_gather_object(owned, sorted(sd))
    assert all(owned), f"a rank owns nothing: {owned}"
    assert sorted(k for keys in owned for k in keys) == sorted(full)
    for k, v in sd.items():
        assert not isinstance(v, DTensor), k
        assert v.dtype == full[k].dtype and torch.equal(v, full[k]), k

    # Ranks that disagree must fail with a diagnosis, not hang in a collective or drop a tensor.
    w = torch.ones(1, 2)
    if rank == 0:
        w = DTensor.from_local(w, mesh, [Shard(0)], run_check=False)
    cases = [
        ({"shared": torch.ones(2), f"rank{rank}": torch.ones(2)}, "dense key sets differ"),
        ({"w": w}, "disagree on which tensors are dim-0 sharded"),
    ]
    for sd, diagnosis in cases:
        try:
            with bounded_collectives():
                _materialize_owned(sd, set(), rank, world_size)
        except RuntimeError as error:
            raised = str(error)
        else:
            raised = "no error"
        assert diagnosis in raised, raised


def test_two_ranks_write_one_checkpoint_and_reject_inconsistent_state(cpu_dist_workers, tmp_path):
    """Shards cross ranks point to point, each tensor gets one owner, files share one numbering,
    and the tensors handed to every rank are written once.

    The same job also checks the cross-rank consistency errors.
    """
    src = _save(_mixtral_config(vocab_size=63, num_local_experts=3), tmp_path / "moe")
    _save(_llama_config(), tmp_path / "dense")

    cpu_dist_workers.run(_two_rank_job, tmp_dir=tmp_path)

    shards = sorted(p.name for p in (tmp_path / "moe_export").glob("*.safetensors"))
    n = len(shards)
    assert n > 2, "both ranks should have written several files"
    assert shards == [f"model-{i:05d}-of-{n:05d}.safetensors" for i in range(1, n + 1)]
    want = {**load_file(str(src / "model.safetensors")), **_mtp_tail()}
    _assert_byte_identical(_load_checkpoint(tmp_path / "moe_export"), want)
