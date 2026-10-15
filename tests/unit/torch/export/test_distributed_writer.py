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
from datetime import timedelta
from functools import partial
from unittest import mock

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

pytest.importorskip("safetensors")
pytest.importorskip("transformers")
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, LlamaConfig, MixtralConfig

from modelopt.torch.export import quant_aware_conversion
from modelopt.torch.export.distributed_writer import (
    _even_bins,
    _is_full_world_shard0,
    _materialize_owned,
    _nbytes,
    _revert_whole_sd,
    _size_to_bytes,
    _write_even_shards,
)


@pytest.fixture
def gloo_world_of_one(skip_on_windows, tmp_path_factory):
    """A one-rank gloo process group and its CPU device mesh."""
    store = tmp_path_factory.mktemp("pg") / "store"
    dist.init_process_group("gloo", init_method=store.as_uri(), rank=0, world_size=1)
    yield init_device_mesh("cpu", (1,))
    dist.destroy_process_group()


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
    ("size", "nbytes"),
    [
        ("5GB", 5 * 10**9),
        ("500MB", 500 * 10**6),
        ("10KB", 10 * 10**3),
        ("2TB", 2 * 10**12),
        ("1GiB", 2**30),
        ("3MiB", 3 * 2**20),
        ("4KiB", 4 * 2**10),
        ("1TiB", 2**40),
        ("1.5GB", 1_500_000_000),
        (" 2 mb ", 2 * 10**6),
        ("123", 123),
        (4096, 4096),
    ],
)
def test_size_to_bytes_reads_hf_shard_sizes(size, nbytes):
    assert _size_to_bytes(size) == nbytes


@pytest.mark.parametrize("size", ["5B", "5 gigs", ""])
def test_size_to_bytes_rejects_an_unknown_unit(size):
    with pytest.raises(ValueError):
        _size_to_bytes(size)


def test_even_bins_places_every_key_once_within_the_bound():
    sizes = {f"t{i}": i + 1 for i in range(20)}  # 210 bytes in all

    bins = _even_bins(list(sizes), sizes, math.ceil(210 / 50), 50)

    assert sorted(k for b in bins for k in b) == sorted(sizes)
    assert all(0 < sum(sizes[k] for k in b) <= 50 for b in bins)


def test_even_bins_balances_the_files_instead_of_filling_them_in_turn():
    """``max_shard_size`` is a bound, not a target: two files of 8 rather than 10 + 6."""
    sizes = {"a": 4, "b": 3, "c": 3, "d": 2, "e": 2, "f": 2}

    bins = _even_bins(list(sizes), sizes, 2, 10)

    assert sorted(sum(sizes[k] for k in b) for b in bins) == [8, 8]


def test_even_bins_adds_bins_until_every_bin_fits():
    sizes = {"a": 6, "b": 6, "c": 6}

    bins = _even_bins(list(sizes), sizes, 1, 10)

    assert sorted(map(sorted, bins)) == [["a"], ["b"], ["c"]]


def test_even_bins_gives_a_tensor_larger_than_the_bound_a_bin_of_its_own():
    sizes = {"big": 100, "a": 1, "b": 1, "c": 1}

    bins = _even_bins(list(sizes), sizes, 1, 10)

    assert ["big"] in bins
    assert sorted(k for b in bins for k in b) == sorted(sizes)


def test_even_bins_still_packs_the_rest_when_one_tensor_exceeds_the_bound():
    sizes = {"big": 100, "a": 1, "b": 1, "c": 1}

    bins = _even_bins(list(sizes), sizes, 1, 10)

    assert len(bins) == 2


def test_even_bins_never_returns_an_empty_bin():
    sizes = {"a": 1, "b": 1}
    assert sorted(map(sorted, _even_bins(list(sizes), sizes, 5, 10))) == [["a"], ["b"]]
    assert _even_bins([], {}, 3, 10) == []


def test_nbytes_counts_the_global_size_of_a_dtensor(gloo_world_of_one):
    assert _nbytes(torch.zeros(3, 4, dtype=torch.bfloat16)) == 24
    assert _nbytes(torch.zeros(5, dtype=torch.uint8)) == 5
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


@pytest.mark.parametrize(
    ("placement", "expected"),
    [(Shard(0), True), (Shard(1), False), (Replicate(), False)],
    ids=["shard0", "shard1", "replicate"],
)
def test_is_full_world_shard0_accepts_only_a_dim0_shard_over_the_whole_world(
    gloo_world_of_one, placement, expected
):
    tensor = distribute_tensor(torch.arange(12.0).reshape(4, 3), gloo_world_of_one, [placement])
    assert _is_full_world_shard0(tensor, 1) is expected


def test_is_full_world_shard0_rejects_a_mesh_that_is_not_the_whole_world(gloo_world_of_one):
    tensor = distribute_tensor(torch.zeros(4, 3), gloo_world_of_one, [Shard(0)])
    assert _is_full_world_shard0(tensor, 2) is False
    mesh_2d = init_device_mesh("cpu", (1, 1))
    tensor_2d = distribute_tensor(torch.zeros(4, 3), mesh_2d, [Shard(0), Replicate()])
    assert _is_full_world_shard0(tensor_2d, 1) is False


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


def test_materialize_owned_rejects_a_placement_it_cannot_assemble(gloo_world_of_one):
    sd = {"w": distribute_tensor(torch.zeros(4, 4), gloo_world_of_one, [Shard(1)])}
    with pytest.raises(RuntimeError, match=r"cannot materialize.*'w'"):
        _materialize_owned(sd, set(), 0, 1)


def test_materialize_owned_rejects_shards_that_do_not_tile_the_global_shape(gloo_world_of_one):
    """A padded local chunk would concatenate into a wrong tensor; it must raise instead."""
    padded = DTensor.from_local(
        torch.zeros(2, 4),
        gloo_world_of_one,
        [Shard(0)],
        run_check=False,
        shape=torch.Size([3, 4]),
        stride=(4, 1),
    )
    with pytest.raises(RuntimeError, match="do not tile their global shape"):
        _materialize_owned({"w": padded}, set(), 0, 1)


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


def test_revert_whole_sd_keeps_in_memory_names_when_the_reverse_fails(monkeypatch):
    def _fail(model, state_dict):
        raise RuntimeError("unsupported op")

    monkeypatch.setattr(quant_aware_conversion, "revert_weight_conversion_quant_aware", _fail)
    sd = {"model.layers.0.mlp.experts.0.up_proj.weight": torch.ones(2)}
    before = dict(sd)

    with pytest.warns(UserWarning, match="reverse weight conversion skipped"):
        assert _revert_whole_sd(torch.nn.Module(), sd) is False

    assert sd == before


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


def test_write_even_shards_fails_loudly_when_no_rank_has_anything_to_write(
    gloo_world_of_one, tmp_path
):
    with pytest.raises(RuntimeError, match="produced no shards"):
        _write_even_shards({}, tmp_path, "1MB", 0, 1)


@contextlib.contextmanager
def _bounded_collectives(seconds=10):
    """Run the writer's default-group collectives on a new group that gives up after ``seconds``.

    A rank that fails alone can leave a peer waiting in a collective it never joins. On this group
    that wait raises in seconds instead of hanging for the default group's timeout, and the default
    group, which the worker pool keeps for the next test, never falls out of step.
    """
    group = dist.new_group(timeout=timedelta(seconds=seconds))
    with (
        mock.patch.object(dist, "all_gather_object", partial(dist.all_gather_object, group=group)),
        mock.patch.object(dist, "barrier", partial(dist.barrier, group=group)),
    ):
        yield


def _one_rank_fails_job(rank, world_size, tmp_dir, fails):
    failing_rank, step, target = {
        "shard": (1, "shards written", "safetensors.torch.save_file"),
        "index": (0, "index written", "json.dump"),
    }[fails]
    if rank == failing_rank:
        failure = mock.patch(target, side_effect=OSError("disk full"))
        expected = pytest.raises(OSError, match="disk full")
    else:
        failure = contextlib.nullcontext()
        expected = pytest.raises(
            RuntimeError, match=f"rank {failing_rank} failed to compute {step}: OSError: disk full"
        )
    with _bounded_collectives(), failure, expected:
        _write_even_shards({f"rank{rank}.weight": torch.ones(2)}, tmp_dir, "1MB", rank, world_size)


@pytest.mark.parametrize("fails", ["shard", "index"])
def test_a_write_that_fails_on_one_rank_fails_on_every_rank(cpu_dist_workers, tmp_path, fails):
    """Not just on the rank that failed, while its peers wait in a collective it never joins."""
    cpu_dist_workers.run(_one_rank_fails_job, tmp_dir=tmp_path, fails=fails)


def _cross_rank_collision_job(rank, world_size, tmp_dir):
    with (
        _bounded_collectives(),
        pytest.raises(RuntimeError, match="name collision across ranks"),
    ):
        _write_even_shards({"model.norm.weight": torch.ones(2)}, tmp_dir, "1MB", rank, world_size)


def test_a_name_two_ranks_both_write_fails_the_write_on_every_rank(cpu_dist_workers, tmp_path):
    cpu_dist_workers.run(_cross_rank_collision_job, tmp_dir=tmp_path)
