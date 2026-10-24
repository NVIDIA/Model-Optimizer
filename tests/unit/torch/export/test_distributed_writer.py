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
from unittest import mock

import pytest
import torch
from _test_utils.torch.distributed.utils import bounded_collectives
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
        ("1GiB", 2**30),
        (" 1.5 mb ", 1_500_000),
        ("123", 123),
        (4096, 4096),
        ("5B", ValueError),
        ("5 gigs", ValueError),
        ("", ValueError),
    ],
)
def test_size_to_bytes_reads_hf_shard_sizes(size, nbytes):
    """Decimal and binary units, any case and padding, a bare number; anything else is refused."""
    if nbytes is ValueError:
        with pytest.raises(ValueError):
            _size_to_bytes(size)
    else:
        assert _size_to_bytes(size) == nbytes


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
    ],
    ids=["shard1", "2d-mesh", "padded"],
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
    else:
        mesh = init_device_mesh("cpu", (world_size,))
        if case == "placement":
            # Only rank 1 holds a placement the export cannot materialize.
            local, placement = (
                (torch.zeros(2, 4), Shard(0)) if rank == 0 else (torch.zeros(4, 2), Shard(1))
            )
            gshape = (4, 4)
        else:
            # A rewrap that re-inferred the global shape per rank: rank 0's own view tiles, rank 1's not.
            local, placement, gshape = torch.zeros(2, 4), Shard(0), (4 if rank == 0 else 3, 4)
        w = DTensor.from_local(
            local, mesh, [placement], run_check=False, shape=torch.Size(gshape), stride=(4, 1)
        )
    try:
        with bounded_collectives():
            _materialize_owned({"w": w}, set(), rank, world_size)
    except RuntimeError as error:
        raised = str(error)
    else:
        raised = "no error"
    match = "do not tile their global shape" if case == "global-shape" else "cannot materialize"
    assert match in raised, raised


@pytest.mark.parametrize("case", ["smaller-mesh", "placement", "global-shape"])
def test_materialize_owned_refuses_on_every_rank(cpu_dist_workers, case):
    """Every rank refuses, the ranks whose own view looks fine included, instead of waiting in a
    collective the refusing rank never joins."""
    cpu_dist_workers.run(_refusing_materialize_job, case=case)


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
    failure = mock.patch.object(
        quant_aware_conversion,
        "revert_weight_conversion_quant_aware",
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


def _one_rank_fails_job(rank, world_size, tmp_dir, fails):
    failing_rank, target = {
        "dir": (1, "pathlib.Path.mkdir"),
        "shard": (1, "safetensors.torch.save_file"),
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
    """Not just on the rank that failed, while its peers wait in a collective it never joins."""
    cpu_dist_workers.run(_one_rank_fails_job, tmp_dir=tmp_path, fails=fails)


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
    cpu_dist_workers.run(_cross_rank_collision_job, tmp_dir=tmp_path)
