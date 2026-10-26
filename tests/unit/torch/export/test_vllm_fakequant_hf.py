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

"""Tests for the vLLM fake-quant exporter's checkpoint-weight carry-over."""

import json

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from modelopt.torch.export.plugins import vllm_fakequant_hf as vfq


def test_carry_over_unplaced_weights_writes_extra_shard_and_reindexes(tmp_path, monkeypatch):
    """A single-file base checkpoint (the common, unindexed case) gains a shard and an index.

    export_hf_vllm_fq_checkpoint's inplace_mem_efficient path deliberately never passes an
    explicit state_dict= to save_pretrained (it would crash on meta tensors for offloaded
    params -- see the comment at its call site), so there is no state_dict to merge unplaced
    weights into the way export_hf_checkpoint does. _carry_over_unplaced_weights instead writes
    them as their own shard after save_pretrained has already run, and rebuilds the index from
    every shard on disk so a checkpoint that started as a single unindexed file still ends up
    correctly indexed once a second file exists.
    """
    save_file({"model.embed.weight": torch.zeros(4, 4)}, str(tmp_path / "model.safetensors"))
    assert not (tmp_path / "model.safetensors.index.json").exists()

    extra = {
        "mtp.eh_proj.weight": torch.full((2, 2), 3.0),
        "mtp.norm.weight": torch.full((2,), 5.0),
    }
    monkeypatch.setattr(vfq, "read_unplaced_weights", lambda model: extra)

    vfq._carry_over_unplaced_weights(tmp_path, model=torch.nn.Module())

    shard_path = tmp_path / "model-carried-over.safetensors"
    assert shard_path.exists()

    index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    assert weight_map == {
        "model.embed.weight": "model.safetensors",
        "mtp.eh_proj.weight": "model-carried-over.safetensors",
        "mtp.norm.weight": "model-carried-over.safetensors",
    }
    assert index["metadata"]["total_size"] > 0

    with safe_open(str(shard_path), framework="pt") as f:
        assert set(f.keys()) == set(extra)
        assert torch.equal(f.get_tensor("mtp.eh_proj.weight"), extra["mtp.eh_proj.weight"])


def test_carry_over_unplaced_weights_extends_an_existing_sharded_index(tmp_path, monkeypatch):
    """A checkpoint save_pretrained already sharded keeps its own entries after the extra shard lands."""
    save_file({"a.weight": torch.zeros(2)}, str(tmp_path / "model-00001-of-00002.safetensors"))
    save_file({"b.weight": torch.zeros(2)}, str(tmp_path / "model-00002-of-00002.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 16},
                "weight_map": {
                    "a.weight": "model-00001-of-00002.safetensors",
                    "b.weight": "model-00002-of-00002.safetensors",
                },
            }
        )
    )

    extra = {"mtp.eh_proj.weight": torch.full((2, 2), 3.0)}
    monkeypatch.setattr(vfq, "read_unplaced_weights", lambda model: extra)

    vfq._carry_over_unplaced_weights(tmp_path, model=torch.nn.Module())

    weight_map = json.loads((tmp_path / "model.safetensors.index.json").read_text())["weight_map"]
    assert weight_map["a.weight"] == "model-00001-of-00002.safetensors"
    assert weight_map["b.weight"] == "model-00002-of-00002.safetensors"
    assert weight_map["mtp.eh_proj.weight"] == "model-carried-over.safetensors"


def test_carry_over_unplaced_weights_is_a_true_noop_when_nothing_to_carry(tmp_path, monkeypatch):
    """No unplaced weights means the checkpoint on disk is untouched, not merely unchanged in content.

    Writing a same-content index anyway would still be a behavior change for the (common) case of
    a model with nothing to carry: every plain export would gain an index.json it did not have
    before.
    """
    save_file({"model.embed.weight": torch.zeros(4, 4)}, str(tmp_path / "model.safetensors"))
    monkeypatch.setattr(vfq, "read_unplaced_weights", lambda model: {})

    vfq._carry_over_unplaced_weights(tmp_path, model=torch.nn.Module())

    assert not (tmp_path / "model-carried-over.safetensors").exists()
    assert not (tmp_path / "model.safetensors.index.json").exists()
