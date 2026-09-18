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

"""Carry-over must catch every on-disk layout an unplaced weight can arrive in.

These drive a real ``from_pretrained``, not a stub, because the thing under test is what the
Transformers loader does and does not report. The three MTP storage conventions come from the
support matrix the previous name-matching implementation carried:

    inlined         GLM-5.1, DeepSeek-V3   ``model.layers.{N}.*`` past ``num_hidden_layers``
    standalone      GLM-4.7                a ``mtp.safetensors`` the index does not reference
    indexed shard   Qwen3-Next             an ``mtp.*`` tail shard listed in the index

The standalone case is handled differently from the other two, and deliberately so. The loader
only opens shards named in the index, so tensors in an extra file produce no ``unexpected_keys``
at all -- but they are also untouched by quantization and absent from the export, which makes them
sidecars rather than carry-over. They are copied verbatim, which costs no host memory and keeps
the bytes and filename a consumer looks for. The other two conventions land inside shards the
loader does read, so those keys ride the state dict.

CPU-only and small -- the mechanism is about bookkeeping during load, so a GPU adds nothing.
"""

import json

import pytest
import torch
from _test_utils.examples.hf_ptq_example_utils import example_utils
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, LlamaConfig

from modelopt.torch.utils.plugins.hf_checkpoint_utils import copy_off_index_safetensors

NUM_HIDDEN_LAYERS = 2


def _build_checkpoint(d):
    """A tiny real model saved to disk, so the loader has something genuine to place."""
    cfg = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
    )
    AutoModelForCausalLM.from_config(cfg).save_pretrained(d, safe_serialization=True)


def _add_to_main_shard(d, extra):
    f = d / "model.safetensors"
    tensors = load_file(str(f))
    tensors.update(extra)
    save_file(tensors, str(f), metadata={"format": "pt"})


def _recorded(d):
    model = example_utils._from_pretrained_recording(AutoModelForCausalLM, str(d))
    return model._modelopt_unplaced_source_keys


@pytest.fixture
def ckpt(tmp_path):
    _build_checkpoint(tmp_path)
    return tmp_path


def test_inlined_layer_past_num_hidden_layers(ckpt):
    """GLM-5.1 / DeepSeek-V3: the head is decoder layer N, which the model never builds."""
    keys = [
        f"model.layers.{NUM_HIDDEN_LAYERS}.input_layernorm.weight",
        f"model.layers.{NUM_HIDDEN_LAYERS}.mlp.gate_proj.weight",
    ]
    _add_to_main_shard(ckpt, {keys[0]: torch.zeros(32), keys[1]: torch.zeros(64, 32)})

    assert set(_recorded(ckpt)) == set(keys)


def test_standalone_off_index_file_is_copied_not_carried(ckpt, tmp_path):
    """GLM-4.7: a separate mtp.safetensors. The loader never opens it, so it contributes no
    unexpected_keys -- it is preserved by copying the file, not by routing its tensors through
    the state dict."""
    payload = {"mtp.fc.weight": torch.zeros(32, 32), "mtp.layers.0.enorm.weight": torch.zeros(32)}
    save_file(payload, str(ckpt / "mtp.safetensors"), metadata={"format": "pt"})

    assert _recorded(ckpt) == [], "the loader never saw it, so it cannot report it"

    export = tmp_path / "export"
    export.mkdir()
    assert copy_off_index_safetensors(ckpt, export) == ["mtp.safetensors"]
    assert load_file(str(export / "mtp.safetensors")).keys() == payload.keys()


def test_indexed_shards_are_not_copied(ckpt, tmp_path):
    """The guard on the above: shards the loader does read are the source weights, and copying
    them into the export would sit alongside the quantized ones."""
    export = tmp_path / "export"
    export.mkdir()
    assert copy_off_index_safetensors(ckpt, export) == []


def test_indexed_tail_shard(ckpt):
    """Qwen3-Next: an mtp.* shard the index does reference, so the loader reads and rejects it."""
    main = ckpt / "model.safetensors"
    base = load_file(str(main))
    first, tail = "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"
    mtp = {"mtp.fc.weight": torch.zeros(32, 32), "mtp.layers.0.enorm.weight": torch.zeros(32)}
    save_file(base, str(ckpt / first), metadata={"format": "pt"})
    save_file(mtp, str(ckpt / tail), metadata={"format": "pt"})
    main.unlink()
    weight_map = {**dict.fromkeys(base, first), **dict.fromkeys(mtp, tail)}
    (ckpt / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map})
    )

    assert set(_recorded(ckpt)) == set(mtp)


def test_auxiliary_tower_is_carried_too(ckpt):
    """Nothing is keyed to MTP: any tensor the model has no home for is carried."""
    _add_to_main_shard(ckpt, {"aux_tower.blocks.0.weight": torch.zeros(8, 8)})

    assert set(_recorded(ckpt)) == {"aux_tower.blocks.0.weight"}


def test_two_layouts_at_once(ckpt, tmp_path):
    """An inlined head and an off-index sidecar together: each must be picked up by its own
    mechanism, and neither by both -- a tensor carried *and* copied would be exported twice."""
    inlined = f"model.layers.{NUM_HIDDEN_LAYERS}.input_layernorm.weight"
    _add_to_main_shard(ckpt, {inlined: torch.zeros(32)})
    save_file(
        {"mtp.fc.weight": torch.zeros(32, 32)},
        str(ckpt / "mtp.safetensors"),
        metadata={"format": "pt"},
    )

    export = tmp_path / "export"
    export.mkdir()
    recorded = set(_recorded(ckpt))
    copied = copy_off_index_safetensors(ckpt, export)

    assert recorded == {inlined}
    assert copied == ["mtp.safetensors"]
    assert "mtp.fc.weight" not in recorded, "copied files must not also ride the state dict"


def test_clean_checkpoint_records_nothing(ckpt):
    """No stray tensors -> an empty answer, distinct from never having asked."""
    model = example_utils._from_pretrained_recording(AutoModelForCausalLM, str(ckpt))

    assert model._modelopt_unplaced_source_keys == []
    assert model._modelopt_source_checkpoint == str(ckpt)


def test_recorded_keys_are_sorted_and_deduplicated(ckpt):
    """The two sources can overlap; the export wants a stable, duplicate-free list."""
    _add_to_main_shard(
        ckpt, {"zzz_orphan.weight": torch.zeros(4), "aaa_orphan.weight": torch.zeros(4)}
    )

    keys = _recorded(ckpt)
    assert keys == sorted(keys) and len(keys) == len(set(keys))
