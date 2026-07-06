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

import json

import torch

from modelopt.torch.export.plugins import mcore_custom


def test_save_safetensors_by_layer_index_uses_single_snapshot(monkeypatch, tmp_path):
    """Shard JSON and safetensors must be produced from the same key snapshot."""
    layer_state_dicts = {1: {"layers.0.weight": torch.arange(4, dtype=torch.float32)}}
    late_key = "mtp.late.weight"
    written_keys_by_shard = {}

    def _fake_save_file(tensors, path, metadata=None):
        written_keys_by_shard[path.split("/")[-1]] = set(tensors.keys())
        # Simulate a late mutation against the original source dict (not the writer snapshot).
        layer_state_dicts[1][late_key] = torch.ones(1, dtype=torch.float32)

    monkeypatch.setattr(mcore_custom, "save_file", _fake_save_file)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)

    mcore_custom.save_safetensors_by_layer_index(
        layer_state_dicts=layer_state_dicts,
        total_layers=1,
        save_directory=str(tmp_path),
        name_template="model-{:05d}-of-{:05d}",
    )

    shard_name = "model-00001-of-00001.safetensors"
    with open(tmp_path / "model-00001-of-00001.json") as f:
        shard_meta = json.load(f)
    with open(tmp_path / "model.safetensors.index.json") as f:
        index_meta = json.load(f)

    json_keys = set(shard_meta["weight_map"].keys())
    assert json_keys == written_keys_by_shard[shard_name]
    assert late_key not in json_keys
    assert late_key not in index_meta["weight_map"]
