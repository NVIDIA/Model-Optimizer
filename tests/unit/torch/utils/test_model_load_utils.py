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

"""Pure-function tests for ``modelopt.torch.utils.model_load_utils``."""

import json

import pytest
import torch
from safetensors.torch import save_file

from modelopt.torch.utils.model_load_utils import read_safetensors_subset, weight_map_for


def test_weight_map_for_sharded(tmp_path):
    save_file({"a.weight": torch.zeros(2)}, str(tmp_path / "shard1.safetensors"))
    save_file({"b.weight": torch.zeros(2)}, str(tmp_path / "shard2.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": {"a.weight": "shard1.safetensors", "b.weight": "shard2.safetensors"}}
        )
    )

    assert weight_map_for(str(tmp_path)) == {
        "a.weight": "shard1.safetensors",
        "b.weight": "shard2.safetensors",
    }


def test_weight_map_for_single_file(tmp_path):
    save_file(
        {"a.weight": torch.zeros(2), "b.weight": torch.zeros(2)},
        str(tmp_path / "model.safetensors"),
    )

    assert weight_map_for(str(tmp_path)) == {
        "a.weight": "model.safetensors",
        "b.weight": "model.safetensors",
    }


def test_weight_map_for_missing(tmp_path):
    with pytest.raises(RuntimeError, match="No safetensors checkpoint"):
        weight_map_for(str(tmp_path))


def test_read_safetensors_subset(tmp_path):
    save_file(
        {"a.weight": torch.tensor([1.0, 2.0]), "a.bias": torch.tensor([3.0])},
        str(tmp_path / "shard1.safetensors"),
    )
    save_file({"b.weight": torch.tensor([4.0])}, str(tmp_path / "shard2.safetensors"))
    weight_map = {
        "a.weight": "shard1.safetensors",
        "a.bias": "shard1.safetensors",
        "b.weight": "shard2.safetensors",
    }

    result = read_safetensors_subset(str(tmp_path), weight_map, lambda n: n.startswith("a."))

    assert set(result.keys()) == {"a.weight", "a.bias"}
    assert torch.equal(result["a.weight"], torch.tensor([1.0, 2.0]))
    assert torch.equal(result["a.bias"], torch.tensor([3.0]))
