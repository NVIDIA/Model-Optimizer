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

"""Checkpoint configuration contracts for captured attention geometry."""

import json
from types import SimpleNamespace

import pytest

from modelopt.torch.sparsity.attention_sparsity.calibration.mask_reuse_contracts import (
    validate_model_geometry,
)


@pytest.mark.parametrize("explicit_count", [True, False])
@pytest.mark.parametrize("nested", [True, False])
def test_full_attention_geometry_from_hybrid_block_list(tmp_path, explicit_count, nested):
    config = {
        "model_type": "nemotron_h",
        "layers_block_type": ["mamba", "attention", "moe", "attention"],
        "num_attention_heads": 64,
        "mtp_layers_block_type": ["attention", "moe"],
    }
    if explicit_count:
        config["num_hidden_layers"] = 4
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"text_config": config} if nested else config))
    checkpoint = SimpleNamespace(checkpoint_root=tmp_path)
    validate_model_geometry(checkpoint, [1, 3], 64)
    for layers, heads in (([1], 64), ([0, 1, 3], 64), ([1, 3], 32)):
        with pytest.raises(ValueError):
            validate_model_geometry(checkpoint, layers, heads)
    config["num_hidden_layers"] = 5
    path.write_text(json.dumps({"text_config": config} if nested else config))
    with pytest.raises(ValueError):
        validate_model_geometry(checkpoint, [1, 3], 64)
