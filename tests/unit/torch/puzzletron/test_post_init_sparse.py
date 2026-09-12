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

"""Tests for post-initialization structured sparsity."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from modelopt.torch.puzzletron.tools.post_init_sparse import SparsityMethod2o4


@pytest.mark.parametrize("layer_name", ["dense", "gate", "proj"])
def test_do_sparsity_preserves_layer_name(layer_name):
    block = nn.Module()
    block.mlp = nn.Module()
    linear = nn.Linear(4, 2, bias=False)
    block.mlp.add_module(layer_name, linear)

    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([block])
    model.config = SimpleNamespace(
        block_configs=[
            SimpleNamespace(
                ffn=SimpleNamespace(sparsify=[layer_name]),
                attention=SimpleNamespace(sparsify=[]),
            )
        ]
    )
    with torch.no_grad():
        linear.weight.copy_(torch.arange(1, 9, dtype=torch.float32).reshape(2, 4))

    SparsityMethod2o4().do_sparsity(model)

    expected_mask = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=torch.float32)
    torch.testing.assert_close(linear.weight_mask, expected_mask)
    torch.testing.assert_close(linear.weight, linear.weight_orig * expected_mask)
