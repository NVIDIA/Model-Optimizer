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

import pytest
import torch

from modelopt.torch.nas.registry import DMRegistry
from modelopt.torch.utils import make_divisible

pytest.importorskip("transformers")
from transformers.models.gptj.modeling_gptj import GPTJAttention, GPTJConfig


@pytest.mark.parametrize(
    ("n_attn_head", "embed_dim", "rotary_dim", "active_n_attn_head", "active_embed_dim"),
    [
        (2, 12, 4, 1, 6),
        (4, 24, 6, 3, 12),
    ],
)
def test_dynamic_gptj_attention(
    n_attn_head: int,
    embed_dim: int,
    rotary_dim: int,
    active_n_attn_head: int,
    active_embed_dim: int,
) -> None:
    def _get_dyn_model():
        config = GPTJConfig(n_embd=embed_dim, n_head=n_attn_head, rotary_dim=rotary_dim)
        model = GPTJAttention(config)
        return DMRegistry.convert(model)

    model = _get_dyn_model()

    # check if for max subnet the n_attn_head are the original ones
    model.num_attention_heads = model.get_hparam("num_attention_heads").max
    assert model.q_proj.weight is model.q_proj._parameters["weight"]

    hidden_dim_per_head = (
        model.get_hparam("hidden_dim").original // model.get_hparam("num_attention_heads").original
    )
    active_hidden_dim = hidden_dim_per_head * active_n_attn_head
    model.num_attention_heads = active_n_attn_head
    model.hidden_dim = active_hidden_dim
    model.embed_dim = active_embed_dim

    # store removable attributes before export
    attr_to_be_removed = model._dm_attribute_manager.attr_keys()

    model = model.export()
    assert type(model) is GPTJAttention

    model.train()
    inputs = torch.randn(1, 6, active_embed_dim)
    position_ids = torch.arange(6, dtype=torch.long)
    targets = model(inputs, position_ids=position_ids)[0]
    loss = targets.sum()
    loss.backward()

    assert torch.allclose(
        model.q_proj.weight.grad[active_hidden_dim:, active_embed_dim:],
        torch.tensor([0.0]),
    )

    model.eval()
    assert model.training is False
    assert model.embed_dim == active_embed_dim
    assert model.hidden_dim == active_hidden_dim

    for attr in attr_to_be_removed:
        assert not hasattr(model, attr)

    outputs = model(inputs, position_ids=position_ids)[0]
    assert outputs.shape == targets.shape
    assert torch.allclose(outputs, targets)

    # get new model and check that we can correctly modify choices
    model = _get_dyn_model()

    # # now check if we can reduce the choices
    n_heads_ratio = [0.5, 1.0]
    n_heads_divisor = 2
    model.modify(n_heads_ratio=n_heads_ratio, n_heads_divisor=n_heads_divisor)

    hp = model.get_hparam("num_attention_heads")
    assert all(c % n_heads_divisor == 0 for c in hp.choices)
    assert hp.choices == list(
        {make_divisible(r * hp.original, n_heads_divisor) for r in n_heads_ratio}
    )
