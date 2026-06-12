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

pytest.importorskip("transformers")

from transformers.models.bert.modeling_bert import BertAttention, BertConfig
from transformers.models.gptj.modeling_gptj import GPTJAttention, GPTJConfig

from modelopt.torch.trace.plugins.transformers import get_hf_attn_sym_info


@pytest.mark.parametrize(
    ("cls_type", "config"),
    [
        (
            BertAttention,
            BertConfig(hidden_size=8, num_attention_heads=2, attn_implementation="eager"),
        ),
        (GPTJAttention, GPTJConfig(n_embd=12, n_head=2, rotary_dim=4)),
    ],
)
def test_attn_sym_info(cls_type, config):
    attn = cls_type(config)
    sym_info = get_hf_attn_sym_info(attn)

    assert sym_info.is_shape_preserving is True
    assert sym_info.symbols.keys() == {"num_attention_heads", "embed_dim", "hidden_dim"}
    assert sym_info.symbols["num_attention_heads"].is_searchable, "Expected searchable symbol"
    assert sym_info.symbols["embed_dim"].is_incoming, "Expected incoming symbol"
    assert sym_info.symbols["hidden_dim"].is_dynamic, "Expected dynamic symbol"
