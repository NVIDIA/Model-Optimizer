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

import pytest
import torch
from transformers import AutoModelForCausalLM, Qwen3Config

from modelopt.torch.utils import (
    DETERMINISTIC_ATTN_IMPLEMENTATIONS,
    deterministic_attn_implementation,
)


@pytest.mark.parametrize("attn_implementation", DETERMINISTIC_ATTN_IMPLEMENTATIONS)
def test_a_sample_has_the_same_logits_whether_a_neighbour_in_its_batch_is_padded(
    attn_implementation,
):
    """The default SDPA runs the flash kernel on a batch without padding and another one on a batch
    with a padded row, so a sample's logits depend on its neighbours."""
    config = Qwen3Config(
        vocab_size=512,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=512,
    )
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(
        config, attn_implementation=deterministic_attn_implementation(attn_implementation)
    )
    model = model.to("cuda", torch.bfloat16).eval()

    input_ids = torch.randint(1, config.vocab_size, (4, 256), device="cuda")
    attention_mask = torch.ones_like(input_ids)
    padded_neighbour = attention_mask.clone()
    padded_neighbour[1, :100] = 0
    with torch.no_grad():
        no_padding = model(input_ids, attention_mask=attention_mask).logits[0]
        with_padding = model(input_ids, attention_mask=padded_neighbour).logits[0]

    assert torch.equal(no_padding, with_padding)
