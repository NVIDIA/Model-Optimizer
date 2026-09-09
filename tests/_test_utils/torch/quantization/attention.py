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

"""Shared attention-quantization fixtures for the unit and gpu attention tests."""

import pytest

pytest.importorskip("transformers")

from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention

from modelopt.torch.quantization.plugins.huggingface import _QuantAttention


def make_quant_attention(hidden_size=128, num_q_heads=4, num_kv_heads=2):
    """A single ``_QuantAttention``-converted Llama attention layer, pinned to the sdpa impl."""
    config = LlamaConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_q_heads,
        num_key_value_heads=num_kv_heads,
    )
    quant_attention = _QuantAttention.convert(LlamaAttention(config, layer_idx=0))
    quant_attention.config._attn_implementation = "sdpa"
    return quant_attention
