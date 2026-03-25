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

"""Utilities to describe symbols in the dynamic attention module."""

import torch
import transformers
from packaging.version import Version
from torch import nn
from transformers.models.bert.modeling_bert import BertAttention, BertLayer
from transformers.models.gptj.modeling_gptj import GPTJAttention

from ..symbols import Symbol, SymInfo, SymMap

__all__ = ["SymAttentionHead"]


class SymAttentionHead(Symbol):
    """Just a special class to mark the attention head symbol."""


def get_hf_attn_sym_info(sortable_attn: bool = False) -> SymInfo:
    # embed_dim is registered as elastic incoming symbol (we don't support sorting for now!)
    embed_dim = Symbol(is_sortable=False, cl_type=Symbol.CLType.INCOMING, elastic_dims={-1})

    # num_attention_heads is registered as a special symbol
    num_attention_heads = SymAttentionHead(is_sortable=sortable_attn, is_searchable=True)

    # hidden_dim is linked to num_attention_heads. Correct handling of dependencies done in hps
    # NOTE: we assume hidden_dim is 1st dependency of num_attention_heads in hps!
    hidden_dim = Symbol(is_sortable=sortable_attn, elastic_dims={-1})
    hidden_dim.link_to(num_attention_heads)

    return SymInfo(
        is_shape_preserving=True,
        num_attention_heads=num_attention_heads,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
    )


@SymMap.register([BertAttention])
def get_hf_attn_sym_info_sortable(mod: nn.Module) -> SymInfo:
    return get_hf_attn_sym_info(sortable_attn=True)


@SymMap.register([GPTJAttention])
def get_hf_attn_sym_info_unsortable(mod: nn.Module) -> SymInfo:
    return get_hf_attn_sym_info(sortable_attn=True)


# In transformers>=5.0, BertLayer.forward uses tuple unpacking on the BertAttention output
# (e.g. `self_attn_out, _ = self.attention(...)`), which FX symbolic tracing cannot handle when
# BertAttention is a registered leaf (the proxy is not iterable). Patch BertLayer.forward to use
# indexing instead, and call feed_forward_chunk directly (equivalent to apply_chunking_to_forward
# with chunk_size=0, which is the default for BERT).
if Version(transformers.__version__) >= Version("5.0"):

    def _fx_friendly_bert_layer_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        # Use indexing instead of tuple-unpacking so FX can trace through BertLayer
        # when BertAttention is a registered leaf (returns an opaque Proxy).
        # Accept **kwargs so that a parent trace (e.g. BertEncoder) passing extra kwargs
        # like position_ids does not mark BertLayer as failed. However, do NOT forward
        # **kwargs into self.attention: FX represents **kwargs as a Proxy(_kwargs), so
        # unpacking it with ** would trigger "Proxy cannot be iterated". Additionally,
        # BertSelfAttention ignores these kwargs (e.g. position_ids) in practice.
        _attn_outputs = self.attention(
            hidden_states,
            attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )
        attention_output = _attn_outputs[0]

        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with"
                    " cross-attention layers by setting `config.add_cross_attention=True`"
                )
            _cross_outputs = self.crossattention(
                attention_output,
                None,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_values=past_key_values,
            )
            attention_output = _cross_outputs[0]

        # Call feed_forward_chunk directly (equivalent to apply_chunking_to_forward when
        # chunk_size_feed_forward=0, which is the BERT default).
        return self.feed_forward_chunk(attention_output)

    BertLayer.forward = _fx_friendly_bert_layer_forward
