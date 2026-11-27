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

"""A stand-in for Kimi-K3's own remote modeling code: the classes and functions the fixes for it
patch, for tests on tiny CPU tensors.
"""

import types

import torch
from torch import nn


def remote_module(name="kimi_k3_remote_stub"):
    """A stand-in for the remote modeling namespace: what the fixes patch, as shipped."""
    remote = types.ModuleType(name)
    remote.calls = []

    class KimiDeltaAttention(nn.Module):
        def __init__(self, num_heads, head_dim):
            super().__init__()
            self.head_dim = head_dim
            self.A_log = nn.Parameter(torch.zeros(num_heads, dtype=torch.bfloat16))  # stale shape

    # fla-core 0.5's signatures: use_beta_sigmoid_in_kernel is no parameter of either kernel; it lands
    # in **kwargs, which is why the kernels ignore it.
    def chunk_kda(q, k, v, g, beta, scale=None, initial_state=None, **kwargs):
        remote.calls.append(("chunk_kda", beta, kwargs.get("use_beta_sigmoid_in_kernel", False)))
        return "chunk_kda"

    def fused_recurrent_kda(q, k, v, g, beta, A_log=None, dt_bias=None, scale=None, **kwargs):  # noqa: N803
        flag = kwargs.get("use_beta_sigmoid_in_kernel", False)
        remote.calls.append(("fused_recurrent_kda", beta, flag))
        return "fused_recurrent_kda"

    def create_causal_mask(
        config, inputs_embeds, attention_mask, past_key_values, position_ids=None
    ):
        # current transformers' signature: ``inputs_embeds``, and no ``cache_position``
        args = (config, inputs_embeds, attention_mask, past_key_values, position_ids)
        remote.calls.append(("create_causal_mask", *args))
        return "mask"

    class KimiDynamicCache:
        def __init__(self, seen_tokens):
            self.seen_tokens = seen_tokens

        def get_seq_length(self, layer_idx=0):
            remote.calls.append(("get_seq_length", layer_idx))
            return self.seen_tokens

        def get_mask_sizes(self, cache_position, layer_idx):  # wants a tensor, as older callers
            return cache_position.shape[0] + self.get_seq_length(layer_idx), 0

    class KimiK3ForConditionalGeneration:
        def tie_weights(self):  # the remote override, with the old no-kwarg signature
            remote.calls.append(("tie_weights", self))
            return "tied"

    for obj in (
        KimiDeltaAttention,
        chunk_kda,
        fused_recurrent_kda,
        create_causal_mask,
        KimiDynamicCache,
        KimiK3ForConditionalGeneration,
    ):
        obj.__module__ = name  # defined in the remote module, as far as any lookup can tell
        setattr(remote, obj.__name__, obj)
    return remote
