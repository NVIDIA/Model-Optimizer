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

"""A stand-in for Kimi-K3's own remote modeling code: the classes and functions its adapter patches,
with the per-expert ``nn.ModuleList`` MoE the checkpoint's code ships, for tests on tiny CPU tensors.
"""

import types

import torch
import torch.nn.functional as F
from torch import nn


def situ(gate_up, beta, linear_beta):
    """The stand-in ``situ`` gated activation over a full [gate; up] row."""
    gate, up = gate_up.chunk(2, dim=-1)
    return gate * torch.sigmoid(beta * gate) * (up + linear_beta)


def remote_module(name="kimi_k3_remote_stub"):
    """A stand-in for the remote modeling namespace: everything the adapter patches, as shipped."""
    remote = types.ModuleType(name)
    remote.calls = []

    def _get_situ_activation_params(config):
        return config.situ_beta, config.situ_linear_beta

    class SituAndMul(nn.Module):
        def __init__(self, beta, linear_beta):
            super().__init__()
            self.beta, self.linear_beta = beta, linear_beta

        def forward(self, gate_up):
            return situ(gate_up, self.beta, self.linear_beta)

    class KimiBlockSparseMLP(nn.Module):
        """One expert as the checkpoint stores it: w1 = gate, w3 = up, w2 = down."""

        def __init__(self, config, hidden):
            super().__init__()
            inter = config.moe_intermediate_size
            self.w1 = nn.Linear(hidden, inter, bias=False)
            self.w3 = nn.Linear(hidden, inter, bias=False)
            self.w2 = nn.Linear(inter, hidden, bias=False)
            situ = config.hidden_act == "situ"
            self.situ_params = _get_situ_activation_params(config) if situ else None

        def forward(self, x):
            gate, up = self.w1(x), self.w3(x)
            if self.situ_params is None:
                return self.w2(F.silu(gate) * up)
            return self.w2(situ(torch.cat([gate, up], dim=-1), *self.situ_params))

    class KimiMoEGate(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.top_k = config.num_experts_per_token
            self.weight = nn.Parameter(torch.randn(config.num_experts, config.hidden_size))

        def forward(self, hidden_states):
            flat = hidden_states.reshape(-1, hidden_states.shape[-1])
            weights, idx = torch.topk(torch.sigmoid(flat @ self.weight.t()), self.top_k, dim=-1)
            return idx, weights

    class KimiSparseMoeBlock(nn.Module):
        """Latent projections around a per-expert Python loop, plus always-on shared experts."""

        def __init__(self, config):
            super().__init__()
            self.config = config
            # Kimi-K3's config has no use_latent_moe field: the block derives it, as vLLM does.
            self.use_latent_moe = config.routed_expert_hidden_size is not None
            self.latent_moe_use_norm = config.latent_moe_use_norm
            hidden = config.hidden_size
            width = config.routed_expert_hidden_size if self.use_latent_moe else hidden
            self.experts = nn.ModuleList(
                KimiBlockSparseMLP(config, width) for _ in range(config.num_experts)
            )
            self.gate = KimiMoEGate(config)
            if self.use_latent_moe:
                self.routed_expert_down_proj = nn.Linear(hidden, width, bias=False)
                self.routed_expert_up_proj = nn.Linear(width, hidden, bias=False)
                if config.latent_moe_use_norm:
                    self.routed_expert_norm = nn.LayerNorm(width)
            if config.num_shared_experts is not None:
                self.shared_experts = KimiBlockSparseMLP(config, hidden)

        def forward(self, hidden_states):
            idx, weights = self.gate(hidden_states)
            x = hidden_states.reshape(-1, hidden_states.shape[-1])
            if self.use_latent_moe:
                x = self.routed_expert_down_proj(x)
            y = torch.stack(
                [
                    sum(w * self.experts[int(e)](token) for e, w in zip(top_e, top_w))
                    for token, top_e, top_w in zip(x, idx, weights)
                ]
            )
            if self.use_latent_moe:
                if self.latent_moe_use_norm:
                    y = self.routed_expert_norm(y)
                y = self.routed_expert_up_proj(y)
            y = y.reshape(hidden_states.shape)
            if self.config.num_shared_experts is not None:
                y = y + self.shared_experts(hidden_states)
            return y

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
        _get_situ_activation_params,
        SituAndMul,
        KimiBlockSparseMLP,
        KimiMoEGate,
        KimiSparseMoeBlock,
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
