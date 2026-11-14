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


def remote_module(name="kimi_k3_remote_stub"):
    """A stand-in for the remote modeling namespace: everything the adapter patches, as shipped.

    Its MoE block computes what the checkpoint's ``modeling_kimi_linear.py`` computes: the ``situ``
    activation in fp32, the sigmoid router with its correction bias, grouped routing,
    renormalization and scaling factor, RMSNorm, and ``moe_infer``'s sum in the routing weights'
    dtype. The adapter keeps the gate, the activation and the norm and swaps the experts, so the
    tests compare its fused block against this one.
    """
    remote = types.ModuleType(name)
    remote.calls = []

    def _get_situ_activation_params(config):
        beta = getattr(config, "activation_situ_beta", None)
        return beta or 1.0, getattr(config, "activation_situ_linear_beta", None)

    class SituAndMul(nn.Module):
        """``beta * tanh(gate / beta) * sigmoid(gate) * up`` over a full [gate; up] row, in fp32; a
        ``linear_beta`` soft-caps ``up`` the same way."""

        def __init__(self, beta=1.0, linear_beta=None):
            super().__init__()
            self.beta, self.linear_beta = beta, linear_beta

        def forward(self, gate_up):
            gate, up = gate_up.float().chunk(2, dim=-1)
            act = self.beta * torch.tanh(gate / self.beta) * torch.sigmoid(gate)
            if self.linear_beta is not None:
                up = self.linear_beta * torch.tanh(up / self.linear_beta)
            return (act * up).to(gate_up.dtype)

    def _gated(config, gate, up):
        """The gated activation of one MLP: ``situ`` over the [gate; up] row, else silu(gate) * up."""
        if config.hidden_act == "situ":
            return SituAndMul(*_get_situ_activation_params(config))(torch.cat([gate, up], dim=-1))
        return F.silu(gate) * up

    class KimiRMSNorm(nn.Module):
        def __init__(self, width, eps):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(width))
            self.eps = eps

        def forward(self, x):
            normed = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
            return self.weight * normed.to(x.dtype)

    class KimiBlockSparseMLP(nn.Module):
        """One expert as the checkpoint stores it: w1 = gate, w3 = up, w2 = down."""

        def __init__(self, config, hidden):
            super().__init__()
            self.config = config
            inter = config.moe_intermediate_size
            self.w1 = nn.Linear(hidden, inter, bias=False)
            self.w3 = nn.Linear(hidden, inter, bias=False)
            self.w2 = nn.Linear(inter, hidden, bias=False)

        def forward(self, x):
            return self.w2(_gated(self.config, self.w1(x), self.w3(x)))

    class KimiMLP(nn.Module):
        """The always-on shared experts, one MLP as wide as all of them."""

        def __init__(self, config, intermediate_size):
            super().__init__()
            self.config = config
            hidden = config.hidden_size
            self.gate_proj = nn.Linear(hidden, intermediate_size, bias=False)
            self.up_proj = nn.Linear(hidden, intermediate_size, bias=False)
            self.down_proj = nn.Linear(intermediate_size, hidden, bias=False)

        def forward(self, x):
            return self.down_proj(_gated(self.config, self.gate_proj(x), self.up_proj(x)))

    class KimiMoEGate(nn.Module):
        """The sigmoid router: scores in fp32, the experts picked on the scores plus a correction
        bias (within the best groups under grouped routing), and weighted by the plain scores,
        renormalized and scaled."""

        def __init__(self, config):
            super().__init__()
            self.top_k = config.num_experts_per_token
            self.num_expert_group = getattr(config, "num_expert_group", 1)
            self.topk_group = getattr(config, "topk_group", 1)
            self.moe_renormalize = config.moe_renormalize
            self.routed_scaling_factor = config.routed_scaling_factor
            self.weight = nn.Parameter(torch.randn(config.num_experts, config.hidden_size))
            self.e_score_correction_bias = nn.Parameter(torch.zeros(config.num_experts))

        def forward(self, hidden_states):
            flat = hidden_states.reshape(-1, hidden_states.shape[-1])
            scores = (flat.float() @ self.weight.float().t()).sigmoid()
            choice = scores + self.e_score_correction_bias
            if self.num_expert_group > 1 and self.num_expert_group > self.topk_group:
                groups = choice.view(len(choice), self.num_expert_group, -1)
                best = groups.topk(2, dim=-1).values.sum(-1).topk(self.topk_group, dim=-1).indices
                kept = torch.zeros(groups.shape[:2], dtype=torch.bool, device=groups.device)
                kept.scatter_(1, best, True)
                choice = groups.masked_fill(~kept[..., None], float("-inf")).view(len(choice), -1)
            idx = choice.topk(self.top_k, dim=-1, sorted=False).indices
            weights = scores.gather(1, idx)
            if self.top_k > 1 and self.moe_renormalize:
                weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
            return idx, weights * self.routed_scaling_factor

    class KimiSparseMoeBlock(nn.Module):
        """Latent projections around a per-expert Python loop, plus always-on shared experts."""

        def __init__(self, config):
            super().__init__()
            self.config = config
            # Kimi-K3's config has no use_latent_moe field: the block derives it, as vLLM does.
            self.use_latent_moe = getattr(config, "routed_expert_hidden_size", None) is not None
            self.latent_moe_use_norm = getattr(config, "latent_moe_use_norm", False)
            hidden = config.hidden_size
            width = config.routed_expert_hidden_size if self.use_latent_moe else hidden
            self.experts = nn.ModuleList(
                KimiBlockSparseMLP(config, width) for _ in range(config.num_experts)
            )
            self.gate = KimiMoEGate(config)
            if config.num_shared_experts is not None:
                shared_width = config.moe_intermediate_size * config.num_shared_experts
                self.shared_experts = KimiMLP(config, shared_width)
            if self.use_latent_moe:
                self.routed_expert_down_proj = nn.Linear(hidden, width, bias=False)
                self.routed_expert_up_proj = nn.Linear(width, hidden, bias=False)
                if self.latent_moe_use_norm:
                    self.routed_expert_norm = KimiRMSNorm(width, config.rms_norm_eps)

        def forward(self, hidden_states):
            idx, weights = self.gate(hidden_states)
            x = hidden_states.reshape(-1, hidden_states.shape[-1])
            if self.use_latent_moe:
                x = self.routed_expert_down_proj(x)
            y = self.moe_infer(x, idx, weights)
            if self.use_latent_moe:
                if self.latent_moe_use_norm:
                    y = self.routed_expert_norm(y)
                y = self.routed_expert_up_proj(y)
            y = y.reshape(hidden_states.shape)
            if self.config.num_shared_experts is not None:
                y = y + self.shared_experts(hidden_states)
            return y

        @torch.no_grad()
        def moe_infer(self, x, idx, weights):
            """Each token's experts on it in ``x``'s dtype; their results weighted in the routing
            weights' dtype, summed over the token's experts there, and cast back once."""
            results = x.new_empty(*idx.shape, x.shape[-1])
            for e, expert in enumerate(self.experts):
                token, slot = (idx == e).nonzero(as_tuple=True)
                if len(token):
                    results[token, slot] = expert(x[token])
            weighted = results.to(weights.dtype) * weights[..., None]
            return weighted.sum(dim=1).to(x.dtype)

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
        KimiRMSNorm,
        KimiBlockSparseMLP,
        KimiMLP,
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
