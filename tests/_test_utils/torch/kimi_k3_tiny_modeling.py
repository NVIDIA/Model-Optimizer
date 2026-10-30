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

"""A tiny Kimi-K3-shaped model, as remote code: tests copy it into a checkpoint as ``modeling_kimi_k3.py``.

It has the module tree, the names and the MoE the Kimi-K3 adapter patches, as the checkpoint's own
code builds them: a VLM-shaped ``KimiK3ForConditionalGeneration`` whose ``language_model`` is a
``KimiLinearForCausalLM`` (model type ``kimi_linear``), a dense first layer, and then
``block_sparse_moe`` blocks of per-expert ``w1``/``w3``/``w2`` in a latent space, with the latent norm,
shared experts and the ``situ`` gate. A per-token linear stands in for the attention, which the
expert-parallel path leaves alone.
"""

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast


class KimiLinearConfig(PretrainedConfig):
    model_type = "kimi_linear"

    def __init__(
        self,
        vocab_size=64,
        hidden_size=64,
        intermediate_size=64,
        routed_expert_hidden_size=32,
        moe_intermediate_size=32,
        num_experts=8,
        num_experts_per_token=2,
        num_hidden_layers=3,
        first_k_dense_replace=1,
        num_shared_experts=1,
        hidden_act="situ",
        situ_beta=1.5,
        situ_linear_beta=0.25,
        latent_moe_use_norm=True,
        rms_norm_eps=1e-6,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.routed_expert_hidden_size = routed_expert_hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.num_hidden_layers = num_hidden_layers
        self.first_k_dense_replace = first_k_dense_replace
        self.num_shared_experts = num_shared_experts
        self.hidden_act = hidden_act
        self.situ_beta = situ_beta
        self.situ_linear_beta = situ_linear_beta
        self.latent_moe_use_norm = latent_moe_use_norm
        self.rms_norm_eps = rms_norm_eps
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(**kwargs)


class KimiK3Config(PretrainedConfig):
    model_type = "kimi_k3"
    sub_configs = {"text_config": KimiLinearConfig}

    def __init__(self, text_config=None, **kwargs):
        if isinstance(text_config, dict):
            text_config = KimiLinearConfig(**text_config)
        self.text_config = text_config if text_config is not None else KimiLinearConfig()
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(**kwargs)


def _get_situ_activation_params(config):
    return config.situ_beta, config.situ_linear_beta


class SituAndMul(nn.Module):
    """``situ`` on a full ``[gate; up]`` row."""

    def __init__(self, beta, linear_beta):
        super().__init__()
        self.beta, self.linear_beta = beta, linear_beta

    def forward(self, gate_up):
        gate, up = gate_up.chunk(2, dim=-1)
        return gate * torch.sigmoid(self.beta * gate) * (up + self.linear_beta)


class KimiRMSNorm(nn.Module):
    def __init__(self, size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x):
        normed = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * normed).to(x.dtype)


class KimiMLP(nn.Module):
    """The dense MLP: the first layer's, and the shared experts'."""

    def __init__(self, hidden, inter):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class KimiBlockSparseMLP(nn.Module):
    """One routed expert, as the checkpoint stores it: ``w1`` = gate, ``w3`` = up, ``w2`` = down."""

    def __init__(self, config, hidden):
        super().__init__()
        inter = config.moe_intermediate_size
        self.w1 = nn.Linear(hidden, inter, bias=False)
        self.w3 = nn.Linear(hidden, inter, bias=False)
        self.w2 = nn.Linear(inter, hidden, bias=False)
        situ = config.hidden_act == "situ"
        self.act = SituAndMul(*_get_situ_activation_params(config)) if situ else None

    def forward(self, x):
        gate_up = torch.cat([self.w1(x), self.w3(x)], dim=-1)
        if self.act is not None:
            return self.w2(self.act(gate_up))
        gate, up = gate_up.chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)


class KimiMoEGate(nn.Module):
    """Sigmoid scores over the replicated router weight, normalized over each token's top k."""

    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_token
        self.weight = nn.Parameter(torch.empty(config.num_experts, config.hidden_size))

    def forward(self, hidden_states):
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        scores = torch.sigmoid(F.linear(flat.float(), self.weight.float()))
        weights, idx = torch.topk(scores, self.top_k, dim=-1)
        return idx, (weights / weights.sum(dim=-1, keepdim=True)).to(hidden_states.dtype)


class KimiSparseMoeBlock(nn.Module):
    """Latent projections around a per-expert loop, plus the always-on shared experts."""

    def __init__(self, config):
        super().__init__()
        self.config = config
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
                self.routed_expert_norm = KimiRMSNorm(width, config.rms_norm_eps)
        if config.num_shared_experts is not None:
            inter = config.moe_intermediate_size * config.num_shared_experts
            self.shared_experts = KimiMLP(hidden, inter)

    def forward(self, hidden_states):
        idx, weights = self.gate(hidden_states)
        x = hidden_states.reshape(-1, hidden_states.shape[-1])
        if self.use_latent_moe:
            x = self.routed_expert_down_proj(x)
        y = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):
            token, slot = torch.where(idx == e)
            if token.numel():
                y.index_add_(0, token, expert(x[token]) * weights[token, slot, None])
        if self.use_latent_moe:
            if self.latent_moe_use_norm:
                y = self.routed_expert_norm(y)
            y = self.routed_expert_up_proj(y)
        y = y.reshape(hidden_states.shape)
        if self.config.num_shared_experts is not None:
            y = y + self.shared_experts(hidden_states)
        return y


class KimiDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        hidden = config.hidden_size
        self.input_layernorm = KimiRMSNorm(hidden, config.rms_norm_eps)
        self.self_attn = nn.Linear(hidden, hidden, bias=False)  # no token mixing: not under test
        self.post_attention_layernorm = KimiRMSNorm(hidden, config.rms_norm_eps)
        if layer_idx < config.first_k_dense_replace:
            self.mlp = KimiMLP(hidden, config.intermediate_size)
        else:
            self.block_sparse_moe = KimiSparseMoeBlock(config)

    def forward(self, hidden_states):
        hidden_states = hidden_states + self.self_attn(self.input_layernorm(hidden_states))
        ffn = self.mlp if hasattr(self, "mlp") else self.block_sparse_moe
        return hidden_states + ffn(self.post_attention_layernorm(hidden_states))


class KimiLinearPreTrainedModel(PreTrainedModel):
    config_class = KimiLinearConfig
    base_model_prefix = "model"
    _no_split_modules = ["KimiDecoderLayer"]


class KimiLinearModel(KimiLinearPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            KimiDecoderLayer(config, i) for i in range(config.num_hidden_layers)
        )
        self.norm = KimiRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_init()

    def forward(self, input_ids):
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.norm(hidden_states)


class KimiLinearForCausalLM(KimiLinearPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = KimiLinearModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(self, input_ids=None, **kwargs):
        return CausalLMOutputWithPast(logits=self.lm_head(self.model(input_ids)))


class KimiK3PreTrainedModel(PreTrainedModel):
    config_class = KimiK3Config
    _no_split_modules = ["KimiDecoderLayer"]


class KimiK3ForConditionalGeneration(KimiK3PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.language_model = KimiLinearForCausalLM(config.text_config)
        self.post_init()

    def forward(self, input_ids=None, **kwargs):
        return self.language_model(input_ids=input_ids)
