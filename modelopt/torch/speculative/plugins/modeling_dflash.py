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

"""DFlash draft model architecture (DFlashModule) and related components.

Draft model components use Qwen3 (MLP, RMSNorm, RotaryEmbedding) from
``transformers.models.qwen3``, matching z-lab's reference checkpoint format.
The draft architecture is independent of the target model.
"""

from dataclasses import dataclass

import torch
from torch import nn
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP as _MLP_CLS  # noqa: N814
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm as _NORM_CLS  # noqa: N814
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RotaryEmbedding as _ROTARY_CLS,  # noqa: N814
)
from transformers.models.qwen3.modeling_qwen3 import rotate_half as _rotate_half

__all__ = ["DFlashBaseModelOutput", "DFlashModule", "build_target_layer_ids"]


@dataclass
class DFlashBaseModelOutput:
    """Output container for base model forward pass in DFlash training."""

    target_hidden: torch.Tensor  # concatenated hidden states from target layers [B, seq, N*H]
    logits: torch.Tensor | None = None  # base model logits [B, seq, vocab]

    @classmethod
    def from_offline_dict(cls, d: dict):
        """Construct from a dict of pre-computed base model outputs (offline training).

        ``aux_hidden_states`` is required — missing it raises KeyError at the entry point
        rather than producing a cryptic failure deeper in the forward.
        """
        return cls(
            target_hidden=d["aux_hidden_states"],
            logits=d.get("base_model_logits"),
        )


def build_target_layer_ids(num_target_layers, num_draft_layers):
    """Select layers uniformly from the target model for feature extraction."""
    if num_target_layers < num_draft_layers:
        raise ValueError(
            f"num_target_layers ({num_target_layers}) must be >= num_draft_layers ({num_draft_layers})"
        )
    if num_draft_layers == 1:
        return [num_target_layers // 2]
    start = min(1, num_target_layers - 1)
    end = max(start, num_target_layers - 3)
    span = end - start
    return [round(start + (i * span) / (num_draft_layers - 1)) for i in range(num_draft_layers)]


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply RoPE. Q uses last q_len positions, K uses all positions."""
    cos = cos.unsqueeze(1)  # [B, 1, seq, dim]
    sin = sin.unsqueeze(1)
    q_len = q.size(2)
    q_embed = (q * cos[:, :, -q_len:, :]) + (_rotate_half(q) * sin[:, :, -q_len:, :])
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class DFlashAttention(nn.Module):
    """Attention with KV injection, using HF's attention dispatch."""

    def __init__(self, config, layer_idx):
        """Initialize DFlash attention with KV injection projections and QK-norm."""
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = getattr(config, "attention_dropout", 0.0)
        self.is_causal = False

        attn_bias = getattr(config, "attention_bias", False)
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=attn_bias)
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=attn_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=attn_bias
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=attn_bias)

        self.q_norm = _NORM_CLS(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = _NORM_CLS(self.head_dim, eps=config.rms_norm_eps)

        # Resolve HF attention function
        self._attn_fn = None
        # Qwen3 uses sliding window attention on some layers (config.layer_types)
        if hasattr(config, "layer_types") and hasattr(config, "sliding_window"):
            is_sliding = config.layer_types[layer_idx] == "sliding_attention"
            self.sliding_window = config.sliding_window if is_sliding else None
        else:
            self.sliding_window = None

    def _get_attn_fn(self):
        """Lazily resolve the HF attention function (default: sdpa)."""
        if self._attn_fn is not None:
            return self._attn_fn
        impl = self.config._attn_implementation  # default set in dflash/default_config.py
        self._attn_fn = ALL_ATTENTION_FUNCTIONS.get(impl, ALL_ATTENTION_FUNCTIONS["sdpa"])
        return self._attn_fn

    def forward(self, hidden_states, target_hidden, position_embeddings, attention_mask=None):
        """Forward with KV injection.

        Q is projected from the noise block (draft token embeddings: [anchor, mask, mask, ...]).
        K and V are projected from the concatenation of target hidden states (context from the
        base model) and noise block, so the draft can attend to both context and its own block.
        """
        bsz, q_len, _ = hidden_states.shape
        ctx_len = target_hidden.shape[1]

        # Q from noise block only (the draft tokens being predicted), with QK-norm
        q = self.q_proj(hidden_states).view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)

        # K from context + noise, with QK-norm
        k_ctx = self.k_proj(target_hidden)
        k_noise = self.k_proj(hidden_states)
        k = torch.cat([k_ctx, k_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
        k = self.k_norm(k).transpose(1, 2)

        # V from context + noise (no norm)
        v_ctx = self.v_proj(target_hidden)
        v_noise = self.v_proj(hidden_states)
        v = (
            torch.cat([v_ctx, v_noise], dim=1)
            .view(bsz, ctx_len + q_len, -1, self.head_dim)
            .transpose(1, 2)
        )

        # RoPE
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Use HF's attention dispatch (handles GQA internally)
        attn_fn = self._get_attn_fn()
        attn_output, _ = attn_fn(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
        )
        attn_output = attn_output.reshape(bsz, q_len, -1)
        return self.o_proj(attn_output)


class DFlashDecoderLayer(nn.Module):
    """Draft decoder layer with KV injection."""

    def __init__(self, config, layer_idx):
        """Initialize decoder layer with attention, MLP, and layer norms."""
        super().__init__()
        self.self_attn = DFlashAttention(config, layer_idx)
        self.mlp = _MLP_CLS(config)
        self.input_layernorm = _NORM_CLS(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = _NORM_CLS(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, target_hidden, position_embeddings, attention_mask=None):
        """Forward pass with residual connections."""
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, target_hidden, position_embeddings, attention_mask
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class DFlashModule(nn.Module):
    """DFlash draft module using Qwen3 components (MLP, RMSNorm, RotaryEmbedding)."""

    def __init__(self, config):
        """Initialize DFlash module with feature fusion, decoder layers, and rotary embeddings."""
        super().__init__()
        self.config = config
        self.block_size = config.block_size

        # Feature fusion
        num_fused_layers = len(config.target_layer_ids)
        self.fc = nn.Linear(num_fused_layers * config.hidden_size, config.hidden_size, bias=False)
        self.hidden_norm = _NORM_CLS(config.hidden_size, eps=config.rms_norm_eps)

        # Decoder layers
        self.layers = nn.ModuleList(
            [DFlashDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = _NORM_CLS(config.hidden_size, eps=config.rms_norm_eps)
        self._rotary_config = config  # Used by _maybe_init_rotary_emb

        # Explicit weight init is needed because DFlashModule is instantiated via
        # mtsp.convert() AFTER the base model's post_init() has already run, so HF's
        # automatic _init_weights walk doesn't reach these new layers.
        self._init_weights(config)

    def _maybe_init_rotary_emb(self, device=None):
        """Lazily initialize rotary embeddings on first forward call.

        Same pattern as EAGLE3's _maybe_init_rope. Avoids creating rotary_emb
        during __init__ (which runs on meta device during from_pretrained),
        preventing the meta-tensor inv_freq issue on checkpoint resume.
        """
        if not hasattr(self, "rotary_emb"):
            self.rotary_emb = _ROTARY_CLS(config=self._rotary_config, device=device)

    def _init_weights(self, config):
        """Initialize weights matching HF PreTrainedModel._init_weights."""
        std = getattr(config, "initializer_range", 0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, noise_embedding, target_hidden, position_ids, attention_mask=None):
        """Forward with feature fusion, KV injection, and position embeddings."""
        hidden_states = noise_embedding
        target_hidden = self.hidden_norm(self.fc(target_hidden))
        self._maybe_init_rotary_emb(device=hidden_states.device)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states, target_hidden, position_embeddings, attention_mask)

        return self.norm(hidden_states)
