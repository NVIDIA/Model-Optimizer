# Adapted from https://github.com/sgl-project/SpecForge/blob/8ea5ca6/specforge/modeling/draft/dflash.py
# Copyright (c) 2025 sgl-project
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND MIT
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
import torch.nn.functional as F
from torch import nn
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP as _MLP_CLS  # noqa: N814
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm as _NORM_CLS  # noqa: N814
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RotaryEmbedding as _ROTARY_CLS,  # noqa: N814
)
from transformers.models.qwen3.modeling_qwen3 import repeat_kv
from transformers.models.qwen3.modeling_qwen3 import rotate_half as _rotate_half

from .modeling_final_norm import _maybe_apply_base_final_norm

__all__ = ["DFlashBaseModelOutput", "DFlashModule", "build_target_layer_ids"]


def _sink_attention_impl(q, k, v, attention_mask, sink_bias, scaling, dropout_p, training):
    """Eager attention with a learnable per-head sink logit.

    Free function rather than a method so ``torch.compile`` traces one graph shared by every
    draft layer, instead of one per module instance.

    Mirrors ``transformers``' GPT-OSS ``eager_attention_forward``, minus its explicit
    max-subtraction: ``F.softmax`` already subtracts the row max internally, so the extra
    step is a mathematical no-op, and under ``torch.compile`` its backward returns NaN
    sink gradients for a bf16 mask built from ``finfo.min`` (the forward stays finite, so
    this surfaces only as a dead sink parameter).
    """
    attn_weights = torch.matmul(q, k.transpose(2, 3)) * scaling
    if attention_mask is not None:
        # [B, 1, Q, KV] additive mask, already sliced to the kv length by the caller.
        attn_weights = attn_weights + attention_mask[..., : k.shape[-2]]

    sinks = sink_bias.view(1, -1, 1, 1).expand(
        attn_weights.shape[0], -1, attn_weights.shape[-2], -1
    )
    combined = torch.cat([attn_weights, sinks.to(attn_weights.dtype)], dim=-1)
    probs = F.softmax(combined, dim=-1, dtype=torch.float32).to(q.dtype)
    attn_weights = probs[..., :-1]  # drop the sink column
    attn_weights = F.dropout(attn_weights, p=dropout_p, training=training)
    return torch.matmul(attn_weights, v).transpose(1, 2).contiguous()


# Compiled lazily and cached, shared by every draft layer. The sink's extra softmax column
# cannot be expressed by a fused SDPA/flash kernel, so this path materializes the
# [B, H, Q, KV] logits; Inductor fuses the mask add, concat and softmax into one kernel,
# which on the released Nemotron-3.5 draft shape (B=4, H=32, Q=512, KV=2560, bf16, fwd+bwd)
# cuts a draft layer from 15.5 ms / 2.87 GiB to 8.4 ms / 1.13 GiB. Training shapes are static
# (the collator pads every sample to ``train_len``), so this compiles once rather than per
# batch. Compilation is deferred to first use so importing this module — and CPU-only unit
# tests, which never reach a sink layer — pay nothing.
_compiled_sink_attention = None


def _get_sink_attention_fn():
    """Return the sink attention implementation, compiling it on first use.

    No fallback is wired here on purpose. ``torch.compile`` is lazy — it returns a wrapper
    and compiles on first call — so a guard around this line would never fire, and catching
    at the call site instead cannot tell a compiler failure from a genuine error raised by
    the function itself. Set ``torch._dynamo.config.suppress_errors`` to fall back to eager
    on compiler errors; the compiled and uncompiled paths are numerically interchangeable.
    """
    global _compiled_sink_attention
    if _compiled_sink_attention is None:
        _compiled_sink_attention = torch.compile(_sink_attention_impl)
    return _compiled_sink_attention


@dataclass
class DFlashBaseModelOutput:
    """Output container for base model forward pass in DFlash training."""

    target_hidden: torch.Tensor  # concatenated hidden states from target layers [B, seq, N*H]
    logits: torch.Tensor | None = None  # base model logits [B, seq, vocab]

    @classmethod
    def from_offline_dict(
        cls, d: dict, base_model_norm=None, base_model_lm_head=None, need_logits=False
    ):
        """Construct from a dict of pre-computed base model outputs (offline training).

        ``aux_hidden_states`` is required — missing it raises KeyError at the entry point
        rather than producing a cryptic failure deeper in the forward.

        When ``need_logits`` (self-logit-distillation) and the producer didn't supply
        ``base_model_logits``, logits are reconstructed from the captured final hidden via
        ``base_model_lm_head`` — first re-applying the base final norm when the producer captured
        a pre-(final-)norm hidden (``base_hidden_prenorm``), so the reconstruction is correct
        regardless of capture format. Anything missing on that path raises rather than silently
        yielding None logits: no ``base_model_lm_head`` (ValueError), no captured hidden
        (KeyError), or a pre-norm hidden with no ``base_model_norm`` (feeding an un-normed hidden
        to lm_head would be a corrupt distillation target).
        """
        logits = d.get("base_model_logits")
        if need_logits and logits is None:
            if base_model_lm_head is None:
                raise ValueError(
                    "need_logits=True but base_model_lm_head is None; cannot reconstruct logits."
                )
            out_hiddens = d.get("base_model_hidden_states")
            if out_hiddens is None:
                raise KeyError("base_model_hidden_states")
            out_hiddens = _maybe_apply_base_final_norm(out_hiddens, d, base_model_norm)
            logits = base_model_lm_head(out_hiddens)
        return cls(
            target_hidden=d["aux_hidden_states"],
            logits=logits,
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

        # Learnable per-head attention sink (GPT-OSS / Nemotron formulation): one extra
        # logit per head, appended before the softmax and dropped after, so a head can put
        # probability mass "nowhere" instead of being forced to attend inside its window.
        # Named to match the deployed checkpoints' `self_attn.attention_sink_bias`.
        self.attention_sink_bias = (
            nn.Parameter(torch.zeros(self.num_heads))
            if getattr(config, "attention_sink_bias", False)
            else None
        )

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

        if self.attention_sink_bias is not None:
            if self.sliding_window is not None:
                # The eager sink path applies only the caller-supplied mask; a per-layer
                # window from config.layer_types would be silently dropped. DFlash windows
                # the context through the attention mask instead (dflash_swa_window_size),
                # so this combination is rejected rather than trained with the wrong mask.
                raise NotImplementedError(
                    "dflash_attention_sink is not supported together with a per-layer "
                    "sliding window from dflash_architecture_config.layer_types. Use "
                    "dflash_swa_window_size for the draft's sliding window instead."
                )
            attn_output = self._sink_attention(q, k, v, attention_mask)
        else:
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

    def _sink_attention(self, q, k, v, attention_mask):
        """Attention with a learnable per-head sink logit.

        The sink is an extra column appended to the attention logits before the softmax and
        dropped immediately after, so it consumes probability mass without contributing to
        the output. Fused SDPA/flash kernels cannot express that extra column, so the logits
        are materialized and the kernel fusion is left to ``torch.compile``. Runs only when
        ``dflash_attention_sink`` is enabled.

        Returns ``[B, q_len, num_heads, head_dim]`` to match the HF attention interface.
        """
        sink_bias = self.attention_sink_bias
        assert sink_bias is not None, "_sink_attention requires dflash_attention_sink=True"

        return _get_sink_attention_fn()(
            q,
            repeat_kv(k, self.num_key_value_groups),
            repeat_kv(v, self.num_key_value_groups),
            attention_mask,
            sink_bias,
            self.scaling,
            self.attention_dropout if self.training else 0.0,
            self.training,
        )


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
