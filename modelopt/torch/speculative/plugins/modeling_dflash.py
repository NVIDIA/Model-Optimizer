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

import copy
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

from .modeling_final_norm import _maybe_apply_base_final_norm

__all__ = ["DFlashBaseModelOutput", "DFlashModule", "build_target_layer_ids"]


@dataclass
class DFlashBaseModelOutput:
    """Output container for base model forward pass in DFlash training."""

    target_hidden: torch.Tensor  # concatenated hidden states from target layers [B, seq, N*H]
    logits: torch.Tensor | None = None  # base model logits [B, seq, vocab]
    # Post-final-norm base hidden [B, seq, H], i.e. lm_head's input. Consumers that only
    # need the base distribution at a handful of positions project THIS at those rows
    # instead of materialising (and then gathering out of) full-sequence logits.
    base_hidden: torch.Tensor | None = None

    @classmethod
    def from_offline_dict(
        cls,
        d: dict,
        base_model_norm=None,
        base_model_lm_head=None,
        need_logits=False,
        defer_lm_head=False,
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
        base_hidden = None
        if need_logits and logits is None:
            out_hiddens = d.get("base_model_hidden_states")
            if out_hiddens is None:
                raise KeyError("base_model_hidden_states")
            base_hidden = _maybe_apply_base_final_norm(out_hiddens, d, base_model_norm)
            if defer_lm_head:
                # Caller will project only the rows it needs; skip the full-sequence
                # [B, seq, vocab] materialisation entirely.
                return cls(target_hidden=d["aux_hidden_states"], base_hidden=base_hidden)
            if base_model_lm_head is None:
                raise ValueError(
                    "need_logits=True but base_model_lm_head is None; cannot reconstruct logits."
                )
            logits = base_model_lm_head(base_hidden)
        return cls(
            target_hidden=d["aux_hidden_states"],
            logits=logits,
            base_hidden=base_hidden,
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
        # DFlash/DSpark drafts attend bidirectionally: a block of draft tokens is
        # predicted in one shot, so those tokens must see each other. Serving must
        # agree -- vLLM resolves per-layer causality in
        # qwen3_dflash._dflash_layer_causal(): an explicit ``dflash_config.causal``
        # overrides all layers, otherwise a layer is causal only when
        # ``layer_types[i] == "sliding_attention"``. The exporter emits no
        # ``causal`` field for a plain full-attention draft, so it stays non-causal
        # on both sides. With ``dflash_swa_window_size`` set, the exporter instead
        # emits ``use_swa: True`` + an explicit ``causal: False`` and leaves
        # ``layer_types`` all-full, which keeps vLLM non-causal too. Only mark
        # layers ``sliding_attention`` if you also intend them to be CAUSAL at
        # serving time, and train them that way -- see _build_draft_attention_mask.
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

    def _attend(self, q, k, v, attention_mask, bsz, q_len):
        """Run attention and project, routing a FlexAttention BlockMask to the flex kernel.

        ``attention_mask`` is either the dense additive [B, 1, Q, KV] tensor (HF attention
        dispatch) or a BlockMask carrying the same predicate block-sparsely.
        """
        from .dflash_flex_attention import flex_attention_forward, is_block_mask

        if is_block_mask(attention_mask):
            dropout = 0.0 if not self.training else self.attention_dropout
            if dropout:
                raise ValueError(
                    "FlexAttention path does not support attention_dropout > 0 "
                    f"(got {dropout}); unset dflash_use_flex_attention."
                )
            attn_output = flex_attention_forward(q, k, v, attention_mask, self.scaling)
        else:
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
        return self.o_proj(attn_output.reshape(bsz, q_len, -1))

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

        return self._attend(q, k, v, attention_mask, bsz, q_len)


class DFlashGemma4Attention(DFlashAttention):
    """DFlash attention for a Gemma4-style draft.

    Two deltas versus the Qwen3-style :class:`DFlashAttention`:

    * ``attention_k_eq_v``: Gemma4 can derive V from the K projection instead of
      carrying a separate ``v_proj``, halving the KV parameters. vLLM's
      ``Gemma4DSparkAttention`` does exactly this (``v_src = k`` when
      ``use_k_eq_v``), and its fused context-KV precompute *asserts* every draft
      layer is built this way, so a draft trained with a separate ``v_proj``
      cannot be served by that path at all.
    * ``v_norm``: applied to V, with **no learnable weight**, mirroring vLLM's
      ``RMSNorm(..., has_weight=False)``. Plain (Qwen3) DFlash does not norm V.

    ``use_k_eq_v`` follows vLLM: full-attention layers only, and only when the
    config opts in. Sliding layers keep their own ``v_proj``.
    """

    def __init__(self, config, layer_idx):
        """Initialize Gemma4 draft attention with per-layer dims, dropping ``v_proj`` under k_eq_v."""
        super().__init__(config, layer_idx)
        layer_types = getattr(config, "layer_types", None)
        is_full = layer_types is None or layer_types[layer_idx] == "full_attention"
        self.use_k_eq_v = is_full and getattr(config, "attention_k_eq_v", False)

        # Gemma4's attention dims are PER LAYER: full-attention layers use a larger
        # ``global_head_dim`` and, under k_eq_v, a smaller ``num_global_key_value_heads``.
        # This mirrors vLLM's ``gemma4_layer_config`` (transformers_utils/configs/gemma4.py),
        # which Gemma4DSparkAttention calls to size q/k/o. Getting this wrong is silent:
        # the DSpark weight loader only fills names it finds, so a mis-shaped k_proj is
        # simply left randomly initialized.
        if is_full:
            self.head_dim = getattr(config, "global_head_dim", None) or self.head_dim
            if self.use_k_eq_v:
                self.num_kv_heads = (
                    getattr(config, "num_global_key_value_heads", None) or self.num_kv_heads
                )
            self.num_key_value_groups = self.num_heads // self.num_kv_heads
            # Keep ``head_dim**-0.5`` even though vLLM serves this draft with
            # ``scaling = 1.0``. That looks like a train/serve mismatch and it IS one, but
            # aligning it is measurably WORSE -- do not "fix" this again without repeating
            # the experiment below.
            #
            # vLLM's Gemma4MTPAttention (which Gemma4DSparkAttention inherits) hardcodes
            # 1.0, matching the Gemma 4 base, which documents that "unlike Gemma2/3,
            # query_pre_attn_scalar is NOT used here; Q/K norms with learnable weights
            # handle scaling implicitly". Two full lr 2e-3 runs, identical except for this
            # line, evaluated under REAL vLLM on 80q MT-Bench at num_spec=7:
            #
            #     step   trained 1/sqrt(512)   trained 1.0
            #     1000   2.0645                1.9710   (-4.5%)
            #     5000   2.5715                2.5234   (-1.9%)
            #
            # The reason is that ``q_norm`` is learnable, so it absorbs the scale: serving
            # a 1/sqrt(512)-trained draft at 1.0 costs only 0.3% (step 1000) to 2.2%
            # (step 5000), while TRAINING at 1.0 costs more than that. The small scale is
            # the better training configuration -- smoother attention, better conditioned --
            # and it transfers almost intact.
            #
            # An earlier estimate of a 7% mismatch penalty came from simulating scale 1.0
            # inside the hand-written harness rather than measuring vLLM, and overstated it.
            self.scaling = self.head_dim**-0.5
            attn_bias = getattr(config, "attention_bias", False)
            self.q_proj = nn.Linear(
                config.hidden_size, self.num_heads * self.head_dim, bias=attn_bias
            )
            self.k_proj = nn.Linear(
                config.hidden_size, self.num_kv_heads * self.head_dim, bias=attn_bias
            )
            self.o_proj = nn.Linear(
                self.num_heads * self.head_dim, config.hidden_size, bias=attn_bias
            )
            self.q_norm = _NORM_CLS(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = _NORM_CLS(self.head_dim, eps=config.rms_norm_eps)

        if self.use_k_eq_v:
            # Registered by the parent; drop it so it is neither trained nor exported.
            del self.v_proj
            self.v_proj = None
        elif is_full:
            self.v_proj = nn.Linear(
                config.hidden_size,
                self.num_kv_heads * self.head_dim,
                bias=getattr(config, "attention_bias", False),
            )
        # vLLM builds this as ``RMSNorm(..., has_weight=False)`` and the reference
        # checkpoint ships NO v_norm tensor, so keep the scale fixed at ones and
        # non-persistent: it must not appear in the exported state_dict.
        self.v_norm = _NORM_CLS(self.head_dim, eps=config.rms_norm_eps)
        del self.v_norm.weight
        self.v_norm.register_buffer("weight", torch.ones(self.head_dim), persistent=False)

    def _project_v(self, target_hidden, hidden_states, k_ctx, k_noise):
        """Return the V sequence, from K under k_eq_v or from ``v_proj`` otherwise."""
        if self.use_k_eq_v:
            return k_ctx, k_noise
        return self.v_proj(target_hidden), self.v_proj(hidden_states)

    def forward(self, hidden_states, target_hidden, position_embeddings, attention_mask=None):
        """Forward with KV injection; V is normed and, under k_eq_v, shares K's projection."""
        bsz, q_len, _ = hidden_states.shape
        ctx_len = target_hidden.shape[1]

        q = self.q_proj(hidden_states).view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)

        k_ctx = self.k_proj(target_hidden)
        k_noise = self.k_proj(hidden_states)
        k = torch.cat([k_ctx, k_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
        k = self.k_norm(k).transpose(1, 2)

        v_ctx, v_noise = self._project_v(target_hidden, hidden_states, k_ctx, k_noise)
        v = torch.cat([v_ctx, v_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
        # vLLM norms V (no RoPE on V), unlike the Qwen3-style path.
        v = self.v_norm(v).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        return self._attend(q, k, v, attention_mask, bsz, q_len)


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


class DFlashGemma4DecoderLayer(nn.Module):
    """Draft decoder layer matching Gemma4's block, with KV injection.

    Gemma4 wraps each sub-block in a *pair* of norms ("sandwich norm") and scales
    the layer output by a learned ``layer_scalar``, where Qwen3 uses a single
    pre-norm per sub-block. vLLM's ``Gemma4MTPDecoderLayer`` -- which
    ``Gemma4DSparkDecoderLayer`` inherits -- looks up
    ``pre_feedforward_layernorm`` / ``post_feedforward_layernorm`` /
    ``layer_scalar`` by name, and its DSpark weight loader silently leaves any
    parameter it cannot find randomly initialized. A Qwen3-shaped draft
    therefore *loads without error* and produces garbage, so the shapes must
    match exactly.

    The residual/norm order below mirrors ``Gemma4MTPDecoderLayer.forward``:
    norm -> attn -> norm -> +residual -> norm -> mlp -> norm -> +residual, then
    scale by ``layer_scalar``.
    """

    def __init__(self, config, layer_idx):
        """Initialize a Gemma4-style draft layer (sandwich norms + layer scalar)."""
        super().__init__()
        self.self_attn = DFlashGemma4Attention(config, layer_idx)
        self.mlp = _MLP_CLS(config)
        self.input_layernorm = _NORM_CLS(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = _NORM_CLS(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = _NORM_CLS(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = _NORM_CLS(config.hidden_size, eps=config.rms_norm_eps)
        # A buffer (not a parameter) to match vLLM's `register_buffer`, so the
        # exported tensor name and shape line up with the reference checkpoint.
        self.register_buffer("layer_scalar", torch.ones(1))

    def forward(self, hidden_states, target_hidden, position_embeddings, attention_mask=None):
        """Forward with sandwich norms, KV injection, and the layer scalar."""
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, target_hidden, position_embeddings, attention_mask
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = hidden_states + residual

        return hidden_states * self.layer_scalar


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
        # Gemma4 drafts need Gemma4's block shape (sandwich norms + layer_scalar,
        # optional k_eq_v); everything else keeps the Qwen3-style block.
        layer_cls = (
            DFlashGemma4DecoderLayer
            if str(getattr(config, "model_type", "")).startswith("gemma4")
            else DFlashDecoderLayer
        )
        self.layers = nn.ModuleList(
            [layer_cls(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = _NORM_CLS(config.hidden_size, eps=config.rms_norm_eps)
        self._rotary_config = config  # Used by _maybe_init_rotary_emb
        self._gemma4_rope_kinds = self._build_gemma4_rope_kinds(config)
        self._layer_types = list(getattr(config, "layer_types", []) or [])

        # Explicit weight init is needed because DFlashModule is instantiated via
        # mtsp.convert() AFTER the base model's post_init() has already run, so HF's
        # automatic _init_weights walk doesn't reach these new layers.
        self._init_weights(config)

    def _maybe_init_rotary_emb(self, device=None):
        """Lazily initialize rotary embeddings on first forward call.

        Same pattern as EAGLE3's _maybe_init_rope. Avoids creating rotary_emb
        during __init__ (which runs on meta device during from_pretrained),
        preventing the meta-tensor inv_freq issue on checkpoint resume.

        Gemma4 needs one module PER attention kind, not one for the whole draft:
        its full-attention layers use ``global_head_dim`` while sliding layers use
        ``head_dim``, and the two kinds carry different ``rope_parameters`` (theta
        1e6 vs 1e4). vLLM builds RoPE per layer for exactly this reason; a single
        shared module silently mismatches the head dim on one of the two kinds.
        """
        # The shared module is only used when there are no per-kind ones. Building it anyway
        # would crash on a Gemma4 draft: its config carries the NESTED, dict-of-dicts
        # ``rope_parameters`` (keyed by attention kind), and ``_ROTARY_CLS`` indexes it as a
        # flat dict via ``rope_parameters["rope_type"]``. The per-kind configs built below
        # each hold a flattened single-kind dict, so they are the ones that can be built.
        if not self._gemma4_rope_kinds and not hasattr(self, "rotary_emb"):
            self.rotary_emb = _ROTARY_CLS(config=self._rotary_config, device=device)
        if self._gemma4_rope_kinds and not hasattr(self, "rotary_emb_by_kind"):
            self.rotary_emb_by_kind = nn.ModuleDict(
                {
                    kind: _ROTARY_CLS(config=cfg, device=device)
                    for kind, cfg in self._gemma4_rope_kinds.items()
                }
            )

    @staticmethod
    def _build_gemma4_rope_kinds(config):
        """Per-attention-kind rotary configs for a Gemma4 draft, or ``{}`` otherwise.

        Returns a shallow copy of ``config`` per distinct ``layer_types`` entry with
        ``head_dim`` and ``rope_parameters`` resolved for that kind.
        """
        if not str(getattr(config, "model_type", "")).startswith("gemma4"):
            return {}
        layer_types = getattr(config, "layer_types", None)
        if not layer_types:
            return {}
        rope_params = getattr(config, "rope_parameters", None)
        kinds = {}
        for kind in dict.fromkeys(layer_types):
            cfg = copy.copy(config)
            if kind == "full_attention":
                cfg.head_dim = getattr(config, "global_head_dim", None) or config.head_dim
            entry = rope_params.get(kind) if isinstance(rope_params, dict) else None
            if entry is None and isinstance(rope_params, dict):
                # The RoPE kind need not equal the LAYER kind. `rope_attention_kind` lets a
                # recipe inherit e.g. the sliding entry for an all-full_attention draft, so
                # rope_params can hold exactly one entry keyed by a name absent from
                # layer_types. Fall back to that sole entry -- leaving rope_parameters
                # nested here makes the rotary class raise KeyError('rope_type'), since it
                # indexes rope_parameters as a FLAT dict.
                sole = [v for v in rope_params.values() if isinstance(v, dict)]
                entry = sole[0] if len(sole) == 1 else None
            if isinstance(entry, dict):
                cfg.rope_parameters = dict(entry)
                cfg.rope_theta = cfg.rope_parameters.get("rope_theta", config.rope_theta)
            kinds[kind] = cfg
        return kinds

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
        per_kind = {
            kind: emb(hidden_states, position_ids)
            for kind, emb in getattr(self, "rotary_emb_by_kind", {}).items()
        }
        # A Gemma4 draft has per-kind modules and NO shared one (its nested rope_parameters
        # cannot build a single rotary module — see _maybe_init_rotary_emb), so fall back to
        # the first kind rather than to a `self.rotary_emb` that does not exist.
        position_embeddings = (
            next(iter(per_kind.values()))
            if per_kind
            else self.rotary_emb(hidden_states, position_ids)
        )

        for layer_idx, layer in enumerate(self.layers):
            layer_pos = position_embeddings
            if per_kind and layer_idx < len(self._layer_types):
                layer_pos = per_kind.get(self._layer_types[layer_idx], position_embeddings)
            hidden_states = layer(hidden_states, target_hidden, layer_pos, attention_mask)

        return self.norm(hidden_states)
