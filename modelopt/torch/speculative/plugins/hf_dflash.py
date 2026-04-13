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

"""DFlash speculative decoding plugin for HuggingFace models.

Architecture:
- Feature Fusion: multi-layer target hidden states → FC + RMSNorm
- KV Injection: fused features as K/V in every draft layer with QK-norm
- Parallel Drafting: mask_token_id for unknown positions, bidirectional within blocks
- Random anchor sampling with exponential loss decay
- Logit distillation from target model

Reference: "DFlash: Block Diffusion for Flash Speculative Decoding" (arXiv:2602.06036)

Draft model components:
    The draft model currently uses Qwen3 components (MLP, RMSNorm, RotaryEmbedding)
    from ``transformers.models.qwen3``, matching z-lab's reference checkpoint format.
    Qwen3 sliding window attention is supported via ``config.layer_types``.
    The draft architecture is independent of the target model — any target model can
    be used as long as it provides hidden states.

    To add support for other draft architectures:

    Qwen3MoE (MoE MLP):
        1. Import ``Qwen3MoeMLP`` from ``transformers.models.qwen3_moe``
        2. Add a config flag (e.g., ``use_moe``) in ``dflash_architecture_config``
        3. In ``DFlashDecoderLayer.__init__``, select MLP based on the flag
        RMSNorm, RotaryEmbedding, and attention are shared across Qwen3 variants.

    MLA (Multi-head Latent Attention, e.g., DeepseekV3/Kimi-K2):
        MLA compresses K/V into a low-rank latent space. To support MLA in DFlash:
        1. Replace ``DFlashAttention`` with an MLA-aware variant that handles
           compressed KV injection (project target_hidden through MLA's down/up
           projections before concatenating with noise K/V)
        2. Handle lazy rope initialization (see ``_setup_kimi_k2_decoder`` in
           ``modelopt.torch.speculative.utils`` for the EAGLE3 approach)
        3. The ``_apply`` meta buffer fix in ``DFlashModule`` already handles the
           lazy rope pattern needed for MLA models.
"""

import logging

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config as _Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP as _MLP_CLS  # noqa: N814
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm as _NORM_CLS  # noqa: N814
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RotaryEmbedding as _ROTARY_CLS,  # noqa: N814
)
from transformers.models.qwen3.modeling_qwen3 import rotate_half as _rotate_half
from transformers.trainer_pt_utils import LabelSmoother
from transformers.utils import ModelOutput

from ..dflash.conversion import DFlashDMRegistry
from ..dflash.dflash_model import DFlashModel
from .modeling_fakebase import _BASE_MODEL_PATHS, _EMBED_TOKENS_PATHS, _LM_HEAD_PATHS

logger = logging.getLogger(__name__)

__all__ = ["HFDFlashModel"]


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


@DFlashDMRegistry.register({PreTrainedModel: "hf.PreTrainedModel"})
class HFDFlashModel(DFlashModel):
    """DFlash Model for HuggingFace transformers."""

    @property
    def _base_model(self):
        return self.get_submodule(self.base_model_path)

    @property
    def _base_model_embeddings(self):
        return self.get_submodule(self.base_model_embeddings_path)

    @property
    def _base_model_lm_head(self):
        return self.get_submodule(self.base_model_lm_head_path)

    @property
    def _base_llm_config(self):
        return (
            getattr(self.config, "text_config", None)
            or getattr(self.config, "llm_config", None)
            or self.config
        )

    def _find_base_model_parts(self):
        """Locate base model submodules (backbone, embeddings, lm_head) by probing known paths.

        Reuses the shared path constants from modeling_fakebase (same as EAGLE).
        """
        for name, paths in {
            "base_model_path": _BASE_MODEL_PATHS,
            "base_model_embeddings_path": _EMBED_TOKENS_PATHS,
            "base_model_lm_head_path": _LM_HEAD_PATHS,
        }.items():
            for path in paths:
                try:
                    submodule = self.get_submodule(path)
                    assert isinstance(submodule, torch.nn.Module)
                    setattr(self, name, path)
                    break
                except Exception:
                    continue
            else:
                raise ValueError(f"Part {name} not found in model")

    def modify(self, config):
        """Initialize DFlash draft module."""
        super().modify(config)

        base_config = self._base_llm_config
        # Use Qwen3Config (not generic PretrainedConfig) so rope_parameters is
        # auto-populated from rope_theta. DFlash draft uses Qwen3 components.
        self.dflash_config = _Qwen3Config(**config.dflash_architecture_config)

        # hidden_size and vocab_size MUST match the base model.
        self.dflash_config.hidden_size = base_config.hidden_size
        self.dflash_config.vocab_size = base_config.vocab_size

        # Inherit architecture settings from base model when not specified by user.
        # Static defaults (hidden_act, attention_bias, etc.) are in dflash/default_config.py.
        # NOTE: rope_scaling is intentionally excluded. DFlash draft uses Qwen3
        # RotaryEmbedding which only supports standard RoPE. Inheriting M-RoPE
        # config from multimodal models (e.g. Qwen3.5) would be incorrect.
        _base_model_attrs = [
            "max_position_embeddings",
            "intermediate_size",
            "num_attention_heads",
            "num_key_value_heads",
            "rope_theta",
            "rope_type",
            "rope_interleaved",
            "rms_norm_eps",
        ]
        for attr in _base_model_attrs:
            if not hasattr(self.dflash_config, attr) or getattr(self.dflash_config, attr) is None:
                if hasattr(base_config, attr):
                    setattr(self.dflash_config, attr, getattr(base_config, attr))

        self.dflash_config.head_dim = getattr(
            self.dflash_config,
            "head_dim",
            self.dflash_config.hidden_size // self.dflash_config.num_attention_heads,
        )
        self.dflash_config.block_size = self.dflash_block_size

        # Target layer IDs
        num_target_layers = base_config.num_hidden_layers
        num_draft_layers = self.dflash_config.num_hidden_layers
        self.target_layer_ids = build_target_layer_ids(num_target_layers, num_draft_layers)
        self.dflash_config.target_layer_ids = self.target_layer_ids

        # mask_token_id: set in DFlashConfig (or auto-detected by main.py from tokenizer)
        mask_id = config.dflash_mask_token_id
        if mask_id is None:
            raise ValueError(
                "dflash_mask_token_id is required. Set it in the config YAML "
                "(dflash.dflash_mask_token_id=TOKEN_ID) or let main.py auto-detect "
                "from tokenizer.mask_token_id."
            )
        self.mask_token_id = mask_id
        logger.info("DFlash mask_token_id: %s", self.mask_token_id)

        # Freeze base model
        if self.dflash_freeze_base_model:
            for param in self.parameters():
                param.requires_grad = False

        self._find_base_model_parts()

        self.dflash_module = DFlashModule(self.dflash_config)
        # Match base model dtype/device. Skip if base is on meta (during from_pretrained
        # restore — the model will be moved to the correct device after weight loading).
        base_device = next(self._base_model.layers[-1].parameters()).device
        if base_device.type != "meta":
            self.dflash_module.to(self._base_model.dtype).to(base_device)

        self.is_quantized = False
        self._num_anchors = self.dflash_num_anchors

    def get_exporter(self):
        """Get the exporter for the DFlash draft model."""
        from modelopt.torch.export.plugins.hf_spec_export import DFlashExporter

        return DFlashExporter(self)

    def _sample_anchor_positions(self, seq_len, loss_mask, device):
        """Randomly sample anchor positions per sample.

        Returns (anchor_positions [B, N], block_keep_mask [B, N]).

        TODO: Fix the random seed per epoch (change between epochs) so that anchor
        positions are deterministic within an epoch. This would allow caching the derived
        masks and position IDs across steps while preserving the same data augmentation
        effect. Currently, anchors are re-sampled every forward pass.
        """
        bs = self.dflash_block_size
        bsz = loss_mask.shape[0]
        max_anchor = max(seq_len - bs, 0)
        num_anchors = getattr(self, "_num_anchors", 512)

        valid = loss_mask[:, : max_anchor + 1] > 0.5
        valid_counts = valid.sum(dim=1)
        max_n = min(num_anchors, int(valid_counts.max().item()) - 1)

        if max_n <= 0:
            # No valid anchors — return empty
            anchors = torch.zeros(bsz, 1, dtype=torch.long, device=device)
            keep = torch.zeros(bsz, 1, dtype=torch.bool, device=device)
            return anchors, keep

        indices = torch.arange(max_anchor + 1, device=device).unsqueeze(0).expand(bsz, -1)
        masked_indices = torch.where(valid, indices, torch.tensor(seq_len + 1, device=device))

        random_vals = torch.rand(bsz, max_anchor + 1, device=device)
        random_vals = torch.where(valid, random_vals, torch.tensor(2.0, device=device))

        _, sorted_idx = random_vals.sort(dim=1)
        gathered = torch.gather(masked_indices, 1, sorted_idx)
        anchors = gathered[:, :max_n].sort(dim=1).values

        keep = torch.arange(max_n, device=device).unsqueeze(0) < valid_counts.unsqueeze(1).clamp(
            max=max_n
        )
        anchors = torch.where(keep, anchors, torch.tensor(0, dtype=torch.long, device=device))
        return anchors, keep

    def _build_noise_embedding(self, input_ids, anchor_positions, block_keep_mask, n_blocks):
        """Build noise embeddings: anchor token at block start, mask_token elsewhere."""
        bsz, seq_len = input_ids.shape
        block_size = self.dflash_block_size
        device = input_ids.device

        noise_ids = torch.full(
            (bsz, n_blocks * block_size), self.mask_token_id, dtype=torch.long, device=device
        )
        block_starts = torch.arange(n_blocks, device=device) * block_size
        block_starts_exp = block_starts.unsqueeze(0).expand(bsz, -1)
        valid_anchors = anchor_positions.clamp(0, seq_len - 1)
        anchor_tokens = torch.gather(input_ids, 1, valid_anchors)
        batch_idx = torch.arange(bsz, device=device).unsqueeze(1).expand(bsz, n_blocks)
        noise_ids[batch_idx, block_starts_exp] = torch.where(
            block_keep_mask,
            anchor_tokens,
            torch.tensor(self.mask_token_id, dtype=torch.long, device=device),
        )
        return self._base_model_embeddings(noise_ids)

    def _build_position_ids(self, seq_len, anchor_positions, device):
        """Build position IDs: context [0..S-1], draft blocks [anchor+0..anchor+B-1]."""
        bsz = anchor_positions.shape[0]
        block_size = self.dflash_block_size

        ctx_pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        offsets = torch.arange(block_size, device=device).view(1, 1, -1)
        draft_pos = (anchor_positions.unsqueeze(-1) + offsets).view(bsz, -1)
        return torch.cat([ctx_pos, draft_pos], dim=1)

    def _build_draft_attention_mask(
        self, seq_len, anchor_positions, block_keep_mask, n_blocks, dtype, device
    ):
        """Build SDPA attention mask: context (causal) + draft (bidirectional within block)."""
        bsz = anchor_positions.shape[0]
        block_size = self.dflash_block_size
        q_len = n_blocks * block_size
        kv_len = seq_len + q_len

        q_indices = torch.arange(q_len, device=device).view(1, 1, -1, 1)
        kv_indices = torch.arange(kv_len, device=device).view(1, 1, 1, -1)
        q_block_ids = q_indices // block_size

        anchor_exp = anchor_positions.view(bsz, 1, n_blocks, 1).repeat_interleave(block_size, dim=2)

        # Context: kv < S and kv < anchor
        mask_ctx = (kv_indices < seq_len) & (kv_indices < anchor_exp)
        # Draft: kv >= S and same block
        is_draft = kv_indices >= seq_len
        kv_block_ids = (kv_indices - seq_len) // block_size
        mask_draft = is_draft & (q_block_ids == kv_block_ids)
        # Valid block
        valid_block = block_keep_mask.view(bsz, 1, n_blocks, 1).repeat_interleave(block_size, dim=2)

        final_mask = (mask_ctx | mask_draft) & valid_block  # [B, 1, Q, KV]

        # Convert bool mask to float additive mask for SDPA
        attn_mask = torch.zeros(bsz, 1, q_len, kv_len, device=device, dtype=dtype)
        attn_mask.masked_fill_(~final_mask, torch.finfo(dtype).min)
        return attn_mask

    def _compute_loss(
        self, logits, input_ids, anchor_positions, block_keep_mask, loss_mask, base_logits=None
    ):
        """Compute weighted cross-entropy (or KD) loss and accuracy.

        Args:
            logits: Draft model output [B, N*block_size, vocab].
            input_ids: Original input token IDs [B, seq_len].
            anchor_positions: Anchor positions per block [B, N].
            block_keep_mask: Valid block mask [B, N].
            loss_mask: Token-level loss mask [B, seq_len].
            base_logits: Base model logits for KD loss [B, seq_len, vocab], or None for CE.

        Returns:
            (loss, accuracy) tuple.
        """
        bsz, seq_len = input_ids.shape
        block_size = self.dflash_block_size
        n_blocks = anchor_positions.shape[1]
        device = input_ids.device

        label_offsets = torch.arange(0, block_size, device=device).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        valid_label = label_indices < seq_len
        safe_label_indices = label_indices.clamp(max=seq_len - 1)

        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, n_blocks, -1), 2, safe_label_indices
        )

        # Weight mask: valid block * in bounds * exclude anchor (pos 0) * loss_mask
        weight_mask = block_keep_mask.unsqueeze(-1).expand(-1, -1, block_size).float()
        weight_mask = weight_mask * valid_label.float()
        pos_in_block = torch.arange(block_size, device=device).view(1, 1, -1)
        weight_mask = weight_mask * (pos_in_block > 0).float()

        orig_loss_mask = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, n_blocks, -1), 2, safe_label_indices
        )
        weight_mask = weight_mask * orig_loss_mask

        binary_eval_mask = weight_mask.view(-1)

        # Optional loss decay
        if self.dflash_loss_decay_factor > 0:
            k = torch.arange(block_size, device=device).view(1, 1, -1)
            decay = torch.exp(-(k - 1).clamp(min=0).float() / self.dflash_loss_decay_factor)
            weight_mask = weight_mask * decay

        flat_logits = logits.view(-1, logits.size(-1))
        flat_targets = target_ids.view(-1)
        flat_weights = weight_mask.view(-1)
        valid_count = flat_weights.sum() + 1e-6

        if valid_count > 1.0:
            if base_logits is not None:
                # KD loss: teacher logits for token anchor+k are at position anchor+k-1
                teacher_indices = (safe_label_indices - 1).clamp(min=0)
                teacher_logits = torch.gather(
                    base_logits.unsqueeze(1).expand(-1, n_blocks, -1, -1),
                    2,
                    teacher_indices.unsqueeze(-1).expand(-1, -1, -1, base_logits.size(-1)),
                )
                flat_teacher = teacher_logits.reshape(-1, base_logits.size(-1)).detach()
                target_soft = torch.softmax(flat_teacher, dim=-1)
                draft_logsoft = torch.log_softmax(flat_logits, dim=-1)
                kd_loss = -(target_soft * draft_logsoft).sum(dim=-1)
                loss = (kd_loss * flat_weights).sum() / valid_count
            else:
                loss_per_token = F.cross_entropy(flat_logits, flat_targets, reduction="none")
                loss = (loss_per_token * flat_weights).sum() / valid_count

            with torch.no_grad():
                preds = flat_logits.argmax(dim=-1)
                correct = (preds == flat_targets) & (binary_eval_mask > 0.5)
                accuracy = correct.sum().float() / (binary_eval_mask.sum() + 1e-6)
                accuracy = accuracy.item()
        else:
            loss = flat_logits.sum() * 0.0
            accuracy = 0.0

        return loss, accuracy

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        cache_position=None,
        **kwargs,
    ):
        """Training forward with random anchor sampling.

        - Random anchor sampling instead of uniform block division
        - Bidirectional intra-block attention (no causal constraint)
        - Context sees strictly before anchor position
        - Label alignment: position k predicts token at anchor+k
        - Optional loss decay weighting
        """
        if not self.training:
            # Don't pass labels to base model — DFlash uses unshifted labels
            # which are incompatible with the base model's shifted loss.
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                cache_position=cache_position,
                **kwargs,
            )

        bsz, seq_len = input_ids.shape
        block_size = self.dflash_block_size
        device = input_ids.device

        if seq_len % block_size != 0:
            raise ValueError(
                f"seq_len ({seq_len}) must be divisible by block_size ({block_size}). "
                f"Adjust training_seq_len or use padding."
            )

        # 1. Run base model → hidden states
        # TODO: For co-training the base model, remove no_grad and eval() switch.
        with torch.no_grad():
            base_outputs = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        offset = 1
        selected = [base_outputs.hidden_states[lid + offset] for lid in self.target_layer_ids]
        target_hidden = torch.cat(selected, dim=-1)  # [B, seq, num_layers * H]

        # 2. Build loss mask.
        # When labels are provided (answer_only_loss), they already encode both
        # assistant masking and padding (-100 for both). When labels are not
        # provided, fall back to attention_mask for padding only.
        if labels is not None:
            loss_mask = (labels != LabelSmoother.ignore_index).float()
        elif attention_mask is not None:
            loss_mask = attention_mask.float()
        else:
            loss_mask = torch.ones(bsz, seq_len, device=device)

        # 3. Random anchor sampling
        anchor_positions, block_keep_mask = self._sample_anchor_positions(
            seq_len, loss_mask, device
        )
        n_blocks = anchor_positions.shape[1]

        if n_blocks == 0 or not block_keep_mask.any():
            # Zero loss that still flows through dflash_module for DDP gradient sync
            dummy = self.dflash_module.fc.weight.sum() * 0.0
            return ModelOutput(loss=dummy, logits=base_outputs.logits, train_acc=[[0.0]])

        # 4. Build draft inputs
        noise_embedding = self._build_noise_embedding(
            input_ids, anchor_positions, block_keep_mask, n_blocks
        )
        full_pos = self._build_position_ids(seq_len, anchor_positions, device)
        attn_mask = self._build_draft_attention_mask(
            seq_len, anchor_positions, block_keep_mask, n_blocks, target_hidden.dtype, device
        )

        # 5. Draft forward
        hidden = self.dflash_module(
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            position_ids=full_pos,
            attention_mask=attn_mask,
        )

        # 6. Compute loss and accuracy
        logits = self._base_model_lm_head(hidden)
        loss, accuracy = self._compute_loss(
            logits,
            input_ids,
            anchor_positions,
            block_keep_mask,
            loss_mask,
            base_outputs.logits if self.dflash_self_logit_distillation else None,
        )

        return ModelOutput(
            loss=loss,
            logits=base_outputs.logits,
            train_acc=[[accuracy]],
        )

    @torch.no_grad()
    def pseudo_speculative_generate(self, input_ids, steps=1):
        """Generate draft tokens using one DFlash block for AR validation.

        This method implements a single speculative decoding step:

        1. **Base model forward**: Run the full target model on ``input_ids`` to get:
           - ``base_token``: greedy next token (argmax of last position logits)
           - ``hidden_states``: intermediate hidden states from target layers

        2. **Extract target hidden states**: Concatenate hidden states from
           ``target_layer_ids`` (e.g., layers [1, 9, 17, 25, 33] for 5-layer draft).
           Shape: ``[B, seq_len, num_layers * hidden_size]``.

        3. **Build block input**: Create a block of ``block_size`` tokens where:
           - Position 0 = ``base_token`` (the anchor/known token)
           - Positions 1..block_size-1 = ``mask_token_id`` (unknown, to be predicted)
           Embed this block via the base model's embedding layer.

        4. **Position IDs**: Context positions ``[0..seq_len-1]`` followed by block
           positions ``[seq_len..seq_len+block_size-1]``. The draft model's attention
           uses RoPE on these positions so Q (block only) attends to K (context + block)
           with correct relative position encoding.

        5. **Draft forward**: Run ``DFlashModule`` with:
           - ``noise_embedding``: embedded block tokens
           - ``target_hidden``: extracted hidden states from step 2
           - ``position_ids``: context + block positions
           - ``attention_mask=None``: no mask at inference (all positions attend freely)
           The draft model's KV injection concatenates projected target_hidden as K/V
           with the block's own K/V, enabling the draft to "see" the target's context.

        6. **Decode**: Apply ``lm_head`` to draft hidden states at positions 1..block_size-1
           (skip position 0 which is the known anchor). Argmax gives draft tokens.

        7. **Return**: ``(base_token, draft_tokens[:steps])`` — base token is always
           returned; draft tokens are truncated to ``steps`` (default: block_size-1).

        Note:
            This method re-runs the full target model from scratch on each call
            (no KV cache). For AR validation, it is called repeatedly with growing
            ``input_ids`` by ``AcceptanceRateValidation.validate()``. The ``steps``
            parameter should be set to ``block_size - 1`` for full block evaluation.

        Args:
            input_ids: Input token IDs [B, seq_len].
            steps: Number of draft tokens to return (capped at block_size-1).

        Returns:
            base_token: Next token from base model [B, 1].
            draft_tokens: Draft tokens [B, min(steps, block_size-1)] or None if steps < 1.
        """
        # Call the base model's inner model directly (avoids DynamicModule dispatch)
        model_output = self._base_model(
            input_ids=input_ids,
            output_hidden_states=True,
        )
        # Compute logits via lm_head
        base_logits = self._base_model_lm_head(model_output.last_hidden_state)
        # Build output with hidden_states
        base_outputs = ModelOutput(
            logits=base_logits,
            hidden_states=model_output.hidden_states,
        )
        base_logits = base_outputs.logits
        base_token = base_logits[:, -1:, :].argmax(dim=-1).to(input_ids.device)

        if steps < 1:
            return base_token, None

        # Extract target hidden states (raw, before FC projection)
        hid_offset = 1
        selected = [base_outputs.hidden_states[lid + hid_offset] for lid in self.target_layer_ids]
        target_hidden = torch.cat(selected, dim=-1)

        block_size = self.dflash_block_size
        bsz = input_ids.shape[0]
        device = input_ids.device

        # Block: first token is base_token (anchor), rest are mask
        block_ids = torch.full(
            (bsz, block_size), self.mask_token_id, dtype=torch.long, device=device
        )
        block_ids[:, 0] = base_token.squeeze(-1)
        noise_embedding = self._base_model_embeddings(block_ids)

        # Position IDs: training uses [0..L-1, 0..L-1] where noise positions
        # mirror context positions. At inference, block predicts tokens at
        # seq_len..seq_len+B-1, so noise positions continue from ctx_len.
        ctx_len = target_hidden.shape[1]
        ctx_positions = torch.arange(ctx_len, device=device)
        block_positions = torch.arange(ctx_len, ctx_len + block_size, device=device)
        pos_ids = torch.cat([ctx_positions, block_positions]).unsqueeze(0).expand(bsz, -1)

        # No attention mask at inference
        # which uses KV cache with no mask. All positions attend freely to
        # context and each other within the block.

        # Draft forward
        draft_hidden = self.dflash_module(
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            position_ids=pos_ids,
            attention_mask=None,
        )

        # Logits on positions 1..block_size-1 (skip anchor at position 0)
        draft_logits = self._base_model_lm_head(draft_hidden[:, 1:, :])
        draft_tokens = draft_logits.argmax(dim=-1)  # [B, block_size-1]

        # Return up to `steps` tokens
        num_tokens = min(steps, block_size - 1)
        return base_token, draft_tokens[:, :num_tokens]
