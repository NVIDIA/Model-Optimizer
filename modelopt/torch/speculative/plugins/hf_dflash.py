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

DFlash (Block Diffusion for Flash Speculative Decoding) uses three key mechanisms:

1. Feature Fusion: Extract hidden states from uniformly sampled target model layers,
   concatenate and project via a lightweight FC layer.

2. KV Injection: The fused features are injected as Key/Value entries into EVERY
   draft model layer's attention. Unlike EAGLE-3 which only feeds features to the
   first layer, DFlash ensures every layer has full target model context.

3. Parallel Drafting: All tokens in a block are predicted in a single forward pass.
   The draft model uses mask tokens for unknown positions and predicts them all
   simultaneously via cross-entropy against target model logits.

Reference: "DFlash: Block Diffusion for Flash Speculative Decoding" (arXiv:2602.06036)
"""

import contextlib
import math
import random

import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.models.llama.modeling_llama import (
    LlamaMLP,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    rotate_half,
)
from transformers.utils import ModelOutput

from ..dflash.conversion import DFlashDMRegistry
from ..dflash.dflash_model import DFlashModel

__all__ = ["HFDFlashModel"]


def build_target_layer_ids(num_target_layers, num_sample_layers):
    """Select layers uniformly from the target model for feature extraction."""
    if num_sample_layers == 1:
        return [num_target_layers // 2]
    start = 1
    end = num_target_layers - 3
    span = end - start
    return [round(start + (i * span) / (num_sample_layers - 1)) for i in range(num_sample_layers)]


class DFlashAttention(nn.Module):
    """Attention with KV injection from target model features.

    Key difference from standard attention: K and V are computed from BOTH
    the target model's fused features (context) AND the draft tokens (noise).
    Q is computed only from draft tokens.

    Attention pattern: [k_ctx | k_noise] where draft queries attend to
    both context KV and draft KV with appropriate masking.
    """

    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states,
        target_hidden,
        position_embeddings,
        attention_mask=None,
    ):
        bsz, q_len, _ = hidden_states.shape
        ctx_len = target_hidden.shape[1]

        # Q from draft tokens only
        q = (
            self.q_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # K, V from both context (target features) and noise (draft tokens)
        k_ctx = (
            self.k_proj(target_hidden)
            .view(bsz, ctx_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v_ctx = (
            self.v_proj(target_hidden)
            .view(bsz, ctx_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        k_noise = (
            self.k_proj(hidden_states)
            .view(bsz, q_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v_noise = (
            self.v_proj(hidden_states)
            .view(bsz, q_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Apply rotary embeddings
        if position_embeddings is not None:
            cos, sin = position_embeddings
            # Rotary for noise Q and K
            q = self._apply_rotary(
                q, cos[:, ctx_len : ctx_len + q_len], sin[:, ctx_len : ctx_len + q_len]
            )
            k_noise = self._apply_rotary(
                k_noise, cos[:, ctx_len : ctx_len + q_len], sin[:, ctx_len : ctx_len + q_len]
            )
            k_ctx = self._apply_rotary(k_ctx, cos[:, :ctx_len], sin[:, :ctx_len])

        # Concatenate context and noise KV
        k = torch.cat([k_ctx, k_noise], dim=2)  # [B, num_kv_heads, ctx+q, head_dim]
        v = torch.cat([v_ctx, v_noise], dim=2)

        # GQA: expand KV heads to match Q heads
        if self.num_kv_heads != self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        # Scaled dot product attention
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, is_causal=False
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        return self.o_proj(attn_output)

    @staticmethod
    def _apply_rotary(x, cos, sin):
        """Apply rotary positional embeddings (HF Llama convention)."""
        cos = cos.unsqueeze(1)  # [B, 1, seq, dim]
        sin = sin.unsqueeze(1)
        return (x * cos) + (rotate_half(x) * sin)


class DFlashDecoderLayer(nn.Module):
    """Draft decoder layer with KV injection."""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.self_attn = DFlashAttention(config, layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, target_hidden, position_embeddings, attention_mask=None):
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
    """DFlash draft module with feature fusion + KV injection.

    Architecture:
    - FC layer fuses multi-layer target hidden states → hidden_size
    - N decoder layers, each with KV injection from fused target features
    - Shares embeddings and lm_head with the target model
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Feature fusion: project concatenated multi-layer hidden states
        num_fused_layers = len(config.target_layer_ids)
        self.fc = nn.Linear(num_fused_layers * config.hidden_size, config.hidden_size, bias=False)
        self.hidden_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Learnable mask embedding for unknown positions in blocks
        self.mask_embedding = nn.Parameter(torch.randn(config.hidden_size) * 0.02)

        # Draft decoder layers with KV injection
        self.layers = nn.ModuleList(
            [DFlashDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

    def fuse_target_features(self, target_hidden_states, target_layer_ids):
        """Extract and fuse hidden states from sampled target layers."""
        selected = [target_hidden_states[lid + 1] for lid in target_layer_ids]
        concatenated = torch.cat(selected, dim=-1)  # [B, seq, num_layers * H]
        fused = self.hidden_norm(self.fc(concatenated))  # [B, seq, H]
        return fused

    def forward(self, hidden_states, target_hidden, attention_mask=None):
        """Forward pass with KV injection.

        Args:
            hidden_states: Draft token embeddings [B, noise_len, H].
            target_hidden: Fused target features [B, ctx_len, H].
            attention_mask: Attention mask for [ctx + noise] positions.
        """
        total_len = target_hidden.shape[1] + hidden_states.shape[1]
        position_ids = torch.arange(total_len, device=hidden_states.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states, target_hidden, position_embeddings, attention_mask)

        return self.norm(hidden_states)


def build_dflash_attention_mask(ctx_len, block_anchors, block_size, seq_len, device, dtype):
    """Build DFlash attention mask for training.

    Each draft query can attend to:
    1. Context positions strictly before its block's anchor position
    2. All positions within the same block (bidirectional)
    3. Nothing from other blocks

    Args:
        ctx_len: Number of context tokens (= seq_len of the input).
        block_anchors: List of anchor positions (indices into the original sequence).
        block_size: Number of tokens per block.
        seq_len: Original sequence length.
        device: Target device.
        dtype: Target dtype.

    Returns:
        Attention mask [1, 1, noise_len, ctx_len + noise_len].
    """
    num_blocks = len(block_anchors)
    noise_len = num_blocks * block_size

    # Mask shape: [noise_len, ctx_len + noise_len]
    # Q dimension = noise tokens, KV dimension = context + noise tokens
    mask = torch.full((noise_len, ctx_len + noise_len), float("-inf"), device=device, dtype=dtype)

    for block_idx, anchor in enumerate(block_anchors):
        block_start = block_idx * block_size
        block_end = block_start + block_size

        # 1. Context: each block sees all context up to its anchor position
        mask[block_start:block_end, : min(anchor, ctx_len)] = 0.0

        # 2. Within-block: BIDIRECTIONAL attention (all positions see each other)
        # This is key to DFlash's parallel drafting — all positions in a block
        # have the same information and predict independently.
        mask[block_start:block_end, ctx_len + block_start : ctx_len + block_end] = 0.0

    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, noise, ctx+noise]


@DFlashDMRegistry.register({PreTrainedModel: "hf.PreTrainedModel"})
class HFDFlashModel(DFlashModel):
    """DFlash Model for HuggingFace models with KV injection + parallel drafting."""

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
        for name, paths in {
            "base_model_path": ["model.language_model", "model", "backbone"],
            "base_model_embeddings_path": [
                "model.embed_tokens",
                "backbone.embeddings",
                "model.language_model.embed_tokens",
            ],
            "base_model_lm_head_path": ["lm_head", "language_model.lm_head"],
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
        """Initialize DFlash with feature fusion + KV injection."""
        super().modify(config)

        base_config = self._base_llm_config
        self.dflash_config = PretrainedConfig.from_dict(config.dflash_architecture_config)
        self.dflash_config.hidden_size = base_config.hidden_size
        self.dflash_config.vocab_size = base_config.vocab_size
        self.dflash_config.max_position_embeddings = base_config.max_position_embeddings
        self.dflash_config.intermediate_size = getattr(
            self.dflash_config, "intermediate_size", base_config.intermediate_size
        )
        # head_dim for rotary embeddings: match base model
        actual_head_dim = base_config.hidden_size // base_config.num_attention_heads
        self.dflash_config.head_dim = actual_head_dim
        if self.dflash_config._attn_implementation is None:
            self.dflash_config._attn_implementation = "sdpa"

        # Determine target layer IDs for feature extraction
        num_target_layers = base_config.num_hidden_layers
        num_sample_layers = self.dflash_config.num_hidden_layers  # sample as many as draft layers
        self.target_layer_ids = build_target_layer_ids(num_target_layers, num_sample_layers)
        self.dflash_config.target_layer_ids = self.target_layer_ids

        # Freeze base model
        if self.dflash_freeze_base_model:
            for param in self.parameters():
                param.requires_grad = False

        self._find_base_model_parts()

        # Build DFlash module
        self.dflash_module = DFlashModule(self.dflash_config)
        self.dflash_module.to(self._base_model.dtype).to(
            next(self._base_model.layers[-1].parameters()).device
        )

        # Register hooks to collect hidden states from target layers
        self._target_hidden_states = []
        for layer_idx, layer in enumerate(self._base_model.layers):
            if layer_idx in self.target_layer_ids:
                layer.register_forward_hook(self._collect_hidden_hook)

        self._cached_masks = {}
        self.is_quantized = False

    def _collect_hidden_hook(self, module, input, output):
        hidden = (
            output.clone().detach()
            if isinstance(output, torch.Tensor)
            else output[0].clone().detach()
        )
        self._target_hidden_states.append(hidden)

    def _run_base_model(self, input_ids, attention_mask, labels=None, **kwargs):
        """Run base model, collect hidden states from target layers."""
        self._target_hidden_states = []
        with torch.no_grad() if self.dflash_freeze_base_model else contextlib.nullcontext():
            outputs = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                **kwargs,
            )
        logits = outputs.logits

        # Fuse collected hidden states
        target_hidden = self.dflash_module.fuse_target_features(
            outputs.hidden_states, self.target_layer_ids
        )
        self._target_hidden_states = []

        base_loss = None
        if labels is not None and not self.dflash_freeze_base_model:
            loss_fct = nn.CrossEntropyLoss()
            base_loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))

        return target_hidden, logits, base_loss

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
        """DFlash training forward pass.

        1. Run base model → get hidden states from sampled layers + logits
        2. Fuse multi-layer hidden states via FC projection
        3. Sample random anchors from the sequence → form blocks
        4. Create noise input (mask tokens + anchor tokens at block starts)
        5. Run draft model with KV injection (fused features as K/V in every layer)
        6. Compute CE loss with exponential position decay
        """
        if not self.training:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                cache_position=cache_position,
                **kwargs,
            )

        batch_size, seq_len = input_ids.shape
        block_size = self.dflash_block_size
        device = input_ids.device

        # 1. Run base model → fused target features + logits
        target_hidden, base_logits, base_loss = self._run_base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

        # 2. Sample anchor positions (start of each block)
        # Anchors are random positions in the valid (non-padding) region
        if attention_mask is not None:
            actual_len = attention_mask.sum(dim=1).min().int().item()
        else:
            actual_len = seq_len

        # Number of blocks we can fit: leave room for block_size predictions after each anchor
        max_anchor = actual_len - block_size
        max_anchor = max(max_anchor, 1)
        num_blocks = max(1, max_anchor // block_size)
        # Sample anchor positions uniformly
        anchors = sorted(
            random.sample(range(1, max(2, max_anchor)), min(num_blocks, max(1, max_anchor - 1)))
        )

        noise_len = len(anchors) * block_size

        # 3. Create noise embeddings: anchor token at position 0, MASK for positions 1..B-1
        # This matches inference where only the anchor token is known.
        # The draft model must predict all other positions from context KV alone.
        noise_embeds = (
            self.dflash_module.mask_embedding.unsqueeze(0)
            .unsqueeze(0)
            .expand(batch_size, noise_len, -1)
            .clone()
        )
        for b, anchor in enumerate(anchors):
            # Only position 0 of each block gets the real anchor token embedding
            anchor_embed = self._base_model_embeddings(input_ids[:, anchor : anchor + 1])
            noise_embeds[:, b * block_size : b * block_size + 1] = anchor_embed

        # 4. Build attention mask
        dtype = target_hidden.dtype
        attn_mask = build_dflash_attention_mask(
            seq_len, anchors, block_size, seq_len, device, dtype
        )

        # 5. Run DFlash draft model with KV injection
        draft_hidden = self.dflash_module(
            hidden_states=noise_embeds,
            target_hidden=target_hidden,
            attention_mask=attn_mask,
        )
        draft_logits = self._base_model_lm_head(draft_hidden)  # [B, noise_len, V]

        # 6. Compute loss with exponential position decay
        # For block b, position k: target is token at anchors[b] + k
        total_loss = torch.tensor(0.0, device=device, dtype=dtype)
        total_correct = 0
        total_valid = 0
        decay_gamma = block_size  # decay rate

        for b, anchor in enumerate(anchors):
            for k in range(1, block_size):  # skip position 0 (anchor itself)
                target_pos = anchor + k
                if target_pos >= seq_len:
                    break

                draft_idx = b * block_size + k
                logit = draft_logits[:, draft_idx, :]  # [B, V]

                # Target: base_logits[anchor + k - 1] predicts token at position anchor + k
                # This is the base model's autoregressive prediction for this position
                target = base_logits[:, target_pos - 1, :].detach()

                # Logit distillation loss
                target_soft = torch.softmax(target, dim=-1)
                draft_logsoft = torch.log_softmax(logit, dim=-1)
                kd_loss = -torch.sum(target_soft * draft_logsoft, dim=-1).mean()

                # Position decay weight
                weight = math.exp(-(k - 1) / decay_gamma)
                total_loss = total_loss + weight * kd_loss

                # Accuracy: does draft predict the same token as the base model?
                target_tok = input_ids[:, target_pos]
                draft_tok = logit.detach().argmax(dim=-1)
                total_correct += (target_tok == draft_tok).sum().item()
                total_valid += batch_size

        # Normalize by number of predictions
        num_predictions = sum(min(block_size - 1, seq_len - a - 1) for a in anchors)
        if num_predictions > 0:
            total_loss = total_loss / num_predictions

        accuracy = total_correct / max(total_valid, 1)
        final_loss = (base_loss or 0) + total_loss

        return ModelOutput(
            loss=final_loss,
            logits=base_logits,
            hidden_states=target_hidden,
            train_acc=[[accuracy]],
        )

    @torch.no_grad()
    def pseudo_speculative_generate(self, input_ids, steps=1):
        """Generate draft tokens using DFlash parallel block prediction.

        Args:
            input_ids: Prompt token IDs [B, seq_len].
            steps: Number of blocks to generate.

        Returns:
            base_token: Next token from base model [B, 1].
            draft_tokens: Draft tokens [B, steps * block_size] or None.
        """
        # Run base model
        self._target_hidden_states = []
        base_outputs = super().forward(input_ids=input_ids, output_hidden_states=True)
        base_logits = base_outputs.logits
        base_token = base_logits[:, -1:, :].argmax(dim=-1).to(input_ids.device)

        if steps < 1:
            return base_token, None

        # Fuse target features
        target_hidden = self.dflash_module.fuse_target_features(
            base_outputs.hidden_states, self.target_layer_ids
        )

        block_size = self.dflash_block_size
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        device = input_ids.device
        dtype = target_hidden.dtype

        all_draft_tokens = []
        current_token = base_token  # [B, 1]

        for step in range(steps):
            # Build noise: anchor token at position 0, mask embedding for rest
            noise_embeds = (
                self.dflash_module.mask_embedding.unsqueeze(0)
                .unsqueeze(0)
                .expand(batch_size, block_size, -1)
                .clone()
            )
            anchor_embed = self._base_model_embeddings(current_token)  # [B, 1, H]
            noise_embeds[:, :1] = anchor_embed

            # Attention mask: block sees all context, bidirectional within block
            anchor = seq_len + step * block_size
            attn_mask = build_dflash_attention_mask(
                seq_len, [anchor], block_size, seq_len + (step + 1) * block_size, device, dtype
            )

            # Run draft with KV injection
            draft_hidden = self.dflash_module(
                hidden_states=noise_embeds,
                target_hidden=target_hidden,
                attention_mask=attn_mask,
            )
            draft_logits = self._base_model_lm_head(draft_hidden)
            block_tokens = draft_logits.argmax(dim=-1)  # [B, block_size]
            all_draft_tokens.append(block_tokens[:, 1:])  # skip anchor position

            # Next block starts with last predicted token
            current_token = block_tokens[:, -1:]

        draft_tokens = torch.cat(all_draft_tokens, dim=-1)
        return base_token, draft_tokens
