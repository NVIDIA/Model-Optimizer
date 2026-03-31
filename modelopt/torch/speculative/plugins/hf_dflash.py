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

Matches the reference SpecForge implementation (github.com/sgl-project/SpecForge PR #415).

Architecture:
- Feature Fusion: multi-layer target hidden states → FC + RMSNorm
- KV Injection: fused features as K/V in every draft layer with QK-norm
- Parallel Drafting: mask_token_id for unknown positions, causal within blocks
- Loss: hard CE on input_ids[i] (position i predicts token i)

Reference: "DFlash: Block Diffusion for Flash Speculative Decoding" (arXiv:2602.06036)
"""

import torch
import torch.nn.functional as F
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


def build_target_layer_ids(num_target_layers, num_draft_layers):
    """Select layers uniformly from the target model for feature extraction."""
    if num_draft_layers == 1:
        return [num_target_layers // 2]
    start = 1
    end = num_target_layers - 3
    span = end - start
    return [
        int(round(start + (i * span) / (num_draft_layers - 1))) for i in range(num_draft_layers)
    ]


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply RoPE. Q uses last q_len positions, K uses all positions."""
    cos = cos.unsqueeze(1)  # [B, 1, seq, dim]
    sin = sin.unsqueeze(1)
    q_len = q.size(2)
    q_embed = (q * cos[:, :, -q_len:, :]) + (rotate_half(q) * sin[:, :, -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class DFlashAttention(nn.Module):
    """Attention with KV injection, matching SpecForge Qwen3DFlashAttention."""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.is_causal = False

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        # QK norm (matches Qwen3DFlashAttention)
        self.q_norm = LlamaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = LlamaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, hidden_states, target_hidden, position_embeddings, attention_mask=None):
        bsz, q_len, _ = hidden_states.shape
        ctx_len = target_hidden.shape[1]

        # Q from noise only, with QK-norm
        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)

        # K from context + noise, with QK-norm
        k_ctx = self.k_proj(target_hidden)
        k_noise = self.k_proj(hidden_states)
        k = torch.cat([k_ctx, k_noise], dim=1).view(
            bsz, ctx_len + q_len, self.num_kv_heads, self.head_dim
        )
        k = self.k_norm(k).transpose(1, 2)

        # V from context + noise (no norm)
        v_ctx = self.v_proj(target_hidden)
        v_noise = self.v_proj(hidden_states)
        v = (
            torch.cat([v_ctx, v_noise], dim=1)
            .view(bsz, ctx_len + q_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        # RoPE: applied to full 2L positions, Q gets last q_len, K gets all
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # GQA expand
        if self.num_kv_heads != self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, is_causal=False, scale=self.scaling
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        return self.o_proj(attn_output)


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
    """DFlash draft module matching SpecForge DFlashDraftModel."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.block_size = config.block_size

        # Feature fusion
        num_fused_layers = len(config.target_layer_ids)
        self.fc = nn.Linear(num_fused_layers * config.hidden_size, config.hidden_size, bias=False)
        self.hidden_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Decoder layers
        self.layers = nn.ModuleList(
            [DFlashDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

    def forward(self, noise_embedding, target_hidden, position_ids, attention_mask=None):
        """Forward matching SpecForge DFlashDraftModel.forward."""
        hidden_states = noise_embedding
        target_hidden = self.hidden_norm(self.fc(target_hidden))
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states, target_hidden, position_embeddings, attention_mask)

        return self.norm(hidden_states)


def create_dflash_attention_mask(seq_len, block_size, device, dtype):
    """Create [L, 2L] attention mask matching SpecForge.

    Context (cols 0..L-1): Block B sees blocks 0..B-1 (strictly previous).
    Noise (cols L..2L-1): causal within same block only.
    """
    indices = torch.arange(seq_len, device=device)
    block_ids = indices // block_size

    q_block_ids = block_ids.unsqueeze(1)  # [L, 1]
    k_block_ids = block_ids.unsqueeze(0)  # [1, L]

    ctx_mask = k_block_ids < q_block_ids
    same_block = q_block_ids == k_block_ids
    causal = indices.unsqueeze(0) >= indices.unsqueeze(1)
    noise_mask = same_block & causal

    full_mask_bool = torch.cat([ctx_mask, noise_mask], dim=1)

    full_mask = torch.zeros(seq_len, 2 * seq_len, device=device, dtype=dtype)
    full_mask.masked_fill_(~full_mask_bool, torch.finfo(dtype).min)

    return full_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, L, 2L]


def create_dflash_loss_mask(seq_len, block_size, device):
    """Create loss mask: exclude Block 0 and block starts."""
    positions = torch.arange(seq_len, device=device)
    block_ids = positions // block_size
    is_block_0 = block_ids == 0
    is_block_start = (positions % block_size) == 0
    return (~is_block_0 & ~is_block_start).float()


@DFlashDMRegistry.register({PreTrainedModel: "hf.PreTrainedModel"})
class HFDFlashModel(DFlashModel):
    """DFlash Model matching SpecForge OnlineDFlashModel."""

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

    @staticmethod
    def _auto_detect_mask_token_id(base_config):
        """Auto-detect an appropriate mask token ID for DFlash.

        Different model families use different strategies:
        - Qwen3/3.5: built-in [MASK] token in vocabulary
        - Llama3: reserved special tokens (128002 = reserved_special_token_0)
        - Others: try tokenizer.mask_token_id, then fall back to pad/eos
        """
        model_type = getattr(base_config, "model_type", "")
        vocab_size = getattr(base_config, "vocab_size", 0)

        # Qwen3/3.5: known mask token positions
        if "qwen3" in model_type.lower() or "qwen" in model_type.lower():
            # Qwen3 vocab has dedicated mask tokens
            # Qwen3.5-4B: 248070, Qwen3-8B: similar range
            # Heuristic: eos_token_id + some offset, or check known values
            eos = getattr(base_config, "eos_token_id", None)
            if isinstance(eos, list):
                eos = eos[0]
            if eos and vocab_size > 200000:
                # Large Qwen vocab — mask token is typically near end of special tokens
                # Known: Qwen3.5 eos=248044, mask=248070 (offset ~26)
                # Try common offsets
                for offset in [26, 25, 24]:
                    candidate = eos + offset
                    if candidate < vocab_size:
                        return candidate
            # Fallback for smaller Qwen models
            if vocab_size > 150000:
                return vocab_size - 250  # heuristic for Qwen special token region

        # Llama3: use reserved_special_token_0 (128002)
        if "llama" in model_type.lower():
            if vocab_size >= 128256:  # Llama3 vocab size
                return 128002  # <|reserved_special_token_0|>

        # Generic: try pad_token_id, then eos
        pad_id = getattr(base_config, "pad_token_id", None)
        eos_id = getattr(base_config, "eos_token_id", None)
        if isinstance(eos_id, list):
            eos_id = eos_id[0]

        # Prefer pad over eos (pad is less likely to interfere)
        if pad_id is not None and pad_id != eos_id:
            return pad_id

        # Last resort
        return eos_id or 0

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
        """Initialize DFlash draft module."""
        super().modify(config)

        base_config = self._base_llm_config
        self.dflash_config = PretrainedConfig.from_dict(config.dflash_architecture_config)

        # Inherit settings from base model, but only those NOT already in the user config.
        # hidden_size and vocab_size MUST match. Others (heads, intermediate_size) can differ.
        # This allows the draft model to have a different architecture than the base model.
        self.dflash_config.hidden_size = base_config.hidden_size
        self.dflash_config.vocab_size = base_config.vocab_size

        # These use base model defaults if not specified in dflash_architecture_config
        for attr, default_from_base in [
            ("max_position_embeddings", True),
            ("intermediate_size", True),
            ("num_attention_heads", True),
            ("num_key_value_heads", True),
            ("hidden_act", True),
            ("rope_theta", True),
            ("rope_scaling", True),
            ("rope_type", False),
            ("position_embedding_type", False),
            ("rope_interleaved", False),
            ("rms_norm_eps", True),
            ("attention_bias", False),
            ("tie_word_embeddings", False),
        ]:
            if not hasattr(self.dflash_config, attr) or getattr(self.dflash_config, attr) is None:
                if default_from_base and hasattr(base_config, attr):
                    setattr(self.dflash_config, attr, getattr(base_config, attr))

        # Ensure required attrs have defaults
        if not hasattr(self.dflash_config, "mlp_bias") or self.dflash_config.mlp_bias is None:
            self.dflash_config.mlp_bias = False

        self.dflash_config.head_dim = getattr(
            self.dflash_config, "head_dim",
            self.dflash_config.hidden_size // self.dflash_config.num_attention_heads,
        )
        self.dflash_config.block_size = self.dflash_block_size
        if self.dflash_config._attn_implementation is None:
            self.dflash_config._attn_implementation = "eager"

        # Target layer IDs
        num_target_layers = base_config.num_hidden_layers
        num_draft_layers = self.dflash_config.num_hidden_layers
        self.target_layer_ids = build_target_layer_ids(num_target_layers, num_draft_layers)
        self.dflash_config.target_layer_ids = self.target_layer_ids

        # mask_token_id resolution order:
        # 1. Explicit in dflash_architecture_config (user override)
        # 2. Auto-detect from model vocabulary:
        #    - Qwen3/3.5: built-in [MASK] token
        #    - Llama3: reserved_special_token_0 (128002)
        #    - Others: tokenizer.mask_token_id
        # 3. Fallback to pad_token_id or eos_token_id (suboptimal)
        mask_id = config.dflash_architecture_config.get("mask_token_id", None)
        if mask_id is None:
            mask_id = self._auto_detect_mask_token_id(base_config)
        self.mask_token_id = mask_id[0] if isinstance(mask_id, list) else mask_id
        print(f"DFlash mask_token_id: {self.mask_token_id}")

        # Freeze base model
        if self.dflash_freeze_base_model:
            for param in self.parameters():
                param.requires_grad = False

        self._find_base_model_parts()

        self.dflash_module = DFlashModule(self.dflash_config)
        self.dflash_module.to(self._base_model.dtype).to(
            next(self._base_model.layers[-1].parameters()).device
        )

        self.is_quantized = False

        # Store bound reference to the original model class's forward.
        # DynamicModule changes type(self) but the original class is in _original_cls.
        # Find the original HF model class (e.g., Qwen3_5ForConditionalGeneration)
        # by walking MRO and skipping DFlash/DynamicModule classes
        skip_names = {"HFDFlashModel", "DFlashModel", "DynamicModule", "DFlashPreTrainedModel", "DFlashDraftModel"}
        original_cls = None
        for cls in type(self).__mro__:
            if (
                hasattr(cls, "forward")
                and cls.__name__ not in skip_names
                and cls is not type(self)
                and issubclass(cls, PreTrainedModel)
                and cls is not PreTrainedModel
            ):
                original_cls = cls
                break
        if original_cls is None:
            # Last resort: use the class two levels up (skip DFlash wrapper + DynamicModule)
            original_cls = type(self).__mro__[2]
        self._original_forward_cls = original_cls
        print(f"DFlash: using {original_cls.__name__}.forward as base forward")

    def _base_forward(self, **kwargs):
        """Call the original model's forward, bypassing DFlash wrapper."""
        return self._original_forward_cls.forward(self, **kwargs)

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
        """Training forward matching SpecForge OnlineDFlashModel.forward."""
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

        bsz, seq_len = input_ids.shape
        block_size = self.dflash_block_size
        device = input_ids.device

        # 1. Run base model → raw multi-layer hidden states
        # Use super().forward() which goes through DynamicModule → original model
        # (same pattern as EAGLE's HFEagleModel)
        with torch.no_grad():
            base_outputs = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        # Extract and concatenate target layer hidden states
        offset = 1
        selected = [base_outputs.hidden_states[lid + offset] for lid in self.target_layer_ids]
        target_hidden = torch.cat(selected, dim=-1)  # [B, seq, num_layers * H]

        # 2. Truncate to multiple of block_size
        n_blocks = seq_len // block_size
        effective_len = n_blocks * block_size
        input_ids_trunc = input_ids[:, :effective_len]
        target_hidden = target_hidden[:, :effective_len, :]
        # Loss mask: use labels to identify response-only tokens
        # labels has -100 (IGNORE_TOKEN_ID) for prompt/padding, valid ids for response
        # This matches SpecForge's loss_mask behavior
        if labels is not None:
            labels_trunc = labels[:, :effective_len]
            loss_mask_input = (labels_trunc != -100).float()
        elif attention_mask is not None:
            loss_mask_input = attention_mask[:, :effective_len].float()
        else:
            loss_mask_input = torch.ones(bsz, effective_len, device=device)

        # 3. Prepare noise: mask_token_id everywhere, real token at block starts
        positions = torch.arange(effective_len, device=device)
        is_block_start = (positions % block_size) == 0
        noise_input_ids = torch.full_like(input_ids_trunc, self.mask_token_id)
        noise_input_ids[:, is_block_start] = input_ids_trunc[:, is_block_start]
        noise_embedding = self._base_model_embeddings(noise_input_ids)

        # 4. Position IDs: [0..L-1, 0..L-1]
        pos_seq = torch.arange(effective_len, device=device)
        position_ids_2l = torch.cat([pos_seq, pos_seq]).unsqueeze(0).expand(bsz, -1)

        # 5. Attention mask: [1, 1, L, 2L]
        dtype = target_hidden.dtype
        dflash_attn_mask = create_dflash_attention_mask(effective_len, block_size, device, dtype)

        # 6. Draft forward
        hidden = self.dflash_module(
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            position_ids=position_ids_2l,
            attention_mask=dflash_attn_mask,
        )

        # 7. Loss computation
        logits = self._base_model_lm_head(hidden)
        dflash_loss_mask = create_dflash_loss_mask(effective_len, block_size, device)
        combined_mask = loss_mask_input * dflash_loss_mask.unsqueeze(0)

        logits_flat = logits.reshape(-1, logits.size(-1))
        labels_flat = input_ids_trunc.reshape(-1)
        mask_flat = combined_mask.reshape(-1)

        active_indices = mask_flat > 0.5
        active_logits = logits_flat[active_indices]
        active_labels = labels_flat[active_indices]

        if active_logits.numel() > 0:
            if self.dflash_self_logit_distillation:
                # Logit distillation: learn from target model's output distribution
                # This works regardless of whether training data matches the target model
                base_logits_trunc = base_outputs.logits[:, :effective_len, :]
                base_logits_flat = base_logits_trunc.reshape(-1, base_logits_trunc.size(-1))
                active_base_logits = base_logits_flat[active_indices].detach()
                target_soft = torch.softmax(active_base_logits, dim=-1)
                draft_logsoft = torch.log_softmax(active_logits, dim=-1)
                loss = -(target_soft * draft_logsoft).sum(dim=-1).mean()
            else:
                # Hard CE: predict ground truth tokens directly
                # Only works well when training data is synthesized by the target model
                loss = F.cross_entropy(active_logits, active_labels)

            with torch.no_grad():
                preds = active_logits.argmax(dim=-1)
                accuracy = (preds == active_labels).float().mean().item()
        else:
            # No valid positions — compute a zero loss that still flows through
            # dflash_module parameters to keep DDP gradient sync happy
            loss = logits.sum() * 0.0
            accuracy = 0.0

        return ModelOutput(
            loss=loss,
            logits=base_outputs.logits,
            train_acc=[[accuracy]],
        )

    @torch.no_grad()
    def pseudo_speculative_generate(self, input_ids, steps=1):
        """Generate draft tokens using one DFlash block.

        DFlash generates block_size-1 draft tokens in a single forward pass.
        The `steps` parameter is used as the number of tokens to return
        (capped at block_size-1).

        Returns:
            base_token: Next token from base model [B, 1].
            draft_tokens: Draft tokens [B, min(steps, block_size-1)] or None.
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
        if not hasattr(self, '_psg_debug'):
            self._psg_debug = True
            sel = [base_outputs.hidden_states[lid + hid_offset] for lid in self.target_layer_ids]
            th = torch.cat(sel, dim=-1)
            print(f"[psg] hidden_states layers: {len(base_outputs.hidden_states)}, target_hidden norm: {th.norm().item():.2f}, shape: {th.shape}")
            print(f"[psg] base_token: {base_token.item()}, mask_token_id: {self.mask_token_id}")
            print(f"[psg] block_ids: {[self.mask_token_id]*self.dflash_block_size}")
            bi = torch.full((1, self.dflash_block_size), self.mask_token_id, dtype=torch.long, device=input_ids.device)
            bi[0, 0] = base_token[0, 0]
            ne = self._base_model_embeddings(bi)
            print(f"[psg] noise_emb norm: {ne.norm().item():.2f}, shape: {ne.shape}")
            print(f"[psg] pos_ids will be: ctx=[0..{input_ids.shape[1]-1}], blk=[{input_ids.shape[1]}..{input_ids.shape[1]+self.dflash_block_size-1}]")
        selected = [base_outputs.hidden_states[lid + hid_offset] for lid in self.target_layer_ids]
        target_hidden = torch.cat(selected, dim=-1)

        block_size = self.dflash_block_size
        bsz = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        device = input_ids.device
        dtype = target_hidden.dtype

        # Block: first token is base_token (anchor), rest are mask
        block_ids = torch.full(
            (bsz, block_size), self.mask_token_id, dtype=torch.long, device=device
        )
        block_ids[:, 0] = base_token.squeeze(-1)
        noise_embedding = self._base_model_embeddings(block_ids)

        # Position IDs: context 0..ctx_len-1, block seq_len..seq_len+block_size-1
        ctx_len = target_hidden.shape[1]
        ctx_positions = torch.arange(ctx_len, device=device)
        block_positions = torch.arange(seq_len, seq_len + block_size, device=device)
        pos_ids = torch.cat([ctx_positions, block_positions]).unsqueeze(0).expand(bsz, -1)

        # Attention mask: block sees ALL context + causal within block
        attn_mask = torch.zeros(
            1, 1, block_size, ctx_len + block_size, device=device, dtype=dtype
        )
        block_causal = torch.triu(
            torch.full(
                (block_size, block_size), torch.finfo(dtype).min, device=device, dtype=dtype
            ),
            diagonal=1,
        )
        attn_mask[:, :, :, ctx_len:] = block_causal

        # Draft forward
        draft_hidden = self.dflash_module(
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            position_ids=pos_ids,
            attention_mask=attn_mask,
        )

        # Logits on positions 1..block_size-1 (skip anchor at position 0)
        draft_logits = self._base_model_lm_head(draft_hidden[:, 1:, :])
        draft_tokens = draft_logits.argmax(dim=-1)  # [B, block_size-1]

        # Return up to `steps` tokens
        num_tokens = min(steps, block_size - 1)
        return base_token, draft_tokens[:, :num_tokens]
