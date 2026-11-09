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

# Architecture reference: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/inference/model.py
# Copyright (c) 2023 DeepSeek. MIT License.
# SPDX-License-Identifier: Apache-2.0 AND MIT

"""Native, differentiable HF text port of the published DeepSeek V4.1 backbone.

The eager Torch path preserves compressed KV/index ownership and mHC mixing.
Vision input and DSpark speculative draft heads are explicitly outside this text
interface. Generation uses an exact prefix-recomputation cache; it is intended
for correctness and repair workloads, not production serving.
"""

import re
from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import nn
from transformers import GenerationMixin, PreTrainedModel
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from ._hf_attention import V41Attention
from ._hf_engram import V41Engram, V41NgramHash
from ._hf_math import (
    V41GroupedProjection,
    V41Linear,
    V41LMHead,
    V41RMSNorm,
    hyper_mixes,
    hyper_post,
    hyper_pre,
)
from ._hf_moe import V41FusedExperts, V41MoE
from .configuration_hf import DeepseekV41Config

__all__ = [
    "DeepseekV41DecoderLayer",
    "DeepseekV41ForCausalLM",
    "DeepseekV41Model",
    "DeepseekV41PreTrainedModel",
    "DeepseekV41RecomputeCache",
]


class DeepseekV41RecomputeCache(Cache):
    """Store prefix inputs for exact replay of shared CSA2 and Engram dependencies."""

    def __init__(self):
        """Initialize an empty prefix cache."""
        super().__init__(layers=[])
        self.input_ids = self.inputs_embeds = self.position_ids = self.attention_mask = None

    def get_seq_length(self, layer_idx=0):
        """Return the number of retained prefix positions."""
        return 0 if self.position_ids is None else self.position_ids.shape[1]

    def get_max_cache_shape(self, layer_idx=0):
        """Report an unbounded logical prefix length."""
        return -1

    @property
    def batch_size(self):
        """Return the current prefix batch size."""
        return -1 if self.position_ids is None else self.position_ids.shape[0]

    @property
    def is_initialized(self):
        """Report whether prefix inputs have been cached."""
        return self.position_ids is not None

    def reorder_cache(self, beam_idx):
        """Select prefix rows for beam search."""
        for name in ("input_ids", "inputs_embeds", "position_ids", "attention_mask"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, value.index_select(0, beam_idx.to(value.device)))

    def batch_select_indices(self, indices):
        """Select the requested prefix batch rows."""
        self.reorder_cache(indices)

    def batch_repeat_interleave(self, repeats):
        """Expand prefix rows for beam or candidate generation."""
        for name in ("input_ids", "inputs_embeds", "position_ids", "attention_mask"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, value.repeat_interleave(repeats, dim=0))

    def crop(self, tokens_to_remove):
        """Remove the requested negative count of trailing prefix positions."""
        if tokens_to_remove > 0:
            raise ValueError("Use a negative integer to remove tokens from the prefix cache.")
        if tokens_to_remove == 0:
            return
        length = max(0, self.get_seq_length() + tokens_to_remove)
        for name in ("input_ids", "inputs_embeds", "position_ids", "attention_mask"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, value[:, :length])

    def reset(self):
        """Discard all retained prefix inputs."""
        self.input_ids = self.inputs_embeds = self.position_ids = self.attention_mask = None


class DeepseekV41DecoderLayer(nn.Module):
    """mHC attention and MoE sublayers, with delayed pre-mix propagation."""

    def __init__(self, config, layer_id):
        """Create one published mHC attention and MoE block."""
        super().__init__()
        self.config, self.layer_id = config, layer_id
        self.attn = V41Attention(config, layer_id)
        self.ffn = V41MoE(config)
        self.engram = V41Engram(config, layer_id) if layer_id in config.engram_layer_ids else None
        self.attn_norm = V41RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn_norm = V41RMSNorm(config.hidden_size, config.rms_norm_eps)
        mix_width, stream_width = (
            (config.hc_mult + 2) * config.hc_mult,
            config.hc_mult * config.hidden_size,
        )
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_width, stream_width, dtype=torch.float32))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_width, stream_width, dtype=torch.float32))
        self.hc_attn_base = nn.Parameter(torch.zeros(mix_width, dtype=torch.float32))
        self.hc_ffn_base = nn.Parameter(torch.zeros(mix_width, dtype=torch.float32))
        self.hc_attn_scale = nn.Parameter(torch.ones(3, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.ones(3, dtype=torch.float32))

    def forward(
        self,
        hidden_states,
        pre_mix,
        position_ids,
        shared_state,
        attention_mask=None,
        engram_hashes=None,
    ):
        """Return the residual stream and the next attention pre-mix."""
        if self.engram is not None:
            if engram_hashes is None:
                raise ValueError("An Engram decoder requires its token hashes.")
            hidden_states = self.engram(
                hidden_states, engram_hashes[:, :, self.engram.layer_hash_index], attention_mask
            )
        residual = hidden_states
        attn_pre, attn_post, attn_comb = hyper_mixes(
            hidden_states, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base, self.config
        )
        value = self.attn_norm(hyper_pre(hidden_states, pre_mix))
        value = self.attn(value, position_ids, shared_state, attention_mask)
        hidden_states = hyper_post(value, residual, attn_post, attn_comb)
        residual = hidden_states
        ffn_pre, ffn_post, ffn_comb = hyper_mixes(
            hidden_states, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base, self.config
        )
        value = self.ffn_norm(hyper_pre(hidden_states, attn_pre))
        return hyper_post(self.ffn(value), residual, ffn_post, ffn_comb), ffn_pre


class DeepseekV41PreTrainedModel(PreTrainedModel):
    """HF base class with ordinary autograd parameters and public EP/TP plans."""

    config_class = DeepseekV41Config
    base_model_prefix = "model"
    _no_split_modules = ["DeepseekV41DecoderLayer"]
    supports_gradient_checkpointing = False
    _keep_in_fp32_modules_strict = [
        "hc_attn_fn",
        "hc_ffn_fn",
        "hc_attn_base",
        "hc_ffn_base",
        "hc_attn_scale",
        "hc_ffn_scale",
        "attn_sink",
        r"ffn\.gate\.",
        r"attn\.compressor\.wgate\.",
    ]
    _supports_sdpa = False
    _can_compile_fullgraph = False
    _can_set_experts_implementation_cached_value = True
    supports_context_parallel = False
    context_parallel_fallback = "full_context_teacher_local_unit_replay"

    def _get_dtype_plan(self, dtype):
        plan = super()._get_dtype_plan(dtype)
        if dtype in (torch.float16, torch.bfloat16):
            for layer in self.config.kv_source_layer_ids:
                if (
                    0 <= layer < self.config.num_hidden_layers
                    and self.config.compress_ratios[layer] > 1
                ):
                    plan[rf"layers\.{layer}\.attn\.compressor\.wkv\."] = torch.float32
        return plan

    def get_correct_experts_implementation(self, requested_experts):
        """Select an expert backend that preserves V4.1 activation ordering."""
        implementation = requested_experts or "eager"
        if implementation not in ("eager", "grouped_mm"):
            raise ValueError("V4.1 supports eager and its exact grouped_mm experts implementation.")
        return implementation

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Embedding, V41GroupedProjection)):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, V41RMSNorm):
            nn.init.ones_(module.weight)
        elif isinstance(module, V41FusedExperts):
            nn.init.normal_(module.gate_up_proj, std=std)
            nn.init.normal_(module.down_proj, std=std)
        elif isinstance(module, V41NgramHash):
            module.reset_hash_buffers()
        elif isinstance(module, DeepseekV41DecoderLayer):
            nn.init.normal_(module.hc_attn_fn, std=std)
            nn.init.normal_(module.hc_ffn_fn, std=std)
            nn.init.zeros_(module.hc_attn_base)
            nn.init.zeros_(module.hc_ffn_base)
            nn.init.ones_(module.hc_attn_scale)
            nn.init.ones_(module.hc_ffn_scale)


class DeepseekV41Model(DeepseekV41PreTrainedModel):
    """Published text backbone with forward-local shared compressed attention state."""

    def __init__(self, config):
        """Build the backbone and its exact Engram hash layout."""
        super().__init__(config)
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            DeepseekV41DecoderLayer(config, layer) for layer in range(config.num_hidden_layers)
        )
        self.norm = V41RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.engram_hash = V41NgramHash(config) if config.engram_layer_ids else None
        self.post_init()

    def get_input_embeddings(self):
        """Return the token embedding table."""
        return self.embed

    def set_input_embeddings(self, value):
        """Replace the token embedding table."""
        self.embed = value

    def set_engram_tokenizer(self, tokenizer):
        """Build the checkpoint's exact normalized token map, verifying its vocabulary size."""
        if self.engram_hash is not None:
            self.engram_hash.set_tokenizer(tokenizer)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_hidden_states=None,
        output_attentions=None,
        return_dict=None,
        **kwargs,
    ):
        """Run the complete causal text backbone, optionally replaying a cached prefix."""
        if kwargs:
            unsupported = {
                key
                for key, value in kwargs.items()
                if value is not None and key not in ("cache_position", "num_items_in_batch")
            }
            if unsupported:
                raise ValueError(f"Unsupported V4.1 text inputs: {sorted(unsupported)}")
        if output_attentions:
            raise NotImplementedError(
                "V4.1 sparse attention probabilities are not returned by this text interface."
            )
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Supply exactly one of input_ids and inputs_embeds.")
        if self.engram_hash is not None and input_ids is None:
            raise ValueError("Engram hashing requires input_ids.")
        if past_key_values is not None and not isinstance(
            past_key_values, DeepseekV41RecomputeCache
        ):
            raise ValueError("V4.1 generation requires DeepseekV41RecomputeCache.")
        use_cache = self.config.use_cache if use_cache is None else use_cache
        return_dict = self.config.return_dict if return_dict is None else return_dict
        output_hidden_states = (
            self.config.output_hidden_states
            if output_hidden_states is None
            else output_hidden_states
        )
        current = input_ids if input_ids is not None else inputs_embeds
        batch, new_length = current.shape[:2]
        cache = past_key_values
        prefix_length = cache.get_seq_length() if cache is not None else 0
        if position_ids is None:
            position_ids = torch.arange(
                prefix_length, prefix_length + new_length, device=current.device
            ).expand(batch, -1)
        elif position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0).expand(batch, -1)
        if prefix_length:
            assert cache is not None
            if (input_ids is None) != (cache.input_ids is None):
                raise ValueError("Cached input_ids and inputs_embeds modes cannot be mixed.")
            if input_ids is not None:
                input_ids = torch.cat((cache.input_ids, input_ids), 1)
            else:
                inputs_embeds = torch.cat((cache.inputs_embeds, inputs_embeds), 1)
            position_ids = torch.cat((cache.position_ids, position_ids), 1)
        if attention_mask is None:
            attention_mask = torch.ones(
                batch, prefix_length + new_length, dtype=torch.bool, device=current.device
            )
        elif attention_mask.ndim != 2:
            raise ValueError("V4.1 text attention_mask must be a two-dimensional token mask.")
        elif prefix_length and attention_mask.shape[1] == new_length:
            assert cache is not None
            attention_mask = torch.cat((cache.attention_mask, attention_mask), 1)
        if attention_mask.shape != (batch, prefix_length + new_length):
            raise ValueError("V4.1 attention mask must cover the complete input prefix.")
        attention_mask = attention_mask.bool()
        hidden_states = self.embed(input_ids) if inputs_embeds is None else inputs_embeds
        hashes = (
            self.engram_hash(input_ids, attention_mask) if self.engram_hash is not None else None
        )
        hidden_states = hidden_states.unsqueeze(-2).expand(-1, -1, self.config.hc_mult, -1)
        pre_mix = F.one_hot(
            torch.zeros(hidden_states.shape[:2], dtype=torch.long, device=hidden_states.device),
            self.config.hc_mult,
        ).float()
        collected = () if output_hidden_states else None
        shared = {}
        for layer in self.layers:
            if output_hidden_states:
                collected += (hyper_pre(hidden_states, pre_mix)[:, prefix_length:],)
            hidden_states, pre_mix = layer(
                hidden_states,
                pre_mix,
                position_ids,
                shared,
                attention_mask,
                engram_hashes=hashes,
            )
        hidden_states = self.norm(hyper_pre(hidden_states, pre_mix))[:, prefix_length:]
        if output_hidden_states:
            collected += (hidden_states,)
        if use_cache:
            cache = cache if cache is not None else DeepseekV41RecomputeCache()
            cache.input_ids = None if input_ids is None else input_ids.detach()
            cache.inputs_embeds = None if inputs_embeds is None else inputs_embeds.detach()
            cache.position_ids = position_ids.detach()
            cache.attention_mask = attention_mask.detach()
        else:
            cache = None
        result = BaseModelOutputWithPast(
            last_hidden_state=hidden_states, past_key_values=cache, hidden_states=collected
        )
        return result if return_dict else result.to_tuple()


class DeepseekV41ForCausalLM(DeepseekV41PreTrainedModel, GenerationMixin):
    """Differentiable text causal LM; optional source vision/draft branches are unsupported."""

    _tied_weights_keys = {"lm_head.weight": "model.embed.weight"}
    _fsdp_plan = {"lm_head": "keep_full_weight"}

    def __init__(self, config):
        """Attach the vocabulary head to the text backbone."""
        super().__init__(config)
        self.model = DeepseekV41Model(config)
        self.lm_head = V41LMHead(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        self._apply_checkpoint_precision()

    @classmethod
    def _supports_default_dynamic_cache(cls):
        return False

    def get_input_embeddings(self):
        """Return the backbone token embedding table."""
        return self.model.embed

    def set_input_embeddings(self, value):
        """Replace the backbone token embedding table."""
        self.model.embed = value

    def get_output_embeddings(self):
        """Return the vocabulary projection."""
        return self.lm_head

    def set_output_embeddings(self, value):
        """Replace the vocabulary projection."""
        self.lm_head = value

    def set_engram_tokenizer(self, tokenizer):
        """Install the exact normalized tokenizer mapping used by Engram."""
        self.model.set_engram_tokenizer(tokenizer)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        logits_to_keep=0,
        output_hidden_states=None,
        output_attentions=None,
        return_dict=None,
        **kwargs,
    ):
        """Return causal text logits and optional shifted cross-entropy loss."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
            **kwargs,
        )
        selection = (
            slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        )
        if isinstance(logits_to_keep, int) and logits_to_keep < 0:
            raise ValueError("logits_to_keep must be nonnegative.")
        hidden = outputs.last_hidden_state
        loss = None
        if labels is not None:
            full_logits = self.lm_head(hidden).float()
            loss = F.cross_entropy(
                full_logits[:, :-1].reshape(-1, self.config.vocab_size),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
            logits = full_logits[:, selection]
        else:
            logits = self.lm_head(hidden[:, selection]).float()
        result = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
        )
        return (
            result
            if (self.config.return_dict if return_dict is None else return_dict)
            else result.to_tuple()
        )

    def _apply_checkpoint_precision(self):
        for path in self.config.checkpoint_quantized_projections:
            match = re.fullmatch(r"(.+\.experts)\.(\d+)\.(w1|w2|w3)", path)
            if match:
                owner, expert, projection = match.groups()
                module = self.get_submodule(owner)
                target = "quantized_down" if projection == "w2" else "quantized_gate_up"
                setattr(module, target, getattr(module, target) | {int(expert)})
            else:
                owner, _, projection = path.rpartition(".")
                module = self.get_submodule(owner)
                if isinstance(module, V41Linear):
                    raise ValueError(f"Checkpoint projection must name the linear itself: {path}")
                if hasattr(module, "checkpoint_quantized_projections"):
                    module.checkpoint_quantized_projections |= {projection}
                else:
                    self.get_submodule(path).checkpoint_quantized = True

    @classmethod
    def from_checkpoint(
        cls,
        config,
        state_dict: Mapping[str, torch.Tensor],
        *,
        tokenizer=None,
        dtype=torch.bfloat16,
        device=None,
        allow_auxiliary=False,
    ):
        """Convert a supplied published text state dict, decoding FP8/FP4 and fusing experts.

        No files or model weights are downloaded. ``allow_auxiliary=True`` explicitly
        discards vision and DSpark tensors, which have no text-forward role. Source
        quantized projections keep their published MXFP8 activation rounding via STE.
        """
        from ._hf_checkpoint import (
            load_published_text_checkpoint,  # Avoid a model/converter import cycle.
        )

        return load_published_text_checkpoint(
            cls,
            config,
            state_dict,
            tokenizer=tokenizer,
            dtype=dtype,
            device=device,
            allow_auxiliary=allow_auxiliary,
        )
