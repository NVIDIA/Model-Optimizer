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

# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Qwen3 model with DMS (Dynamic Memory Sparsification)."""

import torch
from dms.attention import dms_attention
from dms.cache import DMSCache, Mode
from dms.core import (
    DMSBaseModelOutputWithPastAndCR,
    DMSCausalLMOutputWithPastAndCR,
    DMSTrainingStateAux,
    dms_chunked_prefill,
    post_process_attention_output,
    prepare_attention_input,
)
from dms.logging import get_logger
from torch import nn
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import (
    GenericForQuestionAnswering,
    GenericForSequenceClassification,
    GenericForTokenClassification,
    GradientCheckpointingLayer,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3MLP,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, can_return_tuple
from transformers.utils.deprecation import deprecate_kwarg

from .configuration_qwen3_dms import Qwen3ConfigDMS

logger = get_logger("Qwen3ForCausalLMDMS")


class Qwen3AttentionDMS(Qwen3Attention):
    """Multi-headed attention from 'Attention Is All You Need' paper."""

    def __init__(self, config: Qwen3ConfigDMS, layer_idx: int):
        """Initialize the Qwen3AttentionDMS model."""
        super().__init__(config=config, layer_idx=layer_idx)
        self.dms_alpha_scale = config.dms_alpha_scale
        self.dms_initial_alpha_offset = config.dms_initial_alpha_offset
        self.dms_window_size = config.dms_window_size
        self.dms_disable_eviction = config.dms_disable_eviction

        self.num_key_value_heads = config.num_key_value_heads

        self.dms_tau = config.dms_tau

        if self.config.dms_separate_alpha:
            self.dms_proj_alpha_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.dms_proj_alpha = nn.Linear(
                config.hidden_size, self.num_key_value_heads, bias=config.attention_bias
            )
        else:
            self.dms_proj_alpha_norm = None
            self.dms_proj_alpha = None

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        pre_attn_norm_hidden_states: torch.Tensor,
        post_attn_norm_hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: DMSCache | None = None,
        cache_position: torch.LongTensor | None = None,
        dms_state: DMSTrainingStateAux = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """A modified version of the forward pass from the transformers Qwen3Attention model."""
        if self.training:
            assert dms_state is not None, "dms_state is None in training mode"
        flash_attn_query_states, key_states, value_states, decisions, decision_logits = (
            prepare_attention_input(
                pre_attn_norm_hidden_states=pre_attn_norm_hidden_states,
                post_attn_norm_hidden_states=post_attn_norm_hidden_states,
                q_proj_fn=self.q_proj,
                k_proj_fn=self.k_proj,
                v_proj_fn=self.v_proj,
                q_norm_fn=self.q_norm,
                k_norm_fn=self.k_norm,
                head_dim=self.head_dim,
                cos=position_embeddings[0],
                sin=position_embeddings[1],
                dms_proj_alpha_norm_fn=self.dms_proj_alpha_norm,
                dms_proj_alpha_fn=self.dms_proj_alpha,
                dms_alpha_per=self.config.dms_alpha_per,
                dms_decision_scale=self.dms_alpha_scale,
                dms_initial_decision_offset=self.dms_initial_alpha_offset,
                dms_training=self.training,
                dms_disable_eviction=self.dms_disable_eviction,
                dms_tau=self.dms_tau,
                apply_rotary_pos_emb_fn=apply_rotary_pos_emb,
                dms_teacher_mode=(dms_state is not None and dms_state.dms_teacher_mode),
                dms_noise=dms_state.noise[self.layer_idx]
                if (dms_state is not None and dms_state.noise is not None)
                else None,
            )
        )

        attn_output = dms_attention(
            new_q_flash=flash_attn_query_states,
            new_k=key_states,
            new_v=value_states,
            decisions=decisions,
            decision_logits=decision_logits,
            attention_mask=attention_mask,
            layer_idx=self.layer_idx,
            dms_cache=past_key_values,
            attn_scaling=self.scaling,
            window_size=self.dms_window_size,
            train_attn_kwargs=kwargs.get("train_attn_kwargs", {}),
        )

        attn_output = post_process_attention_output(
            attn_output=attn_output,
            o_proj=self.o_proj,
        )

        return attn_output, decisions


class Qwen3DecoderLayerDMS(GradientCheckpointingLayer):
    """A modified version of the transformers Qwen3DecoderLayer model."""

    def __init__(self, config: Qwen3ConfigDMS, layer_idx: int):
        """Initialize the Qwen3DecoderLayerDMS model."""
        super().__init__()
        self.dms_window_size = config.dms_window_size
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen3AttentionDMS(config=config, layer_idx=layer_idx)

        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: DMSCache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor]
        | None = None,  # necessary, but kept here for BC
        dms_state: DMSTrainingStateAux | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        """A modified version of the forward pass from the transformers Qwen3DecoderLayer model."""
        residual = hidden_states
        # Self Attention
        pre_attn_norm_hidden_states = hidden_states
        post_attn_norm_hidden_states = self.input_layernorm(hidden_states)
        hidden_states, decisions = self.self_attn(
            pre_attn_norm_hidden_states=pre_attn_norm_hidden_states,
            post_attn_norm_hidden_states=post_attn_norm_hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            dms_state=dms_state,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        if self.training and dms_state is not None and not dms_state.dms_teacher_mode:
            assert dms_state.kv_cache_shape[1:] == decisions.shape, (
                f"dms_state.kv_cache_shape[1:]: {dms_state.kv_cache_shape[1:]} != decisions.shape: {decisions.shape}"
            )
            if dms_state.right_padding_size > 0:
                decisions = decisions[:, :, : -dms_state.right_padding_size]
            dms_frac_closed = decisions.float().mean(dim=(1, 2))
        else:
            dms_frac_closed = None

        return hidden_states, dms_frac_closed


class Qwen3PreTrainedModelDMS(PreTrainedModel):
    """A modified version of the transformers Qwen3PreTrainedModel model."""

    config: Qwen3ConfigDMS
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3DecoderLayerDMS"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Qwen3DecoderLayerDMS,
        "attentions": Qwen3AttentionDMS,
    }


class Qwen3ModelDMS(Qwen3PreTrainedModelDMS):
    """A modified version of the transformers Qwen3Model model."""

    def __init__(self, config: Qwen3ConfigDMS):
        """Initialize the Qwen3ModelDMS model."""
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayerDMS(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: DMSCache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        dms_state: DMSTrainingStateAux = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> DMSBaseModelOutputWithPastAndCR:
        """A modified version of the forward pass from the transformers Qwen3Model model."""
        if self.training:
            assert dms_state is not None
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        assert past_key_values is None or isinstance(past_key_values, DMSCache), (
            f"past_key_values is not a DMSCache: {type(past_key_values)}"
        )

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(
                    **mask_kwargs
                )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        hidden_states, dms_frac_closed = dms_chunked_prefill(
            decoder_layers=self.layers[: self.config.num_hidden_layers],
            hidden_states=hidden_states,
            attention_mask=causal_mask_mapping["full_attention"],
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            dms_state=dms_state,
            dms_manual_inference_mode=self.config.dms_manual_inference_mode,
            dms_chunked_prefill=self.config.dms_chunked_prefill,
            **kwargs,
        )

        hidden_states = self.norm(hidden_states)
        return DMSBaseModelOutputWithPastAndCR(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            cr=past_key_values.get_cr() if past_key_values is not None else None,
            dms_frac_closed=dms_frac_closed,
        )


class Qwen3ForCausalLMDMS(Qwen3PreTrainedModelDMS, GenerationMixin):
    """A modified version of the transformers Qwen3ForCausalLM model."""

    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config: Qwen3ConfigDMS):
        """Initialize the Qwen3ForCausalLMDMS model."""
        super().__init__(config)
        self.model = Qwen3ModelDMS(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_cache(self, preallocate_for_tokens: int | None = None):
        """Get the DMS cache for inference."""
        if preallocate_for_tokens is None:
            preallocate_for_tokens = self.config.dms_preallocate_for_tokens
        return DMSCache(
            dms_window_size=self.config.dms_window_size + 1,
            max_context_length=self.config.max_position_embeddings,
            accommodate_min_initial_context_length=preallocate_for_tokens,
            disable_eviction=self.config.dms_disable_eviction,
            block_size=self.config.dms_paged_attention_block_size,
        )

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: DMSCache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        dms_state: DMSTrainingStateAux = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> DMSCausalLMOutputWithPastAndCR:
        """A modified version of the forward pass from the transformers Qwen3ForCausalLM model."""
        if self.training:
            assert dms_state is not None, "dms_state is None in training mode"
        if (not self.training) and (
            (use_cache and past_key_values is None) or not isinstance(past_key_values, DMSCache)
        ):
            if past_key_values is not None:
                logger.warning(
                    f"past_key_values is of type {type(past_key_values)}, it will be replaced with an empty DMSCache!"
                )
            past_key_values = self.get_cache()

        outputs: DMSBaseModelOutputWithPastAndCR = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            dms_state=dms_state,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = (
            slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs
            )

        dms_frac_closed: torch.Tensor | None = outputs.dms_frac_closed
        dms_cr = None

        if dms_state is not None and dms_state.target_frac_to_close is not None:
            assert self.training, (
                "dms_state.target_frac_to_close is only supported in training mode"
            )
            assert dms_frac_closed is not None, "dms_frac_closed is None during training"
            dms_loss = torch.clamp(dms_state.target_frac_to_close - dms_frac_closed, min=0.0).mean()
            dms_frac_open = 1 - dms_frac_closed.detach().mean()
            dms_cr = 1 / torch.clamp(dms_frac_open, min=1e-6)
        else:
            assert (not self.training) or dms_state.dms_teacher_mode, (
                "dms_state.target_frac_to_close is required in training mode"
            )
            dms_loss = None

        if past_key_values is not None and past_key_values.current_mode == Mode.INFERENCE:
            dms_cr = past_key_values.get_cr()

        return DMSCausalLMOutputWithPastAndCR(
            loss=loss,
            dms_loss=dms_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cr=dms_cr,
            dms_frac_closed=dms_frac_closed.detach().mean()
            if dms_frac_closed is not None
            else None,
        )


class Qwen3ForSequenceClassificationDMS(GenericForSequenceClassification, Qwen3PreTrainedModelDMS):
    """Qwen3 model for sequence classification with DMS."""


class Qwen3ForTokenClassificationDMS(GenericForTokenClassification, Qwen3PreTrainedModelDMS):
    """Qwen3 model for token classification with DMS."""


class Qwen3ForQuestionAnsweringDMS(GenericForQuestionAnswering, Qwen3PreTrainedModelDMS):
    """Qwen3 model for question answering with DMS."""

    base_model_prefix = "transformer"  # For BC, where `transformer` was used instead of `model`


__all__ = [
    "Qwen3ForCausalLMDMS",
    "Qwen3ForQuestionAnsweringDMS",
    "Qwen3ForSequenceClassificationDMS",
    "Qwen3ForTokenClassificationDMS",
    "Qwen3ModelDMS",
    "Qwen3PreTrainedModelDMS",
]
