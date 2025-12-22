# Adapted from: https://github.com/ctlllll/axolotl/blob/f86767e/src/axolotl/monkeypatch/medusa_utils.py
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

"""Support speculative decoding for huggingface models."""

import contextlib
import copy
from typing import Any, Optional, Unpack

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn.attention.flex_attention import BlockMask, create_block_mask
from transformers import Cache, DynamicCache, GradientCheckpointingLayer, LlamaConfig, PretrainedConfig, PreTrainedModel
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from transformers.activations import ACT2FN
from transformers.trainer_pt_utils import LabelSmoother
from transformers.utils import ModelOutput, TransformersKwargs

from ..eagle.conversion import EagleDMRegistry
from ..eagle.eagle_model import EagleModel
from ..eagle.utils import RMSNorm, expand_mask, make_causal_mask
from ..medusa.conversion import MedusaDMRegistry
from ..medusa.medusa_model import MedusaModel
from ..utils import AcceptanceRateValidation, ResBlock, temporary_set_config_value

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@MedusaDMRegistry.register({PreTrainedModel: "hf.PreTrainedModel"})
class HFMedusaModel(MedusaModel):
    """Medusa Model Class for huggingface models."""

    def modify(self, medusa_num_heads=0, medusa_num_layers=0):
        """Constructor.

        Args:
            medusa_num_heads: number of medusa heads.
            medusa_num_layers: number of ResBlock layers in each head.
        """
        super().modify(medusa_num_heads=medusa_num_heads, medusa_num_layers=medusa_num_layers)
        self.config.medusa = {
            "num_medusa_heads": medusa_num_heads,
            "num_medusa_layers": medusa_num_layers,
        }

        hidden_size = self.lm_head.weight.shape[-1]
        vocab_size = self.lm_head.weight.shape[0]

        # Create a list of Medusa heads
        self.medusa_heads = nn.ModuleList(
            [
                nn.Sequential(
                    *([ResBlock(hidden_size)] * self.medusa_num_layers),
                    nn.Linear(hidden_size, vocab_size, bias=False),
                )
                for _ in range(self.medusa_num_heads)
            ]
        )

        # Ensure medusa_head's dtype and device align with the base_model
        self.medusa_heads.to(self.lm_head.weight.dtype).to(self.lm_head.weight.device)
        self.medusa_heads.device = self.lm_head.weight.device
        if hasattr(self, "hf_device_map") and "lm_head" in self.hf_device_map:
            self.hf_device_map["medusa_heads"] = self.hf_device_map["lm_head"]

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        freeze_base_model: bool = True,
        medusa_heads_coefficient: float | None = 0.2,
        medusa_decay_coefficient: float | None = 0.8,
        **kwargs,
    ) -> Any:
        """Forward pass of the MedusaModel.

        Returns:
            torch.Tensor: A tensor containing predictions from all Medusa heads.
        """
        # Pass input through the base model
        with torch.no_grad() if freeze_base_model else contextlib.nullcontext():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                rcache_position=cache_position,
                **kwargs,
            )
            hidden_states = outputs.last_hidden_state
            # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
            slice_indices = (
                slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            )
            logits = self.lm_head(hidden_states[:, slice_indices, :])

        medusa_logits = [
            self.medusa_heads[i](hidden_states[:, slice_indices, :])
            for i in range(self.medusa_num_heads)
        ]

        if labels is not None:
            loss = 0
            loss_fct = CrossEntropyLoss()
            # Base model loss
            if not freeze_base_model:
                loss_logits = logits.view(-1, logits.shape[-1])
                loss_labels = labels.view(-1)
                base_model_loss = loss_fct(loss_logits, loss_labels)
                loss += base_model_loss
            # Medusa loss
            for i in range(self.medusa_num_heads):
                labels = labels[..., 1:].contiguous()
                loss_logits = medusa_logits[i][:, : -(1 + i)].contiguous()
                loss_logits = loss_logits.view(-1, loss_logits.shape[-1])
                loss_labels = labels.view(-1)
                loss += (
                    loss_fct(loss_logits, loss_labels)
                    * medusa_decay_coefficient**i
                    * medusa_heads_coefficient
                )
        else:
            loss = None

        return ModelOutput(
            loss=loss,
            logits=logits,
            medusa_logits=medusa_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.context_down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        inter_states = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        token_out_h = self.down_proj(inter_states)
        return token_out_h, token_out_h, None
        context_out_h = self.context_down_proj(inter_states)
        return token_out_h, context_out_h, None


DS_CONFIG_FOR_NEMOTRON = {
    "n_routed_experts": 128,
    "n_shared_experts": 1,
    "num_experts_per_tok": 6,
    "moe_intermediate_size": 1856,
    "moe_shared_expert_intermediate_size": 3712,
    "n_group": 1,
    "n_groups": 8,
    "topk_group": 1,
    "norm_topk_prob": True,
    "routed_scaling_factor": 2.5,
    "bias_update_speed": 0.001,
    "moe_balancing_loss_weight": 0.0001,
}

import torch.nn.functional as F

class DeepseekV3MLP(nn.Module):
    def __init__(self, config, intermediate_size):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

class DeepseekV3TopkRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_routed_experts = DS_CONFIG_FOR_NEMOTRON["n_routed_experts"]
        self.bias_update_speed = DS_CONFIG_FOR_NEMOTRON["bias_update_speed"]

        self.weight = nn.Linear(config.hidden_size, self.n_routed_experts, bias=False, dtype=torch.float32)

        # Temp accumulator for expert score correction bias between update steps
        self.register_buffer("e_score_correction_bias_temp_acc", torch.zeros(self.n_routed_experts, dtype=torch.int64))
        # Total net bias count for each expert. Tracking done in int64 to avoid accumulation errors.
        self.register_buffer("e_score_correction_bias_total_count", torch.zeros(self.n_routed_experts, dtype=torch.int64))
        # Bias scores calculated using the update speed
        self.register_buffer("e_score_correction_bias_factor", torch.zeros(self.n_routed_experts, dtype=torch.float32))

    def clear_temp_correction_accumulator(self):
        self.e_score_correction_bias_temp_acc.data.zero_()

    def apply_bias_update(self):
        expert_usage_counts = self.e_score_correction_bias_temp_acc.data
        total_selections = expert_usage_counts.sum().item()
        usage_fraction = expert_usage_counts.to(torch.float32) / (total_selections + 1e-20)
        target_fraction = 1.0 / self.n_routed_experts

        error = usage_fraction - target_fraction
        self.e_score_correction_bias_total_count.data -= error.sign().to(torch.int64)
        min_count = self.e_score_correction_bias_total_count.data.min().item()
        if min_count > 0:
            # Prevent unbounded growth of the bias counts
            self.e_score_correction_bias_total_count.data -= min_count
        max_bias_magnitude = 1.0 / self.bias_update_speed
        new_bias_scores = self.bias_update_speed * \
            torch.clamp(self.e_score_correction_bias_total_count.data.to(torch.float32),
                        None, max_bias_magnitude)
        self.e_score_correction_bias_factor.data[:] = new_bias_scores

        violation = error / target_fraction
        return violation

    def get_e_score_correction_bias(self):
        return self.e_score_correction_bias_factor.data

    def forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, self.config.hidden_size)
        router_logits = self.weight(hidden_states.to(torch.float32))
        return router_logits


class DeepseekV3NaiveMoe(nn.Module):
    """Collection of expert weights stored as 3D tensors."""

    def __init__(self, config):
        super().__init__()
        self.num_experts = DS_CONFIG_FOR_NEMOTRON["n_routed_experts"]
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = DS_CONFIG_FOR_NEMOTRON["moe_intermediate_size"]
        self.gate_up_projections = nn.ModuleList([
            nn.Linear(self.hidden_dim, 2 * self.intermediate_dim, bias=False)
            for _ in range(self.num_experts)
        ])
        self.down_projections = nn.ModuleList([
            nn.Linear(self.intermediate_dim, self.hidden_dim, bias=False)
            for _ in range(self.num_experts)
        ])
        # self.gate_up_proj = nn.Linear(self.hidden_dim, 2 * self.intermediate_dim, bias=False)
        # self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))

        # self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = self.gate_up_projections[expert_idx](current_state).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = self.down_projections[expert_idx](current_hidden_states)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


class DeepseekV3MoE(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.experts = DeepseekV3NaiveMoe(config)
        self.gate = DeepseekV3TopkRouter(config)
        self.shared_experts = DeepseekV3MLP(
            config=config, intermediate_size=DS_CONFIG_FOR_NEMOTRON["moe_shared_expert_intermediate_size"] * DS_CONFIG_FOR_NEMOTRON["n_shared_experts"]
        )
        self.n_routed_experts = DS_CONFIG_FOR_NEMOTRON["n_routed_experts"]
        self.n_group = DS_CONFIG_FOR_NEMOTRON["n_group"]
        self.topk_group = DS_CONFIG_FOR_NEMOTRON["topk_group"]
        self.norm_topk_prob = DS_CONFIG_FOR_NEMOTRON["norm_topk_prob"]
        self.routed_scaling_factor = DS_CONFIG_FOR_NEMOTRON["routed_scaling_factor"]
        self.top_k = DS_CONFIG_FOR_NEMOTRON["num_experts_per_tok"]

    def route_tokens_to_experts(self, router_logits):
        router_probs = router_logits.to(torch.float32).softmax(dim=-1)
        router_logits = router_logits.to(torch.float32).sigmoid()
        router_logits_for_choice = router_logits.to(torch.float32) + self.gate.get_e_score_correction_bias()
        group_scores = (
            router_logits_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = router_logits_for_choice.masked_fill(~score_mask.bool(), 0.0)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = router_logits.gather(1, topk_indices)
        load_balancing_loss = None
        if self.training:
            # Update expert score correction bias
            # 1. Calculate actual expert usage in this batch
            # Shape: (batch_size * seq_len, top_k) -> flattened
            with torch.no_grad():
                chosen_experts = topk_indices.flatten()
                # Count how many times each expert was selected
                expert_usage = torch.bincount(
                    chosen_experts, 
                    minlength=self.n_routed_experts
                )
                fraction_of_tokens_per_expert = expert_usage.to(torch.float32) / router_logits.shape[0]

            # Compute per-worker load-balancing loss
            gate_probs_mean_per_expert = router_probs.mean(dim=0)
            assert gate_probs_mean_per_expert.shape == fraction_of_tokens_per_expert.shape, \
                f"Shape mismatch: {gate_probs_mean_per_expert.shape} vs {fraction_of_tokens_per_expert.shape}"
            load_balancing_loss = torch.sum(
                gate_probs_mean_per_expert.to(torch.float32) * fraction_of_tokens_per_expert.to(torch.float32)
            ) * DS_CONFIG_FOR_NEMOTRON["moe_balancing_loss_weight"] / DS_CONFIG_FOR_NEMOTRON["n_routed_experts"]

            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(expert_usage, op=torch.distributed.ReduceOp.SUM)
            
            self.gate.e_score_correction_bias_temp_acc.data += expert_usage.to(torch.int64)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights, load_balancing_loss

    def forward(self, hidden_states):
        residuals = hidden_states
        orig_shape = hidden_states.shape
        router_logits = self.gate(hidden_states)
        topk_indices, topk_weights, aux_loss = self.route_tokens_to_experts(router_logits)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self.experts(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states, hidden_states, aux_loss

class LlamaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)

        # self.mlp = LlamaMLP(config)
        self.mlp = DeepseekV3MoE(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(hidden_states)
        assert isinstance(mlp_output, tuple), \
            f"MLP output should be a tuple, got {type(mlp_output)}"
        hidden_states, context_out_h, aux_loss = mlp_output
        hidden_states = residual + hidden_states
        context_out_h = residual + context_out_h
        return hidden_states, context_out_h, aux_loss

class EagleModule(nn.Module):
    """Eagle module used in EAGLE model."""

    def __init__(self, config, decoder_layer_cls, bias=False):
        """Init function for EagleModule."""
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        if config.use_last_layernorm:
            self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        # Optionally, we use a smaller vocab table for eagle module
        if config.draft_vocab_size != config.vocab_size or config.has_lm_head:
            # Need an extra lm_head for eagle module since vocab size is reduced.
            assert config.draft_vocab_size <= config.vocab_size, (
                "EAGLE module's vocab size should be <= base model vocab size!"
            )

            # Initialize the buffers to zero.
            # Their values depend on specific tokenzier and calibrate dataset, and should be set in training script.
            if config.draft_vocab_size < config.vocab_size:
                self.register_buffer("d2t", torch.zeros(config.draft_vocab_size, dtype=torch.int64))
            self.eagle_lm_head = nn.Linear(
                config.hidden_size,
                config.draft_vocab_size,
                bias=False,
            )

        if not config.use_aux_hidden_state:
            num_layer_features = 0
        else:
            # In EAGLE3, the FC concatenates hidden states from multiple base model layers
            num_layer_features = len(config.eagle_aux_hidden_state_layer_ids)
        if num_layer_features > 0:
            self.fc = nn.Linear(
                num_layer_features * config.hidden_size,
                config.hidden_size,
                bias=bias,
            )
        
        if True:
            first_layer_attn = self.layers[0].self_attn
            if not isinstance(first_layer_attn, LlamaAttention):
                raise ValueError("EAGLE-3 only support LlamaAttention.")

            # EAGLE-3's first attention require [input_layernorm_output, aux_hidden_states]
            first_layer_attn.register_forward_pre_hook(
                self._eagle3_attention_forward_pre_hook, with_kwargs=True
            )

            # Modify qkv projection in first layer to accept 2h hidden size.
            first_layer_attn.q_proj = nn.Linear(
                first_layer_attn.q_proj.in_features * 2,
                first_layer_attn.q_proj.out_features,
                bias=first_layer_attn.config.attention_bias,
            )
            first_layer_attn.k_proj = nn.Linear(
                first_layer_attn.k_proj.in_features * 2,
                first_layer_attn.k_proj.out_features,
                bias=first_layer_attn.config.attention_bias,
            )
            first_layer_attn.v_proj = nn.Linear(
                first_layer_attn.v_proj.in_features * 2,
                first_layer_attn.v_proj.out_features,
                bias=first_layer_attn.config.attention_bias,
            )

            # In EAGLE-3, input_embeds and hidden_states are normalized separately before concatenation.
            self.input_embeds_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.hidden_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

            # Disable input norm in first layer. We normed embeds and h individually before.
            self.layers[0].input_layernorm = nn.Identity()

    def _eagle3_attention_forward_pre_hook(self, module, args, kwargs):
        """Concat input_embeds and hidden_states for EAGLE-3's first attention layer."""
        if "hidden_states" not in kwargs:
            raise ValueError("hidden_states not found in kwargs")
        if self._input_embeds is None:
            raise ValueError("self._input_embeds is None")

        input_embeds = self._input_embeds
        self._input_embeds = None
        kwargs["hidden_states"] = torch.cat(
            (input_embeds, self.hidden_norm(kwargs["hidden_states"])), dim=-1
        )

        return args, kwargs

    def forward(
        self,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor | None = None,
        logits: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = False,
        position_embeddings: torch.Tensor | None = None,
    ):
        """Forward function for EagleModule."""
        batch_size, seq_length, _ = hidden_states.shape
        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values.get_seq_length()
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if position_ids is None:
            device = hidden_states.device if hidden_states is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        inputs_embeds = inputs_embeds.to(hidden_states.dtype).to(hidden_states.device)
        if self.config.use_aux_hidden_state or not hasattr(self, "fc"):
            # In EAGLE-3, we save input embeddings to attribute, and use it in first decoder layer by hook function
            # Also, we normalize input embeddings and hidden states before concatenating them.
            # The default input norm in first layer attn will be disabled.
            self._input_embeds = self.input_embeds_norm(inputs_embeds)
        else:
            # EAGLE-1
            hidden_states = self.fc(torch.cat((inputs_embeds, hidden_states), dim=-1))


        aux_loss_acc = None
        for decoder_layer in self.layers:
            hidden_states, context_out_h, aux_loss = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
            )
            if aux_loss is not None:
                if aux_loss_acc is None:
                    aux_loss_acc = aux_loss
                else:
                    aux_loss_acc = aux_loss_acc + aux_loss

        pre_norm_h = context_out_h

        post_norm_h = self.norm(hidden_states) if hasattr(self, "norm") else hidden_states
        return post_norm_h, pre_norm_h, past_key_values, aux_loss_acc


@EagleDMRegistry.register({PreTrainedModel: "hf.PreTrainedModel"})
class HFEagleModel(EagleModel):
    """Eagle Model Class for huggingface models."""

    # Use functions to get base model parts without creating tied modules.
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
        """Return the llm config for the base model, from LLM or VLM."""
        return self.config.llm_config if hasattr(self.config, "llm_config") else self.config

    def _find_base_model_parts(self):
        """Find model parts from different models and set base_{part}_path attributes."""
        base_model_parts_mapping = {
            "base_model_path": ["model", "backbone", "language_model.backbone"],
            "base_model_embeddings_path": [
                "model.embed_tokens",
                "backbone.embeddings",
                "language_model.backbone.embeddings",
            ],
            "base_model_lm_head_path": ["lm_head", "language_model.lm_head"],
        }

        for name, paths in base_model_parts_mapping.items():
            found_submodule = False
            for path in paths:
                try:
                    submodule = self.get_submodule(path)
                    assert isinstance(submodule, torch.nn.Module)
                    print(f"Found {name} at {path}")
                    found_submodule = True
                    setattr(self, name, path)
                    break
                except Exception:
                    continue
            if not found_submodule:
                raise ValueError(f"Part {name} not found in model")

    def _set_default_aux_hidden_state_layers(self):
        # Read a custom config attribute since we override num_hidden_layers for offline training
        num_layers = self._base_llm_config.num_hidden_layers
        if self.eagle_offline and (num_layers is None or num_layers <= 0):
            num_layers = getattr(self.config, "num_orig_hidden_layers", 0)

        self.eagle_config.eagle_aux_hidden_state_layer_ids = [
            1,
            max(0, num_layers // 2 - 1),
            max(0, num_layers - 4),
        ]
        self.eagle_config.eagle_aux_hidden_state_layer_ids = list(
            set(self.eagle_config.eagle_aux_hidden_state_layer_ids)
        )

    def _collect_aux_hidden_states_forward_hook(self, module, input, output) -> None:
        """Collect auxiliary hidden states from base model intermediate layers, save them in attribute."""
        hidden_states = (
            output.clone().detach()
            if isinstance(output, torch.Tensor)
            else output[0].clone().detach()
        )
        self._aux_hidden_states.append(hidden_states)

    def pop_and_gather_aux_hiddens(self):
        """Pop auxiliary hidden states from base model and gather them on the draft model device."""
        # In PTQ, forward method will be called with try and except to find max batch size.
        # This leads to uncleared aux hidden states in the front of the list.
        # To fix it, we only return the last num_aux_h items in the list.
        num_aux_h = len(self.eagle_config.eagle_aux_hidden_state_layer_ids)
        aux_h_list = self._aux_hidden_states[-num_aux_h:]
        self._aux_hidden_states.clear()

        # Gather aux hidden states on the draft model device
        aux_h_list = [h.to(self.eagle_module.fc.weight.device) for h in aux_h_list]

        return aux_h_list

    def _get_eagle_device(self):
        """Return the device where we should place eagle module."""
        if self.eagle_offline:
            # For offline training, the base model has no layers.
            # Read the device from the base model lm_head instead.
            return self._base_model_lm_head.weight.device
        else:
            # When there is a base model, put eagle on the last layer's device.
            base_model_last_layer = self._base_model.layers[-1]
            return next(base_model_last_layer.parameters()).device

    def modify(
        self,
        eagle_offline,
        eagle_hidden_state_distillation,
        eagle_self_logit_distillation,
        eagle_freeze_base_model,
        eagle_report_acc,
        eagle_reuse_base_decoder,
        eagle_loss_decay_factor,
        eagle_architecture_config,
    ):
        """Constructor.

        Args:
            config: The config for eagle decoder layers.
        """
        super().modify(
            eagle_offline=eagle_offline,
            eagle_hidden_state_distillation=eagle_hidden_state_distillation,
            eagle_self_logit_distillation=eagle_self_logit_distillation,
            eagle_freeze_base_model=eagle_freeze_base_model,
            eagle_report_acc=eagle_report_acc,
            eagle_reuse_base_decoder=eagle_reuse_base_decoder,
            eagle_loss_decay_factor=eagle_loss_decay_factor,
            eagle_architecture_config=eagle_architecture_config,
        )
        self.eagle_config = PretrainedConfig.from_dict(eagle_architecture_config)
        if self.eagle_config._attn_implementation is None:
            self.eagle_config._attn_implementation = "sdpa"
        assert not self.eagle_reuse_base_decoder, "EAGLE-3 does not support reusing base decoder."
        decoder_cls = None
        # decoder_cls = (
        #     type(self.model.layers[-1]) if self.eagle_reuse_base_decoder else LlamaDecoderLayer
        # )

        # Use default aux_hidden_state layers if use_aux_hidden_state is True
        # but no layer id is given
        if (
            self.eagle_config.use_aux_hidden_state
            and len(self.eagle_config.eagle_aux_hidden_state_layer_ids) == 0
        ):
            self._set_default_aux_hidden_state_layers()

        if self._base_llm_config.hidden_size != self.eagle_config.hidden_size:
            raise ValueError(
                "EAGLE module hidden size "
                f"{self.eagle_config.hidden_size} must match base model hidden size "
                f"{self._base_llm_config.hidden_size}!"
            )

        self.eagle_module = EagleModule(
            self.eagle_config,
            decoder_cls,
        )
        self.eagle_rotary_emb = LlamaRotaryEmbedding(config=self.eagle_config)

        # find base model, lm head, and embeddings paths
        self._find_base_model_parts()
        self.eagle_module.to(self._base_model.dtype).to(self._get_eagle_device())

        # Make sure word embedding and lm head are frozen
        for param in self._base_model_embeddings.parameters():
            param.requires_grad = False
        for param in self._base_model_lm_head.parameters():
            param.requires_grad = False

        # EAGLE-3 auxiliary hidden_states
        if (not eagle_offline) and self.eagle_config.use_aux_hidden_state:
            self._aux_hidden_states = []
            for layer_idx, layer in enumerate(self._base_model.layers):
                if layer_idx in self.eagle_config.eagle_aux_hidden_state_layer_ids:
                    layer.register_forward_hook(self._collect_aux_hidden_states_forward_hook)

        # delete base model layers for offline training
        if eagle_offline:
            self._base_model._modules.pop("layers")

        # NOTE: this is a temporary hack to bypass hf trainer check:
        # https://github.com/huggingface/transformers/blob/v4.56-release/src/transformers/trainer.py#L566
        self.is_quantized = False

        self.num_ttt_steps = 3  # NOTE: (hg) hardcoded for now. Should add to cfg.
        self._cached_attn_blk_masks = {}

    def _get_ttt_attention_mask(self, seq_length, ttt_step):
        # compile and cached flex attention masks in first call
        if ttt_step not in self._cached_attn_blk_masks:
            self._cached_attn_blk_masks.update(
                {ttt_step: self._compute_ttt_attention_mask(seq_length, ttt_step)}
            )
        return self._cached_attn_blk_masks[ttt_step]

    def _prepare_decoder_attention_mask(
        self, attention_mask, input_shape, inputs_embeds, past_key_values_length
    ):
        """Expand the 2-D attention mask to 4-D and apply causal mask."""
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            ).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def _get_eagle_module_inputs(
        self,
        input_ids,
        eagle_input_hidden_states,
        attention_mask,
        position_ids,
        eagle_cache,
    ):
        """Helper function to prepare eagle inputs for the 0th eagle forward pass."""
        b, seq_length, _ = eagle_input_hidden_states.shape
        past_key_values_length = eagle_cache.get_seq_length() if eagle_cache is not None else 0
        seq_length_with_past = seq_length + past_key_values_length

        # Prepare eagle_input_ids: Shift left 1 token
        zeropadding = torch.zeros(
            input_ids.shape[0], 1, dtype=input_ids.dtype, device=input_ids.device
        )
        eagle_input_ids = torch.cat((input_ids[:, 1:], zeropadding), dim=1)

        # Prepare attention_mask
        if attention_mask is not None:  # Shift left 1 token for attention_mask
            zeropadding = torch.zeros(
                attention_mask.shape[0], 1, dtype=attention_mask.dtype, device=attention_mask.device
            )
            attention_mask = torch.cat((attention_mask[:, 1:], zeropadding), dim=1)
        else:
            attention_mask = torch.ones(  # Initialize default attention_mask
                (b, seq_length_with_past), dtype=torch.bool, device=eagle_input_hidden_states.device
            )

        # Expand the 2-D attention mask to 4-D and apply causal mask.
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (b, seq_length), eagle_input_hidden_states, past_key_values_length
        )

        # Prepare position_ids
        if position_ids is None:
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=eagle_input_hidden_states.device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        return eagle_input_ids, attention_mask, position_ids

    def _compute_ttt_attention_mask(self, seq_length, ttt_step) -> BlockMask | torch.Tensor:
        """Return TTT attention_mask tensor of type BlockMask or Tensor depends on eagle attn impl."""

        def msk_func(b, h, q_idx, kv_idx):
            mask = kv_idx <= (q_idx - ttt_step)
            for i in range(1, ttt_step + 1):
                mask_block_i = (kv_idx == q_idx + i * seq_length - (ttt_step - i)) & (
                    kv_idx >= seq_length * i
                )
                mask = mask | mask_block_i
            return mask

        dtypemin = torch.finfo(self._base_llm_config.dtype).min
        q_len = seq_length
        kv_len = seq_length * (1 + ttt_step)
        if self.eagle_module.config._attn_implementation == "flex_attention":
            # Return block mask for flex attention
            block_mask = create_block_mask(msk_func, B=None, H=None, Q_LEN=q_len, KV_LEN=kv_len)
            return block_mask
        else:
            # Return tensor mask for non-flex attention
            tensor_mask = msk_func(
                None,
                None,
                torch.arange(q_len).view(1, 1, q_len, 1),
                torch.arange(kv_len).view(1, 1, 1, kv_len),
            ).to(self.device)
            tensor_mask = torch.full_like(
                tensor_mask, 0, dtype=self._base_llm_config.dtype, device=self.device
            ).masked_fill(~tensor_mask, dtypemin)
            return tensor_mask

    def _llm_or_vlm_embedding(self, input_ids, kwargs):
        """Return input embeddings with possibly vision embeddings for VLM."""
        tok_embeds = self._base_model_embeddings(input_ids)

        # LLM only have token embeddings
        if "pixel_values" not in kwargs:
            return tok_embeds

        # Otherwise, insert vision embeddings in tok_embeds
        if self.config.model_type == "NemotronH_Nano_VL_V2":
            vit_embeds = self.extract_feature(kwargs["pixel_values"])
            vit_embeds = vit_embeds[kwargs["image_flags"] == 1]
            bs, seq_len, hid_size = tok_embeds.shape
            tok_embeds = tok_embeds.reshape(bs * seq_len, hid_size)
            input_ids = input_ids.reshape(bs * seq_len)
            selected = input_ids == self.img_context_token_id
            try:
                tok_embeds[selected] = tok_embeds[selected] * 0.0 + vit_embeds.reshape(-1, hid_size)
            except Exception as e:
                vit_embeds = vit_embeds.reshape(-1, hid_size)
                print(
                    f"warning: {e}, tok_embeds[selected].shape={tok_embeds[selected].shape}, "
                    f"vit_embeds.shape={vit_embeds.shape}"
                )
                n_token = selected.sum()
                tok_embeds[selected] = tok_embeds[selected] * 0.0 + vit_embeds[:n_token]
            del vit_embeds
            return tok_embeds.reshape(bs, seq_len, hid_size)
        else:
            raise ValueError(f"VLM model type {self.config.model_type} not supported")

    def _base_model_forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        past_key_values,
        freeze_base_model,
        labels,
        **kwargs,
    ):
        # TODO: This function still use eagle_module. Ideally we should remove it,
        # so we can del model.eagle_module on the base model ranks to save memory.
        with torch.no_grad() if freeze_base_model else contextlib.nullcontext():
            outputs = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_hidden_states=True,
                **kwargs,
            )
            past_key_values = getattr(outputs, "past_key_values", None)
            base_model_hidden_states = outputs.hidden_states[-1]
            base_model_logits = outputs.logits

            # Optionally, compute base model loss when we want to tune the base model.
            base_model_loss = None
            if not freeze_base_model and labels is not None:  # Base model loss
                loss_fct = CrossEntropyLoss()
                loss_logits = base_model_logits.view(-1, base_model_logits.shape[-1])
                labels = labels.view(-1)
                base_model_loss = loss_fct(loss_logits, labels)

        # Map the base model logits to the draft vocab
        if self.eagle_config.draft_vocab_size != self.eagle_config.vocab_size and self.training:
            assert hasattr(self.eagle_module, "d2t"), "d2t buffer not initialized"
            base_model_logits = self._map_logits_to_draft_vocab(base_model_logits)

        return base_model_hidden_states, base_model_logits, base_model_loss, past_key_values

    def _map_logits_to_draft_vocab(self, full_logits):
        reverse_mapping = (
            torch.arange(len(self.eagle_module.d2t)).to(self.eagle_module.d2t.device)
            + self.eagle_module.d2t
        )
        return full_logits[:, :, reverse_mapping]

    def _eagle_forward(
        self,
        eagle_input_hidden_states,
        inputs_embeds,
        attention_mask,
        position_ids,
        position_embeddings,
        eagle_cache=None,
    ):
        eagle_postnorm_h, eagle_prenorm_h, eagle_cache, aux_loss = self.eagle_module(
            eagle_input_hidden_states,
            inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            position_embeddings=position_embeddings,
            past_key_values=eagle_cache,
        )
        eagle_lm_head = (
            self.eagle_module.eagle_lm_head
            if hasattr(self.eagle_module, "eagle_lm_head")
            else self._base_model_lm_head
        )
        eagle_logits = eagle_lm_head(eagle_postnorm_h)

        return eagle_postnorm_h, eagle_prenorm_h, eagle_logits, eagle_cache, aux_loss

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int = 0,
        loss_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> Any:
        """Forward pass of the EagleModel.

        Returns:
            hidden_states: The hidden state from the base model.
            logits: logits from the base model.
            eagle_hidden_states: The hidden state from eagle_module.
            eagle_logits: logits from the eagle_module.
        """
        if past_key_values is not None and hasattr(past_key_values, "eagle_cache"):
            eagle_cache = past_key_values.eagle_cache
        else:
            eagle_cache = None

        if self.training:
            assert eagle_cache is None, "eagle_cache should be None in training"
            assert past_key_values is None, "past_key_values should be None in training"

        if loss_mask is None:
            loss_mask = torch.ones_like(input_ids, dtype=torch.bool, device=input_ids.device)

        # ====First, we run base model forward====
        if "base_model_outputs" in kwargs:
            # Parse base model outputs forwarded from teacher
            base_outputs = kwargs["base_model_outputs"]
            base_model_hidden_states = base_outputs["base_model_hidden_states"]
            if "base_model_logits" in base_outputs:
                base_model_logits = base_outputs["base_model_logits"]
            else:
                base_model_logits = self.lm_head(self.backbone.norm_f(base_model_hidden_states))
                if self.eagle_config.draft_vocab_size != self.eagle_config.vocab_size:
                    base_model_logits = self._map_logits_to_draft_vocab(base_model_logits)
            base_model_loss = None
            past_key_values = DynamicCache()  # Dummy cache
        else:
            base_model_hidden_states, base_model_logits, base_model_loss, past_key_values = (
                self._base_model_forward(
                    input_ids,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    self.eagle_freeze_base_model,
                    labels,
                    **kwargs,
                )
            )

        if not isinstance(past_key_values, Cache):
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        if not isinstance(eagle_cache, Cache):
            eagle_cache = DynamicCache.from_legacy_cache(eagle_cache)

        # ====Run eagle forward====
        eagle_losses = []
        aux_losses = []
        train_accs = []
        # In EAGLE-3, we have an additional FC layer to concentrate hidden states from multiple base model layers
        b, seq_length, h = base_model_hidden_states.shape
        if self.eagle_config.use_aux_hidden_state:
            if "base_model_outputs" in kwargs:
                aux_hidden_states = kwargs["base_model_outputs"]["aux_hidden_states"]
            else:
                aux_hidden_states = torch.cat(self.pop_and_gather_aux_hiddens(), dim=-1)
            eagle_input_hidden_states = aux_hidden_states
        else:
            eagle_input_hidden_states = base_model_hidden_states

        if self.training:
            with torch.no_grad():
                # Add some noise to the input hidden states
                noise = (
                    (torch.rand_like(eagle_input_hidden_states) - 0.5)
                    * 0.2 # Magic number: noise scale used in EAGLE paper
                    * 512
                    / eagle_input_hidden_states.shape[-1]
                )
                eagle_input_hidden_states = eagle_input_hidden_states + noise
        
        if self.eagle_config.use_aux_hidden_state:
            eagle_input_hidden_states = self.eagle_module.fc(eagle_input_hidden_states)

        # Get eagle inputs for the first eagle forward pass
        eagle_input_ids, attention_mask_0, position_ids = self._get_eagle_module_inputs(
            input_ids,
            eagle_input_hidden_states,
            attention_mask,
            position_ids,
            eagle_cache,
        )
        with torch.no_grad():
            inputs_embeds = self._llm_or_vlm_embedding(eagle_input_ids, kwargs)

        position_embeddings = self.eagle_rotary_emb(eagle_input_hidden_states, position_ids)
        past_key_values.eagle_cache = eagle_cache

        # ====Perform training-time-testing with 3 extra eagle forward passes====
        for ttt_step in range(self.num_ttt_steps):
            attention_mask = (
                attention_mask_0
                if ttt_step == 0
                else self._get_ttt_attention_mask(seq_length, ttt_step)
            )
            _, eagle_input_hidden_states, eagle_logits, eagle_cache, aux_loss = self._eagle_forward(
                eagle_input_hidden_states,
                inputs_embeds,
                attention_mask,
                position_ids,
                position_embeddings,
                eagle_cache,
            )
            eagle_input_hidden_states = torch.cat(
                (
                    torch.zeros(
                        (b, 1, h),
                        dtype=eagle_input_hidden_states.dtype,
                        device=eagle_input_hidden_states.device,
                    ),
                    eagle_input_hidden_states[:, :-1, :],
                ),
                dim=1,
            )
            classification_loss, acc = self._eagle_loss(
                # base model predict +1 tok, while eagle predict +2
                # so we shift base model outputs compared to eagle outputs
                base_model_logits[:, 1:],
                eagle_logits[:, :-1],
                # additionally, we mask the first n tok of eagle outputs at nth TTT step
                torch.cat(
                    (
                        torch.zeros(b, ttt_step, dtype=loss_mask.dtype, device=loss_mask.device),
                        loss_mask[:, 1 + ttt_step :],
                    ),
                    dim=1,
                ),
            )
            # is_in_last_2_steps = ttt_step >= (self.num_ttt_steps - 2)
            # if is_in_last_2_steps:
            eagle_losses.append(classification_loss)
            train_accs.append(acc)
            if aux_loss is not None:
                aux_losses.append(aux_loss)
        eagle_loss = torch.stack(eagle_losses).mean()
        aux_loss = torch.mean(torch.stack(aux_losses)) if len(aux_losses) > 0 else None
        # Finally, we merge base model loss and eagle loss, raise error if both are None
        if base_model_loss is not None and eagle_loss is not None:
            loss = base_model_loss + eagle_loss
        elif base_model_loss is not None:
            loss = base_model_loss
        elif eagle_loss is not None:
            loss = eagle_loss
        else:
            loss = None
            assert not self.training, ValueError(
                "Both base_model_loss and eagle_loss are skipped. At least one loss must be computed."
            )
        if aux_loss is not None:
            loss = loss + aux_loss

        return ModelOutput(
            loss=loss,
            logits=base_model_logits,
            past_key_values=past_key_values,
            hidden_states=base_model_hidden_states,
            train_acc=train_accs,
        )

    def _eagle_loss(
        self,
        base_model_logits,
        eagle_logits,
        loss_mask,
    ):
        """Function for EAGLE loss computing."""
        loss_mask = loss_mask[:, :, None]
        classification_loss = nn.Softmax(dim=2)(base_model_logits) * nn.LogSoftmax(dim=2)(
            eagle_logits
        )
        classification_loss = -torch.sum(torch.sum(loss_mask * classification_loss, 2)) / (
            loss_mask.sum() + 1e-5
        )
        # Compute accuracy
        base_predict_tok = base_model_logits.clone().detach().argmax(dim=-1)
        eagle_predict_tok = eagle_logits.clone().detach().argmax(dim=-1)
        valid = loss_mask[:, :, 0].bool()
        correct = (base_predict_tok == eagle_predict_tok) & valid
        denom = valid.sum().clamp_min(1).float()
        accuracy = round(correct.sum().float().div(denom).item(), 3)

        return classification_loss, accuracy

    @torch.no_grad()
    def pseudo_speculative_generate(
        self,
        input_ids: torch.Tensor,
        steps: int = 1,
    ):
        """Pseudo generate of the EAGLE GPTModel.

        Returns:
            base_token (torch.Tensor): token from base model
            draft_tokens (torch.Tensor): draft tokens from eagle module
        """
        base_model_outputs = super().forward(
            input_ids=input_ids,
            output_hidden_states=True,
        )

        base_model_hidden_states = base_model_outputs.hidden_states[-1]
        base_model_logits = base_model_outputs.logits
        base_token = base_model_logits[:, -1:, :].argmax(dim=-1).to(input_ids.device)

        # Early return
        if steps < 1:
            if hasattr(self, "_aux_hidden_states"):
                _ = self.pop_and_gather_aux_hiddens()
            return base_token, None

        eagle_ids = torch.cat((input_ids[:, 1:], base_token), dim=-1)

        if self.eagle_config.use_aux_hidden_state:
            # EAGLE-3
            # Only the first iteration input_hidden_states are from aux_hidden_state layers
            # Gather _aux_hidden_states from all devices before concatenation
            gathered_aux_hidden_states = self.pop_and_gather_aux_hiddens()
            eagle_input_hidden_states = self.eagle_module.fc(
                torch.cat(gathered_aux_hidden_states, dim=-1)
            )

        else:
            eagle_input_hidden_states = base_model_hidden_states

        draft_tokens = []
        for _ in range(steps):
            # Get eagle inputs for the first eagle forward pass
            _, eagle_attention_mask, eagle_position_ids = self._get_eagle_module_inputs(
                input_ids,
                eagle_input_hidden_states,
                None,
                None,
                None,
            )
            position_embeddings = self.eagle_rotary_emb(
                eagle_input_hidden_states, eagle_position_ids
            )

            # Use SDPA attention during generation for both stability and performance
            with temporary_set_config_value(
                self.eagle_module.config, "_attn_implementation", "sdpa"
            ):
                _, eagle_prenorm_h, eagle_logits, _, _ = self._eagle_forward(
                    eagle_input_hidden_states,
                    self._base_model_embeddings(eagle_ids),
                    eagle_attention_mask,
                    eagle_position_ids,
                    position_embeddings,
                )

            draft_token = eagle_logits[:, -1:, :].argmax(dim=-1)
            if self.eagle_config.draft_vocab_size != self.eagle_config.vocab_size:
                draft_token += self.eagle_module.d2t[draft_token]
            draft_tokens.append(draft_token)

            eagle_ids = torch.cat((eagle_ids, draft_token.to(eagle_ids.device)), dim=-1)
            eagle_input_hidden_states = torch.cat(
                (eagle_input_hidden_states, eagle_prenorm_h[:, -1:, :]), dim=1
            )

        draft_tokens = torch.cat(draft_tokens, dim=-1).to(base_token.device)

        return base_token, draft_tokens


class HFARValidation(AcceptanceRateValidation):
    """This is the subclass for HF model AR validation."""

    def get_ground_truth(self, input_ids, osl):
        """This function returns ground truth output tokens from the base model."""
        input_ids = copy.deepcopy(input_ids).to(torch.cuda.current_device())
        for _ in range(osl):
            input_id, _ = self.model.pseudo_speculative_generate(input_ids, steps=0)
            input_ids = torch.cat((input_ids, input_id.to(input_ids.device)), dim=-1)
            if input_id[0, 0] == self.end_token:
                break
        return input_ids
