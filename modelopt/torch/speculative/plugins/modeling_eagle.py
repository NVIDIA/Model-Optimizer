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

"""EAGLE draft model architecture (EagleModule) and related data structures."""

from dataclasses import dataclass

import torch
from torch import nn
from transformers import Cache
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding

__all__ = ["EagleBaseModelOutput", "EagleModule", "EagleShareKVCache"]


class EagleShareKVCache(Cache):
    """HF Cache subclass implementing EAGLE3 ShareKV (sliding window of one).

    Each TTT step N reuses the (K, V) computed at step N-1, mirroring the Megatron
    MTP ShareKV reference (``prev_kv_list`` is injected for the current iteration and
    then refreshed from the current iteration's captured KV):

    - ``"capture"`` (TTT step 0): store the local post-RoPE (K, V) per layer and use
      them as-is (no prior step to reuse).
    - ``"inject"`` (TTT step N>0): store *this* step's local (K, V) for step N+1, but
      return the (K, V) captured at step N-1 so attention reuses the prior step's KV.

    Captured tensors keep their grad graph so backward at step N flows into step
    N-1's q/k/v projections.
    """

    def __init__(self, num_layers: int):
        """Init function for EagleShareKVCache."""
        # transformers 5.x Cache.__init__ requires `layers` or `layer_class_to_replicate`;
        # transformers 4.x takes no args. Try the new signature first and fall back.
        try:
            super().__init__(layers=[])
        except TypeError:
            super().__init__()
        self.num_layers = num_layers
        self._stored: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * num_layers
        self._mode: str = "passthrough"

    def set_mode(self, mode: str) -> None:
        """Set the mode of the cache."""
        assert mode in ("passthrough", "capture", "inject"), f"Unknown mode {mode!r}"
        self._mode = mode

    def reset(self) -> None:
        """Reset the cache."""
        self._stored = [None] * self.num_layers

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        """Capture local (K,V); in inject mode return the *previous* step's (K,V)."""
        if self._mode == "inject":
            prev = self._stored[layer_idx]
            # Refresh storage with this step's local KV so step N+1 reuses step N.
            self._stored[layer_idx] = (key_states, value_states)
            return prev if prev is not None else (key_states, value_states)
        if self._mode == "capture":
            self._stored[layer_idx] = (key_states, value_states)
        return key_states, value_states

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Get the sequence length of the cache, not used in ShareKV and always return 0."""
        return 0


class EagleModule(nn.Module):
    """Eagle module used in EAGLE model."""

    def __init__(self, config, decoder_layer_cls, bias=False):
        """Init function for EagleModule."""
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList(
            [decoder_layer_cls(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        if config.use_last_layernorm:
            self.norm = LlamaRMSNorm(config.hidden_size, config.rms_norm_eps)

        # Optionally, we use a smaller vocab table for eagle module
        if config.draft_vocab_size != config.vocab_size or config.has_lm_head:
            # Need an extra lm_head for eagle module since vocab size is reduced.
            assert config.draft_vocab_size <= config.vocab_size, (
                "EAGLE module's vocab size should be <= base model vocab size!"
            )
            # Initialize the buffers to zero.
            # Their values depend on specific tokenizer and calibration dataset, and should be set in training script.
            if config.draft_vocab_size < config.vocab_size:
                self.register_buffer("d2t", torch.zeros(config.draft_vocab_size, dtype=torch.int64))
            self.lm_head = nn.Linear(
                config.hidden_size,
                config.draft_vocab_size,
                bias=False,
            )

        if config.use_aux_hidden_state:
            # In EAGLE-3, the FC concentrate hidden states from multiple base model layers
            self.fc = nn.Linear(
                len(config.eagle_aux_hidden_state_layer_ids) * config.hidden_size,
                config.hidden_size,
                bias=bias,
            )

            first_layer_attn = self.layers[0].self_attn

            # Expand first attn input dim since it accepts cat(input_embeds, hidden_states)
            self._expand_first_attn_in_dim(first_layer_attn)

            # EAGLE-3's first attention require [input_layernorm_output, aux_hidden_states]
            first_layer_attn.register_forward_pre_hook(
                self._eagle3_attention_forward_pre_hook, with_kwargs=True
            )

            # In EAGLE-3, input_embeds and hidden_states are normalized separately before concatenation.
            self.layers[0].input_layernorm = LlamaRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.layers[0].hidden_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # ShareKV roll-query marker. Read by the attention-function patch in hf_eagle.py.
        for layer in self.layers:
            layer.self_attn._eagle_roll_query_active = False

    def _set_roll_query_active(self, enabled: bool) -> None:
        """Toggle the roll-query marker on every draft self-attention."""
        for layer in self.layers:
            layer.self_attn._eagle_roll_query_active = enabled

    def _maybe_init_rope(self, device=None):
        if self.config.eagle_decoder_type == "llama" and not hasattr(self, "rotary_emb"):
            self.rotary_emb = LlamaRotaryEmbedding(config=self.config, device=device)

    def _expand_first_attn_in_dim(self, first_layer_attn):
        """Modify qkv projection in first layer to accept 2h hidden size."""
        # Find Linear modules to expand
        eagle_attn_type = type(first_layer_attn)
        if eagle_attn_type.__name__ == "LlamaAttention":
            expand_modules = ["q_proj", "k_proj", "v_proj"]
        elif eagle_attn_type.__name__ == "DeepseekV3Attention":
            if first_layer_attn.q_lora_rank is None:
                expand_modules = ["q_proj", "kv_a_proj_with_mqa"]
            else:
                expand_modules = ["q_a_proj", "kv_a_proj_with_mqa"]
        else:
            raise ValueError(f"Unsupported attention type: {eagle_attn_type}")

        # Replace Linear with 2x input dim
        for module in expand_modules:
            original_linear = getattr(first_layer_attn, module)
            assert isinstance(original_linear, nn.Linear), f"Module {module} is not a Linear"
            setattr(
                first_layer_attn,
                module,
                nn.Linear(
                    original_linear.in_features * 2,
                    original_linear.out_features,
                    bias=first_layer_attn.config.attention_bias,
                ),
            )

    def _eagle3_attention_forward_pre_hook(self, module, args, kwargs):
        """Concat input_embeds and hidden_states for EAGLE-3's first attention layer."""
        if "hidden_states" not in kwargs:
            raise ValueError("hidden_states not found in kwargs")
        if self._input_embeds is None:
            raise ValueError("self._input_embeds is None")

        input_embeds = self._input_embeds
        self._input_embeds = None
        kwargs["hidden_states"] = torch.cat(
            (input_embeds, self.layers[0].hidden_norm(kwargs["hidden_states"])), dim=-1
        )

        return args, kwargs

    def forward(
        self,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = False,
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
        # In EAGLE-3, we save input embeddings to attribute, and use it in first decoder layer by hook function
        # Also, we normalize input embeddings and hidden states before concatenating them.
        # The default input norm in first layer attn will be disabled.
        self._input_embeds = self.layers[0].input_layernorm(inputs_embeds)

        if self.config.eagle_decoder_type == "llama":
            # rotary_emb must be pre-initialized by the caller (see HFEagleModel);
            # lazy init here would allocate inv_freq inside the torch.compile/CUDAGraph
            # capture region and get overwritten by subsequent runs.
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
        else:
            position_embeddings = None

        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
            )
            # For HF>= 4.54.0, the layer_outputs is a tensor, for older, it is a tuple.
            if isinstance(layer_outputs, tuple):
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs

        pre_norm_h = hidden_states

        post_norm_h = self.norm(hidden_states) if hasattr(self, "norm") else hidden_states

        return post_norm_h, pre_norm_h, past_key_values


@dataclass
class EagleBaseModelOutput:
    """Output container for base model forward pass in EAGLE training."""

    out_hiddens: torch.Tensor
    aux_hiddens: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    input_embeds: torch.Tensor | None = None
    loss: torch.Tensor | None = None

    @classmethod
    def from_offline_dict(cls, d: dict):
        """Construct from a dict of pre-computed base model outputs (offline training)."""
        return cls(
            out_hiddens=d.get("base_model_hidden_states"),
            aux_hiddens=d.get("aux_hidden_states"),
            logits=d.get("base_model_logits"),
            input_embeds=d.get("base_model_input_embeds"),
            loss=None,
        )
