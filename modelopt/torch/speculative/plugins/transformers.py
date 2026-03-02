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
from dataclasses import dataclass
from typing import Any

import torch
import transformers
from packaging.version import Version
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn.attention.flex_attention import BlockMask, create_block_mask
from transformers import Cache, DynamicCache, PretrainedConfig, PreTrainedModel
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from transformers.trainer_pt_utils import LabelSmoother
from transformers.utils import ModelOutput
from transformers.utils.quantization_config import CompressedTensorsConfig

from ..eagle.conversion import EagleDMRegistry
from ..eagle.eagle_model import EagleModel
from ..eagle.utils import expand_mask, make_causal_mask
from ..medusa.conversion import MedusaDMRegistry
from ..medusa.medusa_model import MedusaModel
from ..utils import (
    AcceptanceRateValidation,
    ResBlock,
    _setup_kimi_k2_decoder,
    enable_cp_ttt_patch,
    get_ttt_msk_func,
    temporary_set_config_value,
)

__all__ = ["HFARValidation", "HFEagleModel", "HFMedusaModel"]

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
ENABLE_CP_TTT_PATCH = False
# module variable to cache attention mask for cp ttt
CACHED_SHARD_TTT_MASKS = {}


def _get_empty_cache(config):
    """Return an empty cache. Handle different versions of transformers for unit tests."""
    if Version(transformers.__version__) >= Version("4.54"):
        return DynamicCache(config=config)
    else:
        return DynamicCache()


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
                    *([ResBlock(hidden_size) for _ in range(self.medusa_num_layers)]),
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


class ParallelDraft(nn.Module):
    """ParallelDraft module with multiple Medusa heads and a shared lm head."""

    def __init__(self, hidden_size: int, vocab_size: int, num_heads: int = 1, num_layers: int = 1):
        """Init function for ParallelDraft."""
        super().__init__()

        self.medusa_heads = torch.nn.ModuleList(
            [
                nn.Sequential(
                    *([ResBlock(hidden_size) for _ in range(num_layers)]),
                )
                for _ in range(num_heads)
            ]
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x):
        """Forward function."""
        output = []
        for head in self.medusa_heads:
            x_head = head(x)
            output.append(self.lm_head(x_head))
        return output


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
            # Their values depend on specific tokenzier and calibrate dataset, and should be set in training script.
            if config.draft_vocab_size < config.vocab_size:
                self.register_buffer("d2t", torch.zeros(config.draft_vocab_size, dtype=torch.int64))
            self.lm_head = nn.Linear(
                config.hidden_size,
                config.draft_vocab_size,
                bias=False,
            )

        if not config.use_aux_hidden_state:
            # In Eagle-1, the FC concentrate input embeddings and hidden states
            self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=bias)
        else:
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

        if self.config.parallel_draft_step > 1:
            self.parallel_draft_heads = ParallelDraft(
                config.hidden_size,
                config.draft_vocab_size,
                num_heads=self.config.parallel_draft_step - 1,
                num_layers=self.config.parallel_draft_heads_num_layers,
            )

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
        if self.config.use_aux_hidden_state:
            # In EAGLE-3, we save input embeddings to attribute, and use it in first decoder layer by hook function
            # Also, we normalize input embeddings and hidden states before concatenating them.
            # The default input norm in first layer attn will be disabled.
            self._input_embeds = self.layers[0].input_layernorm(inputs_embeds)
        else:  # EAGLE-1
            hidden_states = self.fc(torch.cat((inputs_embeds, hidden_states), dim=-1))

        if self.config.eagle_decoder_type == "llama":
            # Lazy init rope to avoid save/load meta tensor error
            if not hasattr(self, "rotary_emb"):
                self.rotary_emb = LlamaRotaryEmbedding(
                    config=self.config, device=hidden_states.device
                )
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
    out_hiddens: torch.Tensor
    aux_hiddens: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    input_embeds: torch.Tensor | None = None
    loss: torch.Tensor | None = None

    @classmethod
    def from_offline_dict(cls, d: dict):
        return cls(
            out_hiddens=d.get("base_model_hidden_states"),
            aux_hiddens=d.get("aux_hidden_states"),
            logits=d.get("base_model_logits"),
            input_embeds=d.get("base_model_input_embeds"),
            loss=None,
        )


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
        return (
            getattr(self.config, "text_config", None)
            or getattr(self.config, "llm_config", None)
            or self.config
        )

    def _find_base_model_parts(self):
        """Find model parts from different models and set base_{part}_path attributes."""
        base_model_parts_mapping = {
            "base_model_path": [
                "model.language_model",
                "model",
                "backbone",
                "language_model.backbone",
            ],
            "base_model_embeddings_path": [
                "model.embed_tokens",
                "backbone.embeddings",
                "language_model.backbone.embeddings",
                "model.language_model.embed_tokens",
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
        if not self.eagle_config.use_aux_hidden_state:
            return None
        # In PTQ, forward method will be called with try and except to find max batch size.
        # This leads to uncleared aux hidden states in the front of the list.
        # To fix it, we only return the last num_aux_h items in the list.
        num_aux_h = len(self.eagle_config.eagle_aux_hidden_state_layer_ids)
        aux_h_list = self._aux_hidden_states[-num_aux_h:]
        self._aux_hidden_states.clear()

        # Gather aux hidden states on the draft model device
        aux_hiddens = torch.cat(
            [h.to(self.eagle_module.fc.weight.device) for h in aux_h_list], dim=-1
        )

        return aux_hiddens

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
        eagle_decoder_type,
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
            eagle_decoder_type=eagle_decoder_type,
        )

        if eagle_decoder_type == "llama":
            # Use default eagle config
            decoder_cls = LlamaDecoderLayer
        elif eagle_decoder_type == "kimik2":
            decoder_cls = _setup_kimi_k2_decoder()

        self.eagle_config = PretrainedConfig.from_dict(eagle_architecture_config)
        self.eagle_config.eagle_decoder_type = eagle_decoder_type
        # Hidden size and vocab size must match base model
        self.eagle_config.hidden_size = self._base_llm_config.hidden_size
        self.eagle_config.vocab_size = self._base_llm_config.vocab_size
        self.eagle_config.max_position_embeddings = self._base_llm_config.max_position_embeddings
        self.eagle_config.draft_vocab_size = getattr(
            self.eagle_config, "draft_vocab_size", self.eagle_config.vocab_size
        )

        if self.eagle_config._attn_implementation is None:
            self.eagle_config._attn_implementation = "sdpa"

        # Patch for Kimi-K2-Thinking, avoid quantizing drafter
        quant_config = getattr(self.config, "quantization_config", None)
        if isinstance(quant_config, CompressedTensorsConfig):
            quant_config.ignore.append("re:.*eagle_module.*")

        # Set default aux_hidden_state layers
        if (
            self.eagle_config.use_aux_hidden_state
            and len(self.eagle_config.eagle_aux_hidden_state_layer_ids) == 0
        ):
            self._set_default_aux_hidden_state_layers()

        # Freeze all parameters
        if self.eagle_freeze_base_model:
            for name, param in self.named_parameters():
                param.requires_grad = False

        self.eagle_module = EagleModule(
            self.eagle_config,
            decoder_cls,
        )

        # find base model, lm head, and embeddings paths
        self._find_base_model_parts()
        self.eagle_module.to(self._base_model.dtype).to(self._get_eagle_device())

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

        self.num_ttt_steps = 4  # NOTE: (hg) hardcoded for now. Might add to config later.
        self._cached_attn_blk_masks = {}

    def _get_ttt_attention_mask(self, batch_size, seq_length, ttt_step):
        # compile and cached flex attention masks in first call
        if ttt_step not in self._cached_attn_blk_masks:
            self._cached_attn_blk_masks.update(
                {ttt_step: self._compute_ttt_attention_mask(batch_size, seq_length, ttt_step)}
            )
        return self._cached_attn_blk_masks[ttt_step]

    def _prepare_decoder_attention_mask(
        self, attention_mask, input_shape, past_key_values_length, device, dtype
    ):
        """Expand the 2-D attention mask to 4-D and apply causal mask."""
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        # construct causal mask
        if input_shape[-1] > 1:
            combined_attention_mask = make_causal_mask(
                input_shape,
                dtype,
                device=device,
                past_key_values_length=past_key_values_length,
            )
        # merge causal mask with padding mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = expand_mask(attention_mask, dtype, tgt_len=input_shape[-1]).to(
                device
            )
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def _prepare_eagle_inputs(
        self,
        input_ids,
        attention_mask,
        position_ids,
        eagle_cache,
        base_outputs,
    ):
        """Helper function to prepare eagle inputs for the 0th eagle forward pass."""
        b, seq_length = input_ids.shape
        past_kv_len = eagle_cache.get_seq_length() if eagle_cache is not None else 0
        seq_len_with_past = seq_length + past_kv_len

        # Prepare eagle_input_embeds: Shift left 1 token
        with torch.no_grad():
            if base_outputs.input_embeds is None:
                eagle_input_embeds = self._base_model_embeddings(input_ids.roll(-1, 1))
            else:
                eagle_input_embeds = base_outputs.input_embeds.roll(-1, 1)

        # Prepare eagle_input_hiddens
        if self.eagle_config.use_aux_hidden_state:
            # Eagle3: concat base model intermediate (pre-norm) hiddens
            eagle_input_hiddens = self.eagle_module.fc(base_outputs.aux_hiddens)
        else:
            # Eagle1: use base model output (post-norm)hiddens
            eagle_input_hiddens = base_outputs.out_hiddens

        # Prepare attention_mask
        if attention_mask is None:
            eagle_attention_mask = torch.ones(  # default: all tokens are valid
                (b, seq_len_with_past), dtype=torch.bool, device=eagle_input_hiddens.device
            )
        else:
            eagle_attention_mask = attention_mask.roll(-1, 1)  # Shift left 1 token
        # Expand the 2-D attention mask to 4-D and apply causal mask.
        eagle_attention_mask = self._prepare_decoder_attention_mask(
            eagle_attention_mask,
            (b, seq_length),
            past_kv_len,
            eagle_input_hiddens.device,
            eagle_input_hiddens.dtype,
        )

        # Prepare position_ids
        if position_ids is None:
            eagle_position_ids = (
                torch.arange(
                    past_kv_len,
                    seq_len_with_past,
                    dtype=torch.long,
                    device=eagle_input_hiddens.device,
                )
                .unsqueeze(0)
                .view(-1, seq_length)
            )
        else:
            eagle_position_ids = position_ids.view(-1, seq_length).long()

        return eagle_input_embeds, eagle_input_hiddens, eagle_attention_mask, eagle_position_ids

    def _compute_ttt_attention_mask(
        self, batch_size, seq_length, ttt_step
    ) -> BlockMask | torch.Tensor:
        """Return TTT attention_mask tensor of type BlockMask or Tensor depends on eagle attn impl."""
        msk_func = get_ttt_msk_func(seq_length, ttt_step)
        dtypemin = torch.finfo(self._base_llm_config.dtype).min
        q_len = seq_length
        kv_len = seq_length * (1 + ttt_step)
        if self.eagle_config._attn_implementation == "flex_attention":
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

            # Note: (hg) repeat mask for kimi-k2 compatibility
            tensor_mask = tensor_mask.repeat(batch_size, 1, 1, 1)
            return tensor_mask

    def _base_model_forward(
        self,
        input_ids,
        attention_mask,
        position_ids=None,
        past_key_values=None,
        freeze_base_model=True,
        labels=None,
        **kwargs,
    ):
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
            base_input_embeds = outputs.hidden_states[0]
            base_model_hidden_states = outputs.hidden_states[-1]
            base_model_logits = outputs.logits

            # Optionally, compute base model loss when we want to tune the base model.
            base_model_loss = None
            if not freeze_base_model and labels is not None:  # Base model loss
                loss_fct = CrossEntropyLoss()
                loss_logits = base_model_logits.view(-1, base_model_logits.shape[-1])
                labels = labels.view(-1)
                base_model_loss = loss_fct(loss_logits, labels)

        return EagleBaseModelOutput(
            input_embeds=base_input_embeds,
            aux_hiddens=self.pop_and_gather_aux_hiddens(),
            out_hiddens=base_model_hidden_states,
            logits=base_model_logits,
            loss=base_model_loss,
        ), past_key_values

    def _map_logits_to_draft_vocab(self, full_logits):
        assert hasattr(self.eagle_module, "d2t"), "d2t buffer not initialized"
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
        eagle_cache=None,
    ):
        eagle_postnorm_h, eagle_prenorm_h, eagle_cache = self.eagle_module(
            eagle_input_hidden_states,
            inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            past_key_values=eagle_cache,
        )
        eagle_lm_head = (
            self.eagle_module.lm_head
            if hasattr(self.eagle_module, "lm_head")
            else self._base_model_lm_head
        )
        eagle_logits = eagle_lm_head(eagle_postnorm_h)

        draft_logits_list = [eagle_logits]
        if self.eagle_config.parallel_draft_step > 1:
            # Get additional draft logits from parallel draft heads
            draft_logits = self.eagle_module.parallel_draft_heads(eagle_postnorm_h)
            draft_logits_list += draft_logits

        return eagle_postnorm_h, eagle_prenorm_h, draft_logits_list, eagle_cache

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
            loss: Loss of base model or eagle model.
            logits: Base model logits.
            past_key_values: Base model past key values with eagle cache attached.
            hidden_states: Base model hidden states.
            train_acc: Drafter training accuracies.
        """
        eagle_cache = getattr(past_key_values, "eagle_cache", None)

        if self.training:
            assert past_key_values is None, "past_key_values should be None in training"

        if loss_mask is None:
            # By default, mask out padding tokens in loss computation
            loss_mask = (
                attention_mask.clone().detach()
                if attention_mask is not None
                else torch.ones_like(input_ids, dtype=torch.bool)
            )

        # ====First, run base model forward====
        if self.eagle_offline:
            # Parse base model outputs forwarded from teacher
            assert "base_model_outputs" in kwargs
            base_outputs = EagleBaseModelOutput.from_offline_dict(kwargs["base_model_outputs"])
            if base_outputs.logits is None:
                base_outputs.logits = self.lm_head(base_outputs.out_hiddens)
            past_key_values = None
        else:
            base_outputs, past_key_values = self._base_model_forward(
                input_ids,
                attention_mask,
                position_ids,
                past_key_values,
                self.eagle_freeze_base_model,
                labels,
                **kwargs,
            )

        if not isinstance(past_key_values, Cache):
            past_key_values = _get_empty_cache(self._base_llm_config)
        if not isinstance(eagle_cache, Cache):
            eagle_cache = _get_empty_cache(self.eagle_module.config)
        past_key_values.eagle_cache = eagle_cache

        # ====Prepare inputs for the first eagle forward pass====
        eagle_loss = None
        train_accs = [[] for _ in range(self.eagle_config.parallel_draft_step)]
        b, seq_length, _ = base_outputs.out_hiddens.shape
        (
            eagle_input_embeds,
            eagle_input_hiddens,
            eagle_attn_mask_0,
            eagle_position_ids,
        ) = self._prepare_eagle_inputs(
            input_ids,
            attention_mask,
            position_ids,
            eagle_cache,
            base_outputs,
        )

        # ====Run eagle forward with extra training-time-test steps====
        for ttt_step in range(self.num_ttt_steps):
            # TODO: (hg) during cp training, this mask is not used. Maybe turn it off then.
            eagle_attention_mask = (
                eagle_attn_mask_0
                if ttt_step == 0
                else self._get_ttt_attention_mask(b, seq_length, ttt_step)
            )
            with enable_cp_ttt_patch() if self.training else contextlib.nullcontext():
                _, eagle_input_hiddens, eagle_logits, eagle_cache = self._eagle_forward(
                    eagle_input_hiddens,
                    eagle_input_embeds,
                    eagle_attention_mask,
                    eagle_position_ids,
                    eagle_cache,
                )
            eagle_input_hiddens = eagle_input_hiddens.roll(1, 1)
            for i in range(self.eagle_config.parallel_draft_step):
                eagle_logit = eagle_logits[i]
                classification_loss, acc = self._eagle_loss(
                    # base model predict +1 tok, while eagle predict +2
                    # so we shift base model outputs compared to eagle outputs
                    # additionally, we mask the first n tok of eagle outputs at nth TTT step
                    base_outputs.logits[:, 1 + i + ttt_step :],
                    eagle_logit[:, ttt_step : -(1 + i)],
                    loss_mask[:, 1 + ttt_step :] if i == 0 else loss_mask[:, 1 + ttt_step : -i],
                )
                # Apply loss decay factor to focus on early steps
                classification_loss *= self.eagle_loss_decay_factor ** (ttt_step + i)
                eagle_loss = (
                    classification_loss if eagle_loss is None else eagle_loss + classification_loss
                )
                train_accs[i].append(acc)
            if not self.training:
                break

        # Merge base model loss and eagle loss
        if base_outputs.loss is None and eagle_loss is None:
            loss = None
            assert not self.training, "At least one loss must be computed for training."
        else:
            loss = (base_outputs.loss or 0) + (eagle_loss or 0)

        return ModelOutput(
            loss=loss,
            logits=base_outputs.logits,
            past_key_values=past_key_values,
            hidden_states=base_outputs.out_hiddens,
            train_acc=train_accs,
        )

    def _eagle_loss(
        self,
        base_model_logits,
        eagle_logits,
        loss_mask,
    ):
        """Function for EAGLE loss computing."""
        if self.eagle_config.draft_vocab_size != self.eagle_config.vocab_size:
            base_model_logits = self._map_logits_to_draft_vocab(base_model_logits)
        loss_mask = loss_mask[:, : eagle_logits.shape[1], None]
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
        accuracy = correct.sum().float().div(denom)

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
            eagle_input_hidden_states = self.eagle_module.fc(self.pop_and_gather_aux_hiddens())
        else:
            eagle_input_hidden_states = base_model_hidden_states

        draft_tokens = []
        for step in range(steps):
            b, seq_length = eagle_ids.shape
            eagle_attention_mask = self._prepare_decoder_attention_mask(
                None,
                (b, seq_length),
                0,
                eagle_input_hidden_states.device,
                eagle_input_hidden_states.dtype,
            )

            # Use SDPA attention during generation for both stability and performance
            with temporary_set_config_value(self.eagle_config, "_attn_implementation", "sdpa"):
                _, eagle_prenorm_h, eagle_logits, _ = self._eagle_forward(
                    eagle_input_hidden_states,
                    self._base_model_embeddings(eagle_ids),
                    eagle_attention_mask,
                    None,
                )

            # parallel logits are only used after the last step
            if step == steps - 1 and self.eagle_config.parallel_draft_step > 1:
                parallel_logits = [
                    eagle_logits[i][:, -1:, :]
                    for i in range(1, self.eagle_config.parallel_draft_step)
                ]
            draft_token = eagle_logits[0][:, -1:, :].argmax(dim=-1)
            if self.eagle_config.draft_vocab_size != self.eagle_config.vocab_size:
                draft_token += self.eagle_module.d2t[draft_token]
            draft_tokens.append(draft_token)

            eagle_ids = torch.cat((eagle_ids, draft_token.to(eagle_ids.device)), dim=-1)
            eagle_input_hidden_states = torch.cat(
                (eagle_input_hidden_states, eagle_prenorm_h[:, -1:, :]), dim=1
            )

        draft_tokens = torch.cat(draft_tokens, dim=-1).to(base_token.device)
        if self.eagle_config.parallel_draft_step > 1:
            parallel_logits = torch.cat(parallel_logits, dim=1)
            parallel_tokens = parallel_logits.argmax(dim=-1)
            if self.eagle_config.draft_vocab_size != self.eagle_config.vocab_size:
                parallel_tokens += self.eagle_module.d2t[parallel_tokens]
            draft_tokens = torch.cat((draft_tokens, parallel_tokens), dim=-1).to(base_token.device)

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
