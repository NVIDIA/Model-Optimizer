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

# mypy: ignore-errors

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List

from torch import nn
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5TextRotaryEmbedding,
)

from ....block_config import BlockConfig
from ....pruning.ffn_intermediate_pruning_mixin import FFNIntermediateLayerDescriptor
from ....pruning.kv_heads_pruning_mixin import KVHeadsLayerDescriptor
from ....utils.dummy_modules import DummyBlock
from ...model_descriptor import ModelDescriptor, ModelDescriptorFactory
from ...puzzformer.no_op import MatchingZeros, Same, return_tuple_of_size

__all__ = [
    "Qwen3_5ModelDescriptor",
    "Qwen3_5FFNIntermediateLayerDescriptor",
    "Qwen3_5KVHeadsLayerDescriptor",
]

# Weight prefixes that belong to the vision encoder and MTP head — not part of the
# text model and skipped during subblock conversion.
_NON_TEXT_PREFIXES = ("model.visual.", "mtp.")


@ModelDescriptorFactory.register_decorator("qwen3_5")
class Qwen3_5ModelDescriptor(ModelDescriptor):
    @staticmethod
    def get_language_model_config(config):
        """Qwen3.5 is a VLM; language model parameters live in the nested text_config."""
        return config.text_config if hasattr(config, "text_config") else config

    @staticmethod
    def decoder_layer_cls():
        return Qwen3_5DecoderLayer

    @classmethod
    def create_dummy_block(cls, original_layer: nn.Module, block_index: int) -> nn.Module:
        """Preserve layer_type so the model forward can select the right attention path."""
        dummy = DummyBlock(block_index=block_index)
        if hasattr(original_layer, "layer_type"):
            dummy.layer_type = original_layer.layer_type
        return dummy

    @staticmethod
    def block_config_to_layer_overrides(block_config: BlockConfig):
        return {
            "intermediate_size": block_config.ffn.intermediate_size,
            "num_key_value_heads": block_config.attention.num_key_value_heads,
        }

    @staticmethod
    def attn_no_op_post_init(decoder_layer: Qwen3_5DecoderLayer):
        """Zero out the attention sub-block, branching on the hybrid layer type.

        full_attention layers return a (hidden_states, attn_weights) tuple;
        linear_attention (GatedDeltaNet) layers return hidden_states directly.
        """
        decoder_layer.input_layernorm = Same()
        if decoder_layer.layer_type == "full_attention":
            decoder_layer.self_attn = return_tuple_of_size(MatchingZeros, size=2)()
        else:
            decoder_layer.linear_attn = MatchingZeros()

    @staticmethod
    def mlp_no_op_post_init(decoder_layer: Qwen3_5DecoderLayer):
        decoder_layer.post_attention_layernorm = Same()
        decoder_layer.mlp = MatchingZeros()

    @staticmethod
    def init_rotary_embedding(model, runtime):
        # After conversion the model is Qwen3_5ForCausalLM; text model is at model.model
        model.model.rotary_emb = Qwen3_5TextRotaryEmbedding(config=model.config).to(
            device=runtime.device
        )

    @staticmethod
    def input_embedding_name():
        return "model.embed_tokens"

    @staticmethod
    def output_embedding_name():
        return "lm_head"

    @staticmethod
    def final_norm_name():
        return "model.norm"

    @staticmethod
    def layer_block_name(index: int):
        return f"model.layers.{index}"

    @classmethod
    def get_weight_groups(
        cls, layer_names: Iterable[str], num_hidden_layers: int
    ) -> Dict[str, List[str]]:
        """Filter out vision/MTP weights before grouping.

        get_weight_groups is called from two places with different name formats:
        - convert_model_weights: original VLM checkpoint names (model.language_model.*)
        - _save_checkpoint: already-converted state dict names (model.*)

        Predicates use model.* format. When original names are detected we remap
        internally for matching but restore originals in the returned groups so
        that the param_to_file lookup in convert_model_weights still works.
        """
        _lm_prefix = "model.language_model."
        text_names = [n for n in layer_names if not n.startswith(_NON_TEXT_PREFIXES)]

        if not any(n.startswith(_lm_prefix) for n in text_names):
            # Already-converted names — pass through directly.
            return super().get_weight_groups(text_names, num_hidden_layers)

        # Original checkpoint names: remap to model.* for predicate matching,
        # then un-remap so returned groups contain the original names.
        name_map: Dict[str, str] = {}  # remapped → original
        remapped = []
        for n in text_names:
            r = "model." + n[len(_lm_prefix) :] if n.startswith(_lm_prefix) else n
            name_map[r] = n
            remapped.append(r)

        groups_remapped = super().get_weight_groups(remapped, num_hidden_layers)
        return {group: [name_map[r] for r in names] for group, names in groups_remapped.items()}

    @staticmethod
    def layer_name_predicates(num_layers: int) -> Dict[str, re.Pattern]:
        # Predicates use converted model.* names (matching Qwen3_5ForCausalLM).
        # get_weight_groups normalises original checkpoint names before matching.
        layer_name_patterns = {
            "embeddings": re.compile(r"^model\.embed_tokens\.weight$"),
            "lm_head": re.compile(r"^(model\.norm\.weight|lm_head\.weight)$"),
        }

        def build_ffn_predicates() -> Dict[str, re.Pattern]:
            return {
                f"block_{layer_idx}_ffn": re.compile(
                    rf"^model\.layers\.{layer_idx}\.(post_attention_layernorm\.weight"
                    r"|mlp\.up_proj\.weight"
                    r"|mlp\.gate_proj\.weight"
                    r"|mlp\.down_proj\.weight)$"
                )
                for layer_idx in range(num_layers)
            }

        def build_attention_predicates() -> Dict[str, re.Pattern]:
            return {
                f"block_{layer_idx}_attention": re.compile(
                    rf"^model\.layers\.{layer_idx}\.(input_layernorm\.weight"
                    # full_attention (Qwen3_5Attention) weights
                    r"|self_attn\.q_proj\.weight"
                    r"|self_attn\.k_proj\.weight"
                    r"|self_attn\.v_proj\.weight"
                    r"|self_attn\.o_proj\.weight"
                    r"|self_attn\.q_norm\.weight"
                    r"|self_attn\.k_norm\.weight"
                    # linear_attention (GatedDeltaNet) weights
                    r"|linear_attn\.in_proj_qkv\.weight"
                    r"|linear_attn\.in_proj_z\.weight"
                    r"|linear_attn\.in_proj_b\.weight"
                    r"|linear_attn\.in_proj_a\.weight"
                    r"|linear_attn\.out_proj\.weight"
                    r"|linear_attn\.conv1d\.weight"
                    r"|linear_attn\.norm\.weight"
                    r"|linear_attn\.dt_bias"
                    r"|linear_attn\.A_log)$"
                )
                for layer_idx in range(num_layers)
            }

        layer_name_patterns.update(**build_ffn_predicates(), **build_attention_predicates())
        return layer_name_patterns


@dataclass
class Qwen3_5FFNIntermediateLayerDescriptor(FFNIntermediateLayerDescriptor):
    down_proj_name: str = "mlp.down_proj"
    ffn_prefix_name: str = "model.layers.{layer_idx}.mlp"
    linear_weight_names: List[str] = field(
        default_factory=lambda: ["down_proj", "gate_proj", "up_proj"]
    )


@dataclass
class Qwen3_5KVHeadsLayerDescriptor(KVHeadsLayerDescriptor):
    o_proj_name: str = "self_attn.o_proj"
    attn_prefix_name: str = "model.layers.{layer_idx}.self_attn"
    qkvo_weight_names: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
