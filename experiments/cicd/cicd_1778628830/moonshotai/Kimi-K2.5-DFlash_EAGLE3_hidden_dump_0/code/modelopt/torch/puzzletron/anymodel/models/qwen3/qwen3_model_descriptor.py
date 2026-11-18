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
from typing import Dict, List

from torch import nn
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer,
    Qwen3ForCausalLM,
    Qwen3RotaryEmbedding,
)

from ....block_config import BlockConfig
from ....pruning.ffn_intermediate_pruning_mixin import FFNIntermediateLayerDescriptor
from ....pruning.kv_heads_pruning_mixin import KVHeadsLayerDescriptor
from ....utils.dummy_modules import DummyBlock
from ...model_descriptor import ModelDescriptor, ModelDescriptorFactory
from ...puzzformer.no_op import MatchingZeros, Same, return_tuple_of_size

__all__ = [
    "Qwen3ModelDescriptor",
    "Qwen3FFNIntermediateLayerDescriptor",
    "Qwen3KVHeadsLayerDescriptor",
]


@ModelDescriptorFactory.register_decorator("qwen3")
class Qwen3ModelDescriptor(ModelDescriptor):
    @staticmethod
    def decoder_layer_cls():
        return Qwen3DecoderLayer

    @classmethod
    def create_dummy_block(cls, original_layer: nn.Module, block_index: int) -> nn.Module:
        """Create a dummy block that preserves Qwen3-specific attributes like attention_type.

        Qwen3's forward pass accesses decoder_layer.attention_type for attention mask selection.
        """
        dummy = DummyBlock(block_index=block_index)
        # Copy attention_type from original layer (required by Qwen3's forward pass)
        if hasattr(original_layer, "attention_type"):
            dummy.attention_type = original_layer.attention_type
        return dummy

    @staticmethod
    def block_config_to_layer_overrides(block_config: BlockConfig):
        return {
            "intermediate_size": block_config.ffn.intermediate_size,
            "num_key_value_heads": block_config.attention.num_key_value_heads,
        }

    @staticmethod
    def attn_no_op_post_init(decoder_layer: Qwen3DecoderLayer):
        decoder_layer.input_layernorm = Same()
        decoder_layer.self_attn = return_tuple_of_size(MatchingZeros, size=2)()

    @staticmethod
    def mlp_no_op_post_init(decoder_layer: Qwen3DecoderLayer):
        decoder_layer.post_attention_layernorm = Same()
        decoder_layer.mlp = MatchingZeros()

    @staticmethod
    def init_rotary_embedding(model: Qwen3ForCausalLM, runtime):
        model.model.rotary_emb = Qwen3RotaryEmbedding(model.config, runtime.device)

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

    @staticmethod
    def layer_name_predicates(num_layers: int) -> Dict[str, re.Pattern]:
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
                    r"|self_attn\.q_proj\.weight"
                    r"|self_attn\.k_proj\.weight"
                    r"|self_attn\.v_proj\.weight"
                    r"|self_attn\.o_proj\.weight"
                    r"|self_attn\.q_norm\.weight"
                    r"|self_attn\.k_norm\.weight)$"
                )
                for layer_idx in range(num_layers)
            }

        layer_name_patterns.update(**build_ffn_predicates(), **build_attention_predicates())
        return layer_name_patterns


@dataclass
class Qwen3FFNIntermediateLayerDescriptor(FFNIntermediateLayerDescriptor):
    down_proj_name: str = "mlp.down_proj"
    ffn_prefix_name: str = "model.layers.{layer_idx}.mlp"
    linear_weight_names: List[str] = field(
        default_factory=lambda: ["down_proj", "gate_proj", "up_proj"]
    )


@dataclass
class Qwen3KVHeadsLayerDescriptor(KVHeadsLayerDescriptor):
    o_proj_name: str = "self_attn.o_proj"
    attn_prefix_name: str = "model.layers.{layer_idx}.self_attn"
    qkvo_weight_names: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
