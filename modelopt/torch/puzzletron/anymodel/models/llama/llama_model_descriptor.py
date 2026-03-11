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

"""Llama model descriptor for AnyModel compression."""

import re
from dataclasses import dataclass, field
from typing import Dict, List

from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaRotaryEmbedding,
)

from modelopt.torch.puzzletron.anymodel.model_descriptor import (
    ModelDescriptor,
    ModelDescriptorFactory,
)
from modelopt.torch.puzzletron.anymodel.puzzformer.no_op import (
    MatchingZeros,
    Same,
    return_tuple_of_size,
)
from modelopt.torch.puzzletron.decilm.deci_lm_hf_code.block_config import BlockConfig
from modelopt.torch.puzzletron.pruning.ffn_intermediate_pruning_mixin import (
    FFNIntermediateLayerDescriptor,
)


@ModelDescriptorFactory.register_decorator("llama")
class LlamaModelDescriptor(ModelDescriptor):
    """Model descriptor for Llama models (Llama 2, Llama 3, Llama 3.1, Llama 3.2)."""

    @staticmethod
    def decoder_layer_cls():
        return LlamaDecoderLayer

    @staticmethod
    def block_config_to_layer_overrides(block_config: BlockConfig):
        return {
            "intermediate_size": block_config.ffn.intermediate_size,
            "num_key_value_heads": block_config.attention.num_key_value_heads,
        }

    @staticmethod
    def attn_no_op_post_init(decoder_layer: LlamaDecoderLayer):
        decoder_layer.input_layernorm = Same()
        decoder_layer.self_attn = return_tuple_of_size(MatchingZeros, size=2)()

    @staticmethod
    def mlp_no_op_post_init(decoder_layer: LlamaDecoderLayer):
        decoder_layer.post_attention_layernorm = Same()
        decoder_layer.mlp = MatchingZeros()

    @staticmethod
    def init_rotary_embedding(model: LlamaForCausalLM, runtime):
        model.model.rotary_emb = LlamaRotaryEmbedding(model.config, runtime.device)

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
                    r"|self_attn\.o_proj\.weight)$"
                )
                for layer_idx in range(num_layers)
            }

        layer_name_patterns.update(**build_ffn_predicates(), **build_attention_predicates())
        return layer_name_patterns


@dataclass
class LlamaFFNIntermediateLayerDescriptor(FFNIntermediateLayerDescriptor):
    """Layer descriptor for Llama FFN intermediate pruning."""

    down_proj_name: str = "mlp.down_proj"
    ffn_prefix_name: str = "model.layers.{layer_idx}.mlp"
    linear_weight_names: List[str] = field(
        default_factory=lambda: ["down_proj", "gate_proj", "up_proj"]
    )
