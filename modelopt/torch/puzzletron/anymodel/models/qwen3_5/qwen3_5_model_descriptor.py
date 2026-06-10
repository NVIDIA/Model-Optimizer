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

# mypy: ignore-errors

import re
from dataclasses import dataclass, field
from typing import Dict, List

from torch import nn
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5TextRotaryEmbedding,
    Qwen3_5VisionRotaryEmbedding,
)

from ....block_config import BlockConfig, maybe_cast_block_configs
from ....pruning.ffn_intermediate_pruning_mixin import (
    FFNIntermediateLayerDescriptor,
    FFNIntermediatePruningMixIn,
)
from ....pruning.pruning_mixin import PruningMixIn
from ....utils.dummy_modules import DummyBlock
from ...model_descriptor import ModelDescriptor, ModelDescriptorFactory
from ...puzzformer.no_op import MatchingZeros, Same, return_tuple_of_size

__all__ = [
    "Qwen3P5TextModelDescriptor",
    "Qwen3P5VLModelDescriptor",
    "Qwen3P5TextFFNIntermediateLayerDescriptor",
    "Qwen3P5VLFFNIntermediateLayerDescriptor",
]


class _Qwen3P5BaseModelDescriptor(ModelDescriptor):
    @staticmethod
    def decoder_layer_cls():
        return Qwen3_5DecoderLayer

    @staticmethod
    def pruning_mixins() -> Dict[str, PruningMixIn]:
        return {
            "ffn_intermediate": FFNIntermediatePruningMixIn(
                Qwen3P5TextFFNIntermediateLayerDescriptor()
            )
        }

    @staticmethod
    def passthrough_weight_name_predicates() -> Dict[str, re.Pattern]:
        return {"mtp": re.compile(r"^mtp.*")}

    @classmethod
    def create_dummy_block(cls, original_layer: nn.Module, block_index: int) -> nn.Module:
        dummy = DummyBlock(block_index=block_index)
        if hasattr(original_layer, "layer_type"):
            dummy.layer_type = original_layer.layer_type
        return dummy

    @staticmethod
    def block_config_to_layer_overrides(block_config: BlockConfig):
        override_kwargs = {}
        if block_config.ffn is not None:
            override_kwargs["intermediate_size"] = block_config.ffn.intermediate_size
        if (
            block_config.attention is not None
            and not block_config.attention.is_mamba
            and block_config.attention.num_key_value_heads is not None
        ):
            override_kwargs["num_key_value_heads"] = block_config.attention.num_key_value_heads
        return override_kwargs

    @staticmethod
    def attn_no_op_post_init(decoder_layer: Qwen3_5DecoderLayer):
        decoder_layer.input_layernorm = Same()
        if decoder_layer.layer_type == "linear_attention":
            decoder_layer.linear_attn = MatchingZeros()
        else:
            decoder_layer.self_attn = return_tuple_of_size(MatchingZeros, size=2)()

    @staticmethod
    def mlp_no_op_post_init(decoder_layer: Qwen3_5DecoderLayer):
        decoder_layer.post_attention_layernorm = Same()
        decoder_layer.mlp = MatchingZeros()

    @classmethod
    def set_block_configs(cls, model_config, block_configs: list[BlockConfig | dict]) -> None:
        block_configs = maybe_cast_block_configs(block_configs)
        super().set_block_configs(model_config, block_configs)
        lm_config = cls.get_language_model_config(model_config)
        lm_config.layer_types = [
            "linear_attention"
            if block_config.attention is not None and block_config.attention.is_mamba
            else "full_attention"
            for block_config in block_configs
        ]

    @classmethod
    def truncate_pattern_for_subblock(
        cls, lm_config, parent_layer_index: int | None = None
    ) -> None:
        layer_types = getattr(lm_config, "layer_types", None)
        if not layer_types:
            return super().truncate_pattern_for_subblock(lm_config, parent_layer_index)

        if parent_layer_index is not None and 0 <= parent_layer_index < len(layer_types):
            lm_config.layer_types = [layer_types[parent_layer_index]]
        else:
            lm_config.layer_types = [layer_types[0]]

    @staticmethod
    def _text_attention_pattern(prefix: str, layer_idx: int) -> re.Pattern:
        return re.compile(
            rf"^{prefix}\.{layer_idx}\.(input_layernorm\.weight"
            r"|self_attn\.q_proj\.weight"
            r"|self_attn\.k_proj\.weight"
            r"|self_attn\.v_proj\.weight"
            r"|self_attn\.o_proj\.weight"
            r"|self_attn\.q_norm\.weight"
            r"|self_attn\.k_norm\.weight"
            r"|linear_attn\.conv1d\.weight"
            r"|linear_attn\.dt_bias"
            r"|linear_attn\.A_log"
            r"|linear_attn\.norm\.weight"
            r"|linear_attn\.out_proj\.weight"
            r"|linear_attn\.in_proj_qkv\.weight"
            r"|linear_attn\.in_proj_z\.weight"
            r"|linear_attn\.in_proj_b\.weight"
            r"|linear_attn\.in_proj_a\.weight)$"
        )

    @staticmethod
    def _text_ffn_pattern(prefix: str, layer_idx: int) -> re.Pattern:
        return re.compile(
            rf"^{prefix}\.{layer_idx}\.(post_attention_layernorm\.weight"
            r"|mlp\.up_proj\.weight"
            r"|mlp\.gate_proj\.weight"
            r"|mlp\.down_proj\.weight)$"
        )


@ModelDescriptorFactory.register_decorator("qwen3_6_text")
@ModelDescriptorFactory.register_decorator("qwen3_5_text")
class Qwen3P5TextModelDescriptor(_Qwen3P5BaseModelDescriptor):
    @staticmethod
    def pruning_mixins() -> Dict[str, PruningMixIn]:
        return {
            "ffn_intermediate": FFNIntermediatePruningMixIn(
                Qwen3P5TextFFNIntermediateLayerDescriptor()
            )
        }

    @staticmethod
    def init_rotary_embedding(model, runtime):
        model.model.rotary_emb = Qwen3_5TextRotaryEmbedding(config=model.config).to(
            device=runtime.device, dtype=runtime.dtype
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

    @staticmethod
    def layer_name_predicates(num_layers: int) -> Dict[str, re.Pattern]:
        layer_name_patterns = {
            "embeddings": re.compile(r"^model\.embed_tokens\.weight$"),
            "lm_head": re.compile(r"^(model\.norm\.weight|lm_head\.weight)$"),
        }
        layer_name_patterns.update(
            **{
                f"block_{layer_idx}_ffn": _Qwen3P5BaseModelDescriptor._text_ffn_pattern(
                    "model\\.layers", layer_idx
                )
                for layer_idx in range(num_layers)
            },
            **{
                f"block_{layer_idx}_attention": _Qwen3P5BaseModelDescriptor._text_attention_pattern(
                    "model\\.layers", layer_idx
                )
                for layer_idx in range(num_layers)
            },
        )
        return layer_name_patterns


@ModelDescriptorFactory.register_decorator("qwen3_6")
@ModelDescriptorFactory.register_decorator("qwen3_5")
class Qwen3P5VLModelDescriptor(_Qwen3P5BaseModelDescriptor):
    @staticmethod
    def get_language_model_config(config):
        return config.text_config if hasattr(config, "text_config") else config

    @staticmethod
    def pruning_mixins() -> Dict[str, PruningMixIn]:
        return {
            "ffn_intermediate": FFNIntermediatePruningMixIn(
                Qwen3P5VLFFNIntermediateLayerDescriptor()
            )
        }

    @staticmethod
    def init_rotary_embedding(model, runtime):
        text_config = Qwen3P5VLModelDescriptor.get_language_model_config(model.config)
        model.model.language_model.rotary_emb = Qwen3_5TextRotaryEmbedding(config=text_config).to(
            device=runtime.device, dtype=runtime.dtype
        )
        vision_config = (
            model.config.vision_config if hasattr(model.config, "vision_config") else None
        )
        if vision_config is not None:
            head_dim = vision_config.hidden_size // vision_config.num_heads
            model.model.visual.rotary_pos_emb = Qwen3_5VisionRotaryEmbedding(head_dim // 2).to(
                device=runtime.device, dtype=runtime.dtype
            )

    @staticmethod
    def input_embedding_name():
        return "model.language_model.embed_tokens"

    @staticmethod
    def output_embedding_name():
        return "lm_head"

    @staticmethod
    def final_norm_name():
        return "model.language_model.norm"

    @staticmethod
    def layer_block_name(index: int):
        return f"model.language_model.layers.{index}"

    @staticmethod
    def layer_name_predicates(num_layers: int) -> Dict[str, re.Pattern]:
        layer_name_patterns = {
            "embeddings": re.compile(r"^model\.language_model\.embed_tokens\.weight$"),
            "lm_head": re.compile(r"^(model\.language_model\.norm\.weight|lm_head\.weight)$"),
            "vision_encoding": re.compile(r"^model\.visual\..*"),
        }
        layer_name_patterns.update(
            **{
                f"block_{layer_idx}_ffn": _Qwen3P5BaseModelDescriptor._text_ffn_pattern(
                    "model\\.language_model\\.layers", layer_idx
                )
                for layer_idx in range(num_layers)
            },
            **{
                f"block_{layer_idx}_attention": _Qwen3P5BaseModelDescriptor._text_attention_pattern(
                    "model\\.language_model\\.layers", layer_idx
                )
                for layer_idx in range(num_layers)
            },
        )
        return layer_name_patterns


@dataclass
class Qwen3P5TextFFNIntermediateLayerDescriptor(FFNIntermediateLayerDescriptor):
    down_proj_name: str = "mlp.down_proj"
    ffn_prefix_name: str = "model.layers.{layer_idx}.mlp"
    linear_weight_names: List[str] = field(
        default_factory=lambda: ["down_proj", "gate_proj", "up_proj"]
    )


@dataclass
class Qwen3P5VLFFNIntermediateLayerDescriptor(FFNIntermediateLayerDescriptor):
    down_proj_name: str = "mlp.down_proj"
    ffn_prefix_name: str = "model.language_model.layers.{layer_idx}.mlp"
    linear_weight_names: List[str] = field(
        default_factory=lambda: ["down_proj", "gate_proj", "up_proj"]
    )
