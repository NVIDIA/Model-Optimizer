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
from typing import Any, Dict, List, Optional, Type

import torch
from torch import nn
from transformers import PretrainedConfig
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5ForCausalLM,
    Qwen3_5TextRotaryEmbedding,
    Qwen3_5VisionRotaryEmbedding,
)

from ....block_config import BlockConfig, maybe_cast_block_configs
from ....pruning.ffn_intermediate_pruning_mixin import (
    FFNIntermediateLayerDescriptor,
    FFNIntermediatePruningMixIn,
)
from ....pruning.kv_heads_pruning_mixin import KVHeadsLayerDescriptor, KVHeadsPruningMixIn
from ....pruning.pruning_mixin import PruningMixIn
from ....pruning.pruning_utils import (
    GQAInitMode,
    _init_attention_biases,
    _init_attention_weights,
    _lm_head_dim,
)
from ....utils.dummy_modules import DummyBlock
from ...model_descriptor import ModelDescriptor, ModelDescriptorFactory
from ...puzzformer import deci_x_patcher
from ...puzzformer.no_op import MatchingZeros, Same, return_tuple_of_size

__all__ = [
    "Qwen3P5TextModelDescriptor",
    "Qwen3P5VLModelDescriptor",
    "Qwen3P5TextFFNIntermediateLayerDescriptor",
    "Qwen3P5VLFFNIntermediateLayerDescriptor",
    "Qwen3P5TextKVHeadsLayerDescriptor",
    "Qwen3P5VLKVHeadsLayerDescriptor",
    "Qwen3P5KVHeadsPruningMixIn",
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
            ),
            "kv_heads": Qwen3P5KVHeadsPruningMixIn(Qwen3P5TextKVHeadsLayerDescriptor()),
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

    @classmethod
    def runtime_benchmark_config_fields(cls, lm_config) -> dict[str, Any]:
        head_dim = (
            getattr(lm_config, "head_dim", None)
            or lm_config.hidden_size // lm_config.num_attention_heads
        )
        return {
            "head_dim": head_dim,
            "hidden_act": getattr(lm_config, "hidden_act", "silu"),
            "intermediate_size": 256,
            "linear_conv_kernel_dim": getattr(lm_config, "linear_conv_kernel_dim", 4),
            "linear_key_head_dim": getattr(lm_config, "linear_key_head_dim", head_dim),
            "linear_num_key_heads": getattr(
                lm_config, "linear_num_key_heads", lm_config.num_key_value_heads
            ),
            "linear_num_value_heads": getattr(
                lm_config, "linear_num_value_heads", lm_config.num_attention_heads
            ),
            "linear_value_head_dim": getattr(lm_config, "linear_value_head_dim", head_dim),
            "rms_norm_eps": getattr(lm_config, "rms_norm_eps", 1e-6),
            "tie_word_embeddings": getattr(lm_config, "tie_word_embeddings", False),
        }

    @classmethod
    def create_runtime_benchmark_model(cls, runtime_config, block_configs: list[BlockConfig]):
        model_config = Qwen3_5TextConfig(
            max_position_embeddings=runtime_config.prefill_seq_len
            + runtime_config.generation_seq_len,
            vocab_size=runtime_config.vocab_size,
            hidden_size=runtime_config.hidden_size,
            intermediate_size=runtime_config.model_config_value("intermediate_size", 256),
            num_attention_heads=runtime_config.num_attention_heads,
            num_key_value_heads=runtime_config.num_key_value_heads,
            num_hidden_layers=len(block_configs),
            head_dim=runtime_config.model_config_value("head_dim"),
            hidden_act=runtime_config.model_config_value("hidden_act", "silu"),
            linear_conv_kernel_dim=runtime_config.model_config_value("linear_conv_kernel_dim", 4),
            linear_key_head_dim=runtime_config.model_config_value("linear_key_head_dim"),
            linear_num_key_heads=runtime_config.model_config_value("linear_num_key_heads"),
            linear_num_value_heads=runtime_config.model_config_value("linear_num_value_heads"),
            linear_value_head_dim=runtime_config.model_config_value("linear_value_head_dim"),
            rms_norm_eps=runtime_config.model_config_value("rms_norm_eps", 1e-6),
            tie_word_embeddings=runtime_config.model_config_value("tie_word_embeddings", False),
        )

        cls.set_block_configs(model_config, block_configs)
        with deci_x_patcher(cls, block_configs):
            model = Qwen3_5ForCausalLM(model_config)

        model.config.block_configs = [block_config.to_dict() for block_config in block_configs]
        model.config.architectures = ["AnyModel"]
        model.config.base_architecture = "Qwen3_5ForCausalLM"
        return model

    @classmethod
    def update_runtime_benchmark_config(cls, config_data: dict[str, Any]) -> None:
        if config_data.get("model_type") in {"qwen3_5_text", "qwen3_6_text"}:
            config_data["model_type"] = "qwen3"

    @classmethod
    def runtime_vllm_benchmark_args(cls, config: dict[str, Any]) -> list[str]:
        text_config = config.get("text_config", config)
        layer_types = text_config.get("layer_types", [])
        if "linear_attention" in layer_types:
            return ["--mamba-cache-mode", "align"]
        return []


@ModelDescriptorFactory.register_decorator("qwen3_6_text")
@ModelDescriptorFactory.register_decorator("qwen3_5_text")
class Qwen3P5TextModelDescriptor(_Qwen3P5BaseModelDescriptor):
    @staticmethod
    def pruning_mixins() -> Dict[str, PruningMixIn]:
        return {
            "ffn_intermediate": FFNIntermediatePruningMixIn(
                Qwen3P5TextFFNIntermediateLayerDescriptor()
            ),
            "kv_heads": Qwen3P5KVHeadsPruningMixIn(Qwen3P5TextKVHeadsLayerDescriptor()),
        }

    @classmethod
    def runtime_benchmark_config_fields(cls, lm_config) -> dict[str, Any]:
        head_dim = (
            getattr(lm_config, "head_dim", None)
            or lm_config.hidden_size // lm_config.num_attention_heads
        )
        return {
            "head_dim": head_dim,
            "hidden_act": getattr(lm_config, "hidden_act", "silu"),
            "intermediate_size": 256,
            "linear_conv_kernel_dim": getattr(lm_config, "linear_conv_kernel_dim", 4),
            "linear_key_head_dim": getattr(lm_config, "linear_key_head_dim", head_dim),
            "linear_num_key_heads": getattr(
                lm_config, "linear_num_key_heads", lm_config.num_key_value_heads
            ),
            "linear_num_value_heads": getattr(
                lm_config, "linear_num_value_heads", lm_config.num_attention_heads
            ),
            "linear_value_head_dim": getattr(lm_config, "linear_value_head_dim", head_dim),
            "rms_norm_eps": getattr(lm_config, "rms_norm_eps", 1e-6),
            "tie_word_embeddings": getattr(lm_config, "tie_word_embeddings", False),
        }

    @classmethod
    def create_runtime_benchmark_model(cls, runtime_config, block_configs: list[BlockConfig]):
        model_config = Qwen3_5TextConfig(
            max_position_embeddings=runtime_config.prefill_seq_len
            + runtime_config.generation_seq_len,
            vocab_size=runtime_config.vocab_size,
            hidden_size=runtime_config.hidden_size,
            intermediate_size=runtime_config.model_config_value("intermediate_size", 256),
            num_attention_heads=runtime_config.num_attention_heads,
            num_key_value_heads=runtime_config.num_key_value_heads,
            num_hidden_layers=len(block_configs),
            head_dim=runtime_config.model_config_value("head_dim"),
            hidden_act=runtime_config.model_config_value("hidden_act", "silu"),
            linear_conv_kernel_dim=runtime_config.model_config_value("linear_conv_kernel_dim", 4),
            linear_key_head_dim=runtime_config.model_config_value("linear_key_head_dim"),
            linear_num_key_heads=runtime_config.model_config_value("linear_num_key_heads"),
            linear_num_value_heads=runtime_config.model_config_value("linear_num_value_heads"),
            linear_value_head_dim=runtime_config.model_config_value("linear_value_head_dim"),
            rms_norm_eps=runtime_config.model_config_value("rms_norm_eps", 1e-6),
            tie_word_embeddings=runtime_config.model_config_value("tie_word_embeddings", False),
        )

        cls.set_block_configs(model_config, block_configs)
        with deci_x_patcher(cls, block_configs):
            model = Qwen3_5ForCausalLM(model_config)

        model.config.block_configs = [block_config.to_dict() for block_config in block_configs]
        model.config.architectures = ["AnyModel"]
        model.config.base_architecture = "Qwen3_5ForCausalLM"
        return model

    @classmethod
    def runtime_vllm_benchmark_args(cls, config: dict[str, Any]) -> list[str]:
        text_config = config.get("text_config", config)
        layer_types = text_config.get("layer_types", [])
        if "linear_attention" in layer_types:
            return ["--mamba-cache-mode", "align"]
        return []

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

    @classmethod
    def runtime_benchmark_export_descriptor(cls) -> Type[ModelDescriptor]:
        return Qwen3P5TextModelDescriptor

    @staticmethod
    def pruning_mixins() -> Dict[str, PruningMixIn]:
        return {
            "ffn_intermediate": FFNIntermediatePruningMixIn(
                Qwen3P5VLFFNIntermediateLayerDescriptor()
            ),
            "kv_heads": Qwen3P5KVHeadsPruningMixIn(Qwen3P5VLKVHeadsLayerDescriptor()),
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


@dataclass
class Qwen3P5TextKVHeadsLayerDescriptor(KVHeadsLayerDescriptor):
    o_proj_name: str = "self_attn.o_proj"
    attn_prefix_name: str = "model.layers.{layer_idx}.self_attn"
    qkvo_weight_names: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )


@dataclass
class Qwen3P5VLKVHeadsLayerDescriptor(KVHeadsLayerDescriptor):
    o_proj_name: str = "self_attn.o_proj"
    attn_prefix_name: str = "model.language_model.layers.{layer_idx}.self_attn"
    qkvo_weight_names: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )


def _split_qwen3p5_q_proj(
    tensor: torch.Tensor, num_q_heads: int, head_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    q_proj = tensor.reshape(num_q_heads, 2 * head_size, *tensor.shape[1:])
    query = q_proj[:, :head_size].reshape(num_q_heads * head_size, *tensor.shape[1:])
    gate = q_proj[:, head_size:].reshape(num_q_heads * head_size, *tensor.shape[1:])
    return query, gate


def _merge_qwen3p5_q_proj(
    query: torch.Tensor, gate: torch.Tensor, num_q_heads: int, head_size: int
) -> torch.Tensor:
    trailing_shape = query.shape[1:]
    query = query.reshape(num_q_heads, head_size, *trailing_shape)
    gate = gate.reshape(num_q_heads, head_size, *trailing_shape)
    return torch.cat([query, gate], dim=1).reshape(num_q_heads * 2 * head_size, *trailing_shape)


def _state_dict_with_tensor(
    state_dict: dict[str, torch.Tensor], key: str, value: torch.Tensor
) -> dict[str, torch.Tensor]:
    patched_state_dict = dict(state_dict)
    patched_state_dict[key] = value
    return patched_state_dict


class Qwen3P5KVHeadsPruningMixIn(KVHeadsPruningMixIn):
    """KV-head pruning for Qwen3.5 gated full-attention layers."""

    def __init__(self, layer_descriptor: KVHeadsLayerDescriptor):
        assert isinstance(layer_descriptor, KVHeadsLayerDescriptor)
        super().__init__(layer_descriptor)

    def _init_gated_attention_weights(
        self,
        *,
        layer_idx: int,
        parent_state_dict: dict,
        new_state_dict: dict,
        original_config: PretrainedConfig,
        new_config: PretrainedConfig,
        descriptor: Type[_Qwen3P5BaseModelDescriptor],
        q_key: str,
        k_key: str,
        v_key: str,
        o_key: str,
        gqa_init_mode: GQAInitMode,
        mlp_init_config: Optional[dict[str, Any]],
        is_original_mha: bool,
        head_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_q_heads = descriptor.get_language_model_config(new_config).num_attention_heads
        orig_query, orig_gate = _split_qwen3p5_q_proj(
            parent_state_dict[q_key], num_q_heads, head_size
        )
        new_query, new_gate = _split_qwen3p5_q_proj(new_state_dict[q_key], num_q_heads, head_size)

        query_state_dict = _state_dict_with_tensor(parent_state_dict, q_key, orig_query)
        query_new_state_dict = _state_dict_with_tensor(new_state_dict, q_key, new_query)
        wq, wk, wv, wo = _init_attention_weights(
            gqa_init_mode=gqa_init_mode,
            layer_idx=layer_idx,
            new_state_dict=query_new_state_dict,
            new_config=new_config,
            descriptor=descriptor,
            original_state_dict=query_state_dict,
            original_config=original_config,
            q_key=q_key,
            k_key=k_key,
            v_key=v_key,
            o_key=o_key,
            is_original_mha=is_original_mha,
            head_size=head_size,
            mlp_init_config=mlp_init_config,
        )

        gate_state_dict = _state_dict_with_tensor(parent_state_dict, q_key, orig_gate)
        gate_new_state_dict = _state_dict_with_tensor(new_state_dict, q_key, new_gate)
        wg, _, _, _ = _init_attention_weights(
            gqa_init_mode=gqa_init_mode,
            layer_idx=layer_idx,
            new_state_dict=gate_new_state_dict,
            new_config=new_config,
            descriptor=descriptor,
            original_state_dict=gate_state_dict,
            original_config=original_config,
            q_key=q_key,
            k_key=k_key,
            v_key=v_key,
            o_key=o_key,
            is_original_mha=is_original_mha,
            head_size=head_size,
            mlp_init_config=mlp_init_config,
        )
        wq = _merge_qwen3p5_q_proj(wq, wg, num_q_heads, head_size)
        return wq, wk, wv, wo

    def _init_gated_attention_biases(
        self,
        *,
        layer_idx: int,
        parent_state_dict: dict,
        new_state_dict: dict,
        original_config: PretrainedConfig,
        new_config: PretrainedConfig,
        descriptor: Type[_Qwen3P5BaseModelDescriptor],
        q_key: str,
        k_key: str,
        v_key: str,
        o_key: str,
        gqa_init_mode: GQAInitMode,
        mlp_init_config: Optional[dict[str, Any]],
        is_original_mha: bool,
        head_size: int,
    ) -> dict[str, torch.Tensor]:
        num_q_heads = descriptor.get_language_model_config(new_config).num_attention_heads
        orig_query, orig_gate = _split_qwen3p5_q_proj(
            parent_state_dict[q_key], num_q_heads, head_size
        )
        new_query, new_gate = _split_qwen3p5_q_proj(new_state_dict[q_key], num_q_heads, head_size)

        query_state_dict = _state_dict_with_tensor(parent_state_dict, q_key, orig_query)
        query_new_state_dict = _state_dict_with_tensor(new_state_dict, q_key, new_query)
        bias_sd = _init_attention_biases(
            gqa_init_mode=gqa_init_mode,
            layer_idx=layer_idx,
            new_state_dict=query_new_state_dict,
            new_config=new_config,
            descriptor=descriptor,
            original_state_dict=query_state_dict,
            original_config=original_config,
            q_key=q_key,
            k_key=k_key,
            v_key=v_key,
            o_key=o_key,
            is_original_mha=is_original_mha,
            head_size=head_size,
            mlp_init_config=mlp_init_config,
        )

        gate_state_dict = _state_dict_with_tensor(parent_state_dict, q_key, orig_gate)
        gate_new_state_dict = _state_dict_with_tensor(new_state_dict, q_key, new_gate)
        gate_bias_sd = _init_attention_biases(
            gqa_init_mode=gqa_init_mode,
            layer_idx=layer_idx,
            new_state_dict=gate_new_state_dict,
            new_config=new_config,
            descriptor=descriptor,
            original_state_dict=gate_state_dict,
            original_config=original_config,
            q_key=q_key,
            k_key=k_key,
            v_key=v_key,
            o_key=o_key,
            is_original_mha=is_original_mha,
            head_size=head_size,
            mlp_init_config=mlp_init_config,
        )
        if "q" in bias_sd:
            bias_sd["q"] = _merge_qwen3p5_q_proj(
                bias_sd["q"], gate_bias_sd["q"], num_q_heads, head_size
            )
        return bias_sd

    def prune_single_layer(
        self,
        layer_idx: int,
        parent_state_dict: dict,
        new_state_dict: dict,
        original_config: PretrainedConfig,
        new_config: PretrainedConfig,
        descriptor,
        gqa_init_mode: GQAInitMode,
        mlp_init_config: Optional[dict[str, Any]],
        is_original_mha: bool,
        keys: dict,
        keys_to_remove: dict,
        **kwargs,
    ):
        layer_out_state_dict = {}
        attn_prefix = self.layer_descriptor.attn_prefix(layer_idx)
        q_name, k_name, v_name, o_name = [
            f"{attn_prefix}.{proj_name}" for proj_name in self.layer_descriptor.qkvo_weight_names
        ]

        head_size = _lm_head_dim(new_config, descriptor)
        for part in ["weight", "bias"]:
            attn_keys = [f"{name}.{part}" for name in [q_name, k_name, v_name, o_name]]
            q_key, k_key, v_key, o_key = attn_keys
            attn_keys = [key for key in attn_keys if key in new_state_dict]
            if not attn_keys or not all(key in keys for key in attn_keys):
                continue

            for key in attn_keys:
                keys_to_remove[key] = keys[key]

            if not all(key in new_state_dict for key in attn_keys):
                continue

            if q_key not in new_state_dict:
                continue

            if part == "weight":
                wq, wk, wv, wo = self._init_gated_attention_weights(
                    layer_idx=layer_idx,
                    parent_state_dict=parent_state_dict,
                    new_state_dict=new_state_dict,
                    original_config=original_config,
                    new_config=new_config,
                    descriptor=descriptor,
                    q_key=q_key,
                    k_key=k_key,
                    v_key=v_key,
                    o_key=o_key,
                    gqa_init_mode=gqa_init_mode,
                    mlp_init_config=mlp_init_config,
                    is_original_mha=is_original_mha,
                    head_size=head_size,
                )
                layer_out_state_dict[q_key], layer_out_state_dict[k_key] = wq, wk
                layer_out_state_dict[v_key], layer_out_state_dict[o_key] = wv, wo
            else:
                bias_sd = self._init_gated_attention_biases(
                    layer_idx=layer_idx,
                    parent_state_dict=parent_state_dict,
                    new_state_dict=new_state_dict,
                    original_config=original_config,
                    new_config=new_config,
                    descriptor=descriptor,
                    q_key=q_key,
                    k_key=k_key,
                    v_key=v_key,
                    o_key=o_key,
                    gqa_init_mode=gqa_init_mode,
                    mlp_init_config=mlp_init_config,
                    is_original_mha=is_original_mha,
                    head_size=head_size,
                )
                for bias_key, sd_key in zip("qkvo", [q_key, k_key, v_key, o_key]):
                    if bias_key in bias_sd:
                        layer_out_state_dict[sd_key] = bias_sd[bias_key]

        return layer_out_state_dict
