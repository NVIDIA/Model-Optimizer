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

from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeTextDecoderLayer,
    Qwen3VLMoeTextRotaryEmbedding,
    Qwen3VLMoeVisionRotaryEmbedding,
)

from ....block_config import BlockConfig
from ....pruning.expert_removal_pruning_mixin import ExpertRemovalLayerDescriptor
from ....pruning.ffn_intermediate_pruning_mixin import FFNIntermediateLayerDescriptor
from ....pruning.kv_heads_pruning_mixin import KVHeadsLayerDescriptor
from ...model_descriptor import ModelDescriptor, ModelDescriptorFactory
from ...puzzformer.no_op import MatchingZeros, Same, return_tuple_of_size

__all__ = [
    "Qwen3VLModelDescriptor",
    "Qwen3VLFFNIntermediateLayerDescriptor",
    "Qwen3VLKVHeadsLayerDescriptor",
    "Qwen3VLExpertRemovalLayerDescriptor",
]


@ModelDescriptorFactory.register_decorator("qwen3_vl")
class Qwen3VLModelDescriptor(ModelDescriptor):
    @staticmethod
    def uses_autocast() -> bool:
        """
        Qwen3-VL MoE has a dtype bug in HuggingFace transformers under torch.autocast:
        scatter() in MoE routing fails with dtype mismatch. Use native bfloat16 instead.
        See: https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct (recommended approach)
        """
        return False

    @staticmethod
    def get_language_model_config(config):
        """Qwen3-VL has nested text_config for language model parameters."""
        return config.text_config if hasattr(config, "text_config") else config

    @staticmethod
    def decoder_layer_cls():
        return Qwen3VLMoeTextDecoderLayer

    @staticmethod
    def block_config_to_layer_overrides(block_config: BlockConfig):
        override_kwargs = {"num_key_value_heads": block_config.attention.num_key_value_heads}

        if block_config.ffn.moe:
            override_kwargs["moe_intermediate_size"] = block_config.ffn.moe.expert_intermediate_dim
            override_kwargs["num_experts"] = block_config.ffn.moe.num_local_experts
        else:
            override_kwargs["intermediate_size"] = block_config.ffn.intermediate_size

        return override_kwargs

    @staticmethod
    def attn_no_op_post_init(decoder_layer: Qwen3VLMoeTextDecoderLayer):
        decoder_layer.input_layernorm = Same()
        decoder_layer.self_attn = return_tuple_of_size(MatchingZeros, size=2)()

    @staticmethod
    def mlp_no_op_post_init(decoder_layer: Qwen3VLMoeTextDecoderLayer):
        decoder_layer.post_attention_layernorm = Same()
        decoder_layer.mlp = MatchingZeros()

    @staticmethod
    def init_rotary_embedding(model, runtime):
        # Re-initialize text rotary embedding on correct device and dtype
        text_config = Qwen3VLModelDescriptor.get_language_model_config(model.config)
        model.model.language_model.rotary_emb = Qwen3VLMoeTextRotaryEmbedding(
            config=text_config
        ).to(device=runtime.device, dtype=runtime.dtype)
        # Re-initialize vision rotary embedding on correct device and dtype
        vision_config = (
            model.config.vision_config if hasattr(model.config, "vision_config") else None
        )
        if vision_config is not None:
            head_dim = vision_config.hidden_size // vision_config.num_heads
            model.model.visual.rotary_pos_emb = Qwen3VLMoeVisionRotaryEmbedding(head_dim // 2).to(
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
        # Qwen3-VL has text model under model.language_model.* prefix
        layer_name_patterns = {
            "embeddings": re.compile(r"^model\.language_model\.embed_tokens\.weight$"),
            "lm_head": re.compile(r"^(model\.language_model\.norm\.weight|lm_head\.weight)$"),
            # Vision encoder (includes merger under model.visual.deepstack_merger_list.*)
            "vision_encoding": re.compile(r"^model\.visual\..*"),
        }

        def build_ffn_predicates() -> Dict[str, re.Pattern]:
            return {
                f"block_{layer_idx}_ffn": re.compile(
                    rf"^model\.language_model\.layers\.{layer_idx}\.(post_attention_layernorm\.weight"
                    # MoE router
                    r"|mlp\.gate\.weight"
                    # MoE experts - fused format (gate_up_proj, down_proj without .weight suffix)
                    r"|mlp\.experts\.gate_up_proj"
                    r"|mlp\.experts\.down_proj"
                    # Shared expert (if present)
                    r"|mlp\.shared_expert\.up_proj\.weight"
                    r"|mlp\.shared_expert\.gate_proj\.weight"
                    r"|mlp\.shared_expert\.down_proj\.weight"
                    r"|mlp\.shared_expert_gate\.weight"
                    # Dense MLP fallback (for non-MoE layers)
                    r"|mlp\.up_proj\.weight"
                    r"|mlp\.gate_proj\.weight"
                    r"|mlp\.down_proj\.weight)$"
                )
                for layer_idx in range(num_layers)
            }

        def build_attention_predicates() -> Dict[str, re.Pattern]:
            return {
                f"block_{layer_idx}_attention": re.compile(
                    rf"^model\.language_model\.layers\.{layer_idx}\.(input_layernorm\.weight"
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
class Qwen3VLFFNIntermediateLayerDescriptor(FFNIntermediateLayerDescriptor):
    down_proj_name: str = "mlp.down_proj"
    ffn_prefix_name: str = "model.language_model.layers.{layer_idx}.mlp"
    linear_weight_names: List[str] = field(
        default_factory=lambda: ["down_proj", "gate_proj", "up_proj"]
    )


@dataclass
class Qwen3VLKVHeadsLayerDescriptor(KVHeadsLayerDescriptor):
    o_proj_name: str = "self_attn.o_proj"
    attn_prefix_name: str = "model.language_model.layers.{layer_idx}.self_attn"
    qkvo_weight_names: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )


@dataclass
class Qwen3VLExpertRemovalLayerDescriptor(ExpertRemovalLayerDescriptor):
    """
    Qwen3-VL MoE layer descriptor.

    Reference: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl_moe/modeling_qwen3_vl_moe.py
    - Qwen3VLMoeTextSparseMoeBlock: MoE block with .gate (router) and .experts
    - Qwen3VLMoeTextTopKRouter: Router with .weight (no bias)
    - Qwen3VLMoeTextExperts: Fused experts with .gate_up_proj and .down_proj tensors
    """

    target_name: str = "mlp"
    moe_prefix_name: str = "model.language_model.layers.{layer_idx}.mlp"
    # Router: Qwen3VLMoeTextTopKRouter has self.weight, no bias
    router_weights: List[str] = field(default_factory=lambda: ["gate.weight"])
    router_biases: List[str] = field(default_factory=list)
    # Fused expert format: Qwen3VLMoeTextExperts stores all experts in single tensors
    # with shape [num_experts, ...] instead of separate tensors per expert.
    is_fused_experts: bool = True
    fused_expert_weights: List[str] = field(
        default_factory=lambda: ["experts.gate_up_proj", "experts.down_proj"]
    )
