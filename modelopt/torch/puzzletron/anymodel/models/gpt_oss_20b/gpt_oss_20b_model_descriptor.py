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

"""GPT-OSS-20B model descriptor for AnyModel compression."""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type

import torch.nn as nn
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssDecoderLayer, GptOssRotaryEmbedding

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
from modelopt.torch.puzzletron.pruning.expert_removal_pruning_mixin import (
    ExpertRemovalLayerDescriptor,
    ExpertRemovalPruningMixIn,
)

# Expert removal is supported for unquantized models (test models).
# Production models use MXFP4 quantized MoE with combined tensors
# (gate_up_proj_blocks, down_proj_blocks), which is not yet supported.
from modelopt.torch.puzzletron.pruning.pruning_mixin import PruningMixIn
from modelopt.torch.puzzletron.utils.dummy_modules import DummyBlock


@ModelDescriptorFactory.register_decorator("gpt_oss_20b")
class GptOss20bModelDescriptor(ModelDescriptor):
    """Model descriptor for GPT-OSS-20B (pure MoE model)."""

    _DECODER_LAYER_CLS: Type[nn.Module] = None

    @classmethod
    def create_dummy_block(cls, original_layer: GptOssDecoderLayer, block_index: int) -> nn.Module:
        dummy_block = DummyBlock(block_index=block_index)
        # Required by `GptOssModel.forward`.
        dummy_block.attention_type = original_layer.attention_type
        return dummy_block

    @staticmethod
    def decoder_layer_cls():
        """Get the decoder layer class for GPT-OSS models.

        GPT-OSS is a standard transformers model in recent versions.
        Import directly from transformers.models.gpt_oss.modeling_gpt_oss.
        """
        return GptOssDecoderLayer

    @staticmethod
    def block_config_to_layer_overrides(block_config: BlockConfig):
        """Map BlockConfig to layer constructor overrides."""
        override_kwargs = {}

        if block_config.attention.num_key_value_heads is not None:
            override_kwargs["num_key_value_heads"] = block_config.attention.num_key_value_heads

        if block_config.ffn.moe is not None:
            override_kwargs["moe_intermediate_size"] = block_config.ffn.moe.expert_intermediate_dim
            override_kwargs["num_local_experts"] = block_config.ffn.moe.num_local_experts
            override_kwargs["num_experts_per_tok"] = block_config.ffn.moe.num_experts_per_tok

        return override_kwargs

    @staticmethod
    def attn_no_op_post_init(decoder_layer):
        """Replace attention sublayers with no-op modules."""
        decoder_layer.input_layernorm = Same()
        decoder_layer.self_attn = return_tuple_of_size(MatchingZeros, size=2)()

    @staticmethod
    def mlp_no_op_post_init(decoder_layer):
        """Replace MLP sublayers with no-op modules.

        Note: GPT-OSS MoE layers return (hidden_states, router_scores), so we need
        to return a tuple of 2 values.
        """
        decoder_layer.post_attention_layernorm = Same()
        decoder_layer.mlp = return_tuple_of_size(MatchingZeros, size=2)()

    @staticmethod
    def init_rotary_embedding(model, runtime):
        """Initialize rotary embeddings on the correct device."""
        # GPT-OSS uses RoPE with YARN scaling

        model.model.rotary_emb = GptOssRotaryEmbedding(
            config=model.config,
            device=runtime.device,
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
        """Define regex patterns for grouping weights into subblocks."""
        layer_name_patterns = {
            "embeddings": re.compile(r"^model\.embed_tokens\.weight$"),
            "lm_head": re.compile(r"^(model\.norm\.weight|lm_head\.weight)$"),
        }

        def build_ffn_predicates() -> Dict[str, re.Pattern]:
            """FFN is MoE in GPT-OSS-20B with MXFP4 quantization."""
            return {
                f"block_{layer_idx}_ffn": re.compile(
                    rf"^model\.layers\.{layer_idx}\."
                    r"(post_attention_layernorm\.weight"
                    r"|mlp\.router\.weight"
                    r"|mlp\.router\.bias"
                    r"|mlp\.experts\.(gate_up_proj|down_proj)(_(bias|blocks|scales))?)$"
                )
                for layer_idx in range(num_layers)
            }

        def build_attention_predicates() -> Dict[str, re.Pattern]:
            return {
                f"block_{layer_idx}_attention": re.compile(
                    rf"^model\.layers\.{layer_idx}\."
                    r"(input_layernorm\.weight"
                    r"|self_attn\.q_proj\.weight"
                    r"|self_attn\.q_proj\.bias"
                    r"|self_attn\.k_proj\.weight"
                    r"|self_attn\.k_proj\.bias"
                    r"|self_attn\.v_proj\.weight"
                    r"|self_attn\.v_proj\.bias"
                    r"|self_attn\.o_proj\.weight"
                    r"|self_attn\.o_proj\.bias"
                    r"|self_attn\.sinks)$"
                )
                for layer_idx in range(num_layers)
            }

        layer_name_patterns.update(
            **build_ffn_predicates(),
            **build_attention_predicates(),
        )

        return layer_name_patterns

    @staticmethod
    def pruning_mixins() -> Dict[str, PruningMixIn]:
        """Return available pruning mixins for GPT-OSS-20B.

        Note: Expert removal works for unquantized models (test models).
        Production models use MXFP4 quantization which is not yet supported.
        """
        return {
            "expert_removal": ExpertRemovalPruningMixIn(GptOss20bExpertRemovalLayerDescriptor())
        }


@dataclass
class GptOss20bExpertRemovalLayerDescriptor(ExpertRemovalLayerDescriptor):
    """
    GPT-OSS-20B MoE layer descriptor for expert removal.

    Note: This only works for unquantized models (e.g., test models).
    Production GPT-OSS models use MXFP4 quantization with fused experts
    (_blocks, _scales, _bias), which requires a different approach.

    Structure:
    - Router: mlp.router with .weight and .bias
    - Experts: mlp.experts.{idx}.{gate_up_proj,down_proj} with .weight and .bias
    """

    target_name: str = "mlp"
    moe_prefix_name: str = "model.layers.{layer_idx}.mlp"
    expert_prefix_name: str = "experts"

    # Router has both weight and bias
    router_weights: List[str] = field(default_factory=lambda: ["router.weight"])
    router_biases: List[str] = field(default_factory=lambda: ["router.bias"])

    # Fused format: experts stored as single tensors
    is_fused_experts: bool = True

    # Fused format: single tensors containing all experts (test models)
    fused_expert_weights: List[str] = field(
        default_factory=lambda: [
            "experts.gate_up_proj",
            "experts.gate_up_proj_bias",
            "experts.down_proj",
            "experts.down_proj_bias",
        ]
    )

    # Not used for fused format, but kept for compatibility
    expert_weights: List[str] = field(default_factory=lambda: ["gate_up_proj", "down_proj"])
    expert_biases: List[str] = field(
        default_factory=lambda: ["gate_up_proj_bias", "down_proj_bias"]
    )

    def get_modules_names_to_hook(self, model) -> List[Tuple[int, str]]:
        target_class_name = "GptOssTopKRouter"

        module_names_to_hook = []
        for module_name, module in model.named_modules():
            if (
                module_name.endswith(self.target_name)
                and module.__class__.__name__ == target_class_name
            ):
                module_names_to_hook.append(
                    (self.block_idx_from_module_name(module_name), module_name)
                )
        return module_names_to_hook
