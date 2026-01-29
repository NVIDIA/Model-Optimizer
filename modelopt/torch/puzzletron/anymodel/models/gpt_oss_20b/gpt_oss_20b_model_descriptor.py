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

import importlib
import inspect
import pkgutil
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type

import torch.nn as nn

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

# Note: Expert removal pruning not supported for MXFP4 quantized MoE
# Experts are stored in combined tensors (gate_up_proj_blocks, down_proj_blocks)
# rather than individual per-expert weights
from modelopt.torch.puzzletron.pruning.pruning_mixin import PruningMixIn


def get_dynamic_modules(module_cls_str: str) -> List[Type[nn.Module]]:
    """Get dynamically loaded module classes from transformers_modules."""
    import transformers_modules

    matches = []
    for finder, modname, ispkg in pkgutil.walk_packages(
        transformers_modules.__path__, transformers_modules.__name__ + "."
    ):
        module = importlib.import_module(modname)
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__name__ == module_cls_str:
                matches.append(obj)

    return matches


@ModelDescriptorFactory.register_decorator("gpt_oss_20b")
class GptOss20bModelDescriptor(ModelDescriptor):
    """Model descriptor for GPT-OSS-20B (pure MoE model)."""

    _DECODER_LAYER_CLS: Type[nn.Module] = None

    @staticmethod
    def decoder_layer_cls():
        """Get the decoder layer class for GPT-OSS models."""
        decoder_cls_list = get_dynamic_modules("GptOssDecoderLayer")
        if not decoder_cls_list:
            raise AssertionError(
                "GPT-OSS contains dynamic modules that should be cached beforehand. "
                "Make sure to load your config using `load_model_config` or manually call "
                "`force_cache_dynamic_modules(config, checkpoint_dir)`"
            )
        return decoder_cls_list

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
        """Replace MLP sublayers with no-op modules."""
        decoder_layer.post_attention_layernorm = Same()
        decoder_layer.mlp = MatchingZeros()

    @staticmethod
    def init_rotary_embedding(model, runtime):
        """Initialize rotary embeddings on the correct device."""
        # GPT-OSS uses RoPE with YARN scaling
        # Get the dynamically loaded GptOssRotaryEmbedding class
        rotary_emb_classes = get_dynamic_modules("GptOssRotaryEmbedding")
        if rotary_emb_classes:
            GptOssRotaryEmbedding = rotary_emb_classes[0]
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
                    r"|mlp\.experts\.((\d+\.)?(gate_up_proj|down_proj)(\.(weight|bias|blocks|scales))?|gate_up_proj_(bias|blocks|scales)|down_proj_(bias|blocks|scales)))$"
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

        Note: Expert removal is not supported because GPT-OSS uses MXFP4
        quantized MoE where all experts are stored in combined tensors
        (gate_up_proj_blocks, down_proj_blocks) rather than individual
        per-expert weights.
        """
        return {}
