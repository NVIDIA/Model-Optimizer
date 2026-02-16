#!/usr/bin/env python3
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

"""
Megatron Bridge for Puzzletron Llama-based AnyModel checkpoints.

This module contains bridge classes for Llama-based AnyModel checkpoints,
allowing for architecture-specific optimizations and weight mappings.

This bridge handles conversion between Puzzletron Llama AnyModel (heterogeneous
layer architecture) and Megatron-Core GPT models.

As a user you would not use this bridge directly, but through `AutoBridge`.

Example:
    >>> from megatron.bridge.models.conversion.auto_bridge import AutoBridge
    >>> import llama_anymodel_bridge  # Register the bridge
    >>> bridge = AutoBridge.from_hf_pretrained("path/to/llama/anymodel/checkpoint")
    >>> provider = bridge.to_megatron_provider()
"""

import json
import logging

import torch
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import AutoMapping, GatedMLPMapping, QKVMapping
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.llama_nemotron.llama_nemotron_provider import (
    LlamaNemotronHeterogeneousProvider,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from transformers import LlamaForCausalLM

logger = logging.getLogger(__name__)


@MegatronModelBridge.register_bridge(source=LlamaForCausalLM, target=GPTModel, model_type="llama")
class PuzzletronLlamaAnyModelBridge(MegatronModelBridge):
    """
    Megatron Bridge for Puzzletron Llama-based AnyModel checkpoints.

    Supports:
    - Llama 2
    - Llama 3
    - Llama 3.1
    - Llama 3.2
    - Any other Llama-based model with block_configs

    Uses LlamaModelDescriptor for understanding model structure.
    """

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> GPTModelProvider:
        """Convert HuggingFace Llama AnyModel config to Megatron GPTModelProvider.

        This bridge is registered for all LlamaForCausalLM models, but only handles
        AnyModel checkpoints (those with block_configs). For regular Llama models,
        it delegates to the default LlamaBridge.

        Args:
            hf_pretrained: HuggingFace PreTrainedCausalLM containing the Llama config

        Returns:
            GPTModelProvider configured for Llama AnyModel architecture
        """
        hf_config = hf_pretrained.config

        # Check if this is an AnyModel checkpoint (has block_configs)
        is_anymodel = hasattr(hf_config, "block_configs") and hf_config.block_configs

        if not is_anymodel:
            # Not an AnyModel checkpoint - delegate to default LlamaBridge
            from megatron.bridge.models.llama.llama_bridge import LlamaBridge

            default_bridge = LlamaBridge()
            default_bridge.hf_pretrained = hf_pretrained
            return default_bridge.provider_bridge(hf_pretrained)

        # This is an AnyModel checkpoint - handle it with AnyModel-specific logic
        # Extract num_query_groups for heterogeneous models
        # For heterogeneous models, GQA may be defined in each block config
        # First try global config, then fall back to block_configs
        num_query_groups = None
        if hasattr(hf_config, "num_key_value_heads") and hf_config.num_key_value_heads is not None:
            # Standard Llama configs have num_key_value_heads in global config
            num_query_groups = hf_config.num_attention_heads // hf_config.num_key_value_heads
        elif hasattr(hf_config, "block_configs") and hf_config.block_configs:
            # Extract from block_configs if not in global config
            for block in hf_config.block_configs:
                if hasattr(block, "attention") and hasattr(block.attention, "n_heads_in_group"):
                    n_heads_in_group = block.attention.n_heads_in_group
                    if n_heads_in_group is not None:
                        num_query_groups = hf_config.num_attention_heads // n_heads_in_group
                        break

        # Extract ffn_hidden_size as a default/placeholder value
        # For heterogeneous models, this is just a placeholder; actual values come from block_configs
        ffn_hidden_size = None
        if hasattr(hf_config, "intermediate_size") and hf_config.intermediate_size is not None:
            # Standard Llama configs have intermediate_size in global config
            ffn_hidden_size = hf_config.intermediate_size
        elif hasattr(hf_config, "block_configs") and hf_config.block_configs:
            # Extract from block_configs if not in global config
            for block in hf_config.block_configs:
                ffn_config = getattr(block, "ffn", None) or getattr(block, "mlp", None)
                if ffn_config is not None and hasattr(ffn_config, "intermediate_size"):
                    if ffn_config.intermediate_size is not None:
                        ffn_hidden_size = ffn_config.intermediate_size
                        break

        # Prepare kwargs for provider creation
        provider_kwargs = {
            "num_layers": hf_config.num_hidden_layers,
            "hidden_size": hf_config.hidden_size,
            "ffn_hidden_size": ffn_hidden_size,
            "num_attention_heads": hf_config.num_attention_heads,
            "init_method_std": hf_config.initializer_range,
            "layernorm_epsilon": hf_config.rms_norm_eps,
            "num_query_groups": num_query_groups,
            "seq_length": hf_config.max_position_embeddings,
            "rotary_base": getattr(hf_config, "rope_theta", 10000.0),
            "kv_channels": getattr(hf_config, "head_dim", None),
            "gated_linear_unit": True,  # Llama uses SwiGLU
            "make_vocab_size_divisible_by": self.make_vocab_size_divisible_by(hf_config.vocab_size),
            "share_embeddings_and_output_weights": getattr(hf_config, "tie_word_embeddings", False),
            "fp16": (self.dtype_from_hf(hf_config, default=torch.float32) == torch.float16),
            "bf16": (self.dtype_from_hf(hf_config, default=torch.float32) == torch.bfloat16),
            "params_dtype": self.dtype_from_hf(hf_config, default=torch.float32),
            "generation_config": getattr(hf_pretrained, "generation_config", None),
            "vocab_size": hf_config.vocab_size,
        }

        # Handle rope scaling for Llama 3.1/3.2
        if hasattr(hf_config, "rope_scaling") and hf_config.rope_scaling:
            if hf_config.rope_scaling.get("rope_type") == "llama3":
                provider_kwargs["rope_scaling_factor"] = hf_config.rope_scaling.get("factor", 8.0)

        # Convert HF config format to MCore format for heterogeneous_layers_config_encoded_json
        # MCore's MLPConfig.build_config_from_dict() expects ffn_hidden_size,
        # but HF configs use intermediate_size. We convert intermediate_size -> ffn_hidden_size.
        hf_config_dict = json.loads(hf_config.to_json_string())
        mcore_block_configs = []

        # Calculate default num_query_groups from global config (if not per-layer)
        default_num_query_groups = num_query_groups
        if default_num_query_groups is None and hasattr(hf_config, "num_key_value_heads"):
            if hf_config.num_key_value_heads is not None:
                default_num_query_groups = (
                    hf_config.num_attention_heads // hf_config.num_key_value_heads
                )

        for block in hf_config_dict.get("block_configs", []):
            mcore_block = {}

            # Process attention config - ensure it has num_query_groups or n_heads_in_group
            # AnyModel uses num_key_value_heads, DeciLM uses n_heads_in_group
            # MCore accepts either num_query_groups (preferred) or n_heads_in_group
            if "attention" in block and block["attention"] is not None:
                attention_config = (
                    block["attention"].copy() if isinstance(block["attention"], dict) else {}
                )

                # Convert num_key_value_heads (AnyModel format) to num_query_groups (MCore format)
                # Since num_query_groups = num_key_value_heads, we can use it directly
                if (
                    "num_key_value_heads" in attention_config
                    and attention_config["num_key_value_heads"] is not None
                ):
                    # AnyModel format: num_key_value_heads is the same as num_query_groups
                    attention_config["num_query_groups"] = attention_config["num_key_value_heads"]
                    # Remove num_key_value_heads as MCore doesn't expect it
                    attention_config.pop("num_key_value_heads", None)

                # If attention config still doesn't have num_query_groups or n_heads_in_group, add it
                if (
                    "num_query_groups" not in attention_config
                    and "n_heads_in_group" not in attention_config
                ):
                    # Calculate num_query_groups from global config
                    if default_num_query_groups is not None:
                        attention_config["num_query_groups"] = default_num_query_groups
                    elif (
                        hasattr(hf_config, "num_key_value_heads")
                        and hf_config.num_key_value_heads is not None
                    ):
                        # Use global num_key_value_heads (same as num_query_groups)
                        attention_config["num_query_groups"] = hf_config.num_key_value_heads
                    else:
                        # Default to MHA (no grouping) - num_query_groups equals num_attention_heads
                        attention_config["num_query_groups"] = hf_config.num_attention_heads

                # Ensure no_op and replace_with_linear are set (default to False if not present)
                if "no_op" not in attention_config:
                    attention_config["no_op"] = False
                if "replace_with_linear" not in attention_config:
                    attention_config["replace_with_linear"] = False

                mcore_block["attention"] = attention_config

            # Convert FFN config: intermediate_size -> ffn_hidden_size
            if "ffn" in block:
                ffn_config = block["ffn"].copy()
                if (
                    "intermediate_size" in ffn_config
                    and ffn_config["intermediate_size"] is not None
                ):
                    # Convert intermediate_size to ffn_hidden_size (MCore expects this)
                    ffn_config["ffn_hidden_size"] = ffn_config.pop("intermediate_size")
                mcore_block["ffn"] = ffn_config
            elif "mlp" in block:
                # Some configs use "mlp" instead of "ffn"
                mlp_config = block["mlp"].copy()
                if (
                    "intermediate_size" in mlp_config
                    and mlp_config["intermediate_size"] is not None
                ):
                    mlp_config["ffn_hidden_size"] = mlp_config.pop("intermediate_size")
                mcore_block["ffn"] = mlp_config  # MCore expects "ffn" key

            mcore_block_configs.append(mcore_block)

        # Build MCore format JSON
        mcore_config = {"block_configs": mcore_block_configs}
        if "rope_scaling" in hf_config_dict:
            mcore_config["rope_scaling"] = hf_config_dict["rope_scaling"]

        provider_kwargs["heterogeneous_layers_config_encoded_json"] = json.dumps(
            mcore_config, ensure_ascii=False
        )

        provider = LlamaNemotronHeterogeneousProvider(**provider_kwargs)
        return provider

    @classmethod
    def megatron_to_hf_config(cls, provider: GPTModelProvider) -> dict:
        """Convert Megatron GPTModelProvider config to HuggingFace Llama config dict.

        Args:
            provider: GPTModelProvider with Llama AnyModel configuration

        Returns:
            Dictionary of HuggingFace LlamaConfig parameters
        """
        raise NotImplementedError("megatron_to_hf_config() not yet implemented for Llama")

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Define weight mappings between Llama AnyModel and Megatron formats.

        Uses standard Llama-style mappings:
        - QKV concatenation (q_proj, k_proj, v_proj -> linear_qkv)
        - GatedMLP concatenation (gate_proj, up_proj -> linear_fc1)

        AnyModel checkpoints use the same weight structure as standard Llama models,
        so the mappings are identical.

        Returns:
            MegatronMappingRegistry containing Llama weight mappings
        """
        # Base mappings (same as standard LlamaBridge)
        param_mappings = {
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "output_layer.weight": "lm_head.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
            # te implementation
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": (
                "model.layers.*.input_layernorm.weight"
            ),
            # local implementation
            "decoder.layers.*.input_layernorm.weight": "model.layers.*.input_layernorm.weight",
            # te implementation
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": (
                "model.layers.*.post_attention_layernorm.weight"
            ),
            # local implementation
            "decoder.layers.*.pre_mlp_layernorm.weight": (
                "model.layers.*.post_attention_layernorm.weight"
            ),
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
        }

        mapping_list = []
        # Convert each dictionary entry to AutoMapping(megatron_param, hf_param)
        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        # Add special mappings that require parameter concatenation/transformation
        mapping_list.extend(
            [
                # QKV: Combine separate Q, K, V matrices into single QKV matrix
                QKVMapping(
                    megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                    q="model.layers.*.self_attn.q_proj.weight",
                    k="model.layers.*.self_attn.k_proj.weight",
                    v="model.layers.*.self_attn.v_proj.weight",
                ),
                # Gated MLP: Combine gate and up projection matrices into single FC1 matrix
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                    gate="model.layers.*.mlp.gate_proj.weight",
                    up="model.layers.*.mlp.up_proj.weight",
                ),
            ]
        )

        return MegatronMappingRegistry(*mapping_list)
