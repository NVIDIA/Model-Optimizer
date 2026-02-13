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
Megatron Bridge for Puzzletron DeciLM models.

This bridge handles conversion between Puzzletron DeciLM (heterogeneous layer architecture)
and Megatron-Core GPT models.

As a user you would not use this bridge directly, but through `AutoBridge`.

Example:
    >>> from megatron.bridge.models.conversion.auto_bridge import AutoBridge
    >>> import puzzletron_decilm_bridge  # Register the bridge
    >>> bridge = AutoBridge.from_hf_pretrained("path/to/decilm/checkpoint", trust_remote_code=True)
    >>> provider = bridge.to_megatron_provider()
"""

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

logger = logging.getLogger(__name__)


# Register bridge using string-based registration for DeciLMForCausalLM
# This matches the architecture string in the checkpoint's config.json
# This allows registration even if DeciLMForCausalLM is not importable at module level
# (e.g., when using trust_remote_code=True)
# Note: This will override LlamaNemotronBridge registration for DeciLMForCausalLM
@MegatronModelBridge.register_bridge(
    source="DeciLMForCausalLM", target=GPTModel, model_type="decilm"
)
class PuzzletronDeciLMBridge(MegatronModelBridge):
    """
    Megatron Bridge for Puzzletron DeciLM Causal LM.

    DeciLM models have heterogeneous layers where each layer can have different
    configurations (intermediate_size, num_heads, etc.) defined in block_configs.

    This bridge handles:
    - Converting DeciLM config to Megatron GPTModelProvider
    - Mapping DeciLM weight names to Megatron weight names
    - Handling heterogeneous layer configurations
    """

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> GPTModelProvider:
        """Convert HuggingFace DeciLM config to Megatron GPTModelProvider.

        Based on LlamaNemotronBridge.provider_bridge() but adapted for Puzzletron DeciLM models.
        Key difference: Puzzletron models may have num_key_value_heads per-layer in block_configs,
        not as a global config attribute.

        Args:
            hf_pretrained: HuggingFace PreTrainedCausalLM containing the DeciLM config

        Returns:
            GPTModelProvider configured for DeciLM architecture
        """
        hf_config = hf_pretrained.config

        # Validate heterogeneous DeciLM config
        if not (hasattr(hf_config, "block_configs") and hf_config.block_configs):
            num_layers = getattr(hf_config, "num_hidden_layers", "unknown")
            raise ValueError(
                "PuzzletronDeciLMBridge requires block_configs. "
                f"Model with {num_layers} layers and no block_configs is not supported."
            )

        # Extract num_query_groups for heterogeneous models
        # For heterogeneous models, GQA is defined in each block config
        # Note: This is just a default value; actual per-layer values come from heterogeneous_layers_config_encoded_json
        #
        # Puzzletron DeciLM removes num_key_value_heads from global config (it's per-layer only),
        # so we extract from block_configs. We iterate to find first block with valid n_heads_in_group
        # (skip no_op/replace_with_linear blocks where n_heads_in_group is None).
        num_query_groups = None
        if hasattr(hf_config, "block_configs") and hf_config.block_configs:
            # Find first block with valid n_heads_in_group (skip no_op/replace_with_linear blocks)
            for block in hf_config.block_configs:
                if hasattr(block, "attention") and hasattr(block.attention, "n_heads_in_group"):
                    n_heads_in_group = block.attention.n_heads_in_group
                    if n_heads_in_group is not None:
                        num_query_groups = hf_config.num_attention_heads // n_heads_in_group
                        break

        # Extract ffn_hidden_size as a default/placeholder value
        #
        # IMPORTANT: For heterogeneous models, this value is NOT actually used during model building.
        # It's only required because TransformerConfig.ffn_hidden_size is a required field (not Optional).
        #
        # The actual per-layer ffn_hidden_size values come from:
        #   1. heterogeneous_layers_config_encoded_json (contains all block_configs)
        #   2. Parsed by finalize() into per_block_parameters
        #   3. MCore uses get_config_for_layer() to get per-layer ffn_hidden_size from per_block_parameters
        #
        # This value is just a placeholder to satisfy the dataclass requirement.
        # Puzzletron DeciLM removes intermediate_size from global config (it's per-layer only),
        # so we extract from block_configs. We iterate to find first block with valid intermediate_size
        # (skip no_op/replace_with_linear blocks where intermediate_size is None).
        ffn_hidden_size = None
        if hasattr(hf_config, "block_configs") and hf_config.block_configs:
            # Find first block with valid intermediate_size (skip no_op/replace_with_linear blocks)
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
            "rotary_base": hf_config.rope_theta,
            "kv_channels": getattr(hf_config, "head_dim", None),
            "gated_linear_unit": True,  # DeciLM uses SwiGLU
            "make_vocab_size_divisible_by": self.make_vocab_size_divisible_by(hf_config.vocab_size),
            "share_embeddings_and_output_weights": getattr(hf_config, "tie_word_embeddings", False),
            "fp16": (self.dtype_from_hf(hf_config, default=torch.float32) == torch.float16),
            "bf16": (self.dtype_from_hf(hf_config, default=torch.float32) == torch.bfloat16),
            "params_dtype": self.dtype_from_hf(hf_config, default=torch.float32),
            "generation_config": hf_pretrained.generation_config,
            "vocab_size": hf_config.vocab_size,
        }

        # Handle rope scaling for Llama 3.1/3.3
        if hasattr(hf_config, "rope_scaling") and hf_config.rope_scaling:
            if hf_config.rope_scaling.get("rope_type") == "llama3":
                provider_kwargs["rope_scaling_factor"] = hf_config.rope_scaling.get("factor", 8.0)

        # Convert HF config format to MCore format for heterogeneous_layers_config_encoded_json
        # MCore's MLPConfig.build_config_from_dict() expects ffn_hidden_size or ffn_mult,
        # but HF configs use intermediate_size. We convert intermediate_size -> ffn_hidden_size.
        # Note: LlamaNemotronBridge works because Nemotron models may already have ffn_mult
        # in their block_configs, but Puzzletron models use intermediate_size.
        import json

        hf_config_dict = json.loads(hf_config.to_json_string())
        mcore_block_configs = []

        for block in hf_config_dict.get("block_configs", []):
            mcore_block = {}

            # Copy attention config as-is (n_heads_in_group is already in correct format)
            if "attention" in block:
                mcore_block["attention"] = block["attention"]

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
        """Convert Megatron GPTModelProvider config to HuggingFace DeciLM config dict.

        This method should:
        1. Call super().megatron_to_hf_config() for base conversion
        2. Add DeciLM-specific config fields (block_configs, etc.)

        Args:
            provider: GPTModelProvider with DeciLM configuration

        Returns:
            Dictionary of HuggingFace DeciLMConfig parameters

        Raises:
            NotImplementedError: Method not yet implemented
        """
        raise NotImplementedError(
            "megatron_to_hf_config() not yet implemented. "
            "This method should convert GPTModelProvider back to DeciLM config format."
        )

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Define weight mappings between DeciLM and Megatron formats.

        Uses same mappings as LlamaNemotronBridge - DeciLM models use the same
        weight structure as standard Llama models (q_proj, k_proj, v_proj, etc.).
        Wildcard patterns (*) work for standard attention/MLP layers.

        Note: If Puzzletron models have Mamba/MoE layers, we'd need per-layer mappings,
        but MBridge's MegatronMappingRegistry only supports wildcard patterns.

        Returns:
            MegatronMappingRegistry containing all weight mapping definitions
        """
        # Base mappings (same as LlamaNemotronBridge)
        param_mappings = {
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "output_layer.weight": "lm_head.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",
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
