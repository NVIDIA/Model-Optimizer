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

import logging

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
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
        # TODO: Implement Llama-specific config conversion
        # - Validate block_configs exist
        # - Extract global config values
        # - Extract num_query_groups from first valid block
        # - Extract ffn_hidden_size from first valid block
        # - Convert intermediate_size -> ffn_hidden_size in block_configs
        # - Create LlamaNemotronHeterogeneousProvider
        raise NotImplementedError("provider_bridge() not yet implemented for Llama AnyModel")

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

        Returns:
            MegatronMappingRegistry containing Llama weight mappings
        """
        # TODO: Implement Llama-specific weight mappings
        # - Base mappings (embedding, output, layernorm)
        # - QKV mapping
        # - GatedMLP mapping
        raise NotImplementedError("mapping_registry() not yet implemented for Llama")
