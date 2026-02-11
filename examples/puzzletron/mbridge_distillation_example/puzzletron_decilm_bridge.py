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

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.core.models.gpt.gpt_model import GPTModel

logger = logging.getLogger(__name__)


# Register bridge using string-based registration for DeciLMModel
# This allows registration even if DeciLMModel is not importable at module level
# (e.g., when using trust_remote_code=True)
@MegatronModelBridge.register_bridge(source="DeciLMModel", target=GPTModel, model_type="decilm")
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

        This method should:
        1. Call super().provider_bridge() to get base GPTModelProvider
        2. Set DeciLM-specific defaults (normalization, activation, etc.)
        3. Handle heterogeneous layers if needed (block_configs)

        Args:
            hf_pretrained: HuggingFace PreTrainedCausalLM containing the DeciLM config

        Returns:
            GPTModelProvider configured for DeciLM architecture

        Raises:
            NotImplementedError: Method not yet implemented
        """
        raise NotImplementedError(
            "provider_bridge() not yet implemented. "
            "This method should convert DeciLM config to GPTModelProvider, "
            "handling heterogeneous layers defined in block_configs."
        )

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

        This method should return a MegatronMappingRegistry containing:
        - AutoMapping for simple 1:1 parameter mappings
        - QKVMapping for attention Q/K/V weight concatenation
        - GatedMLPMapping for gated MLP weight concatenation
        - Any other special mappings needed for DeciLM

        The mappings should handle:
        - Standard transformer layers (embedding, output, layernorms)
        - Attention layers (QKV, output projection)
        - MLP layers (gate, up, down projections)
        - Heterogeneous layer configurations if they affect naming

        Returns:
            MegatronMappingRegistry containing all weight mapping definitions

        Raises:
            NotImplementedError: Method not yet implemented
        """
        raise NotImplementedError(
            "mapping_registry() not yet implemented. "
            "This method should return a MegatronMappingRegistry with mappings "
            "from DeciLM parameter names to Megatron parameter names. "
            "Reference: LlamaBridge.mapping_registry() for similar architecture."
        )
