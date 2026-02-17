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
Mixin class for bridges that support heterogeneous layer architectures.

This module provides a mixin class for converting models with block_configs
(heterogeneous layer configurations) to Megatron-Core format via Megatron-Bridge.
"""

import dataclasses
import json
from collections.abc import Callable
from dataclasses import dataclass

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.transformer_config import HeterogeneousTransformerConfig
from megatron.core.models.gpt.heterogeneous.heterogeneous_layer_specs import (
    get_gpt_heterogeneous_layer_spec,
)
from megatron.core.transformer.spec_utils import ModuleSpec


def heterogeneous_layer_spec(config) -> ModuleSpec:
    """Determine the most appropriate layer specification based on availability.

    Uses Transformer Engine specs since TE is a required dependency.

    Args:
        config: GPT configuration object

    Returns:
        ModuleSpec: The selected module specification
    """
    return get_gpt_heterogeneous_layer_spec(config, use_te=True)


@dataclass
class GenericHeterogeneousProvider(GPTModelProvider, HeterogeneousTransformerConfig):
    """
    Generic provider for heterogeneous (AnyModel) checkpoints.

    This provider is model-agnostic and works with any model architecture
    (Llama, Qwen3, Mistral, etc.) that uses block_configs for heterogeneous
    layer specifications. Model-specific settings should be provided via kwargs.
    """

    # Heterogeneous configuration fields
    heterogeneous_layers_config_path: str | None = None
    heterogeneous_layers_config_encoded_json: str = ""
    transformer_layer_spec: ModuleSpec | Callable = heterogeneous_layer_spec


class HeterogeneousBridgeMixin:
    """
    Mixin class for bridges that support heterogeneous layer architectures.

    Provides shared functionality for handling models with block_configs (heterogeneous
    layer configurations where each layer can have different dimensions, attention heads, etc.).
    This is a mixin - it must be used with multiple inheritance alongside a model-specific
    bridge (e.g., LlamaBridge, Qwen3Bridge). The mixin calls super() to access methods
    from the model-specific bridge via Python's Method Resolution Order (MRO).

    Example:
        class PuzzletronLlamaAnyModelBridge(HeterogeneousBridgeMixin, LlamaBridge):
            pass
    """

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> GPTModelProvider:
        """Convert HuggingFace AnyModel config to Megatron GPTModelProvider.

        Reuses the parent bridge's provider_bridge() to get all model-specific settings,
        then extends it with heterogeneous layer configuration (block_configs).

        This method works for any model bridge that inherits from both a model-specific
        bridge (e.g., LlamaBridge, Qwen3Bridge) and this base class.

        Args:
            hf_pretrained: HuggingFace PreTrainedCausalLM containing the model config

        Returns:
            GenericHeterogeneousProvider configured for AnyModel architecture
        """

        # Get fully configured provider from parent bridge (includes all model-specific settings)
        # This calls the parent's provider_bridge() method (e.g., LlamaBridge.provider_bridge())
        # via Python's Method Resolution Order (MRO). The mixin doesn't inherit from
        # MegatronModelBridge, but super() finds the method in the other parent class.
        # Note: mypy can't verify this, but it works correctly at runtime.
        parent_provider = super().provider_bridge(hf_pretrained)  # type: ignore[misc]

        # Convert provider to dict (captures ALL fields automatically, no duplication)
        provider_kwargs = dataclasses.asdict(parent_provider)

        # Add heterogeneous layer configuration (block_configs)
        # Reuse the base class method to avoid duplicating the conversion logic
        # The method will extract num_query_groups from config if not provided
        hf_config = hf_pretrained.config
        provider_kwargs["heterogeneous_layers_config_encoded_json"] = (
            self._build_heterogeneous_config_json(hf_config)
        )

        # Create heterogeneous provider with all settings from parent + heterogeneous config
        return GenericHeterogeneousProvider(**provider_kwargs)

    @classmethod
    def megatron_to_hf_config(cls, provider: GPTModelProvider) -> dict:
        """Convert Megatron GPTModelProvider config to HuggingFace config dict.

        This method is not yet implemented for AnyModel bridges, as it requires
        handling heterogeneous layer configurations (block_configs) which is
        more complex than standard model conversion.

        Args:
            provider: GPTModelProvider with AnyModel configuration

        Returns:
            Dictionary of HuggingFace config parameters

        Raises:
            NotImplementedError: This method is not yet implemented for AnyModel bridges
        """
        raise NotImplementedError(
            "megatron_to_hf_config() not yet implemented for AnyModel bridges. "
            "AnyModel bridges require special handling for heterogeneous layer configurations."
        )

    def _extract_num_query_groups(self, hf_config) -> int | None:
        """
        Extract num_query_groups from global config or block_configs.

        Note: num_query_groups in Megatron = num_key_value_heads in HF (they're the same).
        The base class CONFIG_MAPPING already handles this, so we only need this for
        extracting from block_configs when not in global config.
        """
        # Base class will handle global config via CONFIG_MAPPING, so we only
        # need to extract from block_configs if not in global config
        if hasattr(hf_config, "num_key_value_heads") and hf_config.num_key_value_heads is not None:
            # Base class will handle this, return None to use base class value
            return None

        # Fall back to block_configs - extract num_key_value_heads from first valid block
        for block in hf_config.block_configs:
            if hasattr(block, "attention") and hasattr(block.attention, "num_key_value_heads"):
                num_kv_heads = block.attention.num_key_value_heads
                if num_kv_heads is not None:
                    return num_kv_heads
            # Also check for n_heads_in_group (DeciLM format) and convert
            if hasattr(block, "attention") and hasattr(block.attention, "n_heads_in_group"):
                n_heads_in_group = block.attention.n_heads_in_group
                if n_heads_in_group is not None:
                    # n_heads_in_group = num_attention_heads / num_query_groups
                    # So num_query_groups = num_attention_heads / n_heads_in_group
                    return hf_config.num_attention_heads // n_heads_in_group
        return None

    def _convert_block_config(
        self, block: dict, hf_config, default_num_query_groups: int | None
    ) -> dict:
        """Convert a single block config from HF format to MCore format."""
        mcore_block = {}

        # Process attention config
        if "attention" in block and block["attention"] is not None:
            mcore_block["attention"] = self._convert_attention_config(
                block["attention"], hf_config, default_num_query_groups
            )

        # Process FFN/MLP config
        ffn_key = block.get("ffn") or block.get("mlp")
        if ffn_key:
            mcore_block["ffn"] = self._convert_ffn_config(ffn_key)

        return mcore_block

    def _convert_attention_config(
        self, attention_config: dict, hf_config, default_num_query_groups: int | None
    ) -> dict:
        """Convert attention config from HF format to MCore format."""
        if isinstance(attention_config, dict):
            attention_config = attention_config.copy()
        else:
            attention_config = {}

        # Convert num_key_value_heads (AnyModel format) to num_query_groups (MCore format)
        if (
            "num_key_value_heads" in attention_config
            and attention_config["num_key_value_heads"] is not None
        ):
            attention_config["num_query_groups"] = attention_config["num_key_value_heads"]
            attention_config.pop("num_key_value_heads", None)

        # Set num_query_groups if missing
        if (
            "num_query_groups" not in attention_config
            and "n_heads_in_group" not in attention_config
        ):
            if default_num_query_groups is not None:
                attention_config["num_query_groups"] = default_num_query_groups
            else:
                # Default to MHA (no grouping)
                attention_config["num_query_groups"] = hf_config.num_attention_heads

        # Ensure required fields are set
        attention_config.setdefault("no_op", False)
        attention_config.setdefault("replace_with_linear", False)

        return attention_config

    def _convert_ffn_config(self, ffn_config: dict) -> dict:
        """Convert FFN/MLP config from HF format to MCore format."""
        if isinstance(ffn_config, dict):
            ffn_config = ffn_config.copy()
        else:
            ffn_config = {}

        # Convert intermediate_size to ffn_hidden_size (MCore expects this)
        if "intermediate_size" in ffn_config and ffn_config["intermediate_size"] is not None:
            ffn_config["ffn_hidden_size"] = ffn_config.pop("intermediate_size")

        return ffn_config

    def _build_heterogeneous_config_json(
        self, hf_config, num_query_groups: int | None = None
    ) -> str:
        """Build the heterogeneous layers config JSON from HF config.

        Args:
            hf_config: HuggingFace model configuration
            num_query_groups: Optional num_query_groups value (if not provided, extracted from config)

        Returns:
            JSON string for heterogeneous_layers_config_encoded_json
        """
        if num_query_groups is None:
            num_query_groups = self._extract_num_query_groups(hf_config)

        hf_config_dict = json.loads(hf_config.to_json_string())
        mcore_block_configs = [
            self._convert_block_config(block, hf_config, num_query_groups)
            for block in hf_config_dict.get("block_configs", [])
        ]

        # Build MCore format JSON
        mcore_config = {"block_configs": mcore_block_configs}
        if "rope_scaling" in hf_config_dict:
            mcore_config["rope_scaling"] = hf_config_dict["rope_scaling"]

        return json.dumps(mcore_config, ensure_ascii=False)
