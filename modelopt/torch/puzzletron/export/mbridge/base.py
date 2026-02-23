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
    """Get GPT heterogeneous layer spec using Transformer Engine."""
    return get_gpt_heterogeneous_layer_spec(config, use_te=True)


@dataclass
class GenericHeterogeneousProvider(GPTModelProvider, HeterogeneousTransformerConfig):
    """Generic provider for AnyModel checkpoints with block_configs."""

    # Heterogeneous configuration fields
    heterogeneous_layers_config_path: str | None = None
    heterogeneous_layers_config_encoded_json: str = ""
    transformer_layer_spec: ModuleSpec | Callable = heterogeneous_layer_spec

    def __getattr__(self, name: str):
        """Handle missing attributes for OmegaConf compatibility.

        Returns empty list for per_block_parameters if not yet initialized (before finalize()).
        This allows OmegaConf to serialize/deserialize configs without errors. Actual usage
        should call finalize() first to set per_block_parameters as a real attribute.
        """
        if name == "per_block_parameters":
            # Return existing attribute if set, otherwise [] for OmegaConf compatibility
            try:
                return object.__getattribute__(self, name)
            except AttributeError:
                return []
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


class HeterogeneousBridgeMixin:
    """Mixin for bridges supporting heterogeneous layer architectures (block_configs).

    Must be used with multiple inheritance alongside a model-specific bridge.
    Example: class PuzzletronLlamaAnyModelBridge(HeterogeneousBridgeMixin, LlamaBridge)
    """

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> GPTModelProvider:
        """Convert HF AnyModel config to Megatron GPTModelProvider."""

        parent_provider = super().provider_bridge(hf_pretrained)  # type: ignore[misc]

        provider_kwargs = dataclasses.asdict(parent_provider)

        provider_kwargs["heterogeneous_layers_config_encoded_json"] = (
            self._build_heterogeneous_config_json(hf_pretrained.config)
        )
        return GenericHeterogeneousProvider(**provider_kwargs)

    @classmethod
    def megatron_to_hf_config(cls, provider: GPTModelProvider) -> dict:
        raise NotImplementedError(
            "megatron_to_hf_config() not yet implemented for AnyModel bridges. "
            "AnyModel bridges require special handling for heterogeneous layer configurations."
        )

    def _build_heterogeneous_config_json(self, hf_config) -> str:
        """Build heterogeneous layers config JSON from HF config."""

        hf_config_dict = json.loads(hf_config.to_json_string())

        mcore_block_configs = [
            self._convert_block_config(block) for block in hf_config_dict["block_configs"]
        ]
        return json.dumps({"block_configs": mcore_block_configs}, ensure_ascii=False)

    def _convert_block_config(self, block: dict) -> dict:
        """Convert a single block config from HF format to MCore format."""
        return {
            "attention": self._convert_attention_config(block["attention"]),
            "ffn": self._convert_ffn_config(block["ffn"]),
        }

    def _convert_attention_config(self, attention_config: dict) -> dict:
        """Convert attention config from HF format to MCore format."""
        attention_config = attention_config.copy()
        attention_config["num_query_groups"] = attention_config.pop("num_key_value_heads")
        return attention_config

    def _convert_ffn_config(self, ffn_config: dict) -> dict:
        """Convert FFN/MLP config from HF format to MCore format."""
        ffn_config = ffn_config.copy()
        ffn_config["ffn_hidden_size"] = ffn_config.pop("intermediate_size")
        return ffn_config
