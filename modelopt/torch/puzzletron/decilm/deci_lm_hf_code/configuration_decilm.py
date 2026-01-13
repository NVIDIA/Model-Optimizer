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

import copy
import dataclasses
import warnings
from typing import Any

from transformers.utils import is_flash_attn_2_available, is_torch_sdpa_available

from .block_config import BlockConfig
from .transformers_4_44_2__configuration_llama import LlamaConfig

# fakes imports to make AutoConfig infer dependencies
from .transformers_4_44_2__modeling_rope_utils import rope_config_validation
from .transformers_4_51_3__cache_utils import HybridChunkedCache
from .transformers_4_51_3__configuration_llama4 import Llama4Config

# make sure that auto-formatting doesn't remove the fake imports
rope_config_validation
Llama4Config
HybridChunkedCache


class DeciLMConfig(LlamaConfig):
    model_type = "nemotron-nas"

    # Mapping from global attribute names to their per-layer equivalents in block_configs
    # Format: 'global_name': ('block_section', 'layer_name')
    PER_LAYER_ATTRIBUTE_MAPPING = {
        "intermediate_size": ("ffn", "intermediate_size"),
        "num_key_value_heads": (
            "attention",
            "n_heads_in_group",
        ),  # Note: derived value (num_heads / num_kv_heads)
        "hidden_act": ("ffn", "hidden_act"),
        "sliding_window": ("attention", "window_length"),  # Note: different name!
    }

    def __init__(
        self,
        block_configs: list[dict] | list[BlockConfig] | None = None,
        position_embedding_type: str = "rope",
        llama4_attn_implementation: str | None = None,
        block_return_only_hidden_states: bool = False,
        router_aux_loss_coef: float = 0.01,
        router_z_loss_coef: float = 0.0,
        output_router_logits: bool = False,
        head_dim: int | None = 128,
        o_proj_bias: bool = False,
        **kwargs,
    ):
        self.block_configs: list[BlockConfig] = block_configs
        if self.block_configs is not None:
            if isinstance(self.block_configs[0], dict):
                self.block_configs = [BlockConfig(**conf) for conf in self.block_configs]

        assert position_embedding_type in ["rope", "rope_llama4", "none", "mistral_yarn"]
        self.position_embedding_type = position_embedding_type
        if self.position_embedding_type == "none":
            self.rope_theta = None
            self.rope_scaling = None

        self.block_return_only_hidden_states = block_return_only_hidden_states
        self.router_aux_loss_coef = router_aux_loss_coef
        self.router_z_loss_coef = router_z_loss_coef
        self.output_router_logits = output_router_logits
        self.o_proj_bias = o_proj_bias

        self._choose_llama4_attn_implementation(llama4_attn_implementation)
        attn_implementation = self._choose_llama3_attn_implementation(kwargs)
        super().__init__(attn_implementation=attn_implementation, **kwargs)
        self.head_dim = (
            head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads
        )

        # Delete per-layer attributes after parent init (they should only exist in block_configs)
        self._delete_per_layer_attributes()

        if self.block_configs is not None:
            assert len(self.block_configs) == self.num_hidden_layers

    def _delete_per_layer_attributes(self):
        """Delete per-layer attributes that should only exist in block_configs.

        These attributes are intentionally deleted AFTER super().__init__() to ensure
        they don't exist at the global config level. Deleting them (rather than setting
        to None) makes it clear they shouldn't be accessed globally.
        """
        present_attrs = {
            attr: getattr(self, attr)
            for attr in self.PER_LAYER_ATTRIBUTE_MAPPING
            if hasattr(self, attr)
        }
        if present_attrs:
            warnings.warn(
                f"Deleting global per-layer attributes (should only be in block_configs): {present_attrs}",
                UserWarning,
                stacklevel=3,
            )
        for attr in self.PER_LAYER_ATTRIBUTE_MAPPING:
            if hasattr(self, attr):
                delattr(self, attr)

    def _choose_llama4_attn_implementation(self, llama4_attn_implementation):
        self.llama4_attn_implementation = llama4_attn_implementation
        if self.llama4_attn_implementation is None:
            if is_torch_sdpa_available():
                _print_once("auto-setting llama4_attn_implementation to sdpa")
                self.llama4_attn_implementation = "sdpa"
            else:
                _print_once("auto-setting llama4_attn_implementation to eager")
                self.llama4_attn_implementation = "eager"

    def _choose_llama3_attn_implementation(self, kwargs: dict[str, Any]) -> str:
        attn_implementation = kwargs.pop("attn_implementation", None)
        if attn_implementation is None and is_flash_attn_2_available():
            _print_once("auto-setting attn_implementation (for Llama3 layers) to flash_attention_2")
            attn_implementation = "flash_attention_2"

        if self.block_configs is not None:
            using_unshifted_sink = any(
                block_config.attention.unshifted_sink for block_config in self.block_configs
            )
            if using_unshifted_sink and attn_implementation != "eager":
                warnings.warn(
                    "Forcing attn_implementation='eager' since some attention layers use unshifted sink"
                )
                attn_implementation = "eager"
        return attn_implementation

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary, removing per-layer-only attributes."""
        self_dict = super().to_dict()
        if self.block_configs is not None:
            self_dict["block_configs"] = [dataclasses.asdict(conf) for conf in self.block_configs]

        # Remove global keys that should only exist per-layer in block_configs
        for key in self.PER_LAYER_ATTRIBUTE_MAPPING:
            self_dict.pop(key, None)

        return self_dict

    def set_block_configs(self, block_configs: list[BlockConfig]) -> "DeciLMConfig":
        new_model_config = copy.deepcopy(self)
        new_model_config.block_configs = block_configs
        new_model_config.num_hidden_layers = len(block_configs)
        return new_model_config

    def get_num_hidden_layers(self) -> int:
        return self.num_hidden_layers

    def get_hidden_size(self) -> int:
        return self.hidden_size

    def get_embedding_layer_name(self) -> str:
        return "model.embed_tokens"

    def get_final_layer_norm_layer_name(self) -> str:
        return "model.norm"

    def get_lm_head_layer_name(self) -> str:
        return "lm_head"

    def get_layers_layer_name(self) -> str:
        return "model.layers"

    def get_block_config(self, layer_idx: int | tuple[int, ...]) -> BlockConfig:
        if isinstance(layer_idx, tuple) and len(layer_idx) == 1:
            layer_idx = layer_idx[0]

        if isinstance(layer_idx, int):
            return self.block_configs[layer_idx]

        external_layer_idx, internal_layer_idx = layer_idx
        return self.block_configs[external_layer_idx].parallel_blocks[internal_layer_idx]

    def get_min_attention_chunk_size(self) -> int | None:
        min_chunk_size = float("inf")
        for block_config in self.block_configs:
            if block_config.attention.llama4 is not None:
                attention_chunk_size = block_config.attention.llama4.attention_chunk_size
                if attention_chunk_size is not None:
                    min_chunk_size = min(min_chunk_size, attention_chunk_size)

        if min_chunk_size == float("inf"):
            return None
        return min_chunk_size


def _print_once(message: str):
    if not hasattr(_print_once, "was_printed"):
        _print_once.was_printed = set()
    if message not in _print_once.was_printed:
        _print_once.was_printed.add(message)
        print(message)
