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

"""GPT-OSS-20B converter for AnyModel compression."""

from typing import List

from transformers import PretrainedConfig

from ....block_config import AttentionConfig, BlockConfig, MoEConfig
from ...converter import Converter, ConverterFactory

__all__ = ["GptOssConverter"]


@ConverterFactory.register_decorator("gpt_oss")
class GptOssConverter(Converter):
    """Converter for GPT-OSS models to AnyModel format.

    GPT-OSS is a pure MoE model with 32/128 experts per layer and 4/16 active experts.
    All layers use MoE FFN (no standard dense FFN layers).
    """

    @staticmethod
    def create_block_configs_from_main_config(config: PretrainedConfig) -> List[BlockConfig]:
        """Create block configs for GPT-OSS layers.

        GPT-OSS uses MoE for all MLP layers with:
        - 32/128 experts
        - 4/16 active experts per token
        - No dense/standard FFN layers
        """
        num_hidden_layers = config.num_hidden_layers
        num_experts = config.num_local_experts
        top_k = config.experts_per_token
        expert_intermediate_size = config.intermediate_size
        layer_types = tuple(getattr(config, "layer_types", ()) or ())
        sliding_window = getattr(config, "sliding_window", None)

        block_configs = []
        for layer_idx in range(num_hidden_layers):
            attention_type = (
                layer_types[layer_idx] if layer_idx < len(layer_types) else None
            )
            window = (
                "full"
                if attention_type == "full_attention"
                else int(sliding_window) if sliding_window is not None else None
            )
            block_config = BlockConfig(
                subblock_configs=(
                    AttentionConfig(
                        no_op=False,
                        num_kv_heads=config.num_key_value_heads,
                        num_query_heads=config.num_attention_heads,
                        sliding_window_size=window,
                    ),
                    MoEConfig(
                        num_experts=num_experts,
                        top_k=top_k,
                        expert_intermediate_size=expert_intermediate_size,
                    ),
                ),
            ).to_dict()
            block_configs.append(block_config)

        return block_configs
