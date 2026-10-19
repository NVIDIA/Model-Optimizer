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

from typing import TYPE_CHECKING, List

from ....block_config import AttentionConfig, BlockConfig, FFNConfig, MoEConfig
from ...converter import Converter, ConverterFactory

if TYPE_CHECKING:
    from transformers import Qwen3VLMoeConfig

__all__ = ["Qwen3VLConverter"]


@ConverterFactory.register_decorator("qwen3_vl")
class Qwen3VLConverter(Converter):
    @staticmethod
    def create_block_configs_from_main_config(config: "Qwen3VLMoeConfig") -> List[BlockConfig]:
        # Qwen3-VL MoE has nested text_config
        text_config = config.text_config if hasattr(config, "text_config") else config

        num_hidden_layers = text_config.num_hidden_layers
        decoder_sparse_step = getattr(text_config, "decoder_sparse_step", 1)
        mlp_only_layers = getattr(text_config, "mlp_only_layers", [])

        block_configs = []
        for layer_idx in range(num_hidden_layers):
            # Check if this layer is MoE or dense
            is_moe_layer = (layer_idx % decoder_sparse_step == 0) and (
                layer_idx not in mlp_only_layers
            )

            if is_moe_layer:
                # MoE layer
                block_config = BlockConfig(
                    attention=AttentionConfig(
                        no_op=False, num_key_value_heads=text_config.num_key_value_heads
                    ),
                    ffn=FFNConfig(
                        moe=MoEConfig(
                            num_local_experts=text_config.num_experts,
                            expert_intermediate_dim=text_config.moe_intermediate_size,
                            num_experts_per_tok=text_config.num_experts_per_tok,
                        )
                    ),
                )
            else:
                # Dense layer
                block_config = BlockConfig(
                    attention=AttentionConfig(
                        no_op=False, num_key_value_heads=text_config.num_key_value_heads
                    ),
                    ffn=FFNConfig(no_op=False, intermediate_size=text_config.intermediate_size),
                )

            block_configs.append(block_config)

        print(
            f"Created {len(block_configs)} block configs for Qwen3-VL MoE (decoder_sparse_step={decoder_sparse_step})"
        )
        return block_configs
