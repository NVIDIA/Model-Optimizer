# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from ....block_config import AttentionConfig, BlockConfig, FFNConfig, MambaConfig, MoEConfig
from ...converter import Converter, ConverterFactory

if TYPE_CHECKING:
    from transformers import PretrainedConfig

__all__ = ["Qwen3P5Converter"]


@ConverterFactory.register_decorator("qwen3_6_text")
@ConverterFactory.register_decorator("qwen3_6")
@ConverterFactory.register_decorator("qwen3_5_text")
@ConverterFactory.register_decorator("qwen3_5")
@ConverterFactory.register_decorator("qwen3_5_moe_text")
@ConverterFactory.register_decorator("qwen3_5_moe")
class Qwen3P5Converter(Converter):
    @staticmethod
    def create_block_configs_from_main_config(config: "PretrainedConfig") -> List[BlockConfig]:
        text_config = config.text_config if hasattr(config, "text_config") else config
        layer_types = getattr(text_config, "layer_types", None)
        if layer_types is None:
            layer_types = [
                "linear_attention" if bool((i + 1) % 4) else "full_attention"
                for i in range(text_config.num_hidden_layers)
            ]
        if len(layer_types) < text_config.num_hidden_layers:
            raise ValueError(
                f"Qwen3.5 layer_types has {len(layer_types)} entries, "
                f"expected at least {text_config.num_hidden_layers}"
            )

        block_configs = []
        for layer_idx, layer_type in enumerate(layer_types[: text_config.num_hidden_layers]):
            if layer_type == "linear_attention":
                attention_config = None
                mamba_config = MambaConfig(
                    state_dim=text_config.linear_key_head_dim,
                    num_heads=text_config.linear_num_value_heads,
                    head_dim=text_config.linear_value_head_dim,
                    num_groups=text_config.linear_num_key_heads,
                    conv_kernel_size=text_config.linear_conv_kernel_dim,
                )
            elif layer_type == "full_attention":
                attention_config = AttentionConfig(
                    no_op=False,
                    num_kv_heads=text_config.num_key_value_heads,
                    num_query_heads=text_config.num_attention_heads,
                )
                mamba_config = None
            else:
                raise ValueError(
                    f"Unsupported Qwen3.5 layer type at layer {layer_idx}: {layer_type}"
                )

            if getattr(text_config, "num_experts", None) is not None:
                ffn_config = MoEConfig(
                    num_experts=text_config.num_experts,
                    top_k=text_config.num_experts_per_tok,
                    expert_intermediate_size=text_config.moe_intermediate_size,
                    shared_expert_intermediate_size=getattr(
                        text_config, "shared_expert_intermediate_size", None
                    ),
                )
            else:
                ffn_config = FFNConfig(
                    no_op=False, intermediate_size=text_config.intermediate_size
                )
            if mamba_config is not None:
                block_configs.append(BlockConfig(subblock_configs=(mamba_config, ffn_config)))
            else:
                block_configs.append(
                    BlockConfig(subblock_configs=(attention_config, ffn_config))
                )

        return block_configs
