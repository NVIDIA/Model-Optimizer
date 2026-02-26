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

from modelopt.torch.puzzletron.anymodel.converter import Converter, ConverterFactory
from modelopt.torch.puzzletron.decilm.deci_lm_hf_code.block_config import (
    AttentionConfig,
    BlockConfig,
    FFNConfig,
    MoEConfig,
)


@ConverterFactory.register_decorator("gpt_oss_20b")
class GptOss20bConverter(Converter):
    """Converter for GPT-OSS-20B models to AnyModel format.

    GPT-OSS-20B is a pure MoE model with 32 experts per layer and 4 active experts.
    All layers use MoE FFN (no standard dense FFN layers).
    """

    quantized = "mxfp4"

    @staticmethod
    def create_block_configs_from_main_config(config: PretrainedConfig) -> List[BlockConfig]:
        """Create block configs for GPT-OSS-20B layers.

        GPT-OSS-20B uses MoE for all FFN layers with:
        - 32 local experts (num_local_experts)
        - 4 active experts per token (experts_per_token)
        - No dense/standard FFN layers
        """
        num_hidden_layers = config.num_hidden_layers
        num_local_experts = config.num_local_experts
        experts_per_token = config.experts_per_token
        intermediate_size = config.intermediate_size

        block_configs = []
        for layer_idx in range(num_hidden_layers):
            block_config = BlockConfig(
                attention=AttentionConfig(
                    no_op=False, num_key_value_heads=config.num_key_value_heads
                ),
                ffn=FFNConfig(
                    no_op=False,
                    intermediate_size=None,  # MoE doesn't use this field
                    moe=MoEConfig(
                        num_local_experts=num_local_experts,
                        num_experts_per_tok=experts_per_token,
                        expert_intermediate_dim=intermediate_size,
                    ),
                ),
            ).to_dict()
            block_configs.append(block_config)

        return block_configs
