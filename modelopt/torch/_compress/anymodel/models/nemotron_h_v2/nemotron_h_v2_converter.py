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

from typing import List

from modelopt.torch._compress.anymodel.converter import Converter, ConverterFactory
from modelopt.torch._compress.decilm.deci_lm_hf_code.block_config import (
    AttentionConfig,
    BlockConfig,
    FFNConfig,
    MambaConfig,
    MoEConfig,
)


@ConverterFactory.register_decorator("nemotron_h_v2")
class NemotronHV2Converter(Converter):
    @staticmethod
    def create_block_configs_from_main_config(config) -> List[BlockConfig]:
        # Create block configs for each layer based on the hybrid_override_pattern
        block_configs = []

        # Parse the hybrid_override_pattern: "M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-"
        pattern = config.hybrid_override_pattern
        print(f"Parsing hybrid pattern: {pattern}")

        for i, char in enumerate(pattern):
            if char == "M":
                _block_config = BlockConfig(
                    attention=AttentionConfig(
                        mamba=MambaConfig(  # Those parameters are currently used only for calc_block_stats.
                            state_dim=config.ssm_state_size,
                            num_heads=config.mamba_num_heads,
                            head_dim=config.mamba_head_dim,
                            num_groups=config.n_groups,
                        )
                    ),
                    ffn=FFNConfig(no_op=True),
                )

            elif char == "-":
                _block_config = BlockConfig(
                    attention=AttentionConfig(no_op=True),
                    ffn=FFNConfig(intermediate_size=config.intermediate_size),
                )

            elif char == "*":
                _block_config = BlockConfig(
                    attention=AttentionConfig(num_key_value_heads=config.num_key_value_heads),
                    ffn=FFNConfig(no_op=True),
                )

            elif char == "E":
                _block_config = BlockConfig(
                    attention=AttentionConfig(no_op=True),
                    ffn=FFNConfig(
                        moe=MoEConfig(
                            num_local_experts=config.n_routed_experts,
                            expert_intermediate_dim=config.moe_intermediate_size,
                            num_experts_per_tok=config.num_experts_per_tok,
                        )
                    ),
                )
            else:
                raise ValueError(
                    f"Unknown character '{char}' in hybrid_override_pattern at position {i}"
                )

            block_configs.append(_block_config)

        print(f"Created {len(block_configs)} block configs from pattern")
        return block_configs
