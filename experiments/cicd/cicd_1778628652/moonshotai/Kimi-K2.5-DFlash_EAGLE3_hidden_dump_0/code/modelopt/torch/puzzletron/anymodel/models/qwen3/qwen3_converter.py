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

from typing import List

from transformers import Qwen3Config

from ....block_config import AttentionConfig, BlockConfig, FFNConfig
from ...converter import Converter, ConverterFactory

__all__ = ["Qwen3Converter"]


@ConverterFactory.register_decorator("qwen3")
class Qwen3Converter(Converter):
    @staticmethod
    def create_block_configs_from_main_config(config: Qwen3Config) -> List[BlockConfig]:
        num_hidden_layers = config.num_hidden_layers

        block_configs = [
            BlockConfig(
                attention=AttentionConfig(
                    no_op=False, num_key_value_heads=config.num_key_value_heads
                ),
                ffn=FFNConfig(no_op=False, intermediate_size=config.intermediate_size),
            ).to_dict()
            for _ in range(num_hidden_layers)
        ]
        return block_configs
