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

"""Llama converter for AnyModel compression."""

from typing import List

from transformers import LlamaConfig

from modelopt.torch.puzzletron.anymodel.converter import Converter, ConverterFactory
from modelopt.torch.puzzletron.decilm.deci_lm_hf_code.block_config import (
    AttentionConfig,
    BlockConfig,
    FFNConfig,
)


@ConverterFactory.register_decorator("llama")
class LlamaConverter(Converter):
    """Converter for Llama models to AnyModel format."""

    @staticmethod
    def create_block_configs_from_main_config(config: LlamaConfig) -> List[BlockConfig]:
        """Create uniform block configs for all Llama layers.

        Llama models have uniform architecture across all layers, so we create
        the same BlockConfig for each layer.
        """
        num_hidden_layers = config.num_hidden_layers

        block_config = BlockConfig(
            attention=AttentionConfig(no_op=False, num_key_value_heads=config.num_key_value_heads),
            ffn=FFNConfig(no_op=False, intermediate_size=config.intermediate_size),
        ).to_dict()

        block_configs = [block_config] * num_hidden_layers
        return block_configs
