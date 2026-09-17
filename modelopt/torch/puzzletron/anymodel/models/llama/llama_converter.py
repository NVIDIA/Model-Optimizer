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

from ....block_config import BlockConfig
from ...converter import ConverterFactory, GenericDecoderConverter
from .llama_model_descriptor import LlamaModelDescriptor

__all__ = ["LlamaConverter"]


@ConverterFactory.register_decorator("llama")
class LlamaConverter(GenericDecoderConverter):
    """Converter for Llama models to AnyModel format."""

    @staticmethod
    def create_block_configs_from_main_config(config: LlamaConfig) -> List[BlockConfig]:
        """Create the descriptor-owned Llama block schema for legacy callers."""
        return [
            block.to_dict()
            for block in GenericDecoderConverter.create_block_configs(
                LlamaModelDescriptor,
                config,
            )
        ]
