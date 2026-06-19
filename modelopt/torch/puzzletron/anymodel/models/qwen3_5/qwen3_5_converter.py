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
from pathlib import Path
from typing import List

from ....block_config import AttentionConfig, BlockConfig, FFNConfig
from ....tools.checkpoint_utils_hf import load_model_config, save_model_config
from ...converter import Converter, ConverterFactory

__all__ = ["Qwen3_5Converter"]

_LANGUAGE_MODEL_PREFIX = "model.language_model."


@ConverterFactory.register_decorator("qwen3_5")
class Qwen3_5Converter(Converter):
    @staticmethod
    def create_block_configs_from_main_config(config) -> List[BlockConfig]:
        text_config = config.text_config if hasattr(config, "text_config") else config
        return [
            BlockConfig(
                attention=AttentionConfig(
                    no_op=False, num_key_value_heads=text_config.num_key_value_heads
                ),
                ffn=FFNConfig(no_op=False, intermediate_size=text_config.intermediate_size),
            ).to_dict()
            for _ in range(text_config.num_hidden_layers)
        ]

    @classmethod
    def convert_configs_in_dirs(
        cls, input_dir: Path, output_dir: Path, trust_remote_code: bool = False
    ):
        """Save text_config (not the full VLM config) so downstream code can access
        num_hidden_layers and other text-model fields directly."""
        config = load_model_config(input_dir, trust_remote_code=trust_remote_code)
        text_config = config.text_config if hasattr(config, "text_config") else config
        block_configs = cls.create_block_configs_from_main_config(config)
        out_config = copy.deepcopy(text_config)
        out_config.block_configs = block_configs
        save_model_config(out_config, output_dir)
        return out_config

    @staticmethod
    def convert_weight_name(name: str) -> str:
        """Remap VLM weight names to text-model paths.

        model.language_model.X  →  model.X
        All other names are unchanged (lm_head, etc.).
        """
        if name.startswith(_LANGUAGE_MODEL_PREFIX):
            return "model." + name[len(_LANGUAGE_MODEL_PREFIX) :]
        return name
