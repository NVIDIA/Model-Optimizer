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
"""This module provides helper functions for parsing, sorting, and analyzing layer replacement
configurations used in the replacement library for model compression.
"""

# mypy: ignore-errors
import json
from copy import deepcopy
from pathlib import Path

from modelopt.torch.puzzletron.decilm.deci_lm_hf_code.block_config import BlockConfig
from modelopt.torch.puzzletron.decilm.deci_lm_hf_code.configuration_decilm import DeciLMConfig
from modelopt.torch.puzzletron.mip.utils import sort_replacements


def parse_layer_replacement(layer_replacement: dict | str) -> dict:
    if isinstance(layer_replacement, str):
        layer_replacement = json.loads(layer_replacement)
    else:
        layer_replacement = deepcopy(layer_replacement)

    if "layer_replacement" in layer_replacement:  # happens in puzzle solutions
        layer_replacement = layer_replacement["layer_replacement"]

    layer_replacement["child_block_configs"] = [
        BlockConfig(**block_config) if isinstance(block_config, dict) else block_config
        for block_config in layer_replacement["child_block_configs"]
    ]
    layer_replacement["weight_paths"] = [Path(p) for p in layer_replacement["weight_paths"]]
    return layer_replacement


# sort_replacements moved to modelopt.torch.puzzletron.mip.utils and imported above


def extract_block_configs_and_locations(
    layer_replacements: list[dict],
) -> tuple[list[BlockConfig], list[tuple[dict, int]]]:
    layer_replacements = sort_replacements(layer_replacements)
    block_configs = []
    block_locations = []
    for layer_replacement in layer_replacements:
        child_block_configs = layer_replacement["child_block_configs"]
        if not isinstance(child_block_configs, list | tuple):
            child_block_configs = [child_block_configs]
        for block_idx_in_replacement, block_config in enumerate(child_block_configs):
            block_configs.append(block_config)
            block_locations.append((layer_replacement, block_idx_in_replacement))
    return block_configs, block_locations


def weights_path_to_checkpoint_dir(weights_path: Path) -> Path:
    checkpoint_dir: Path = weights_path
    while checkpoint_dir != Path("/"):
        if (checkpoint_dir / "config.json").exists():
            return checkpoint_dir
        checkpoint_dir = checkpoint_dir.parent
    raise FileNotFoundError(f"Couldn't find checkpoint dir for weights path {weights_path}")


def replacement_is_teacher(
    layer_replacement: dict,
    teacher_model_config: DeciLMConfig,
    teacher_checkpoint_dir: Path,
) -> bool:
    paths_all_teacher = all(
        p.is_relative_to(teacher_checkpoint_dir) for p in layer_replacement["weight_paths"]
    )
    return paths_all_teacher and is_replacement_identical_to_teacher(
        layer_replacement, teacher_model_config
    )


def is_replacement_identical_to_teacher(
    layer_replacement: dict,
    teacher_model_config: DeciLMConfig,
) -> bool:
    if len(layer_replacement["parent_layer_indices"]) == 1:
        block_idx = layer_replacement["parent_layer_indices"][0]
        teacher_block_config = teacher_model_config.block_configs[block_idx]
        if len(child_block_configs := layer_replacement["child_block_configs"]) == 1:
            replacement_block_config: BlockConfig = child_block_configs[0]
            if replacement_block_config == teacher_block_config:
                return True
            else:
                parallel_blocks = getattr(replacement_block_config, "parallel_blocks", None)
                if (
                    parallel_blocks is not None
                    and len(parallel_blocks) == 1
                    and parallel_blocks[0].attention == teacher_block_config.attention
                    and parallel_blocks[0].ffn == teacher_block_config.ffn
                ):
                    return True
    return False


def split_replacements_to_teacher_and_student(
    replacements: list[dict],
    teacher_model_config: DeciLMConfig,
    teacher_checkpoint_dir: Path,
) -> tuple[list[dict], list[dict]]:
    teacher_replacements, student_replacements = [], []
    for replacement in replacements:
        if replacement_is_teacher(replacement, teacher_model_config, teacher_checkpoint_dir):
            teacher_replacements.append(replacement)
        else:
            student_replacements.append(replacement)
    return teacher_replacements, student_replacements
