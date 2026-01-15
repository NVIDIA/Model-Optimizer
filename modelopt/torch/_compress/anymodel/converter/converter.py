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
import fnmatch
import json
import os
import shutil
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from safetensors.torch import load_file, save_file
from tqdm import tqdm
from transformers import PretrainedConfig

from modelopt.torch._compress.anymodel.model_descriptor import ModelDescriptor
from modelopt.torch._compress.decilm.deci_lm_hf_code.block_config import BlockConfig
from modelopt.torch._compress.tools.checkpoint_utils_hf import load_model_config, save_model_config

__all__ = ["Converter"]


class Converter(ABC):
    """Base class for converting HuggingFace models to Puzzletron/AnyModel format."""

    @staticmethod
    def _get_weight_map(input_dir: Path) -> Dict[str, str]:
        """Load weight map from checkpoint directory (supports both sharded and single-file models).

        Returns a dict mapping parameter names to their safetensors filenames.
        """
        index_path = input_dir / "model.safetensors.index.json"
        single_file_path = input_dir / "model.safetensors"

        if index_path.exists():
            # Sharded model
            with open(index_path, "r") as f:
                index = json.load(f)
            return index["weight_map"]
        elif single_file_path.exists():
            # Single file model - create a synthetic weight map
            data = load_file(single_file_path)
            return {name: "model.safetensors" for name in data.keys()}
        else:
            raise FileNotFoundError(
                f"Neither {index_path} nor {single_file_path} found. Cannot determine model format."
            )

    @staticmethod
    def convert_model_weights(
        input_dir: Path, output_dir: Path, descriptor: ModelDescriptor, num_hidden_layers: int
    ):
        """Convert model weights to subblock format."""
        param_to_file = Converter._get_weight_map(input_dir)
        all_param_names = list(param_to_file.keys())

        # Reverse map: file -> set of params
        file_to_params = defaultdict(set)
        for name, file in param_to_file.items():
            file_to_params[file].add(name)

        # Determine subblocks needed
        subblocks = descriptor.get_weight_groups(
            all_param_names, num_hidden_layers=num_hidden_layers
        )

        # Output directory
        out_dir = output_dir / "subblocks_safetensors"
        os.makedirs(out_dir, exist_ok=True)

        # New weight index
        new_index = {"metadata": {"format": "pt"}, "weight_map": {}}

        for subblock, param_names in tqdm(subblocks.items(), desc="Processing subblocks"):
            param_files = set(param_to_file[name] for name in param_names)
            tensors = {}

            # Load only needed files for this subblock
            for file in param_files:
                data = load_file(os.path.join(input_dir, file))
                for name in param_names:
                    if param_to_file[name] == file and name in data:
                        tensors[name] = data[name]

            # Save this subblock
            print(f"\n✅ Group: {subblock} ({len(tensors)} layers)")
            for layer in tensors.keys():
                print(f"  - {layer}")

            subblock_file = f"{subblock}.safetensors"
            save_file(tensors, os.path.join(out_dir, subblock_file))

            # Update index
            for new_name in tensors.keys():
                new_index["weight_map"][new_name] = f"subblocks_safetensors/{subblock_file}"

        # Save new index file
        with (output_dir / "model.safetensors.index.json").open("w") as f:
            json.dump(new_index, f, indent=2)

        print(f"✅ Finished saving subblocks and index to {output_dir}")

    @classmethod
    def convert_configs_in_dirs(
        cls,
        input_dir: Path,
        output_dir: Path,
    ):
        """Convert config and add block_configs."""
        config = load_model_config(input_dir)

        block_configs = cls.create_block_configs_from_main_config(config)
        out_config = copy.deepcopy(config)
        out_config.block_configs = block_configs

        save_model_config(out_config, output_dir)
        return out_config

    @staticmethod
    def copy_checkpoint_files(input_dir: Path, output_dir: Path):
        """Copy checkpoint files except model weights (which will be converted)."""
        ignore_patterns = [
            "model-*.safetensors",
            "model.safetensors",
            "model.safetensors.index.json",
            "subblocks_safetensors",
        ]

        def ignore_func(dir, files):
            ignored = set()
            for pattern in ignore_patterns:
                ignored.update(fnmatch.filter(files, pattern))
            return ignored

        shutil.copytree(str(input_dir), str(output_dir), ignore=ignore_func, dirs_exist_ok=True)

    @classmethod
    def convert(
        cls,
        descriptor: ModelDescriptor,
        input_dir: Path,
        output_dir: Path,
    ):
        """Convert a HuggingFace model to AnyModel format.

        Args:
            descriptor: Model descriptor for the model type.
            input_dir: Path to the input HuggingFace checkpoint.
            output_dir: Path to the output AnyModel checkpoint.
        """
        cls.copy_checkpoint_files(input_dir, output_dir)
        config = cls.convert_configs_in_dirs(input_dir, output_dir)
        cls.convert_model_weights(
            input_dir, output_dir, descriptor=descriptor, num_hidden_layers=config.num_hidden_layers
        )

    @staticmethod
    @abstractmethod
    def create_block_configs_from_main_config(config: PretrainedConfig) -> List[BlockConfig]:
        """Create per-layer BlockConfig list from a HuggingFace model config.

        This method extracts layer-specific parameters (e.g., intermediate_size,
        num_key_value_heads) from the main model config and creates a BlockConfig
        for each layer. These BlockConfigs enable layer-specific pruning and
        modifications during the compression pipeline.

        Args:
            config: HuggingFace PretrainedConfig (e.g., LlamaConfig, Qwen2Config)

        Returns:
            List of BlockConfig, one per hidden layer. Each BlockConfig contains:
            - AttentionConfig: attention settings (no_op, num_key_value_heads)
            - FFNConfig: FFN settings (no_op, intermediate_size)

        Example:
            For a model with uniform layers (e.g., Llama):
                return [BlockConfig(...)] * config.num_hidden_layers

            For a model with heterogeneous layers (e.g., NemotronH with Mamba/Attention):
                return [BlockConfig(...) for layer_idx in range(num_layers)]
        """
        raise NotImplementedError
