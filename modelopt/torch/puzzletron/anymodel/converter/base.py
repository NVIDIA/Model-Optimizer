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
import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from transformers import PretrainedConfig

from ...block_config import BlockConfig
from ...tools.checkpoint_utils_hf import load_model_config, save_model_config
from ..model_descriptor import ModelDescriptor

__all__ = ["Converter"]


class Converter(ABC):
    """Base class for adding AnyModel metadata to HuggingFace checkpoints."""

    @classmethod
    def convert_configs_in_dirs(
        cls,
        input_dir: Path,
        output_dir: Path,
        trust_remote_code: bool = False,
        descriptor: ModelDescriptor | None = None,
    ):
        """Attach typed block_configs and save standard HuggingFace config artifacts."""
        config = load_model_config(input_dir, trust_remote_code=trust_remote_code)

        block_configs = cls.create_block_configs_from_main_config(config)
        out_config = copy.deepcopy(config)
        if descriptor is None:
            out_config.block_configs = block_configs
        else:
            descriptor.set_block_configs(out_config, block_configs)

        save_model_config(out_config, output_dir)
        return out_config

    @staticmethod
    def copy_checkpoint_files(input_dir: Path, output_dir: Path):
        """Materialize a checkpoint while preserving standard HuggingFace weight files.

        Metadata/config files are copied so Puzzletron can rewrite config.json
        without mutating the source checkpoint. Large weight shards are hardlinked
        when possible and copied otherwise. This keeps 100B+ smoke conversions
        fast when the HF cache and run directory share a filesystem, while still
        producing self-contained checkpoints when they do not.
        """
        ignore_patterns = [
            "subblocks",
            "subblocks_safetensors",
        ]

        def should_ignore(path: Path) -> bool:
            rel_parts = path.relative_to(input_dir).parts
            return any(
                fnmatch.fnmatch(part, pattern)
                for part in rel_parts
                for pattern in ignore_patterns
            )

        def is_weight_file(path: Path) -> bool:
            return path.name.endswith((".safetensors", ".bin"))

        def materialize_file(src: Path, dst: Path) -> None:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists() or dst.is_symlink():
                try:
                    if dst.stat().st_size == src.stat().st_size:
                        return
                except OSError:
                    pass
                dst.unlink()

            if is_weight_file(src):
                # HF snapshots often store weights as relative symlinks into a
                # cache-level blobs/ directory. Never reproduce that relative
                # symlink in Puzzletron's teacher dir; it points at the wrong
                # location after conversion. Link/copy the resolved blob.
                real_src = src.resolve(strict=True)
                try:
                    os.link(real_src, dst)
                    return
                except OSError:
                    pass
                shutil.copy2(real_src, dst)
                return
            shutil.copy2(src, dst)

        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for src in input_dir.rglob("*"):
            if should_ignore(src):
                continue
            dst = output_dir / src.relative_to(input_dir)
            if src.is_dir():
                dst.mkdir(parents=True, exist_ok=True)
            elif src.is_file():
                materialize_file(src, dst)

    @classmethod
    def convert(
        cls,
        descriptor: ModelDescriptor,
        input_dir: Path,
        output_dir: Path,
    ):
        """Attach AnyModel block configs to a standard HuggingFace checkpoint.

        The normal conversion path copies the original HF checkpoint files,
        including safetensors and index files, then writes config artifacts with
        typed block_configs. It does not reorganize model weights.

        Args:
            descriptor: Model descriptor for the model type.
            input_dir: Path to the input HuggingFace checkpoint.
            output_dir: Path to the output AnyModel checkpoint.
        """
        cls.copy_checkpoint_files(input_dir, output_dir)
        trust_remote_code = descriptor.requires_trust_remote_code()
        cls.convert_configs_in_dirs(
            input_dir, output_dir, trust_remote_code=trust_remote_code, descriptor=descriptor
        )

    @staticmethod
    @abstractmethod
    def create_block_configs_from_main_config(config: PretrainedConfig) -> List[BlockConfig]:
        """Create per-layer BlockConfig list from a HuggingFace model config.

        This method extracts layer-specific parameters (e.g., intermediate_size,
        num_kv_heads) from the main model config and creates a BlockConfig
        for each layer. These BlockConfigs enable layer-specific pruning and
        modifications during the compression pipeline.

        Args:
            config: HuggingFace PretrainedConfig (e.g., LlamaConfig, Qwen2Config)

        Returns:
            List of BlockConfig, one per hidden layer. Each BlockConfig contains
            typed subblock_configs, for example AttentionConfig, FFNConfig,
            MoEConfig, or MambaConfig entries.

        Example:
            For a model with uniform layers (e.g., Llama):
                return [BlockConfig(...)] * config.num_hidden_layers

            For a model with heterogeneous layers (e.g., NemotronH with Mamba/Attention):
                return [BlockConfig(...) for layer_idx in range(num_layers)]
        """
        raise NotImplementedError
