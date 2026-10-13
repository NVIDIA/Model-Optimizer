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
"""
Replacement library for loading models with layer replacements (AnyModel / sharded HF checkpoints).
"""
# mypy: ignore-errors

import copy
import json
import tempfile
from pathlib import Path
from typing import List, Optional

from immutabledict import immutabledict
from safetensors import safe_open
from transformers import PretrainedConfig, PreTrainedModel

from ..anymodel.converter import Converter
from ..tools.checkpoint_utils import SAFETENSORS_SUBBLOCKS_DIR_NAME, load_model_config
from ..tools.checkpoint_utils_hf import save_model_config
from ..tools.sharded_checkpoint_utils import load_and_shard_model
from .replacement_utils import (
    extract_block_configs_and_locations,
    parse_layer_replacement,
    weights_path_to_checkpoint_dir,
)

__all__ = [
    "ReplacementLibrary",
]


class ReplacementLibrary:
    def __init__(
        self,
        replacement_library_path: str | Path,
        descriptor,
        model_config_overrides: Optional[dict] = None,
    ):
        self.descriptor = descriptor
        self.replacement_library = self._load_replacement_library(replacement_library_path)
        self._ensure_all_checkpoints_are_split()
        self.model_config_overrides = (
            immutabledict(model_config_overrides) if (model_config_overrides is not None) else None
        )

        self._model_config = None
        self._arbitrary_checkpoint_dir = None

    @staticmethod
    def _load_replacement_library(replacement_library_path: str | Path) -> list[dict]:
        replacement_library = json.loads(Path(replacement_library_path).read_text())
        replacement_library = [
            parse_layer_replacement(layer_replacement) for layer_replacement in replacement_library
        ]
        return replacement_library

    def _ensure_all_checkpoints_are_split(self) -> None:
        checkpoint_dirs = self._get_all_checkpoint_dirs()
        unsplit_checkpoints = []
        for checkpoint_dir in checkpoint_dirs:
            if not (checkpoint_dir / SAFETENSORS_SUBBLOCKS_DIR_NAME).exists():
                unsplit_checkpoints.append(checkpoint_dir)
        assert len(unsplit_checkpoints) == 0, f"Found unsplit checkpoints: {unsplit_checkpoints}"

    @property
    def model_config(self) -> PretrainedConfig:
        if self._model_config is None:
            trust_remote_code = self.descriptor.requires_trust_remote_code()
            self._model_config = load_model_config(
                self.get_arbitrary_checkpoint_dir(),
                self.model_config_overrides,
                ignore_unexpected_config_keys=True,
                trust_remote_code=trust_remote_code,
            )
        return self._model_config

    def create_model_config(self, layer_replacements: list[dict]):
        block_configs, _ = extract_block_configs_and_locations(layer_replacements)
        model_config = copy.deepcopy(self.model_config)
        model_config.block_configs = block_configs
        model_config.num_hidden_layers = len(block_configs)
        return model_config

    def _get_arbitrary_non_block_checkpoint_paths(self):
        checkpoint_dir = Path(self.get_arbitrary_checkpoint_dir())
        subblocks_dir = checkpoint_dir / SAFETENSORS_SUBBLOCKS_DIR_NAME
        non_block_paths = [p for p in subblocks_dir.glob("*.safetensors") if "block_" not in p.name]
        return non_block_paths

    def create_index_file_from_weights(self, weight_paths: List[str]):
        weight_map = {}
        for weight_path in weight_paths:
            weight_path = Path(weight_path)
            with safe_open(str(weight_path), framework="pt", device="cpu") as f:
                for tensor_name in f.keys():
                    weight_map[tensor_name] = f"{SAFETENSORS_SUBBLOCKS_DIR_NAME}/{weight_path.name}"
        index = {"metadata": {"format": "pt"}, "weight_map": weight_map}
        return index

    def prepare_tmp_checkpoint_dir(
        self,
        tmpdir: Path,
        model_config: PretrainedConfig,
        layer_replacements: List[dict],
    ):
        arbitrary_checkpoint_dir = Path(self.get_arbitrary_checkpoint_dir())

        weight_paths = self._get_arbitrary_non_block_checkpoint_paths()
        for layer_replacement in layer_replacements:
            weight_paths += layer_replacement["weight_paths"]

        weights_index = self.create_index_file_from_weights(weight_paths)
        index_path = tmpdir / "model.safetensors.index.json"
        with index_path.open("w", encoding="utf-8") as out:
            json.dump(weights_index, out, indent=2, sort_keys=True)

        Converter.copy_checkpoint_files(arbitrary_checkpoint_dir, tmpdir)
        save_model_config(model_config, tmpdir)

        # create symlinks inside tmpdir
        subblocks_dir = tmpdir / SAFETENSORS_SUBBLOCKS_DIR_NAME
        subblocks_dir.mkdir(exist_ok=True)
        for weight_path in weight_paths:
            link_path = subblocks_dir / weight_path.name
            link_path.symlink_to(weight_path)

    def load_model(
        self,
        layer_replacements: list[dict],
    ) -> PreTrainedModel:
        """Load model using AnyModel approach with temporary checkpoint directory."""
        model_config = self.create_model_config(layer_replacements)
        with tempfile.TemporaryDirectory(prefix="replacement_solution_") as tmpdir:
            tmpdir = Path(tmpdir)
            self.prepare_tmp_checkpoint_dir(
                tmpdir, model_config=model_config, layer_replacements=layer_replacements
            )
            model = load_and_shard_model(descriptor=self.descriptor, checkpoint_path=tmpdir)
        return model

    def get_arbitrary_checkpoint_dir(self) -> Path:
        if self._arbitrary_checkpoint_dir is None:
            self._arbitrary_checkpoint_dir = self._get_arbitrary_checkpoint_dir()
        return self._arbitrary_checkpoint_dir

    def _get_arbitrary_checkpoint_dir(self) -> Path:
        for layer_replacement in self.replacement_library:
            weight_paths = layer_replacement["weight_paths"]
            if len(weight_paths) > 0:
                return weights_path_to_checkpoint_dir(weight_paths[0])

    def _get_all_checkpoint_dirs(self) -> list[Path]:
        checkpoint_dirs = set()
        for layer_replacement in self.replacement_library:
            weight_paths = layer_replacement["weight_paths"]
            for weights_path in weight_paths:
                checkpoint_dir = weights_path_to_checkpoint_dir(weights_path)
                checkpoint_dirs.add(checkpoint_dir)
        return list(checkpoint_dirs)
