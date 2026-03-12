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
Replacement library for efficiently loading and managing layer-replaced DeciLM models.
- Uses replacement_utils for parsing, sorting, and analyzing layer replacement configurations
"""
# mypy: ignore-errors

import copy
import json
import re
import tempfile
from pathlib import Path
from typing import List, Optional

import torch
from immutabledict import immutabledict
from lru import LRU
from safetensors import safe_open
from safetensors.torch import load_file as safe_load_file
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel

import modelopt.torch.utils.distributed as dist
from modelopt.torch.puzzletron.anymodel.converter.converter import Converter
from modelopt.torch.puzzletron.decilm.deci_lm_hf_code.configuration_decilm import DeciLMConfig
from modelopt.torch.puzzletron.decilm.deci_lm_hf_code.modeling_decilm import (
    DeciLMDecoderLayer,
    DeciLMForCausalLM,
    DeciLMMultiDecoderLayer,
    DeciLMRMSNorm,
    LMHead,
)
from modelopt.torch.puzzletron.replacement_library.replacement_utils import (
    extract_block_configs_and_locations,
    parse_layer_replacement,
    sort_replacements,
    weights_path_to_checkpoint_dir,
)
from modelopt.torch.puzzletron.tools.checkpoint_utils import (
    PTH_SUBBLOCKS_DIR_NAME,
    SAFETENSORS_SUBBLOCKS_DIR_NAME,
    infer_weights_dtype,
    init_empty_module,
    init_module_with_state_dict,
    load_model_config,
)
from modelopt.torch.puzzletron.tools.checkpoint_utils_hf import save_model_config
from modelopt.torch.puzzletron.tools.sharded_checkpoint_utils import (
    create_dummy_model,
    is_in_safetensors_format,
    load_and_shard_model,
    load_sharded_state_dict,
)


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

        self._loaded_replacements: dict[str, nn.ModuleList] = LRU(
            size=256
        )  # least-recently-used dict: a dict of fixed size that evicts old items

        self._dtype = None

        self.teacher_dir = Path(replacement_library_path).parent / "ckpts" / "teacher"
        self._model_config = None
        self._embedding = None
        self._ln_f = None
        self._lm_head = None
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
    def dtype(self) -> torch.dtype:
        if self._dtype is None:
            ln_f = self.get_ln_f()
            self._dtype = ln_f.weight.dtype
        return self._dtype

    @property
    def n_layer(self) -> int:
        return self.model_config.get_num_hidden_layers()

    @property
    def model_config(self) -> DeciLMConfig:
        if self._model_config is None:
            self._model_config = load_model_config(
                self.get_arbitrary_checkpoint_dir(),
                self.model_config_overrides,
                ignore_unexpected_config_keys=True,
            )
        return self._model_config

    def create_model_config(self, layer_replacements: list[dict]):
        block_configs, _ = extract_block_configs_and_locations(layer_replacements)
        model_config = copy.deepcopy(self.model_config)
        model_config.block_configs = block_configs
        model_config.num_hidden_layers = len(block_configs)
        return model_config

    def _get_arbitrary_block_checkpoint_paths(self):
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

        weight_paths = self._get_arbitrary_block_checkpoint_paths()
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

    def load_checkpoint(self, checkpoint_dir: str | Path) -> PreTrainedModel:
        checkpoint_dir = Path(checkpoint_dir).resolve()
        layer_replacements = self._locate_replacements_of_entire_checkpoint(checkpoint_dir)
        model = self.load_model(layer_replacements)
        return model

    def _locate_replacements_of_entire_checkpoint(self, checkpoint_dir: str | Path) -> list[dict]:
        weight_paths_located = []
        layer_replacements = []
        for layer_replacement in self.replacement_library:
            weight_paths = layer_replacement["weight_paths"]
            weight_paths = [Path(p).absolute().resolve() for p in weight_paths]
            layer_replacement["weight_paths"] = weight_paths
            if len(weight_paths) > 0 and all(
                p.is_relative_to(checkpoint_dir) for p in weight_paths
            ):
                layer_replacements.append(layer_replacement)
                weight_paths_located.extend(weight_paths)

        all_block_weight_paths = [
            p
            for p in list((checkpoint_dir / SAFETENSORS_SUBBLOCKS_DIR_NAME).iterdir())
            if p.name not in ("embeddings.safetensors", "lm_head.safetensors")
        ]
        missing_paths = set(all_block_weight_paths) - set(weight_paths_located)
        assert len(missing_paths) == 0, (
            f"Couldn't locate replacements for the entire checkpoint {checkpoint_dir}, missing weights: {missing_paths}"
        )

        dedupped_layer_replacements = []
        for weights_path in all_block_weight_paths:
            replacements_with_path = [
                rep for rep in layer_replacements if weights_path in rep["weight_paths"]
            ]
            largets_replacement_with_path = max(
                replacements_with_path, key=lambda rep: len(rep["weight_paths"])
            )
            if largets_replacement_with_path not in dedupped_layer_replacements:
                dedupped_layer_replacements.append(largets_replacement_with_path)

        dedupped_layer_replacements = sort_replacements(dedupped_layer_replacements)
        return dedupped_layer_replacements

    def get_block(
        self, layer_replacement: dict, block_idx_in_replacement: int
    ) -> DeciLMDecoderLayer | DeciLMMultiDecoderLayer:
        if str(layer_replacement) not in self._loaded_replacements.keys():
            self._loaded_replacements[str(layer_replacement)] = self._load_layer_replacement(
                layer_replacement
            )
        module_list = self._loaded_replacements[str(layer_replacement)]
        block = module_list[block_idx_in_replacement]
        return block

    def _load_layer_replacement(self, layer_replacement: dict) -> nn.ModuleList:
        state_dict = dict()
        for weights_path in layer_replacement["weight_paths"]:
            if weights_path.suffix == ".safetensors":
                curr_state_dict = safe_load_file(weights_path)
            elif weights_path.suffix == ".pth":
                curr_state_dict = torch.load(weights_path, weights_only=True)
            else:
                raise ValueError(f"Unrecognized suffix of {weights_path=}")
            for param_name in curr_state_dict.keys():
                assert param_name not in state_dict, (
                    f"Duplicate entries for {param_name=} in {layer_replacement=}"
                )
            state_dict.update(curr_state_dict)

        if len(state_dict) > 0:
            block_indices = [
                int(re.findall(r"^model\.layers\.(\d+)\.", param_name)[0])
                for param_name in state_dict.keys()
            ]
            assert sorted(set(block_indices)) == list(
                range(min(block_indices), max(block_indices) + 1)
            ), (
                f"Block indices in loaded weight files must be consecutive, but found {sorted(set(block_indices))} in {layer_replacement=}"
            )

            min_block_idx = min(block_indices)

            state_dict = {
                param_name.replace(
                    f"model.layers.{block_idx}.", f"{block_idx - min_block_idx}."
                ): param_weight
                for block_idx, (param_name, param_weight) in zip(block_indices, state_dict.items())
            }

        dtype = infer_weights_dtype(state_dict)
        model_config = copy.deepcopy(self.model_config)
        model_config.block_configs = layer_replacement["child_block_configs"]
        model_config.num_hidden_layers = len(layer_replacement["child_block_configs"])

        module_list = nn.ModuleList(
            [
                (
                    init_empty_module(DeciLMDecoderLayer, dtype, model_config, layer_idx)
                    if (block_config.parallel_blocks is None)
                    else init_empty_module(DeciLMMultiDecoderLayer, dtype, model_config, layer_idx)
                )
                for layer_idx, block_config in enumerate(layer_replacement["child_block_configs"])
            ]
        )

        module_list.load_state_dict(state_dict, strict=True)
        return module_list

    def _move_inactive_blocks_to_cpu(self, active_blocks: list[nn.Module]) -> None:
        for module_list in self._loaded_replacements.values():
            for module in module_list:
                if module not in active_blocks:
                    module.to("cpu")

    def get_embedding(self) -> nn.Embedding:
        if self._embedding is None:
            state_dict = {
                "weight": self._get_arbitrary_non_block_param(
                    self.model_config.get_embedding_layer_name() + ".weight"
                )
            }
            self._embedding = init_module_with_state_dict(
                state_dict,
                nn.Embedding,
                num_embeddings=self.model_config.vocab_size,
                embedding_dim=self.model_config.hidden_size,
            )
        return self._embedding

    def get_ln_f(self) -> DeciLMRMSNorm:
        if self._ln_f is None:
            state_dict = {
                "weight": self._get_arbitrary_non_block_param(
                    self.model_config.get_final_layer_norm_layer_name() + ".weight"
                )
            }
            self._ln_f = init_module_with_state_dict(
                state_dict,
                DeciLMRMSNorm,
                hidden_size=self.model_config.hidden_size,
                eps=self.model_config.rms_norm_eps,
            )
        return self._ln_f

    def get_lm_head(self) -> nn.Linear:
        if self._lm_head is None:
            state_dict = {
                "weight": self._get_arbitrary_non_block_param(
                    self.model_config.get_lm_head_layer_name() + ".weight"
                )
            }
            self._lm_head = init_module_with_state_dict(
                state_dict,
                LMHead,
                out_features=self.model_config.vocab_size,
                in_features=self.model_config.hidden_size,
                bias=False,
            )
        return self._lm_head

    def _get_arbitrary_non_block_param(self, param_name: str) -> torch.Tensor:
        checkpoint_dir = self.get_arbitrary_checkpoint_dir()
        if (
            is_in_safetensors_format(checkpoint_dir)
            or (checkpoint_dir / SAFETENSORS_SUBBLOCKS_DIR_NAME).exists()
        ):
            partial_state_dict = load_sharded_state_dict(checkpoint_dir, [param_name])
            return partial_state_dict[param_name]

        non_block_pth_path = checkpoint_dir / PTH_SUBBLOCKS_DIR_NAME / f"non_block.pth"
        assert non_block_pth_path.exists(), _error_message_ensure_split(checkpoint_dir)
        non_block_state_dict = torch.load(non_block_pth_path)
        return non_block_state_dict[param_name]

    def get_arbitrary_checkpoint_dir(self) -> Path:
        if self._arbitrary_checkpoint_dir is None:
            self._arbitrary_checkpoint_dir = self._get_arbitrary_checkpoint_dir()
        return self._arbitrary_checkpoint_dir

    def get_teacher_dir(self) -> Path:
        return self.teacher_dir

    def get_teacher_lm_head_path(self) -> Path:
        return self.get_teacher_dir() / SAFETENSORS_SUBBLOCKS_DIR_NAME / "lm_head.safetensors"

    def get_teacher_embedding_path(self) -> Path:
        return self.get_teacher_dir() / SAFETENSORS_SUBBLOCKS_DIR_NAME / "embeddings.safetensors"

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


def _error_message_ensure_split(checkpoint_dir: Path) -> str:
    return (
        f"Encountered unsplit checkpoint dir '{checkpoint_dir}', "
        f"please call `ensure_all_checkpoints_are_split`"
    )


def _get_owned_block_indexes(n_layer: int) -> list[int]:
    last_process_blocks = np.array([n_layer - 1])  # less params in last gpu, leave room for logits

    if dist.size() == 1:
        # Only one process: assign everything (including the "last process" block) to rank 0
        owned_block_indexes_per_process = [
            np.concatenate([np.arange(n_layer - 1), last_process_blocks])
        ]
    else:
        # Multiple processes: split n_layer-1 blocks, reserve the last for "last process"
        owned_block_indexes_per_process = np.array_split(range(n_layer - 1), dist.size() - 1)
        owned_block_indexes_per_process.append(last_process_blocks)

    owned_block_indexes = owned_block_indexes_per_process[dist.rank()].tolist()
    return owned_block_indexes
