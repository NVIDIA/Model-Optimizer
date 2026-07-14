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
"""Replacement library for virtual candidates backed by one sorted teacher checkpoint."""
# mypy: ignore-errors

import copy
import json
from pathlib import Path
from typing import Optional

from immutabledict import immutabledict
from transformers import PretrainedConfig

from ..tools.checkpoint_utils_hf import load_model_config
from .replacement_utils import extract_block_configs_and_locations, parse_layer_replacement

__all__ = [
    "ReplacementLibrary",
]


class ReplacementLibrary:
    def __init__(
        self,
        replacement_library_path: str | Path,
        descriptor,
        model_config_overrides: Optional[dict] = None,
        sorted_teacher_dir: str | Path | None = None,
    ):
        self.descriptor = descriptor
        (
            self.replacement_library,
            lib_sorted_teacher_dir,
            self.hidden_width,
            self.teacher_hidden_width,
        ) = self._load_replacement_library(replacement_library_path)
        effective_sorted_teacher_dir = (
            sorted_teacher_dir if sorted_teacher_dir is not None else lib_sorted_teacher_dir
        )
        self.sorted_teacher_dir = (
            Path(effective_sorted_teacher_dir) if effective_sorted_teacher_dir is not None else None
        )
        if self.sorted_teacher_dir is None:
            raise ValueError(
                "ReplacementLibrary requires a v2 library with sorted_teacher_dir. "
                "Realized subblock checkpoint libraries are no longer supported."
            )
        self.model_config_overrides = (
            immutabledict(model_config_overrides) if (model_config_overrides is not None) else None
        )

        self._model_config = None

    @staticmethod
    def _load_replacement_library(
        replacement_library_path: str | Path,
    ) -> tuple[list[dict], Path | None, int | None, int | None]:
        """Parse a modern v2 replacement library JSON."""
        raw = json.loads(Path(replacement_library_path).read_text())
        if not isinstance(raw, dict) or raw.get("version") != 2:
            raise ValueError(
                "Replacement libraries must use v2 JSON: "
                '{"version": 2, "sorted_teacher_dir": "...", "entries": [...]}'
            )
        st = raw.get("sorted_teacher_dir")
        if not st:
            raise ValueError("Replacement library v2 JSON is missing sorted_teacher_dir")
        sorted_teacher_dir = Path(st)
        entries = raw.get("entries", [])
        for entry in entries:
            entry.setdefault("weight_paths", [])
        entries = [parse_layer_replacement(e) for e in entries]
        hidden_width = raw.get("hidden_width")
        teacher_hidden_width = raw.get("teacher_hidden_width")
        return (
            entries,
            sorted_teacher_dir,
            None if hidden_width is None else int(hidden_width),
            None if teacher_hidden_width is None else int(teacher_hidden_width),
        )

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
        self.descriptor.set_block_configs(model_config, block_configs)
        return model_config

    def load_model(
        self,
        layer_replacements: list[dict],
    ):
        """Build a variant model by slicing/merging from the sorted teacher."""
        model_config = self.create_model_config(layer_replacements)
        from ..pruning.materialize import materialize_model_from_sorted

        return materialize_model_from_sorted(
            self.sorted_teacher_dir, layer_replacements, self.descriptor, model_config
        )

    def materialize_checkpoint(
        self,
        layer_replacements: list[dict],
        output_dir: str | Path,
        *,
        model_config: PretrainedConfig | None = None,
        overwrite: bool = False,
        solution_identity: str | None = None,
    ) -> Path:
        """Stream a realized child checkpoint without constructing a dense model."""

        from ..pruning.materialize import materialize_checkpoint_from_sorted

        model_config = model_config or self.create_model_config(layer_replacements)
        return materialize_checkpoint_from_sorted(
            self.sorted_teacher_dir,
            layer_replacements,
            self.descriptor,
            model_config,
            output_dir,
            overwrite=overwrite,
            solution_identity=solution_identity,
        )

    def get_arbitrary_checkpoint_dir(self) -> Path:
        return self.sorted_teacher_dir
