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

"""Pydantic schemas for HF-trainer-based speculative-decoding experiments.

These are the typed section models used inside speculative-decoding recipes
(:class:`modelopt.recipe.config.ModelOptEagleRecipe` /
:class:`modelopt.recipe.config.ModelOptDFlashRecipe`). They mirror the HF dataclasses used
by :mod:`examples/speculative_decoding/main.py` so that recipe YAMLs are Pydantic-validated
at load time.

The module intentionally does NOT import ``transformers`` — it is pure Pydantic schema.
``transformers.TrainingArguments`` is extended separately in the example script to stay
compatible with HF's ``Trainer`` API; the ``TrainingArguments`` model here only declares the
seven speculative-decoding extension fields plus ``extra='allow'`` so HF trainer fields
(learning_rate, num_train_epochs, ...) flow through untouched.

``TrainingArguments`` does read ``WORLD_SIZE`` and ``torch.cuda.device_count()`` at validation
time to auto-fill ``dp_shard_size`` and derive a ``parallelism_config`` (accelerate's
``ParallelismConfig``) when the run is actually distributed. ``torch`` and ``accelerate`` are
imported lazily from within the validator so importing this module stays cheap and
``accelerate`` only becomes a hard requirement when ``cp_size>1`` or ``dp_shard_size>1``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class ModelArguments(BaseModel):
    """Arguments for loading the base HF model."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_name_or_path: str | None = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    use_fake_base_for_offline: bool = False
    trust_remote_code: bool = False


class DataArguments(BaseModel):
    """Arguments for the training dataset."""

    model_config = ConfigDict(extra="forbid")

    data_path: str | None = None
    offline_data_path: str | None = None
    lazy_preprocess: bool = True
    draft_vocab_cache: str | None = None
    chat_template: str | None = None
    vlm_img_dir: str | None = None
    vlm_processor: str | None = None
    sample_size: int = -1

    @field_validator("sample_size")
    @classmethod
    def _check_sample_size(cls, v: int) -> int:
        if v == 0 or v < -1:
            raise ValueError("sample_size must be -1 (use all samples) or a positive integer")
        return v


class TrainingArguments(BaseModel):
    """Speculative-decoding extensions on top of ``transformers.TrainingArguments``.

    HF trainer fields (``learning_rate``, ``num_train_epochs``, ...) flow through as extras
    via ``extra='allow'`` — they're re-validated later when the dict is passed to
    ``HfTrainingArguments(**recipe.training.model_dump())`` in main.py.
    """

    model_config = ConfigDict(extra="allow")

    training_seq_len: int = 2048
    estimate_ar: bool = False
    ar_validate_steps: int = 1000
    answer_only_loss: bool = False
    cp_size: int = 1
    dp_shard_size: int | None = None
    # Derived at validation time from cp_size/dp_shard_size/WORLD_SIZE; typed as Any so this
    # module doesn't need to import accelerate.ParallelismConfig just to annotate the field.
    parallelism_config: Any = None

    @model_validator(mode="after")
    def _fill_parallelism(self) -> TrainingArguments:
        # Read WORLD_SIZE (set by torchrun/accelerate, multi-node aware); fall back to the
        # local GPU count for single-process runs.
        import os

        import torch

        world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
        if self.dp_shard_size is None:
            self.dp_shard_size = world_size // self.cp_size

        # Build a ParallelismConfig only when actually running distributed — matches the
        # previous main.py guard and avoids requiring accelerate on single-GPU dev boxes.
        if self.cp_size > 1 or self.dp_shard_size > 1:
            parallel_size = self.dp_shard_size * self.cp_size
            if world_size % parallel_size != 0:
                raise ValueError(
                    f"world_size ({world_size}) must be divisible by "
                    f"dp_shard_size ({self.dp_shard_size}) * cp_size ({self.cp_size}) "
                    f"= {parallel_size}"
                )
            try:
                from accelerate import ParallelismConfig
            except ImportError as e:
                raise ImportError(
                    "cp_size>1 or dp_shard_size>1 requires `accelerate` for ParallelismConfig. "
                    "Install it via `pip install accelerate`."
                ) from e
            self.parallelism_config = ParallelismConfig(
                cp_size=self.cp_size,
                dp_shard_size=self.dp_shard_size,
                dp_replicate_size=world_size // parallel_size,
            )
        return self
