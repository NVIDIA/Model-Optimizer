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

The module is pure Pydantic schema with no runtime dependencies on ``transformers``,
``torch``, or ``accelerate`` — distributed-environment resolution (``WORLD_SIZE`` lookup,
``ParallelismConfig`` construction) is the caller's responsibility, see
``init_distributed_env`` in ``examples/speculative_decoding/main.py``.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator


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
