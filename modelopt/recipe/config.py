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

"""ModelOpt's pydantic BaseModel for recipes."""

from __future__ import annotations

import warnings
from enum import Enum
from typing import Literal

from pydantic import Field, field_validator, model_validator

from modelopt.torch.opt.config import ModeloptBaseConfig, ModeloptField
from modelopt.torch.quantization.config import QuantizeConfig  # noqa: TC001
from modelopt.torch.speculative.config import DFlashConfig, EagleConfig, MedusaConfig
from modelopt.torch.speculative.plugins.hf_training_args import DataArguments as SpecDataArgs
from modelopt.torch.speculative.plugins.hf_training_args import ModelArguments as SpecModelArgs
from modelopt.torch.speculative.plugins.hf_training_args import (
    TrainingArguments as SpecTrainingArgs,
)


class RecipeType(str, Enum):
    """List of recipe types. See ``RECIPE_TYPE_TO_CLASS`` at the bottom for the schema mapping."""

    PTQ = "ptq"
    AUTO_QUANTIZE = "auto_quantize"
    SPECULATIVE_EAGLE = "speculative_eagle"
    SPECULATIVE_DFLASH = "speculative_dflash"
    SPECULATIVE_MEDUSA = "speculative_medusa"
    # QAT = "qat" # Not implemented yet, will be added in the future.


_DEFAULT_RECIPE_DESCRIPTION = "Model optimization recipe."


class RecipeMetadataConfig(ModeloptBaseConfig):
    """YAML shape of the recipe metadata section."""

    recipe_type: RecipeType = Field(
        title="Recipe type",
        description="The type of the recipe (e.g. PTQ).",
    )
    description: str = ModeloptField(
        default=_DEFAULT_RECIPE_DESCRIPTION,
        title="Description",
        description="Human-readable description of the recipe.",
    )


def _metadata_field(recipe_type: RecipeType):
    """Build the metadata Pydantic field with the recipe_type baked into the default."""
    return ModeloptField(
        default={"recipe_type": recipe_type, "description": _DEFAULT_RECIPE_DESCRIPTION},
        title="Metadata",
        description="Recipe metadata containing the recipe type and description.",
        validate_default=True,
    )


class ModelOptRecipeBase(ModeloptBaseConfig):
    """Base configuration class for model optimization recipes.

    If a layer name matches ``"*output_layer*"``, the attributes will be replaced with ``{"enable": False}``.
    """

    metadata: RecipeMetadataConfig = Field(
        title="Metadata",
        description="Recipe metadata containing the recipe type and description. "
        "Required: a recipe without a ``metadata`` section is rejected so that a "
        "missing section can't silently fall back to a default recipe type.",
    )

    @property
    def recipe_type(self) -> RecipeType:
        """Return the recipe type from metadata."""
        return self.metadata.recipe_type

    @property
    def description(self) -> str:
        """Return the recipe description from metadata."""
        return self.metadata.description


class ModelOptPTQRecipe(ModelOptRecipeBase):
    """Our config class for PTQ recipes."""

    quantize: QuantizeConfig = Field(
        title="PTQ config",
        description="PTQ config containing quant_cfg and algorithm. Required: a PTQ "
        "recipe without a ``quantize`` section is rejected so that a missing section "
        "can't silently fall back to the default INT8 config.",
    )


class AutoQuantizeConstraints(ModeloptBaseConfig):
    """Constraints passed to ``mtq.auto_quantize`` (matches its dict shape).

    Today only ``effective_bits`` is supported upstream. When new constraint
    keys land (e.g., ``cost_model`` / ``cost`` from PR #1497), add them as
    fields here so ``.model_dump(exclude_none=True)`` produces the dict
    upstream expects.
    """

    effective_bits: float = ModeloptField(
        default=4.8,
        title="Effective bits per weight",
        description="Average weight-storage bits target for the LP, in (0, 16].",
    )

    @field_validator("effective_bits")
    @classmethod
    def _validate_effective_bits(cls, v: float) -> float:
        if not (0 < v <= 16):
            raise ValueError(f"effective_bits must be in (0, 16], got {v}")
        return v


class AutoQuantizeConfig(ModeloptBaseConfig):
    """Schema for the ``auto_quantize`` block in an AutoQuantize recipe."""

    constraints: AutoQuantizeConstraints = Field(
        title="Search constraints + cost model",
        description="LP budget and cost model.",
    )

    candidate_formats: list[QuantizeConfig] = ModeloptField(
        default=[],
        title="Candidate quantization formats",
        description="Per-layer search space; each entry is a full QuantizeConfig. "
        "At least 2 entries required.",
    )

    method: Literal["gradient", "kl_div"] = ModeloptField(
        default="gradient",
        title="Sensitivity scoring method",
        description="'gradient' (Taylor + Fisher, needs labels) or 'kl_div' (no labels).",
    )

    num_score_steps: int = ModeloptField(
        default=128,
        title="Phase-3 scoring sample count",
        description="Number of batches for sensitivity scoring.",
    )

    disabled_layers: list[str] = ModeloptField(
        default=[],
        title="Excluded layer patterns",
        description="Glob patterns; matching layers are excluded from the search.",
    )

    kv_cache: QuantizeConfig | None = ModeloptField(
        default=None,
        title="KV cache QuantizeConfig (optional)",
        description="Optional full QuantizeConfig applied as a uniform post-step after the "
        "LP search. Typically uses ``$import: configs/ptq/units/kv_*`` for a built-in KV "
        "preset, or inlines a custom config. If omitted, the runtime --kv_cache_qformat "
        "CLI flag is used as a fallback.",
    )

    @field_validator("candidate_formats")
    @classmethod
    def _at_least_two_candidates(cls, v: list[QuantizeConfig]) -> list[QuantizeConfig]:
        if len(v) < 2:
            raise ValueError(
                "auto_quantize requires at least 2 candidate_formats. "
                "For uniform quantization, use a PTQ recipe instead."
            )
        return v


class ModelOptAutoQuantizeRecipe(ModelOptRecipeBase):
    """Our config class for AutoQuantize recipes."""

    metadata: RecipeMetadataConfig = _metadata_field(RecipeType.AUTO_QUANTIZE)

    auto_quantize: AutoQuantizeConfig = Field(
        title="AutoQuantize config",
        description="AutoQuantize search configuration. Required.",
    )


class ModelOptSpeculativeRecipeBase(ModelOptRecipeBase):
    """Base class for speculative-decoding recipes.

    Unlike PTQ, speculative-decoding is a training-time optimization: the draft head is trained
    with HF Trainer. We therefore bundle ``model`` / ``data`` / ``training`` sections into the
    recipe so a single YAML is the full experiment spec. Each section is a typed Pydantic model
    (see :mod:`modelopt.torch.speculative.plugins.hf_training_args`) so field typos and bad
    values are caught at recipe-load time; HF trainer fields pass through
    ``TrainingArguments`` via ``extra='allow'``.
    """

    model: SpecModelArgs = ModeloptField(
        default=SpecModelArgs(),
        title="HF model args",
        description="ModelArguments for the base HF model to train a draft head against.",
        validate_default=True,
    )
    data: SpecDataArgs = ModeloptField(
        default=SpecDataArgs(),
        title="HF data args",
        description="DataArguments for the training/offline dataset.",
        validate_default=True,
    )
    training: SpecTrainingArgs = ModeloptField(
        default=SpecTrainingArgs(),
        title="HF training args",
        description="Speculative-decoding extensions; HF trainer fields flow through as extras.",
        validate_default=True,
    )


class ModelOptEagleRecipe(ModelOptSpeculativeRecipeBase):
    """Our config class for EAGLE speculative decoding recipes."""

    metadata: RecipeMetadataConfig = _metadata_field(RecipeType.SPECULATIVE_EAGLE)

    eagle: EagleConfig = ModeloptField(
        default=EagleConfig(),
        title="EAGLE config",
        description="EAGLE speculative decoding configuration.",
        validate_default=True,
    )

    @model_validator(mode="after")
    def _derive_eagle_offline(self) -> ModelOptEagleRecipe:
        self.eagle.eagle_offline = self.data.offline_data_path is not None
        return self

    @model_validator(mode="after")
    def _warn_rope_vs_training_seq_len(self) -> ModelOptEagleRecipe:
        orig_max_pos = self.eagle.eagle_export_rope_scaling.get("original_max_position_embeddings")
        if orig_max_pos is not None and orig_max_pos != self.training.training_seq_len:
            warnings.warn(
                f"eagle.eagle_export_rope_scaling.original_max_position_embeddings ({orig_max_pos}) "
                f"differs from training.training_seq_len ({self.training.training_seq_len}). "
                f"This may affect long-context inference quality."
            )
        return self


class ModelOptDFlashRecipe(ModelOptSpeculativeRecipeBase):
    """Our config class for DFlash speculative decoding recipes."""

    metadata: RecipeMetadataConfig = _metadata_field(RecipeType.SPECULATIVE_DFLASH)

    dflash: DFlashConfig = ModeloptField(
        default=DFlashConfig(),
        title="DFlash config",
        description="DFlash speculative decoding configuration.",
        validate_default=True,
    )

    @model_validator(mode="after")
    def _derive_dflash_offline(self) -> ModelOptDFlashRecipe:
        self.dflash.dflash_offline = self.data.offline_data_path is not None
        return self


class ModelOptMedusaRecipe(ModelOptSpeculativeRecipeBase):
    """Our config class for Medusa speculative decoding recipes."""

    metadata: RecipeMetadataConfig = _metadata_field(RecipeType.SPECULATIVE_MEDUSA)

    medusa: MedusaConfig = ModeloptField(
        default=MedusaConfig(),
        title="Medusa config",
        description="Medusa speculative decoding configuration.",
        validate_default=True,
    )


# Single source of truth mapping YAML ``metadata.recipe_type`` to its schema class. The loader
# uses this for typed-list ``$import`` resolution; add a new entry when introducing a recipe.
RECIPE_TYPE_TO_CLASS: dict[RecipeType, type[ModelOptRecipeBase]] = {
    RecipeType.PTQ: ModelOptPTQRecipe,
    RecipeType.AUTO_QUANTIZE: ModelOptAutoQuantizeRecipe,
    RecipeType.SPECULATIVE_EAGLE: ModelOptEagleRecipe,
    RecipeType.SPECULATIVE_DFLASH: ModelOptDFlashRecipe,
    RecipeType.SPECULATIVE_MEDUSA: ModelOptMedusaRecipe,
}
