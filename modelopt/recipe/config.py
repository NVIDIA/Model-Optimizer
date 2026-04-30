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

from enum import Enum

from pydantic import field_validator

from modelopt.torch.opt.config import ModeloptBaseConfig, ModeloptField
from modelopt.torch.quantization.config import QuantizeConfig
from modelopt.torch.speculative.config import DFlashConfig, EagleConfig, MedusaConfig
from modelopt.torch.speculative.plugins.hf_training_args import DataArguments as SpecDataArgs
from modelopt.torch.speculative.plugins.hf_training_args import ModelArguments as SpecModelArgs
from modelopt.torch.speculative.plugins.hf_training_args import (
    TrainingArguments as SpecTrainingArgs,
)


class RecipeType(str, Enum):
    """List of recipe types."""

    PTQ = "ptq"
    SPECULATIVE_EAGLE = "speculative_eagle"
    SPECULATIVE_DFLASH = "speculative_dflash"
    SPECULATIVE_MEDUSA = "speculative_medusa"
    # QAT = "qat" # Not implemented yet, will be added in the future.


class ModelOptRecipeBase(ModeloptBaseConfig):
    """Base configuration class for model optimization recipes.

    If a layer name matches ``"*output_layer*"``, the attributes will be replaced with ``{"enable": False}``.
    """

    recipe_type: RecipeType = ModeloptField(
        default=RecipeType.PTQ,
        title="type of the recipe",
        description="The type of the recipe.",
        validate_default=True,
    )

    description: str = ModeloptField(
        default="Model optimization recipe.",
        title="Description",
        description="A brief description of the model optimization recipe.",
        validate_default=False,
    )

    @field_validator("recipe_type")
    @classmethod
    def validate_recipe_type(cls, v):
        """Validate recipe type."""
        if v not in RecipeType:
            raise ValueError(
                f"Unsupported recipe type: {v}. Only {list(RecipeType)} are currently supported."
            )
        return v


class ModelOptPTQRecipe(ModelOptRecipeBase):
    """Our config class for PTQ recipes."""

    quantize: QuantizeConfig = ModeloptField(
        default=QuantizeConfig(),
        title="PTQ config",
        description="PTQ config containing quant_cfg and algorithm.",
        validate_default=True,
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

    recipe_type: RecipeType = RecipeType.SPECULATIVE_EAGLE

    eagle: EagleConfig = ModeloptField(
        default=EagleConfig(),
        title="EAGLE config",
        description="EAGLE speculative decoding configuration.",
        validate_default=True,
    )


class ModelOptDFlashRecipe(ModelOptSpeculativeRecipeBase):
    """Our config class for DFlash speculative decoding recipes."""

    recipe_type: RecipeType = RecipeType.SPECULATIVE_DFLASH

    dflash: DFlashConfig = ModeloptField(
        default=DFlashConfig(),
        title="DFlash config",
        description="DFlash speculative decoding configuration.",
        validate_default=True,
    )


class ModelOptMedusaRecipe(ModelOptSpeculativeRecipeBase):
    """Our config class for Medusa speculative decoding recipes."""

    recipe_type: RecipeType = RecipeType.SPECULATIVE_MEDUSA

    medusa: MedusaConfig = ModeloptField(
        default=MedusaConfig(),
        title="Medusa config",
        description="Medusa speculative decoding configuration.",
        validate_default=True,
    )
