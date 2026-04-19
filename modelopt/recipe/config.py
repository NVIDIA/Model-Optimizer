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


class RecipeType(str, Enum):
    """List of recipe types."""

    PTQ = "ptq"
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
