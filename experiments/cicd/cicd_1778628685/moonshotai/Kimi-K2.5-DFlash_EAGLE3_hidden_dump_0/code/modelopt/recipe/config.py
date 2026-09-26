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
from typing_extensions import NotRequired, TypedDict

from modelopt.torch.opt.config import ModeloptBaseConfig, ModeloptField
from modelopt.torch.quantization.config import QuantizeConfig


class RecipeType(str, Enum):
    """List of recipe types."""

    PTQ = "ptq"
    # QAT = "qat" # Not implemented yet, will be added in the future.


class RecipeMetadataConfig(TypedDict):
    """YAML shape of the recipe metadata section."""

    recipe_type: RecipeType
    description: NotRequired[str]


_DEFAULT_RECIPE_DESCRIPTION = "Model optimization recipe."


class ModelOptRecipeBase(ModeloptBaseConfig):
    """Base configuration class for model optimization recipes.

    If a layer name matches ``"*output_layer*"``, the attributes will be replaced with ``{"enable": False}``.
    """

    metadata: RecipeMetadataConfig = ModeloptField(
        default={"recipe_type": RecipeType.PTQ, "description": _DEFAULT_RECIPE_DESCRIPTION},
        title="Metadata",
        description="Recipe metadata containing the recipe type and description.",
        validate_default=True,
    )

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, metadata: RecipeMetadataConfig) -> RecipeMetadataConfig:
        """Validate recipe metadata and fill defaults for optional fields."""
        if metadata["recipe_type"] not in RecipeType:
            raise ValueError(
                f"Unsupported recipe type: {metadata['recipe_type']}. "
                f"Only {list(RecipeType)} are currently supported."
            )
        return {"description": _DEFAULT_RECIPE_DESCRIPTION, **metadata}

    @property
    def recipe_type(self) -> RecipeType:
        """Return the recipe type from metadata."""
        return self.metadata["recipe_type"]

    @property
    def description(self) -> str:
        """Return the recipe description from metadata."""
        return self.metadata.get("description", _DEFAULT_RECIPE_DESCRIPTION)


class ModelOptPTQRecipe(ModelOptRecipeBase):
    """Our config class for PTQ recipes."""

    quantize: QuantizeConfig = ModeloptField(
        default=QuantizeConfig(),
        title="PTQ config",
        description="PTQ config containing quant_cfg and algorithm.",
        validate_default=True,
    )
