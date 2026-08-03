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
from modelopt.torch.opt.config_loader import load_config
from modelopt.torch.quantization.config import QuantizeConfig  # noqa: TC001
from modelopt.torch.speculative.config import DFlashConfig, EagleConfig, MedusaConfig
from modelopt.torch.speculative.plugins.hf_training_args import DataArguments as SpecDataArgs
from modelopt.torch.speculative.plugins.hf_training_args import ModelArguments as SpecModelArgs
from modelopt.torch.speculative.plugins.hf_training_args import (
    TrainingArguments as SpecTrainingArgs,
)

__all__ = [
    "RECIPE_TYPE_TO_CLASS",
    "AutoQuantizeConfig",
    "AutoQuantizeConstraints",
    "AutoQuantizeCost",
    "AutoQuantizeModuleSearchSpace",
    "ModelOptAutoQuantizeRecipe",
    "ModelOptDFlashRecipe",
    "ModelOptEagleRecipe",
    "ModelOptMedusaRecipe",
    "ModelOptPTQRecipe",
    "ModelOptRecipeBase",
    "ModelOptSpeculativeRecipeBase",
    "RecipeMetadataConfig",
    "RecipeType",
]


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


# Named alias so a shared layer-pattern unit (e.g. configs/auto_quantize/units/base_disabled_layers)
# can declare ``modelopt-schema: modelopt.recipe.config.LayerPatternList`` and be spliced into a
# ``list[str]`` field — mirrors how base_disable_all is imported into a PTQ quant_cfg list.
LayerPatternList = list[str]


def _load_layer_pattern_list(config_path: str) -> list[str]:
    """Load a ``list[str]`` layer-pattern unit (e.g. AutoQuantize base disabled/cost-excluded).

    Relies on the unit's ``modelopt-schema: ...LayerPatternList`` comment (like
    _load_quantizer_cfg_dict_list) rather than an explicit ``list[str]`` schema_type.
    """
    return list(load_config(config_path))


# Base AutoQuantize layer-pattern sets, loaded once (used by the deprecated --auto_quantize_* CLI shim).
AUTOQUANT_BASE_DISABLED_LAYERS: list[str] = _load_layer_pattern_list(
    "configs/auto_quantize/units/base_disabled_layers"
)
AUTOQUANT_BASE_COST_EXCLUDED_LAYERS: list[str] = _load_layer_pattern_list(
    "configs/auto_quantize/units/base_cost_excluded_layers"
)


class AutoQuantizeCost(ModeloptBaseConfig):
    """Cost-model parameters (the ``cost`` sub-dict of ``mtq.auto_quantize`` constraints)."""

    active_moe_expert_ratio: float | None = ModeloptField(
        default=None,
        title="Active MoE expert ratio",
        description="Routed experts active per token, in (0, 1]. Used by the 'active_moe' cost model.",
    )

    @field_validator("active_moe_expert_ratio")
    @classmethod
    def _validate_active_moe_expert_ratio(cls, v: float | None) -> float | None:
        if v is not None and not (0 < v <= 1):
            raise ValueError(f"active_moe_expert_ratio must be in (0, 1], got {v}")
        return v


class AutoQuantizeConstraints(ModeloptBaseConfig):
    """LP search constraints + cost model; matches the ``mtq.auto_quantize`` constraints dict."""

    effective_bits: float = ModeloptField(
        default=4.8,
        title="Effective bits per weight",
        description="Average weight-storage bits target for the LP, in (0, 16].",
    )
    cost_model: Literal["weight", "active_moe"] = ModeloptField(
        default="weight",
        title="Cost model",
        description="'weight' counts all weights equally; 'active_moe' scales routed-expert weights.",
    )
    cost: AutoQuantizeCost | None = ModeloptField(
        default=None,
        title="Cost-model parameters",
        description="Extra cost-model parameters; omit for the 'weight' cost model.",
    )

    @field_validator("effective_bits")
    @classmethod
    def _validate_effective_bits(cls, v: float) -> float:
        if not (0 < v <= 16):
            raise ValueError(f"effective_bits must be in (0, 16], got {v}")
        return v


class AutoQuantizeModuleSearchSpace(ModeloptBaseConfig):
    """Candidate formats selectable for modules matching one or more name patterns."""

    module_name_patterns: LayerPatternList = ModeloptField(
        default=[],
        title="Module name patterns",
        description="Glob patterns matched against quantizable module names. A grouped AutoQuantize "
        "decision must match a rule for every module in the group or for none of them.",
        validate_default=True,
    )
    candidate_formats: list[QuantizeConfig] = ModeloptField(
        default=[],
        title="Module candidate quantization formats",
        description="Formats selectable for matching modules. These override the top-level "
        "candidate_formats for the matching AutoQuantize decision group.",
        validate_default=True,
    )
    allow_no_quant: bool = ModeloptField(
        default=True,
        title="Allow no-quant selection",
        description="Whether BF16/no-quant is selectable for matching modules. AutoQuantize keeps "
        "an internal no-quant baseline for sensitivity scoring and cost normalization even when "
        "this is false.",
    )

    @field_validator("module_name_patterns")
    @classmethod
    def _at_least_one_module_pattern(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("module_search_spaces requires at least 1 module_name_pattern")
        return v

    @field_validator("candidate_formats")
    @classmethod
    def _at_least_one_module_candidate(cls, v: list[QuantizeConfig]) -> list[QuantizeConfig]:
        if not v:
            raise ValueError("module_search_spaces requires at least 1 candidate_format")
        return v


class AutoQuantizeConfig(ModeloptBaseConfig):
    """Schema for the ``auto_quantize`` block of an AutoQuantize recipe."""

    constraints: AutoQuantizeConstraints = Field(
        title="Search constraints + cost model",
        description="LP budget and cost model.",
    )
    candidate_formats: list[QuantizeConfig] = ModeloptField(
        default=[],
        title="Candidate quantization formats",
        description="Fallback per-layer search space for modules not matched by "
        "module_search_spaces. Each entry is a full QuantizeConfig. BF16/no-quant is always an "
        "implicit additional choice. Omit this field when the parent recipe supplies a fixed "
        "quantize baseline and explicitly lists every searched family in module_search_spaces.",
        validate_default=True,
    )
    module_search_spaces: list[AutoQuantizeModuleSearchSpace] = ModeloptField(
        default=[],
        title="Module-specific search spaces",
        description="Optional per-module overrides for candidate formats and BF16/no-quant "
        "selectability. Matching is performed after runtime-fusion grouping.",
    )
    auto_quantize_method: Literal["gradient", "kl_div"] = ModeloptField(
        default="gradient",
        title="Sensitivity scoring method",
        description="'gradient' (Taylor + Fisher, needs labels) or 'kl_div' (no labels).",
    )
    score_size: int = ModeloptField(
        default=128,
        title="Scoring sample count",
        description="Number of samples used for sensitivity scoring (divided by batch_size to get "
        "the number of mtq scoring steps). Matches the former --auto_quantize_score_size.",
    )
    disabled_layers: LayerPatternList = ModeloptField(
        default=[],
        title="Search-excluded layer patterns",
        description="Glob patterns; matching layers are excluded from the search (kept full precision).",
    )
    cost_excluded_layers: LayerPatternList = ModeloptField(
        default=[],
        title="Cost-excluded layer patterns",
        description="Glob patterns excluded from the bit-budget accounting (cost_weight 0) — e.g. VL "
        "vision towers. Distinct from disabled_layers: those are removed from the search; these still "
        "get searched but don't count toward effective_bits. The two roles overlap but are independent.",
    )
    kv_cache: QuantizeConfig | None = ModeloptField(
        default=None,
        title="KV cache config (optional)",
        description="QuantizeConfig applied as a uniform post-step; falls back to "
        "the --kv_cache_qformat CLI flag when omitted.",
    )

    @model_validator(mode="after")
    def _has_search_space(self):
        if not self.candidate_formats and not self.module_search_spaces:
            raise ValueError(
                "auto_quantize requires candidate_formats or at least one module_search_spaces "
                "entry. For uniform quantization, use a PTQ recipe instead."
            )
        return self


class ModelOptAutoQuantizeRecipe(ModelOptRecipeBase):
    """Our config class for AutoQuantize recipes."""

    metadata: RecipeMetadataConfig = _metadata_field(RecipeType.AUTO_QUANTIZE)

    quantize: QuantizeConfig | None = ModeloptField(
        default=None,
        title="Fixed PTQ baseline",
        description="Optional normal PTQ QuantizeConfig for modules outside the explicit "
        "AutoQuantize module_search_spaces. Fixed and searched modules are calibrated, scored, "
        "costed, and exported in one integrated AutoQuantize operation.",
    )

    auto_quantize: AutoQuantizeConfig = Field(
        title="AutoQuantize config",
        description="AutoQuantize search configuration. Required.",
    )

    @model_validator(mode="after")
    def _validate_fixed_and_searched_spaces(self):
        has_fixed_baseline = self.quantize is not None
        has_global_search = bool(self.auto_quantize.candidate_formats)
        if has_fixed_baseline and has_global_search:
            raise ValueError(
                "An AutoQuantize recipe with a fixed quantize baseline must omit top-level "
                "auto_quantize.candidate_formats and explicitly list searched modules under "
                "auto_quantize.module_search_spaces."
            )
        if has_fixed_baseline and not self.auto_quantize.module_search_spaces:
            raise ValueError(
                "An AutoQuantize recipe with a fixed quantize baseline requires at least one "
                "auto_quantize.module_search_spaces entry."
            )
        if not has_fixed_baseline and not has_global_search:
            raise ValueError(
                "An AutoQuantize recipe without a fixed quantize baseline requires top-level "
                "auto_quantize.candidate_formats for unmatched modules."
            )
        return self


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
        self.eagle.eagle_offline = self.data.mode != "online"
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
        # offline (dumped .pt) and streaming (hidden states via NIXL RDMA from a vLLM
        # serve) both feed pre-computed base hidden states to the DFlash module, so
        # both set dflash_offline. Only fully-online training runs the base model.
        # Mirrors ModelOptEagleRecipe._derive_eagle_offline.
        self.dflash.dflash_offline = self.data.mode != "online"
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
