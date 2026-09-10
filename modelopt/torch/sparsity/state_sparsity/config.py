# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Configuration and result schemas for DASC state sparsity."""

import math
from typing import Literal

from pydantic import Field, field_validator, model_validator

from modelopt.torch.opt.config import ModeloptBaseConfig, ModeloptField

__all__ = [
    "DASCCalibrationMeasurement",
    "DASCConfig",
    "DASCPolicy",
    "DASCQualityMeasurement",
]


class DASCQualityMeasurement(ModeloptBaseConfig):
    """Quality and lifecycle measurements for one calibration slice."""

    slice_id: str = Field(min_length=1)
    perplexity_retention: float = Field(gt=0.0, le=1.0, allow_inf_nan=False)
    top1_agreement: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    finite_continuation_logits: bool = Field(strict=True)
    retained_state_exact: bool = Field(strict=True)
    omitted_state_matches_recovery: bool = Field(strict=True)
    convolution_state_exact: bool = Field(strict=True)


class DASCCalibrationMeasurement(ModeloptBaseConfig):
    """Caller-supplied measurements for one candidate recovery window."""

    variant: Literal["dasc_nr", "dasc_wr"]
    wmax: int = Field(strict=True, gt=0)
    retained_heads: int = Field(strict=True, ge=0)
    total_heads: int = Field(strict=True, gt=0)
    checkpoint_savings: float = Field(ge=0.0, lt=1.0, allow_inf_nan=False)
    quality: list[DASCQualityMeasurement] = Field(min_length=1)

    @field_validator("quality")
    @classmethod
    def validate_unique_slices(
        cls, quality: list[DASCQualityMeasurement]
    ) -> list[DASCQualityMeasurement]:
        """Require one result per named calibration slice."""
        slice_ids = [measurement.slice_id for measurement in quality]
        if len(slice_ids) != len(set(slice_ids)):
            raise ValueError("quality slice_id values must be unique")
        return quality


class DASCConfig(ModeloptBaseConfig):
    """Configuration for GDN decay-aware state checkpoint sparsity."""

    variant: Literal["dasc_nr", "dasc_wr"] = ModeloptField(
        default="dasc_wr",
        description="Use zero recovery (DASC-NR) or suffix replay recovery (DASC-WR).",
    )
    epsilon: float = ModeloptField(
        default=1e-3,
        description="Retained contribution threshold used to derive static decay horizons.",
    )
    static_gate_input: float = ModeloptField(
        default=-0.3,
        description="Static gate input added to each GDN head's dt_bias.",
    )
    wmax_candidates: list[int] = ModeloptField(
        default=[8, 16, 32, 64, 128, 256],
        description="Positive candidate windows evaluated during offline calibration.",
    )
    min_perplexity_retention: float = ModeloptField(default=0.995)
    min_top1_agreement: float = ModeloptField(default=0.98)
    min_checkpoint_savings: float = ModeloptField(default=0.2)
    model_id: str = ModeloptField(default="", validate_default=True)
    model_revision: str = ModeloptField(default="", validate_default=True)
    model_config_id: str = ModeloptField(
        default="",
        description="Immutable identifier or digest for the model architecture configuration.",
        validate_default=True,
    )
    calibration_data_id: str = ModeloptField(
        default="",
        description="Immutable identifier or digest for the calibration data and protocol.",
        validate_default=True,
    )
    granularity: Literal["gdn_head"] = ModeloptField(default="gdn_head")
    preserve_convolution_state: Literal[True] = ModeloptField(default=True)

    @field_validator("epsilon")
    @classmethod
    def validate_epsilon(cls, epsilon: float) -> float:
        """Require a finite decay threshold strictly between zero and one."""
        if not math.isfinite(epsilon) or not 0.0 < epsilon < 1.0:
            raise ValueError("epsilon must be finite and in (0, 1)")
        return epsilon

    @field_validator("static_gate_input")
    @classmethod
    def validate_static_gate_input(cls, value: float) -> float:
        """Require a finite representative gate input."""
        if not math.isfinite(value):
            raise ValueError("static_gate_input must be finite")
        return value

    @field_validator("wmax_candidates", mode="before")
    @classmethod
    def validate_wmax_candidates(cls, candidates: object) -> object:
        """Require unique positive integer windows without power-of-two restrictions."""
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("wmax_candidates must be a non-empty list")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in candidates
        ):
            raise ValueError("wmax_candidates must contain only positive integers")
        if len(candidates) != len(set(candidates)):
            raise ValueError("wmax_candidates must be unique")
        return sorted(candidates)

    @field_validator("min_perplexity_retention")
    @classmethod
    def validate_perplexity_gate(cls, value: float) -> float:
        """Require a finite retention gate in (0, 1]."""
        if not math.isfinite(value) or not 0.0 < value <= 1.0:
            raise ValueError("min_perplexity_retention must be finite and in (0, 1]")
        return value

    @field_validator("min_top1_agreement")
    @classmethod
    def validate_top1_gate(cls, value: float) -> float:
        """Require a finite agreement gate in [0, 1]."""
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("min_top1_agreement must be finite and in [0, 1]")
        return value

    @field_validator("min_checkpoint_savings")
    @classmethod
    def validate_savings_gate(cls, value: float) -> float:
        """Require a finite physical checkpoint-savings gate in [0, 1)."""
        if not math.isfinite(value) or not 0.0 <= value < 1.0:
            raise ValueError("min_checkpoint_savings must be finite and in [0, 1)")
        return value

    @field_validator("model_id", "model_revision", "model_config_id", "calibration_data_id")
    @classmethod
    def validate_provenance(cls, value: str) -> str:
        """Require explicit immutable provenance instead of inferred defaults."""
        if not value.strip():
            raise ValueError("DASC provenance fields must be non-empty")
        return value


class DASCLayerPolicy(ModeloptBaseConfig):
    """Serializable whole-head policy for one GDN layer."""

    num_heads: int = Field(strict=True, gt=0)
    static_horizons: list[float] = Field(min_length=1)
    retained_heads: list[int]
    omitted_heads: list[int]

    @model_validator(mode="after")
    def validate_partition(self) -> "DASCLayerPolicy":
        """Validate that retained and omitted indices partition every GDN head."""
        if len(self.static_horizons) != self.num_heads or not all(
            math.isfinite(value) and value > 0.0 for value in self.static_horizons
        ):
            raise ValueError("static_horizons must contain one finite positive value per head")
        if sorted(self.retained_heads + self.omitted_heads) != list(range(self.num_heads)):
            raise ValueError("retained_heads and omitted_heads must partition all head indices")
        return self


class DASCPolicy(ModeloptBaseConfig):
    """Standalone JSON-safe DASC deployment policy."""

    format_version: Literal[1] = 1
    variant: Literal["dasc_nr", "dasc_wr"]
    recovery: Literal["zero", "suffix_replay"]
    epsilon: float
    static_gate_input: float
    selected_wmax: int = Field(strict=True, gt=0)
    wmax_candidates: list[int] = Field(min_length=1)
    quality_gates: dict[str, float]
    model_id: str
    model_revision: str
    model_config_id: str
    calibration_data_id: str
    granularity: Literal["gdn_head"]
    preserve_convolution_state: Literal[True]
    active_runtime_state: Literal["dense"] = "dense"
    model_structure_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decay_parameters_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    layers: dict[str, DASCLayerPolicy] = Field(min_length=1)
    measurements: list[DASCCalibrationMeasurement] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_policy(self) -> "DASCPolicy":
        """Reject inconsistent variant, candidate, measurement, or mask metadata."""
        expected_recovery = "zero" if self.variant == "dasc_nr" else "suffix_replay"
        if self.recovery != expected_recovery:
            raise ValueError(f"{self.variant} requires recovery={expected_recovery!r}")
        if self.selected_wmax not in self.wmax_candidates:
            raise ValueError("selected_wmax must be present in wmax_candidates")
        measured = [measurement.wmax for measurement in self.measurements]
        if sorted(measured) != sorted(self.wmax_candidates) or len(measured) != len(set(measured)):
            raise ValueError("measurements must contain exactly one result per wmax candidate")
        if any(measurement.variant != self.variant for measurement in self.measurements):
            raise ValueError("measurement variants must match the policy variant")
        for layer in self.layers.values():
            expected_retained = [
                head
                for head, horizon in enumerate(layer.static_horizons)
                if horizon > self.selected_wmax
            ]
            if layer.retained_heads != expected_retained:
                raise ValueError("retained_heads do not match the selected decay threshold")
        return self
