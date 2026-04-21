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

"""Configuration for KV cache sparsity modes."""

from __future__ import annotations

from typing import Literal

from pydantic import field_validator, model_validator

from modelopt.torch.opt.config import ModeloptBaseConfig, ModeloptField

__all__ = ["TriAttentionConfig"]


class TriAttentionConfig(ModeloptBaseConfig):
    """Configuration for TriAttention KV cache eviction.

    TriAttention scores cached KV entries using a trigonometric model derived from
    pre-RoPE Q/K concentration. Calibration computes per-head frequency statistics;
    at runtime, the serving engine scores and evicts tokens periodically.

    Exactly one of ``budget`` or ``target_sparsity_ratio`` must be set:

    - ``budget``: absolute token count to retain per head (fixed-size cache).
    - ``target_sparsity_ratio``: fraction of tokens to evict at each pruning step.
      Cache size auto-scales with generation length. Value in (0, 1).
    """

    # Eviction policy (exactly one must be set)
    budget: int | None = ModeloptField(
        default=None,
        title="KV token budget (absolute).",
        description=(
            "Number of KV tokens to retain per head after pruning. "
            "Mutually exclusive with target_sparsity_ratio."
        ),
    )
    target_sparsity_ratio: float | None = ModeloptField(
        default=None,
        title="Target sparsity ratio (percentile-based).",
        description=(
            "Fraction of tokens to evict at each pruning step, in (0, 1). "
            "Example: 0.7 means evict 70% of tokens (keep top 30% by score). "
            "Mutually exclusive with budget."
        ),
    )

    # Pruning schedule
    prune_interval: int = ModeloptField(
        default=128,
        title="Pruning interval.",
        description="Re-score and evict every N generated tokens.",
    )
    window_size: int = ModeloptField(
        default=128,
        title="Protected window size.",
        description="Number of most recent tokens always retained.",
    )

    # Scoring
    pruning_mode: Literal["per_head", "per_layer_per_head"] = ModeloptField(
        default="per_head",
        title="Pruning mode.",
        description=(
            "'per_head': independent budget per KV head. "
            "'per_layer_per_head': budget allocated per layer and head."
        ),
    )
    score_aggregation: Literal["mean", "max"] = ModeloptField(
        default="mean",
        title="Offset score aggregation.",
        description="How to aggregate scores across geometric offsets.",
    )
    offset_max_length: int = ModeloptField(
        default=65536,
        title="Maximum geometric offset.",
        description="Offsets are [1, 2, 4, ..., offset_max_length].",
    )
    disable_mlr: bool = ModeloptField(
        default=False,
        title="Disable MLR term.",
        description="If True, disable the magnitude linear regression extra term.",
    )
    disable_trig: bool = ModeloptField(
        default=False,
        title="Disable trigonometric term.",
        description="If True, use only the additive (MLR) term for scoring.",
    )

    # Calibration
    calib_size: int = ModeloptField(
        default=100000,
        title="Calibration tokens.",
        description="Number of tokens for calibration. 50K-960K, any domain.",
    )

    @field_validator("pruning_mode")
    @classmethod
    def validate_pruning_mode(cls, v: str) -> str:
        """Validate pruning_mode is a supported value."""
        valid = {"per_head", "per_layer_per_head"}
        if v not in valid:
            raise ValueError(f"pruning_mode must be one of {valid}, got '{v}'")
        return v

    @field_validator("score_aggregation")
    @classmethod
    def validate_score_aggregation(cls, v: str) -> str:
        """Validate score_aggregation is a supported value."""
        valid = {"mean", "max"}
        if v not in valid:
            raise ValueError(f"score_aggregation must be one of {valid}, got '{v}'")
        return v

    @model_validator(mode="after")
    def validate_budget_or_sparsity(self) -> TriAttentionConfig:
        """Exactly one of budget or target_sparsity_ratio must be set."""
        budget_set = self.budget is not None
        sparsity_set = self.target_sparsity_ratio is not None
        if not budget_set and not sparsity_set:
            raise ValueError("Must set exactly one of 'budget' or 'target_sparsity_ratio'")
        if budget_set and sparsity_set:
            raise ValueError("Cannot set both 'budget' and 'target_sparsity_ratio'; pick one")
        if sparsity_set and not (0.0 < self.target_sparsity_ratio < 1.0):
            raise ValueError(
                f"target_sparsity_ratio must be in (0, 1), got {self.target_sparsity_ratio}"
            )
        return self
