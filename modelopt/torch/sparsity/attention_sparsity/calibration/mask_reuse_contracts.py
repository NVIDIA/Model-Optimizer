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

"""Shared numerical and input contracts for mask-reuse capture and calibration."""

import json
import math
from fractions import Fraction

__all__ = [
    "required_skipped_tiles",
    "selection_target_met",
    "threshold_pair_matches",
    "validate_capture_scope",
    "validate_context_buckets",
    "validate_manifest_counts",
    "validate_model_geometry",
]


def required_skipped_tiles(ratio: float, eligible_tiles: int) -> int:
    """Round the user's decimal saving target upward using exact integer arithmetic."""
    if isinstance(ratio, bool) or not math.isfinite(ratio) or not 0 <= ratio <= 1:
        raise ValueError("BMM1 skip ratio must be finite and in [0, 1]")
    if type(eligible_tiles) is not int or eligible_tiles < 0:
        raise ValueError("eligible tiles must be a nonnegative integer")
    target = Fraction(str(ratio)) * eligible_tiles
    return (target.numerator + target.denominator - 1) // target.denominator


def threshold_pair_matches(log2_threshold: float, threshold: float) -> bool:
    """Validate diagnostic lambda against authoritative log2, allowing one binary64 ULP."""
    if not math.isfinite(log2_threshold) or not math.isfinite(threshold) or threshold <= 0:
        return False
    try:
        expected = 2.0**log2_threshold
    except OverflowError:
        return False
    return math.isfinite(expected) and abs(expected - threshold) <= math.ulp(expected)


def validate_context_buckets(buckets) -> None:
    """Bounds are inclusive: distinct ranges cannot touch or overlap."""
    previous_max = 0
    for minimum, maximum in sorted(set(buckets), key=lambda pair: pair[0]):
        if type(minimum) is not int or minimum <= 0:
            raise ValueError("context bucket minimum must be a positive integer")
        if maximum is not None and (type(maximum) is not int or maximum < minimum):
            raise ValueError("context bucket maximum must be an integer at least its minimum")
        if previous_max is None or minimum <= previous_max:
            raise ValueError("context buckets must be ordered and non-overlapping")
        previous_max = maximum


def validate_manifest_counts(manifest) -> None:
    """Verify row counts before opening the capture in a solver."""
    for name in ("capture_count", "candidate_cell_count"):
        if type(manifest.get(name)) is not int or manifest[name] <= 0:
            raise ValueError(f"capture manifest {name} must be positive")
    rows = manifest.get("captures")
    if not isinstance(rows, list) or len(rows) != manifest["capture_count"]:
        raise ValueError("capture manifest captures do not match capture_count")
    total = 0
    for row in rows:
        value = row.get("candidate_cell_count") if isinstance(row, dict) else None
        if type(value) is not int or value < 0:
            raise ValueError("capture candidate_cell_count must be a nonnegative integer")
        total += value
    if total != manifest["candidate_cell_count"]:
        raise ValueError("capture manifest candidate-cell total is inconsistent")


def validate_model_geometry(checkpoint, layers, heads) -> None:
    """Bind captured full-attention layers and query heads to the checkpoint config."""
    config = json.loads((checkpoint.checkpoint_root / "config.json").read_bytes())
    config = config.get("text_config", config)
    count = config.get("num_hidden_layers")
    if type(count) is not int or count <= 0 or type(config.get("num_attention_heads")) is not int:
        raise ValueError("checkpoint config must declare layer and attention-head counts")
    types = config.get("layers_block_type", config.get("layer_types"))
    pattern = config.get("hybrid_override_pattern")
    if types is not None:
        if not isinstance(types, list) or len(types) != count:
            raise ValueError("checkpoint attention layer types do not match num_hidden_layers")
        expected = [i for i, kind in enumerate(types) if kind in ("attention", "full_attention")]
    elif pattern is not None:
        if not isinstance(pattern, str) or len(pattern) != count:
            raise ValueError("checkpoint hybrid pattern does not match num_hidden_layers")
        expected = [i for i, kind in enumerate(pattern) if kind == "*"]
    else:
        expected = list(range(count))
    if sorted(layers) != expected or heads != config["num_attention_heads"]:
        raise ValueError(
            "capture/topology does not cover the checkpoint's full-attention layers and heads"
        )


def validate_capture_scope(manifest, *, diagnostic=False) -> None:
    """Keep final-chunk captures out of ordinary prefill policy calibration."""
    if not diagnostic and manifest.get("capture_manifest_schema_version") not in (10, 11):
        raise ValueError(
            "policy calibration requires whole-prefill captures; use --diagnostic-final-chunk for diagnostics"
        )


def selection_target_met(artifact) -> bool:
    """Read the selection outcome without using a rounded display ratio."""
    report = artifact["calibration_report"]
    if "bmm1_skip_objective" in report:
        return report["bmm1_skip_objective"]["target_met"] is True
    rows = report.get("by_bucket", [])
    return bool(rows) and all(row["bmm1_skip_objective"]["target_met"] is True for row in rows)
