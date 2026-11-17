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

"""Shared mask-reuse calibration types and validation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

_FORMULA = "a * exp(b * target_sparsity)"

_CALIBRATION_PROTOCOL = "modelopt_mask_reuse_target_sparsity_v1"

_EVIDENCE_FIELDS = frozenset(
    {
        "calibration_plan_sha256",
        "family_registry_sha256",
        "vanilla_fit_sha256",
        "reuse_bundle_sha256",
        "grouped_fit_sha256",
        "outer_report_sha256",
    }
)

_DEPLOYMENT_GEOMETRY_CONTRACT: dict[str, object] = {
    "schema_version": 1,
    "batch_size": 1,
    "max_query_chunk_tokens": 8192,
    "query_block_tokens": 128,
    "key_block_tokens": 128,
    "qstage2_query_pair_tokens": 256,
    "kv_page_tokens": 16,
    "head_dim": 128,
    "causal": True,
    "bottom_right_aligned": True,
    "query_chunk_start_alignment_tokens": 128,
    "attention_dtype": "bfloat16",
    "kv_cache_dtype": "bfloat16",
    "common_prefix": False,
    "cascade_attention": False,
    "context_parallel_size": 1,
    "pipeline_parallel_size": 1,
}

Bucket = tuple[int, int | None]

ConsumerHead = tuple[int, int]


class MaskReuseCalibrationError(ValueError):
    """Raised when observations cannot produce a trustworthy reuse policy."""


@dataclass(frozen=True, slots=True)
class AnchorLayerStats:
    """Per-head BLASST mask statistics for one topology anchor layer."""

    retained_tiles: tuple[int, ...]
    dropped_mass: tuple[float, ...]


def _integer(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise MaskReuseCalibrationError(f"{name} must be an integer >= {minimum}")
    return value


def _number(
    value: object,
    name: str,
    *,
    minimum: float,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise MaskReuseCalibrationError(f"{name} must be a finite number")
    result = float(value)
    below = result < minimum if minimum_inclusive else result <= minimum
    if not math.isfinite(result) or below or (maximum is not None and result > maximum):
        raise MaskReuseCalibrationError(f"{name} is outside its valid range")
    return result


def _dropped_mass_matrix(raw: object, *, heads: int, label: str) -> tuple[tuple[float, ...], ...]:
    """Validate a head-by-donor matrix without formatting a label for every element."""
    if not isinstance(raw, list) or len(raw) != heads:
        raise MaskReuseCalibrationError(f"{label} must have {heads} consumer rows")
    rows = []
    for head, row in enumerate(raw):
        if not isinstance(row, list) or len(row) != heads:
            raise MaskReuseCalibrationError(f"{label}[{head}] must cover every global donor head")
        for donor, value in enumerate(row):
            if isinstance(value, bool) or not isinstance(value, int | float) or not 0 <= value <= 1:
                raise MaskReuseCalibrationError(
                    f"{label}[{head}][{donor}] must be finite and in [0, 1]"
                )
        rows.append(tuple(map(float, row)))
    return tuple(rows)


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise MaskReuseCalibrationError(f"{name} must be a lowercase SHA256")
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise MaskReuseCalibrationError(f"{name} must be a lowercase SHA256")
    return normalized


def _find_skip_softmax_group(raw: Mapping[str, object]) -> Mapping[str, object]:
    current: object = raw
    if "sparse_attention_config" in raw:
        current = raw["sparse_attention_config"]
    if not isinstance(current, Mapping):
        raise MaskReuseCalibrationError("sparse_attention_config must be an object")
    if "config_groups" in current:
        groups = current["config_groups"]
        if not isinstance(groups, Mapping):
            raise MaskReuseCalibrationError("config_groups must be an object")
        matches = [
            group
            for group in groups.values()
            if isinstance(group, Mapping)
            and (
                group.get("algorithm") == "skip_softmax"
                or group.get("sparse_algo") == "softmax_skip"
            )
        ]
        if len(matches) != 1:
            raise MaskReuseCalibrationError(
                "vanilla config must contain exactly one skip_softmax config group"
            )
        selected = matches[0]
        if selected.get("sparse_algo") == "softmax_skip" and "threshold_scale_factor" in current:
            # Older ModelOpt serving calibration stored the fit beside a
            # ``sparse_algo: softmax_skip`` group instead of inside it.
            current = current["threshold_scale_factor"]
        else:
            current = selected
    if isinstance(current, Mapping) and "threshold_scale_factor" in current:
        current = current["threshold_scale_factor"]
    if not isinstance(current, Mapping):
        raise MaskReuseCalibrationError("threshold_scale_factor must be an object")
    return current


def canonical_prefill_threshold_scale_factor(
    vanilla_calibration: Mapping[str, object],
) -> dict[str, object]:
    """Canonicalize ModelOpt fit parameters or exported skip-softmax metadata."""
    raw = _find_skip_softmax_group(vanilla_calibration)
    if "calibration_params" in raw:
        raw = _find_skip_softmax_group(raw["calibration_params"])  # type: ignore[arg-type]
    formula = raw.get("formula", _FORMULA)
    if formula != _FORMULA:
        raise MaskReuseCalibrationError("vanilla calibration uses an unsupported formula")
    params = raw.get("prefill")
    if not isinstance(params, Mapping):
        raise MaskReuseCalibrationError("vanilla calibration requires prefill fit parameters")
    unknown = params.keys() - {
        "a",
        "b",
        "min_observed_sparsity",
        "max_observed_sparsity",
    }
    if unknown or not {"a", "b"} <= params.keys():
        raise MaskReuseCalibrationError(
            f"prefill fit requires a and b and contains unknown fields {sorted(unknown)}"
        )
    prefill: dict[str, float] = {
        "a": _number(params["a"], "prefill.a", minimum=0.0, minimum_inclusive=False),
        "b": _number(params["b"], "prefill.b", minimum=0.0, maximum=20.0),
    }
    bounds = {"min_observed_sparsity", "max_observed_sparsity"} & params.keys()
    if bounds and len(bounds) != 2:
        raise MaskReuseCalibrationError("observed sparsity bounds must appear together")
    if bounds:
        lower = _number(
            params["min_observed_sparsity"],
            "prefill.min_observed_sparsity",
            minimum=0.0,
            maximum=1.0,
        )
        upper = _number(
            params["max_observed_sparsity"],
            "prefill.max_observed_sparsity",
            minimum=0.0,
            maximum=1.0,
        )
        if lower > upper:
            raise MaskReuseCalibrationError("observed sparsity range is reversed")
        prefill.update(min_observed_sparsity=lower, max_observed_sparsity=upper)
    return {"formula": _FORMULA, "prefill": prefill}


def _normalize_topology(raw: Mapping[str, object]) -> tuple[tuple[int, ...], dict[int, int]]:
    if set(raw) != {"anchors", "nearest"}:
        raise MaskReuseCalibrationError("topology must contain exactly anchors and nearest")
    raw_anchors = raw["anchors"]
    if not isinstance(raw_anchors, list) or not raw_anchors:
        raise MaskReuseCalibrationError("topology anchors must be a non-empty list")
    anchors = tuple(sorted(_integer(value, "topology anchor", minimum=0) for value in raw_anchors))
    if len(anchors) != len(set(anchors)):
        raise MaskReuseCalibrationError("topology anchors must be unique")
    raw_nearest = raw["nearest"]
    if not isinstance(raw_nearest, Mapping):
        raise MaskReuseCalibrationError("topology nearest must be an object")
    nearest: dict[int, int] = {}
    for raw_layer, raw_anchor in raw_nearest.items():
        try:
            layer = int(raw_layer)
        except (TypeError, ValueError) as error:
            raise MaskReuseCalibrationError("topology nearest has a non-integer key") from error
        if str(layer) != str(raw_layer) or layer in nearest:
            raise MaskReuseCalibrationError("topology nearest keys must be canonical and unique")
        nearest[layer] = _integer(raw_anchor, f"topology nearest[{layer}]", minimum=0)
    anchor_set = set(anchors)
    if not anchor_set <= nearest.keys():
        raise MaskReuseCalibrationError("topology nearest must include every anchor")
    for layer, anchor in nearest.items():
        if anchor not in anchor_set or (layer != anchor and anchor >= layer):
            raise MaskReuseCalibrationError(f"topology layer {layer} has invalid anchor {anchor}")
        if layer in anchor_set and anchor != layer:
            raise MaskReuseCalibrationError(f"topology anchor {layer} must map to itself")
        if anchor != max(a for a in anchors if a <= layer):
            raise MaskReuseCalibrationError(
                f"topology layer {layer} must use its nearest preceding anchor"
            )
    return anchors, dict(sorted(nearest.items()))


def _bucket_key(bucket: Bucket) -> tuple[int, float]:
    return bucket[0], math.inf if bucket[1] is None else float(bucket[1])


@dataclass(frozen=True, slots=True)
class _Choice:
    donor_head: int
    fallback: bool
    retained_tiles: int


@dataclass(frozen=True, slots=True)
class _Selection:
    target_sparsity: float | None
    choices: Mapping[ConsumerHead, _Choice]
    frontier: tuple[Mapping[str, object], ...]
    exact_reason: str | None = None
    bmm1_eligible_tiles: int = 0
    bmm1_skipped_tiles: int = 0
    target_bmm1_skip_ratio_met: bool = False
    worst_prompt_reuse_dropped_mass: float = 0.0
    mean_prompt_reuse_dropped_mass: float = 0.0
    worst_individual_reuse_dropped_mass: float = 0.0
    worst_prompt_total_dropped_mass: float = 0.0
    mean_prompt_total_dropped_mass: float = 0.0
    worst_individual_dropped_mass: float = 0.0


@dataclass(slots=True)
class _ReuseEvaluation:
    eligible_tiles: int = 0
    retained_tiles: int = 0
    sparse_observations: int = 0
    violations: int = 0
    dropped_mass_sum: float = 0.0
    worst_dropped_mass: float = 0.0
    model_wide_eligible_tiles: int = 0
    prompt_count: int = 0
    total_prompt_risk_sum: float = 0.0
    worst_prompt_total_risk: float = 0.0

    def add(self, other: _ReuseEvaluation) -> None:
        self.eligible_tiles += other.eligible_tiles
        self.retained_tiles += other.retained_tiles
        self.sparse_observations += other.sparse_observations
        self.violations += other.violations
        self.dropped_mass_sum += other.dropped_mass_sum
        self.worst_dropped_mass = max(self.worst_dropped_mass, other.worst_dropped_mass)
        self.model_wide_eligible_tiles += other.model_wide_eligible_tiles
        self.prompt_count += other.prompt_count
        self.total_prompt_risk_sum += other.total_prompt_risk_sum
        self.worst_prompt_total_risk = max(
            self.worst_prompt_total_risk, other.worst_prompt_total_risk
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "eligible_tiles": self.eligible_tiles,
            "retained_tiles": self.retained_tiles,
            "bmm1_tile_savings_fraction": (
                1.0 - self.retained_tiles / self.eligible_tiles if self.eligible_tiles else 0.0
            ),
            "sparse_head_prompt_observations": self.sparse_observations,
            "report_threshold_exceedance_count": self.violations,
            "report_threshold_exceedance_rate": (
                self.violations / self.sparse_observations if self.sparse_observations else 0.0
            ),
            "mean_dropped_mass": (
                self.dropped_mass_sum / self.sparse_observations
                if self.sparse_observations
                else 0.0
            ),
            "worst_dropped_mass": self.worst_dropped_mass,
            "bmm1_tile_savings_denominator": "consumer_heads_only",
            **(
                {
                    "model_wide_eligible_tiles": self.model_wide_eligible_tiles,
                    "model_wide_bmm1_tile_skip_ratio": (self.eligible_tiles - self.retained_tiles)
                    / self.model_wide_eligible_tiles,
                    "worst_prompt_model_wide_total_dropped_mass": self.worst_prompt_total_risk,
                    "mean_prompt_model_wide_total_dropped_mass": self.total_prompt_risk_sum
                    / self.prompt_count,
                }
                if self.model_wide_eligible_tiles
                else {}
            ),
        }


@dataclass(slots=True)
class _AnchorEvaluation:
    eligible_tiles: int = 0
    retained_tiles: int = 0
    prompt_count: int = 0
    violations: int = 0
    prompt_mean_sum: float = 0.0
    worst_prompt_mean: float = 0.0

    def add(self, other: _AnchorEvaluation) -> None:
        self.eligible_tiles += other.eligible_tiles
        self.retained_tiles += other.retained_tiles
        self.prompt_count += other.prompt_count
        self.violations += other.violations
        self.prompt_mean_sum += other.prompt_mean_sum
        self.worst_prompt_mean = max(self.worst_prompt_mean, other.worst_prompt_mean)

    def to_mapping(self, *, exact: bool = False, report_only: bool = False) -> dict[str, object]:
        result = {
            "policy_exact": exact,
            "constraint_statistic": "worst_prompt_mean_anchor_dropped_mass",
            "eligible_tiles": self.eligible_tiles,
            "retained_tiles": self.retained_tiles,
            "bmm2_tile_savings_fraction": (
                1.0 - self.retained_tiles / self.eligible_tiles if self.eligible_tiles else 0.0
            ),
            "evaluated_prompt_count": self.prompt_count,
            "constraint_violation_count": self.violations,
            "constraint_violation_rate": (
                self.violations / self.prompt_count if self.prompt_count else 0.0
            ),
            "mean_prompt_mean_anchor_dropped_mass": (
                self.prompt_mean_sum / self.prompt_count if self.prompt_count else 0.0
            ),
            "worst_prompt_mean_anchor_dropped_mass": self.worst_prompt_mean,
        }
        if report_only:
            result.pop("constraint_statistic")
            result["report_metric"] = "per_prompt_mean_across_selected_anchor_heads_dropped_mass"
            result["report_threshold_exceedance_count"] = result.pop("constraint_violation_count")
            result["report_threshold_exceedance_rate"] = result.pop("constraint_violation_rate")
        return result


def _risk_variant(max_anchor_dropped_mass, anchor_dropped_mass_report_threshold):
    if max_anchor_dropped_mass is not None:
        if anchor_dropped_mass_report_threshold is not None:
            raise MaskReuseCalibrationError(
                "choose the legacy anchor cap OR the report-only threshold"
            )
        bound = _number(
            max_anchor_dropped_mass, "max_anchor_dropped_mass", minimum=0.0, maximum=1.0
        )
        return bound, bound, "legacy_anchor_cap_v1"
    threshold = (
        0.02
        if anchor_dropped_mass_report_threshold is None
        else _number(
            anchor_dropped_mass_report_threshold,
            "anchor_dropped_mass_report_threshold",
            minimum=0.0,
            maximum=1.0,
        )
    )
    return None, threshold, "anchor_inclusive_v1"


def _pareto_donor_options(options):
    """Prune only dominance that also preserves the canonical donor-map tie-break."""
    skipped = np.asarray([option.bmm1_skipped_tiles for option in options])
    ranks = np.asarray(
        [0 if option.choice.fallback else option.choice.donor_head + 1 for option in options]
    )
    risks = np.asarray([option.risk_by_prompt for option in options], dtype=np.float64)
    dominates = skipped[:, None] >= skipped[None, :]
    for prompt in risks.T:
        dominates &= prompt[:, None] <= prompt[None, :]
    dominates &= (skipped[:, None] > skipped[None, :]) | (ranks[:, None] < ranks[None, :])
    keep = ~dominates.any(axis=0)
    retained_risks = risks[keep]
    # A zero MIP gap does not resolve distinct coefficients below the native
    # feasibility precision. Refuse these alternatives rather than treating
    # them as exact ties and preferring cheaper work over lower risk.
    separations = np.zeros((len(retained_risks), len(retained_risks)))
    for prompt in retained_risks.T:
        np.maximum(separations, np.abs(prompt[:, None] - prompt[None, :]), out=separations)
    if np.any((separations > 0) & (separations <= 1e-9)):
        from .mask_reuse_solver import MaskReuseSolverError

        raise MaskReuseSolverError(
            "distinct donor risks are below native solver precision; selection is unresolved", []
        )
    return tuple(option for option, retained in zip(options, keep, strict=True) if retained)


def _validate_donor_head_mode(mode: str) -> None:
    if mode not in ("calibrated", "same_index"):
        raise MaskReuseCalibrationError("donor_head_mode must be calibrated or same_index")


def _donor_head_candidates(mode: str, consumer_head: int, global_num_heads: int):
    return (consumer_head,) if mode == "same_index" else range(global_num_heads)


def _canonical_evidence(raw: Mapping[str, object]) -> dict[str, str]:
    missing = {"vanilla_fit_sha256", "reuse_bundle_sha256"} - raw.keys()
    extra = raw.keys() - _EVIDENCE_FIELDS
    if missing or extra:
        raise MaskReuseCalibrationError(
            f"evidence fields do not match schema v3; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return {field: _sha256(raw[field], f"evidence.{field}") for field in sorted(raw)}


__all__ = [
    "AnchorLayerStats",
    "MaskReuseCalibrationError",
    "canonical_prefill_threshold_scale_factor",
]
