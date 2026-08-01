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

"""Offline calibration and schema-v3 export for cross-layer mask reuse.

The selector consumes prompt-level observations measured at target sparsities
derived from an existing ModelOpt skip-softmax fit. Calibration observations
alone select one target sparsity per context bucket and one donor head (or an
exact fallback) per consumer head. Held-out observations only evaluate the
frozen policy.

This module intentionally has no serving-backend dependency. It exports the
JSON-safe schema consumed by the mask-reuse attention backend.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import cast

import modelopt

from .checkpoint_manifest import VerifiedCheckpointManifest

__all__ = [
    "AnchorLayerStats",
    "MaskReuseCalibrationError",
    "MaskReuseObservation",
    "calibrate_mask_reuse_policy",
    "canonical_prefill_threshold_scale_factor",
    "load_mask_reuse_observations",
    "parse_mask_reuse_observations",
]


_FORMULA = "a * exp(b * target_sparsity)"
_SPLITS = frozenset({"calibration", "heldout"})
_OBSERVATION_FIELDS = frozenset(
    {
        "model",
        "min_kv_tokens",
        "max_kv_tokens",
        "target_sparsity",
        "sample_length",
        "threshold_lambda",
        "threshold_log2",
        "q_tokens",
        "kv_tokens",
        "q_start_tokens",
        "split",
        "prompt_id",
        "source_capture_sha256",
        "anchor_layer",
        "consumer_layer",
        "consumer_head",
        "donor_head",
        "retained_tiles",
        "eligible_tiles",
        "anchor_dropped_mass",
        "anchor_stats_by_layer",
        "dropped_mass",
    }
)
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
ObservationKey = tuple[str, str, float, int, int, int]
AnchorKey = tuple[str, str, float, int, int]


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


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise MaskReuseCalibrationError(f"{name} must be a lowercase SHA256")
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise MaskReuseCalibrationError(f"{name} must be a lowercase SHA256")
    return normalized


def _parse_anchor_stats_by_layer(
    value: object,
    *,
    require_canonical_string_keys: bool,
) -> dict[int, AnchorLayerStats]:
    if not isinstance(value, Mapping) or not value:
        raise MaskReuseCalibrationError("anchor_stats_by_layer must be a non-empty object")
    parsed: dict[int, AnchorLayerStats] = {}
    for raw_layer, raw_stats in value.items():
        if isinstance(raw_layer, bool):
            raise MaskReuseCalibrationError("anchor_stats_by_layer has a non-integer layer key")
        try:
            layer = int(raw_layer)
        except (TypeError, ValueError) as error:
            raise MaskReuseCalibrationError(
                "anchor_stats_by_layer has a non-integer layer key"
            ) from error
        if layer < 0 or (
            require_canonical_string_keys
            and (not isinstance(raw_layer, str) or raw_layer != str(layer))
        ):
            raise MaskReuseCalibrationError(
                f"anchor_stats_by_layer layer key {raw_layer!r} is not canonical"
            )
        if layer in parsed:
            raise MaskReuseCalibrationError(f"anchor_stats_by_layer repeats layer {layer}")
        if isinstance(raw_stats, AnchorLayerStats):
            raw_retained = raw_stats.retained_tiles
            raw_dropped = raw_stats.dropped_mass
        elif isinstance(raw_stats, Mapping):
            missing = {"retained_tiles", "dropped_mass"} - raw_stats.keys()
            extra = raw_stats.keys() - {"retained_tiles", "dropped_mass"}
            if missing or extra:
                raise MaskReuseCalibrationError(
                    f"anchor_stats_by_layer[{layer}] requires exactly retained_tiles "
                    f"and dropped_mass; missing={sorted(missing)}, extra={sorted(extra)}"
                )
            raw_retained = raw_stats["retained_tiles"]
            raw_dropped = raw_stats["dropped_mass"]
        else:
            raise MaskReuseCalibrationError(f"anchor_stats_by_layer[{layer}] must be an object")
        if not isinstance(raw_retained, list | tuple) or not raw_retained:
            raise MaskReuseCalibrationError(
                f"anchor_stats_by_layer[{layer}].retained_tiles must be a non-empty list"
            )
        if not isinstance(raw_dropped, list | tuple) or not raw_dropped:
            raise MaskReuseCalibrationError(
                f"anchor_stats_by_layer[{layer}].dropped_mass must be a non-empty list"
            )
        retained = tuple(
            _integer(
                item,
                f"anchor_stats_by_layer[{layer}].retained_tiles[{head}]",
                minimum=0,
            )
            for head, item in enumerate(raw_retained)
        )
        dropped = tuple(
            _number(
                item,
                f"anchor_stats_by_layer[{layer}].dropped_mass[{head}]",
                minimum=0.0,
                maximum=1.0,
            )
            for head, item in enumerate(raw_dropped)
        )
        if len(retained) != len(dropped):
            raise MaskReuseCalibrationError(
                f"anchor_stats_by_layer[{layer}] head arrays differ in width"
            )
        parsed[layer] = AnchorLayerStats(retained, dropped)
    return dict(sorted(parsed.items()))


@dataclass(frozen=True, slots=True)
class MaskReuseObservation:
    """One prompt, target-sparsity, consumer-head, and donor-head observation."""

    model: str
    min_kv_tokens: int
    max_kv_tokens: int | None
    target_sparsity: float
    sample_length: int
    threshold_lambda: float
    threshold_log2: float
    q_tokens: int
    kv_tokens: int
    q_start_tokens: int
    split: str
    prompt_id: str
    source_capture_sha256: str
    anchor_layer: int
    consumer_layer: int
    consumer_head: int
    donor_head: int
    retained_tiles: int
    eligible_tiles: int
    anchor_dropped_mass: float
    anchor_stats_by_layer: Mapping[int, AnchorLayerStats]
    dropped_mass: float

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise MaskReuseCalibrationError("model must be a non-empty string")
        if not isinstance(self.prompt_id, str) or not self.prompt_id.strip():
            raise MaskReuseCalibrationError("prompt_id must be a non-empty string")
        if self.split not in _SPLITS:
            raise MaskReuseCalibrationError(f"split must be one of {sorted(_SPLITS)}")
        minimum = _integer(self.min_kv_tokens, "min_kv_tokens", minimum=1)
        maximum = self.max_kv_tokens
        if maximum is not None:
            maximum = _integer(maximum, "max_kv_tokens", minimum=minimum)
        sample_length = _integer(self.sample_length, "sample_length", minimum=1)
        if sample_length < minimum or (maximum is not None and sample_length > maximum):
            raise MaskReuseCalibrationError("sample_length lies outside its context bucket")
        q_tokens = _integer(self.q_tokens, "q_tokens", minimum=129)
        if q_tokens > int(cast("int", _DEPLOYMENT_GEOMETRY_CONTRACT["max_query_chunk_tokens"])):
            raise MaskReuseCalibrationError("q_tokens exceeds the deployment geometry limit")
        kv_tokens = _integer(self.kv_tokens, "kv_tokens", minimum=1)
        q_start_tokens = _integer(self.q_start_tokens, "q_start_tokens", minimum=0)
        if sample_length != kv_tokens:
            raise MaskReuseCalibrationError("sample_length must equal kv_tokens")
        if q_start_tokens + q_tokens != kv_tokens:
            raise MaskReuseCalibrationError("q_start_tokens + q_tokens must equal kv_tokens")
        alignment = int(
            cast("int", _DEPLOYMENT_GEOMETRY_CONTRACT["query_chunk_start_alignment_tokens"])
        )
        if q_start_tokens % alignment:
            raise MaskReuseCalibrationError("q_start_tokens must be 128-token aligned")
        anchor_layer = _integer(self.anchor_layer, "anchor_layer", minimum=0)
        consumer_layer = _integer(self.consumer_layer, "consumer_layer", minimum=0)
        if anchor_layer >= consumer_layer:
            raise MaskReuseCalibrationError("anchor_layer must precede consumer_layer")
        _integer(self.consumer_head, "consumer_head", minimum=0)
        _integer(self.donor_head, "donor_head", minimum=0)
        retained = _integer(self.retained_tiles, "retained_tiles", minimum=0)
        eligible = _integer(self.eligible_tiles, "eligible_tiles", minimum=1)
        if retained > eligible:
            raise MaskReuseCalibrationError("retained_tiles must not exceed eligible_tiles")

        object.__setattr__(self, "model", self.model.strip())
        object.__setattr__(self, "prompt_id", self.prompt_id.strip())
        object.__setattr__(
            self,
            "source_capture_sha256",
            _sha256(self.source_capture_sha256, "source_capture_sha256"),
        )
        object.__setattr__(
            self,
            "target_sparsity",
            _number(
                self.target_sparsity,
                "target_sparsity",
                minimum=0.0,
                maximum=1.0,
                minimum_inclusive=False,
            ),
        )
        if self.target_sparsity >= 1.0:
            raise MaskReuseCalibrationError("target_sparsity must be in (0, 1)")
        object.__setattr__(
            self,
            "threshold_lambda",
            _number(
                self.threshold_lambda,
                "threshold_lambda",
                minimum=0.0,
                maximum=1.0,
                minimum_inclusive=False,
            ),
        )
        if self.threshold_lambda >= 1.0:
            raise MaskReuseCalibrationError("threshold_lambda must be in (0, 1)")
        object.__setattr__(
            self,
            "threshold_log2",
            _number(self.threshold_log2, "threshold_log2", minimum=-math.inf, maximum=0.0),
        )
        object.__setattr__(
            self,
            "anchor_dropped_mass",
            _number(self.anchor_dropped_mass, "anchor_dropped_mass", minimum=0.0, maximum=1.0),
        )
        object.__setattr__(
            self,
            "anchor_stats_by_layer",
            _parse_anchor_stats_by_layer(
                self.anchor_stats_by_layer,
                require_canonical_string_keys=False,
            ),
        )
        object.__setattr__(
            self,
            "dropped_mass",
            _number(self.dropped_mass, "dropped_mass", minimum=0.0, maximum=1.0),
        )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> MaskReuseObservation:
        """Build a validated observation from normalized JSON."""
        missing = _OBSERVATION_FIELDS - raw.keys()
        extra = raw.keys() - _OBSERVATION_FIELDS
        if missing or extra:
            raise MaskReuseCalibrationError(
                f"observation fields do not match the schema; "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        values = dict(raw)
        values["anchor_stats_by_layer"] = _parse_anchor_stats_by_layer(
            raw["anchor_stats_by_layer"],
            require_canonical_string_keys=True,
        )
        return cls(**values)  # type: ignore[arg-type]

    def to_mapping(self) -> dict[str, object]:
        """Return the normalized JSON representation."""
        result = {
            field: getattr(self, field)
            for field in _OBSERVATION_FIELDS
            if field != "anchor_stats_by_layer"
        }
        result["anchor_stats_by_layer"] = {
            str(layer): {
                "retained_tiles": list(stats.retained_tiles),
                "dropped_mass": list(stats.dropped_mass),
            }
            for layer, stats in self.anchor_stats_by_layer.items()
        }
        return result


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise MaskReuseCalibrationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def parse_mask_reuse_observations(lines: Iterable[str]) -> list[MaskReuseObservation]:
    """Parse strict normalized observation JSONL."""
    observations: list[MaskReuseObservation] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line, object_pairs_hook=_reject_duplicate_json_keys)
        except json.JSONDecodeError as error:
            raise MaskReuseCalibrationError(
                f"line {line_number}: invalid JSON: {error.msg}"
            ) from error
        except MaskReuseCalibrationError as error:
            raise MaskReuseCalibrationError(f"line {line_number}: {error}") from error
        if not isinstance(raw, dict):
            raise MaskReuseCalibrationError(f"line {line_number}: observation must be an object")
        try:
            observations.append(MaskReuseObservation.from_mapping(raw))
        except MaskReuseCalibrationError as error:
            raise MaskReuseCalibrationError(f"line {line_number}: {error}") from error
    if not observations:
        raise MaskReuseCalibrationError("input contains no mask-reuse observations")
    return observations


def load_mask_reuse_observations(path: str | Path) -> list[MaskReuseObservation]:
    """Load normalized mask-reuse observations from JSONL."""
    with Path(path).open(encoding="utf-8") as handle:
        return parse_mask_reuse_observations(handle)


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
    if not any(layer != anchor for layer, anchor in nearest.items()):
        raise MaskReuseCalibrationError("topology must contain at least one reuse layer")
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


@dataclass(frozen=True, slots=True)
class _BucketIndex:
    observations: Mapping[ObservationKey, MaskReuseObservation]
    prompts: Mapping[str, tuple[str, ...]]
    target_menus: Mapping[tuple[str, str], frozenset[float]]
    donor_menus: Mapping[tuple[str, str, float, ConsumerHead], frozenset[int]]
    eligible: Mapping[tuple[str, str, ConsumerHead], int]
    anchor_masks: Mapping[AnchorKey, tuple[int, int, float]]
    anchors: tuple[int, ...]


@dataclass(slots=True)
class _ReuseEvaluation:
    eligible_tiles: int = 0
    retained_tiles: int = 0
    sparse_observations: int = 0
    violations: int = 0
    dropped_mass_sum: float = 0.0
    worst_dropped_mass: float = 0.0

    def add(self, other: _ReuseEvaluation) -> None:
        self.eligible_tiles += other.eligible_tiles
        self.retained_tiles += other.retained_tiles
        self.sparse_observations += other.sparse_observations
        self.violations += other.violations
        self.dropped_mass_sum += other.dropped_mass_sum
        self.worst_dropped_mass = max(self.worst_dropped_mass, other.worst_dropped_mass)

    def to_mapping(self) -> dict[str, object]:
        return {
            "eligible_tiles": self.eligible_tiles,
            "retained_tiles": self.retained_tiles,
            "bmm1_tile_savings_fraction": (
                1.0 - self.retained_tiles / self.eligible_tiles if self.eligible_tiles else 0.0
            ),
            "sparse_head_prompt_observations": self.sparse_observations,
            "constraint_violation_count": self.violations,
            "constraint_violation_rate": (
                self.violations / self.sparse_observations if self.sparse_observations else 0.0
            ),
            "mean_dropped_mass": (
                self.dropped_mass_sum / self.sparse_observations
                if self.sparse_observations
                else 0.0
            ),
            "worst_dropped_mass": self.worst_dropped_mass,
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

    def to_mapping(self, *, exact: bool = False) -> dict[str, object]:
        return {
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


def _normalize_observations(
    values: Sequence[MaskReuseObservation | Mapping[str, object]],
) -> list[MaskReuseObservation]:
    if not values:
        raise MaskReuseCalibrationError("at least one observation is required")
    return [
        value
        if isinstance(value, MaskReuseObservation)
        else MaskReuseObservation.from_mapping(value)
        for value in values
    ]


def _validate_thresholds(
    observations: Sequence[MaskReuseObservation], threshold_scale_factor: Mapping[str, object]
) -> None:
    params = threshold_scale_factor["prefill"]
    assert isinstance(params, Mapping)
    a = float(params["a"])
    b = float(params["b"])
    lower = params.get("min_observed_sparsity")
    upper = params.get("max_observed_sparsity")
    for observation in observations:
        target = observation.target_sparsity
        if (lower is not None and target < float(lower)) or (
            upper is not None and target > float(upper)
        ):
            raise MaskReuseCalibrationError(
                f"target_sparsity={target} is outside the observed vanilla calibration range"
            )
        expected_log2 = (
            math.log2(a) + b * target * math.log2(math.e) - math.log2(observation.sample_length)
        )
        expected_lambda = 2.0**expected_log2
        if not 0.0 < expected_lambda < 1.0:
            raise MaskReuseCalibrationError(
                "vanilla calibration derives a threshold outside (0, 1) for "
                f"prompt={observation.prompt_id!r}, target_sparsity={target}"
            )
        if observation.threshold_log2.hex() != expected_log2.hex():
            raise MaskReuseCalibrationError(
                "threshold_log2 does not match the log-domain vanilla fit for "
                f"prompt={observation.prompt_id!r}, target_sparsity={target}: "
                f"observed={observation.threshold_log2.hex()}, "
                f"expected={expected_log2.hex()}"
            )
        if observation.threshold_lambda.hex() != expected_lambda.hex():
            raise MaskReuseCalibrationError(
                "threshold_lambda does not match "
                "exp2(log2(a) + b * target_sparsity * log2(e) - log2(sample_length)) for "
                f"prompt={observation.prompt_id!r}, target_sparsity={target}: "
                f"observed={observation.threshold_lambda.hex()}, "
                f"expected={expected_lambda.hex()}"
            )


def _validate_dataset(
    observations: Sequence[MaskReuseObservation],
    *,
    nearest: Mapping[int, int],
) -> tuple[
    str, int, tuple[Bucket, ...], tuple[ConsumerHead, ...], dict[Bucket, list[MaskReuseObservation]]
]:
    by_split = {
        split: [observation for observation in observations if observation.split == split]
        for split in _SPLITS
    }
    if any(not rows for rows in by_split.values()):
        raise MaskReuseCalibrationError("observations require calibration and heldout splits")
    models = {observation.model for observation in observations}
    if len(models) != 1:
        raise MaskReuseCalibrationError("observations must contain exactly one model")
    calibration_prompts = {row.prompt_id for row in by_split["calibration"]}
    heldout_prompts = {row.prompt_id for row in by_split["heldout"]}
    if calibration_prompts & heldout_prompts:
        raise MaskReuseCalibrationError("prompt IDs overlap calibration and heldout splits")
    calibration_sources = {row.source_capture_sha256 for row in by_split["calibration"]}
    heldout_sources = {row.source_capture_sha256 for row in by_split["heldout"]}
    if calibration_sources & heldout_sources:
        raise MaskReuseCalibrationError("source captures overlap calibration and heldout splits")

    consumer_to_anchor = {layer: anchor for layer, anchor in nearest.items() if layer != anchor}
    max_head = max(max(row.consumer_head, row.donor_head) for row in observations)
    global_num_heads = max_head + 1
    targets = tuple(
        (layer, head) for layer in sorted(consumer_to_anchor) for head in range(global_num_heads)
    )
    expected_targets = set(targets)
    seen: set[tuple[object, ...]] = set()
    capture_sources: dict[tuple[str, str, Bucket], str] = {}
    by_bucket: dict[Bucket, list[MaskReuseObservation]] = defaultdict(list)
    for row in observations:
        expected_anchor = consumer_to_anchor.get(row.consumer_layer)
        if expected_anchor != row.anchor_layer:
            raise MaskReuseCalibrationError(
                f"consumer layer {row.consumer_layer} does not match the explicit topology"
            )
        bucket = (row.min_kv_tokens, row.max_kv_tokens)
        by_bucket[bucket].append(row)
        identity = (
            row.model,
            bucket,
            row.split,
            row.prompt_id,
            row.target_sparsity,
            row.consumer_layer,
            row.consumer_head,
            row.donor_head,
        )
        if identity in seen:
            raise MaskReuseCalibrationError("observations contain a duplicate candidate row")
        seen.add(identity)
        capture = (row.split, row.prompt_id, bucket)
        previous_source = capture_sources.setdefault(capture, row.source_capture_sha256)
        if previous_source != row.source_capture_sha256:
            raise MaskReuseCalibrationError("one prompt/context capture has multiple fingerprints")

    split_buckets = {
        split: {(row.min_kv_tokens, row.max_kv_tokens) for row in rows}
        for split, rows in by_split.items()
    }
    if split_buckets["calibration"] != split_buckets["heldout"]:
        raise MaskReuseCalibrationError("heldout context buckets must match calibration buckets")
    buckets = tuple(sorted(split_buckets["calibration"], key=_bucket_key))
    previous_max: int | None = 0
    for minimum, maximum in buckets:
        if previous_max is None or minimum <= previous_max:
            raise MaskReuseCalibrationError("context buckets must be ordered and non-overlapping")
        previous_max = maximum
    for bucket in buckets:
        for split in _SPLITS:
            actual = {
                (row.consumer_layer, row.consumer_head)
                for row in by_bucket[bucket]
                if row.split == split
            }
            if actual != expected_targets:
                raise MaskReuseCalibrationError(
                    f"{split} bucket {bucket} does not cover every consumer head"
                )
    return next(iter(models)), global_num_heads, buckets, targets, dict(by_bucket)


def _index_bucket(
    rows: Sequence[MaskReuseObservation],
    *,
    anchors: tuple[int, ...],
    global_num_heads: int,
    targets: Sequence[ConsumerHead],
) -> _BucketIndex:
    observations: dict[ObservationKey, MaskReuseObservation] = {}
    prompts: dict[str, set[str]] = defaultdict(set)
    target_menus: dict[tuple[str, str], set[float]] = defaultdict(set)
    donor_menus: dict[tuple[str, str, float, ConsumerHead], set[int]] = defaultdict(set)
    eligible: dict[tuple[str, str, ConsumerHead], int] = {}
    anchor_masks: dict[AnchorKey, tuple[int, int, float]] = {}
    prompt_targets: dict[tuple[str, str, float], tuple[int, float, float]] = {}
    anchor_payloads: dict[tuple[str, str, float], Mapping[int, AnchorLayerStats]] = {}
    payload_eligible: dict[tuple[str, str, float], int] = {}
    for row in rows:
        target = (row.consumer_layer, row.consumer_head)
        key: ObservationKey = (
            row.split,
            row.prompt_id,
            row.target_sparsity,
            row.consumer_layer,
            row.consumer_head,
            row.donor_head,
        )
        observations[key] = row
        prompts[row.split].add(row.prompt_id)
        target_menus[(row.split, row.prompt_id)].add(row.target_sparsity)
        donor_menus[(row.split, row.prompt_id, row.target_sparsity, target)].add(row.donor_head)
        eligible_key = (row.split, row.prompt_id, target)
        previous_eligible = eligible.setdefault(eligible_key, row.eligible_tiles)
        if previous_eligible != row.eligible_tiles:
            raise MaskReuseCalibrationError("eligible_tiles differs across candidates")
        prompt_target = (row.split, row.prompt_id, row.target_sparsity)
        sample_and_threshold = (
            row.sample_length,
            row.threshold_lambda,
            row.threshold_log2,
        )
        if prompt_targets.setdefault(prompt_target, sample_and_threshold) != sample_and_threshold:
            raise MaskReuseCalibrationError(
                "sample_length, threshold_lambda, or threshold_log2 differs within a capture"
            )
        previous_payload = anchor_payloads.setdefault(prompt_target, row.anchor_stats_by_layer)
        if previous_payload != row.anchor_stats_by_layer:
            raise MaskReuseCalibrationError(
                "anchor_stats_by_layer differs across repeated candidate rows"
            )
        previous_payload_eligible = payload_eligible.setdefault(prompt_target, row.eligible_tiles)
        if previous_payload_eligible != row.eligible_tiles:
            raise MaskReuseCalibrationError(
                "eligible_tiles differs within one capture and target_sparsity"
            )

    expected_anchors = set(anchors)
    for prompt_target, payload in anchor_payloads.items():
        actual_anchors = set(payload)
        if actual_anchors != expected_anchors:
            raise MaskReuseCalibrationError(
                "anchor_stats_by_layer does not exactly cover topology anchors; "
                f"missing={sorted(expected_anchors - actual_anchors)}, "
                f"extra={sorted(actual_anchors - expected_anchors)}"
            )
        eligible_tiles = payload_eligible[prompt_target]
        for anchor, stats in payload.items():
            if len(stats.retained_tiles) != global_num_heads:
                raise MaskReuseCalibrationError(
                    f"anchor_stats_by_layer[{anchor}] has head width "
                    f"{len(stats.retained_tiles)}, expected {global_num_heads}"
                )
            for head, (retained, dropped) in enumerate(
                zip(stats.retained_tiles, stats.dropped_mass, strict=True)
            ):
                if retained > eligible_tiles:
                    raise MaskReuseCalibrationError(
                        f"anchor_stats_by_layer[{anchor}].retained_tiles[{head}] "
                        "exceeds eligible_tiles"
                    )
                split, prompt, target_sparsity = prompt_target
                anchor_masks[(split, prompt, target_sparsity, anchor, head)] = (
                    retained,
                    eligible_tiles,
                    dropped,
                )

    for row in rows:
        stats = row.anchor_stats_by_layer[row.anchor_layer]
        payload_candidate = (
            stats.retained_tiles[row.donor_head],
            stats.dropped_mass[row.donor_head],
        )
        if payload_candidate != (row.retained_tiles, row.anchor_dropped_mass):
            raise MaskReuseCalibrationError(
                "candidate retained_tiles/anchor_dropped_mass does not match anchor_stats_by_layer"
            )

    first_prompt = min(prompts["calibration"])
    expected_targets = target_menus[("calibration", first_prompt)]
    full_donors = set(range(global_num_heads))
    for split in _SPLITS:
        for prompt in prompts[split]:
            if target_menus[(split, prompt)] != expected_targets:
                raise MaskReuseCalibrationError("target_sparsity menu differs across captures")
            for target_sparsity in expected_targets:
                for target in targets:
                    if donor_menus[(split, prompt, target_sparsity, target)] != full_donors:
                        raise MaskReuseCalibrationError("candidate donor menu is incomplete")
                for anchor in anchors:
                    for head in range(global_num_heads):
                        if (split, prompt, target_sparsity, anchor, head) not in anchor_masks:
                            raise MaskReuseCalibrationError(
                                "anchor/head mask observations are incomplete"
                            )
    return _BucketIndex(
        observations=observations,
        prompts={split: tuple(sorted(values)) for split, values in prompts.items()},
        target_menus={key: frozenset(values) for key, values in target_menus.items()},
        donor_menus={key: frozenset(values) for key, values in donor_menus.items()},
        eligible=eligible,
        anchor_masks=anchor_masks,
        anchors=anchors,
    )


def _evaluate_anchor(
    index: _BucketIndex,
    *,
    split: str,
    target_sparsity: float | None,
    global_num_heads: int,
    maximum: float,
) -> _AnchorEvaluation:
    result = _AnchorEvaluation()
    for prompt in index.prompts[split]:
        selected_target = (
            min(index.target_menus[(split, prompt)]) if target_sparsity is None else target_sparsity
        )
        values: list[float] = []
        retained_tiles = 0
        eligible_tiles = 0
        for anchor in index.anchors:
            for head in range(global_num_heads):
                retained, eligible, dropped = index.anchor_masks[
                    (split, prompt, selected_target, anchor, head)
                ]
                values.append(0.0 if target_sparsity is None else dropped)
                retained_tiles += eligible if target_sparsity is None else retained
                eligible_tiles += eligible
        prompt_mean = sum(values) / len(values)
        result.eligible_tiles += eligible_tiles
        result.retained_tiles += retained_tiles
        result.prompt_count += 1
        result.prompt_mean_sum += prompt_mean
        result.worst_prompt_mean = max(result.worst_prompt_mean, prompt_mean)
        result.violations += int(prompt_mean > maximum)
    return result


def _select_bucket(
    index: _BucketIndex,
    *,
    targets: Sequence[ConsumerHead],
    global_num_heads: int,
    max_anchor_dropped_mass: float,
    max_reuse_selection_dropped_mass: float,
) -> _Selection:
    prompts = index.prompts["calibration"]
    target_menu = tuple(sorted(index.target_menus[("calibration", prompts[0])]))
    frontier: list[Mapping[str, object]] = []
    candidates: list[tuple[tuple[object, ...], _Selection]] = []
    for target_sparsity in target_menu:
        anchor = _evaluate_anchor(
            index,
            split="calibration",
            target_sparsity=target_sparsity,
            global_num_heads=global_num_heads,
            maximum=max_anchor_dropped_mass,
        )
        choices: dict[ConsumerHead, _Choice] = {}
        total_retained = 0
        fallback_count = 0
        for target in targets:
            feasible: list[tuple[int, int]] = []
            for donor in range(global_num_heads):
                candidate_rows = [
                    index.observations[
                        (
                            "calibration",
                            prompt,
                            target_sparsity,
                            target[0],
                            target[1],
                            donor,
                        )
                    ]
                    for prompt in prompts
                ]
                if all(
                    row.dropped_mass <= max_reuse_selection_dropped_mass for row in candidate_rows
                ):
                    feasible.append((sum(row.retained_tiles for row in candidate_rows), donor))
            if feasible:
                retained, donor = min(feasible)
                choice = _Choice(donor, False, retained)
            else:
                fallback_count += 1
                retained = sum(
                    index.eligible[("calibration", prompt, target)] for prompt in prompts
                )
                choice = _Choice(0, True, retained)
            choices[target] = choice
            total_retained += retained
        signature = tuple(choices[target].donor_head for target in targets)
        combined_tile_cost = 2 * total_retained + anchor.retained_tiles
        frontier.append(
            {
                "target_sparsity": target_sparsity,
                "anchor_safe": anchor.violations == 0,
                "retained_reuse_tiles": total_retained,
                "retained_anchor_tiles": anchor.retained_tiles,
                "combined_tile_cost": combined_tile_cost,
                "fallback_head_count": fallback_count,
                "anchor_calibration": anchor.to_mapping(),
            }
        )
        if anchor.violations == 0:
            rank = (
                combined_tile_cost,
                fallback_count,
                target_sparsity,
                signature,
            )
            candidates.append((rank, _Selection(target_sparsity, choices, ())))
    if not candidates:
        dense_choices = {
            target: _Choice(
                0, True, sum(index.eligible[("calibration", prompt, target)] for prompt in prompts)
            )
            for target in targets
        }
        return _Selection(
            None,
            dense_choices,
            tuple(frontier),
            "no_target_sparsity_satisfied_anchor_calibration_constraint",
        )
    _, selected = min(candidates, key=lambda item: item[0])
    return _Selection(selected.target_sparsity, selected.choices, tuple(frontier))


def _evaluate_reuse(
    selection: _Selection,
    index: _BucketIndex,
    *,
    split: str,
    maximum: float,
) -> _ReuseEvaluation:
    result = _ReuseEvaluation()
    for prompt in index.prompts[split]:
        for target, choice in sorted(selection.choices.items()):
            eligible = index.eligible[(split, prompt, target)]
            result.eligible_tiles += eligible
            if choice.fallback:
                result.retained_tiles += eligible
                continue
            assert selection.target_sparsity is not None
            row = index.observations[
                (
                    split,
                    prompt,
                    selection.target_sparsity,
                    target[0],
                    target[1],
                    choice.donor_head,
                )
            ]
            result.retained_tiles += row.retained_tiles
            result.sparse_observations += 1
            result.dropped_mass_sum += row.dropped_mass
            result.worst_dropped_mass = max(result.worst_dropped_mass, row.dropped_mass)
            result.violations += int(row.dropped_mass > maximum)
    return result


def _canonical_digest(observations: Sequence[MaskReuseObservation]) -> str:
    rows = [
        json.dumps(row.to_mapping(), sort_keys=True, separators=(",", ":")).encode()
        for row in observations
    ]
    digest = sha256()
    for row in sorted(rows):
        digest.update(sha256(row).digest())
    return digest.hexdigest()


def _deployment_geometry(
    observations: Sequence[MaskReuseObservation],
) -> dict[str, object]:
    by_capture: dict[tuple[object, ...], tuple[int, int, int]] = {}
    for row in observations:
        identity = (
            row.split,
            row.prompt_id,
            row.source_capture_sha256,
            row.min_kv_tokens,
            row.max_kv_tokens,
        )
        geometry = (row.q_tokens, row.kv_tokens, row.q_start_tokens)
        if by_capture.setdefault(identity, geometry) != geometry:
            raise MaskReuseCalibrationError(
                "q_tokens, kv_tokens, or q_start_tokens differs within one capture"
            )
    geometry_rows = [
        {
            "split": identity[0],
            "prompt_id": identity[1],
            "source_capture_sha256": identity[2],
            "min_kv_tokens": identity[3],
            "max_kv_tokens": identity[4],
            "q_tokens": geometry[0],
            "kv_tokens": geometry[1],
            "q_start_tokens": geometry[2],
        }
        for identity, geometry in sorted(by_capture.items(), key=lambda item: str(item[0]))
    ]
    return {
        "contract": dict(_DEPLOYMENT_GEOMETRY_CONTRACT),
        "observations": geometry_rows,
    }


def _canonical_evidence(raw: Mapping[str, object]) -> dict[str, str]:
    missing = _EVIDENCE_FIELDS - raw.keys()
    extra = raw.keys() - _EVIDENCE_FIELDS
    if missing or extra:
        raise MaskReuseCalibrationError(
            f"evidence fields do not match schema v3; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return {field: _sha256(raw[field], f"evidence.{field}") for field in sorted(raw)}


def calibrate_mask_reuse_policy(
    observations: Sequence[MaskReuseObservation | Mapping[str, object]],
    *,
    vanilla_calibration: Mapping[str, object],
    topology: Mapping[str, object],
    checkpoint_manifest: VerifiedCheckpointManifest,
    evidence: Mapping[str, object],
    max_anchor_dropped_mass: float,
    max_reuse_dropped_mass: float,
    max_reuse_selection_dropped_mass: float | None = None,
    source_provenance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Select and evaluate a fail-closed schema-v3 legacy candidate."""
    if not isinstance(checkpoint_manifest, VerifiedCheckpointManifest):
        raise MaskReuseCalibrationError(
            "checkpoint_manifest must be returned by verify_checkpoint_manifest"
        )
    rows = _normalize_observations(observations)
    threshold_scale_factor = canonical_prefill_threshold_scale_factor(vanilla_calibration)
    prefill_params = threshold_scale_factor["prefill"]
    assert isinstance(prefill_params, Mapping)
    if not {"min_observed_sparsity", "max_observed_sparsity"} <= prefill_params.keys():
        raise MaskReuseCalibrationError(
            "schema-v3 export requires observed prefill sparsity bounds"
        )
    _validate_thresholds(rows, threshold_scale_factor)
    anchors, nearest = _normalize_topology(topology)
    checkpoint_identity = checkpoint_manifest.sha256
    geometry = _deployment_geometry(rows)
    canonical_evidence = _canonical_evidence(evidence)
    anchor_bound = _number(
        max_anchor_dropped_mass, "max_anchor_dropped_mass", minimum=0.0, maximum=1.0
    )
    reuse_bound = _number(
        max_reuse_dropped_mass, "max_reuse_dropped_mass", minimum=0.0, maximum=1.0
    )
    selection_bound = (
        reuse_bound
        if max_reuse_selection_dropped_mass is None
        else _number(
            max_reuse_selection_dropped_mass,
            "max_reuse_selection_dropped_mass",
            minimum=0.0,
            maximum=1.0,
        )
    )
    if selection_bound > reuse_bound:
        raise MaskReuseCalibrationError(
            "max_reuse_selection_dropped_mass must not exceed max_reuse_dropped_mass"
        )
    model, global_num_heads, buckets, targets, by_bucket = _validate_dataset(rows, nearest=nearest)
    if model != checkpoint_manifest.model:
        raise MaskReuseCalibrationError(
            "observation model does not match the verified checkpoint manifest"
        )

    context_policies: list[dict[str, object]] = []
    bucket_reports: list[dict[str, object]] = []
    target_menus: list[dict[str, object]] = []
    overall_reuse_calibration = _ReuseEvaluation()
    overall_reuse_heldout = _ReuseEvaluation()
    overall_anchor_calibration = _AnchorEvaluation()
    overall_anchor_heldout = _AnchorEvaluation()
    total_fallback = 0
    for bounds in buckets:
        index = _index_bucket(
            by_bucket[bounds],
            anchors=anchors,
            global_num_heads=global_num_heads,
            targets=targets,
        )
        selection = _select_bucket(
            index,
            targets=targets,
            global_num_heads=global_num_heads,
            max_anchor_dropped_mass=anchor_bound,
            max_reuse_selection_dropped_mass=selection_bound,
        )
        if selection.target_sparsity is not None and bounds[1] is None:
            raise MaskReuseCalibrationError(
                "a deployment-qualified sparse context bucket requires a finite maximum"
            )
        reuse_calibration = _evaluate_reuse(
            selection, index, split="calibration", maximum=reuse_bound
        )
        reuse_heldout = _evaluate_reuse(selection, index, split="heldout", maximum=reuse_bound)
        anchor_calibration = _evaluate_anchor(
            index,
            split="calibration",
            target_sparsity=selection.target_sparsity,
            global_num_heads=global_num_heads,
            maximum=anchor_bound,
        )
        anchor_heldout = _evaluate_anchor(
            index,
            split="heldout",
            target_sparsity=selection.target_sparsity,
            global_num_heads=global_num_heads,
            maximum=anchor_bound,
        )
        overall_reuse_calibration.add(reuse_calibration)
        overall_reuse_heldout.add(reuse_heldout)
        overall_anchor_calibration.add(anchor_calibration)
        overall_anchor_heldout.add(anchor_heldout)

        policy: dict[str, object]
        if selection.target_sparsity is None:
            policy = {
                "min_kv_tokens": bounds[0],
                "max_kv_tokens": bounds[1],
                "exact": True,
            }
            headmaps: dict[str, list[int]] = {}
            fallback_heads: dict[str, list[int]] = {}
        else:
            headmaps = {
                str(layer): [
                    selection.choices[(layer, head)].donor_head for head in range(global_num_heads)
                ]
                for layer in sorted({target[0] for target in targets})
            }
            fallback_heads = {
                str(layer): [
                    head
                    for head in range(global_num_heads)
                    if selection.choices[(layer, head)].fallback
                ]
                for layer in sorted({target[0] for target in targets})
            }
            policy = {
                "min_kv_tokens": bounds[0],
                "max_kv_tokens": bounds[1],
                "target_sparsity": selection.target_sparsity,
                "headmaps": headmaps,
                "fallback_heads": fallback_heads,
            }
        context_policies.append(policy)
        fallback_count = sum(len(heads) for heads in fallback_heads.values())
        total_fallback += fallback_count
        menu = [row["target_sparsity"] for row in selection.frontier]
        target_menus.append(
            {
                "min_kv_tokens": bounds[0],
                "max_kv_tokens": bounds[1],
                "target_sparsities": menu,
            }
        )
        bucket_reports.append(
            {
                "min_kv_tokens": bounds[0],
                "max_kv_tokens": bounds[1],
                "selected_target_sparsity": selection.target_sparsity,
                "selection_status": "sparse"
                if selection.target_sparsity is not None
                else selection.exact_reason,
                "target_sparsity_frontier": list(selection.frontier),
                "fallback_head_count": fallback_count,
                "reuse_calibration": reuse_calibration.to_mapping(),
                "reuse_heldout": reuse_heldout.to_mapping(),
                "anchor_calibration": anchor_calibration.to_mapping(
                    exact=selection.target_sparsity is None
                ),
                "anchor_heldout": anchor_heldout.to_mapping(
                    exact=selection.target_sparsity is None
                ),
            }
        )

    provenance: dict[str, object] = {
        "calibrator": "modelopt.mask_reuse",
        "observation_schema_version": 1,
        "input_observation_count": len(rows),
        "canonical_input_sha256": _canonical_digest(rows),
        "calibration_prompt_ids": sorted(
            {row.prompt_id for row in rows if row.split == "calibration"}
        ),
        "heldout_prompt_ids": sorted({row.prompt_id for row in rows if row.split == "heldout"}),
        "selection_split": "calibration",
        "evaluation_split": "heldout",
        "threshold_semantics": "a * exp(b * target_sparsity) / sample_length",
        "threshold_implementation": {
            "threshold_log2": ("log2(a) + b * target_sparsity * log2(e) - log2(sample_length)"),
            "threshold_lambda": "exp2(threshold_log2)",
            "validation": "exact IEEE-754 binary64 hex equality",
        },
        "checkpoint_manifest_sha256": checkpoint_identity,
        "target_sparsity_menus": target_menus,
        "constraints": {
            "anchor": {
                "metric": "worst_prompt_mean_anchor_dropped_mass",
                "comparison": "<=",
                "maximum": anchor_bound,
            },
            "reuse": {
                "metric": "per_prompt_candidate_reuse_dropped_mass",
                "comparison": "<=",
                "selection_maximum": selection_bound,
                "evaluation_maximum": reuse_bound,
            },
        },
        "tie_breaks": [
            "reject target sparsities exceeding the calibration anchor bound",
            "minimum equal-BMM combined tile cost 2*A_R + A_A",
            "fewest exact-fallback consumer heads",
            "smallest target sparsity",
            "lexicographically smallest donor-head map",
        ],
        "selection_cost": {
            "formula": "2 * retained_reuse_tiles + retained_anchor_tiles",
            "bmm1_weight": 1.0,
            "bmm2_weight": 1.0,
        },
    }
    if source_provenance is not None:
        provenance["source"] = dict(source_provenance)

    report = {
        "model": model,
        "checkpoint_manifest_sha256": checkpoint_identity,
        "constraints": provenance["constraints"],
        "selection_unit": "context_bucket",
        "by_bucket": bucket_reports,
        "overall": {
            "consumer_head_bucket_count": len(targets) * len(buckets),
            "fallback_head_bucket_count": total_fallback,
            "fallback_fraction": total_fallback / (len(targets) * len(buckets)),
            "reuse_calibration": overall_reuse_calibration.to_mapping(),
            "reuse_heldout": overall_reuse_heldout.to_mapping(),
            "anchor_calibration": overall_anchor_calibration.to_mapping(),
            "anchor_heldout": overall_anchor_heldout.to_mapping(),
        },
        "promotion": {
            "status": "candidate_only",
            "eligible": False,
            "reasons": [
                "legacy observations are not capture-schema-v2 checkpoint-bound records",
                "grouped inner-fold and preregistered outer gates were not evaluated",
                "deployment rectangular-geometry promotion gate was not evaluated",
            ],
        },
    }
    return {
        "version": 3,
        "promotion_status": "candidate_only",
        "phase": "prefill",
        "decode": {"mode": "dense"},
        "calibration_protocol": _CALIBRATION_PROTOCOL,
        "producer": {"name": "modelopt", "version": modelopt.__version__},
        "evidence": canonical_evidence,
        "threshold_scale_factor": threshold_scale_factor,
        "model": model,
        "checkpoint_manifest_sha256": checkpoint_identity,
        "global_num_heads": global_num_heads,
        "anchors": list(anchors),
        "nearest": {str(layer): anchor for layer, anchor in nearest.items()},
        "deployment_geometry_validated": False,
        "deployment_geometry": geometry,
        "context_policies": context_policies,
        "provenance": provenance,
        "calibration_report": report,
    }
