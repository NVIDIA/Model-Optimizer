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

"""Development-only joint topology selection for cross-layer mask reuse.

Topology discovery treats every earlier attention layer as a potential mask
donor.  It jointly selects one global anchor topology, one target sparsity per
context bucket, and per-consumer-head donor/fallback choices.  Only development
captures participate in selection; held-out captures evaluate the frozen
result.

The selected topology is intentionally a candidate artifact.  It must be fed
back into the ordinary fixed-topology compact collector and calibrator before
deployment.  This keeps the serving topology global while retaining the
existing context-bucket policy contract.
"""

from __future__ import annotations

import itertools
import json
import math
from collections import defaultdict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import pulp

import modelopt

from .checkpoint_manifest import VerifiedCheckpointManifest
from .mask_reuse import (
    _SOLVER_LEXICOGRAPHIC_ATOL,
    AnchorLayerStats,
    Bucket,
    MaskReuseCalibrationError,
    _AnchorEvaluation,
    _bucket_key,
    _Choice,
    _number,
    _ReuseEvaluation,
    _sha256,
    canonical_prefill_threshold_scale_factor,
)
from .mask_reuse_compact import (
    _ANCHOR_FIELDS,
    _INVOCATION_FIELDS,
    _MONOTONIC_ATOL,
    _PARTITIONS,
    _SPLITS,
    _eligible_tiles,
    _exact_fields,
    _float_hex,
    _geometry,
    _integer,
    _reject_duplicate_json_keys,
    _text,
)

__all__ = [
    "TopologyDiscoveryCapture",
    "TopologyDiscoveryCaptureSource",
    "calibrate_mask_reuse_topology",
    "load_topology_discovery_captures",
]

_CAPTURE_FIELDS = frozenset(
    {
        "topology_discovery_capture_schema_version",
        "invocation",
        "geometry",
        "global_num_heads",
        "eligible_tiles",
        "attention_layers",
        "max_reuse_span",
        "anchor_stats_by_layer",
        "consumer_candidates_by_layer",
    }
)
_CANDIDATE_FIELDS = frozenset({"dropped_mass"})
_MAX_TARGET_COMBINATIONS = 4096


def _canonical_layer_key(value: object, label: str) -> int:
    if not isinstance(value, str) or not value.isdigit() or value != str(int(value)):
        raise MaskReuseCalibrationError(f"{label} keys must be canonical nonnegative integers")
    return int(value)


def _matrix(raw: object, *, heads: int, label: str) -> tuple[tuple[float, ...], ...]:
    if not isinstance(raw, list) or len(raw) != heads:
        raise MaskReuseCalibrationError(f"{label} must have {heads} consumer rows")
    rows = []
    for consumer_head, row in enumerate(raw):
        if not isinstance(row, list) or len(row) != heads:
            raise MaskReuseCalibrationError(
                f"{label}[{consumer_head}] must cover every global donor head"
            )
        rows.append(
            tuple(
                _number(
                    value,
                    f"{label}[{consumer_head}][{donor_head}]",
                    minimum=0.0,
                    maximum=1.0,
                )
                for donor_head, value in enumerate(row)
            )
        )
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class TopologyDiscoveryCapture:
    """One prompt/target all-earlier-layer topology-discovery capture."""

    model: str
    checkpoint_manifest_sha256: str
    split: str
    partition: str
    inner_fold: int | None
    prompt_id: str
    source: str
    source_group_sha256: str
    source_capture_sha256: str
    min_kv_tokens: int
    max_kv_tokens: int | None
    target_sparsity: float
    sample_length: int
    threshold_log2: float
    threshold_lambda: float
    geometry: Mapping[str, int]
    global_num_heads: int
    eligible_tiles: int
    attention_layers: tuple[int, ...]
    max_reuse_span: int
    anchor_stats_by_layer: Mapping[int, AnchorLayerStats]
    consumer_candidates_by_layer: Mapping[int, Mapping[int, tuple[tuple[float, ...], ...]]]

    @property
    def bucket(self) -> Bucket:
        """Return this capture's context bounds."""
        return self.min_kv_tokens, self.max_kv_tokens

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> TopologyDiscoveryCapture:
        """Parse one strict topology-discovery capture object."""
        _exact_fields(raw, _CAPTURE_FIELDS, "topology discovery capture")
        if raw["topology_discovery_capture_schema_version"] != 1:
            raise MaskReuseCalibrationError("topology_discovery_capture_schema_version must be 1")
        invocation = raw["invocation"]
        if not isinstance(invocation, Mapping):
            raise MaskReuseCalibrationError("capture invocation must be an object")
        _exact_fields(invocation, _INVOCATION_FIELDS, "topology discovery invocation")
        if invocation["capture_schema_version"] != 2:
            raise MaskReuseCalibrationError("capture_schema_version must be 2")
        split = invocation["split"]
        if split not in _SPLITS:
            raise MaskReuseCalibrationError("capture split must be calibration or heldout")
        partition = invocation["partition"]
        if partition not in _PARTITIONS:
            raise MaskReuseCalibrationError("capture partition must be development or outer_test")
        if split != ("calibration" if partition == "development" else "heldout"):
            raise MaskReuseCalibrationError("capture split and partition disagree")
        raw_fold = invocation["inner_fold"]
        if partition == "development":
            inner_fold = _integer(raw_fold, "inner_fold")
        elif raw_fold is not None:
            raise MaskReuseCalibrationError("outer_test capture must have null inner_fold")
        else:
            inner_fold = None

        minimum = _integer(invocation["min_kv_tokens"], "min_kv_tokens", minimum=1)
        maximum = invocation["max_kv_tokens"]
        if maximum is not None:
            maximum = _integer(maximum, "max_kv_tokens", minimum=minimum)
        sample_length = _integer(invocation["sample_length"], "sample_length", minimum=1)
        if sample_length < minimum or (maximum is not None and sample_length > maximum):
            raise MaskReuseCalibrationError("sample_length lies outside its context bucket")
        target = _float_hex(invocation["target_sparsity_hex"], "target_sparsity_hex")
        threshold_log2 = _float_hex(invocation["threshold_log2_hex"], "threshold_log2_hex")
        threshold_lambda = _float_hex(invocation["threshold_lambda_hex"], "threshold_lambda_hex")
        if not 0.0 < target < 1.0:
            raise MaskReuseCalibrationError("target_sparsity must be in (0, 1)")
        if threshold_log2 >= 0.0 or not 0.0 < threshold_lambda < 1.0:
            raise MaskReuseCalibrationError("threshold must be in (0, 1)")
        if (2.0**threshold_log2).hex() != threshold_lambda.hex():
            raise MaskReuseCalibrationError("threshold lambda and log2 fields disagree")

        expected_geometry = _geometry(invocation["expected_geometry"], "expected_geometry")
        geometry = _geometry(raw["geometry"], "geometry")
        if expected_geometry != geometry or geometry["kv_tokens"] != sample_length:
            raise MaskReuseCalibrationError("capture geometry does not match its invocation")
        heads = _integer(raw["global_num_heads"], "global_num_heads", minimum=1)
        eligible = _integer(raw["eligible_tiles"], "eligible_tiles", minimum=1)
        if eligible != _eligible_tiles(geometry):
            raise MaskReuseCalibrationError(
                "eligible_tiles does not match 128x128 bottom-right causal geometry"
            )

        raw_layers = raw["attention_layers"]
        if not isinstance(raw_layers, list) or not raw_layers:
            raise MaskReuseCalibrationError("attention_layers must be a non-empty array")
        layers = tuple(
            _integer(layer, f"attention_layers[{index}]") for index, layer in enumerate(raw_layers)
        )
        if tuple(sorted(set(layers))) != layers:
            raise MaskReuseCalibrationError(
                "attention_layers must be strictly increasing and unique"
            )
        if len(layers) < 2:
            raise MaskReuseCalibrationError(
                "topology discovery requires at least two attention layers"
            )
        max_reuse_span = _integer(raw["max_reuse_span"], "max_reuse_span", minimum=1)
        if max_reuse_span >= len(layers):
            raise MaskReuseCalibrationError(
                "max_reuse_span must be smaller than the attention-layer count"
            )

        raw_anchors = raw["anchor_stats_by_layer"]
        if not isinstance(raw_anchors, Mapping):
            raise MaskReuseCalibrationError("anchor_stats_by_layer must be an object")
        anchors: dict[int, AnchorLayerStats] = {}
        for raw_layer, raw_stats in raw_anchors.items():
            layer = _canonical_layer_key(raw_layer, "anchor_stats_by_layer")
            if not isinstance(raw_stats, Mapping):
                raise MaskReuseCalibrationError(f"anchor_stats_by_layer[{layer}] must be an object")
            _exact_fields(raw_stats, _ANCHOR_FIELDS, f"anchor_stats_by_layer[{layer}]")
            retained_raw = raw_stats["retained_tiles"]
            dropped_raw = raw_stats["dropped_mass"]
            if not isinstance(retained_raw, list) or not isinstance(dropped_raw, list):
                raise MaskReuseCalibrationError(f"anchor_stats_by_layer[{layer}] must use arrays")
            if len(retained_raw) != heads or len(dropped_raw) != heads:
                raise MaskReuseCalibrationError(
                    f"anchor_stats_by_layer[{layer}] does not cover all global heads"
                )
            retained = tuple(
                _integer(value, f"anchor {layer} retained[{head}]")
                for head, value in enumerate(retained_raw)
            )
            if any(value > eligible for value in retained):
                raise MaskReuseCalibrationError(f"anchor {layer} retained tiles exceed eligible")
            dropped = tuple(
                _number(
                    value,
                    f"anchor {layer} dropped[{head}]",
                    minimum=0.0,
                    maximum=1.0,
                )
                for head, value in enumerate(dropped_raw)
            )
            anchors[layer] = AnchorLayerStats(retained, dropped)
        if set(anchors) != set(layers):
            raise MaskReuseCalibrationError(
                "anchor_stats_by_layer must cover every attention layer"
            )

        raw_candidates = raw["consumer_candidates_by_layer"]
        if not isinstance(raw_candidates, Mapping):
            raise MaskReuseCalibrationError("consumer_candidates_by_layer must be an object")
        candidates: dict[int, dict[int, tuple[tuple[float, ...], ...]]] = {}
        for raw_consumer, raw_by_anchor in raw_candidates.items():
            consumer = _canonical_layer_key(raw_consumer, "consumer_candidates_by_layer")
            if not isinstance(raw_by_anchor, Mapping):
                raise MaskReuseCalibrationError(
                    f"consumer_candidates_by_layer[{consumer}] must be an object"
                )
            by_anchor: dict[int, tuple[tuple[float, ...], ...]] = {}
            for raw_anchor, raw_stats in raw_by_anchor.items():
                anchor = _canonical_layer_key(
                    raw_anchor, f"consumer_candidates_by_layer[{consumer}]"
                )
                if not isinstance(raw_stats, Mapping):
                    raise MaskReuseCalibrationError(
                        f"candidate {anchor}->{consumer} must be an object"
                    )
                _exact_fields(raw_stats, _CANDIDATE_FIELDS, f"candidate {anchor}->{consumer}")
                by_anchor[anchor] = _matrix(
                    raw_stats["dropped_mass"],
                    heads=heads,
                    label=f"candidate {anchor}->{consumer}.dropped_mass",
                )
            candidates[consumer] = dict(sorted(by_anchor.items()))
        expected_consumers = set(layers[1:])
        if set(candidates) != expected_consumers:
            raise MaskReuseCalibrationError(
                "consumer_candidates_by_layer must cover every non-first attention layer"
            )
        for index, consumer in enumerate(layers[1:], start=1):
            first_candidate = max(0, index - max_reuse_span)
            if set(candidates[consumer]) != set(layers[first_candidate:index]):
                raise MaskReuseCalibrationError(
                    f"consumer {consumer} must evaluate every earlier attention layer "
                    "within max_reuse_span"
                )

        return cls(
            model=_text(invocation["model"], "model"),
            checkpoint_manifest_sha256=_sha256(
                invocation["checkpoint_manifest_sha256"], "checkpoint_manifest_sha256"
            ),
            split=split,
            partition=partition,
            inner_fold=inner_fold,
            prompt_id=_text(invocation["prompt_id"], "prompt_id"),
            source=_text(invocation["source"], "source"),
            source_group_sha256=_sha256(invocation["source_group_sha256"], "source_group_sha256"),
            source_capture_sha256=_sha256(
                invocation["source_capture_sha256"], "source_capture_sha256"
            ),
            min_kv_tokens=minimum,
            max_kv_tokens=maximum,
            target_sparsity=target,
            sample_length=sample_length,
            threshold_log2=threshold_log2,
            threshold_lambda=threshold_lambda,
            geometry=geometry,
            global_num_heads=heads,
            eligible_tiles=eligible,
            attention_layers=layers,
            max_reuse_span=max_reuse_span,
            anchor_stats_by_layer=dict(sorted(anchors.items())),
            consumer_candidates_by_layer=dict(sorted(candidates.items())),
        )


@dataclass(frozen=True, slots=True)
class TopologyDiscoveryCaptureSource:
    """Re-iterable strict JSONL source for topology discovery."""

    path: Path

    def __iter__(self) -> Iterator[TopologyDiscoveryCapture]:
        with self.path.open(encoding="utf-8") as handle:
            seen = False
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                seen = True
                try:
                    raw = json.loads(line, object_pairs_hook=_reject_duplicate_json_keys)
                except json.JSONDecodeError as error:
                    raise MaskReuseCalibrationError(
                        f"line {line_number}: invalid JSON: {error.msg}"
                    ) from error
                except MaskReuseCalibrationError as error:
                    raise MaskReuseCalibrationError(f"line {line_number}: {error}") from error
                if not isinstance(raw, dict):
                    raise MaskReuseCalibrationError(
                        f"line {line_number}: topology discovery capture must be an object"
                    )
                try:
                    yield TopologyDiscoveryCapture.from_mapping(raw)
                except MaskReuseCalibrationError as error:
                    raise MaskReuseCalibrationError(f"line {line_number}: {error}") from error
            if not seen:
                raise MaskReuseCalibrationError("topology discovery capture input is empty")

    def sha256(self) -> str:
        """Hash exact file bytes."""
        digest = sha256()
        with self.path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


def load_topology_discovery_captures(
    path: str | Path,
) -> TopologyDiscoveryCaptureSource:
    """Return a lazy topology-discovery capture source."""
    source = TopologyDiscoveryCaptureSource(Path(path))
    if not source.path.is_file():
        raise MaskReuseCalibrationError(
            f"topology discovery capture file does not exist: {source.path}"
        )
    return source


@dataclass(frozen=True, slots=True)
class _Dataset:
    model: str
    checkpoint_manifest_sha256: str
    global_num_heads: int
    attention_layers: tuple[int, ...]
    max_reuse_span: int
    buckets: tuple[Bucket, ...]
    prompts: Mapping[tuple[Bucket, str], tuple[str, ...]]
    menus: Mapping[Bucket, tuple[float, ...]]
    calibration_prompt_ids: tuple[str, ...]
    heldout_prompt_ids: tuple[str, ...]
    prompt_sources: tuple[Mapping[str, object], ...]
    capture_count: int


def _capture_order_key(capture: TopologyDiscoveryCapture) -> tuple[object, ...]:
    return (
        capture.min_kv_tokens,
        math.inf if capture.max_kv_tokens is None else capture.max_kv_tokens,
        _SPLITS.index(capture.split),
        capture.prompt_id,
        capture.target_sparsity,
    )


def _validate_monotonic_pair(
    previous: TopologyDiscoveryCapture, current: TopologyDiscoveryCapture
) -> None:
    identity = (current.bucket, current.split, current.prompt_id)
    if identity != (previous.bucket, previous.split, previous.prompt_id):
        return
    if current.target_sparsity <= previous.target_sparsity:
        raise MaskReuseCalibrationError("target sparsities must be strictly increasing per prompt")
    for layer in current.attention_layers:
        current_stats = current.anchor_stats_by_layer[layer]
        previous_stats = previous.anchor_stats_by_layer[layer]
        if any(
            current_value > previous_value
            for current_value, previous_value in zip(
                current_stats.retained_tiles, previous_stats.retained_tiles, strict=True
            )
        ):
            raise MaskReuseCalibrationError(
                f"anchor candidate {layer} retained counts increase with target sparsity"
            )
        if any(
            current_value + _MONOTONIC_ATOL < previous_value
            for current_value, previous_value in zip(
                current_stats.dropped_mass, previous_stats.dropped_mass, strict=True
            )
        ):
            raise MaskReuseCalibrationError(
                f"anchor candidate {layer} dropped mass decreases with target sparsity"
            )
    for consumer, by_anchor in current.consumer_candidates_by_layer.items():
        for anchor, matrix in by_anchor.items():
            previous_matrix = previous.consumer_candidates_by_layer[consumer][anchor]
            if any(
                current_value + _MONOTONIC_ATOL < previous_value
                for current_row, previous_row in zip(matrix, previous_matrix, strict=True)
                for current_value, previous_value in zip(current_row, previous_row, strict=True)
            ):
                raise MaskReuseCalibrationError(
                    f"candidate {anchor}->{consumer} dropped mass decreases with target sparsity"
                )


def _validate_topology_threshold(
    capture: TopologyDiscoveryCapture, fit: Mapping[str, object]
) -> None:
    params = fit["prefill"]
    assert isinstance(params, Mapping)
    lower = params.get("min_observed_sparsity")
    upper = params.get("max_observed_sparsity")
    if (lower is not None and capture.target_sparsity < float(lower)) or (
        upper is not None and capture.target_sparsity > float(upper)
    ):
        raise MaskReuseCalibrationError(
            f"target_sparsity={capture.target_sparsity} is outside the vanilla fit range"
        )
    expected_log2 = (
        math.log2(float(params["a"]))
        + float(params["b"]) * capture.target_sparsity * math.log2(math.e)
        - math.log2(capture.sample_length)
    )
    expected_lambda = 2.0**expected_log2
    if capture.threshold_log2.hex() != expected_log2.hex():
        raise MaskReuseCalibrationError(
            "topology discovery capture threshold_log2 differs from vanilla fit"
        )
    if capture.threshold_lambda.hex() != expected_lambda.hex():
        raise MaskReuseCalibrationError(
            "topology discovery capture threshold_lambda differs from vanilla fit"
        )


def _validate_dataset(
    source: TopologyDiscoveryCaptureSource,
    *,
    fit: Mapping[str, object],
) -> _Dataset:
    models: set[str] = set()
    checkpoint_identities: set[str] = set()
    head_counts: set[int] = set()
    layer_sets: set[tuple[int, ...]] = set()
    reuse_spans: set[int] = set()
    split_counts: dict[str, int] = defaultdict(int)
    split_buckets: dict[str, set[Bucket]] = defaultdict(set)
    prompts: dict[tuple[Bucket, str], set[str]] = defaultdict(set)
    menus: dict[tuple[Bucket, str, str], set[float]] = defaultdict(set)
    split_prompt_ids: dict[str, set[str]] = defaultdict(set)
    split_fingerprints: dict[str, set[str]] = defaultdict(set)
    group_assignments: dict[str, tuple[str, int | None]] = {}
    fingerprint_groups: dict[str, str] = {}
    prompt_buckets: dict[tuple[str, str], Bucket] = {}
    prompt_metadata: dict[tuple[Bucket, str, str], tuple[object, ...]] = {}
    prompt_source_rows: dict[tuple[Bucket, str, str], Mapping[str, object]] = {}
    seen: set[tuple[Bucket, str, str, float]] = set()
    capture_count = 0
    previous_order: tuple[object, ...] | None = None
    previous_capture: TopologyDiscoveryCapture | None = None
    for capture in source:
        capture_count += 1
        order = _capture_order_key(capture)
        if previous_order is not None and order <= previous_order:
            raise MaskReuseCalibrationError("topology discovery records are not in canonical order")
        if previous_capture is not None:
            _validate_monotonic_pair(previous_capture, capture)
        previous_order = order
        previous_capture = capture
        _validate_topology_threshold(capture, fit)
        models.add(capture.model)
        checkpoint_identities.add(capture.checkpoint_manifest_sha256)
        head_counts.add(capture.global_num_heads)
        layer_sets.add(capture.attention_layers)
        reuse_spans.add(capture.max_reuse_span)
        split_counts[capture.split] += 1
        split_buckets[capture.split].add(capture.bucket)
        prompts[(capture.bucket, capture.split)].add(capture.prompt_id)
        menus[(capture.bucket, capture.split, capture.prompt_id)].add(capture.target_sparsity)
        split_prompt_ids[capture.split].add(capture.prompt_id)
        split_fingerprints[capture.split].add(capture.source_capture_sha256)
        prompt_identity = (capture.split, capture.prompt_id)
        if prompt_buckets.setdefault(prompt_identity, capture.bucket) != capture.bucket:
            raise MaskReuseCalibrationError("a prompt ID is assigned to multiple context buckets")
        assignment = (capture.partition, capture.inner_fold)
        if group_assignments.setdefault(capture.source_group_sha256, assignment) != assignment:
            raise MaskReuseCalibrationError(
                "one source group is assigned to multiple partitions or inner folds"
            )
        if (
            fingerprint_groups.setdefault(
                capture.source_capture_sha256, capture.source_group_sha256
            )
            != capture.source_group_sha256
        ):
            raise MaskReuseCalibrationError(
                "one rendered source capture is assigned to multiple source groups"
            )
        key = (capture.bucket, capture.split, capture.prompt_id, capture.target_sparsity)
        if key in seen:
            raise MaskReuseCalibrationError(
                "topology discovery captures contain a duplicate prompt target"
            )
        seen.add(key)
        prompt_key = (capture.bucket, capture.split, capture.prompt_id)
        metadata = (
            capture.source,
            capture.source_group_sha256,
            capture.partition,
            capture.inner_fold,
            capture.source_capture_sha256,
            capture.sample_length,
            tuple(sorted(capture.geometry.items())),
            capture.eligible_tiles,
        )
        if prompt_metadata.setdefault(prompt_key, metadata) != metadata:
            raise MaskReuseCalibrationError("prompt metadata differs across target sparsities")
        prompt_source_rows[prompt_key] = {
            "min_kv_tokens": capture.min_kv_tokens,
            "max_kv_tokens": capture.max_kv_tokens,
            "split": capture.split,
            "prompt_id": capture.prompt_id,
            "source": capture.source,
            "source_group_sha256": capture.source_group_sha256,
            "partition": capture.partition,
            "inner_fold": capture.inner_fold,
            "source_capture_sha256": capture.source_capture_sha256,
        }

    if capture_count == 0 or any(not split_counts[split] for split in _SPLITS):
        raise MaskReuseCalibrationError(
            "topology discovery captures require calibration and heldout splits"
        )
    if (
        len(models) != 1
        or len(checkpoint_identities) != 1
        or len(head_counts) != 1
        or len(layer_sets) != 1
        or len(reuse_spans) != 1
    ):
        raise MaskReuseCalibrationError(
            "topology discovery captures must use one model, checkpoint, head count, and layer set"
        )
    if split_prompt_ids["calibration"] & split_prompt_ids["heldout"]:
        raise MaskReuseCalibrationError("prompt IDs overlap calibration and heldout splits")
    if split_fingerprints["calibration"] & split_fingerprints["heldout"]:
        raise MaskReuseCalibrationError("source captures overlap calibration and heldout splits")
    if split_buckets["calibration"] != split_buckets["heldout"]:
        raise MaskReuseCalibrationError("heldout context buckets must match calibration buckets")
    buckets = tuple(sorted(split_buckets["calibration"], key=_bucket_key))
    previous_max: int | None = 0
    for minimum, maximum in buckets:
        if previous_max is None or minimum <= previous_max:
            raise MaskReuseCalibrationError("context buckets must be ordered and non-overlapping")
        previous_max = maximum
    canonical_menus: dict[Bucket, tuple[float, ...]] = {}
    for bucket in buckets:
        calibration_prompts = sorted(prompts[(bucket, "calibration")])
        if not calibration_prompts:
            raise MaskReuseCalibrationError(f"bucket {bucket} has no calibration prompts")
        menu = menus[(bucket, "calibration", calibration_prompts[0])]
        for split in _SPLITS:
            for prompt in prompts[(bucket, split)]:
                if menus[(bucket, split, prompt)] != menu:
                    raise MaskReuseCalibrationError("target-sparsity menu differs across prompts")
        canonical_menus[bucket] = tuple(sorted(menu))
    combinations = math.prod(len(canonical_menus[bucket]) for bucket in buckets)
    if combinations > _MAX_TARGET_COMBINATIONS:
        raise MaskReuseCalibrationError(
            f"joint target menu has {combinations} combinations; maximum is "
            f"{_MAX_TARGET_COMBINATIONS}"
        )
    return _Dataset(
        model=next(iter(models)),
        checkpoint_manifest_sha256=next(iter(checkpoint_identities)),
        global_num_heads=next(iter(head_counts)),
        attention_layers=next(iter(layer_sets)),
        max_reuse_span=next(iter(reuse_spans)),
        buckets=buckets,
        prompts={key: tuple(sorted(value)) for key, value in prompts.items()},
        menus=canonical_menus,
        calibration_prompt_ids=tuple(sorted(split_prompt_ids["calibration"])),
        heldout_prompt_ids=tuple(sorted(split_prompt_ids["heldout"])),
        prompt_sources=tuple(
            prompt_source_rows[key] for key in sorted(prompt_source_rows, key=str)
        ),
        capture_count=capture_count,
    )


@dataclass(slots=True)
class _Accumulator:
    eligible_sum: int
    retained: np.ndarray
    anchor_prompt_means: dict[str, np.ndarray]
    prompt_mass: dict[str, Mapping[tuple[int, int], np.ndarray]]


@dataclass(frozen=True, slots=True)
class _Edge:
    anchor_tile_cost: int


@dataclass(frozen=True, slots=True)
class _DonorOption:
    choice: _Choice
    bmm1_skipped_tiles: int
    risk_by_prompt: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class _FrozenSelection:
    targets: Mapping[Bucket, float] | None
    anchors: tuple[int, ...]
    nearest: Mapping[int, int]
    choices: Mapping[tuple[Bucket, int, int], _Choice]
    combined_tile_cost: int
    fallback_count: int
    bmm1_skipped_tiles: int
    bmm1_eligible_tiles: int
    target_bmm1_skip_ratio_met: bool
    worst_prompt_reuse_dropped_mass: float
    mean_prompt_reuse_dropped_mass: float
    worst_individual_reuse_dropped_mass: float
    exact_reason: str | None = None

    @property
    def bmm1_skip_ratio(self) -> float:
        return (
            self.bmm1_skipped_tiles / self.bmm1_eligible_tiles if self.bmm1_eligible_tiles else 0.0
        )


@dataclass(slots=True)
class _Bmm1Evaluation:
    eligible_tiles: int = 0
    skipped_tiles: int = 0

    def add(self, other: _Bmm1Evaluation) -> None:
        self.eligible_tiles += other.eligible_tiles
        self.skipped_tiles += other.skipped_tiles

    def to_mapping(self) -> dict[str, object]:
        return {
            "eligible_tiles": self.eligible_tiles,
            "skipped_tiles": self.skipped_tiles,
            "model_wide_bmm1_tile_skip_ratio": (
                self.skipped_tiles / self.eligible_tiles if self.eligible_tiles else 0.0
            ),
        }


def _selection_pass(
    source: TopologyDiscoveryCaptureSource,
    dataset: _Dataset,
    *,
    max_anchor_dropped_mass: float,
    target_bmm1_skip_ratio: float,
) -> tuple[_FrozenSelection, list[Mapping[str, object]]]:
    layers = dataset.attention_layers
    layer_index = {layer: index for index, layer in enumerate(layers)}
    accumulators: dict[tuple[Bucket, float], _Accumulator] = {}
    for bucket in dataset.buckets:
        for target in dataset.menus[bucket]:
            accumulators[(bucket, target)] = _Accumulator(
                eligible_sum=0,
                retained=np.zeros((len(layers), dataset.global_num_heads), dtype=np.int64),
                anchor_prompt_means={},
                prompt_mass={},
            )
    for capture in source:
        if capture.split != "calibration":
            continue
        accumulator = accumulators[(capture.bucket, capture.target_sparsity)]
        accumulator.eligible_sum += capture.eligible_tiles
        prompt_means = np.zeros(len(layers), dtype=np.float64)
        for layer, stats in capture.anchor_stats_by_layer.items():
            index = layer_index[layer]
            accumulator.retained[index] += np.asarray(stats.retained_tiles, dtype=np.int64)
            prompt_means[index] = sum(stats.dropped_mass) / dataset.global_num_heads
        if capture.prompt_id in accumulator.anchor_prompt_means:
            raise MaskReuseCalibrationError(
                "topology discovery captures contain a duplicate calibration prompt target"
            )
        accumulator.anchor_prompt_means[capture.prompt_id] = prompt_means
        prompt_mass: dict[tuple[int, int], np.ndarray] = {}
        for consumer, by_anchor in capture.consumer_candidates_by_layer.items():
            for anchor, matrix in by_anchor.items():
                prompt_mass[(anchor, consumer)] = np.asarray(matrix, dtype=np.float64)
        accumulator.prompt_mass[capture.prompt_id] = prompt_mass

    def edge(
        start: int,
        stop: int,
        targets: Mapping[Bucket, float],
    ) -> _Edge:
        cost = 0
        for bucket in dataset.buckets:
            accumulator = accumulators[(bucket, targets[bucket])]
            retained = accumulator.retained[start]
            cost += accumulator.eligible_sum * dataset.global_num_heads
            cost += int(retained.sum())
        return _Edge(cost)

    def donor_options(
        accumulator: _Accumulator,
        *,
        anchor: int,
        consumer: int,
        head: int,
        retained: np.ndarray,
        prompts: tuple[str, ...],
    ) -> tuple[_DonorOption, ...]:
        options = [
            _DonorOption(
                _Choice(0, True, accumulator.eligible_sum),
                0,
                tuple(0.0 for _ in prompts),
            )
        ]
        for donor in range(dataset.global_num_heads):
            retained_tiles = int(retained[donor])
            options.append(
                _DonorOption(
                    _Choice(donor, False, retained_tiles),
                    accumulator.eligible_sum - retained_tiles,
                    tuple(
                        float(accumulator.prompt_mass[prompt][(anchor, consumer)][head, donor])
                        for prompt in prompts
                    ),
                )
            )

        # Keep only risk/skip Pareto choices.  Exact fallback is the canonical
        # zero-risk, zero-skip option; equivalent donors cannot improve either
        # objective and only enlarge the MILP.
        pareto = []
        for candidate_index, candidate in enumerate(options):
            dominated = False
            for other_index, other in enumerate(options):
                if candidate_index == other_index:
                    continue
                no_worse = other.bmm1_skipped_tiles >= candidate.bmm1_skipped_tiles and all(
                    other_risk <= candidate_risk
                    for other_risk, candidate_risk in zip(
                        other.risk_by_prompt,
                        candidate.risk_by_prompt,
                        strict=True,
                    )
                )
                strictly_better = other.bmm1_skipped_tiles > candidate.bmm1_skipped_tiles or any(
                    other_risk < candidate_risk
                    for other_risk, candidate_risk in zip(
                        other.risk_by_prompt,
                        candidate.risk_by_prompt,
                        strict=True,
                    )
                )
                canonical_tie = not strictly_better and (
                    (other.choice.fallback and not candidate.choice.fallback)
                    or (
                        other.choice.fallback == candidate.choice.fallback
                        and other.choice.donor_head < candidate.choice.donor_head
                    )
                )
                if no_worse and (strictly_better or canonical_tie):
                    dominated = True
                    break
            if not dominated:
                pareto.append(candidate)
        return tuple(pareto)

    def eligible_bmm1_tiles(targets: Mapping[Bucket, float]) -> int:
        return sum(
            accumulators[(bucket, targets[bucket])].eligible_sum
            * dataset.global_num_heads
            * len(layers)
            for bucket in dataset.buckets
        )

    def solve_path(
        targets: Mapping[Bucket, float],
        *,
        minimum_bmm1_skipped_tiles: int | None,
        maximize_bmm1_skipped_tiles: bool = False,
        target_met: bool,
    ) -> _FrozenSelection | None:
        edge_rows: dict[tuple[int, int], _Edge] = {}
        for start in range(len(layers)):
            maximum_stop = min(len(layers), start + dataset.max_reuse_span + 1)
            for stop in range(start + 1, maximum_stop + 1):
                row = edge(start, stop, targets)
                if row is not None:
                    edge_rows[(start, stop)] = row

        problem = pulp.LpProblem("mask_reuse_topology", pulp.LpMinimize)
        edge_variables = {
            edge_key: pulp.LpVariable(
                f"edge_{edge_key[0]}_{edge_key[1]}", lowBound=0, upBound=1, cat="Binary"
            )
            for edge_key in edge_rows
        }
        outgoing = {
            node: pulp.lpSum(
                variable for (start, _), variable in edge_variables.items() if start == node
            )
            for node in range(len(layers))
        }
        incoming = {
            node: pulp.lpSum(
                variable for (_, stop), variable in edge_variables.items() if stop == node
            )
            for node in range(1, len(layers) + 1)
        }
        problem += outgoing[0] == 1, "path_start"
        problem += incoming[len(layers)] == 1, "path_end"
        for node in range(1, len(layers)):
            problem += incoming[node] == outgoing[node], f"path_flow_{node}"

        # Match the ordinary fixed-topology calibrator: for each development
        # prompt, constrain the mean dropped mass across all selected anchor
        # layers and heads.  A per-candidate-anchor constraint is stricter and
        # can incorrectly reject a safe aggregate topology when a short run of
        # layers is individually above the bound.
        for bucket_index, bucket in enumerate(dataset.buckets):
            accumulator = accumulators[(bucket, targets[bucket])]
            for prompt_index, prompt in enumerate(dataset.prompts[(bucket, "calibration")]):
                prompt_means = accumulator.anchor_prompt_means[prompt]
                problem += (
                    (
                        pulp.lpSum(
                            (float(prompt_means[start]) - max_anchor_dropped_mass) * variable
                            for (start, _), variable in edge_variables.items()
                        )
                        <= 0.0
                    ),
                    f"anchor_mass_{bucket_index}_{prompt_index}",
                )

        choice_variables: dict[tuple[tuple[int, int], Bucket, int, int, int], pulp.LpVariable] = {}
        choice_options: dict[tuple[tuple[int, int], Bucket, int, int, int], _DonorOption] = {}
        prompt_risk_terms: dict[tuple[Bucket, str], list[object]] = defaultdict(list)
        for edge_key, edge_variable in edge_variables.items():
            start, stop = edge_key
            anchor = layers[start]
            for bucket_index, bucket in enumerate(dataset.buckets):
                accumulator = accumulators[(bucket, targets[bucket])]
                prompts = dataset.prompts[(bucket, "calibration")]
                retained = accumulator.retained[start]
                for consumer in layers[start + 1 : stop]:
                    for head in range(dataset.global_num_heads):
                        options = donor_options(
                            accumulator,
                            anchor=anchor,
                            consumer=consumer,
                            head=head,
                            retained=retained,
                            prompts=prompts,
                        )
                        variables_for_choice = []
                        for option_index, option in enumerate(options):
                            key = (edge_key, bucket, consumer, head, option_index)
                            variable = pulp.LpVariable(
                                (
                                    f"choice_{start}_{stop}_{bucket_index}_{consumer}_"
                                    f"{head}_{option_index}"
                                ),
                                lowBound=0,
                                upBound=1,
                                cat="Binary",
                            )
                            choice_variables[key] = variable
                            choice_options[key] = option
                            variables_for_choice.append(variable)
                            for prompt, risk in zip(prompts, option.risk_by_prompt, strict=True):
                                prompt_risk_terms[(bucket, prompt)].append(risk * variable)
                        problem += (
                            pulp.lpSum(variables_for_choice) == edge_variable,
                            f"choose_{start}_{stop}_{bucket_index}_{consumer}_{head}",
                        )

        cost_expression = pulp.lpSum(
            edge_rows[edge_key].anchor_tile_cost * variable
            for edge_key, variable in edge_variables.items()
        ) + pulp.lpSum(
            2 * choice_options[key].choice.retained_tiles * variable
            for key, variable in choice_variables.items()
        )
        reuse_count_expression = pulp.lpSum(
            int(not choice_options[key].choice.fallback) * variable
            for key, variable in choice_variables.items()
        )
        bmm1_skipped_expression = pulp.lpSum(
            choice_options[key].bmm1_skipped_tiles * variable
            for key, variable in choice_variables.items()
        )
        anchor_count_expression = pulp.lpSum(edge_variables.values())
        risk_normalizer = len(layers) * dataset.global_num_heads
        worst_prompt_risk = pulp.LpVariable("worst_prompt_reuse_dropped_mass", lowBound=0.0)
        for prompt_index, prompt_key in enumerate(sorted(prompt_risk_terms, key=str)):
            problem += (
                pulp.lpSum(prompt_risk_terms[prompt_key]) <= risk_normalizer * worst_prompt_risk,
                f"reuse_risk_{prompt_index}",
            )
        if minimum_bmm1_skipped_tiles is not None:
            problem += (
                (bmm1_skipped_expression >= minimum_bmm1_skipped_tiles),
                "minimum_bmm1_skipped_tiles",
            )
        solver = pulp.PULP_CBC_CMD(msg=False, threads=1, options=["randomSeed 0"])
        warm_solver = pulp.PULP_CBC_CMD(
            msg=False,
            threads=1,
            options=["randomSeed 0"],
            warmStart=True,
        )
        has_incumbent = False

        def minimize(expression: object) -> bool:
            nonlocal has_incumbent
            problem.setObjective(expression)
            status = problem.solve(warm_solver if has_incumbent else solver)
            has_incumbent = status == pulp.LpStatusOptimal
            return has_incumbent

        def minimize_and_fix(
            expression: object,
            name: str,
            *,
            integral: bool,
        ) -> float | None:
            nonlocal problem
            if not minimize(expression):
                return None
            raw_value = pulp.value(expression)
            value = 0.0 if raw_value is None else float(raw_value)
            if integral:
                rounded = round(value)
                problem += expression == rounded, name
                return float(rounded)
            problem += expression <= value + _SOLVER_LEXICOGRAPHIC_ATOL, name
            return value

        if maximize_bmm1_skipped_tiles:
            if (
                minimize_and_fix(
                    -bmm1_skipped_expression,
                    "fix_maximum_bmm1_skipped_tiles",
                    integral=True,
                )
                is None
            ):
                return None
        if (
            minimize_and_fix(
                worst_prompt_risk,
                "fix_worst_prompt_reuse_dropped_mass",
                integral=False,
            )
            is None
        ):
            return None
        if minimize_and_fix(cost_expression, "fix_cost", integral=True) is None:
            raise MaskReuseCalibrationError(
                "topology solver lost feasibility after fixing worst-prompt reuse risk"
            )
        donor_signature = pulp.lpSum(
            (choice_options[key].choice.donor_head + 1) * variable
            for key, variable in choice_variables.items()
            if not choice_options[key].choice.fallback
        )
        maximum_reuse_count = len(dataset.buckets) * dataset.global_num_heads * len(layers)
        donor_base = dataset.global_num_heads * maximum_reuse_count + 1
        anchor_lex_scale = 1 << len(layers)
        anchor_lex_signature = pulp.lpSum(
            (1 << (len(layers) - start - 1)) * outgoing[start] for start in range(1, len(layers))
        )
        deterministic_signature = (
            (reuse_count_expression * (len(layers) + 1) + anchor_count_expression)
            * anchor_lex_scale
            - anchor_lex_signature
        ) * donor_base + donor_signature
        if not minimize(deterministic_signature):
            raise MaskReuseCalibrationError(
                "topology solver lost deterministic tie-break feasibility"
            )

        selected = sorted(
            edge_key for edge_key, variable in edge_variables.items() if variable.value() > 0.5
        )
        if not selected or selected[0][0] != 0 or selected[-1][1] != len(layers):
            raise MaskReuseCalibrationError("topology solver returned a non-covering path")
        anchors = tuple(layers[start] for start, _ in selected)
        nearest: dict[int, int] = {}
        choices: dict[tuple[Bucket, int, int], _Choice] = {}
        combined_tile_cost = 0
        for edge_key in selected:
            start, stop = edge_key
            anchor = layers[start]
            combined_tile_cost += edge_rows[edge_key].anchor_tile_cost
            for layer in layers[start:stop]:
                nearest[layer] = anchor
            for key, variable in choice_variables.items():
                candidate_edge, bucket, consumer, head, _ = key
                if candidate_edge != edge_key or variable.value() <= 0.5:
                    continue
                option = choice_options[key]
                choices[(bucket, consumer, head)] = option.choice
                combined_tile_cost += 2 * option.choice.retained_tiles

        prompt_risks: list[float] = []
        worst_individual_risk = 0.0
        for bucket in dataset.buckets:
            accumulator = accumulators[(bucket, targets[bucket])]
            for prompt in dataset.prompts[(bucket, "calibration")]:
                total = 0.0
                for consumer, anchor in nearest.items():
                    if consumer == anchor:
                        continue
                    matrix = accumulator.prompt_mass[prompt][(anchor, consumer)]
                    for head in range(dataset.global_num_heads):
                        choice = choices[(bucket, consumer, head)]
                        if choice.fallback:
                            continue
                        risk = float(matrix[head, choice.donor_head])
                        total += risk
                        worst_individual_risk = max(worst_individual_risk, risk)
                prompt_risks.append(total / risk_normalizer)
        bmm1_skipped_tiles = sum(
            accumulators[(bucket, targets[bucket])].eligible_sum - choice.retained_tiles
            for (bucket, _, _), choice in choices.items()
        )
        fallback_count = sum(choice.fallback for choice in choices.values())
        return _FrozenSelection(
            targets=targets,
            anchors=anchors,
            nearest=nearest,
            choices=choices,
            combined_tile_cost=combined_tile_cost,
            fallback_count=fallback_count,
            bmm1_skipped_tiles=bmm1_skipped_tiles,
            bmm1_eligible_tiles=eligible_bmm1_tiles(targets),
            target_bmm1_skip_ratio_met=target_met,
            worst_prompt_reuse_dropped_mass=max(prompt_risks, default=0.0),
            mean_prompt_reuse_dropped_mass=(
                sum(prompt_risks) / len(prompt_risks) if prompt_risks else 0.0
            ),
            worst_individual_reuse_dropped_mass=worst_individual_risk,
        )

    frontier: list[Mapping[str, object]] = []
    candidates: list[tuple[tuple[object, ...], _FrozenSelection]] = []
    maximum_candidates: list[tuple[tuple[object, ...], _FrozenSelection]] = []
    infeasible_targets: list[
        tuple[Mapping[Bucket, float], tuple[float, ...], dict[str, object]]
    ] = []
    menus = [dataset.menus[bucket] for bucket in dataset.buckets]
    for target_values in itertools.product(*menus):
        targets = dict(zip(dataset.buckets, target_values, strict=True))
        eligible_tiles = eligible_bmm1_tiles(targets)
        required_tiles = math.ceil(target_bmm1_skip_ratio * eligible_tiles)
        solved = solve_path(
            targets,
            minimum_bmm1_skipped_tiles=required_tiles,
            target_met=True,
        )
        row: dict[str, object] = {
            "target_sparsity_by_bucket": [
                {
                    "min_kv_tokens": bucket[0],
                    "max_kv_tokens": bucket[1],
                    "target_sparsity": targets[bucket],
                }
                for bucket in dataset.buckets
            ],
            "feasible": solved is not None,
            "anchor_constraint_and_bmm1_target_feasible": solved is not None,
            "target_bmm1_skip_ratio": target_bmm1_skip_ratio,
            "required_bmm1_skipped_tiles": required_tiles,
            "bmm1_eligible_tiles": eligible_tiles,
            "target_bmm1_skip_ratio_feasible": solved is not None,
        }
        if solved is not None:
            rank = (
                solved.worst_prompt_reuse_dropped_mass,
                solved.combined_tile_cost,
                len(solved.anchors),
                target_values,
            )
            candidates.append((rank, solved))
            row.update(
                {
                    "combined_tile_cost": solved.combined_tile_cost,
                    "fallback_head_bucket_count": solved.fallback_count,
                    "anchors": list(solved.anchors),
                    "bmm1_skipped_tiles": solved.bmm1_skipped_tiles,
                    "achieved_bmm1_skip_ratio": solved.bmm1_skip_ratio,
                    "worst_prompt_model_wide_reuse_dropped_mass": (
                        solved.worst_prompt_reuse_dropped_mass
                    ),
                    "mean_prompt_model_wide_reuse_dropped_mass": (
                        solved.mean_prompt_reuse_dropped_mass
                    ),
                }
            )
        else:
            infeasible_targets.append((targets, target_values, row))
        frontier.append(row)
    if candidates:
        selection = min(candidates, key=lambda item: item[0])[1]
        return selection, frontier
    for targets, target_values, row in infeasible_targets:
        maximum = solve_path(
            targets,
            minimum_bmm1_skipped_tiles=None,
            maximize_bmm1_skipped_tiles=True,
            target_met=False,
        )
        row["anchor_constraint_feasible"] = maximum is not None
        row["feasible"] = maximum is not None
        if maximum is None:
            continue
        maximum_rank = (
            -maximum.bmm1_skipped_tiles,
            maximum.worst_prompt_reuse_dropped_mass,
            maximum.combined_tile_cost,
            len(maximum.anchors),
            target_values,
        )
        maximum_candidates.append((maximum_rank, maximum))
        row.update(
            {
                "maximum_feasible_bmm1_skipped_tiles": maximum.bmm1_skipped_tiles,
                "maximum_feasible_bmm1_skip_ratio": maximum.bmm1_skip_ratio,
                "maximum_policy_worst_prompt_model_wide_reuse_dropped_mass": (
                    maximum.worst_prompt_reuse_dropped_mass
                ),
            }
        )
    if maximum_candidates:
        selection = min(maximum_candidates, key=lambda item: item[0])[1]
        return selection, frontier
    anchors = layers
    exact_targets = {bucket: dataset.menus[bucket][0] for bucket in dataset.buckets}
    return (
        _FrozenSelection(
            targets=None,
            anchors=anchors,
            nearest={layer: layer for layer in layers},
            choices={},
            combined_tile_cost=0,
            fallback_count=0,
            bmm1_skipped_tiles=0,
            bmm1_eligible_tiles=eligible_bmm1_tiles(exact_targets),
            target_bmm1_skip_ratio_met=False,
            worst_prompt_reuse_dropped_mass=0.0,
            mean_prompt_reuse_dropped_mass=0.0,
            worst_individual_reuse_dropped_mass=0.0,
            exact_reason="no_joint_target_and_topology_satisfied_anchor_constraints",
        ),
        frontier,
    )


def _evaluation_pass(
    source: TopologyDiscoveryCaptureSource,
    dataset: _Dataset,
    selection: _FrozenSelection,
    *,
    max_anchor_dropped_mass: float,
    reuse_dropped_mass_report_threshold: float,
) -> tuple[
    Mapping[tuple[Bucket, str], _ReuseEvaluation],
    Mapping[tuple[Bucket, str], _AnchorEvaluation],
    Mapping[tuple[Bucket, str], _Bmm1Evaluation],
]:
    reuse = {(bucket, split): _ReuseEvaluation() for bucket in dataset.buckets for split in _SPLITS}
    anchor = {
        (bucket, split): _AnchorEvaluation() for bucket in dataset.buckets for split in _SPLITS
    }
    bmm1 = {(bucket, split): _Bmm1Evaluation() for bucket in dataset.buckets for split in _SPLITS}
    for capture in source:
        target = (
            dataset.menus[capture.bucket][0]
            if selection.targets is None
            else selection.targets[capture.bucket]
        )
        if capture.target_sparsity != target:
            continue
        reuse_result = reuse[(capture.bucket, capture.split)]
        bmm1_result = bmm1[(capture.bucket, capture.split)]
        bmm1_result.eligible_tiles += (
            capture.eligible_tiles * dataset.global_num_heads * len(dataset.attention_layers)
        )
        for consumer, selected_anchor in selection.nearest.items():
            if consumer == selected_anchor:
                continue
            anchor_stats = capture.anchor_stats_by_layer[selected_anchor]
            matrix = capture.consumer_candidates_by_layer[consumer][selected_anchor]
            for head in range(dataset.global_num_heads):
                reuse_result.eligible_tiles += capture.eligible_tiles
                if selection.targets is None:
                    reuse_result.retained_tiles += capture.eligible_tiles
                    continue
                choice = selection.choices[(capture.bucket, consumer, head)]
                if choice.fallback:
                    reuse_result.retained_tiles += capture.eligible_tiles
                    continue
                bmm1_result.skipped_tiles += (
                    capture.eligible_tiles - anchor_stats.retained_tiles[choice.donor_head]
                )
                dropped = matrix[head][choice.donor_head]
                reuse_result.retained_tiles += anchor_stats.retained_tiles[choice.donor_head]
                reuse_result.sparse_observations += 1
                reuse_result.dropped_mass_sum += dropped
                reuse_result.worst_dropped_mass = max(reuse_result.worst_dropped_mass, dropped)
                reuse_result.violations += int(dropped > reuse_dropped_mass_report_threshold)

        anchor_result = anchor[(capture.bucket, capture.split)]
        dropped_values: list[float] = []
        for selected_anchor in selection.anchors:
            stats = capture.anchor_stats_by_layer[selected_anchor]
            anchor_result.eligible_tiles += capture.eligible_tiles * dataset.global_num_heads
            if selection.targets is None:
                anchor_result.retained_tiles += capture.eligible_tiles * dataset.global_num_heads
                dropped_values.extend([0.0] * dataset.global_num_heads)
            else:
                anchor_result.retained_tiles += sum(stats.retained_tiles)
                dropped_values.extend(stats.dropped_mass)
        prompt_mean = sum(dropped_values) / len(dropped_values)
        anchor_result.prompt_count += 1
        anchor_result.prompt_mean_sum += prompt_mean
        anchor_result.worst_prompt_mean = max(anchor_result.worst_prompt_mean, prompt_mean)
        anchor_result.violations += int(prompt_mean > max_anchor_dropped_mass)
    for bucket in dataset.buckets:
        for split in _SPLITS:
            expected = len(dataset.prompts[(bucket, split)])
            if anchor[(bucket, split)].prompt_count != expected:
                raise MaskReuseCalibrationError(
                    f"evaluation pass did not cover every {split} prompt in bucket {bucket}"
                )
    return reuse, anchor, bmm1


def _canonical_evidence(evidence: Mapping[str, object]) -> dict[str, str]:
    expected = {
        "topology_discovery_capture_sha256",
        "vanilla_fit_sha256",
        "prompt_plan_sha256",
    }
    if set(evidence) != expected:
        raise MaskReuseCalibrationError(
            "topology evidence fields differ; "
            f"missing={sorted(expected - evidence.keys())}, "
            f"extra={sorted(evidence.keys() - expected)}"
        )
    return {key: _sha256(evidence[key], f"evidence.{key}") for key in sorted(expected)}


def calibrate_mask_reuse_topology(
    captures: TopologyDiscoveryCaptureSource | str | Path,
    *,
    vanilla_calibration: Mapping[str, object],
    checkpoint_manifest: VerifiedCheckpointManifest,
    evidence: Mapping[str, object],
    max_anchor_dropped_mass: float,
    reuse_dropped_mass_report_threshold: float,
    target_bmm1_skip_ratio: float,
) -> dict[str, object]:
    """Select a global candidate topology and evaluate it without held-out tuning.

    ``target_bmm1_skip_ratio`` is the minimum fraction of eligible QK tiles
    skipped by non-fallback reuse consumers across all attention layers in the
    calibration split. Among policies that meet the target, selection minimizes
    development reuse dropped mass. If the target is structurally infeasible,
    the returned candidate maximizes BMM1 skips and explicitly reports the gap.

    ``reuse_dropped_mass_report_threshold`` only counts diagnostic violations
    after selection. It never accepts, rejects, or retunes a policy.
    """
    if not isinstance(checkpoint_manifest, VerifiedCheckpointManifest):
        raise MaskReuseCalibrationError(
            "checkpoint_manifest must be returned by verify_checkpoint_manifest"
        )
    source = (
        captures
        if isinstance(captures, TopologyDiscoveryCaptureSource)
        else load_topology_discovery_captures(captures)
    )
    fit = canonical_prefill_threshold_scale_factor(vanilla_calibration)
    params = fit["prefill"]
    assert isinstance(params, Mapping)
    fit_bounds_available = {
        "min_observed_sparsity",
        "max_observed_sparsity",
    } <= params.keys()
    dataset = _validate_dataset(source, fit=fit)
    if dataset.checkpoint_manifest_sha256 != checkpoint_manifest.sha256:
        raise MaskReuseCalibrationError(
            "topology discovery captures do not match the verified checkpoint manifest"
        )
    if dataset.model != checkpoint_manifest.model:
        raise MaskReuseCalibrationError(
            "topology discovery model does not match the verified checkpoint manifest"
        )
    canonical_evidence = _canonical_evidence(evidence)
    input_sha256 = source.sha256()
    if canonical_evidence["topology_discovery_capture_sha256"] != input_sha256:
        raise MaskReuseCalibrationError(
            "evidence.topology_discovery_capture_sha256 does not match the capture file"
        )
    anchor_bound = _number(
        max_anchor_dropped_mass,
        "max_anchor_dropped_mass",
        minimum=0.0,
        maximum=1.0,
    )
    reuse_report_threshold = _number(
        reuse_dropped_mass_report_threshold,
        "reuse_dropped_mass_report_threshold",
        minimum=0.0,
        maximum=1.0,
    )
    bmm1_target = _number(
        target_bmm1_skip_ratio,
        "target_bmm1_skip_ratio",
        minimum=0.0,
        maximum=1.0,
    )
    selection, frontier = _selection_pass(
        source,
        dataset,
        max_anchor_dropped_mass=anchor_bound,
        target_bmm1_skip_ratio=bmm1_target,
    )
    reuse_evaluations, anchor_evaluations, bmm1_evaluations = _evaluation_pass(
        source,
        dataset,
        selection,
        max_anchor_dropped_mass=anchor_bound,
        reuse_dropped_mass_report_threshold=reuse_report_threshold,
    )
    if source.sha256() != input_sha256:
        raise MaskReuseCalibrationError(
            "topology discovery capture file changed during calibration; discard this result"
        )

    policies = []
    bucket_reports = []
    overall_reuse_calibration = _ReuseEvaluation()
    overall_reuse_heldout = _ReuseEvaluation()
    overall_anchor_calibration = _AnchorEvaluation()
    overall_anchor_heldout = _AnchorEvaluation()
    overall_bmm1_calibration = _Bmm1Evaluation()
    overall_bmm1_heldout = _Bmm1Evaluation()
    consumer_layers = tuple(layer for layer, anchor in selection.nearest.items() if layer != anchor)
    for bucket in dataset.buckets:
        if selection.targets is None:
            policy: dict[str, object] = {
                "min_kv_tokens": bucket[0],
                "max_kv_tokens": bucket[1],
                "exact": True,
            }
            fallback_heads: dict[str, list[int]] = {}
        else:
            headmaps = {
                str(layer): [
                    selection.choices[(bucket, layer, head)].donor_head
                    for head in range(dataset.global_num_heads)
                ]
                for layer in consumer_layers
            }
            fallback_heads = {
                str(layer): [
                    head
                    for head in range(dataset.global_num_heads)
                    if selection.choices[(bucket, layer, head)].fallback
                ]
                for layer in consumer_layers
            }
            policy = {
                "min_kv_tokens": bucket[0],
                "max_kv_tokens": bucket[1],
                "target_sparsity": selection.targets[bucket],
                "headmaps": headmaps,
                "fallback_heads": fallback_heads,
            }
        policies.append(policy)
        reuse_calibration = reuse_evaluations[(bucket, "calibration")]
        reuse_heldout = reuse_evaluations[(bucket, "heldout")]
        anchor_calibration = anchor_evaluations[(bucket, "calibration")]
        anchor_heldout = anchor_evaluations[(bucket, "heldout")]
        bmm1_calibration = bmm1_evaluations[(bucket, "calibration")]
        bmm1_heldout = bmm1_evaluations[(bucket, "heldout")]
        overall_reuse_calibration.add(reuse_calibration)
        overall_reuse_heldout.add(reuse_heldout)
        overall_anchor_calibration.add(anchor_calibration)
        overall_anchor_heldout.add(anchor_heldout)
        overall_bmm1_calibration.add(bmm1_calibration)
        overall_bmm1_heldout.add(bmm1_heldout)
        bucket_reports.append(
            {
                "min_kv_tokens": bucket[0],
                "max_kv_tokens": bucket[1],
                "selected_target_sparsity": (
                    None if selection.targets is None else selection.targets[bucket]
                ),
                "fallback_head_count": sum(len(heads) for heads in fallback_heads.values()),
                "reuse_calibration": reuse_calibration.to_mapping(),
                "reuse_heldout": reuse_heldout.to_mapping(),
                "anchor_calibration": anchor_calibration.to_mapping(
                    exact=selection.targets is None
                ),
                "anchor_heldout": anchor_heldout.to_mapping(exact=selection.targets is None),
                "bmm1_calibration": bmm1_calibration.to_mapping(),
                "bmm1_heldout": bmm1_heldout.to_mapping(),
            }
        )

    if selection.targets is not None and (
        overall_bmm1_calibration.eligible_tiles != selection.bmm1_eligible_tiles
        or overall_bmm1_calibration.skipped_tiles != selection.bmm1_skipped_tiles
    ):
        raise MaskReuseCalibrationError(
            "BMM1 objective accounting differs between selection and calibration evaluation"
        )

    bmm1_objective = {
        "enabled": True,
        "metric": "model_wide_eligible_bmm1_tile_skip_ratio",
        "selection_split": "calibration",
        "comparison": ">=",
        "target": bmm1_target,
        "target_met": selection.target_bmm1_skip_ratio_met,
        "calibration_eligible_tiles": selection.bmm1_eligible_tiles,
        "minimum_required_skipped_tiles": math.ceil(bmm1_target * selection.bmm1_eligible_tiles),
        "calibration_skipped_tiles": selection.bmm1_skipped_tiles,
        "achieved": selection.bmm1_skip_ratio,
        "maximum_feasible": (
            selection.bmm1_skip_ratio if selection.target_bmm1_skip_ratio_met is False else None
        ),
    }
    reuse_selection_objective = {
        "metric": ("per_prompt_mean_across_all_attention_layers_and_heads_reuse_dropped_mass"),
        "selection_split": "calibration",
        "optimization": ["minimum_worst_prompt"],
        "hard_maximum": None,
        "exact_fallback_risk": 0.0,
        "worst_development_prompt_model_wide_dropped_mass": (
            selection.worst_prompt_reuse_dropped_mass
        ),
        "mean_development_prompt_model_wide_dropped_mass": (
            selection.mean_prompt_reuse_dropped_mass
        ),
        "worst_individual_dropped_mass": selection.worst_individual_reuse_dropped_mass,
    }

    constraints = {
        "anchor": {
            "metric": "per_prompt_mean_across_selected_anchor_layers_and_heads_dropped_mass",
            "comparison": "<=",
            "maximum": anchor_bound,
        },
        "reuse": {
            "selection_metric": reuse_selection_objective["metric"],
            "selection_hard_maximum": None,
            "report_metric": "per_prompt_candidate_reuse_dropped_mass",
            "report_threshold": reuse_report_threshold,
            "report_threshold_affects_selection": False,
        },
        "bmm1_skip_ratio": bmm1_objective,
    }
    if selection.targets is None:
        selection_status = selection.exact_reason
    elif selection.target_bmm1_skip_ratio_met:
        selection_status = "target_bmm1_skip_ratio_met"
    else:
        selection_status = "target_bmm1_skip_ratio_unmet_maximum_feasible"
    report = {
        "model": dataset.model,
        "checkpoint_manifest_sha256": checkpoint_manifest.sha256,
        "selection_status": selection_status,
        "selected_anchors": list(selection.anchors),
        "combined_tile_cost": selection.combined_tile_cost,
        "fallback_head_bucket_count": selection.fallback_count,
        "reuse_selection_objective": reuse_selection_objective,
        "constraints": constraints,
        "joint_frontier": frontier,
        "by_bucket": bucket_reports,
        "overall": {
            "reuse_calibration": overall_reuse_calibration.to_mapping(),
            "reuse_heldout": overall_reuse_heldout.to_mapping(),
            "anchor_calibration": overall_anchor_calibration.to_mapping(
                exact=selection.targets is None
            ),
            "anchor_heldout": overall_anchor_heldout.to_mapping(exact=selection.targets is None),
            "bmm1_calibration": overall_bmm1_calibration.to_mapping(),
            "bmm1_heldout": overall_bmm1_heldout.to_mapping(),
        },
    }
    provenance = {
        "calibrator": "modelopt.mask_reuse.topology_discovery",
        "topology_discovery_capture_schema_version": 1,
        "input_capture_count": dataset.capture_count,
        "canonical_input_sha256": input_sha256,
        "calibration_prompt_ids": list(dataset.calibration_prompt_ids),
        "heldout_prompt_ids": list(dataset.heldout_prompt_ids),
        "prompt_sources": list(dataset.prompt_sources),
        "selection_split": "calibration",
        "evaluation_split": "heldout",
        "selection_scope": "one_global_topology_across_context_buckets",
        "max_reuse_span": dataset.max_reuse_span,
        "vanilla_fit_bounds_available": fit_bounds_available,
        "selection_cost": {
            "formula": (
                "anchor: eligible_bmm1 + retained_bmm2; "
                "consumer: 2 * retained_anchor; fallback: 2 * eligible"
            ),
            "bmm1_weight": 1.0,
            "bmm2_weight": 1.0,
        },
        "bmm1_skip_objective": {
            "formula": (
                "BMM1 tiles skipped by non-fallback reuse consumers / "
                "eligible BMM1 tiles across all attention layers"
            ),
            "selection_split": "calibration",
            "requested_target": bmm1_target,
        },
        "tie_breaks": [
            *(
                ["maximum feasible model-wide BMM1 tile skip ratio"]
                if selection.target_bmm1_skip_ratio_met is False
                else []
            ),
            "minimum worst-prompt model-wide reuse dropped mass",
            "minimum equal-BMM total tile cost",
            "fewest reused consumer heads",
            "fewest anchors",
            "smallest target-sparsity tuple",
            "lexicographically smallest anchor tuple",
            "lexicographically smallest donor-head map",
        ],
        "downstream_contract": (
            "freeze anchors and nearest, then recollect with the ordinary "
            "fixed-topology compact calibrator"
        ),
    }
    return {
        "topology_selection_schema_version": 1,
        "promotion_status": "topology_candidate_only",
        "phase": "prefill",
        "producer": {"name": "modelopt", "version": modelopt.__version__},
        "model": dataset.model,
        "checkpoint_manifest_sha256": checkpoint_manifest.sha256,
        "global_num_heads": dataset.global_num_heads,
        "attention_layers": list(dataset.attention_layers),
        "max_reuse_span": dataset.max_reuse_span,
        "target_bmm1_skip_ratio": bmm1_target,
        "bmm1_skip_objective": bmm1_objective,
        "reuse_selection_objective": reuse_selection_objective,
        "anchors": list(selection.anchors),
        "nearest": {str(layer): anchor for layer, anchor in selection.nearest.items()},
        "threshold_scale_factor": fit,
        "evidence": canonical_evidence,
        "discovery_context_policies": policies,
        "constraints": constraints,
        "provenance": provenance,
        "calibration_report": report,
    }
