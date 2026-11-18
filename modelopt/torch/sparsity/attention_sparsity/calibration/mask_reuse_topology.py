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
import time
from array import array
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from hashlib import sha256
from pathlib import Path
from weakref import ref

import numpy as np
import pulp

import modelopt

from .analytical_dense import validate_analytical_dense
from .checkpoint_manifest import VerifiedCheckpointManifest
from .mask_reuse import (
    AnchorLayerStats,
    Bucket,
    MaskReuseCalibrationError,
    _AnchorEvaluation,
    _bucket_key,
    _capture_fingerprint_owner,
    _Choice,
    _donor_head_candidates,
    _dropped_mass_matrix,
    _number,
    _pareto_donor_options,
    _ReuseEvaluation,
    _risk_variant,
    _sha256,
    _validate_donor_head_mode,
    _validate_shared_prefix_provenance,
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
from .mask_reuse_contracts import (
    required_skipped_tiles,
    threshold_pair_matches,
    validate_context_buckets,
)
from .mask_reuse_solver import MaskReuseSolverConfig, MaskReuseSolverFailure, _SolverSession

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
# Select a representation, not a smaller candidate family. Every omitted risk
# row is checked before accepting every objective and lexicographic phase.
_GENERATED_RISK_TERM_THRESHOLD = 1_000_000
# Memoryview/bytes headers outweigh tuple savings on very short vectors.
_PACKED_RISK_MIN_PROMPTS = 16


def _prompt_risk_vector(values, count: int) -> Sequence[float]:
    if count < _PACKED_RISK_MIN_PROMPTS:
        return tuple(values)
    return memoryview(array("d", values).tobytes()).cast("d")


def _canonical_layer_key(value: object, label: str) -> int:
    if not isinstance(value, str) or not value.isdigit() or value != str(int(value)):
        raise MaskReuseCalibrationError(f"{label} keys must be canonical nonnegative integers")
    return int(value)


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
        analytical = validate_analytical_dense(
            raw, version_field="topology_discovery_capture_schema_version"
        )
        _exact_fields(
            raw,
            _CAPTURE_FIELDS | ({"analytical_dense"} if analytical else set()),
            "topology discovery capture",
        )
        if raw["topology_discovery_capture_schema_version"] not in ((1, 2) if analytical else (1,)):
            raise MaskReuseCalibrationError("unsupported topology_discovery_capture_schema_version")
        invocation = raw["invocation"]
        if not isinstance(invocation, Mapping):
            raise MaskReuseCalibrationError("capture invocation must be an object")
        _exact_fields(invocation, _INVOCATION_FIELDS, "topology discovery invocation")
        if (
            type(invocation["capture_schema_version"]) is not int
            or invocation["capture_schema_version"] != 2
        ):
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
        if not analytical and (threshold_log2 >= 0.0 or not 0.0 < threshold_lambda < 1.0):
            raise MaskReuseCalibrationError("threshold must be in (0, 1)")
        if not threshold_pair_matches(threshold_log2, threshold_lambda):
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
                by_anchor[anchor] = _dropped_mass_matrix(
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

    def validated_dataset(
        self, *, fit: Mapping[str, object], allow_same_group_prefixes: bool = False
    ) -> _Dataset:
        """Validate records; verified coefficient sources can reuse this result."""
        return _validate_dataset(self, fit=fit, allow_same_group_prefixes=allow_same_group_prefixes)

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
    analytical_dense_count: int


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
    if current.threshold_log2 >= 0:
        # The explicitly tagged native-dense branch resets savings/risk to zero.
        return
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
    if capture.threshold_log2.hex() != expected_log2.hex():
        raise MaskReuseCalibrationError(
            "topology discovery capture threshold_log2 differs from vanilla fit"
        )
    if not threshold_pair_matches(expected_log2, capture.threshold_lambda):
        raise MaskReuseCalibrationError(
            "topology discovery capture threshold_lambda differs from vanilla fit"
        )


def _validate_dataset(
    source: TopologyDiscoveryCaptureSource,
    *,
    fit: Mapping[str, object],
    allow_same_group_prefixes: bool = False,
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
    fingerprint_owners: dict[tuple[str, str], tuple[object, ...]] = {}
    prompt_buckets: dict[tuple[str, str], Bucket] = {}
    prompt_metadata: dict[tuple[Bucket, str, str], tuple[object, ...]] = {}
    prompt_source_rows: dict[tuple[Bucket, str, str], Mapping[str, object]] = {}
    seen: set[tuple[Bucket, str, str, float]] = set()
    capture_count = 0
    analytical_dense_count = 0
    previous_order: tuple[object, ...] | None = None
    previous_capture: TopologyDiscoveryCapture | None = None
    for capture in source:
        capture_count += 1
        analytical_dense_count += capture.threshold_log2 >= 0
        order = _capture_order_key(capture)
        if previous_order is not None and order <= previous_order:
            raise MaskReuseCalibrationError("topology discovery records are not in canonical order")
        if previous_capture is not None and (
            capture.attention_layers != previous_capture.attention_layers
            or capture.global_num_heads != previous_capture.global_num_heads
            or capture.max_reuse_span != previous_capture.max_reuse_span
        ):
            raise MaskReuseCalibrationError("capture layer set, head count, or span differs")
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
        owner = _capture_fingerprint_owner(capture, allow_same_group_prefixes)
        if (
            fingerprint_owners.setdefault((capture.split, capture.source_capture_sha256), owner)
            != owner
        ):
            raise MaskReuseCalibrationError("one source fingerprint names multiple prompt captures")
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
    try:
        validate_context_buckets(buckets)
    except ValueError as error:
        raise MaskReuseCalibrationError(str(error)) from error
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
        analytical_dense_count=analytical_dense_count,
    )


@dataclass(slots=True)
class _Accumulator:
    eligible_sum: int
    retained: np.ndarray
    anchor_prompt_means: dict[str, np.ndarray]
    anchor_prompt_values: dict[str, np.ndarray]
    prompt_mass: dict[str, Mapping[tuple[int, int], np.ndarray]]


@dataclass(frozen=True, slots=True)
class _Edge:
    anchor_tile_cost: int


@dataclass(frozen=True, slots=True)
class _DonorOption:
    choice: _Choice
    bmm1_skipped_tiles: int
    risk_by_prompt: Sequence[float]


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
    worst_prompt_total_dropped_mass: float
    mean_prompt_total_dropped_mass: float
    worst_individual_dropped_mass: float
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


def _stride_paths(layers, choices, max_reuse_span):
    if choices is None:
        return None
    if (
        not isinstance(choices, Sequence)
        or isinstance(choices, str | bytes)
        or not choices
        or any(type(stride) is not int or stride < 2 for stride in choices)
        or len(set(choices)) != len(choices)
    ):
        raise MaskReuseCalibrationError("anchor_stride_choices must be distinct integers >= 2")
    paths = {
        stride: tuple(
            (start, min(start + stride, len(layers))) for start in range(0, len(layers), stride)
        )
        for stride in sorted(choices)
    }
    if any(stop - start - 1 > max_reuse_span for path in paths.values() for start, stop in path):
        raise MaskReuseCalibrationError("capture max_reuse_span cannot support requested stride")
    return paths


def _selection_pass(
    source: TopologyDiscoveryCaptureSource,
    dataset: _Dataset,
    *,
    target_bmm1_skip_ratio: float,
    max_anchor_dropped_mass: float | None = None,
    solver_session: _SolverSession | None = None,
    donor_head_mode: str = "calibrated",
    stride_paths: Mapping[int, tuple[tuple[int, int], ...]] | None = None,
) -> tuple[_FrozenSelection, list[Mapping[str, object]]]:
    session = solver_session or _SolverSession()
    layers = dataset.attention_layers
    layer_index = {layer: index for index, layer in enumerate(layers)}
    allowed_edges = (
        None if stride_paths is None else {edge for path in stride_paths.values() for edge in path}
    )
    restricted = donor_head_mode != "calibrated" or stride_paths is not None
    accumulators: dict[tuple[Bucket, float], _Accumulator] = {}
    for bucket in dataset.buckets:
        for target in dataset.menus[bucket]:
            accumulators[(bucket, target)] = _Accumulator(
                eligible_sum=0,
                retained=np.zeros((len(layers), dataset.global_num_heads), dtype=np.int64),
                anchor_prompt_means={},
                anchor_prompt_values={},
                prompt_mass={},
            )
    for capture in source:
        session._remaining_seconds()
        if capture.split != "calibration":
            continue
        accumulator = accumulators[(capture.bucket, capture.target_sparsity)]
        accumulator.eligible_sum += capture.eligible_tiles
        prompt_means = np.zeros(len(layers), dtype=np.float64)
        prompt_values = np.zeros((len(layers), dataset.global_num_heads), dtype=np.float64)
        for layer, stats in capture.anchor_stats_by_layer.items():
            index = layer_index[layer]
            accumulator.retained[index] += np.asarray(stats.retained_tiles, dtype=np.int64)
            prompt_means[index] = sum(stats.dropped_mass) / dataset.global_num_heads
            prompt_values[index] = np.asarray(stats.dropped_mass, dtype=np.float64)
        if capture.prompt_id in accumulator.anchor_prompt_means:
            raise MaskReuseCalibrationError(
                "topology discovery captures contain a duplicate calibration prompt target"
            )
        accumulator.anchor_prompt_means[capture.prompt_id] = prompt_means
        accumulator.anchor_prompt_values[capture.prompt_id] = prompt_values
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

    # Cache only within this selection pass, whose capture-backed accumulators are fixed.
    @cache
    def donor_options(
        bucket: Bucket,
        target: float,
        *,
        anchor: int,
        consumer: int,
        head: int,
    ) -> tuple[_DonorOption, ...]:
        accumulator = accumulators[(bucket, target)]
        retained = accumulator.retained[layer_index[anchor]]
        prompts = dataset.prompts[(bucket, "calibration")]
        options = [
            _DonorOption(
                _Choice(0, True, accumulator.eligible_sum),
                0,
                _prompt_risk_vector(itertools.repeat(0.0, len(prompts)), len(prompts)),
            )
        ]
        for donor in _donor_head_candidates(donor_head_mode, head, dataset.global_num_heads):
            retained_tiles = int(retained[donor])
            options.append(
                _DonorOption(
                    _Choice(donor, False, retained_tiles),
                    accumulator.eligible_sum - retained_tiles,
                    _prompt_risk_vector(
                        (
                            float(accumulator.prompt_mass[prompt][(anchor, consumer)][head, donor])
                            for prompt in prompts
                        ),
                        len(prompts),
                    ),
                )
            )

        return _pareto_donor_options(options)

    def eligible_bmm1_tiles(targets: Mapping[Bucket, float]) -> int:
        return sum(
            accumulators[(bucket, targets[bucket])].eligible_sum
            * dataset.global_num_heads
            * len(layers)
            for bucket in dataset.buckets
        )

    def freeze_selection(
        targets, anchors, nearest, choices, combined_tile_cost, minimum, target_met
    ):
        prompt_risks: list[float] = []
        worst_individual_risk = 0.0
        for bucket in dataset.buckets:
            accumulator = accumulators[(bucket, targets[bucket])]
            for prompt in dataset.prompts[(bucket, "calibration")]:
                anchor_indices = [layer_index[anchor] for anchor in anchors]
                anchor_values = accumulator.anchor_prompt_values[prompt][anchor_indices]
                total = float(anchor_values.sum()) if max_anchor_dropped_mass is None else 0.0
                worst_individual_risk = max(
                    worst_individual_risk,
                    float(anchor_values.max()) if max_anchor_dropped_mass is None else 0.0,
                )
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
                prompt_risks.append(total / (len(layers) * dataset.global_num_heads))
        skipped = sum(
            accumulators[(bucket, targets[bucket])].eligible_sum - choice.retained_tiles
            for (bucket, _, _), choice in choices.items()
        )
        session.require_work_target(skipped, minimum, phase="topology_policy_extraction")
        if target_met:
            for bucket in dataset.buckets:
                accumulator = accumulators[(bucket, targets[bucket])]
                bucket_skipped = sum(
                    accumulator.eligible_sum - choice.retained_tiles
                    for (choice_bucket, _, _), choice in choices.items()
                    if choice_bucket == bucket
                )
                eligible = accumulator.eligible_sum * len(layers) * dataset.global_num_heads
                session.require_work_target(
                    bucket_skipped,
                    required_skipped_tiles(target_bmm1_skip_ratio, eligible),
                    phase=f"topology_bucket_{bucket}",
                )
        return _FrozenSelection(
            targets=targets,
            anchors=anchors,
            nearest=nearest,
            choices=choices,
            combined_tile_cost=combined_tile_cost,
            fallback_count=sum(choice.fallback for choice in choices.values()),
            bmm1_skipped_tiles=skipped,
            bmm1_eligible_tiles=eligible_bmm1_tiles(targets),
            target_bmm1_skip_ratio_met=target_met,
            worst_prompt_total_dropped_mass=max(prompt_risks, default=0.0),
            mean_prompt_total_dropped_mass=(
                sum(prompt_risks) / len(prompt_risks) if prompt_risks else 0.0
            ),
            worst_individual_dropped_mass=worst_individual_risk,
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
                if allowed_edges is not None and (start, stop) not in allowed_edges:
                    continue
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
        if stride_paths is not None:
            stride_variables = {
                stride: pulp.LpVariable(f"stride_{stride}", lowBound=0, upBound=1, cat="Binary")
                for stride in stride_paths
            }
            problem += pulp.lpSum(stride_variables.values()) == 1, "choose_complete_stride_path"
            for edge_key, variable in edge_variables.items():
                # A union of edges alone would admit hybrid paths mixing strides.
                problem += (
                    variable
                    == pulp.lpSum(
                        stride_variables[stride]
                        for stride, path in stride_paths.items()
                        if edge_key in path
                    ),
                    f"stride_edge_{edge_key[0]}_{edge_key[1]}",
                )
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

        # Each selected anchor contributes its own dropped mass; each selected
        # reuse consumer contributes its donor-specific dropped mass below.
        # Exact fallback contributes zero. The common denominator includes all
        # attention layer-heads, so topology and donor choices share one risk
        # objective without an anchor-only acceptance threshold.
        # Assemble coefficients directly; one temporary expression per scalar
        # duplicates the large prompt-by-donor matrix in Python objects.
        prompt_keys = tuple(
            sorted(
                (
                    (bucket, prompt)
                    for bucket in dataset.buckets
                    for prompt in dataset.prompts[(bucket, "calibration")]
                ),
                key=str,
            )
        )
        prompt_indices = {
            (bucket, prompt): index
            for bucket in dataset.buckets
            for index, prompt in enumerate(dataset.prompts[(bucket, "calibration")])
        }
        pair_count = sum(
            min(dataset.max_reuse_span, len(layers) - start - 1) for start in range(len(layers))
        )
        donor_count = dataset.global_num_heads if donor_head_mode == "calibrated" else 1
        generated_risk = (
            len(prompt_keys) * pair_count * dataset.global_num_heads * donor_count
            > _GENERATED_RISK_TERM_THRESHOLD
        )
        active_prompts = (
            {next(key for key in prompt_keys if key[0] == bucket) for bucket in dataset.buckets}
            if generated_risk
            else set(prompt_keys)
        )
        active_positions = {
            bucket: tuple(
                (index, prompt)
                for index, prompt in enumerate(dataset.prompts[(bucket, "calibration")])
                if (bucket, prompt) in active_prompts
            )
            for bucket in dataset.buckets
        }
        prompt_risk_terms = {key: pulp.LpAffineExpression() for key in active_prompts}

        def add_anchor_risk(expression, bucket, prompt):
            if max_anchor_dropped_mass is not None:
                return
            prompt_means = accumulators[(bucket, targets[bucket])].anchor_prompt_means[prompt]
            for start in range(len(layers)):
                coefficient = dataset.global_num_heads * float(prompt_means[start])
                if coefficient != 0:
                    for variable, weight in outgoing[start].items():
                        expression.addterm(variable, coefficient * weight)

        for bucket in dataset.buckets:
            accumulator = accumulators[(bucket, targets[bucket])]
            for prompt in dataset.prompts[(bucket, "calibration")]:
                prompt_means = accumulator.anchor_prompt_means[prompt]
                if max_anchor_dropped_mass is None:
                    if (bucket, prompt) in active_prompts:
                        add_anchor_risk(prompt_risk_terms[(bucket, prompt)], bucket, prompt)
                else:
                    problem += (
                        pulp.lpSum(
                            (float(prompt_means[start]) - max_anchor_dropped_mass) * outgoing[start]
                            for start in range(len(layers))
                        )
                        <= 0
                    )

        choice_variables: dict[tuple[int, Bucket, int, int, int], pulp.LpVariable] = {}
        choice_options: dict[tuple[int, Bucket, int, int, int], _DonorOption] = {}
        seed_enabled = (
            session.config.backend == "highs" and max_anchor_dropped_mass is None and not restricted
        )
        seed_choices = {}
        seed_pair_savings = defaultdict(int)
        for start in range(len(layers)):
            anchor = layers[start]
            session._remaining_seconds()
            maximum_consumer_stop = min(len(layers), start + dataset.max_reuse_span + 1)
            for consumer_index in range(start + 1, maximum_consumer_stop):
                if allowed_edges is not None and not any(
                    edge_start == start and stop > consumer_index
                    for edge_start, stop in allowed_edges
                ):
                    continue
                consumer = layers[consumer_index]
                # Donor choices depend on the anchor-consumer pair, not on the
                # selected segment's stop. Link one choice group to every edge
                # that covers this consumer instead of duplicating it per edge.
                pair_active = pulp.lpSum(
                    edge_variable
                    for (edge_start, stop), edge_variable in edge_variables.items()
                    if edge_start == start and stop > consumer_index
                )
                for bucket_index, bucket in enumerate(dataset.buckets):
                    for head in range(dataset.global_num_heads):
                        options = donor_options(
                            bucket,
                            targets[bucket],
                            anchor=anchor,
                            consumer=consumer,
                            head=head,
                        )
                        variables_for_choice = []
                        best_seed = None
                        for option_index, option in enumerate(options):
                            key = (start, bucket, consumer, head, option_index)
                            variable = pulp.LpVariable(
                                (f"choice_{start}_{bucket_index}_{consumer}_{head}_{option_index}"),
                                lowBound=0,
                                upBound=1,
                                cat="Binary",
                            )
                            choice_variables[key] = variable
                            choice_options[key] = option
                            variables_for_choice.append(variable)
                            if seed_enabled and (
                                best_seed is None
                                or option.bmm1_skipped_tiles > best_seed[1].bmm1_skipped_tiles
                            ):
                                best_seed = variable, option
                            for prompt_index, prompt in active_positions[bucket]:
                                risk = option.risk_by_prompt[prompt_index]
                                expression = prompt_risk_terms[(bucket, prompt)]
                                if risk != 0:
                                    expression.addterm(variable, risk)
                        problem += (
                            pulp.lpSum(variables_for_choice) == pair_active,
                            f"choose_{start}_{bucket_index}_{consumer}_{head}",
                        )
                        if seed_enabled:
                            assert best_seed is not None
                            seed_choices[(start, bucket, consumer, head)] = best_seed
                            seed_pair_savings[(start, consumer_index)] += best_seed[
                                1
                            ].bmm1_skipped_tiles

        cost_expression = pulp.LpAffineExpression(
            (variable, edge_rows[edge_key].anchor_tile_cost)
            for edge_key, variable in edge_variables.items()
            if edge_rows[edge_key].anchor_tile_cost != 0
        ) + pulp.LpAffineExpression(
            (variable, 2 * choice_options[key].choice.retained_tiles)
            for key, variable in choice_variables.items()
            if choice_options[key].choice.retained_tiles != 0
        )
        reuse_count_expression = pulp.lpSum(
            variable
            for key, variable in choice_variables.items()
            if not choice_options[key].choice.fallback
        )
        bmm1_skipped_expression = pulp.LpAffineExpression(
            (variable, choice_options[key].bmm1_skipped_tiles)
            for key, variable in choice_variables.items()
            if choice_options[key].bmm1_skipped_tiles != 0
        )
        anchor_count_expression = pulp.lpSum(edge_variables.values())
        worst_prompt_risk = pulp.LpVariable("worst_prompt_total_dropped_mass", lowBound=0.0)
        before_risk_names = tuple(problem.constraints)
        risk_names = {key: f"total_risk_{index}" for index, key in enumerate(prompt_keys)}

        def append_risk_constraint(current_problem, prompt_key, expression):
            # Transfer one expression into the constraint instead of retaining
            # both a source affine map and its full-sized comparison copy.
            expression.addterm(worst_prompt_risk, -1)
            current_problem += (
                pulp.LpConstraint(e=expression, sense=pulp.LpConstraintLE, rhs=0),
                risk_names[prompt_key],
            )

        for prompt_key in prompt_keys:
            if prompt_key in active_prompts:
                append_risk_constraint(problem, prompt_key, prompt_risk_terms.pop(prompt_key))

        def prompt_risk_value(prompt_key, edge_values, choice_values):
            bucket, prompt = prompt_key
            prompt_means = accumulators[(bucket, targets[bucket])].anchor_prompt_means[prompt]
            value = 0
            for start, weight, assignment in edge_values:
                coefficient = dataset.global_num_heads * float(prompt_means[start])
                if coefficient != 0:
                    value += assignment * (coefficient * weight)
            index = prompt_indices[prompt_key]
            for option, assignment in choice_values[bucket]:
                risk = option.risk_by_prompt[index]
                if risk != 0:
                    value += assignment * risk
            return value

        if generated_risk:
            problem_ref = ref(problem)
            all_risk_names = set(risk_names.values())
            before_risk_set = set(before_risk_names)
            diagnostics = {
                "total_prompts": len(prompt_keys),
                "active_prompts": len(active_prompts),
                "separation_passes": 0,
                "all_prompt_checks": 0,
                "row_materialization_seconds": 0.0,
            }
            problem._mask_reuse_risk_generation = diagnostics
            # Retain the original term order without scanning foreign buckets per row.
            choices_by_bucket = defaultdict(list)
            for key, variable in choice_variables.items():
                choices_by_bucket[key[1]].append((variable, choice_options[key]))

            def solution_value(variable):
                value = variable.varValue
                if value is None or not math.isfinite(value):
                    raise MaskReuseSolverFailure(
                        "invalid assignment during prompt-risk separation", session.records
                    )
                return value

            def separate_prompt_risks():
                current_problem = problem_ref()
                assert current_problem is not None
                diagnostics["separation_passes"] += 1
                if len(active_prompts) == len(prompt_keys):
                    # The solver session has already validated every active row.
                    return False
                # Keep every exactly nonzero assignment, including sub-.5
                # values. Native feasibility is checked by the solver session.
                edge_values = []
                if max_anchor_dropped_mass is None:
                    for start in range(len(layers)):
                        for variable, weight in outgoing[start].items():
                            value = solution_value(variable)
                            if value != 0:
                                edge_values.append((start, weight, value))
                choice_values = defaultdict(list)
                for key, variable in choice_variables.items():
                    value = solution_value(variable)
                    if value != 0:
                        choice_values[key[1]].append((choice_options[key], value))
                worst_value = solution_value(worst_prompt_risk)
                violated = []
                for prompt_key in prompt_keys:
                    if prompt_key in active_prompts:
                        continue
                    residual = prompt_risk_value(prompt_key, edge_values, choice_values)
                    residual -= worst_value
                    diagnostics["all_prompt_checks"] += 1
                    # Restore every positive row; the final assignment guard
                    # is looser than native tolerances and cannot omit a row.
                    if residual > 0:
                        violated.append(prompt_key)
                row_started = time.monotonic()
                for prompt_key in violated:
                    bucket, prompt = prompt_key
                    expression = pulp.LpAffineExpression()
                    add_anchor_risk(expression, bucket, prompt)
                    index = prompt_indices[prompt_key]
                    for variable, option in choices_by_bucket[bucket]:
                        risk = option.risk_by_prompt[index]
                        if risk != 0:
                            expression.addterm(variable, risk)
                    append_risk_constraint(current_problem, prompt_key, expression)
                    active_prompts.add(prompt_key)
                if violated:
                    # Preserve the eager risk-row order and position before
                    # saving constraints and already-fixed semantic objectives.
                    constraints = current_problem.constraints
                    ordered = [(name, constraints[name]) for name in before_risk_names]
                    ordered.extend(
                        (risk_names[key], constraints[risk_names[key]])
                        for key in prompt_keys
                        if key in active_prompts
                    )
                    ordered.extend(
                        (name, row)
                        for name, row in constraints.items()
                        if name not in before_risk_set and name not in all_risk_names
                    )
                    constraints.clear()
                    constraints.update(ordered)
                    diagnostics["active_prompts"] = len(active_prompts)
                diagnostics["row_materialization_seconds"] += time.monotonic() - row_started
                return bool(violated)

            problem._mask_reuse_separate_constraints = separate_prompt_risks

            def omitted_risk_rows():
                # Private candidate certification must also check unmaterialized prompts.
                for prompt_key in prompt_keys:
                    if prompt_key in active_prompts:
                        continue
                    bucket, prompt = prompt_key
                    expression = pulp.LpAffineExpression()
                    add_anchor_risk(expression, bucket, prompt)
                    index = prompt_indices[prompt_key]
                    for variable, option in choices_by_bucket[bucket]:
                        risk = option.risk_by_prompt[index]
                        if risk != 0:
                            expression.addterm(variable, risk)
                    expression.addterm(worst_prompt_risk, -1)
                    yield pulp.LpConstraint(e=expression, sense=pulp.LpConstraintLE, rhs=0)

            problem._mask_reuse_omitted_risk_rows = omitted_risk_rows
        if minimum_bmm1_skipped_tiles is not None:
            for bucket_index, bucket in enumerate(dataset.buckets):
                eligible = (
                    accumulators[(bucket, targets[bucket])].eligible_sum
                    * len(layers)
                    * dataset.global_num_heads
                )
                bucket_skipped = pulp.lpSum(
                    choice_options[key].bmm1_skipped_tiles * variable
                    for key, variable in choice_variables.items()
                    if key[1] == bucket
                )
                problem += (
                    bucket_skipped >= required_skipped_tiles(target_bmm1_skip_ratio, eligible),
                    f"minimum_bmm1_skipped_tiles_bucket_{bucket_index}",
                )

        if seed_enabled:
            # With no anchor-risk cap, each edge's maximum saving is separable
            # across heads and contexts. A longest DAG path supplies an incumbent.
            best_saving, predecessor = {0: 0}, {}
            for edge_key in edge_variables:
                start, stop = edge_key
                if start not in best_saving:
                    continue
                saving = best_saving[start] + sum(
                    seed_pair_savings[(start, consumer)] for consumer in range(start + 1, stop)
                )
                if stop not in best_saving or saving > best_saving[stop]:
                    best_saving[stop] = saving
                    predecessor[stop] = edge_key
            maximum_saving = best_saving[len(layers)]
            problem._mask_reuse_initial_diagnostics = {
                "kind": "maximum_bmm1_path_incumbent_not_optimal_risk_policy",
                "maximum_bmm1_skipped_tiles": maximum_saving,
                "minimum_bmm1_skipped_tiles": minimum_bmm1_skipped_tiles,
                "supplied": False,
            }
            if minimum_bmm1_skipped_tiles is None or maximum_saving >= minimum_bmm1_skipped_tiles:
                selected_edges = {}
                stop = len(layers)
                while stop:
                    start, end = predecessor[stop]
                    selected_edges[start] = end
                    stop = start
                initial_values = {
                    variable: 1.0
                    for (start, stop), variable in edge_variables.items()
                    if selected_edges.get(start) == stop
                }
                edge_values = [
                    (start, weight, 1.0)
                    for start in range(len(layers))
                    for variable, weight in outgoing[start].items()
                    if variable in initial_values
                ]
                choice_values = defaultdict(list)
                for (start, bucket, consumer, _), (variable, option) in seed_choices.items():
                    if layer_index[consumer] < selected_edges.get(start, 0):
                        initial_values[variable] = 1.0
                        choice_values[bucket].append((option, 1.0))
                totals = [prompt_risk_value(key, edge_values, choice_values) for key in prompt_keys]
                supplied_risk = max(totals)
                if supplied_risk != 0:
                    initial_values[worst_prompt_risk] = supplied_risk
                variables = problem.variables()
                previous_values = [variable.varValue for variable in variables]
                try:
                    for variable in variables:
                        variable.varValue = initial_values.get(variable, 0.0)
                    seed_feasible = problem.valid(1e-5)
                finally:
                    for variable, previous in zip(variables, previous_values, strict=True):
                        variable.varValue = previous
                # Absent variables are explicitly zero, not unspecified native
                # values. The adapter consumes this complete assignment once.
                # The pooled maximum-saving path need not satisfy every bucket.
                # It is only a warm start: discard it if any new row is violated.
                if seed_feasible:
                    problem._mask_reuse_initial_values = initial_values
                problem._mask_reuse_initial_diagnostics.update(
                    supplied=seed_feasible,
                    worst_prompt_total_dropped_mass=supplied_risk,
                    calibration_prompt_rows_checked=len(totals),
                    all_prompt_risks_valid=True,
                )

        def minimize_and_fix(
            expression: object,
            name: str,
            *,
            integral: bool,
        ) -> float | None:
            # A fixed number cannot rank candidates; the risk solve below still
            # proves feasibility. Avoid PuLP's stale constant-objective dummy.
            if restricted and not len(pulp.lpSum([expression])):
                return float(pulp.value(expression))
            return session.minimize_and_fix(problem, expression, name, integral=integral)

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
                "fix_worst_prompt_total_dropped_mass",
                integral=False,
            )
            is None
        ):
            return None
        if minimize_and_fix(cost_expression, "fix_cost", integral=True) is None:
            raise MaskReuseSolverFailure(
                "topology solver lost feasibility after fixing worst-prompt total risk",
                session.records,
            )
        deterministic_phases = (
            (reuse_count_expression, "fix_reuse_count"),
            (anchor_count_expression, "fix_anchor_count"),
        )
        for expression, fix_name in deterministic_phases:
            if minimize_and_fix(expression, fix_name, integral=True) is None:
                raise MaskReuseSolverFailure(
                    "topology solver lost deterministic tie-break feasibility", session.records
                )
        session.lexicographic(
            problem,
            # Inactive stride positions are constants, not topology tie-breaks.
            # Omitting them also avoids PuLP's transient constant-objective dummy.
            (
                1 - outgoing[start]
                for start in range(1, len(layers))
                if not restricted or len(outgoing[start])
            ),
            base=2,
            phase="fix_anchor_map",
        )
        donor_groups = defaultdict(pulp.LpAffineExpression)
        for key, variable in choice_variables.items():
            _, bucket, consumer, head, _ = key
            choice = choice_options[key].choice
            expression = donor_groups[(bucket, consumer, head)]
            if not choice.fallback:
                expression.addterm(variable, choice.donor_head + 1)
        donor_terms = (donor_groups[key] for key in sorted(donor_groups))
        if restricted:
            donor_terms = (term for term in donor_terms if len(term))
        session.lexicographic(
            problem,
            donor_terms,
            base=dataset.global_num_heads + 1,
            phase="fix_donor_map",
        )

        selected = sorted(
            edge_key for edge_key, variable in edge_variables.items() if variable.value() > 0.5
        )
        if not selected or selected[0][0] != 0 or selected[-1][1] != len(layers):
            raise MaskReuseSolverFailure(
                "topology solver returned a non-covering path", session.records
            )
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
            _, bucket, consumer, head, _ = key
            if variable.value() <= 0.5:
                continue
            option = choice_options[key]
            choices[(bucket, consumer, head)] = option.choice
            combined_tile_cost += 2 * option.choice.retained_tiles

        return freeze_selection(
            targets,
            anchors,
            nearest,
            choices,
            combined_tile_cost,
            minimum_bmm1_skipped_tiles,
            target_met,
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
        required_tiles = sum(
            required_skipped_tiles(
                target_bmm1_skip_ratio,
                accumulators[(bucket, targets[bucket])].eligible_sum
                * len(layers)
                * dataset.global_num_heads,
            )
            for bucket in dataset.buckets
        )
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
            "target_bmm1_skip_ratio": target_bmm1_skip_ratio,
            "required_bmm1_skipped_tiles": required_tiles,
            "bmm1_eligible_tiles": eligible_tiles,
            "target_bmm1_skip_ratio_feasible": solved is not None,
        }
        if solved is not None:
            rank = (
                solved.worst_prompt_total_dropped_mass,
                solved.combined_tile_cost,
                len(solved.choices) - solved.fallback_count,
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
                    "worst_prompt_model_wide_total_dropped_mass": (
                        solved.worst_prompt_total_dropped_mass
                    ),
                    "mean_prompt_model_wide_total_dropped_mass": (
                        solved.mean_prompt_total_dropped_mass
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
        row["feasible"] = maximum is not None
        if maximum is None:
            continue
        maximum_rank = (
            -maximum.bmm1_skipped_tiles,
            maximum.worst_prompt_total_dropped_mass,
            maximum.combined_tile_cost,
            len(maximum.choices) - maximum.fallback_count,
            len(maximum.anchors),
            target_values,
        )
        maximum_candidates.append((maximum_rank, maximum))
        row.update(
            {
                "maximum_feasible_bmm1_skipped_tiles": maximum.bmm1_skipped_tiles,
                "maximum_feasible_bmm1_skip_ratio": maximum.bmm1_skip_ratio,
                "maximum_policy_worst_prompt_model_wide_total_dropped_mass": (
                    maximum.worst_prompt_total_dropped_mass
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
            combined_tile_cost=2 * eligible_bmm1_tiles(exact_targets),
            fallback_count=0,
            bmm1_skipped_tiles=0,
            bmm1_eligible_tiles=eligible_bmm1_tiles(exact_targets),
            target_bmm1_skip_ratio_met=False,
            worst_prompt_total_dropped_mass=0.0,
            mean_prompt_total_dropped_mass=0.0,
            worst_individual_dropped_mass=0.0,
            exact_reason="no_joint_target_and_topology_had_a_feasible_solver_solution",
        ),
        frontier,
    )


def _evaluation_pass(
    source: TopologyDiscoveryCaptureSource,
    dataset: _Dataset,
    selection: _FrozenSelection,
    *,
    anchor_dropped_mass_report_threshold: float,
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
        previous_mass = reuse_result.dropped_mass_sum
        head_count = len(dataset.attention_layers) * dataset.global_num_heads
        reuse_result.model_wide_eligible_tiles += capture.eligible_tiles * head_count
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
        total_risk = (
            reuse_result.dropped_mass_sum - previous_mass + sum(dropped_values)
        ) / head_count
        reuse_result.prompt_count += 1
        reuse_result.total_prompt_risk_sum += total_risk
        reuse_result.worst_prompt_total_risk = max(reuse_result.worst_prompt_total_risk, total_risk)
        anchor_result.prompt_count += 1
        anchor_result.prompt_mean_sum += prompt_mean
        anchor_result.worst_prompt_mean = max(anchor_result.worst_prompt_mean, prompt_mean)
        anchor_result.violations += int(prompt_mean > anchor_dropped_mass_report_threshold)
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
    anchor_dropped_mass_report_threshold: float | None = None,
    max_anchor_dropped_mass: float | None = None,
    reuse_dropped_mass_report_threshold: float,
    target_bmm1_skip_ratio: float,
    solver_config: MaskReuseSolverConfig | None = None,
    donor_head_mode: str = "calibrated",
    anchor_stride_choices: Sequence[int] | None = None,
    allow_same_group_prefixes: bool = False,
    source_provenance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Select a global candidate topology and evaluate it without held-out tuning.

    ``target_bmm1_skip_ratio`` is the minimum fraction of eligible QK tiles
    skipped by non-fallback reuse consumers across all attention layers in the
    calibration split. Among policies that meet the target, selection minimizes
    worst-prompt model-wide anchor-plus-reuse dropped mass. If the target is
    structurally infeasible,
    the returned candidate maximizes BMM1 skips and explicitly reports the gap.

    The anchor and reuse dropped-mass report thresholds only count diagnostic
    exceedances after selection. They never accept, reject, or retune a policy.

    ``same_index`` donor mode restricts reuse to the matching global head index,
    retaining exact fallback. Optional anchor strides refer to full-attention
    layer positions and restrict selection to one complete stride topology.

    Shared prefixes require separately verified whole-prefill parent provenance;
    original request multiplicities are preserved, not deduplicated.
    """
    solver_session = _SolverSession(solver_config)
    _validate_donor_head_mode(donor_head_mode)
    _validate_shared_prefix_provenance(allow_same_group_prefixes, source_provenance)
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
    dataset = (
        source.validated_dataset(fit=fit, allow_same_group_prefixes=True)
        if allow_same_group_prefixes
        else source.validated_dataset(fit=fit)
    )
    if any(hi is None for _, hi in dataset.buckets):
        raise MaskReuseCalibrationError("calibration context ranges require finite maximum lengths")
    stride_paths = _stride_paths(
        dataset.attention_layers, anchor_stride_choices, dataset.max_reuse_span
    )
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
    anchor_bound, anchor_report_threshold, objective_variant = _risk_variant(
        max_anchor_dropped_mass, anchor_dropped_mass_report_threshold
    )
    report_only = anchor_bound is None
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
    solver_session._remaining_seconds()
    selection, frontier = _selection_pass(
        source,
        dataset,
        target_bmm1_skip_ratio=bmm1_target,
        max_anchor_dropped_mass=anchor_bound,
        solver_session=solver_session,
        donor_head_mode=donor_head_mode,
        stride_paths=stride_paths,
    )
    reuse_evaluations, anchor_evaluations, bmm1_evaluations = _evaluation_pass(
        source,
        dataset,
        selection,
        anchor_dropped_mass_report_threshold=anchor_report_threshold,
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
                    exact=selection.targets is None, report_only=report_only
                ),
                "anchor_heldout": anchor_heldout.to_mapping(
                    exact=selection.targets is None, report_only=report_only
                ),
                "bmm1_calibration": bmm1_calibration.to_mapping(),
                "bmm1_heldout": bmm1_heldout.to_mapping(),
                "bmm1_skip_objective": {
                    "target": bmm1_target,
                    "target_met": bmm1_calibration.skipped_tiles
                    >= required_skipped_tiles(bmm1_target, bmm1_calibration.eligible_tiles),
                    "eligible_tiles": bmm1_calibration.eligible_tiles,
                    "skipped_tiles": bmm1_calibration.skipped_tiles,
                },
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
        "minimum_required_skipped_tiles": sum(
            required_skipped_tiles(
                bmm1_target, bmm1_evaluations[(bucket, "calibration")].eligible_tiles
            )
            for bucket in dataset.buckets
        ),
        "scope": "every_context_bucket",
        "calibration_skipped_tiles": selection.bmm1_skipped_tiles,
        "achieved": selection.bmm1_skip_ratio,
        "maximum_pooled_skip_ratio": (
            selection.bmm1_skip_ratio if selection.target_bmm1_skip_ratio_met is False else None
        ),
    }
    risk_selection_objective = {
        "metric": (
            "per_prompt_mean_across_all_attention_layers_and_heads_anchor_or_reuse_dropped_mass"
            if report_only
            else "per_prompt_mean_across_all_attention_layers_and_heads_reuse_dropped_mass"
        ),
        "selection_split": "calibration",
        "optimization": ["minimum_worst_prompt"],
        "hard_maximum": None,
        "exact_fallback_risk": 0.0,
        "worst_development_prompt_model_wide_total_dropped_mass": (
            selection.worst_prompt_total_dropped_mass
        ),
        "mean_development_prompt_model_wide_total_dropped_mass": (
            selection.mean_prompt_total_dropped_mass
        ),
        "worst_individual_dropped_mass": selection.worst_individual_dropped_mass,
    }

    constraints = {
        "anchor": {
            "selection_hard_maximum": anchor_bound,
            "report_metric": (
                "per_prompt_mean_across_selected_anchor_layers_and_heads_dropped_mass"
            ),
            "report_threshold": anchor_report_threshold,
            "report_threshold_affects_selection": not report_only,
        },
        "reuse": {
            "selection_hard_maximum": None,
            "report_metric": "per_prompt_candidate_reuse_dropped_mass",
            "report_threshold": reuse_report_threshold,
            "report_threshold_affects_selection": False,
        },
        "bmm1_skip_ratio": bmm1_objective,
    }
    if not report_only:
        constraints["anchor"] = {
            "metric": "per_prompt_mean_across_selected_anchor_layers_and_heads_dropped_mass",
            "comparison": "<=",
            "maximum": anchor_bound,
        }
        risk_selection_objective["worst_development_prompt_model_wide_dropped_mass"] = (
            selection.worst_prompt_total_dropped_mass
        )
        risk_selection_objective["mean_development_prompt_model_wide_dropped_mass"] = (
            selection.mean_prompt_total_dropped_mass
        )
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
        "risk_selection_objective": risk_selection_objective,
        "constraints": constraints,
        "joint_frontier": frontier,
        "by_bucket": bucket_reports,
        "overall": {
            "reuse_calibration": overall_reuse_calibration.to_mapping(),
            "reuse_heldout": overall_reuse_heldout.to_mapping(),
            "anchor_calibration": overall_anchor_calibration.to_mapping(
                exact=selection.targets is None, report_only=report_only
            ),
            "anchor_heldout": overall_anchor_heldout.to_mapping(
                exact=selection.targets is None, report_only=report_only
            ),
            "bmm1_calibration": overall_bmm1_calibration.to_mapping(),
            "bmm1_heldout": overall_bmm1_heldout.to_mapping(),
        },
    }
    provenance = {
        "calibrator": "modelopt.mask_reuse.topology_discovery",
        "objective_variant": objective_variant,
        "solver": solver_session.provenance(),
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
            "minimum worst-prompt model-wide anchor-plus-reuse dropped mass"
            if report_only
            else "minimum worst-prompt model-wide reuse dropped mass",
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
    restrictions: dict[str, object] = {}
    if donor_head_mode != "calibrated":
        restrictions["donor_head_mode"] = donor_head_mode
    if stride_paths is not None:
        restrictions["anchor_stride_choices"] = list(stride_paths)
        restrictions["selected_anchor_stride"] = next(
            (
                stride
                for stride, path in stride_paths.items()
                if tuple(dataset.attention_layers[start] for start, _ in path) == selection.anchors
            ),
            None,
        )
    if restrictions:
        provenance["candidate_restrictions"] = restrictions
    if dataset.analytical_dense_count:
        provenance.pop("topology_discovery_capture_schema_version")
        provenance["input_capture_schema_versions"] = (
            [2] if dataset.analytical_dense_count == dataset.capture_count else [1, 2]
        )
        provenance["analytical_dense_capture_count"] = dataset.analytical_dense_count
    return {
        "topology_selection_schema_version": 1,
        "promotion_status": "topology_candidate_only",
        "objective_variant": objective_variant,
        "phase": "prefill",
        "producer": {"name": "modelopt", "version": modelopt.__version__},
        "model": dataset.model,
        "checkpoint_manifest_sha256": checkpoint_manifest.sha256,
        "global_num_heads": dataset.global_num_heads,
        "attention_layers": list(dataset.attention_layers),
        "max_reuse_span": dataset.max_reuse_span,
        "target_bmm1_skip_ratio": bmm1_target,
        "bmm1_skip_objective": bmm1_objective,
        "risk_selection_objective": risk_selection_objective,
        **({"reuse_selection_objective": risk_selection_objective} if not report_only else {}),
        "anchors": list(selection.anchors),
        "nearest": {str(layer): anchor for layer, anchor in selection.nearest.items()},
        "threshold_scale_factor": fit,
        "evidence": canonical_evidence,
        "discovery_context_policies": policies,
        "constraints": constraints,
        "provenance": provenance,
        "calibration_report": report,
    }
