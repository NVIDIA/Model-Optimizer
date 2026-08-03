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

"""Streaming selector for compact mask-reuse capture JSONL.

One compact record stores the anchor vectors and consumer ``[H, H]`` risk
matrices for a single prompt and target sparsity.  The selector makes three
semantic passes (validation, calibration selection, frozen-policy evaluation)
and hashes the exact file bytes before and after selection/evaluation to detect
concurrent mutation. Selection retains development risk matrices long enough to
optimize the worst prompt under the requested BMM1 target; held-out captures
remain evaluation-only.
"""

from __future__ import annotations

import json
import math
import unicodedata
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import cast

import numpy as np
import pulp

import modelopt

from .checkpoint_manifest import VerifiedCheckpointManifest
from .mask_reuse import (
    _CALIBRATION_PROTOCOL,
    _DEPLOYMENT_GEOMETRY_CONTRACT,
    _SOLVER_LEXICOGRAPHIC_ATOL,
    AnchorLayerStats,
    Bucket,
    ConsumerHead,
    MaskReuseCalibrationError,
    _AnchorEvaluation,
    _bucket_key,
    _canonical_evidence,
    _Choice,
    _normalize_topology,
    _number,
    _ReuseEvaluation,
    _Selection,
    _sha256,
    canonical_prefill_threshold_scale_factor,
)

__all__ = [
    "CompactMaskReuseCapture",
    "CompactMaskReuseCaptureSource",
    "calibrate_compact_mask_reuse_policy",
    "load_compact_mask_reuse_captures",
]

_CAPTURE_FIELDS = frozenset(
    {
        "compact_capture_schema_version",
        "invocation",
        "geometry",
        "global_num_heads",
        "eligible_tiles",
        "anchor_stats_by_layer",
        "consumer_layers",
    }
)
_INVOCATION_FIELDS = frozenset(
    {
        "capture_schema_version",
        "model",
        "checkpoint_manifest_sha256",
        "split",
        "partition",
        "inner_fold",
        "prompt_id",
        "source",
        "source_group_sha256",
        "source_capture_sha256",
        "min_kv_tokens",
        "max_kv_tokens",
        "target_sparsity_hex",
        "sample_length",
        "threshold_log2_hex",
        "threshold_lambda_hex",
        "expected_geometry",
    }
)
_GEOMETRY_FIELDS = frozenset({"q_tokens", "kv_tokens", "q_start_tokens"})
_ANCHOR_FIELDS = frozenset({"retained_tiles", "dropped_mass"})
_CONSUMER_FIELDS = frozenset({"anchor_layer", "dropped_mass"})
_SPLITS = ("calibration", "heldout")
_PARTITIONS = ("development", "outer_test")
_MONOTONIC_ATOL = 1e-7


def _exact_fields(raw: Mapping[str, object], expected: frozenset[str], label: str) -> None:
    missing = expected - raw.keys()
    extra = raw.keys() - expected
    if missing or extra:
        raise MaskReuseCalibrationError(
            f"{label} fields do not match the schema; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise MaskReuseCalibrationError(f"{label} must be an integer >= {minimum}")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise MaskReuseCalibrationError(f"{label} must be a non-empty string")
    if unicodedata.normalize("NFC", value) != value or any(
        ord(character) < 32 for character in value
    ):
        raise MaskReuseCalibrationError(f"{label} must be NFC text without control characters")
    return value


def _float_hex(value: object, label: str) -> float:
    if not isinstance(value, str):
        raise MaskReuseCalibrationError(f"{label} must be a canonical float.hex string")
    try:
        parsed = float.fromhex(value)
    except ValueError as error:
        raise MaskReuseCalibrationError(f"{label} must be a canonical float.hex string") from error
    if not math.isfinite(parsed) or parsed.hex() != value:
        raise MaskReuseCalibrationError(f"{label} must be a canonical finite float.hex string")
    return parsed


def _geometry(raw: object, label: str) -> dict[str, int]:
    if not isinstance(raw, Mapping):
        raise MaskReuseCalibrationError(f"{label} must be an object")
    _exact_fields(raw, _GEOMETRY_FIELDS, label)
    q_tokens = _integer(raw["q_tokens"], f"{label}.q_tokens", minimum=129)
    if q_tokens > int(cast("int", _DEPLOYMENT_GEOMETRY_CONTRACT["max_query_chunk_tokens"])):
        raise MaskReuseCalibrationError(f"{label}.q_tokens exceeds the deployment maximum")
    kv_tokens = _integer(raw["kv_tokens"], f"{label}.kv_tokens", minimum=q_tokens)
    q_start = _integer(raw["q_start_tokens"], f"{label}.q_start_tokens")
    alignment = int(
        cast("int", _DEPLOYMENT_GEOMETRY_CONTRACT["query_chunk_start_alignment_tokens"])
    )
    if q_start % alignment or q_start + q_tokens != kv_tokens:
        raise MaskReuseCalibrationError(f"{label} is not a bottom-right aligned final chunk")
    return {"q_tokens": q_tokens, "kv_tokens": kv_tokens, "q_start_tokens": q_start}


def _eligible_tiles(geometry: Mapping[str, int]) -> int:
    q_blocks = (geometry["q_tokens"] + 127) // 128
    first_eligible = geometry["q_start_tokens"] // 128 + 1
    return q_blocks * (2 * first_eligible + q_blocks - 1) // 2


@dataclass(frozen=True, slots=True)
class CompactConsumerStats:
    """One consumer layer's global consumer-by-donor dropped-mass matrix."""

    anchor_layer: int
    dropped_mass: Sequence[Sequence[float]]


@dataclass(frozen=True, slots=True)
class CompactMaskReuseCapture:
    """One prompt/target capture decoded without expanding candidate rows."""

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
    anchor_stats_by_layer: Mapping[int, AnchorLayerStats]
    consumer_layers: Mapping[int, CompactConsumerStats]

    @property
    def bucket(self) -> Bucket:
        """Return this capture's context bounds."""
        return self.min_kv_tokens, self.max_kv_tokens

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> CompactMaskReuseCapture:
        """Parse one strict compact-capture object."""
        _exact_fields(raw, _CAPTURE_FIELDS, "compact capture")
        if raw["compact_capture_schema_version"] != 1:
            raise MaskReuseCalibrationError("compact_capture_schema_version must be 1")
        invocation = raw["invocation"]
        if not isinstance(invocation, Mapping):
            raise MaskReuseCalibrationError("compact capture invocation must be an object")
        _exact_fields(invocation, _INVOCATION_FIELDS, "compact capture invocation")
        if invocation["capture_schema_version"] != 2:
            raise MaskReuseCalibrationError("capture_schema_version must be 2")
        split = invocation["split"]
        if split not in _SPLITS:
            raise MaskReuseCalibrationError("capture split must be calibration or heldout")
        partition = invocation["partition"]
        if partition not in _PARTITIONS:
            raise MaskReuseCalibrationError("capture partition must be development or outer_test")
        expected_split = "calibration" if partition == "development" else "heldout"
        if split != expected_split:
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
        observed_geometry = _geometry(raw["geometry"], "geometry")
        if (
            expected_geometry != observed_geometry
            or observed_geometry["kv_tokens"] != sample_length
        ):
            raise MaskReuseCalibrationError("capture geometry does not match its invocation")
        global_num_heads = _integer(raw["global_num_heads"], "global_num_heads", minimum=1)
        eligible_tiles = _integer(raw["eligible_tiles"], "eligible_tiles", minimum=1)
        if eligible_tiles != _eligible_tiles(observed_geometry):
            raise MaskReuseCalibrationError(
                "eligible_tiles does not match 128x128 bottom-right causal geometry"
            )

        raw_anchors = raw["anchor_stats_by_layer"]
        if not isinstance(raw_anchors, Mapping) or not raw_anchors:
            raise MaskReuseCalibrationError("anchor_stats_by_layer must be non-empty")
        anchors: dict[int, AnchorLayerStats] = {}
        for raw_layer, raw_stats in raw_anchors.items():
            if (
                not isinstance(raw_layer, str)
                or not raw_layer.isdigit()
                or raw_layer != str(int(raw_layer))
            ):
                raise MaskReuseCalibrationError("anchor layer keys must be canonical integers")
            layer = int(raw_layer)
            if not isinstance(raw_stats, Mapping):
                raise MaskReuseCalibrationError(f"anchor_stats_by_layer[{layer}] must be an object")
            _exact_fields(raw_stats, _ANCHOR_FIELDS, f"anchor_stats_by_layer[{layer}]")
            retained_raw = raw_stats["retained_tiles"]
            dropped_raw = raw_stats["dropped_mass"]
            if not isinstance(retained_raw, list) or not isinstance(dropped_raw, list):
                raise MaskReuseCalibrationError(f"anchor_stats_by_layer[{layer}] must use arrays")
            if len(retained_raw) != global_num_heads or len(dropped_raw) != global_num_heads:
                raise MaskReuseCalibrationError(
                    f"anchor_stats_by_layer[{layer}] does not cover all global heads"
                )
            retained = tuple(
                _integer(value, f"anchor {layer} retained[{head}]")
                for head, value in enumerate(retained_raw)
            )
            if any(value > eligible_tiles for value in retained):
                raise MaskReuseCalibrationError(f"anchor {layer} retained tiles exceed eligible")
            dropped = tuple(
                _number(value, f"anchor {layer} dropped[{head}]", minimum=0.0, maximum=1.0)
                for head, value in enumerate(dropped_raw)
            )
            anchors[layer] = AnchorLayerStats(retained, dropped)

        raw_consumers = raw["consumer_layers"]
        if not isinstance(raw_consumers, Mapping) or not raw_consumers:
            raise MaskReuseCalibrationError("consumer_layers must be non-empty")
        consumers: dict[int, CompactConsumerStats] = {}
        for raw_layer, raw_stats in raw_consumers.items():
            if (
                not isinstance(raw_layer, str)
                or not raw_layer.isdigit()
                or raw_layer != str(int(raw_layer))
            ):
                raise MaskReuseCalibrationError("consumer layer keys must be canonical integers")
            layer = int(raw_layer)
            if not isinstance(raw_stats, Mapping):
                raise MaskReuseCalibrationError(f"consumer_layers[{layer}] must be an object")
            _exact_fields(raw_stats, _CONSUMER_FIELDS, f"consumer_layers[{layer}]")
            anchor = _integer(raw_stats["anchor_layer"], f"consumer_layers[{layer}].anchor")
            matrix = raw_stats["dropped_mass"]
            if not isinstance(matrix, list) or len(matrix) != global_num_heads:
                raise MaskReuseCalibrationError(
                    f"consumer_layers[{layer}] must have {global_num_heads} consumer rows"
                )
            for consumer_head, row in enumerate(matrix):
                if not isinstance(row, list) or len(row) != global_num_heads:
                    raise MaskReuseCalibrationError(
                        f"consumer_layers[{layer}][{consumer_head}] must cover all donors"
                    )
                for donor_head, value in enumerate(row):
                    _number(
                        value,
                        f"consumer_layers[{layer}][{consumer_head}][{donor_head}]",
                        minimum=0.0,
                        maximum=1.0,
                    )
            consumers[layer] = CompactConsumerStats(anchor, matrix)
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
            geometry=observed_geometry,
            global_num_heads=global_num_heads,
            eligible_tiles=eligible_tiles,
            anchor_stats_by_layer=dict(sorted(anchors.items())),
            consumer_layers=dict(sorted(consumers.items())),
        )


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise MaskReuseCalibrationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


@dataclass(frozen=True, slots=True)
class CompactMaskReuseCaptureSource:
    """Re-iterable strict JSONL source used by the three-pass selector."""

    path: Path

    def __iter__(self) -> Iterator[CompactMaskReuseCapture]:
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
                        f"line {line_number}: compact capture must be an object"
                    )
                try:
                    yield CompactMaskReuseCapture.from_mapping(raw)
                except MaskReuseCalibrationError as error:
                    raise MaskReuseCalibrationError(f"line {line_number}: {error}") from error
            if not seen:
                raise MaskReuseCalibrationError("compact capture input is empty")

    def sha256(self) -> str:
        """Hash exact file bytes without loading the capture bundle."""
        digest = sha256()
        with self.path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


def load_compact_mask_reuse_captures(path: str | Path) -> CompactMaskReuseCaptureSource:
    """Return a lazy, re-iterable compact-capture source."""
    source = CompactMaskReuseCaptureSource(Path(path))
    if not source.path.is_file():
        raise MaskReuseCalibrationError(f"compact capture file does not exist: {source.path}")
    return source


@dataclass(frozen=True, slots=True)
class _Dataset:
    model: str
    checkpoint_manifest_sha256: str
    global_num_heads: int
    buckets: tuple[Bucket, ...]
    anchors: tuple[int, ...]
    nearest: Mapping[int, int]
    consumer_layers: tuple[int, ...]
    targets: tuple[ConsumerHead, ...]
    prompts: Mapping[tuple[Bucket, str], tuple[str, ...]]
    menus: Mapping[Bucket, tuple[float, ...]]
    deployment_geometry: Mapping[str, object]
    prompt_sources: tuple[Mapping[str, object], ...]
    calibration_prompt_ids: tuple[str, ...]
    heldout_prompt_ids: tuple[str, ...]
    capture_count: int


def _validate_threshold(capture: CompactMaskReuseCapture, fit: Mapping[str, object]) -> None:
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
        raise MaskReuseCalibrationError("compact capture threshold_log2 differs from vanilla fit")
    if capture.threshold_lambda.hex() != expected_lambda.hex():
        raise MaskReuseCalibrationError("compact capture threshold_lambda differs from vanilla fit")


def _capture_order_key(capture: CompactMaskReuseCapture) -> tuple[object, ...]:
    return (
        capture.min_kv_tokens,
        math.inf if capture.max_kv_tokens is None else capture.max_kv_tokens,
        _SPLITS.index(capture.split),
        capture.prompt_id,
        capture.target_sparsity,
    )


def _validate_monotonic_pair(
    previous: CompactMaskReuseCapture, current: CompactMaskReuseCapture
) -> None:
    identity = (current.bucket, current.split, current.prompt_id)
    if identity != (previous.bucket, previous.split, previous.prompt_id):
        return
    if current.target_sparsity <= previous.target_sparsity:
        raise MaskReuseCalibrationError("target sparsities must be strictly increasing per prompt")
    for layer, current_stats in current.anchor_stats_by_layer.items():
        previous_stats = previous.anchor_stats_by_layer[layer]
        if any(
            current_value > previous_value
            for current_value, previous_value in zip(
                current_stats.retained_tiles,
                previous_stats.retained_tiles,
                strict=True,
            )
        ):
            raise MaskReuseCalibrationError(
                f"anchor {layer} retained counts increase with target sparsity"
            )
        if any(
            current_value + _MONOTONIC_ATOL < previous_value
            for current_value, previous_value in zip(
                current_stats.dropped_mass,
                previous_stats.dropped_mass,
                strict=True,
            )
        ):
            raise MaskReuseCalibrationError(
                f"anchor {layer} dropped mass decreases with target sparsity"
            )
    for layer, current_consumer_stats in current.consumer_layers.items():
        previous_consumer_stats = previous.consumer_layers[layer]
        for consumer_head, (current_row, previous_row) in enumerate(
            zip(
                current_consumer_stats.dropped_mass,
                previous_consumer_stats.dropped_mass,
                strict=True,
            )
        ):
            if any(
                current_value + _MONOTONIC_ATOL < previous_value
                for current_value, previous_value in zip(current_row, previous_row, strict=True)
            ):
                raise MaskReuseCalibrationError(
                    f"consumer {layer} head {consumer_head} dropped mass decreases "
                    "with target sparsity"
                )


def _validate_dataset(
    source: CompactMaskReuseCaptureSource,
    *,
    fit: Mapping[str, object],
    anchors: tuple[int, ...],
    nearest: Mapping[int, int],
) -> _Dataset:
    models: set[str] = set()
    checkpoint_identities: set[str] = set()
    head_counts: set[int] = set()
    split_counts: dict[str, int] = defaultdict(int)
    split_buckets: dict[str, set[Bucket]] = defaultdict(set)
    prompts: dict[tuple[Bucket, str], set[str]] = defaultdict(set)
    menus: dict[tuple[Bucket, str, str], set[float]] = defaultdict(set)
    capture_sources: dict[tuple[Bucket, str, str], tuple[object, ...]] = {}
    fingerprint_owner: dict[tuple[str, str], tuple[Bucket, str]] = {}
    seen: set[tuple[Bucket, str, str, float]] = set()
    geometry_by_prompt: dict[tuple[Bucket, str, str], tuple[object, ...]] = {}
    prompt_source_rows: dict[tuple[Bucket, str, str], Mapping[str, object]] = {}
    split_prompt_ids: dict[str, set[str]] = defaultdict(set)
    prompt_buckets: dict[tuple[str, str], Bucket] = {}
    split_fingerprints: dict[str, set[str]] = defaultdict(set)
    group_assignments: dict[str, tuple[str, int | None]] = {}
    fingerprint_groups: dict[str, str] = {}
    expected_anchor_set = set(anchors)
    consumer_to_anchor = {layer: anchor for layer, anchor in nearest.items() if layer != anchor}
    expected_consumer_set = set(consumer_to_anchor)
    capture_count = 0
    previous_order: tuple[object, ...] | None = None
    previous_capture: CompactMaskReuseCapture | None = None
    for capture in source:
        capture_count += 1
        order = _capture_order_key(capture)
        if previous_order is not None and order <= previous_order:
            raise MaskReuseCalibrationError("compact capture records are not in canonical order")
        if previous_capture is not None:
            _validate_monotonic_pair(previous_capture, capture)
        previous_order = order
        previous_capture = capture
        _validate_threshold(capture, fit)
        if set(capture.anchor_stats_by_layer) != expected_anchor_set:
            raise MaskReuseCalibrationError("compact capture does not cover every topology anchor")
        if set(capture.consumer_layers) != expected_consumer_set:
            raise MaskReuseCalibrationError("compact capture does not cover every reuse layer")
        for layer, stats in capture.consumer_layers.items():
            if stats.anchor_layer != consumer_to_anchor[layer]:
                raise MaskReuseCalibrationError(
                    f"consumer layer {layer} does not match the explicit topology"
                )
        models.add(capture.model)
        checkpoint_identities.add(capture.checkpoint_manifest_sha256)
        head_counts.add(capture.global_num_heads)
        split_counts[capture.split] += 1
        split_buckets[capture.split].add(capture.bucket)
        prompts[(capture.bucket, capture.split)].add(capture.prompt_id)
        menus[(capture.bucket, capture.split, capture.prompt_id)].add(capture.target_sparsity)
        split_prompt_ids[capture.split].add(capture.prompt_id)
        prompt_identity = (capture.split, capture.prompt_id)
        if prompt_buckets.setdefault(prompt_identity, capture.bucket) != capture.bucket:
            raise MaskReuseCalibrationError("a prompt ID is assigned to multiple context buckets")
        split_fingerprints[capture.split].add(capture.source_capture_sha256)
        assignment = (capture.partition, capture.inner_fold)
        previous_assignment = group_assignments.setdefault(capture.source_group_sha256, assignment)
        if previous_assignment != assignment:
            raise MaskReuseCalibrationError(
                "one source group is assigned to multiple partitions or inner folds"
            )
        previous_group = fingerprint_groups.setdefault(
            capture.source_capture_sha256, capture.source_group_sha256
        )
        if previous_group != capture.source_group_sha256:
            raise MaskReuseCalibrationError(
                "one rendered source capture is assigned to multiple source groups"
            )
        key = (capture.bucket, capture.split, capture.prompt_id, capture.target_sparsity)
        if key in seen:
            raise MaskReuseCalibrationError("compact captures contain a duplicate prompt target")
        seen.add(key)
        prompt_key = (capture.bucket, capture.split, capture.prompt_id)
        source_identity = (
            capture.source,
            capture.source_group_sha256,
            capture.partition,
            capture.inner_fold,
            capture.source_capture_sha256,
            capture.sample_length,
            tuple(sorted(capture.geometry.items())),
            capture.eligible_tiles,
        )
        if capture_sources.setdefault(prompt_key, source_identity) != source_identity:
            raise MaskReuseCalibrationError("prompt metadata differs across target sparsities")
        owner_key = (capture.split, capture.source_capture_sha256)
        owner = (capture.bucket, capture.prompt_id)
        if fingerprint_owner.setdefault(owner_key, owner) != owner:
            raise MaskReuseCalibrationError("one source fingerprint names multiple prompt captures")
        geometry_identity = (
            capture.source_capture_sha256,
            capture.geometry["q_tokens"],
            capture.geometry["kv_tokens"],
            capture.geometry["q_start_tokens"],
        )
        if geometry_by_prompt.setdefault(prompt_key, geometry_identity) != geometry_identity:
            raise MaskReuseCalibrationError("prompt geometry differs across target sparsities")
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
        raise MaskReuseCalibrationError("compact captures require calibration and heldout splits")
    if len(models) != 1 or len(head_counts) != 1 or len(checkpoint_identities) != 1:
        raise MaskReuseCalibrationError(
            "compact captures must use one model, checkpoint, and head count"
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
        if not menu:
            raise MaskReuseCalibrationError(f"bucket {bucket} has no target menu")
        for split in _SPLITS:
            for prompt in prompts[(bucket, split)]:
                if menus[(bucket, split, prompt)] != menu:
                    raise MaskReuseCalibrationError("target-sparsity menu differs across prompts")
        canonical_menus[bucket] = tuple(sorted(menu))

    geometry_rows = []
    for (bucket, split, prompt), identity in sorted(
        geometry_by_prompt.items(), key=lambda item: str(item[0])
    ):
        geometry_rows.append(
            {
                "split": split,
                "prompt_id": prompt,
                "source_capture_sha256": identity[0],
                "min_kv_tokens": bucket[0],
                "max_kv_tokens": bucket[1],
                "q_tokens": identity[1],
                "kv_tokens": identity[2],
                "q_start_tokens": identity[3],
            }
        )
    global_num_heads = next(iter(head_counts))
    consumer_layers = tuple(sorted(expected_consumer_set))
    targets = tuple((layer, head) for layer in consumer_layers for head in range(global_num_heads))
    return _Dataset(
        model=next(iter(models)),
        checkpoint_manifest_sha256=next(iter(checkpoint_identities)),
        global_num_heads=global_num_heads,
        buckets=buckets,
        anchors=anchors,
        nearest=nearest,
        consumer_layers=consumer_layers,
        targets=targets,
        prompts={key: tuple(sorted(value)) for key, value in prompts.items()},
        menus=canonical_menus,
        deployment_geometry={
            "contract": dict(_DEPLOYMENT_GEOMETRY_CONTRACT),
            "observations": geometry_rows,
        },
        prompt_sources=tuple(
            prompt_source_rows[key] for key in sorted(prompt_source_rows, key=str)
        ),
        calibration_prompt_ids=tuple(sorted(split_prompt_ids["calibration"])),
        heldout_prompt_ids=tuple(sorted(split_prompt_ids["heldout"])),
        capture_count=capture_count,
    )


@dataclass(slots=True)
class _SelectionAccumulator:
    prompt_mass: Mapping[float, dict[str, np.ndarray]]
    retained_by_anchor: Mapping[float, np.ndarray]
    eligible_sum: dict[float, int]
    anchor_evaluation: Mapping[float, _AnchorEvaluation]


@dataclass(frozen=True, slots=True)
class _DonorOption:
    choice: _Choice
    bmm1_skipped_tiles: int
    risk_by_prompt: tuple[float, ...]


def _selection_pass(
    source: CompactMaskReuseCaptureSource,
    dataset: _Dataset,
    *,
    max_anchor_dropped_mass: float,
    target_bmm1_skip_ratio: float,
) -> dict[Bucket, _Selection]:
    consumer_index = {layer: index for index, layer in enumerate(dataset.consumer_layers)}
    anchor_index = {layer: index for index, layer in enumerate(dataset.anchors)}
    accumulators: dict[Bucket, _SelectionAccumulator] = {}
    for bucket in dataset.buckets:
        menus = dataset.menus[bucket]
        accumulators[bucket] = _SelectionAccumulator(
            prompt_mass={target: {} for target in menus},
            retained_by_anchor={
                target: np.zeros((len(dataset.anchors), dataset.global_num_heads), dtype=np.int64)
                for target in menus
            },
            eligible_sum=dict.fromkeys(menus, 0),
            anchor_evaluation={target: _AnchorEvaluation() for target in menus},
        )
    for capture in source:
        if capture.split != "calibration":
            continue
        accumulator = accumulators[capture.bucket]
        target = capture.target_sparsity
        accumulator.eligible_sum[target] += capture.eligible_tiles
        for layer, stats in capture.anchor_stats_by_layer.items():
            accumulator.retained_by_anchor[target][anchor_index[layer]] += np.asarray(
                stats.retained_tiles, dtype=np.int64
            )
        evaluation = accumulator.anchor_evaluation[target]
        dropped = [
            value
            for stats in capture.anchor_stats_by_layer.values()
            for value in stats.dropped_mass
        ]
        prompt_mean = sum(dropped) / len(dropped)
        evaluation.eligible_tiles += (
            capture.eligible_tiles * len(dataset.anchors) * dataset.global_num_heads
        )
        evaluation.retained_tiles += sum(
            sum(stats.retained_tiles) for stats in capture.anchor_stats_by_layer.values()
        )
        evaluation.prompt_count += 1
        evaluation.prompt_mean_sum += prompt_mean
        evaluation.worst_prompt_mean = max(evaluation.worst_prompt_mean, prompt_mean)
        evaluation.violations += int(prompt_mean > max_anchor_dropped_mass)
        for layer, stats in capture.consumer_layers.items():
            if capture.prompt_id not in accumulator.prompt_mass[target]:
                accumulator.prompt_mass[target][capture.prompt_id] = np.empty(
                    (
                        len(dataset.consumer_layers),
                        dataset.global_num_heads,
                        dataset.global_num_heads,
                    ),
                    dtype=np.float64,
                )
            accumulator.prompt_mass[target][capture.prompt_id][consumer_index[layer]] = np.asarray(
                stats.dropped_mass, dtype=np.float64
            )

    def donor_options(
        accumulator: _SelectionAccumulator,
        *,
        target: float,
        layer: int,
        head: int,
        retained: np.ndarray,
        prompts: tuple[str, ...],
    ) -> tuple[_DonorOption, ...]:
        options = [
            _DonorOption(
                _Choice(0, True, accumulator.eligible_sum[target]),
                0,
                tuple(0.0 for _ in prompts),
            )
        ]
        layer_row = consumer_index[layer]
        for donor in range(dataset.global_num_heads):
            retained_tiles = int(retained[donor])
            options.append(
                _DonorOption(
                    _Choice(donor, False, retained_tiles),
                    accumulator.eligible_sum[target] - retained_tiles,
                    tuple(
                        float(accumulator.prompt_mass[target][prompt][layer_row, head, donor])
                        for prompt in prompts
                    ),
                )
            )
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

    def solve_target(
        bucket: Bucket,
        target: float,
        *,
        minimum_bmm1_skipped_tiles: int | None,
        maximize_bmm1_skipped_tiles: bool,
        target_met: bool,
    ) -> _Selection | None:
        accumulator = accumulators[bucket]
        prompts = dataset.prompts[(bucket, "calibration")]
        problem = pulp.LpProblem("compact_mask_reuse", pulp.LpMinimize)
        variables: dict[tuple[int, int, int], pulp.LpVariable] = {}
        options: dict[tuple[int, int, int], _DonorOption] = {}
        prompt_risk_terms: dict[str, list[object]] = defaultdict(list)
        for layer in dataset.consumer_layers:
            anchor = dataset.nearest[layer]
            retained = accumulator.retained_by_anchor[target][anchor_index[anchor]]
            for head in range(dataset.global_num_heads):
                menu = donor_options(
                    accumulator,
                    target=target,
                    layer=layer,
                    head=head,
                    retained=retained,
                    prompts=prompts,
                )
                choice_variables = []
                for option_index, option in enumerate(menu):
                    key = (layer, head, option_index)
                    variable = pulp.LpVariable(
                        f"choice_{layer}_{head}_{option_index}",
                        lowBound=0,
                        upBound=1,
                        cat="Binary",
                    )
                    variables[key] = variable
                    options[key] = option
                    choice_variables.append(variable)
                    for prompt, risk in zip(prompts, option.risk_by_prompt, strict=True):
                        prompt_risk_terms[prompt].append(risk * variable)
                problem += pulp.lpSum(choice_variables) == 1, f"choose_{layer}_{head}"

        bmm1_skipped = pulp.lpSum(
            options[key].bmm1_skipped_tiles * variable for key, variable in variables.items()
        )
        retained_reuse = pulp.lpSum(
            options[key].choice.retained_tiles * variable for key, variable in variables.items()
        )
        reuse_count = pulp.lpSum(
            int(not options[key].choice.fallback) * variable for key, variable in variables.items()
        )
        normalizer = len(dataset.nearest) * dataset.global_num_heads
        worst_prompt_risk = pulp.LpVariable("worst_prompt_reuse_dropped_mass", lowBound=0.0)
        for prompt_index, prompt in enumerate(prompts):
            problem += (
                pulp.lpSum(prompt_risk_terms[prompt]) <= normalizer * worst_prompt_risk,
                f"reuse_risk_{prompt_index}",
            )
        if minimum_bmm1_skipped_tiles is not None:
            problem += bmm1_skipped >= minimum_bmm1_skipped_tiles, "minimum_bmm1_skips"
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

        def minimize_and_fix(expression: object, name: str, *, integral: bool) -> bool:
            nonlocal problem
            if not minimize(expression):
                return False
            raw_value = pulp.value(expression)
            value = 0.0 if raw_value is None else float(raw_value)
            if integral:
                problem += expression == round(value), name
            else:
                problem += expression <= value + _SOLVER_LEXICOGRAPHIC_ATOL, name
            return True

        if maximize_bmm1_skipped_tiles and not minimize_and_fix(
            -bmm1_skipped, "fix_maximum_bmm1_skips", integral=True
        ):
            return None
        if not minimize_and_fix(worst_prompt_risk, "fix_worst_prompt_risk", integral=False):
            return None
        if not minimize_and_fix(retained_reuse, "fix_retained_reuse", integral=True):
            raise MaskReuseCalibrationError(
                "compact selector lost feasibility after fixing worst-prompt reuse risk"
            )
        donor_signature = pulp.lpSum(
            (options[key].choice.donor_head + 1) * variable
            for key, variable in variables.items()
            if not options[key].choice.fallback
        )
        donor_base = dataset.global_num_heads * len(dataset.targets) + 1
        if not minimize(reuse_count * donor_base + donor_signature):
            raise MaskReuseCalibrationError(
                "compact selector lost deterministic tie-break feasibility"
            )

        choices: dict[ConsumerHead, _Choice] = {}
        prompt_totals = dict.fromkeys(prompts, 0.0)
        worst_individual = 0.0
        skipped_tiles = 0
        retained_tiles = 0
        for key, variable in variables.items():
            if variable.value() <= 0.5:
                continue
            layer, head, _ = key
            option = options[key]
            choices[(layer, head)] = option.choice
            skipped_tiles += option.bmm1_skipped_tiles
            retained_tiles += option.choice.retained_tiles
            if not option.choice.fallback:
                for prompt, risk in zip(prompts, option.risk_by_prompt, strict=True):
                    prompt_totals[prompt] += risk
                    worst_individual = max(worst_individual, risk)
        prompt_risks = [prompt_totals[prompt] / normalizer for prompt in prompts]
        eligible_tiles = (
            accumulator.eligible_sum[target] * dataset.global_num_heads * len(dataset.nearest)
        )
        return _Selection(
            target,
            choices,
            (),
            bmm1_eligible_tiles=eligible_tiles,
            bmm1_skipped_tiles=skipped_tiles,
            target_bmm1_skip_ratio_met=target_met,
            worst_prompt_reuse_dropped_mass=max(prompt_risks, default=0.0),
            mean_prompt_reuse_dropped_mass=(
                sum(prompt_risks) / len(prompt_risks) if prompt_risks else 0.0
            ),
            worst_individual_reuse_dropped_mass=worst_individual,
        )

    selections: dict[Bucket, _Selection] = {}
    for bucket in dataset.buckets:
        accumulator = accumulators[bucket]
        frontier: list[dict[str, object]] = []
        candidates: list[tuple[tuple[object, ...], _Selection]] = []
        maximum_candidates: list[tuple[tuple[object, ...], _Selection]] = []
        for target in dataset.menus[bucket]:
            anchor_evaluation = accumulator.anchor_evaluation[target]
            eligible_tiles = (
                accumulator.eligible_sum[target] * dataset.global_num_heads * len(dataset.nearest)
            )
            required_tiles = math.ceil(target_bmm1_skip_ratio * eligible_tiles)
            selected = None
            if anchor_evaluation.violations == 0:
                selected = solve_target(
                    bucket,
                    target,
                    minimum_bmm1_skipped_tiles=required_tiles,
                    maximize_bmm1_skipped_tiles=False,
                    target_met=True,
                )
            fallback_count = (
                None
                if selected is None
                else sum(choice.fallback for choice in selected.choices.values())
            )
            retained_reuse = (
                None
                if selected is None
                else sum(choice.retained_tiles for choice in selected.choices.values())
            )
            combined_tile_cost = (
                None
                if retained_reuse is None
                else 2 * retained_reuse + anchor_evaluation.retained_tiles
            )
            frontier.append(
                {
                    "target_sparsity": target,
                    "anchor_safe": anchor_evaluation.violations == 0,
                    "target_bmm1_skip_ratio": target_bmm1_skip_ratio,
                    "target_bmm1_skip_ratio_feasible": selected is not None,
                    "retained_reuse_tiles": retained_reuse,
                    "retained_anchor_tiles": anchor_evaluation.retained_tiles,
                    "combined_tile_cost": combined_tile_cost,
                    "fallback_head_count": fallback_count,
                    "anchor_calibration": anchor_evaluation.to_mapping(),
                }
            )
            if selected is not None:
                rank = (
                    selected.worst_prompt_reuse_dropped_mass,
                    combined_tile_cost,
                    target,
                )
                candidates.append((rank, selected))
                frontier[-1].update(
                    {
                        "bmm1_skipped_tiles": selected.bmm1_skipped_tiles,
                        "achieved_bmm1_skip_ratio": (
                            selected.bmm1_skipped_tiles / selected.bmm1_eligible_tiles
                        ),
                        "worst_prompt_model_wide_reuse_dropped_mass": (
                            selected.worst_prompt_reuse_dropped_mass
                        ),
                    }
                )
            elif anchor_evaluation.violations == 0:
                maximum = solve_target(
                    bucket,
                    target,
                    minimum_bmm1_skipped_tiles=None,
                    maximize_bmm1_skipped_tiles=True,
                    target_met=False,
                )
                if maximum is not None:
                    maximum_candidates.append(
                        (
                            (
                                -maximum.bmm1_skipped_tiles,
                                maximum.worst_prompt_reuse_dropped_mass,
                                target,
                            ),
                            maximum,
                        )
                    )
                    frontier[-1].update(
                        {
                            "maximum_feasible_bmm1_skipped_tiles": (maximum.bmm1_skipped_tiles),
                            "maximum_feasible_bmm1_skip_ratio": (
                                maximum.bmm1_skipped_tiles / maximum.bmm1_eligible_tiles
                            ),
                        }
                    )
        if candidates:
            _, selected = min(candidates, key=lambda item: item[0])
            selections[bucket] = _Selection(
                selected.target_sparsity,
                selected.choices,
                tuple(frontier),
                bmm1_eligible_tiles=selected.bmm1_eligible_tiles,
                bmm1_skipped_tiles=selected.bmm1_skipped_tiles,
                target_bmm1_skip_ratio_met=True,
                worst_prompt_reuse_dropped_mass=selected.worst_prompt_reuse_dropped_mass,
                mean_prompt_reuse_dropped_mass=selected.mean_prompt_reuse_dropped_mass,
                worst_individual_reuse_dropped_mass=(selected.worst_individual_reuse_dropped_mass),
            )
        elif maximum_candidates:
            _, selected = min(maximum_candidates, key=lambda item: item[0])
            selections[bucket] = _Selection(
                selected.target_sparsity,
                selected.choices,
                tuple(frontier),
                bmm1_eligible_tiles=selected.bmm1_eligible_tiles,
                bmm1_skipped_tiles=selected.bmm1_skipped_tiles,
                target_bmm1_skip_ratio_met=False,
                worst_prompt_reuse_dropped_mass=selected.worst_prompt_reuse_dropped_mass,
                mean_prompt_reuse_dropped_mass=selected.mean_prompt_reuse_dropped_mass,
                worst_individual_reuse_dropped_mass=(selected.worst_individual_reuse_dropped_mass),
            )
        else:
            target = dataset.menus[bucket][0]
            dense_choices = {
                item: _Choice(0, True, accumulators[bucket].eligible_sum[target])
                for item in dataset.targets
            }
            selections[bucket] = _Selection(
                None,
                dense_choices,
                tuple(frontier),
                "no_target_sparsity_satisfied_anchor_calibration_constraint",
                bmm1_eligible_tiles=(
                    accumulator.eligible_sum[target]
                    * dataset.global_num_heads
                    * len(dataset.nearest)
                ),
            )
    return selections


def _evaluation_pass(
    source: CompactMaskReuseCaptureSource,
    dataset: _Dataset,
    selections: Mapping[Bucket, _Selection],
    *,
    max_anchor_dropped_mass: float,
    reuse_dropped_mass_report_threshold: float,
) -> tuple[
    dict[tuple[Bucket, str], _ReuseEvaluation],
    dict[tuple[Bucket, str], _AnchorEvaluation],
]:
    reuse = {(bucket, split): _ReuseEvaluation() for bucket in dataset.buckets for split in _SPLITS}
    anchor = {
        (bucket, split): _AnchorEvaluation() for bucket in dataset.buckets for split in _SPLITS
    }
    for capture in source:
        selection = selections[capture.bucket]
        evaluation_target = (
            dataset.menus[capture.bucket][0]
            if selection.target_sparsity is None
            else selection.target_sparsity
        )
        if capture.target_sparsity != evaluation_target:
            continue
        reuse_result = reuse[(capture.bucket, capture.split)]
        for layer, stats in capture.consumer_layers.items():
            anchor_stats = capture.anchor_stats_by_layer[stats.anchor_layer]
            for head in range(dataset.global_num_heads):
                choice = selection.choices[(layer, head)]
                reuse_result.eligible_tiles += capture.eligible_tiles
                if choice.fallback:
                    reuse_result.retained_tiles += capture.eligible_tiles
                    continue
                donor = choice.donor_head
                dropped = float(stats.dropped_mass[head][donor])
                reuse_result.retained_tiles += anchor_stats.retained_tiles[donor]
                reuse_result.sparse_observations += 1
                reuse_result.dropped_mass_sum += dropped
                reuse_result.worst_dropped_mass = max(reuse_result.worst_dropped_mass, dropped)
                reuse_result.violations += int(dropped > reuse_dropped_mass_report_threshold)

        anchor_result = anchor[(capture.bucket, capture.split)]
        dropped_values: list[float] = []
        retained_tiles = 0
        eligible_tiles = 0
        for anchor_stats in capture.anchor_stats_by_layer.values():
            eligible_tiles += capture.eligible_tiles * dataset.global_num_heads
            if selection.target_sparsity is None:
                retained_tiles += capture.eligible_tiles * dataset.global_num_heads
                dropped_values.extend([0.0] * dataset.global_num_heads)
            else:
                retained_tiles += sum(anchor_stats.retained_tiles)
                dropped_values.extend(anchor_stats.dropped_mass)
        prompt_mean = sum(dropped_values) / len(dropped_values)
        anchor_result.eligible_tiles += eligible_tiles
        anchor_result.retained_tiles += retained_tiles
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
    return reuse, anchor


def calibrate_compact_mask_reuse_policy(
    captures: CompactMaskReuseCaptureSource | str | Path,
    *,
    vanilla_calibration: Mapping[str, object],
    topology: Mapping[str, object],
    checkpoint_manifest: VerifiedCheckpointManifest,
    evidence: Mapping[str, object],
    max_anchor_dropped_mass: float,
    reuse_dropped_mass_report_threshold: float,
    target_bmm1_skip_ratio: float,
    source_provenance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Select a minimum-risk schema-v3 candidate under a per-bucket BMM1 target."""
    if not isinstance(checkpoint_manifest, VerifiedCheckpointManifest):
        raise MaskReuseCalibrationError(
            "checkpoint_manifest must be returned by verify_checkpoint_manifest"
        )
    source = (
        captures
        if isinstance(captures, CompactMaskReuseCaptureSource)
        else load_compact_mask_reuse_captures(captures)
    )
    threshold_scale_factor = canonical_prefill_threshold_scale_factor(vanilla_calibration)
    params = threshold_scale_factor["prefill"]
    assert isinstance(params, Mapping)
    if not {"min_observed_sparsity", "max_observed_sparsity"} <= params.keys():
        raise MaskReuseCalibrationError("compact schema-v3 export requires vanilla fit bounds")
    anchors, nearest = _normalize_topology(topology)
    dataset = _validate_dataset(
        source,
        fit=threshold_scale_factor,
        anchors=anchors,
        nearest=nearest,
    )
    checkpoint_identity = checkpoint_manifest.sha256
    if dataset.checkpoint_manifest_sha256 != checkpoint_identity:
        raise MaskReuseCalibrationError(
            "compact captures do not match the verified checkpoint manifest"
        )
    if dataset.model != checkpoint_manifest.model:
        raise MaskReuseCalibrationError(
            "compact capture model does not match the verified checkpoint manifest"
        )
    canonical_evidence = _canonical_evidence(evidence)
    input_sha256 = source.sha256()
    if canonical_evidence["reuse_bundle_sha256"] != input_sha256:
        raise MaskReuseCalibrationError(
            "evidence.reuse_bundle_sha256 does not match the compact capture file"
        )
    anchor_bound = _number(
        max_anchor_dropped_mass, "max_anchor_dropped_mass", minimum=0.0, maximum=1.0
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
    selections = _selection_pass(
        source,
        dataset,
        max_anchor_dropped_mass=anchor_bound,
        target_bmm1_skip_ratio=bmm1_target,
    )
    reuse_evaluations, anchor_evaluations = _evaluation_pass(
        source,
        dataset,
        selections,
        max_anchor_dropped_mass=anchor_bound,
        reuse_dropped_mass_report_threshold=reuse_report_threshold,
    )
    if source.sha256() != input_sha256:
        raise MaskReuseCalibrationError(
            "compact capture file changed during calibration; discard this result"
        )

    context_policies = []
    bucket_reports = []
    target_menus = []
    overall_reuse_calibration = _ReuseEvaluation()
    overall_reuse_heldout = _ReuseEvaluation()
    overall_anchor_calibration = _AnchorEvaluation()
    overall_anchor_heldout = _AnchorEvaluation()
    total_fallback = 0
    total_bmm1_eligible = 0
    total_bmm1_skipped = 0
    for bucket in dataset.buckets:
        selection = selections[bucket]
        if selection.target_sparsity is not None and bucket[1] is None:
            raise MaskReuseCalibrationError(
                "a deployment-qualified sparse context bucket requires a finite maximum"
            )
        reuse_calibration = reuse_evaluations[(bucket, "calibration")]
        reuse_heldout = reuse_evaluations[(bucket, "heldout")]
        anchor_calibration = anchor_evaluations[(bucket, "calibration")]
        anchor_heldout = anchor_evaluations[(bucket, "heldout")]
        overall_reuse_calibration.add(reuse_calibration)
        overall_reuse_heldout.add(reuse_heldout)
        overall_anchor_calibration.add(anchor_calibration)
        overall_anchor_heldout.add(anchor_heldout)
        policy: dict[str, object]
        if selection.target_sparsity is None:
            policy = {"min_kv_tokens": bucket[0], "max_kv_tokens": bucket[1], "exact": True}
            headmaps: dict[str, list[int]] = {}
            fallback_heads: dict[str, list[int]] = {}
        else:
            headmaps = {
                str(layer): [
                    selection.choices[(layer, head)].donor_head
                    for head in range(dataset.global_num_heads)
                ]
                for layer in dataset.consumer_layers
            }
            fallback_heads = {
                str(layer): [
                    head
                    for head in range(dataset.global_num_heads)
                    if selection.choices[(layer, head)].fallback
                ]
                for layer in dataset.consumer_layers
            }
            policy = {
                "min_kv_tokens": bucket[0],
                "max_kv_tokens": bucket[1],
                "target_sparsity": selection.target_sparsity,
                "headmaps": headmaps,
                "fallback_heads": fallback_heads,
            }
        context_policies.append(policy)
        fallback_count = sum(len(heads) for heads in fallback_heads.values())
        total_fallback += fallback_count
        total_bmm1_eligible += selection.bmm1_eligible_tiles
        total_bmm1_skipped += selection.bmm1_skipped_tiles
        target_menus.append(
            {
                "min_kv_tokens": bucket[0],
                "max_kv_tokens": bucket[1],
                "target_sparsities": [row["target_sparsity"] for row in selection.frontier],
            }
        )
        bucket_reports.append(
            {
                "min_kv_tokens": bucket[0],
                "max_kv_tokens": bucket[1],
                "selected_target_sparsity": selection.target_sparsity,
                "selection_status": (
                    "target_bmm1_skip_ratio_met"
                    if selection.target_sparsity is not None
                    and selection.target_bmm1_skip_ratio_met
                    else (
                        "target_bmm1_skip_ratio_unmet_maximum_feasible"
                        if selection.target_sparsity is not None
                        else selection.exact_reason
                    )
                ),
                "target_sparsity_frontier": list(selection.frontier),
                "fallback_head_count": fallback_count,
                "bmm1_skip_objective": {
                    "target": bmm1_target,
                    "target_met": selection.target_bmm1_skip_ratio_met,
                    "eligible_tiles": selection.bmm1_eligible_tiles,
                    "skipped_tiles": selection.bmm1_skipped_tiles,
                    "achieved": (
                        selection.bmm1_skipped_tiles / selection.bmm1_eligible_tiles
                        if selection.bmm1_eligible_tiles
                        else 0.0
                    ),
                },
                "reuse_selection_objective": {
                    "hard_maximum": None,
                    "worst_development_prompt_model_wide_dropped_mass": (
                        selection.worst_prompt_reuse_dropped_mass
                    ),
                    "mean_development_prompt_model_wide_dropped_mass": (
                        selection.mean_prompt_reuse_dropped_mass
                    ),
                    "worst_individual_dropped_mass": (
                        selection.worst_individual_reuse_dropped_mass
                    ),
                },
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

    candidate_cell_count = (
        dataset.capture_count
        * len(dataset.consumer_layers)
        * dataset.global_num_heads
        * dataset.global_num_heads
    )
    constraints = {
        "anchor": {
            "metric": "worst_prompt_mean_anchor_dropped_mass",
            "comparison": "<=",
            "maximum": anchor_bound,
        },
        "reuse": {
            "selection_metric": (
                "per_prompt_mean_across_all_attention_layers_and_heads_reuse_dropped_mass"
            ),
            "selection_hard_maximum": None,
            "report_metric": "per_prompt_candidate_reuse_dropped_mass",
            "report_threshold": reuse_report_threshold,
            "report_threshold_affects_selection": False,
        },
        "bmm1_skip_ratio": {
            "metric": "per_context_bucket_model_wide_eligible_bmm1_tile_skip_ratio",
            "comparison": ">=",
            "target": bmm1_target,
        },
    }
    provenance: dict[str, object] = {
        "calibrator": "modelopt.mask_reuse.compact_streaming",
        "compact_capture_schema_version": 1,
        "input_capture_count": dataset.capture_count,
        "candidate_cell_count": candidate_cell_count,
        "canonical_input_sha256": input_sha256,
        "calibration_prompt_ids": list(dataset.calibration_prompt_ids),
        "heldout_prompt_ids": list(dataset.heldout_prompt_ids),
        "prompt_sources": list(dataset.prompt_sources),
        "development_source_group_sha256": sorted(
            {
                str(row["source_group_sha256"])
                for row in dataset.prompt_sources
                if row["partition"] == "development"
            }
        ),
        "outer_test_source_group_sha256": sorted(
            {
                str(row["source_group_sha256"])
                for row in dataset.prompt_sources
                if row["partition"] == "outer_test"
            }
        ),
        "selection_split": "calibration",
        "evaluation_split": "heldout",
        "streaming_passes": ["validation", "calibration_selection", "frozen_evaluation"],
        "threshold_semantics": "a * exp(b * target_sparsity) / sample_length",
        "threshold_implementation": {
            "threshold_log2": "log2(a) + b * target_sparsity * log2(e) - log2(sample_length)",
            "threshold_lambda": "exp2(threshold_log2)",
            "validation": "exact IEEE-754 binary64 hex equality",
        },
        "checkpoint_manifest_sha256": checkpoint_identity,
        "target_sparsity_menus": target_menus,
        "constraints": constraints,
        "tie_breaks": [
            "reject target sparsities exceeding the calibration anchor bound",
            "meet the requested BMM1 skip ratio in every context bucket",
            "minimum worst-prompt model-wide reuse dropped mass",
            "minimum equal-BMM combined tile cost 2*A_R + A_A",
            "fewest reused consumer heads",
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
    denominator = len(dataset.targets) * len(dataset.buckets)
    report = {
        "model": dataset.model,
        "checkpoint_manifest_sha256": checkpoint_identity,
        "constraints": constraints,
        "selection_unit": "context_bucket",
        "by_bucket": bucket_reports,
        "overall": {
            "consumer_head_bucket_count": denominator,
            "fallback_head_bucket_count": total_fallback,
            "fallback_fraction": total_fallback / denominator,
            "bmm1_skip_objective": {
                "target_per_context_bucket": bmm1_target,
                "all_context_buckets_met": all(
                    selection.target_bmm1_skip_ratio_met for selection in selections.values()
                ),
                "eligible_tiles": total_bmm1_eligible,
                "skipped_tiles": total_bmm1_skipped,
                "achieved_model_wide": (
                    total_bmm1_skipped / total_bmm1_eligible if total_bmm1_eligible else 0.0
                ),
            },
            "reuse_calibration": overall_reuse_calibration.to_mapping(),
            "reuse_heldout": overall_reuse_heldout.to_mapping(),
            "anchor_calibration": overall_anchor_calibration.to_mapping(),
            "anchor_heldout": overall_anchor_heldout.to_mapping(),
        },
        "promotion": {
            "status": "candidate_only",
            "eligible": False,
            "reasons": [
                "grouped inner-fold modal and safety stability not evaluated",
                "preregistered outer gate with at least 99 independent groups per family cell not evaluated",
                "deployment rectangular-geometry promotion gate not evaluated",
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
        "model": dataset.model,
        "checkpoint_manifest_sha256": checkpoint_identity,
        "global_num_heads": dataset.global_num_heads,
        "target_bmm1_skip_ratio": bmm1_target,
        "anchors": list(dataset.anchors),
        "nearest": {str(layer): anchor for layer, anchor in dataset.nearest.items()},
        "deployment_geometry_validated": False,
        "deployment_geometry": dataset.deployment_geometry,
        "context_policies": context_policies,
        "provenance": provenance,
        "calibration_report": report,
    }
