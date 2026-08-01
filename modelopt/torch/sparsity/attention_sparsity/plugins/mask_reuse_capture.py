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

"""Host-side contract for ModelOpt-controlled mask-reuse capture.

The vLLM backend measures rank-local sufficient statistics.  This module owns
the trusted inputs, validates every echoed invocation, merges tensor-parallel
consumer-head shards, and emits the normalized rows consumed by
``calibrate_mask_reuse_policy``.  It never fabricates missing GPU statistics.
"""

from __future__ import annotations

import json
import math
import struct
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import cast

from modelopt.torch.sparsity.attention_sparsity.calibration.mask_reuse import (
    canonical_prefill_threshold_scale_factor,
)

__all__ = [
    "CAPTURE_SCHEMA_VERSION",
    "MAX_QUERY_CHUNK_TOKENS",
    "CaptureContractError",
    "MergedCapture",
    "PromptSpec",
    "build_capture_invocation",
    "canonical_json_sha256",
    "load_prompt_specs",
    "load_vanilla_prefill_fit",
    "merge_rank_captures",
    "parse_prompt_specs_jsonl",
    "parse_vanilla_prefill_fit",
    "source_capture_sha256",
    "validate_begin_acks",
    "validate_capture_statuses",
]

CAPTURE_SCHEMA_VERSION = 2
MAX_QUERY_CHUNK_TOKENS = 8192
_QUERY_START_ALIGNMENT = 128
_SPLITS = frozenset({"calibration", "heldout"})
_PARTITIONS = frozenset({"development", "outer_test"})
_PROMPT_FIELDS = frozenset(
    {
        "split",
        "partition",
        "inner_fold",
        "prompt_id",
        "source",
        "source_group_sha256",
        "prompt",
        "min_kv_tokens",
        "max_kv_tokens",
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
_STATUS_FIELDS = frozenset({"capture_schema_version", "available", "rank", "world_size", "reason"})
_ACK_FIELDS = frozenset(
    {
        "capture_schema_version",
        "armed",
        "rank",
        "world_size",
        "invocation_sha256",
    }
)
_RANK_CAPTURE_FIELDS = frozenset(
    {
        "capture_schema_version",
        "rank",
        "world_size",
        "invocation",
        "invocation_sha256",
        "geometry",
        "global_num_heads",
        "eligible_tiles",
        "anchor_stats_by_layer",
        "consumer_layers",
        "attention_call_counts",
        "tp_head_order_evidence",
        "dense_shadow_evidence",
    }
)
_ANCHOR_STATS_FIELDS = frozenset({"retained_tiles", "dropped_mass"})
_CONSUMER_STATS_FIELDS = frozenset({"anchor_layer", "consumer_head_start", "dropped_mass"})
_ATTENTION_CALL_COUNT_FIELDS = frozenset({"prefill", "decode"})
_TP_HEAD_ORDER_FIELDS = frozenset(
    {
        "sentinel_device_type",
        "gather_dim",
        "local_rank",
        "local_num_heads",
        "gathered_rank_local_head",
    }
)
_DENSE_SHADOW_FIELDS = frozenset({"enabled", "atol_hex", "rtol_hex", "validated_layer_indices"})


class CaptureContractError(ValueError):
    """Raised when capture inputs or backend evidence violate the contract."""


def _exact_fields(value: Mapping[str, object], expected: frozenset[str], label: str) -> None:
    missing = expected - value.keys()
    extra = value.keys() - expected
    if missing or extra:
        raise CaptureContractError(
            f"{label} fields do not match the schema; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CaptureContractError(f"{label} must be an integer >= {minimum}")
    return value


def _finite_number(
    value: object,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CaptureContractError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise CaptureContractError(f"{label} must be a finite number")
    if minimum is not None and result < minimum:
        raise CaptureContractError(f"{label} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise CaptureContractError(f"{label} must be <= {maximum}")
    return result


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CaptureContractError(f"{label} must be a non-empty string")
    result = value.strip()
    if unicodedata.normalize("NFC", result) != result or any(
        ord(character) < 32 for character in result
    ):
        raise CaptureContractError(f"{label} must be NFC text without control characters")
    return result


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CaptureContractError(f"{label} must be a lowercase SHA256")
    return value


def _canonical_float_hex(value: object, label: str) -> float:
    if not isinstance(value, str):
        raise CaptureContractError(f"{label} must be a canonical float.hex string")
    try:
        parsed = float.fromhex(value)
    except ValueError as error:
        raise CaptureContractError(f"{label} must be a canonical float.hex string") from error
    if not math.isfinite(parsed) or parsed.hex() != value:
        raise CaptureContractError(f"{label} must be a canonical finite float.hex string")
    return parsed


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CaptureContractError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def canonical_json_sha256(value: object) -> str:
    """Hash the canonical JSON encoding used by capture RPC acknowledgements."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class PromptSpec:
    """One preregistered prompt/context capture unit."""

    split: str
    partition: str
    inner_fold: int | None
    prompt_id: str
    source: str
    source_group_sha256: str
    prompt: str
    min_kv_tokens: int
    max_kv_tokens: int | None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> PromptSpec:
        """Parse one strict prompt-plan object."""
        _exact_fields(raw, _PROMPT_FIELDS, "prompt")
        split = raw["split"]
        if split not in _SPLITS:
            raise CaptureContractError(f"prompt.split must be one of {sorted(_SPLITS)}")
        partition = raw["partition"]
        if partition not in _PARTITIONS:
            raise CaptureContractError(f"prompt.partition must be one of {sorted(_PARTITIONS)}")
        expected_split = "calibration" if partition == "development" else "heldout"
        if split != expected_split:
            raise CaptureContractError("prompt.split and prompt.partition disagree")
        raw_fold = raw["inner_fold"]
        if partition == "development":
            inner_fold = _integer(raw_fold, "prompt.inner_fold")
        elif raw_fold is not None:
            raise CaptureContractError("outer_test prompts must have null inner_fold")
        else:
            inner_fold = None
        prompt = raw["prompt"]
        if not isinstance(prompt, str) or not prompt:
            raise CaptureContractError("prompt.prompt must be a non-empty string")
        minimum = _integer(raw["min_kv_tokens"], "prompt.min_kv_tokens", minimum=1)
        maximum = raw["max_kv_tokens"]
        if maximum is not None:
            maximum = _integer(maximum, "prompt.max_kv_tokens", minimum=minimum)
        return cls(
            split=split,
            partition=partition,
            inner_fold=inner_fold,
            prompt_id=_text(raw["prompt_id"], "prompt.prompt_id"),
            source=_text(raw["source"], "prompt.source"),
            source_group_sha256=_sha256(raw["source_group_sha256"], "prompt.source_group_sha256"),
            prompt=prompt,
            min_kv_tokens=minimum,
            max_kv_tokens=maximum,
        )

    @property
    def bucket(self) -> tuple[int, int | None]:
        """Return the calibrated context-bucket bounds."""
        return self.min_kv_tokens, self.max_kv_tokens


def parse_prompt_specs_jsonl(payload: bytes) -> list[PromptSpec]:
    """Parse exact strict-JSONL prompt bytes and reject split leakage."""
    if not isinstance(payload, bytes):
        raise CaptureContractError("prompt file payload must be bytes")
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise CaptureContractError("prompt file must be UTF-8") from error
    prompts: list[PromptSpec] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line, object_pairs_hook=_reject_duplicate_json_keys)
        except json.JSONDecodeError as error:
            raise CaptureContractError(f"line {line_number}: invalid JSON: {error.msg}") from error
        except CaptureContractError as error:
            raise CaptureContractError(f"line {line_number}: {error}") from error
        if not isinstance(raw, dict):
            raise CaptureContractError(f"line {line_number}: prompt must be an object")
        try:
            prompts.append(PromptSpec.from_mapping(raw))
        except CaptureContractError as error:
            raise CaptureContractError(f"line {line_number}: {error}") from error
    if not prompts:
        raise CaptureContractError("prompt file contains no prompts")

    by_split = {split: [prompt for prompt in prompts if prompt.split == split] for split in _SPLITS}
    if any(not values for values in by_split.values()):
        raise CaptureContractError("prompt file requires calibration and heldout splits")
    calibration_ids = {prompt.prompt_id for prompt in by_split["calibration"]}
    heldout_ids = {prompt.prompt_id for prompt in by_split["heldout"]}
    for split, values in by_split.items():
        identifiers = [prompt.prompt_id for prompt in values]
        if len(identifiers) != len(set(identifiers)):
            raise CaptureContractError(f"prompt IDs must be unique within the {split} split")
    if calibration_ids & heldout_ids:
        raise CaptureContractError("prompt IDs overlap calibration and heldout splits")
    group_assignments: dict[str, tuple[str, int | None]] = {}
    for prompt in prompts:
        assignment = (prompt.partition, prompt.inner_fold)
        previous = group_assignments.setdefault(prompt.source_group_sha256, assignment)
        if previous != assignment:
            raise CaptureContractError(
                "one source group is assigned to multiple partitions or inner folds"
            )
    split_buckets = {
        split: {prompt.bucket for prompt in values} for split, values in by_split.items()
    }
    if split_buckets["calibration"] != split_buckets["heldout"]:
        raise CaptureContractError("heldout context buckets must match calibration buckets")
    keys = [(prompt.split, prompt.prompt_id, prompt.bucket) for prompt in prompts]
    if len(keys) != len(set(keys)):
        raise CaptureContractError("prompt file repeats a split/prompt/context capture")
    return sorted(
        prompts,
        key=lambda prompt: (
            prompt.min_kv_tokens,
            math.inf if prompt.max_kv_tokens is None else prompt.max_kv_tokens,
            prompt.split,
            prompt.prompt_id,
        ),
    )


def load_prompt_specs(path: str | Path) -> list[PromptSpec]:
    """Load strict JSONL prompt specifications from a path."""
    return parse_prompt_specs_jsonl(Path(path).read_bytes())


def _parse_strict_json_object(payload: bytes, label: str) -> Mapping[str, object]:
    try:
        raw = json.loads(payload, object_pairs_hook=_reject_duplicate_json_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        detail = error.msg if isinstance(error, json.JSONDecodeError) else str(error)
        raise CaptureContractError(f"{label} is invalid JSON: {detail}") from error
    if not isinstance(raw, dict):
        raise CaptureContractError(f"{label} must be a JSON object")
    return raw


def parse_vanilla_prefill_fit(payload: bytes) -> dict[str, object]:
    """Parse and canonicalize exact vanilla-calibration JSON bytes."""
    return canonical_prefill_threshold_scale_factor(
        _parse_strict_json_object(payload, "vanilla calibration")
    )


def load_vanilla_prefill_fit(path: str | Path) -> dict[str, object]:
    """Load and canonicalize an existing ModelOpt vanilla skip-softmax fit."""
    return parse_vanilla_prefill_fit(Path(path).read_bytes())


def source_capture_sha256(prompt_token_ids: Sequence[int]) -> str:
    """Fingerprint exact token IDs independently of prompt labels and splits."""
    if not prompt_token_ids:
        raise CaptureContractError("prompt token IDs must not be empty")
    digest = sha256(b"modelopt-mask-reuse-token-ids-v1\0")
    digest.update(struct.pack("<Q", len(prompt_token_ids)))
    for index, token_id in enumerate(prompt_token_ids):
        token = _integer(token_id, f"prompt_token_ids[{index}]", minimum=0)
        if token >= 2**64:
            raise CaptureContractError(f"prompt_token_ids[{index}] exceeds uint64")
        digest.update(struct.pack("<Q", token))
    return digest.hexdigest()


def _expected_final_geometry(sample_length: int) -> dict[str, int]:
    q_start = ((sample_length - 1) // MAX_QUERY_CHUNK_TOKENS) * MAX_QUERY_CHUNK_TOKENS
    q_tokens = sample_length - q_start
    if q_tokens <= _QUERY_START_ALIGNMENT:
        raise CaptureContractError(
            "the deployed q-stage-2 contract requires a final prefill chunk of at least "
            "129 tokens; adjust the prompt length"
        )
    return {
        "q_tokens": q_tokens,
        "kv_tokens": sample_length,
        "q_start_tokens": q_start,
    }


def build_capture_invocation(
    *,
    model: str,
    checkpoint_manifest_sha256: str,
    prompt: PromptSpec,
    prompt_token_ids: Sequence[int],
    target_sparsity: float,
    threshold_scale_factor: Mapping[str, object],
) -> dict[str, object]:
    """Build the exact invocation the backend must echo before evidence is trusted."""
    model = _text(model, "model")
    checkpoint_identity = _sha256(checkpoint_manifest_sha256, "checkpoint_manifest_sha256")
    sample_length = len(prompt_token_ids)
    if sample_length < prompt.min_kv_tokens or (
        prompt.max_kv_tokens is not None and sample_length > prompt.max_kv_tokens
    ):
        raise CaptureContractError(
            f"prompt {prompt.prompt_id!r} token length {sample_length} lies outside "
            f"bucket [{prompt.min_kv_tokens}, {prompt.max_kv_tokens}]"
        )
    target = _finite_number(target_sparsity, "target_sparsity", minimum=0.0, maximum=1.0)
    if not 0.0 < target < 1.0:
        raise CaptureContractError("target_sparsity must be in (0, 1)")

    if set(threshold_scale_factor) != {"formula", "prefill"}:
        raise CaptureContractError("threshold_scale_factor must be canonical ModelOpt metadata")
    params = threshold_scale_factor["prefill"]
    if not isinstance(params, Mapping):
        raise CaptureContractError("threshold_scale_factor.prefill must be an object")
    lower = params.get("min_observed_sparsity")
    upper = params.get("max_observed_sparsity")
    if (lower is not None and target < float(lower)) or (
        upper is not None and target > float(upper)
    ):
        raise CaptureContractError(
            f"target_sparsity={target} is outside the observed vanilla calibration range"
        )
    a = float(params["a"])
    b = float(params["b"])
    threshold_log2 = math.log2(a) + b * target * math.log2(math.e) - math.log2(sample_length)
    # ``math.exp2`` was added in Python 3.11; ModelOpt still supports 3.10.
    threshold_lambda = 2.0**threshold_log2
    if not math.isfinite(threshold_log2) or not 0.0 < threshold_lambda < 1.0:
        raise CaptureContractError("vanilla fit derives a threshold outside (0, 1)")

    invocation = {
        "capture_schema_version": CAPTURE_SCHEMA_VERSION,
        "model": model,
        "checkpoint_manifest_sha256": checkpoint_identity,
        "split": prompt.split,
        "partition": prompt.partition,
        "inner_fold": prompt.inner_fold,
        "prompt_id": prompt.prompt_id,
        "source": prompt.source,
        "source_group_sha256": prompt.source_group_sha256,
        "source_capture_sha256": source_capture_sha256(prompt_token_ids),
        "min_kv_tokens": prompt.min_kv_tokens,
        "max_kv_tokens": prompt.max_kv_tokens,
        "target_sparsity_hex": target.hex(),
        "sample_length": sample_length,
        "threshold_log2_hex": threshold_log2.hex(),
        "threshold_lambda_hex": threshold_lambda.hex(),
        "expected_geometry": _expected_final_geometry(sample_length),
    }
    _validate_invocation(invocation)
    return invocation


def _validate_geometry(raw: object, label: str) -> dict[str, int]:
    if not isinstance(raw, Mapping):
        raise CaptureContractError(f"{label} must be an object")
    _exact_fields(raw, _GEOMETRY_FIELDS, label)
    q_tokens = _integer(raw["q_tokens"], f"{label}.q_tokens", minimum=129)
    if q_tokens > MAX_QUERY_CHUNK_TOKENS:
        raise CaptureContractError(f"{label}.q_tokens exceeds {MAX_QUERY_CHUNK_TOKENS}")
    kv_tokens = _integer(raw["kv_tokens"], f"{label}.kv_tokens", minimum=q_tokens)
    q_start = _integer(raw["q_start_tokens"], f"{label}.q_start_tokens")
    if q_start % _QUERY_START_ALIGNMENT or q_start + q_tokens != kv_tokens:
        raise CaptureContractError(f"{label} is not a 128-token-aligned final chunk")
    return {"q_tokens": q_tokens, "kv_tokens": kv_tokens, "q_start_tokens": q_start}


def _eligible_tiles(geometry: Mapping[str, int]) -> int:
    q_blocks = (geometry["q_tokens"] + 127) // 128
    first_eligible = geometry["q_start_tokens"] // 128 + 1
    return q_blocks * (2 * first_eligible + q_blocks - 1) // 2


def _validate_invocation(raw: object) -> dict[str, object]:
    if not isinstance(raw, Mapping):
        raise CaptureContractError("capture invocation must be an object")
    _exact_fields(raw, _INVOCATION_FIELDS, "capture invocation")
    if raw["capture_schema_version"] != CAPTURE_SCHEMA_VERSION:
        raise CaptureContractError("capture invocation has an unsupported schema version")
    _text(raw["model"], "capture invocation.model")
    _sha256(
        raw["checkpoint_manifest_sha256"],
        "capture invocation.checkpoint_manifest_sha256",
    )
    if raw["split"] not in _SPLITS:
        raise CaptureContractError("capture invocation.split is invalid")
    partition = raw["partition"]
    if partition not in _PARTITIONS:
        raise CaptureContractError("capture invocation.partition is invalid")
    expected_split = "calibration" if partition == "development" else "heldout"
    if raw["split"] != expected_split:
        raise CaptureContractError("capture invocation split and partition disagree")
    if partition == "development":
        _integer(raw["inner_fold"], "capture invocation.inner_fold")
    elif raw["inner_fold"] is not None:
        raise CaptureContractError("outer_test capture invocation must have null inner_fold")
    _text(raw["prompt_id"], "capture invocation.prompt_id")
    _text(raw["source"], "capture invocation.source")
    _sha256(raw["source_group_sha256"], "capture invocation.source_group_sha256")
    _sha256(raw["source_capture_sha256"], "capture invocation.source_capture_sha256")
    minimum = _integer(raw["min_kv_tokens"], "capture invocation.min_kv_tokens", minimum=1)
    maximum = raw["max_kv_tokens"]
    if maximum is not None:
        _integer(maximum, "capture invocation.max_kv_tokens", minimum=minimum)
    target = _canonical_float_hex(
        raw["target_sparsity_hex"], "capture invocation.target_sparsity_hex"
    )
    if not 0.0 < target < 1.0:
        raise CaptureContractError("capture invocation target sparsity must be in (0, 1)")
    sample_length = _integer(raw["sample_length"], "capture invocation.sample_length", minimum=1)
    threshold_log2 = _canonical_float_hex(
        raw["threshold_log2_hex"], "capture invocation.threshold_log2_hex"
    )
    threshold_lambda = _canonical_float_hex(
        raw["threshold_lambda_hex"], "capture invocation.threshold_lambda_hex"
    )
    if threshold_log2 >= 0.0 or not 0.0 < threshold_lambda < 1.0:
        raise CaptureContractError("capture invocation threshold must be in (0, 1)")
    if (2.0**threshold_log2).hex() != threshold_lambda.hex():
        raise CaptureContractError("capture invocation threshold hex fields disagree")
    geometry = _validate_geometry(raw["expected_geometry"], "expected_geometry")
    if geometry["kv_tokens"] != sample_length:
        raise CaptureContractError("expected geometry does not match sample_length")
    return dict(raw)


def _validate_rank_envelope(
    values: Sequence[Mapping[str, object]], expected_fields: frozenset[str], label: str
) -> tuple[int, list[Mapping[str, object]]]:
    if not values:
        raise CaptureContractError(f"{label} returned no rank payloads")
    world_sizes: set[int] = set()
    ranks: list[int] = []
    for index, value in enumerate(values):
        if not isinstance(value, Mapping):
            raise CaptureContractError(f"{label}[{index}] must be an object")
        _exact_fields(value, expected_fields, f"{label}[{index}]")
        if value["capture_schema_version"] != CAPTURE_SCHEMA_VERSION:
            raise CaptureContractError(f"{label}[{index}] has an unsupported schema version")
        ranks.append(_integer(value["rank"], f"{label}[{index}].rank"))
        world_sizes.add(_integer(value["world_size"], f"{label}[{index}].world_size", minimum=1))
    if len(world_sizes) != 1:
        raise CaptureContractError(f"{label} disagrees on world_size")
    world_size = next(iter(world_sizes))
    if len(values) != world_size or sorted(ranks) != list(range(world_size)):
        raise CaptureContractError(f"{label} does not exactly cover ranks [0, {world_size})")
    return world_size, sorted(values, key=lambda value: cast("int", value["rank"]))


def validate_capture_statuses(values: Sequence[Mapping[str, object]]) -> int:
    """Require the env-gated backend capture sink on every worker rank."""
    world_size, ordered = _validate_rank_envelope(values, _STATUS_FIELDS, "capture status")
    failures = []
    for value in ordered:
        if not isinstance(value["available"], bool):
            raise CaptureContractError("capture status.available must be boolean")
        reason = value["reason"]
        if reason is not None and not isinstance(reason, str):
            raise CaptureContractError("capture status.reason must be a string or null")
        if not value["available"]:
            failures.append(f"rank {value['rank']}: {reason or 'unavailable'}")
    if failures:
        raise CaptureContractError("mask-reuse capture is unavailable: " + "; ".join(failures))
    return world_size


def validate_begin_acks(
    values: Sequence[Mapping[str, object]], invocation: Mapping[str, object]
) -> int:
    """Validate that every rank armed the exact same invocation before generation."""
    _validate_invocation(invocation)
    expected_digest = canonical_json_sha256(invocation)
    world_size, ordered = _validate_rank_envelope(values, _ACK_FIELDS, "capture begin")
    for value in ordered:
        if value["armed"] is not True:
            raise CaptureContractError(f"capture begin rank {value['rank']} did not arm")
        if value["invocation_sha256"] != expected_digest:
            raise CaptureContractError(
                f"capture begin rank {value['rank']} acknowledged the wrong invocation"
            )
    return world_size


def _parse_anchor_stats(
    raw: object, *, global_num_heads: int, eligible_tiles: int
) -> dict[str, dict[str, list[int] | list[float]]]:
    if not isinstance(raw, Mapping) or not raw:
        raise CaptureContractError("anchor_stats_by_layer must be a non-empty object")
    result: dict[str, dict[str, list[int] | list[float]]] = {}
    for raw_layer, value in raw.items():
        if not isinstance(raw_layer, str):
            raise CaptureContractError("anchor layer keys must be canonical integer strings")
        layer = _integer(int(raw_layer), f"anchor layer {raw_layer}") if raw_layer.isdigit() else -1
        if layer < 0 or raw_layer != str(layer):
            raise CaptureContractError("anchor layer keys must be canonical integer strings")
        if not isinstance(value, Mapping):
            raise CaptureContractError(f"anchor_stats_by_layer[{layer}] must be an object")
        _exact_fields(value, _ANCHOR_STATS_FIELDS, f"anchor_stats_by_layer[{layer}]")
        retained_raw = value["retained_tiles"]
        dropped_raw = value["dropped_mass"]
        if not isinstance(retained_raw, list) or not isinstance(dropped_raw, list):
            raise CaptureContractError(f"anchor_stats_by_layer[{layer}] arrays must be lists")
        if len(retained_raw) != global_num_heads or len(dropped_raw) != global_num_heads:
            raise CaptureContractError(
                f"anchor_stats_by_layer[{layer}] must cover {global_num_heads} heads"
            )
        retained = [
            _integer(value, f"anchor_stats_by_layer[{layer}].retained_tiles[{head}]")
            for head, value in enumerate(retained_raw)
        ]
        if any(value > eligible_tiles for value in retained):
            raise CaptureContractError(
                f"anchor_stats_by_layer[{layer}] retained tiles exceed eligible tiles"
            )
        dropped = [
            _finite_number(
                value,
                f"anchor_stats_by_layer[{layer}].dropped_mass[{head}]",
                minimum=0.0,
                maximum=1.0,
            )
            for head, value in enumerate(dropped_raw)
        ]
        result[raw_layer] = {"retained_tiles": retained, "dropped_mass": dropped}
    return dict(sorted(result.items(), key=lambda item: int(item[0])))


def _parse_consumer_layers(
    raw: object, *, rank: int, global_num_heads: int
) -> dict[int, tuple[int, int, list[list[float]]]]:
    if not isinstance(raw, Mapping) or not raw:
        raise CaptureContractError(f"rank {rank} consumer_layers must be a non-empty object")
    result: dict[int, tuple[int, int, list[list[float]]]] = {}
    for raw_layer, value in raw.items():
        if (
            not isinstance(raw_layer, str)
            or not raw_layer.isdigit()
            or raw_layer != str(int(raw_layer))
        ):
            raise CaptureContractError("consumer layer keys must be canonical integer strings")
        layer = int(raw_layer)
        if not isinstance(value, Mapping):
            raise CaptureContractError(f"consumer_layers[{layer}] must be an object")
        _exact_fields(value, _CONSUMER_STATS_FIELDS, f"consumer_layers[{layer}]")
        anchor = _integer(value["anchor_layer"], f"consumer_layers[{layer}].anchor_layer")
        if anchor >= layer:
            raise CaptureContractError(f"consumer layer {layer} must follow anchor {anchor}")
        start = _integer(
            value["consumer_head_start"], f"consumer_layers[{layer}].consumer_head_start"
        )
        matrix = value["dropped_mass"]
        if not isinstance(matrix, list) or not matrix:
            raise CaptureContractError(f"consumer_layers[{layer}].dropped_mass must be non-empty")
        rows: list[list[float]] = []
        for local_head, raw_row in enumerate(matrix):
            if not isinstance(raw_row, list) or len(raw_row) != global_num_heads:
                raise CaptureContractError(
                    f"consumer_layers[{layer}].dropped_mass[{local_head}] must cover "
                    f"{global_num_heads} donor heads"
                )
            rows.append(
                [
                    _finite_number(
                        value,
                        f"consumer_layers[{layer}].dropped_mass[{local_head}][{donor}]",
                        minimum=0.0,
                        maximum=1.0,
                    )
                    for donor, value in enumerate(raw_row)
                ]
            )
        if start + len(rows) > global_num_heads:
            raise CaptureContractError(
                f"rank {rank} consumer layer {layer} shard exceeds head count"
            )
        result[layer] = (anchor, start, rows)
    return result


def _parse_attention_call_counts(raw: object, *, rank: int) -> dict[str, int]:
    if not isinstance(raw, Mapping):
        raise CaptureContractError(f"rank {rank} attention_call_counts must be an object")
    _exact_fields(raw, _ATTENTION_CALL_COUNT_FIELDS, f"rank {rank} attention_call_counts")
    return {
        name: _integer(raw[name], f"rank {rank} attention_call_counts.{name}")
        for name in ("prefill", "decode")
    }


def _parse_tp_head_order_evidence(
    raw: object,
    *,
    rank: int,
    world_size: int,
    global_num_heads: int,
) -> dict[str, object]:
    if not isinstance(raw, Mapping):
        raise CaptureContractError(f"rank {rank} tp_head_order_evidence must be an object")
    _exact_fields(raw, _TP_HEAD_ORDER_FIELDS, f"rank {rank} tp_head_order_evidence")
    if raw["sentinel_device_type"] != "cuda":
        raise CaptureContractError(f"rank {rank} TP sentinel was not gathered on CUDA")
    if raw["gather_dim"] != 0:
        raise CaptureContractError(f"rank {rank} TP sentinel must be gathered along dim 0")
    if _integer(raw["local_rank"], f"rank {rank} TP local_rank") != rank:
        raise CaptureContractError(f"rank {rank} TP sentinel reports a different local rank")
    local_num_heads = _integer(raw["local_num_heads"], f"rank {rank} TP local_num_heads", minimum=1)
    if local_num_heads * world_size != global_num_heads:
        raise CaptureContractError(
            f"rank {rank} local head count does not evenly cover global heads"
        )
    gathered = raw["gathered_rank_local_head"]
    expected = [
        [global_head // local_num_heads, global_head % local_num_heads]
        for global_head in range(global_num_heads)
    ]
    if gathered != expected:
        raise CaptureContractError(
            f"rank {rank} TP all-gather is not rank-major in global-head order"
        )
    return {
        "rank": rank,
        "global_head_start": rank * local_num_heads,
        "local_num_heads": local_num_heads,
        "sentinel_device_type": "cuda",
        "gather_dim": 0,
        "gathered_rank_local_head": expected,
    }


def _parse_dense_shadow_evidence(raw: object, *, rank: int) -> dict[str, object]:
    if not isinstance(raw, Mapping):
        raise CaptureContractError(f"rank {rank} dense_shadow_evidence must be an object")
    _exact_fields(raw, _DENSE_SHADOW_FIELDS, f"rank {rank} dense_shadow_evidence")
    enabled = raw["enabled"]
    if not isinstance(enabled, bool):
        raise CaptureContractError(f"rank {rank} dense_shadow_evidence.enabled must be boolean")
    atol = _canonical_float_hex(raw["atol_hex"], f"rank {rank} dense shadow atol")
    rtol = _canonical_float_hex(raw["rtol_hex"], f"rank {rank} dense shadow rtol")
    if atol != 0.0 or rtol != 0.0:
        raise CaptureContractError("dense shadow evidence must use bitwise zero tolerances")
    raw_layers = raw["validated_layer_indices"]
    if not isinstance(raw_layers, list):
        raise CaptureContractError(
            f"rank {rank} dense_shadow_evidence.validated_layer_indices must be a list"
        )
    layers = [_integer(layer, f"rank {rank} dense shadow layer") for layer in raw_layers]
    if layers != sorted(set(layers)):
        raise CaptureContractError(f"rank {rank} dense shadow layers must be sorted and unique")
    if not enabled and layers:
        raise CaptureContractError(
            f"rank {rank} disabled dense shadow cannot report validated layers"
        )
    return {
        "enabled": enabled,
        "atol_hex": atol.hex(),
        "rtol_hex": rtol.hex(),
        "validated_layer_indices": layers,
    }


@dataclass(frozen=True, slots=True)
class MergedCapture:
    """One compact normalized capture plus auditable metadata."""

    capture: dict[str, object]
    manifest: dict[str, object]


def merge_rank_captures(
    values: Sequence[Mapping[str, object]], invocation: Mapping[str, object]
) -> MergedCapture:
    """Merge raw per-rank sufficient statistics without inventing absent rows."""
    trusted_invocation = _validate_invocation(invocation)
    expected_digest = canonical_json_sha256(trusted_invocation)
    world_size, ordered = _validate_rank_envelope(values, _RANK_CAPTURE_FIELDS, "capture drain")
    global_heads: set[int] = set()
    eligible_values: set[int] = set()
    geometry_values: list[dict[str, int]] = []
    anchor_payloads: list[dict[str, dict[str, list[int] | list[float]]]] = []
    consumer_payloads: list[dict[int, tuple[int, int, list[list[float]]]]] = []
    attention_call_counts: list[dict[str, int]] = []
    tp_head_order_evidence: list[dict[str, object]] = []
    dense_shadow_evidence: list[dict[str, object]] = []
    for value in ordered:
        rank = cast("int", value["rank"])
        if value["invocation"] != trusted_invocation:
            raise CaptureContractError(f"capture drain rank {rank} echoed the wrong invocation")
        if value["invocation_sha256"] != expected_digest:
            raise CaptureContractError(f"capture drain rank {rank} has the wrong invocation digest")
        geometry = _validate_geometry(value["geometry"], f"capture drain rank {rank} geometry")
        if geometry != trusted_invocation["expected_geometry"]:
            raise CaptureContractError(f"capture drain rank {rank} measured the wrong final chunk")
        geometry_values.append(geometry)
        global_num_heads = _integer(
            value["global_num_heads"], f"capture drain rank {rank} global_num_heads", minimum=1
        )
        eligible_tiles = _integer(
            value["eligible_tiles"], f"capture drain rank {rank} eligible_tiles", minimum=1
        )
        global_heads.add(global_num_heads)
        eligible_values.add(eligible_tiles)
        anchor_payloads.append(
            _parse_anchor_stats(
                value["anchor_stats_by_layer"],
                global_num_heads=global_num_heads,
                eligible_tiles=eligible_tiles,
            )
        )
        consumers = _parse_consumer_layers(
            value["consumer_layers"], rank=rank, global_num_heads=global_num_heads
        )
        consumer_payloads.append(consumers)
        counts = _parse_attention_call_counts(value["attention_call_counts"], rank=rank)
        attention_call_counts.append(counts)
        tp_evidence = _parse_tp_head_order_evidence(
            value["tp_head_order_evidence"],
            rank=rank,
            world_size=world_size,
            global_num_heads=global_num_heads,
        )
        tp_head_order_evidence.append(tp_evidence)
        dense_shadow_evidence.append(
            _parse_dense_shadow_evidence(value["dense_shadow_evidence"], rank=rank)
        )
        for layer, (_, start, rows) in consumers.items():
            if (
                start != tp_evidence["global_head_start"]
                or len(rows) != tp_evidence["local_num_heads"]
            ):
                raise CaptureContractError(
                    f"rank {rank} consumer layer {layer} does not match its TP head shard"
                )
    if len(global_heads) != 1 or len(eligible_values) != 1:
        raise CaptureContractError("capture ranks disagree on head count or eligible tiles")
    if any(value != geometry_values[0] for value in geometry_values[1:]):
        raise CaptureContractError("capture ranks disagree on final-chunk geometry")
    if any(value != anchor_payloads[0] for value in anchor_payloads[1:]):
        raise CaptureContractError("capture ranks disagree on global anchor statistics")
    if any(value != attention_call_counts[0] for value in attention_call_counts[1:]):
        raise CaptureContractError("capture ranks disagree on attention call counts")
    if any(value != dense_shadow_evidence[0] for value in dense_shadow_evidence[1:]):
        raise CaptureContractError("capture ranks disagree on dense shadow evidence")
    layer_sets = [set(value) for value in consumer_payloads]
    if any(value != layer_sets[0] for value in layer_sets[1:]):
        raise CaptureContractError("capture ranks disagree on consumer layer coverage")

    global_num_heads = next(iter(global_heads))
    eligible_tiles = next(iter(eligible_values))
    if eligible_tiles != _eligible_tiles(geometry_values[0]):
        raise CaptureContractError(
            "capture eligible_tiles does not match 128x128 bottom-right causal geometry"
        )
    anchors = anchor_payloads[0]
    attention_layers = sorted({int(layer) for layer in anchors} | layer_sets[0])
    expected_prefill_calls = (
        (cast("int", trusted_invocation["sample_length"]) + MAX_QUERY_CHUNK_TOKENS - 1)
        // MAX_QUERY_CHUNK_TOKENS
    ) * len(attention_layers)
    common_counts = attention_call_counts[0]
    if common_counts["prefill"] != expected_prefill_calls:
        raise CaptureContractError(
            "capture prefill attention call count does not match chunks times layers"
        )
    if common_counts["decode"] != 0:
        raise CaptureContractError("capture observed a decode attention call at max_tokens=1")
    common_shadow = dense_shadow_evidence[0]
    if common_shadow["enabled"] and common_shadow["validated_layer_indices"] != attention_layers:
        raise CaptureContractError(
            "dense shadow evidence does not cover every captured attention layer"
        )
    merged_consumers: dict[str, dict[str, object]] = {}
    for layer in sorted(layer_sets[0]):
        shards = sorted(
            (payload[layer] for payload in consumer_payloads), key=lambda value: value[1]
        )
        anchor_layers = {shard[0] for shard in shards}
        if len(anchor_layers) != 1:
            raise CaptureContractError(f"capture ranks disagree on consumer layer {layer}'s anchor")
        anchor_layer = next(iter(anchor_layers))
        anchor_stats = anchors.get(str(anchor_layer))
        if anchor_stats is None:
            raise CaptureContractError(
                f"consumer layer {layer} references missing anchor layer {anchor_layer}"
            )
        cursor = 0
        global_rows: list[list[float]] = []
        for _, start, local_rows in shards:
            if start != cursor:
                raise CaptureContractError(
                    f"consumer layer {layer} head shards are overlapping or incomplete at {cursor}"
                )
            global_rows.extend(local_rows)
            cursor += len(local_rows)
        if cursor != global_num_heads:
            raise CaptureContractError(
                f"consumer layer {layer} head shards cover {cursor}, expected {global_num_heads}"
            )
        merged_consumers[str(layer)] = {
            "anchor_layer": anchor_layer,
            "dropped_mass": global_rows,
        }
    if not merged_consumers:
        raise CaptureContractError("capture drain contained no consumer-head statistics")
    compact_capture = {
        "compact_capture_schema_version": 1,
        "invocation": trusted_invocation,
        "geometry": geometry_values[0],
        "global_num_heads": global_num_heads,
        "eligible_tiles": eligible_tiles,
        "anchor_stats_by_layer": anchors,
        "consumer_layers": merged_consumers,
    }
    manifest = {
        "capture_schema_version": CAPTURE_SCHEMA_VERSION,
        "invocation": trusted_invocation,
        "invocation_sha256": expected_digest,
        "world_size": world_size,
        "global_num_heads": global_num_heads,
        "eligible_tiles": eligible_tiles,
        "candidate_cell_count": sum(
            len(cast("Sequence[object]", value["dropped_mass"])) * global_num_heads
            for value in merged_consumers.values()
        ),
        "attention_call_counts": common_counts,
        "tp_head_order_evidence": tp_head_order_evidence,
        "dense_shadow_evidence": common_shadow,
        "compact_capture_sha256": canonical_json_sha256(compact_capture),
    }
    return MergedCapture(compact_capture, manifest)
