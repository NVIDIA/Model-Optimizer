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

"""Whole-prefill request identities around the unchanged per-chunk capture protocol."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

from ..calibration.analytical_dense import ANALYTICAL_DENSE_EVIDENCE
from .mask_reuse_capture import (
    MAX_QUERY_CHUNK_TOKENS,
    CaptureContractError,
    PromptSpec,
    _canonical_float_hex,
    _eligible_tiles,
    _integer,
    _sha256,
    _text,
    _validate_invocation,
    build_capture_invocation,
    canonical_json_sha256,
    source_capture_sha256,
)

__all__ = [
    "WHOLE_PREFILL_RANGES",
    "build_analytical_dense_capture",
    "build_capture_stream",
    "capture_stream_prompts",
    "capture_stream_work",
    "read_capture_stream_spools",
    "stream_chunk_identity",
    "validate_capture_stream",
    "validate_stream_acknowledgements",
    "validate_whole_prefill_evidence",
]

WHOLE_PREFILL_RANGES = ((129, 65536), (65537, 131072), (131073, 196608), (196609, 253952))
_PARENT_FIELDS = frozenset(
    {
        "model",
        "checkpoint_manifest_sha256",
        "split",
        "partition",
        "inner_fold",
        "prompt_id",
        "source",
        "source_group_sha256",
        "source_capture_sha256",
        "sample_length",
        "target_sparsity_hex",
    }
)
_CHUNK_FIELDS = frozenset(
    {"chunk_index", "geometry", "prefix_source_capture_sha256", "dense_reason", "invocation"}
)


def _chunk_geometries(length: int):
    for start in range(0, length, MAX_QUERY_CHUNK_TOKENS):
        end = min(start + MAX_QUERY_CHUNK_TOKENS, length)
        yield {"q_tokens": end - start, "kv_tokens": end, "q_start_tokens": start}


def _prefix_prompt(prompt: PromptSpec, length: int) -> PromptSpec:
    assert prompt.prompt_token_ids is not None
    bounds = next(
        (bounds for bounds in WHOLE_PREFILL_RANGES if bounds[0] <= length <= bounds[1]), None
    )
    if bounds is None:
        raise CaptureContractError("prefix lies outside the preregistered whole-prefill ranges")
    tokens = prompt.prompt_token_ids[:length]
    return replace(
        prompt,
        prompt_id=f"{prompt.prompt_id}@prefix-{length}",
        prompt=f"Exact token prefix of {prompt.prompt_id}; prompt_token_ids are authoritative.",
        min_kv_tokens=bounds[0],
        max_kv_tokens=bounds[1],
        prompt_token_ids=tokens,
        prompt_token_ids_sha256=source_capture_sha256(tokens),
    )


def capture_stream_prompts(prompts: Sequence[PromptSpec]) -> list[PromptSpec]:
    """Derive fit inputs without treating related prefixes as independent source groups."""
    result = []
    seen = set()
    for prompt in prompts:
        if prompt.prompt_token_ids is None:
            raise CaptureContractError("whole-prefill capture requires frozen exact token IDs")
        for geometry in _chunk_geometries(len(prompt.prompt_token_ids)):
            if geometry["q_tokens"] <= 128 or geometry["kv_tokens"] > WHOLE_PREFILL_RANGES[-1][1]:
                continue
            prefix = _prefix_prompt(prompt, geometry["kv_tokens"])
            if prefix.prompt_id in seen:
                raise CaptureContractError("whole-prefill prefix prompt IDs are not unique")
            seen.add(prefix.prompt_id)
            result.append(prefix)
    return result


def build_capture_stream(
    *,
    model: str,
    checkpoint_manifest_sha256: str,
    prompt: PromptSpec,
    target_sparsity: float,
    threshold_scale_factor: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind every actual chunk to the full native request and its exact token prefix."""
    if prompt.prompt_token_ids is None:
        raise CaptureContractError("whole-prefill capture requires frozen exact token IDs")
    tokens = prompt.prompt_token_ids
    if source_capture_sha256(tokens) != prompt.prompt_token_ids_sha256:
        raise CaptureContractError("whole-prefill prompt token identity differs")
    if not math.isfinite(target_sparsity) or not 0 < target_sparsity < 1:
        raise CaptureContractError("target_sparsity must be in (0, 1)")
    parent = {
        "model": model,
        "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
        **{
            key: getattr(prompt, key)
            for key in (
                "split",
                "partition",
                "inner_fold",
                "prompt_id",
                "source",
                "source_group_sha256",
            )
        },
        "source_capture_sha256": prompt.prompt_token_ids_sha256,
        "sample_length": len(tokens),
        "target_sparsity_hex": target_sparsity.hex(),
    }
    params = threshold_scale_factor["prefill"]
    lower, upper = params.get("min_observed_sparsity"), params.get("max_observed_sparsity")
    if (lower is not None and target_sparsity < lower) or (
        upper is not None and target_sparsity > upper
    ):
        raise CaptureContractError(
            "target_sparsity is outside the observed vanilla calibration range"
        )
    chunks = []
    for index, geometry in enumerate(_chunk_geometries(len(tokens))):
        length = geometry["kv_tokens"]
        log_threshold = (
            math.log2(float(params["a"]))
            + float(params["b"]) * target_sparsity * math.log2(math.e)
            - math.log2(length)
        )
        dense_reason = (
            "short_query"
            if geometry["q_tokens"] <= 128
            else "outside_coverage"
            if length > WHOLE_PREFILL_RANGES[-1][1]
            else "threshold_disabled"
            if log_threshold >= 0
            else None
        )
        invocation = None
        if dense_reason is None:
            prefix = _prefix_prompt(prompt, length)
            assert prefix.prompt_token_ids is not None
            invocation = build_capture_invocation(
                model=model,
                checkpoint_manifest_sha256=checkpoint_manifest_sha256,
                prompt=prefix,
                prompt_token_ids=prefix.prompt_token_ids,
                target_sparsity=target_sparsity,
                threshold_scale_factor=threshold_scale_factor,
            )
        chunks.append(
            {
                "chunk_index": index,
                "geometry": geometry,
                "prefix_source_capture_sha256": (
                    invocation["source_capture_sha256"]
                    if invocation is not None
                    else source_capture_sha256(tokens[:length])
                ),
                "dense_reason": dense_reason,
                "invocation": invocation,
            }
        )
    request = {"stream_schema_version": 1, "parent_invocation": parent, "chunks": chunks}
    validate_capture_stream(request)
    return request


def validate_capture_stream(request: Mapping[str, Any]) -> None:
    """Reject missing, reordered, cross-source, or unaccounted-for chunk identities."""
    if (
        set(request) != {"stream_schema_version", "parent_invocation", "chunks"}
        or request["stream_schema_version"] != 1
    ):
        raise CaptureContractError("unsupported whole-prefill request schema")
    parent = request["parent_invocation"]
    if not isinstance(parent, Mapping) or set(parent) != _PARENT_FIELDS:
        raise CaptureContractError("whole-prefill parent identity fields differ")
    for field in ("checkpoint_manifest_sha256", "source_group_sha256", "source_capture_sha256"):
        _sha256(parent[field], field)
    for field in ("model", "prompt_id", "source"):
        _text(parent[field], field)
    if (parent["partition"], parent["split"]) not in (
        ("development", "calibration"),
        ("outer_test", "heldout"),
    ):
        raise CaptureContractError("whole-prefill parent split and partition disagree")
    if parent["partition"] == "development":
        _integer(parent["inner_fold"], "inner_fold")
    elif parent["inner_fold"] is not None:
        raise CaptureContractError("heldout whole-prefill parent must have null inner_fold")
    target = _canonical_float_hex(parent["target_sparsity_hex"], "target_sparsity_hex")
    if not 0 < target < 1:
        raise CaptureContractError("whole-prefill target sparsity must be in (0, 1)")
    length = _integer(parent["sample_length"], "sample_length", minimum=1)
    expected = list(_chunk_geometries(length))
    chunks = request["chunks"]
    if not isinstance(chunks, list) or len(chunks) != len(expected):
        raise CaptureContractError("whole-prefill request omits or adds chunks")
    for index, (chunk, geometry) in enumerate(zip(chunks, expected, strict=True)):
        if not isinstance(chunk, Mapping) or set(chunk) != _CHUNK_FIELDS:
            raise CaptureContractError("whole-prefill chunk fields differ")
        if (
            type(chunk["chunk_index"]) is not int
            or chunk["chunk_index"] != index
            or chunk["geometry"] != geometry
        ):
            raise CaptureContractError("whole-prefill chunk order or geometry differs")
        _sha256(chunk["prefix_source_capture_sha256"], "prefix_source_capture_sha256")
        if (
            index == len(chunks) - 1
            and chunk["prefix_source_capture_sha256"] != parent["source_capture_sha256"]
        ):
            raise CaptureContractError("final prefix and full request token identities differ")
        reason = chunk["dense_reason"]
        if reason not in (None, "short_query", "threshold_disabled", "outside_coverage"):
            raise CaptureContractError("unknown dense exclusion reason")
        if reason is not None:
            if chunk["invocation"] is not None:
                raise CaptureContractError(
                    "dense exclusion cannot claim measured sparse statistics"
                )
            if reason == "short_query" and geometry["q_tokens"] > 128:
                raise CaptureContractError("short-query exclusion has more than 128 query tokens")
            if (
                reason == "outside_coverage"
                and geometry["kv_tokens"] <= WHOLE_PREFILL_RANGES[-1][1]
            ):
                raise CaptureContractError("outside-coverage exclusion lies inside coverage")
            continue
        invocation = _validate_invocation(chunk["invocation"])
        for field in _PARENT_FIELDS - {"prompt_id", "sample_length", "source_capture_sha256"}:
            if invocation[field] != parent[field]:
                raise CaptureContractError(f"chunk changes parent identity field {field}")
        if (
            invocation["prompt_id"] != f"{parent['prompt_id']}@prefix-{geometry['kv_tokens']}"
            or invocation["source_capture_sha256"] != chunk["prefix_source_capture_sha256"]
            or invocation["expected_geometry"] != geometry
            or (invocation["min_kv_tokens"], invocation["max_kv_tokens"])
            not in WHOLE_PREFILL_RANGES
            or not invocation["min_kv_tokens"]
            <= geometry["kv_tokens"]
            <= invocation["max_kv_tokens"]
        ):
            raise CaptureContractError("chunk invocation differs from its exact prefix")
    # Includes dense exclusions; this is one head/layer's causal logical-tile denominator.
    if sum(_eligible_tiles(chunk["geometry"]) for chunk in chunks) <= 0:
        raise CaptureContractError("whole-prefill request has no eligible work")


def validate_stream_acknowledgements(values, request, *, expected_world_size: int) -> None:
    """Require every TP rank to arm the exact same whole-prefill request."""
    expected_sha = canonical_json_sha256(request)
    fields = {"stream_schema_version", "rank", "world_size", "armed", "request_sha256"}
    if not isinstance(values, list) or len(values) != expected_world_size:
        raise CaptureContractError("stream acknowledgement count differs from TP size")
    ranks = []
    for value in values:
        if (
            not isinstance(value, Mapping)
            or set(value) != fields
            or value["stream_schema_version"] != 1
            or value["armed"] is not True
            or type(value["rank"]) is not int
            or value["world_size"] != expected_world_size
            or value["request_sha256"] != expected_sha
        ):
            raise CaptureContractError("stream acknowledgement identity differs")
        ranks.append(value["rank"])
    if sorted(ranks) != list(range(expected_world_size)):
        raise CaptureContractError("stream acknowledgements do not cover each TP rank exactly once")


def read_capture_stream_spools(
    request: Mapping[str, Any],
    spool_directory: str | Path,
    *,
    world_size: int,
    completed_chunk_count: int = 0,
    allow_partial: bool = False,
):
    """Read one rank-aligned chunk at a time, never loading all request statistics."""
    validate_capture_stream(request)
    _integer(world_size, "world_size", minimum=1)
    _integer(completed_chunk_count, "completed_chunk_count")
    if completed_chunk_count > len(request["chunks"]):
        raise CaptureContractError("completed_chunk_count exceeds the whole-prefill request")
    request_sha = canonical_json_sha256(request)
    expected = [
        chunk["chunk_index"]
        for chunk in request["chunks"]
        if chunk["chunk_index"] >= completed_chunk_count
    ]

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise CaptureContractError("duplicate stream spool JSON key")
            result[key] = value
        return result

    with ExitStack() as stack:
        files = []
        for rank in range(world_size):
            path = Path(spool_directory) / f"rank-{rank}.jsonl"
            if not path.exists() and allow_partial:
                return
            if path.is_symlink():
                raise CaptureContractError("stream spool must not be a symlink")
            files.append(stack.enter_context(path.open("rb")))
        for index in expected:
            lines = [handle.readline() for handle in files]
            payloads = []
            for rank, line in enumerate(lines):
                if allow_partial and (not line or not line.endswith(b"\n")):
                    # A killed writer may leave an uncommitted EOF tail. No part
                    # of that row is accepted; complete malformed rows still fail.
                    continue
                if not line or not line.endswith(b"\n"):
                    raise CaptureContractError("stream spool has a missing or partial chunk record")
                try:
                    value = json.loads(line, object_pairs_hook=pairs)
                except (json.JSONDecodeError, UnicodeDecodeError) as error:
                    raise CaptureContractError("invalid stream spool JSON") from error
                if (
                    not isinstance(value, dict)
                    or set(value)
                    != {"stream_schema_version", "request_sha256", "chunk_index", "payload"}
                    or value["stream_schema_version"] != 1
                    or value["request_sha256"] != request_sha
                    or value["chunk_index"] != index
                ):
                    raise CaptureContractError(
                        "stream spool is not the exact source-bound chunk sequence"
                    )
                invocation = request["chunks"][index]["invocation"]
                if invocation is None:
                    if value["payload"] is not None:
                        raise CaptureContractError("dense chunk must not claim measured statistics")
                elif (
                    not isinstance(value["payload"], Mapping)
                    or value["payload"].get("rank") != rank
                    or value["payload"].get("world_size") != world_size
                    or value["payload"].get("invocation") != invocation
                ):
                    raise CaptureContractError("stream payload rank or invocation differs")
                payloads.append(value["payload"])
            if len(payloads) != world_size:
                return
            yield index, payloads
        if any(handle.read(1) for handle in files):
            raise CaptureContractError("stream spool contains duplicate or extra chunk records")


def stream_chunk_identity(request, chunk):
    """Return progress identity, keeping analytical dense work out of measured captures."""
    if chunk["invocation"] is not None:
        return chunk["invocation"]
    return {
        "stream_request_sha256": canonical_json_sha256(request),
        "chunk_index": chunk["chunk_index"],
        "geometry": chunk["geometry"],
        "prefix_source_capture_sha256": chunk["prefix_source_capture_sha256"],
        "dense_reason": chunk["dense_reason"],
    }


def capture_stream_work(request):
    """Count all logical causal work, including native-dense chunks, per head/layer."""
    return {
        "request_sha256": canonical_json_sha256(request),
        "eligible_tiles_per_head_layer": sum(
            _eligible_tiles(chunk["geometry"]) for chunk in request["chunks"]
        ),
        "native_dense_tiles_per_head_layer": sum(
            _eligible_tiles(chunk["geometry"])
            for chunk in request["chunks"]
            if chunk["invocation"] is None
        ),
    }


def validate_whole_prefill_evidence(raw):
    """Validate explicit coverage and full denominators without claiming a global target passed."""
    whole = raw["whole_prefill"]
    if (
        not isinstance(whole, Mapping)
        or set(whole)
        != {
            "schema_version",
            "coverage_ranges",
            "requests",
            "dense_exclusion_count",
            "measured_capture_count",
            "target_scope",
            "work_by_request",
        }
        or whole["schema_version"] != 1
        or whole["coverage_ranges"] != [list(bounds) for bounds in WHOLE_PREFILL_RANGES]
        or not isinstance(whole["requests"], list)
        or not whole["requests"]
    ):
        raise CaptureContractError("whole-prefill evidence schema or coverage differs")
    for request in whole["requests"]:
        validate_capture_stream(request)
        parent = request["parent_invocation"]
        if (
            parent["model"] != raw["model"]
            or parent["checkpoint_manifest_sha256"] != raw["checkpoint_manifest_sha256"]
            or parent["target_sparsity_hex"] not in raw["target_sparsity_hex"]
        ):
            raise CaptureContractError("whole-prefill request and manifest identities differ")
    chunks = [chunk for request in whole["requests"] for chunk in request["chunks"]]
    measured = sum(chunk["invocation"] is not None for chunk in chunks)
    if (
        whole["work_by_request"] != [capture_stream_work(request) for request in whole["requests"]]
        or whole["dense_exclusion_count"] != len(chunks) - measured
        or whole["measured_capture_count"] != measured
        or raw["capture_count"] != measured
    ):
        raise CaptureContractError("whole-prefill evidence omits measured or dense work")


def build_analytical_dense_capture(request, chunk, *, fit, measured_template):
    """Construct exact all-keep coefficients separately from the original measured file."""
    if chunk["dense_reason"] != "threshold_disabled" or chunk["invocation"] is not None:
        raise CaptureContractError("analytical all-keep input requires a threshold-disabled chunk")
    parent, geometry = request["parent_invocation"], chunk["geometry"]
    length = geometry["kv_tokens"]
    bounds = next(bounds for bounds in WHOLE_PREFILL_RANGES if bounds[0] <= length <= bounds[1])
    params = fit["prefill"]
    target = float.fromhex(parent["target_sparsity_hex"])
    log_threshold = (
        math.log2(float(params["a"]))
        + float(params["b"]) * target * math.log2(math.e)
        - math.log2(length)
    )
    if log_threshold < 0:
        raise CaptureContractError("analytical dense chunk disagrees with the verified vanilla fit")
    invocation = {
        **parent,
        "capture_schema_version": 2,
        "prompt_id": f"{parent['prompt_id']}@prefix-{length}",
        "source_capture_sha256": chunk["prefix_source_capture_sha256"],
        "min_kv_tokens": bounds[0],
        "max_kv_tokens": bounds[1],
        "sample_length": length,
        "threshold_log2_hex": log_threshold.hex(),
        "threshold_lambda_hex": (2.0**log_threshold).hex(),
        "expected_geometry": geometry,
    }
    heads = measured_template["global_num_heads"]
    layers = measured_template["attention_layers"]
    span = measured_template["max_reuse_span"]
    eligible = _eligible_tiles(geometry)
    return {
        "topology_discovery_capture_schema_version": 2,
        "analytical_dense": dict(ANALYTICAL_DENSE_EVIDENCE),
        "invocation": invocation,
        "geometry": geometry,
        "global_num_heads": heads,
        "eligible_tiles": eligible,
        "attention_layers": layers,
        "max_reuse_span": span,
        "anchor_stats_by_layer": {
            str(layer): {"retained_tiles": [eligible] * heads, "dropped_mass": [0.0] * heads}
            for layer in layers
        },
        "consumer_candidates_by_layer": {
            str(consumer): {
                str(anchor): {"dropped_mass": [[0.0] * heads for _ in range(heads)]}
                for anchor in layers[max(0, index - span) : index]
            }
            for index, consumer in enumerate(layers)
            if index > 0
        },
    }
