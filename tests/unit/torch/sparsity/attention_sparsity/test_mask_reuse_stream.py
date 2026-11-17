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

"""Exact-prefix and dense-exclusion contracts; no GPU performance evidence."""

import copy
import json
from dataclasses import asdict

import pytest

from modelopt.torch.sparsity.attention_sparsity.calibration.mask_reuse_compact import (
    CompactMaskReuseCapture,
)
from modelopt.torch.sparsity.attention_sparsity.calibration.mask_reuse_topology import (
    TopologyDiscoveryCapture,
)
from modelopt.torch.sparsity.attention_sparsity.plugins import mask_reuse_capture, mask_reuse_stream
from modelopt.torch.sparsity.attention_sparsity.plugins.mask_reuse_capture import (
    CaptureContractError,
    PromptSpec,
    build_capture_invocation,
    canonical_json_sha256,
    source_capture_sha256,
)
from modelopt.torch.sparsity.attention_sparsity.plugins.mask_reuse_stream import (
    build_analytical_dense_capture,
    build_capture_stream,
    capture_stream_prompts,
    read_capture_stream_spools,
    stream_chunk_identity,
    validate_capture_stream,
    validate_stream_acknowledgements,
)


def _prompt(length=16513, split="calibration"):
    tokens = tuple(range(length))
    return PromptSpec(
        split=split,
        partition="development" if split == "calibration" else "outer_test",
        inner_fold=0 if split == "calibration" else None,
        prompt_id=split,
        source="synthetic",
        source_group_sha256=("a" if split == "calibration" else "b") * 64,
        prompt="synthetic",
        min_kv_tokens=129,
        max_kv_tokens=253952,
        prompt_token_ids=tokens,
        prompt_token_ids_sha256=source_capture_sha256(tokens),
    )


def _fit(a=1.0, b=1.0):
    return {"formula": "a * exp(b * target_sparsity)", "prefill": {"a": a, "b": b}}


def _request(prompt=None, *, a=1.0, b=1.0):
    return build_capture_stream(
        model="toy",
        checkpoint_manifest_sha256="c" * 64,
        prompt=prompt or _prompt(),
        target_sparsity=0.7,
        threshold_scale_factor=_fit(a, b),
    )


def test_stream_invocations_equal_independent_exact_prefixes():
    prompt = _prompt()
    request = _request(prompt)
    prefixes = capture_stream_prompts([prompt])
    assert len(prefixes) == len(request["chunks"]) == 3
    for chunk, prefix in zip(request["chunks"], prefixes, strict=True):
        assert chunk["invocation"] == build_capture_invocation(
            model="toy",
            checkpoint_manifest_sha256="c" * 64,
            prompt=prefix,
            prompt_token_ids=prefix.prompt_token_ids,
            target_sparsity=0.7,
            threshold_scale_factor=_fit(),
        )
        assert prefix.source_group_sha256 == prompt.source_group_sha256
        assert PromptSpec.from_mapping(json.loads(json.dumps(asdict(prefix)))) == prefix
    assert [c["geometry"]["q_tokens"] for c in request["chunks"]] == [8192, 8192, 129]


@pytest.mark.parametrize("dense_prefix", [False, True])
def test_stream_reuses_computed_invocation_identity(monkeypatch, dense_prefix):
    prompt = _prompt()
    calls = []

    def observed(tokens):
        calls.append(len(tokens))
        return source_capture_sha256(tokens)

    monkeypatch.setattr(mask_reuse_capture, "source_capture_sha256", observed)
    monkeypatch.setattr(mask_reuse_stream, "source_capture_sha256", observed)
    request = _request(prompt, a=8192.0 if dense_prefix else 1.0, b=0.0)
    assert calls == [16513, 8192, *([] if dense_prefix else [8192]), 16384, 16384, 16513, 16513]
    assert request["chunks"][0]["dense_reason"] == ("threshold_disabled" if dense_prefix else None)
    assert request["chunks"][-1]["prefix_source_capture_sha256"] == prompt.prompt_token_ids_sha256


def test_dense_exclusions_preserve_full_request_and_boundary():
    request = _request(_prompt(24576 + 100), a=16384.0, b=0.0)
    assert [chunk["dense_reason"] for chunk in request["chunks"]] == [
        "threshold_disabled",
        "threshold_disabled",
        None,
        "short_query",
    ]
    assert request["chunks"][1]["invocation"] is None  # lambda == 1 is dense.
    identity = stream_chunk_identity(request, request["chunks"][0])
    assert identity["dense_reason"] == "threshold_disabled"
    assert "threshold_lambda_hex" not in identity  # Not falsely labeled measured evidence.


@pytest.mark.parametrize(
    "mutation", ["missing", "order", "group", "full_hash", "short", "dense_payload"]
)
def test_stream_rejects_broken_coverage_and_identity(mutation):
    request = copy.deepcopy(_request())
    if mutation == "missing":
        request["chunks"].pop()
    elif mutation == "order":
        request["chunks"].reverse()
    elif mutation == "group":
        request["chunks"][0]["invocation"]["source_group_sha256"] = "d" * 64
    elif mutation == "full_hash":
        request["parent_invocation"]["source_capture_sha256"] = "e" * 64
    elif mutation == "short":
        request["chunks"][0].update(dense_reason="short_query", invocation=None)
    else:
        request["chunks"][0]["dense_reason"] = "threshold_disabled"
    with pytest.raises(CaptureContractError):
        validate_capture_stream(request)


def _spool(tmp_path, request, world=2):
    for rank in range(world):
        rows = []
        for chunk in request["chunks"]:
            payload = (
                None
                if chunk["invocation"] is None
                else {
                    "rank": rank,
                    "world_size": world,
                    "invocation": chunk["invocation"],
                }
            )
            rows.append(
                {
                    "stream_schema_version": 1,
                    "request_sha256": canonical_json_sha256(request),
                    "chunk_index": chunk["chunk_index"],
                    "payload": payload,
                }
            )
        (tmp_path / f"rank-{rank}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )


def test_stream_spools_include_dense_work_and_require_each_rank(tmp_path):
    request = _request(_prompt(16500), a=8192.0, b=0.0)
    _spool(tmp_path, request)
    rows = list(read_capture_stream_spools(request, tmp_path, world_size=2))
    assert [index for index, _ in rows] == [0, 1, 2]
    assert rows[0][1] == [None, None]
    assert rows[-1][1] == [None, None]
    path = tmp_path / "rank-1.jsonl"
    path.write_bytes(path.read_bytes().splitlines(keepends=True)[0])
    assert (
        len(list(read_capture_stream_spools(request, tmp_path, world_size=2, allow_partial=True)))
        == 1
    )
    with pytest.raises(CaptureContractError, match="missing or partial"):
        list(read_capture_stream_spools(request, tmp_path, world_size=2))
    rank0 = tmp_path / "rank-0.jsonl"
    rank0.write_bytes(rank0.read_bytes().splitlines(keepends=True)[0] + b'{"unfinished":')
    assert (
        len(list(read_capture_stream_spools(request, tmp_path, world_size=2, allow_partial=True)))
        == 1
    )
    with pytest.raises(CaptureContractError, match="missing or partial"):
        list(read_capture_stream_spools(request, tmp_path, world_size=2))


@pytest.mark.parametrize("partial_tail", [False, True])
def test_four_rank_interruption_keeps_only_common_validated_prefix(tmp_path, partial_tail):
    request = _request()
    _spool(tmp_path, request, world=4)
    for rank, complete_count in enumerate((3, 2, 1, 2)):
        path = tmp_path / f"rank-{rank}.jsonl"
        lines = path.read_bytes().splitlines(keepends=True)
        path.write_bytes(
            b"".join(lines[:complete_count])
            + (b'{"unfinished":' if partial_tail and rank == 2 else b"")
        )
    original = {path: path.read_bytes() for path in tmp_path.iterdir()}
    assert [
        index
        for index, _ in read_capture_stream_spools(
            request, tmp_path, world_size=4, allow_partial=True
        )
    ] == [0]
    assert all(path.read_bytes() == value for path, value in original.items())
    rank0 = tmp_path / "rank-0.jsonl"
    rank0.write_bytes(rank0.read_bytes().splitlines(keepends=True)[0] + b'{"malformed":}\n')
    with pytest.raises(CaptureContractError, match="invalid stream spool JSON"):
        list(read_capture_stream_spools(request, tmp_path, world_size=4, allow_partial=True))


def test_ack_requires_exact_request_and_tp_membership():
    request = _request()
    values = [
        {
            "stream_schema_version": 1,
            "rank": rank,
            "world_size": 2,
            "armed": True,
            "request_sha256": canonical_json_sha256(request),
        }
        for rank in range(2)
    ]
    validate_stream_acknowledgements(values, request, expected_world_size=2)
    values[1]["rank"] = 0
    with pytest.raises(CaptureContractError, match="exactly once"):
        validate_stream_acknowledgements(values, request, expected_world_size=2)


@pytest.mark.parametrize("tamper", [None, "mass", "retained", "threshold", "claim_measured"])
def test_analytical_dense_is_explicit_and_exact(tamper):
    request = _request(_prompt(16384), a=8192.0, b=0.0)
    raw = build_analytical_dense_capture(
        request,
        request["chunks"][0],
        fit=_fit(8192.0, 0.0),
        measured_template={
            "global_num_heads": 2,
            "attention_layers": [0, 2, 4],
            "max_reuse_span": 2,
        },
    )
    if tamper == "mass":
        raw["consumer_candidates_by_layer"]["2"]["0"]["dropped_mass"][0][0] = 0.001
    elif tamper == "retained":
        raw["anchor_stats_by_layer"]["0"]["retained_tiles"][0] -= 1
    elif tamper == "threshold":
        raw["invocation"].update(threshold_log2_hex=(-1.0).hex(), threshold_lambda_hex=(0.5).hex())
    elif tamper == "claim_measured":
        raw["analytical_dense"]["measured"] = True
    if tamper:
        with pytest.raises(ValueError, match="analytical"):
            TopologyDiscoveryCapture.from_mapping(raw)
        return
    parsed = TopologyDiscoveryCapture.from_mapping(raw)
    assert parsed.threshold_lambda == 1.0 and parsed.threshold_log2 == 0.0
    compact = {
        key: raw[key]
        for key in (
            "analytical_dense",
            "invocation",
            "geometry",
            "global_num_heads",
            "eligible_tiles",
        )
    }
    compact.update(
        compact_capture_schema_version=2,
        anchor_stats_by_layer={"0": raw["anchor_stats_by_layer"]["0"]},
        consumer_layers={
            str(layer): {"anchor_layer": 0, "dropped_mass": [[0.0, 0.0], [0.0, 0.0]]}
            for layer in (2, 4)
        },
    )
    assert CompactMaskReuseCapture.from_mapping(compact).threshold_lambda == 1.0
