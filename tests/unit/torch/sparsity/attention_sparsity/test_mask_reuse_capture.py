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

"""Tests for the strict ModelOpt mask-reuse capture contract."""

import json
import math
from dataclasses import replace
from hashlib import sha256

import pytest

from modelopt.torch.sparsity.attention_sparsity.plugins import mask_reuse_capture
from modelopt.torch.sparsity.attention_sparsity.plugins.mask_reuse_capture import (
    CaptureContractError,
    PromptSpec,
    build_capture_invocation,
    canonical_json_sha256,
    load_prompt_specs,
    merge_rank_captures,
    merge_rank_topology_discovery_captures,
    parse_prompt_specs_jsonl,
    source_capture_sha256,
    validate_begin_acks,
    validate_capture_statuses,
)

_CHECKPOINT = sha256(b"checkpoint-manifest").hexdigest()
_GROUP = sha256(b"source-group-0").hexdigest()


@pytest.mark.parametrize("frozen", [False, True])
def test_invocation_hashes_exact_tokens_once(monkeypatch, frozen):
    tokens = tuple(range(8448))
    identity = source_capture_sha256(tokens)
    prompt = _prompt()
    if frozen:
        prompt = replace(prompt, prompt_token_ids=tokens, prompt_token_ids_sha256=identity)
    calls = []

    def observed(values):
        calls.append(values)
        return source_capture_sha256(values)

    monkeypatch.setattr(mask_reuse_capture, "source_capture_sha256", observed)
    invocation = build_capture_invocation(
        model="test-model",
        checkpoint_manifest_sha256=_CHECKPOINT,
        prompt=prompt,
        prompt_token_ids=tokens,
        target_sparsity=0.7,
        threshold_scale_factor=_fit(),
    )
    assert invocation["source_capture_sha256"] == identity
    assert calls == [tokens]


def _fit():
    return {
        "formula": "a * exp(b * target_sparsity)",
        "prefill": {
            "a": 1.0,
            "b": 1.0,
            "min_observed_sparsity": 0.4,
            "max_observed_sparsity": 0.8,
        },
    }


def _prompt(split="calibration", prompt_id="p0"):
    partition = "development" if split == "calibration" else "outer_test"
    return PromptSpec(
        split=split,
        partition=partition,
        inner_fold=0 if partition == "development" else None,
        prompt_id=prompt_id,
        source="ruler/niah",
        source_group_sha256=_GROUP,
        prompt="prompt",
        min_kv_tokens=8192,
        max_kv_tokens=16384,
    )


def _invocation():
    return build_capture_invocation(
        model="test-model",
        checkpoint_manifest_sha256=_CHECKPOINT,
        prompt=_prompt(),
        prompt_token_ids=list(range(8448)),
        target_sparsity=0.7,
        threshold_scale_factor=_fit(),
    )


def _rank_payload(invocation, rank):
    start = rank * 2
    return {
        "capture_schema_version": 2,
        "rank": rank,
        "world_size": 2,
        "invocation": invocation,
        "invocation_sha256": canonical_json_sha256(invocation),
        "geometry": invocation["expected_geometry"],
        "global_num_heads": 4,
        "eligible_tiles": 131,
        "anchor_stats_by_layer": {
            "0": {
                "retained_tiles": [10, 11, 12, 13],
                "dropped_mass": [0.01, 0.02, 0.03, 0.04],
            },
            "4": {
                "retained_tiles": [14, 15, 16, 17],
                "dropped_mass": [0.05, 0.06, 0.07, 0.08],
            },
        },
        "consumer_layers": {
            "1": {
                "anchor_layer": 0,
                "consumer_head_start": start,
                "dropped_mass": [
                    [0.01 * (start + local + donor + 1) for donor in range(4)] for local in range(2)
                ],
            },
            "5": {
                "anchor_layer": 4,
                "consumer_head_start": start,
                "dropped_mass": [
                    [0.01 * (start + local + donor + 2) for donor in range(4)] for local in range(2)
                ],
            },
        },
        "attention_call_counts": {"prefill": 8, "decode": 0},
        "tp_head_order_evidence": {
            "sentinel_device_type": "cuda",
            "gather_dim": 0,
            "local_rank": rank,
            "local_num_heads": 2,
            "gathered_rank_local_head": [[0, 0], [0, 1], [1, 0], [1, 1]],
        },
        "dense_shadow_evidence": {
            "enabled": True,
            "atol_hex": (0.0).hex(),
            "rtol_hex": (0.0).hex(),
            "validated_layer_indices": [0, 1, 4, 5],
        },
    }


def _topology_rank_payload(invocation, rank):
    payload = _rank_payload(invocation, rank)
    start = rank * 2
    payload.pop("consumer_layers")
    payload.update(
        {
            "capture_mode": "topology_discovery",
            "attention_layers": [0, 1, 2, 3],
            "max_reuse_span": 2,
            "anchor_stats_by_layer": {
                str(layer): {
                    "retained_tiles": [10 + layer] * 4,
                    "dropped_mass": [0.01 * (layer + 1)] * 4,
                }
                for layer in range(4)
            },
            "consumer_candidates_by_layer": {
                str(consumer): {
                    str(anchor): {
                        "consumer_head_start": start,
                        "dropped_mass": [
                            [
                                0.01 * (anchor + consumer + start + local + donor + 1)
                                for donor in range(4)
                            ]
                            for local in range(2)
                        ],
                    }
                    for anchor in range(max(0, consumer - 2), consumer)
                }
                for consumer in range(1, 4)
            },
            "dense_shadow_evidence": {
                "enabled": True,
                "atol_hex": (0.0).hex(),
                "rtol_hex": (0.0).hex(),
                "validated_layer_indices": [0, 1, 2, 3],
            },
        }
    )
    return payload


def test_build_invocation_binds_exact_threshold_source_and_final_chunk():
    invocation = _invocation()
    expected_log2 = math.log2(1.0) + 0.7 * math.log2(math.e) - math.log2(8448)

    assert invocation["target_sparsity_hex"] == (0.7).hex()
    assert invocation["checkpoint_manifest_sha256"] == _CHECKPOINT
    assert invocation["source_group_sha256"] == _GROUP
    assert invocation["partition"] == "development"
    assert invocation["inner_fold"] == 0
    assert invocation["threshold_log2_hex"] == expected_log2.hex()
    assert invocation["threshold_lambda_hex"] == (2.0**expected_log2).hex()
    assert invocation["expected_geometry"] == {
        "q_tokens": 256,
        "kv_tokens": 8448,
        "q_start_tokens": 8192,
    }
    assert invocation["source_capture_sha256"] == source_capture_sha256(list(range(8448)))


def test_backend_schema_v2_invocation_golden_sha_is_byte_exact():
    invocation = build_capture_invocation(
        model="toy",
        checkpoint_manifest_sha256=_CHECKPOINT,
        prompt=PromptSpec(
            split="calibration",
            partition="development",
            inner_fold=0,
            prompt_id="p",
            source="s",
            source_group_sha256=_GROUP,
            prompt="unused",
            min_kv_tokens=1,
            max_kv_tokens=None,
        ),
        prompt_token_ids=list(range(33_024)),
        target_sparsity=0.7,
        threshold_scale_factor={
            "formula": "a * exp(b * target_sparsity)",
            "prefill": {
                "a": 14.47,
                "b": 10.91,
                "min_observed_sparsity": 0.0,
                "max_observed_sparsity": 1.0,
            },
        },
    )

    assert canonical_json_sha256(invocation) == (
        "d1bb38b0611b7a424f70e4567f50380ad8fef9f77ebf05df97643fea08367056"
    )


def test_source_fingerprint_does_not_hide_split_overlap():
    token_ids = [1, 2, 3]
    assert source_capture_sha256(token_ids) == source_capture_sha256(token_ids)
    assert source_capture_sha256(token_ids) != source_capture_sha256([1, 2, 4])


def test_source_fingerprint_matches_binary_oracle_across_block_boundary():
    tokens = [0, 2**64 - 1, *range(4095)]
    payload = b"modelopt-mask-reuse-token-ids-v1\0" + len(tokens).to_bytes(8, "little")
    payload += b"".join(token.to_bytes(8, "little") for token in tokens)
    expected = sha256(payload).hexdigest()
    assert source_capture_sha256(tokens) == expected
    assert source_capture_sha256(tuple(tokens)) == expected

    class Token(int):
        pass

    tokens[4095] = Token(tokens[4095])
    assert source_capture_sha256(tokens) == expected


@pytest.mark.parametrize("invalid", [True, -1, 1.0, 2**64])
def test_source_fingerprint_preserves_first_invalid_index(invalid):
    with pytest.raises(CaptureContractError) as caught:
        source_capture_sha256([0] * 4096 + [invalid, -2])
    suffix = (
        "exceeds uint64" if type(invalid) is int and invalid >= 2**64 else "must be an integer >= 0"
    )
    assert str(caught.value) == f"prompt_token_ids[4096] {suffix}"


@pytest.mark.parametrize("invalid", ["hash", "missing_hash", "empty", "boolean", "range"])
def test_exact_prompt_tokens_fail_closed_before_collection(invalid):
    tokens = list(range(256))
    raw = {
        "split": "calibration",
        "partition": "development",
        "inner_fold": 0,
        "prompt_id": "chat",
        "source": "corpus",
        "source_group_sha256": _GROUP,
        "prompt": "<user>question</user><assistant>",
        "min_kv_tokens": 129,
        "max_kv_tokens": 512,
        "prompt_token_ids": tokens,
        "prompt_token_ids_sha256": source_capture_sha256(tokens),
    }
    if invalid == "hash":
        raw["prompt_token_ids_sha256"] = "0" * 64
    elif invalid == "missing_hash":
        del raw["prompt_token_ids_sha256"]
    elif invalid == "empty":
        raw["prompt_token_ids"] = []
    elif invalid == "boolean":
        raw["prompt_token_ids"][0] = True
    else:
        raw["min_kv_tokens"] = 300
    with pytest.raises(CaptureContractError):
        PromptSpec.from_mapping(raw)


def test_capture_cannot_replace_exact_prompt_tokens():
    tokens = tuple(range(8448))
    prompt = replace(
        _prompt(), prompt_token_ids=tokens, prompt_token_ids_sha256=source_capture_sha256(tokens)
    )
    with pytest.raises(CaptureContractError, match="differ from the frozen"):
        build_capture_invocation(
            model="test-model",
            checkpoint_manifest_sha256=_CHECKPOINT,
            prompt=prompt,
            prompt_token_ids=list(reversed(tokens)),
            target_sparsity=0.7,
            threshold_scale_factor=_fit(),
        )


def test_prompt_plan_requires_unique_ids_within_each_split(tmp_path):
    path = tmp_path / "prompts.jsonl"
    rows = [
        {
            "split": split,
            "partition": "development" if split == "calibration" else "outer_test",
            "inner_fold": 0 if split == "calibration" else None,
            "prompt_id": prompt_id,
            "source": "dataset",
            "source_group_sha256": sha256(f"{split}-{prompt_id}".encode()).hexdigest(),
            "prompt": text,
            "min_kv_tokens": minimum,
            "max_kv_tokens": maximum,
        }
        for split, prompt_id, text, minimum, maximum in (
            ("calibration", "same", "a", 129, 512),
            ("calibration", "same", "b", 513, 1024),
            ("heldout", "held-0", "c", 129, 512),
            ("heldout", "held-1", "d", 513, 1024),
        )
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    with pytest.raises(CaptureContractError, match="unique within"):
        load_prompt_specs(path)


def _unicode_prompt_jsonl(separator, record_separator):
    rows = [
        {
            "split": split,
            "partition": "development" if split == "calibration" else "outer_test",
            "inner_fold": 0 if split == "calibration" else None,
            "prompt_id": split,
            "source": "dataset",
            "source_group_sha256": sha256(split.encode()).hexdigest(),
            "prompt": f"before{separator}after",
            "min_kv_tokens": 129,
            "max_kv_tokens": 512,
        }
        for split in ("calibration", "heldout")
    ]
    return (
        record_separator.join(json.dumps(row, ensure_ascii=False) for row in rows)
        + record_separator
    ).encode("utf-8")


@pytest.mark.parametrize("separator", ["\u0085", "\u2028", "\u2029"])
@pytest.mark.parametrize("record_separator", ["\n", "\r\n"])
def test_jsonl_preserves_unicode_separators_in_prompt_strings(separator, record_separator):
    prompts = parse_prompt_specs_jsonl(_unicode_prompt_jsonl(separator, record_separator))
    assert len(prompts) == 2
    assert {prompt.split for prompt in prompts} == {"calibration", "heldout"}
    assert all(prompt.prompt == f"before{separator}after" for prompt in prompts)


@pytest.mark.parametrize("invalid", ["utf8", "duplicate_key"])
def test_unicode_jsonl_still_rejects_invalid_bytes_and_duplicate_keys(invalid):
    payload = _unicode_prompt_jsonl("\u2028", "\n")
    if invalid == "utf8":
        payload = b"\xff" + payload
        message = "must be UTF-8"
    else:
        payload = payload.replace(
            b'"split": "calibration",', b'"split": "calibration", "split": "calibration",', 1
        )
        message = "duplicate"
    with pytest.raises(CaptureContractError, match=message):
        parse_prompt_specs_jsonl(payload)


def test_build_invocation_rejects_unqualified_final_chunk_and_extrapolated_target():
    with pytest.raises(CaptureContractError, match="at least 129"):
        build_capture_invocation(
            model="test-model",
            checkpoint_manifest_sha256=_CHECKPOINT,
            prompt=replace(_prompt(), max_kv_tokens=9000),
            prompt_token_ids=list(range(8200)),
            target_sparsity=0.7,
            threshold_scale_factor=_fit(),
        )
    with pytest.raises(CaptureContractError, match="outside the observed"):
        build_capture_invocation(
            model="test-model",
            checkpoint_manifest_sha256=_CHECKPOINT,
            prompt=_prompt(),
            prompt_token_ids=list(range(8448)),
            target_sparsity=0.9,
            threshold_scale_factor=_fit(),
        )


def test_status_and_begin_require_every_rank_and_exact_invocation():
    statuses = [
        {
            "capture_schema_version": 2,
            "available": True,
            "rank": rank,
            "world_size": 2,
            "reason": None,
        }
        for rank in range(2)
    ]
    assert validate_capture_statuses(statuses) == 2
    invocation = _invocation()
    digest = canonical_json_sha256(invocation)
    acknowledgements = [
        {
            "capture_schema_version": 2,
            "armed": True,
            "rank": rank,
            "world_size": 2,
            "invocation_sha256": digest,
        }
        for rank in range(2)
    ]
    assert validate_begin_acks(acknowledgements, invocation) == 2

    statuses[1]["available"] = False
    statuses[1]["reason"] = "sink disabled"
    with pytest.raises(CaptureContractError, match="sink disabled"):
        validate_capture_statuses(statuses)
    with pytest.raises(CaptureContractError, match="exactly cover ranks"):
        validate_begin_acks(acknowledgements[:1], invocation)


def test_merge_rank_captures_concatenates_consumer_shards_deterministically():
    invocation = _invocation()
    merged = merge_rank_captures(
        [_rank_payload(invocation, 1), _rank_payload(invocation, 0)], invocation
    )

    assert len(merged.capture["consumer_layers"]) == 2
    assert merged.manifest["world_size"] == 2
    assert merged.manifest["global_num_heads"] == 4
    assert merged.manifest["candidate_cell_count"] == 2 * 4 * 4
    assert merged.manifest["attention_call_counts"] == {"prefill": 8, "decode": 0}
    assert [item["global_head_start"] for item in merged.manifest["tp_head_order_evidence"]] == [
        0,
        2,
    ]
    assert merged.manifest["dense_shadow_evidence"]["validated_layer_indices"] == [0, 1, 4, 5]
    consumer = merged.capture["consumer_layers"]["1"]
    assert consumer["anchor_layer"] == 0
    assert consumer["dropped_mass"][2][3] == pytest.approx(0.06)
    assert set(merged.capture["anchor_stats_by_layer"]) == {"0", "4"}


@pytest.mark.parametrize(
    ("merge", "payload"),
    [
        (merge_rank_captures, _rank_payload),
        (merge_rank_topology_discovery_captures, _topology_rank_payload),
    ],
)
def test_stream_merge_requires_one_chunk_census_without_changing_legacy(merge, payload):
    invocation = _invocation()
    ranks = [payload(invocation, rank) for rank in range(2)]
    legacy = merge(ranks, invocation)
    for rank in ranks:
        rank["attention_call_counts"]["prefill"] = 4
    with pytest.raises(CaptureContractError, match="attention call count"):
        merge(ranks, invocation)
    streamed = merge(ranks, invocation, streamed_chunk=True)
    assert streamed.capture == legacy.capture
    assert streamed.manifest["attention_call_counts"] == {"prefill": 4, "decode": 0}
    ranks[0]["attention_call_counts"]["prefill"] = 3
    with pytest.raises(CaptureContractError, match="attention call counts"):
        merge(ranks, invocation, streamed_chunk=True)


def test_merge_topology_discovery_captures_concatenates_every_candidate_edge():
    invocation = _invocation()
    merged = merge_rank_topology_discovery_captures(
        [
            _topology_rank_payload(invocation, 1),
            _topology_rank_payload(invocation, 0),
        ],
        invocation,
    )

    assert merged.capture["attention_layers"] == [0, 1, 2, 3]
    assert merged.capture["max_reuse_span"] == 2
    assert set(merged.capture["anchor_stats_by_layer"]) == {"0", "1", "2", "3"}
    assert set(merged.capture["consumer_candidates_by_layer"]["3"]) == {"1", "2"}
    assert merged.capture["consumer_candidates_by_layer"]["3"]["1"]["dropped_mass"][2][
        3
    ] == pytest.approx(0.1)
    assert merged.manifest["candidate_edge_count"] == 5
    assert merged.manifest["candidate_cell_count"] == 5 * 4 * 4


@pytest.mark.parametrize(
    "corruption",
    [
        "wrong_invocation",
        "anchor_disagreement",
        "head_gap",
        "rank_permutation",
        "decode_call",
        "prefill_count",
        "missing_dense_shadow",
    ],
)
def test_merge_rank_captures_fails_closed_on_incomplete_or_disagreed_evidence(corruption):
    invocation = _invocation()
    rank0 = _rank_payload(invocation, 0)
    rank1 = _rank_payload(invocation, 1)
    if corruption == "wrong_invocation":
        rank1["invocation"] = dict(invocation, prompt_id="other")
    elif corruption == "anchor_disagreement":
        rank1["anchor_stats_by_layer"]["0"]["retained_tiles"][0] = 9
    elif corruption == "head_gap":
        rank1["consumer_layers"]["1"]["consumer_head_start"] = 3
    elif corruption == "rank_permutation":
        rank1["tp_head_order_evidence"]["gathered_rank_local_head"] = [
            [1, 0],
            [1, 1],
            [0, 0],
            [0, 1],
        ]
    elif corruption == "decode_call":
        rank0["attention_call_counts"]["decode"] = 1
        rank1["attention_call_counts"]["decode"] = 1
    elif corruption == "prefill_count":
        rank0["attention_call_counts"]["prefill"] = 7
        rank1["attention_call_counts"]["prefill"] = 7
    else:
        rank0["dense_shadow_evidence"]["validated_layer_indices"] = [0, 1, 4]
        rank1["dense_shadow_evidence"]["validated_layer_indices"] = [0, 1, 4]

    with pytest.raises(CaptureContractError):
        merge_rank_captures([rank0, rank1], invocation)
