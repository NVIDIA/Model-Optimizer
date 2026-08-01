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

"""Tests for streaming compact-capture policy selection."""

import inspect
import json
import math
from hashlib import sha256

import pytest

from modelopt.torch.sparsity.attention_sparsity.calibration import mask_reuse_compact
from modelopt.torch.sparsity.attention_sparsity.calibration.checkpoint_manifest import (
    create_checkpoint_manifest,
)
from modelopt.torch.sparsity.attention_sparsity.calibration.mask_reuse import (
    MaskReuseCalibrationError,
    calibrate_mask_reuse_policy,
)
from modelopt.torch.sparsity.attention_sparsity.calibration.mask_reuse_compact import (
    calibrate_compact_mask_reuse_policy,
    load_compact_mask_reuse_captures,
)

_FIT = {
    "threshold_scale_factor": {
        "formula": "a * exp(b * target_sparsity)",
        "prefill": {
            "a": 1.0,
            "b": 1.0,
            "min_observed_sparsity": 0.4,
            "max_observed_sparsity": 0.8,
        },
    }
}
_TOPOLOGY = {"anchors": [0], "nearest": {"0": 0, "1": 0}}


def _checkpoint(tmp_path):
    root = tmp_path / "checkpoint"
    root.mkdir()
    (root / "config.json").write_text("{}\n", encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"weights")
    return create_checkpoint_manifest(root, model="test-model")


def _capture(split, prompt_id, target, checkpoint_sha256):
    threshold_log2 = math.log2(1.0) + target * math.log2(math.e) - math.log2(256)
    if target == 0.5:
        retained = [2, 3]
        anchor_dropped = [0.005, 0.005]
        matrix = [[0.001, 0.001], [0.001, 0.001]]
    else:
        retained = [1, 2]
        anchor_dropped = [0.02, 0.02]
        matrix = (
            [[0.01, 0.03], [0.04, 0.01]]
            if split == "calibration"
            else [[0.015, 0.05], [0.05, 0.015]]
        )
    invocation = {
        "capture_schema_version": 2,
        "model": "test-model",
        "checkpoint_manifest_sha256": checkpoint_sha256,
        "split": split,
        "partition": "development" if split == "calibration" else "outer_test",
        "inner_fold": 0 if split == "calibration" else None,
        "prompt_id": prompt_id,
        "source": f"dataset/{split}",
        "source_group_sha256": sha256(f"group/{prompt_id}".encode()).hexdigest(),
        "source_capture_sha256": sha256(prompt_id.encode()).hexdigest(),
        "min_kv_tokens": 129,
        "max_kv_tokens": 512,
        "target_sparsity_hex": target.hex(),
        "sample_length": 256,
        "threshold_log2_hex": threshold_log2.hex(),
        "threshold_lambda_hex": math.exp2(threshold_log2).hex(),
        "expected_geometry": {"q_tokens": 256, "kv_tokens": 256, "q_start_tokens": 0},
    }
    return {
        "compact_capture_schema_version": 1,
        "invocation": invocation,
        "geometry": invocation["expected_geometry"],
        "global_num_heads": 2,
        "eligible_tiles": 3,
        "anchor_stats_by_layer": {
            "0": {"retained_tiles": retained, "dropped_mass": anchor_dropped}
        },
        "consumer_layers": {"1": {"anchor_layer": 0, "dropped_mass": matrix}},
    }


def _write_captures(path, checkpoint_sha256):
    captures = [
        _capture(split, prompt, target, checkpoint_sha256)
        for split, prompt in (("calibration", "cal-0"), ("heldout", "held-0"))
        for target in (0.5, 0.7)
    ]
    payload = b"".join(
        (
            json.dumps(capture, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
        ).encode()
        for capture in captures
    )
    path.write_bytes(payload)
    return captures, sha256(payload).hexdigest()


def _write_payload(path, captures):
    payload = b"".join(
        (
            json.dumps(capture, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
        ).encode()
        for capture in captures
    )
    path.write_bytes(payload)
    return sha256(payload).hexdigest()


def _evidence(reuse_bundle_sha256):
    fields = (
        "calibration_plan_sha256",
        "family_registry_sha256",
        "vanilla_fit_sha256",
        "reuse_bundle_sha256",
        "grouped_fit_sha256",
        "outer_report_sha256",
    )
    return {
        field: reuse_bundle_sha256
        if field == "reuse_bundle_sha256"
        else sha256(field.encode()).hexdigest()
        for field in fields
    }


def _expanded_rows(captures):
    rows = []
    for capture in captures:
        invocation = capture["invocation"]
        anchors = capture["anchor_stats_by_layer"]
        anchor = anchors["0"]
        matrix = capture["consumer_layers"]["1"]["dropped_mass"]
        for consumer_head in range(2):
            rows.extend(
                {
                    "model": invocation["model"],
                    "min_kv_tokens": invocation["min_kv_tokens"],
                    "max_kv_tokens": invocation["max_kv_tokens"],
                    "target_sparsity": float.fromhex(invocation["target_sparsity_hex"]),
                    "sample_length": invocation["sample_length"],
                    "threshold_lambda": float.fromhex(invocation["threshold_lambda_hex"]),
                    "threshold_log2": float.fromhex(invocation["threshold_log2_hex"]),
                    "q_tokens": 256,
                    "kv_tokens": 256,
                    "q_start_tokens": 0,
                    "split": invocation["split"],
                    "prompt_id": invocation["prompt_id"],
                    "source_capture_sha256": invocation["source_capture_sha256"],
                    "anchor_layer": 0,
                    "consumer_layer": 1,
                    "consumer_head": consumer_head,
                    "donor_head": donor_head,
                    "retained_tiles": anchor["retained_tiles"][donor_head],
                    "eligible_tiles": 3,
                    "anchor_dropped_mass": anchor["dropped_mass"][donor_head],
                    "anchor_stats_by_layer": anchors,
                    "dropped_mass": matrix[consumer_head][donor_head],
                }
                for donor_head in range(2)
            )
    return rows


def test_streaming_compact_selector_matches_legacy_row_policy(tmp_path):
    path = tmp_path / "compact.jsonl"
    checkpoint = _checkpoint(tmp_path)
    captures, digest = _write_captures(path, checkpoint.sha256)
    kwargs = {
        "vanilla_calibration": _FIT,
        "topology": _TOPOLOGY,
        "checkpoint_manifest": checkpoint,
        "evidence": _evidence(digest),
        "max_anchor_dropped_mass": 0.03,
        "max_reuse_dropped_mass": 0.025,
        "max_reuse_selection_dropped_mass": 0.02,
    }

    compact = calibrate_compact_mask_reuse_policy(load_compact_mask_reuse_captures(path), **kwargs)
    legacy = calibrate_mask_reuse_policy(_expanded_rows(captures), **kwargs)

    assert compact["context_policies"] == legacy["context_policies"]
    assert compact["context_policies"][0]["target_sparsity"] == 0.7
    assert compact["context_policies"][0]["headmaps"] == {"1": [0, 1]}
    compact_report = dict(compact["calibration_report"])
    legacy_report = dict(legacy["calibration_report"])
    compact_report.pop("promotion")
    legacy_report.pop("promotion")
    assert compact_report == legacy_report
    assert compact["provenance"]["input_capture_count"] == 4
    assert compact["provenance"]["candidate_cell_count"] == 16
    assert compact["provenance"]["streaming_passes"] == [
        "validation",
        "calibration_selection",
        "frozen_evaluation",
    ]
    assert compact["promotion_status"] == "candidate_only"
    assert compact["deployment_geometry_validated"] is False


def test_compact_selector_binds_reuse_bundle_sha(tmp_path):
    path = tmp_path / "compact.jsonl"
    checkpoint = _checkpoint(tmp_path)
    _, digest = _write_captures(path, checkpoint.sha256)
    evidence = _evidence(digest)
    evidence["reuse_bundle_sha256"] = sha256(b"wrong").hexdigest()

    with pytest.raises(MaskReuseCalibrationError, match="does not match"):
        calibrate_compact_mask_reuse_policy(
            path,
            vanilla_calibration=_FIT,
            topology=_TOPOLOGY,
            checkpoint_manifest=checkpoint,
            evidence=evidence,
            max_anchor_dropped_mass=0.03,
            max_reuse_dropped_mass=0.025,
        )


def test_compact_selector_binds_verified_checkpoint_and_disjoint_groups(tmp_path):
    path = tmp_path / "compact.jsonl"
    checkpoint = _checkpoint(tmp_path)
    captures = [
        _capture(split, prompt, target, checkpoint.sha256)
        for split, prompt in (("calibration", "cal-0"), ("heldout", "held-0"))
        for target in (0.5, 0.7)
    ]
    captures[0]["invocation"]["checkpoint_manifest_sha256"] = "0" * 64
    digest = _write_payload(path, captures)
    with pytest.raises(MaskReuseCalibrationError, match="one model, checkpoint"):
        calibrate_compact_mask_reuse_policy(
            path,
            vanilla_calibration=_FIT,
            topology=_TOPOLOGY,
            checkpoint_manifest=checkpoint,
            evidence=_evidence(digest),
            max_anchor_dropped_mass=0.03,
            max_reuse_dropped_mass=0.025,
        )

    shared_group = captures[1]["invocation"]["source_group_sha256"]
    for capture in captures:
        capture["invocation"]["checkpoint_manifest_sha256"] = checkpoint.sha256
        capture["invocation"]["source_group_sha256"] = shared_group
    digest = _write_payload(path, captures)
    with pytest.raises(MaskReuseCalibrationError, match="multiple partitions"):
        calibrate_compact_mask_reuse_policy(
            path,
            vanilla_calibration=_FIT,
            topology=_TOPOLOGY,
            checkpoint_manifest=checkpoint,
            evidence=_evidence(digest),
            max_anchor_dropped_mass=0.03,
            max_reuse_dropped_mass=0.025,
        )


def test_compact_selector_rejects_file_changed_during_evaluation(tmp_path, monkeypatch):
    path = tmp_path / "compact.jsonl"
    checkpoint = _checkpoint(tmp_path)
    _, digest = _write_captures(path, checkpoint.sha256)
    real_evaluation_pass = mask_reuse_compact._evaluation_pass

    def mutate_after_evaluation(*args, **kwargs):
        result = real_evaluation_pass(*args, **kwargs)
        with path.open("ab") as handle:
            handle.write(b"\n")
        return result

    monkeypatch.setattr(mask_reuse_compact, "_evaluation_pass", mutate_after_evaluation)

    with pytest.raises(MaskReuseCalibrationError, match="changed during calibration"):
        calibrate_compact_mask_reuse_policy(
            path,
            vanilla_calibration=_FIT,
            topology=_TOPOLOGY,
            checkpoint_manifest=checkpoint,
            evidence=_evidence(digest),
            max_anchor_dropped_mass=0.03,
            max_reuse_dropped_mass=0.025,
        )


def test_selector_minimizes_declared_equal_bmm_combined_cost(tmp_path):
    path = tmp_path / "compact.jsonl"
    checkpoint = _checkpoint(tmp_path)
    captures = [
        _capture(split, prompt, target, checkpoint.sha256)
        for split, prompt in (("calibration", "cal-0"), ("heldout", "held-0"))
        for target in (0.5, 0.7)
    ]
    for capture in captures:
        target = float.fromhex(capture["invocation"]["target_sparsity_hex"])
        if target == 0.5:
            capture["anchor_stats_by_layer"]["0"] = {
                "retained_tiles": [1, 3],
                "dropped_mass": [0.001, 0.001],
            }
            capture["anchor_stats_by_layer"]["2"] = {
                "retained_tiles": [3, 3],
                "dropped_mass": [0.001, 0.001],
            }
            capture["consumer_layers"]["1"]["dropped_mass"] = [
                [0.001, 0.001],
                [0.001, 0.001],
            ]
        else:
            capture["anchor_stats_by_layer"]["0"] = {
                "retained_tiles": [0, 2],
                "dropped_mass": [0.002, 0.002],
            }
            capture["anchor_stats_by_layer"]["2"] = {
                "retained_tiles": [0, 0],
                "dropped_mass": [0.002, 0.002],
            }
            capture["consumer_layers"]["1"]["dropped_mass"] = [
                [0.03, 0.01],
                [0.03, 0.01],
            ]
    digest = _write_payload(path, captures)

    candidate = calibrate_compact_mask_reuse_policy(
        path,
        vanilla_calibration=_FIT,
        topology={"anchors": [0, 2], "nearest": {"0": 0, "1": 0, "2": 2}},
        checkpoint_manifest=checkpoint,
        evidence=_evidence(digest),
        max_anchor_dropped_mass=0.03,
        max_reuse_dropped_mass=0.025,
        max_reuse_selection_dropped_mass=0.02,
    )

    assert candidate["context_policies"][0]["target_sparsity"] == 0.7
    frontier = candidate["calibration_report"]["by_bucket"][0]["target_sparsity_frontier"]
    assert [row["combined_tile_cost"] for row in frontier] == [14, 10]


def test_validation_pass_does_not_retain_capture_objects():
    implementation = inspect.getsource(mask_reuse_compact._validate_dataset)

    assert ".append(capture)" not in implementation
    assert "list[CompactMaskReuseCapture]" not in implementation


@pytest.mark.parametrize("corruption", ["eligible", "retained_monotonic", "mass_monotonic"])
def test_compact_validation_rejects_impossible_geometry_or_sparsity_trend(tmp_path, corruption):
    path = tmp_path / "compact.jsonl"
    checkpoint = _checkpoint(tmp_path)
    captures = [
        _capture(split, prompt, target, checkpoint.sha256)
        for split, prompt in (("calibration", "cal-0"), ("heldout", "held-0"))
        for target in (0.5, 0.7)
    ]
    if corruption == "eligible":
        captures[0]["eligible_tiles"] = 4
    elif corruption == "retained_monotonic":
        captures[1]["anchor_stats_by_layer"]["0"]["retained_tiles"][0] = 3
    else:
        captures[1]["consumer_layers"]["1"]["dropped_mass"][0][0] = 0.0
    digest = _write_payload(path, captures)

    with pytest.raises(MaskReuseCalibrationError):
        calibrate_compact_mask_reuse_policy(
            path,
            vanilla_calibration=_FIT,
            topology=_TOPOLOGY,
            checkpoint_manifest=checkpoint,
            evidence=_evidence(digest),
            max_anchor_dropped_mass=0.03,
            max_reuse_dropped_mass=0.025,
        )
