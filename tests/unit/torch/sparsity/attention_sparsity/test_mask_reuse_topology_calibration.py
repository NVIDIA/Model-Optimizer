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

"""Tests for development-only mask-reuse topology discovery."""

import inspect
import json
import math
from hashlib import sha256

import pytest

from modelopt.torch.sparsity.attention_sparsity.calibration.checkpoint_manifest import (
    create_checkpoint_manifest,
)
from modelopt.torch.sparsity.attention_sparsity.calibration.mask_reuse import (
    MaskReuseCalibrationError,
)
from modelopt.torch.sparsity.attention_sparsity.calibration.mask_reuse_topology import (
    calibrate_mask_reuse_topology,
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


def _checkpoint(tmp_path):
    root = tmp_path / "checkpoint"
    root.mkdir(parents=True)
    (root / "config.json").write_text("{}\n", encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"weights")
    return create_checkpoint_manifest(root, model="test-model")


def _matrix(value, *, heads=2):
    return [[value for _ in range(heads)] for _ in range(heads)]


def _capture(split, prompt_id, target, checkpoint_sha256, *, hostile_heldout=False):
    threshold_log2 = target * math.log2(math.e) - math.log2(256)
    retained = {
        0: [2, 2] if target == 0.5 else [1, 1],
        1: [3, 3] if target == 0.5 else [2, 2],
        2: [2, 2] if target == 0.5 else [1, 1],
        3: [3, 3] if target == 0.5 else [2, 2],
    }
    anchor_risk = 0.005 if target == 0.5 else 0.01
    safe = 0.005 if target == 0.5 else 0.01
    unsafe = 0.05
    candidates = {
        1: {0: _matrix(safe)},
        2: {0: _matrix(unsafe), 1: _matrix(unsafe)},
        3: {0: _matrix(unsafe), 1: _matrix(unsafe), 2: _matrix(safe)},
    }
    if split == "heldout" and hostile_heldout:
        # A held-out-only reversal must affect evaluation, never the selected
        # topology, target, or donor map.
        candidates[1][0] = _matrix(0.9)
        candidates[3][2] = _matrix(0.9)
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
        "threshold_lambda_hex": (2.0**threshold_log2).hex(),
        "expected_geometry": {"q_tokens": 256, "kv_tokens": 256, "q_start_tokens": 0},
    }
    return {
        "topology_discovery_capture_schema_version": 1,
        "invocation": invocation,
        "geometry": invocation["expected_geometry"],
        "global_num_heads": 2,
        "eligible_tiles": 3,
        "attention_layers": [0, 1, 2, 3],
        "max_reuse_span": 3,
        "anchor_stats_by_layer": {
            str(layer): {
                "retained_tiles": values,
                "dropped_mass": [anchor_risk, anchor_risk],
            }
            for layer, values in retained.items()
        },
        "consumer_candidates_by_layer": {
            str(consumer): {
                str(anchor): {"dropped_mass": matrix} for anchor, matrix in anchors.items()
            }
            for consumer, anchors in candidates.items()
        },
    }


def _write(path, captures):
    payload = b"".join(
        (
            json.dumps(capture, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
        ).encode()
        for capture in captures
    )
    path.write_bytes(payload)
    return sha256(payload).hexdigest()


def _evidence(capture_sha256):
    return {
        "topology_discovery_capture_sha256": capture_sha256,
        "vanilla_fit_sha256": sha256(b"fit").hexdigest(),
        "prompt_plan_sha256": sha256(b"prompts").hexdigest(),
    }


def _run(
    tmp_path,
    *,
    hostile_heldout=False,
    target_bmm1_skip_ratio=0.3,
    reuse_dropped_mass_report_threshold=0.025,
):
    checkpoint = _checkpoint(tmp_path)
    path = tmp_path / "topology.jsonl"
    captures = [
        _capture(
            split,
            prompt,
            target,
            checkpoint.sha256,
            hostile_heldout=hostile_heldout,
        )
        for split, prompt in (("calibration", "cal-0"), ("heldout", "held-0"))
        for target in (0.5, 0.7)
    ]
    digest = _write(path, captures)
    result = calibrate_mask_reuse_topology(
        path,
        vanilla_calibration=_FIT,
        checkpoint_manifest=checkpoint,
        evidence=_evidence(digest),
        max_anchor_dropped_mass=0.03,
        reuse_dropped_mass_report_threshold=reuse_dropped_mass_report_threshold,
        target_bmm1_skip_ratio=target_bmm1_skip_ratio,
    )
    return result, captures, path, checkpoint


def test_public_interface_has_one_required_performance_knob():
    parameters = inspect.signature(calibrate_mask_reuse_topology).parameters

    assert parameters["target_bmm1_skip_ratio"].default is inspect.Parameter.empty
    assert parameters["reuse_dropped_mass_report_threshold"].default is inspect.Parameter.empty
    assert "max_reuse_selection_dropped_mass" not in parameters


def test_joint_selector_recovers_anchor_placement_target_and_consumers(tmp_path):
    result, _, _, _ = _run(tmp_path)

    assert result["anchors"] == [0, 2]
    assert result["nearest"] == {"0": 0, "1": 0, "2": 2, "3": 2}
    policy = result["discovery_context_policies"][0]
    assert policy["target_sparsity"] == 0.7
    assert policy["headmaps"] == {"1": [0, 0], "3": [0, 0]}
    assert policy["fallback_heads"] == {"1": [], "3": []}
    assert result["provenance"]["selection_split"] == "calibration"
    assert result["provenance"]["evaluation_split"] == "heldout"
    assert result["promotion_status"] == "topology_candidate_only"
    assert result["target_bmm1_skip_ratio"] == 0.3
    assert result["bmm1_skip_objective"]["enabled"] is True


def test_target_bmm1_skip_ratio_is_a_model_wide_calibration_objective(tmp_path):
    result, _, _, _ = _run(tmp_path, target_bmm1_skip_ratio=0.3)

    objective = result["bmm1_skip_objective"]
    assert objective == {
        "enabled": True,
        "metric": "model_wide_eligible_bmm1_tile_skip_ratio",
        "selection_split": "calibration",
        "comparison": ">=",
        "target": 0.3,
        "target_met": True,
        "calibration_eligible_tiles": 24,
        "minimum_required_skipped_tiles": 8,
        "calibration_skipped_tiles": 8,
        "achieved": pytest.approx(1 / 3),
        "maximum_feasible": None,
    }
    assert result["calibration_report"]["selection_status"] == ("target_bmm1_skip_ratio_met")
    assert result["calibration_report"]["overall"]["bmm1_calibration"] == {
        "eligible_tiles": 24,
        "skipped_tiles": 8,
        "model_wide_bmm1_tile_skip_ratio": pytest.approx(1 / 3),
    }


def test_unmet_target_returns_the_structural_maximum_policy(tmp_path):
    result, _, _, _ = _run(tmp_path, target_bmm1_skip_ratio=0.8)

    objective = result["bmm1_skip_objective"]
    assert objective["target_met"] is False
    assert objective["achieved"] == objective["maximum_feasible"]
    assert objective["achieved"] < 0.8
    assert result["calibration_report"]["selection_status"] == (
        "target_bmm1_skip_ratio_unmet_maximum_feasible"
    )


def test_bmm1_objective_accounts_for_each_calibration_prompt_once(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    path = tmp_path / "topology.jsonl"
    captures = [
        _capture(split, prompt, target, checkpoint.sha256)
        for split, prompt in (
            ("calibration", "cal-0"),
            ("calibration", "cal-1"),
            ("heldout", "held-0"),
        )
        for target in (0.5, 0.7)
    ]
    digest = _write(path, captures)

    result = calibrate_mask_reuse_topology(
        path,
        vanilla_calibration=_FIT,
        checkpoint_manifest=checkpoint,
        evidence=_evidence(digest),
        max_anchor_dropped_mass=0.03,
        reuse_dropped_mass_report_threshold=0.025,
        target_bmm1_skip_ratio=0.3,
    )

    assert result["bmm1_skip_objective"]["calibration_eligible_tiles"] == 48
    assert result["bmm1_skip_objective"]["calibration_skipped_tiles"] == 16
    assert result["bmm1_skip_objective"]["achieved"] == pytest.approx(1 / 3)
    assert result["calibration_report"]["overall"]["bmm1_calibration"] == {
        "eligible_tiles": 48,
        "skipped_tiles": 16,
        "model_wide_bmm1_tile_skip_ratio": pytest.approx(1 / 3),
    }


@pytest.mark.parametrize("invalid", [True, -0.1, 1.1, float("nan")])
def test_target_bmm1_skip_ratio_rejects_invalid_values(tmp_path, invalid):
    with pytest.raises(MaskReuseCalibrationError, match="target_bmm1_skip_ratio"):
        _run(tmp_path, target_bmm1_skip_ratio=invalid)


def test_target_bmm1_skip_ratio_can_override_equal_bmm_cost(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    path = tmp_path / "topology.jsonl"
    captures = [
        _capture(split, prompt, target, checkpoint.sha256)
        for split, prompt in (("calibration", "cal-0"), ("heldout", "held-0"))
        for target in (0.5, 0.7)
    ]
    for capture in captures:
        # Reusing layers 1 and 3 retains two of three tiles per head. Making
        # those layers anchors instead retains zero BMM2 tiles, so the ordinary
        # equal-BMM objective prefers four anchors. A 15% model-wide BMM1 target
        # requires both safe reuse edges: 4 skipped / 24 eligible = 1/6.
        capture["anchor_stats_by_layer"]["0"]["retained_tiles"] = [2, 2]
        capture["anchor_stats_by_layer"]["1"]["retained_tiles"] = [0, 0]
        capture["anchor_stats_by_layer"]["2"]["retained_tiles"] = [2, 2]
        capture["anchor_stats_by_layer"]["3"]["retained_tiles"] = [0, 0]
    digest = _write(path, captures)

    def calibrate(target_bmm1_skip_ratio):
        return calibrate_mask_reuse_topology(
            path,
            vanilla_calibration=_FIT,
            checkpoint_manifest=checkpoint,
            evidence=_evidence(digest),
            max_anchor_dropped_mass=0.03,
            reuse_dropped_mass_report_threshold=0.025,
            target_bmm1_skip_ratio=target_bmm1_skip_ratio,
        )

    baseline = calibrate(0.0)
    targeted = calibrate(0.15)

    assert baseline["anchors"] == [0, 1, 2, 3]
    assert targeted["anchors"] == [0, 2]
    assert targeted["bmm1_skip_objective"]["achieved"] == pytest.approx(1 / 6)


def test_heldout_statistics_cannot_retune_frozen_topology(tmp_path):
    baseline, _, _, _ = _run(tmp_path / "baseline")
    hostile, _, _, _ = _run(tmp_path / "hostile", hostile_heldout=True)

    keys = ("anchors", "nearest", "discovery_context_policies")
    assert {key: baseline[key] for key in keys} == {key: hostile[key] for key in keys}
    assert (
        hostile["calibration_report"]["overall"]["reuse_heldout"][
            "report_threshold_exceedance_count"
        ]
        == 4
    )


def test_reporting_threshold_cannot_retune_frozen_policy(tmp_path):
    strict, _, _, _ = _run(
        tmp_path / "strict",
        reuse_dropped_mass_report_threshold=0.001,
    )
    permissive, _, _, _ = _run(
        tmp_path / "permissive",
        reuse_dropped_mass_report_threshold=0.9,
    )

    keys = ("anchors", "nearest", "discovery_context_policies")
    assert {key: strict[key] for key in keys} == {key: permissive[key] for key in keys}
    assert strict["constraints"]["reuse"]["selection_hard_maximum"] is None
    assert strict["constraints"]["reuse"]["report_threshold"] == 0.001
    assert permissive["constraints"]["reuse"]["report_threshold"] == 0.9
    assert (
        strict["calibration_report"]["overall"]["reuse_heldout"][
            "report_threshold_exceedance_count"
        ]
        > permissive["calibration_report"]["overall"]["reuse_heldout"][
            "report_threshold_exceedance_count"
        ]
    )


def test_selector_uses_exact_fallback_for_unneeded_higher_risk_heads(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    path = tmp_path / "topology.jsonl"
    captures = [
        _capture(split, prompt, target, checkpoint.sha256)
        for split, prompt in (("calibration", "cal-0"), ("heldout", "held-0"))
        for target in (0.5, 0.7)
    ]
    for capture in captures:
        target = float.fromhex(capture["invocation"]["target_sparsity_hex"])
        if target == 0.5:
            for stats in capture["anchor_stats_by_layer"].values():
                stats["retained_tiles"] = [3, 3]
        for candidates in capture["consumer_candidates_by_layer"].values():
            for stats in candidates.values():
                stats["dropped_mass"] = _matrix(0.0001 if target == 0.5 else 0.8)
        if target == 0.7:
            capture["consumer_candidates_by_layer"]["1"]["0"]["dropped_mass"] = [
                [0.001, 0.001],
                [0.1, 0.1],
            ]
            capture["consumer_candidates_by_layer"]["3"]["2"]["dropped_mass"] = [
                [0.002, 0.002],
                [0.2, 0.2],
            ]
    digest = _write(path, captures)

    result = calibrate_mask_reuse_topology(
        path,
        vanilla_calibration=_FIT,
        checkpoint_manifest=checkpoint,
        evidence=_evidence(digest),
        max_anchor_dropped_mass=0.03,
        reuse_dropped_mass_report_threshold=0.025,
        target_bmm1_skip_ratio=0.1,
    )

    policy = result["discovery_context_policies"][0]
    assert result["anchors"] == [0, 2]
    assert policy["fallback_heads"] == {"1": [1], "3": [1]}
    assert result["bmm1_skip_objective"]["achieved"] == pytest.approx(1 / 6)
    objective = result["reuse_selection_objective"]
    assert objective["hard_maximum"] is None
    assert objective["optimization"] == ["minimum_worst_prompt"]
    assert objective["worst_development_prompt_model_wide_dropped_mass"] == pytest.approx(0.003 / 8)


def test_selector_has_no_arbitrary_reuse_risk_rejection_threshold(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    path = tmp_path / "topology.jsonl"
    captures = [
        _capture(split, prompt, target, checkpoint.sha256)
        for split, prompt in (("calibration", "cal-0"), ("heldout", "held-0"))
        for target in (0.5, 0.7)
    ]
    for capture in captures:
        for candidates in capture["consumer_candidates_by_layer"].values():
            for stats in candidates.values():
                stats["dropped_mass"] = _matrix(0.2)
    digest = _write(path, captures)

    result = calibrate_mask_reuse_topology(
        path,
        vanilla_calibration=_FIT,
        checkpoint_manifest=checkpoint,
        evidence=_evidence(digest),
        max_anchor_dropped_mass=0.03,
        reuse_dropped_mass_report_threshold=0.025,
        target_bmm1_skip_ratio=0.3,
    )

    assert result["bmm1_skip_objective"]["target_met"] is True
    assert result["calibration_report"]["fallback_head_bucket_count"] == 0
    assert result["reuse_selection_objective"]["worst_individual_dropped_mass"] == 0.2


def test_anchor_constraint_matches_fixed_topology_prompt_mean(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    path = tmp_path / "topology.jsonl"
    captures = [
        _capture(split, prompt, target, checkpoint.sha256)
        for split, prompt in (("calibration", "cal-0"), ("heldout", "held-0"))
        for target in (0.5, 0.7)
    ]
    for capture in captures:
        if float.fromhex(capture["invocation"]["target_sparsity_hex"]) == 0.7:
            # Layer 2 is individually above the 0.03 bound, while the selected
            # anchors [0, 2] have prompt mean (0.01 + 0.04) / 2 = 0.025.
            capture["anchor_stats_by_layer"]["2"]["dropped_mass"] = [0.04, 0.04]
    digest = _write(path, captures)

    result = calibrate_mask_reuse_topology(
        path,
        vanilla_calibration=_FIT,
        checkpoint_manifest=checkpoint,
        evidence=_evidence(digest),
        max_anchor_dropped_mass=0.03,
        reuse_dropped_mass_report_threshold=0.025,
        target_bmm1_skip_ratio=0.3,
    )

    assert result["anchors"] == [0, 2]
    assert result["calibration_report"]["overall"]["anchor_calibration"][
        "worst_prompt_mean_anchor_dropped_mass"
    ] == pytest.approx(0.025)
    assert result["constraints"]["anchor"]["metric"] == (
        "per_prompt_mean_across_selected_anchor_layers_and_heads_dropped_mass"
    )


def test_selector_rejects_missing_candidate_edge(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    path = tmp_path / "topology.jsonl"
    captures = [
        _capture(split, prompt, target, checkpoint.sha256)
        for split, prompt in (("calibration", "cal-0"), ("heldout", "held-0"))
        for target in (0.5, 0.7)
    ]
    del captures[0]["consumer_candidates_by_layer"]["3"]["1"]
    digest = _write(path, captures)

    with pytest.raises(MaskReuseCalibrationError, match="every earlier attention layer"):
        calibrate_mask_reuse_topology(
            path,
            vanilla_calibration=_FIT,
            checkpoint_manifest=checkpoint,
            evidence=_evidence(digest),
            max_anchor_dropped_mass=0.03,
            reuse_dropped_mass_report_threshold=0.025,
            target_bmm1_skip_ratio=0.3,
        )


def test_selector_rejects_development_outer_group_overlap(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    path = tmp_path / "topology.jsonl"
    captures = [
        _capture(split, prompt, target, checkpoint.sha256)
        for split, prompt in (("calibration", "cal-0"), ("heldout", "held-0"))
        for target in (0.5, 0.7)
    ]
    shared = captures[0]["invocation"]["source_group_sha256"]
    for capture in captures:
        capture["invocation"]["source_group_sha256"] = shared
    digest = _write(path, captures)

    with pytest.raises(MaskReuseCalibrationError, match="multiple partitions"):
        calibrate_mask_reuse_topology(
            path,
            vanilla_calibration=_FIT,
            checkpoint_manifest=checkpoint,
            evidence=_evidence(digest),
            max_anchor_dropped_mass=0.03,
            reuse_dropped_mass_report_threshold=0.025,
            target_bmm1_skip_ratio=0.3,
        )


def test_selector_is_deterministic_under_exact_ties(tmp_path):
    result, captures, path, checkpoint = _run(tmp_path)
    for capture in captures:
        # Both donor heads have identical risk and retained cost. Canonical
        # tie-breaking must keep the lowest donor index.
        for candidates in capture["consumer_candidates_by_layer"].values():
            for stats in candidates.values():
                stats["dropped_mass"] = _matrix(stats["dropped_mass"][0][0])
    digest = _write(path, captures)
    tied = calibrate_mask_reuse_topology(
        path,
        vanilla_calibration=_FIT,
        checkpoint_manifest=checkpoint,
        evidence=_evidence(digest),
        max_anchor_dropped_mass=0.03,
        reuse_dropped_mass_report_threshold=0.025,
        target_bmm1_skip_ratio=0.3,
    )

    assert result["anchors"] == tied["anchors"]
    assert all(
        donor == 0
        for donors in tied["discovery_context_policies"][0]["headmaps"].values()
        for donor in donors
    )


def test_candidate_discovery_accepts_historical_fit_without_observed_bounds(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    path = tmp_path / "topology.jsonl"
    captures = [
        _capture(split, prompt, target, checkpoint.sha256)
        for split, prompt in (("calibration", "cal-0"), ("heldout", "held-0"))
        for target in (0.5, 0.7)
    ]
    digest = _write(path, captures)
    fit = json.loads(json.dumps(_FIT))
    del fit["threshold_scale_factor"]["prefill"]["min_observed_sparsity"]
    del fit["threshold_scale_factor"]["prefill"]["max_observed_sparsity"]

    result = calibrate_mask_reuse_topology(
        path,
        vanilla_calibration=fit,
        checkpoint_manifest=checkpoint,
        evidence=_evidence(digest),
        max_anchor_dropped_mass=0.03,
        reuse_dropped_mass_report_threshold=0.025,
        target_bmm1_skip_ratio=0.3,
    )

    assert result["provenance"]["vanilla_fit_bounds_available"] is False
    assert result["promotion_status"] == "topology_candidate_only"
