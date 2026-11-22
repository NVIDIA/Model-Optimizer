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

import pulp
import pytest

from modelopt.torch.sparsity.attention_sparsity.calibration import (
    assess_mask_reuse_benchmark_gates,
    mask_reuse_compact,
)
from modelopt.torch.sparsity.attention_sparsity.calibration.checkpoint_manifest import (
    create_checkpoint_manifest,
)
from modelopt.torch.sparsity.attention_sparsity.calibration.mask_reuse import (
    MaskReuseCalibrationError,
)
from modelopt.torch.sparsity.attention_sparsity.calibration.mask_reuse_compact import (
    calibrate_compact_mask_reuse_policy,
)
from modelopt.torch.sparsity.attention_sparsity.calibration.mask_reuse_solver import (
    MaskReuseSolverConfig,
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


@pytest.mark.parametrize("backend", ["highs"])
def test_anchor_inclusive_objective_and_legacy_cap_are_explicit_variants(tmp_path, backend):
    if backend == "highs" and not pulp.HiGHS().available():
        pytest.skip("optional highspy solver is not installed")
    checkpoint = _checkpoint(tmp_path)
    path = tmp_path / "compact.jsonl"
    captures, _ = _write_captures(path, checkpoint.sha256)
    for capture in captures:
        target = float.fromhex(capture["invocation"]["target_sparsity_hex"])
        capture["anchor_stats_by_layer"]["0"]["dropped_mass"] = [0.4 if target == 0.5 else 0.41] * 2
    payload = b"".join(
        (json.dumps(c, sort_keys=True, separators=(",", ":")) + "\n").encode() for c in captures
    )
    path.write_bytes(payload)
    kwargs = {
        "vanilla_calibration": _FIT,
        "topology": _TOPOLOGY,
        "checkpoint_manifest": checkpoint,
        "evidence": _evidence(sha256(payload).hexdigest()),
        "reuse_dropped_mass_report_threshold": 0.02,
        "target_bmm1_skip_ratio": 0.1,
        "solver_config": MaskReuseSolverConfig(backend=backend),
    }
    current = calibrate_compact_mask_reuse_policy(
        path, anchor_dropped_mass_report_threshold=0.02, **kwargs
    )
    repeated = calibrate_compact_mask_reuse_policy(
        path, anchor_dropped_mass_report_threshold=0.02, **kwargs
    )
    assert json.dumps(current, sort_keys=True) == json.dumps(repeated, sort_keys=True)
    permissive_report = calibrate_compact_mask_reuse_policy(
        path, anchor_dropped_mass_report_threshold=0.9, **kwargs
    )
    legacy = calibrate_compact_mask_reuse_policy(path, max_anchor_dropped_mass=0.03, **kwargs)
    assert current["objective_variant"] == "anchor_inclusive_v1"
    assert current["context_policies"] == permissive_report["context_policies"]
    assert current["calibration_report"]["by_bucket"][0]["risk_selection_objective"][
        "worst_development_prompt_model_wide_total_dropped_mass"
    ] == pytest.approx((0.8 + 0.002) / 4)
    assert legacy["objective_variant"] == "legacy_anchor_cap_v1"
    assert legacy["context_policies"][0]["exact"] is True
    assert (
        current["calibration_report"]["overall"]["anchor_calibration"][
            "report_threshold_exceedance_count"
        ]
        == 1
    )
    with pytest.raises(MaskReuseCalibrationError, match="choose the legacy"):
        calibrate_compact_mask_reuse_policy(
            path, max_anchor_dropped_mass=0.03, anchor_dropped_mass_report_threshold=0.02, **kwargs
        )


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
        "threshold_lambda_hex": (2.0**threshold_log2).hex(),
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


def test_compact_public_interface_has_no_hard_reuse_risk_gate():
    parameters = inspect.signature(calibrate_compact_mask_reuse_policy).parameters

    assert parameters["target_bmm1_skip_ratio"].default is inspect.Parameter.empty
    assert "max_reuse_selection_dropped_mass" not in parameters


def test_reference_page_binding_changes_only_runtime_geometry(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    path = tmp_path / "compact.jsonl"
    _, digest = _write_captures(path, checkpoint.sha256)
    kwargs = {
        "vanilla_calibration": _FIT,
        "topology": _TOPOLOGY,
        "checkpoint_manifest": checkpoint,
        "evidence": _evidence(digest),
        "reuse_dropped_mass_report_threshold": 0.025,
        "target_bmm1_skip_ratio": 0.25,
    }
    baseline = calibrate_compact_mask_reuse_policy(path, **kwargs)
    provider = {
        "provider_kind": "triton-reference",
        "evidence_level": "accuracy-reference-only",
        "observed_kv_page_tokens": 4160,
    }
    actual = calibrate_compact_mask_reuse_policy(
        path, **kwargs, source_provenance={"capture_provider": provider}
    )
    assert baseline["deployment_geometry"]["contract"]["kv_page_tokens"] == 16
    assert actual["deployment_geometry"]["contract"]["kv_page_tokens"] == 4160
    assert actual["context_policies"] == baseline["context_policies"]
    assert actual["calibration_report"] == baseline["calibration_report"]
    provider["provider_kind"] = "fa4"
    with pytest.raises(ValueError, match="explicit reference"):
        calibrate_compact_mask_reuse_policy(
            path, **kwargs, source_provenance={"capture_provider": provider}
        )


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
            reuse_dropped_mass_report_threshold=0.025,
            target_bmm1_skip_ratio=0.25,
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
            reuse_dropped_mass_report_threshold=0.025,
            target_bmm1_skip_ratio=0.25,
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
            reuse_dropped_mass_report_threshold=0.025,
            target_bmm1_skip_ratio=0.25,
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
            reuse_dropped_mass_report_threshold=0.025,
            target_bmm1_skip_ratio=0.25,
        )


def test_selector_meets_bmm1_target_before_minimizing_reuse_risk(tmp_path):
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
        reuse_dropped_mass_report_threshold=0.025,
        target_bmm1_skip_ratio=0.25,
    )

    assert candidate["context_policies"][0]["target_sparsity"] == 0.7
    frontier = candidate["calibration_report"]["by_bucket"][0]["target_sparsity_frontier"]
    assert [row["target_bmm1_skip_ratio_feasible"] for row in frontier] == [False, True]
    assert frontier[1]["combined_tile_cost"] == 2


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
            reuse_dropped_mass_report_threshold=0.025,
            target_bmm1_skip_ratio=0.25,
        )


@pytest.mark.parametrize("delta", [1e-17, 1e-13, 1e-10])
def test_refit_accepts_close_donor_risks(tmp_path, delta):
    checkpoint = _checkpoint(tmp_path)
    path = tmp_path / "near.jsonl"
    captures = [
        _capture(split, prompt, 0.5, checkpoint.sha256)
        for split, prompt in (("calibration", "cal-0"), ("heldout", "held-0"))
    ]
    for capture in captures:
        capture["anchor_stats_by_layer"]["0"]["retained_tiles"] = [2, 1]
        capture["consumer_layers"]["1"]["dropped_mass"] = [[0.01, 0.01 + delta]] * 2
    digest = _write_payload(path, captures)
    candidate = calibrate_compact_mask_reuse_policy(
        path,
        vanilla_calibration=_FIT,
        topology=_TOPOLOGY,
        checkpoint_manifest=checkpoint,
        evidence=_evidence(digest),
        reuse_dropped_mass_report_threshold=0.02,
        target_bmm1_skip_ratio=0.1,
    )
    assert candidate["calibration_report"]["by_bucket"][0]["bmm1_skip_objective"]["target_met"]


def test_cost_pass_spends_one_shared_risk_band_across_targets(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    path = tmp_path / "capture.jsonl"
    captures, _ = _write_captures(path, checkpoint.sha256)
    for row in captures:
        expensive = float.fromhex(row["invocation"]["target_sparsity_hex"]) == 0.5
        row["anchor_stats_by_layer"]["0"]["dropped_mass"] = [0.0 if expensive else 8e-8] * 2
        row["consumer_layers"]["1"]["dropped_mass"] = [[0.0 if expensive else 1.6e-7] * 2] * 2
    result = calibrate_compact_mask_reuse_policy(
        path,
        vanilla_calibration=_FIT,
        topology=_TOPOLOGY,
        checkpoint_manifest=checkpoint,
        evidence=_evidence(_write_payload(path, captures)),
        reuse_dropped_mass_report_threshold=0.02,
        target_bmm1_skip_ratio=0.05,
    )
    row = result["calibration_report"]["by_bucket"][0]
    chosen = next(
        t
        for t in row["target_sparsity_frontier"]
        if t["target_sparsity"] == row["selected_target_sparsity"]
    )
    # The 0.5 target costs 13. At 0.7 one reused head costs 11;
    # reusing both costs 7 but exceeds the single shared ceiling.
    assert (chosen["target_sparsity"], chosen["combined_tile_cost"]) == (0.7, 11)
    assert row["fallback_head_count"] == 1


def test_benchmark_replays_sub_ulp_winner_and_gates_quality(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    path = tmp_path / "capture.jsonl"
    captures, _ = _write_captures(path, checkpoint.sha256)
    # Both targets qualify under the same risk ceiling. The cheaper target wins.
    for row in captures:
        target = float.fromhex(row["invocation"]["target_sparsity_hex"])
        row["anchor_stats_by_layer"]["0"]["dropped_mass"] = [0.5, 0.5]
        diff = 2**-60 if target == 0.5 else 2**-59
        row["consumer_layers"]["1"]["dropped_mass"] = [[diff, diff], [diff, diff]]
    kwargs = {
        "vanilla_calibration": _FIT,
        "topology": _TOPOLOGY,
        "checkpoint_manifest": checkpoint,
        "evidence": _evidence(_write_payload(path, captures)),
        "target_bmm1_skip_ratio": 0.05,
        "reuse_dropped_mass_report_threshold": 0.02,
        "source_provenance": {"capture_scope": "whole_prefill"},
    }
    candidate = calibrate_compact_mask_reuse_policy(path, **kwargs)
    assert candidate["context_policies"][0]["target_sparsity"] == 0.7
    assert candidate["context_policies"][0]["fallback_heads"] == {"1": []}
    proof = candidate["calibration_report"]["by_bucket"][0]["selection_result"]
    assert proof["status"] == "within_tolerance"
    assert proof["worst_prompt_risk_exact"] == f"{2**58 + 1}/{2**60}"
    candidate_bytes = json.dumps(candidate).encode()
    digest = sha256(candidate_bytes).hexdigest()
    common = {"evaluation_sha256": "a" * 64, "checkpoint_manifest_sha256": checkpoint.sha256}
    pair = {
        "candidate": dict(common, accuracy_percent=79.1, policy_sha256=digest),
        "dense": dict(common, accuracy_percent=80.1, attention_mode="dense"),
    }
    gate = {
        "expected_policy_sha256": digest,
        "search_scope": proof["scope"],
        "required_benchmarks": ["AA-LCR"],
        "matched_scores": {"AA-LCR": pair},
    }
    assert assess_mask_reuse_benchmark_gates(candidate_bytes, **gate)["selection_and_accuracy_pass"]
    assert not assess_mask_reuse_benchmark_gates(
        candidate_bytes, **(gate | {"search_scope": "joint_topology_and_donors"})
    )["selection_and_accuracy_pass"]
    assert not assess_mask_reuse_benchmark_gates(
        candidate_bytes, **(gate | {"required_benchmarks": ["AA-LCR", "HLE"]})
    )["selection_and_accuracy_pass"]
    pair["candidate"]["accuracy_percent"] = 79.099
    assert not assess_mask_reuse_benchmark_gates(candidate_bytes, **gate)[
        "selection_and_accuracy_pass"
    ]
    pair["candidate"]["policy_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="unmatched"):
        assess_mask_reuse_benchmark_gates(candidate_bytes, **gate)
    bad_prior = {**candidate, "checkpoint_manifest_sha256": "c" * 64}
    with pytest.raises(MaskReuseCalibrationError, match="prior candidate"):
        calibrate_compact_mask_reuse_policy(path, previous_candidate=bad_prior, **kwargs)
    unmet = calibrate_compact_mask_reuse_policy(path, **(kwargs | {"target_bmm1_skip_ratio": 1}))
    assert not unmet["calibration_report"]["overall"]["bmm1_skip_objective"][
        "all_context_buckets_met"
    ]
