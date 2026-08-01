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

"""Deterministic tests for ModelOpt-owned mask-reuse calibration."""

import json
import math
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest

import modelopt
from modelopt.torch.sparsity.attention_sparsity.calibration import (
    AnchorLayerStats,
    MaskReuseCalibrationError,
    MaskReuseObservation,
    calibrate_mask_reuse_policy,
    canonical_prefill_threshold_scale_factor,
    parse_mask_reuse_observations,
)
from modelopt.torch.sparsity.attention_sparsity.calibration.checkpoint_manifest import (
    VerifiedCheckpointManifest,
)

A = 14.47
B = 10.91
VANILLA_FIT = {
    "prefill": {
        "a": A,
        "b": B,
        "min_observed_sparsity": 0.4,
        "max_observed_sparsity": 0.8,
    }
}
TOPOLOGY = {"anchors": [0, 2], "nearest": {"0": 0, "1": 0, "2": 2}}
CHECKPOINT = sha256(b"checkpoint").hexdigest()
VERIFIED_CHECKPOINT = VerifiedCheckpointManifest(
    checkpoint_root=Path("/verified-checkpoint"),
    manifest_path=Path("/verified-checkpoint/checkpoint_manifest.json"),
    model="toy",
    sha256=CHECKPOINT,
    file_count=2,
    total_size_bytes=1,
)
EVIDENCE = {
    field: sha256(field.encode()).hexdigest()
    for field in (
        "calibration_plan_sha256",
        "family_registry_sha256",
        "vanilla_fit_sha256",
        "reuse_bundle_sha256",
        "grouped_fit_sha256",
        "outer_report_sha256",
    )
}


def _source(prompt: str) -> str:
    return sha256(prompt.encode()).hexdigest()


def _thresholds(target_sparsity: float, sample_length: int) -> tuple[float, float]:
    threshold_log2 = (
        math.log2(A) + B * target_sparsity * math.log2(math.e) - math.log2(sample_length)
    )
    return math.exp2(threshold_log2), threshold_log2


def _observations() -> list[MaskReuseObservation]:
    rows = []
    prompts = {
        "calibration": (("cal-0", 65_536), ("cal-1", 98_304)),
        "heldout": (("held-0", 65_536), ("held-1", 98_304)),
    }
    for split, samples in prompts.items():
        for prompt, sample_length in samples:
            for target_sparsity in (0.5, 0.7):
                threshold, threshold_log2 = _thresholds(target_sparsity, sample_length)
                anchor_retained = {
                    0.5: (80, 90),
                    0.7: (40, 20),
                }[target_sparsity]
                anchor_dropped = (0.03, 0.03) if target_sparsity == 0.5 else (0.07, 0.08)
                anchor_stats = {
                    0: AnchorLayerStats(anchor_retained, anchor_dropped),
                    2: AnchorLayerStats(
                        (60, 60) if target_sparsity == 0.5 else (30, 30),
                        (0.02, 0.02),
                    ),
                }
                for consumer_head in range(2):
                    for donor_head in range(2):
                        retained = anchor_retained[donor_head]
                        if target_sparsity == 0.7 and consumer_head == 1:
                            dropped_mass = 0.07 + 0.01 * donor_head
                        else:
                            dropped_mass = 0.02 + 0.01 * donor_head
                        rows.append(
                            MaskReuseObservation(
                                model="toy",
                                min_kv_tokens=65_536,
                                max_kv_tokens=131_072,
                                target_sparsity=target_sparsity,
                                sample_length=sample_length,
                                threshold_lambda=threshold,
                                threshold_log2=threshold_log2,
                                q_tokens=8192,
                                kv_tokens=sample_length,
                                q_start_tokens=sample_length - 8192,
                                split=split,
                                prompt_id=prompt,
                                source_capture_sha256=_source(prompt),
                                anchor_layer=0,
                                consumer_layer=1,
                                consumer_head=consumer_head,
                                donor_head=donor_head,
                                retained_tiles=retained,
                                eligible_tiles=100,
                                anchor_dropped_mass=(
                                    0.03 if target_sparsity == 0.5 else 0.07 + 0.01 * donor_head
                                ),
                                anchor_stats_by_layer=anchor_stats,
                                dropped_mass=dropped_mass,
                            )
                        )
    return rows


def _calibrate(rows):
    return calibrate_mask_reuse_policy(
        rows,
        vanilla_calibration=VANILLA_FIT,
        topology=TOPOLOGY,
        checkpoint_manifest=VERIFIED_CHECKPOINT,
        evidence=EVIDENCE,
        max_anchor_dropped_mass=0.1,
        max_reuse_dropped_mass=0.1,
        max_reuse_selection_dropped_mass=0.05,
    )


def test_selects_target_sparsity_and_exports_backend_v3():
    artifact = _calibrate(_observations())

    assert artifact["version"] == 3
    assert artifact["phase"] == "prefill"
    assert artifact["decode"] == {"mode": "dense"}
    assert artifact["calibration_protocol"] == "modelopt_mask_reuse_target_sparsity_v1"
    assert artifact["producer"] == {"name": "modelopt", "version": modelopt.__version__}
    assert artifact["evidence"] == EVIDENCE
    assert artifact["threshold_scale_factor"] == {
        "formula": "a * exp(b * target_sparsity)",
        "prefill": VANILLA_FIT["prefill"],
    }
    assert artifact["context_policies"] == [
        {
            "min_kv_tokens": 65_536,
            "max_kv_tokens": 131_072,
            "target_sparsity": 0.7,
            "headmaps": {"1": [1, 0]},
            "fallback_heads": {"1": [1]},
        }
    ]
    assert artifact["promotion_status"] == "candidate_only"
    assert artifact["deployment_geometry_validated"] is False
    assert artifact["deployment_geometry"]["contract"]["kv_page_tokens"] == 16
    assert len(artifact["deployment_geometry"]["observations"]) == 4
    assert (
        artifact["calibration_report"]["overall"]["reuse_heldout"]["constraint_violation_rate"]
        == 0.0
    )
    json.dumps(artifact)


def test_heldout_values_evaluate_but_cannot_change_selection():
    baseline = _calibrate(_observations())
    hostile_rows = [
        replace(row, dropped_mass=0.9)
        if row.split == "heldout"
        and row.target_sparsity == 0.7
        and row.consumer_head == 0
        and row.donor_head == 1
        else row
        for row in _observations()
    ]

    hostile = _calibrate(hostile_rows)

    assert hostile["context_policies"] == baseline["context_policies"]
    assert (
        hostile["calibration_report"]["overall"]["reuse_heldout"]["constraint_violation_rate"]
        == 1.0
    )
    assert hostile["calibration_report"]["overall"]["reuse_heldout"]["worst_dropped_mass"] == 0.9


def test_rejects_relabelled_fixed_lambda_observation():
    rows = _observations()
    rows[0] = replace(
        rows[0],
        threshold_lambda=math.nextafter(rows[0].threshold_lambda, math.inf),
    )

    with pytest.raises(MaskReuseCalibrationError, match="threshold_lambda does not match"):
        _calibrate(rows)


def test_rejects_inexact_log2_launch_argument():
    rows = _observations()
    rows[0] = replace(
        rows[0],
        threshold_log2=math.nextafter(rows[0].threshold_log2, math.inf),
    )

    with pytest.raises(MaskReuseCalibrationError, match="threshold_log2 does not match"):
        _calibrate(rows)


def test_anchor_gate_includes_anchor_without_consumer_layer():
    rows = [
        replace(
            row,
            anchor_stats_by_layer={
                **row.anchor_stats_by_layer,
                2: AnchorLayerStats((30, 30), (0.2, 0.2)),
            },
        )
        if row.target_sparsity == 0.7
        else row
        for row in _observations()
    ]

    artifact = _calibrate(rows)

    assert artifact["context_policies"][0]["target_sparsity"] == 0.5


def test_rejects_inconsistent_repeated_anchor_payload():
    rows = _observations()
    rows[0] = replace(
        rows[0],
        anchor_stats_by_layer={
            **rows[0].anchor_stats_by_layer,
            2: AnchorLayerStats((59, 60), (0.02, 0.02)),
        },
    )

    with pytest.raises(MaskReuseCalibrationError, match="anchor_stats_by_layer differs"):
        _calibrate(rows)


def test_jsonl_rejects_duplicate_keys():
    with pytest.raises(MaskReuseCalibrationError, match="duplicate JSON key 'model'"):
        parse_mask_reuse_observations(['{"model":"first","model":"second"}'])


def test_prefill_fit_rejects_b_above_backend_limit():
    invalid = {"prefill": {**VANILLA_FIT["prefill"], "b": 20.000_001}}

    with pytest.raises(MaskReuseCalibrationError, match=r"prefill\.b"):
        canonical_prefill_threshold_scale_factor(invalid)


def test_canonicalizes_existing_sparse_attention_config():
    exported = {
        "sparse_attention_config": {
            "config_groups": {
                "group_0": {
                    "algorithm": "skip_softmax",
                    "threshold_scale_factor": {
                        "formula": "a * exp(b * target_sparsity)",
                        "prefill": VANILLA_FIT["prefill"],
                    },
                }
            }
        }
    }

    assert canonical_prefill_threshold_scale_factor(exported) == {
        "formula": "a * exp(b * target_sparsity)",
        "prefill": VANILLA_FIT["prefill"],
    }
