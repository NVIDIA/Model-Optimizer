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


"""Solver integration checks; CBC spawns subprocesses, no GPU required."""

import copy
import math
from hashlib import sha256

import pytest
from _test_utils.torch.mask_reuse import FIT, brute, captures, checkpoint, solve

from modelopt.torch.sparsity.attention_sparsity.calibration.mask_reuse_compact import (
    calibrate_compact_mask_reuse_policy,
    load_compact_mask_reuse_captures,
)
from modelopt.torch.sparsity.attention_sparsity.calibration.mask_reuse_contracts import (
    required_skipped_tiles,
    threshold_pair_matches,
    validate_context_buckets,
    validate_manifest_counts,
)


@pytest.mark.parametrize("backend", ["highs", "cbc"])
@pytest.mark.parametrize("scale", [1.0, 1e-3])
def test_small_topologies_match_exhaustive_optimum(tmp_path, backend, scale):
    cp = checkpoint(tmp_path)
    for seed in range(16):
        rows = captures(cp, seed=seed, scale=scale)
        expected = brute(rows[0], 0.1)
        result = solve(tmp_path, cp, rows, 0.1, backend)
        if expected is None:
            assert result["bmm1_skip_objective"]["target_met"] is False
            continue
        key, consumers, donors = expected
        report = result["calibration_report"]
        assert result["bmm1_skip_objective"]["target_met"] is True
        assert report["risk_selection_objective"][
            "worst_development_prompt_model_wide_total_dropped_mass"
        ] == pytest.approx(key[0], abs=1e-13)
        assert report["combined_tile_cost"] == key[1]
        assert result["anchors"] == list(key[4])
        policy = result["discovery_context_policies"][0]
        got = [
            (-1 if h in policy["fallback_heads"][str(layer)] else policy["headmaps"][str(layer)][h])
            for layer, h in consumers
        ]
        assert got == list(donors)


@pytest.mark.parametrize("backend", ["highs", "cbc"])
@pytest.mark.parametrize("rho", [0.0, 0.1])
def test_dense_inputs_have_a_defined_result(tmp_path, backend, rho):
    cp = checkpoint(tmp_path)
    result = solve(tmp_path, cp, captures(cp, dense=True), rho, backend)
    assert result["bmm1_skip_objective"]["target_met"] == (rho == 0)


@pytest.mark.parametrize("backend", ["highs", "cbc"])
def test_per_bucket_target_chooses_feasible_topology(tmp_path, backend):
    cp = checkpoint(tmp_path)
    rows = captures(cp, buckets=((129, 512, 256), (513, 4096, 1024)))
    for row in rows:
        for a, stats in row["anchor_stats_by_layer"].items():
            stats["retained_tiles"] = [
                row["eligible_tiles"]
                if a == "0" and row["invocation"]["sample_length"] == 256
                else 0
            ] * 2
            stats["dropped_mass"] = [0.01] * 2
        for by_anchor in row["consumer_candidates_by_layer"].values():
            for a, stats in by_anchor.items():
                stats["dropped_mass"] = [[0.001 if a == "0" else 0.01] * 2 for _ in range(2)]
    result = solve(tmp_path, cp, rows, 0.3, backend)
    assert result["anchors"] == [0, 1]
    assert all(
        b["bmm1_skip_objective"]["target_met"] for b in result["calibration_report"]["by_bucket"]
    )


def test_duplicate_source_and_layer_mismatch_fail_cleanly(tmp_path):
    cp = checkpoint(tmp_path)
    rows = captures(cp)
    duplicate = copy.deepcopy(rows[0])
    duplicate["invocation"]["prompt_id"] += "-copy"
    with pytest.raises(ValueError, match="fingerprint"):
        solve(tmp_path, cp, [*rows, duplicate], 0.1, "highs")
    rows[1]["max_reuse_span"] = 1
    rows[1]["consumer_candidates_by_layer"]["2"].pop("0")
    with pytest.raises(ValueError, match="span"):
        solve(tmp_path, cp, rows, 0.1, "highs")


def test_decimal_target_and_one_ulp_threshold_contract():
    assert required_skipped_tiles(0.14, 50) == 7
    assert required_skipped_tiles(0.5, 2**61 + 1) == 2**60 + 1
    log2 = -7.123
    value = 2.0**log2
    adjacent = math.nextafter(value, math.inf)
    assert threshold_pair_matches(log2, adjacent)
    assert not threshold_pair_matches(log2, math.nextafter(adjacent, math.inf))


def test_touching_buckets_and_negative_manifest_counts_rejected():
    with pytest.raises(ValueError, match="non-overlapping"):
        validate_context_buckets([(129, 512), (512, 1024)])
    with pytest.raises(ValueError, match="nonnegative"):
        validate_manifest_counts(
            {
                "capture_count": 2,
                "candidate_cell_count": 4,
                "captures": [{"candidate_cell_count": -1}, {"candidate_cell_count": 5}],
            }
        )


@pytest.mark.timeout(120)
def test_94_layers_64_heads_bounded_ties(tmp_path):
    cp = checkpoint(tmp_path, layers=94, heads=64)
    rows = captures(cp, layers=94, heads=64, dense=True, span=1)
    result = solve(tmp_path, cp, rows, 0.0, "highs")
    assert result["anchors"] == list(range(0, 94, 2))
    assert result["bmm1_skip_objective"]["target_met"]
    assert all(
        len(v) == 64 for v in result["discovery_context_policies"][0]["fallback_heads"].values()
    )


@pytest.mark.parametrize("all_anchor", [False, True])
def test_discovery_captures_can_be_refit_without_recapture(tmp_path, all_anchor):
    cp = checkpoint(tmp_path)
    topology = solve(tmp_path, cp, captures(cp), 0.1, "highs")
    topology = {key: topology[key] for key in ("anchors", "nearest")}
    if all_anchor:
        topology = {"anchors": [0, 1, 2], "nearest": {"0": 0, "1": 1, "2": 2}}
    path = tmp_path / "captures.jsonl"
    policy = calibrate_compact_mask_reuse_policy(
        load_compact_mask_reuse_captures(path, topology=topology),
        topology=topology,
        vanilla_calibration=FIT,
        checkpoint_manifest=cp,
        evidence={
            "vanilla_fit_sha256": "a" * 64,
            "reuse_bundle_sha256": sha256(path.read_bytes()).hexdigest(),
        },
        target_bmm1_skip_ratio=0 if all_anchor else 0.1,
        reuse_dropped_mass_report_threshold=0.02,
    )
    assert policy["calibration_report"]["overall"]["bmm1_skip_objective"]["all_context_buckets_met"]
    assert policy["anchors"] == topology["anchors"]
    assert policy["provenance"]["input_capture_count"] == 2
