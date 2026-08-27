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

import json

import pytest

from modelopt.torch.puzzletron.diagnostics.campaign_findings import MetricSpec
from modelopt.torch.puzzletron.diagnostics.width_sanity import (
    aggregate_parent_sweep_sanity,
    aggregate_width_sanity,
)
from modelopt.torch.puzzletron.stages.diagnostics import (
    _hidden_width_realization_tolerance,
    _publish_parent_sweep_sanity,
    _validate_parent_sweep_checkpoint_loads,
)


def test_parent_sweep_resume_allows_zero_or_one_checkpoint_load():
    _validate_parent_sweep_checkpoint_loads(
        {"checkpoint_loads": {"original": 1, "activation": 0, "realized_0000": 0}}
    )


def test_parent_sweep_resume_rejects_repeated_checkpoint_load():
    with pytest.raises(RuntimeError, match="activation more than once"):
        _validate_parent_sweep_checkpoint_loads(
            {"checkpoint_loads": {"original": 1, "activation": 2}}
        )


@pytest.mark.parametrize(
    ("config", "metric", "expected"),
    [
        ({}, "raw_replacement_loss", 0.0),
        ({"comparison_tolerance": 1.0e-5}, "raw_replacement_loss", 1.0e-5),
        (
            {
                "comparison_tolerance": 1.0e-5,
                "physical_equivalence_tolerance": 1.0e-3,
            },
            "raw_replacement_loss",
            1.0e-3,
        ),
        (
            {
                "physical_equivalence_tolerance": 1.0e-3,
                "physical_equivalence_tolerances": {"raw_replacement_loss": 2.0e-3},
            },
            "raw_replacement_loss",
            2.0e-3,
        ),
    ],
)
def test_hidden_width_realization_uses_physical_tolerance(config, metric, expected):
    assert _hidden_width_realization_tolerance(config, metric) == pytest.approx(expected)


def test_aggregation_separates_ranking_from_physical_equivalence():
    summaries = {
        "axis_a": {
            "rows": [
                {
                    "axis": "axis_a",
                    "layer_idx": 2,
                    "target_value": 4,
                    "method": method,
                    "loss": loss,
                }
                for method, loss in (
                    ("activation", 0.5),
                    ("random", 0.8),
                    ("reverse", 0.9),
                    ("realized", 0.5),
                )
            ]
        },
        "hidden_width": {
            "hidden_width": 7,
            "teacher_hidden_width": 8,
            "rows": [
                {"role": role, "metrics": {"loss": loss}}
                for role, loss in (
                    ("activation", 1.0),
                    ("original", 0.9),
                    ("reverse", 0.8),
                    ("realized", 1.2),
                )
            ],
        },
    }

    width, slicing = aggregate_width_sanity(
        summaries, metric_specs={"loss": MetricSpec("loss", "lower", abs_tolerance=1e-5)}
    )

    assert {row["method"] for row in width["rows"]} == {"sorted", "original", "reverse"}
    assert {row["method"] for row in slicing["rows"]} == {"sorted", "physical"}
    assert width["axes"] == ["axis_a", "hidden_width"]
    assert slicing["axes"] == ["axis_a", "hidden_width"]
    assert any(
        finding["evidence"]["group"]["axis"] == "hidden_width" for finding in slicing["findings"]
    )
    assert all("hidden_width" not in finding["message"] for finding in slicing["findings"])


def test_parent_sweep_aggregation_groups_axis_rows_and_hidden_width():
    parent_summary = {
        "rows": [
            {
                "axis": axis,
                "layer_idx": 3,
                "target_value": 4,
                "method": method,
                "lm_loss": loss,
            }
            for axis in ("kv_groups", "moe_experts")
            for method, loss in (
                ("activation", 0.5),
                ("random", 0.7),
                ("reverse", 0.8),
                ("realized", 0.50001),
            )
        ]
    }
    hidden_summary = {
        "hidden_width": 3840,
        "teacher_hidden_width": 4096,
        "rows": [
            {"role": role, "metrics": {"lm_loss": loss}}
            for role, loss in (
                ("activation", 0.6),
                ("original", 0.7),
                ("reverse", 0.8),
                ("realized", 0.60001),
            )
        ],
    }

    width, slicing, axes = aggregate_parent_sweep_sanity(
        parent_summary,
        hidden_summary,
        metric_specs={"lm_loss": MetricSpec("lm_loss", "lower", abs_tolerance=1e-3)},
    )

    assert axes == ["hidden_width", "kv_groups", "moe_experts"]
    assert {row["axis"] for row in slicing["rows"]} == set(axes)
    assert slicing["findings"] == []
    assert width["stage"] == "width_sanity"


def test_failed_descriptor_realization_gate_is_promoted_to_slicing_finding():
    summaries = {
        "hidden_width": {
            "hidden_width": 3840,
            "teacher_hidden_width": 4096,
            "primary_metric": "raw_replacement_loss",
            "realization_delta": 6.6e-4,
            "realization_passed": False,
            "rows": [
                {"role": "activation", "metrics": {"raw_replacement_loss": 0.4112}},
                {"role": "realized", "metrics": {"raw_replacement_loss": 0.41186}},
            ],
        }
    }

    _, slicing = aggregate_width_sanity(
        summaries,
        metric_specs={
            "raw_replacement_loss": MetricSpec(
                "raw_replacement_loss", "lower", abs_tolerance=5.0e-3
            )
        },
    )

    assert len(slicing["findings"]) == 1
    finding = slicing["findings"][0]
    assert finding["evidence"]["kind"] == "descriptor_realization_gate"
    assert finding["evidence"]["group"] == {
        "axis": "hidden_width",
        "layer_idx": "global",
        "target_value": 3840,
    }
    assert finding["evidence"]["delta"] == 6.6e-4


def test_parent_sweep_publication_accepts_per_metric_physical_tolerances(tmp_path):
    parent_summary = {
        "rows": [
            {
                "axis": "mamba_head_dim",
                "layer_idx": 3,
                "target_value": 32,
                "method": method,
                "raw_replacement_loss": raw,
                "lm_loss": lm_loss,
            }
            for method, raw, lm_loss in (
                ("activation", 0.040, 2.100),
                ("random", 0.050, 2.110),
                ("reverse", 0.060, 2.120),
                ("realized", 0.036, 2.109),
            )
        ]
    }

    _, slicing_path = _publish_parent_sweep_sanity(
        puzzle_dir=tmp_path,
        parent_summary=parent_summary,
        hidden_width_summary=None,
        diag_cfg={
            "physical_equivalence_tolerance": 1.0e-3,
            "physical_equivalence_tolerances": {
                "raw_replacement_loss": 5.0e-3,
                "lm_loss": 1.0e-2,
            },
            "require_physical_equivalence": True,
        },
        sort_equivalence={"passed": True},
    )

    summary = json.loads(slicing_path.read_text())
    assert summary["findings"] == []
    assert summary["provenance"]["physical_equivalence_tolerances"] == {
        "lm_loss": 1.0e-2,
        "raw_replacement_loss": 5.0e-3,
    }


def test_parent_sweep_physical_miss_is_published_as_correctness_failure(tmp_path):
    parent_summary = {
        "rows": [
            {
                "axis": "moe_experts",
                "layer_idx": 1,
                "target_value": 4,
                "method": method,
                "lm_loss": loss,
            }
            for method, loss in (
                ("activation", 1.0),
                ("random", 1.1),
                ("reverse", 1.2),
                ("realized", 1.5),
            )
        ]
    }

    _, slicing_path = _publish_parent_sweep_sanity(
        puzzle_dir=tmp_path,
        parent_summary=parent_summary,
        hidden_width_summary=None,
        diag_cfg={"physical_equivalence_tolerance": 1.0e-3},
        sort_equivalence={"passed": True},
    )

    summary = json.loads(slicing_path.read_text())
    assert summary["passed"] is False
    assert summary["verdict"] == "failed"
    assert summary["findings"]
    assert all(finding["severity"] == "error" for finding in summary["findings"])
