# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from modelopt.torch.puzzletron.diagnostics.campaign_findings import MetricSpec
from modelopt.torch.puzzletron.diagnostics.width_sanity import aggregate_width_sanity


def test_aggregation_separates_ranking_from_physical_equivalence():
    summaries = {
        "axis_a": {
            "rows": [
                {"axis": "axis_a", "layer_idx": 2, "target_value": 4, "method": method, "loss": loss}
                for method, loss in (
                    ("activation", 0.5), ("random", 0.8), ("reverse", 0.9), ("realized", 0.5)
                )
            ]
        },
        "hidden_width": {
            "hidden_width": 7,
            "teacher_hidden_width": 8,
            "rows": [
                {"role": role, "metrics": {"loss": loss}}
                for role, loss in (
                    ("activation", 1.0), ("original", 0.9), ("reverse", 0.8), ("realized", 1.2)
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
        finding["evidence"]["group"]["axis"] == "hidden_width"
        for finding in slicing["findings"]
    )
    assert all("hidden_width" not in finding["message"] for finding in slicing["findings"])
