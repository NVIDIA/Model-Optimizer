# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from modelopt.torch.puzzletron.diagnostics.campaign_findings import (
    MetricSpec,
    equivalence_findings,
    loss_trend_findings,
    ranking_findings,
    structural_findings,
)


def test_equivalence_error_is_derived_from_values_not_axis_name():
    rows = [
        {"axis": "arbitrary_axis", "method": "dynamic", "lm_loss": 4.9582},
        {"axis": "arbitrary_axis", "method": "physical", "lm_loss": 4.9697},
    ]

    findings = equivalence_findings(
        stage="slicing_sanity",
        rows=rows,
        left="dynamic",
        right="physical",
        metrics={"lm_loss": MetricSpec("lm_loss", "lower", abs_tolerance=1e-5)},
        group_keys=("axis",),
    )

    assert findings[0].evidence["delta"] == pytest.approx(0.0115)
    assert findings[0].evidence["group"] == {"axis": "arbitrary_axis"}
    assert findings[0].severity == "error"
    assert "arbitrary_axis" not in findings[0].message


@pytest.mark.parametrize(
    ("direction", "preferred", "comparison", "warns"),
    (("lower", 1.2, 1.0, True), ("lower", 0.8, 1.0, False), ("higher", 0.8, 1.0, True)),
)
def test_ranking_findings_obey_metric_direction(direction, preferred, comparison, warns):
    rows = [
        {"axis": "x", "method": "activation", "score": preferred},
        {"axis": "x", "method": "reverse", "score": comparison},
    ]

    findings = ranking_findings(
        stage="width_sanity",
        rows=rows,
        preferred="activation",
        comparisons=("reverse",),
        metrics={"score": MetricSpec("score", direction)},
        group_keys=("axis",),
    )

    assert bool(findings) is warns
    if findings:
        assert findings[0].severity == "warning"


def test_loss_trend_warns_when_ending_window_does_not_improve():
    records = [
        {"mode": "fixed", "step": step, "loss": loss}
        for step, loss in enumerate((1.0, 0.9, 1.1, 1.2), start=1)
    ]

    findings = loss_trend_findings(
        stage="bypass_sanity", records=records, group_key="mode", window=2
    )

    assert len(findings) == 1
    assert findings[0].evidence["start_median"] == pytest.approx(0.95)
    assert findings[0].evidence["end_median"] == pytest.approx(1.15)


def test_structural_findings_report_non_finite_and_duplicate_rows():
    rows = [{"id": "a", "loss": 1.0}, {"id": "a", "loss": float("nan")}]

    findings = structural_findings(
        stage="replacement_scoring", rows=rows, id_keys=("id",), finite_metrics=("loss",)
    )

    assert {finding.evidence["kind"] for finding in findings} == {
        "duplicate_identifier",
        "non_finite_metric",
    }
