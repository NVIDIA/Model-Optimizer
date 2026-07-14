# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Normalize one-case-per-axis results into width and slicing sanity artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

from .campaign_findings import MetricSpec, equivalence_findings, ranking_findings

__all__ = ["aggregate_width_sanity"]

_METHODS = {
    "activation": "sorted",
    "sorted": "sorted",
    "random": "original",
    "original": "original",
    "reverse": "reverse",
    "realized": "physical",
    "physical": "physical",
}


def _normalize_summary(axis: str, summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    normalized = []
    for source in summary.get("rows") or ():
        if not isinstance(source, Mapping):
            continue
        role = source.get("method", source.get("role"))
        method = _METHODS.get(str(role))
        if method is None:
            continue
        metrics = source.get("metrics") if isinstance(source.get("metrics"), Mapping) else {}
        row = {
            key: value
            for key, value in source.items()
            if key not in {"metrics", "method", "role"}
        }
        row.update(metrics)
        row.update(
            axis=str(source.get("axis") or axis),
            layer_idx=source.get("layer_idx", "global"),
            target_value=source.get("target_value", source.get("hidden_width")),
            teacher_value=source.get(
                "teacher_value", summary.get("teacher_hidden_width")
            ),
            method=method,
        )
        normalized.append(row)
    return normalized


def _metric_payload(metric_specs: Mapping[str, MetricSpec]) -> dict[str, dict[str, Any]]:
    return {name: asdict(spec) for name, spec in metric_specs.items()}


def aggregate_width_sanity(
    axis_summaries: Mapping[str, Mapping[str, Any]],
    *,
    metric_specs: Mapping[str, MetricSpec],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return separate ranking and physical-equivalence summaries for all axes."""

    rows = [
        row
        for axis, summary in axis_summaries.items()
        for row in _normalize_summary(str(axis), summary)
    ]
    axes = sorted({str(row["axis"]) for row in rows})
    group_keys = ("axis", "layer_idx", "target_value")
    width_rows = [row for row in rows if row["method"] in {"sorted", "original", "reverse"}]
    slicing_rows = [row for row in rows if row["method"] in {"sorted", "physical"}]
    width_findings = ranking_findings(
        stage="width_sanity",
        rows=width_rows,
        preferred="sorted",
        comparisons=("original", "reverse"),
        metrics=metric_specs,
        group_keys=group_keys,
    )
    slicing_findings = equivalence_findings(
        stage="slicing_sanity",
        rows=slicing_rows,
        left="sorted",
        right="physical",
        metrics=metric_specs,
        group_keys=group_keys,
    )
    common = {
        "schema_version": 1,
        "axes": axes,
        "metric_specs": _metric_payload(metric_specs),
    }
    return (
        {
            **common,
            "stage": "width_sanity",
            "rows": width_rows,
            "findings": [asdict(finding) for finding in width_findings],
        },
        {
            **common,
            "stage": "slicing_sanity",
            "rows": slicing_rows,
            "findings": [asdict(finding) for finding in slicing_findings],
        },
    )
