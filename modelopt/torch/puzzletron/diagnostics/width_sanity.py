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

"""Normalize one-case-per-axis results into width and slicing sanity artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

from .campaign_findings import Finding, MetricSpec, equivalence_findings, ranking_findings

__all__ = [
    "aggregate_parent_sweep_sanity",
    "aggregate_width_sanity",
    "descriptor_realization_findings",
]

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


def descriptor_realization_findings(
    axis_summaries: Mapping[str, Mapping[str, Any]],
) -> list[Finding]:
    """Promote descriptor-owned physical-slice gates into public findings."""

    findings = []
    for axis, summary in axis_summaries.items():
        cases = summary.get("cases") or (summary,)
        for case in cases:
            if not isinstance(case, Mapping) or case.get("realization_passed") is not False:
                continue
            target = case.get(
                "target_value",
                case.get("hidden_width", summary.get("hidden_width")),
            )
            metric = str(
                case.get("primary_metric")
                or summary.get("primary_metric")
                or "physical_realization"
            )
            delta = case.get("realization_delta", summary.get("realization_delta"))
            findings.append(
                Finding(
                    stage="slicing_sanity",
                    message=(
                        f"sorted and physical failed the descriptor realization gate for {metric}"
                        + (f": delta {float(delta):.6g}." if isinstance(delta, (int, float)) else ".")
                    ),
                    evidence={
                        "kind": "descriptor_realization_gate",
                        "group": {
                            "axis": str(axis),
                            "layer_idx": case.get("layer_idx", "global"),
                            "target_value": target,
                        },
                        "metric": metric,
                        "left_method": "sorted",
                        "right_method": "physical",
                        "delta": delta,
                    },
                    severity="error",
                )
            )
    return findings


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
    slicing_findings.extend(descriptor_realization_findings(axis_summaries))
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


def aggregate_parent_sweep_sanity(
    parent_summary: Mapping[str, Any],
    hidden_width_summary: Mapping[str, Any] | None,
    *,
    metric_specs: Mapping[str, MetricSpec],
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """Normalize a distributed three-parent sweep into public sanity artifacts."""

    axis_summaries: dict[str, dict[str, Any]] = {}
    for row in parent_summary.get("rows") or ():
        if not isinstance(row, Mapping) or not row.get("axis"):
            continue
        axis = str(row["axis"])
        axis_summaries.setdefault(axis, {"rows": []})["rows"].append(dict(row))
    if hidden_width_summary and hidden_width_summary.get("rows"):
        axis_summaries["hidden_width"] = dict(hidden_width_summary)

    width, slicing = aggregate_width_sanity(
        axis_summaries,
        metric_specs=metric_specs,
    )
    axes = sorted(axis_summaries)
    width["axis_summaries"] = axis_summaries
    slicing["axis_summaries"] = axis_summaries
    return width, slicing, axes
