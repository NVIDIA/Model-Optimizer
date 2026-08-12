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

"""Generic evidence-derived findings for incremental campaign reports."""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

__all__ = [
    "Finding",
    "MetricSpec",
    "equivalence_findings",
    "loss_trend_findings",
    "ranking_findings",
    "structural_findings",
]


@dataclass(frozen=True)
class MetricSpec:
    """Comparison semantics for one numeric report metric."""

    name: str
    direction: Literal["lower", "higher"]
    abs_tolerance: float = 0.0
    rel_tolerance: float = 0.0


@dataclass(frozen=True)
class Finding:
    """One evidence-derived advisory warning or correctness error."""

    stage: str
    message: str
    evidence: Mapping[str, Any]
    severity: Literal["warning", "error"] = "warning"


def _allowed(left: float, right: float, spec: MetricSpec) -> float:
    return max(
        float(spec.abs_tolerance),
        float(spec.rel_tolerance) * max(abs(left), abs(right)),
    )


def _groups(
    rows: Iterable[Mapping[str, Any]], group_keys: Sequence[str]
) -> dict[tuple[Any, ...], list[Mapping[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in group_keys)].append(row)
    return grouped


def _group_evidence(group_keys: Sequence[str], values: Sequence[Any]) -> dict[str, Any]:
    return dict(zip(group_keys, values))


def equivalence_findings(
    *,
    stage: str,
    rows: Iterable[Mapping[str, Any]],
    left: str,
    right: str,
    metrics: Mapping[str, MetricSpec],
    group_keys: Sequence[str],
    method_key: str = "method",
) -> list[Finding]:
    """Report a correctness error when equivalent methods disagree beyond tolerance."""

    findings = []
    for group, values in _groups(rows, group_keys).items():
        by_method = {str(row.get(method_key)): row for row in values}
        if left not in by_method or right not in by_method:
            continue
        for name, spec in metrics.items():
            left_value = by_method[left].get(name)
            right_value = by_method[right].get(name)
            if not isinstance(left_value, (int, float)) or not isinstance(
                right_value, (int, float)
            ):
                continue
            left_float, right_float = float(left_value), float(right_value)
            if not math.isfinite(left_float) or not math.isfinite(right_float):
                continue
            delta = abs(left_float - right_float)
            allowed = _allowed(left_float, right_float, spec)
            if delta <= allowed:
                continue
            findings.append(
                Finding(
                    stage=stage,
                    message=(
                        f"{left} and {right} differ for {name}: "
                        f"delta {delta:.6g} exceeds tolerance {allowed:.6g}."
                    ),
                    evidence={
                        "kind": "equivalence_tolerance",
                        "group": _group_evidence(group_keys, group),
                        "metric": name,
                        "left_method": left,
                        "left_value": left_float,
                        "right_method": right,
                        "right_value": right_float,
                        "delta": delta,
                        "tolerance": allowed,
                    },
                    severity="error",
                )
            )
    return findings


def ranking_findings(
    *,
    stage: str,
    rows: Iterable[Mapping[str, Any]],
    preferred: str,
    comparisons: Sequence[str],
    metrics: Mapping[str, MetricSpec],
    group_keys: Sequence[str],
    method_key: str = "method",
) -> list[Finding]:
    """Report a quality warning when a preferred ranking is worse than a control.

    The warning evaluates the ranking heuristic, not whether the compared model
    transformations are equivalent. A caller may still promote it through a
    stricter qualification policy.
    """

    findings = []
    for group, values in _groups(rows, group_keys).items():
        by_method = {str(row.get(method_key)): row for row in values}
        if preferred not in by_method:
            continue
        for comparison in comparisons:
            if comparison not in by_method:
                continue
            for name, spec in metrics.items():
                preferred_value = by_method[preferred].get(name)
                comparison_value = by_method[comparison].get(name)
                if not isinstance(preferred_value, (int, float)) or not isinstance(
                    comparison_value, (int, float)
                ):
                    continue
                preferred_float = float(preferred_value)
                comparison_float = float(comparison_value)
                if not math.isfinite(preferred_float) or not math.isfinite(comparison_float):
                    continue
                degradation = (
                    preferred_float - comparison_float
                    if spec.direction == "lower"
                    else comparison_float - preferred_float
                )
                allowed = _allowed(preferred_float, comparison_float, spec)
                if degradation <= allowed:
                    continue
                findings.append(
                    Finding(
                        stage=stage,
                        message=(
                            f"{preferred} is worse than {comparison} for {name}: "
                            f"degradation {degradation:.6g} exceeds tolerance {allowed:.6g}."
                        ),
                        evidence={
                            "kind": "ranking_direction",
                            "group": _group_evidence(group_keys, group),
                            "metric": name,
                            "direction": spec.direction,
                            "preferred_method": preferred,
                            "preferred_value": preferred_float,
                            "comparison_method": comparison,
                            "comparison_value": comparison_float,
                            "degradation": degradation,
                            "tolerance": allowed,
                        },
                    )
                )
    return findings


def loss_trend_findings(
    *,
    stage: str,
    records: Iterable[Mapping[str, Any]],
    group_key: str,
    window: int = 4,
    step_key: str = "step",
    loss_key: str = "loss",
) -> list[Finding]:
    """Warn when the ending median loss does not improve over the starting median."""

    grouped: dict[Any, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record.get(group_key)].append(record)
    findings = []
    for group, values in grouped.items():
        finite = [
            row
            for row in sorted(values, key=lambda item: item.get(step_key, 0))
            if isinstance(row.get(loss_key), (int, float)) and math.isfinite(float(row[loss_key]))
        ]
        if len(finite) < 2:
            continue
        size = min(max(1, int(window)), len(finite) // 2)
        start = statistics.median(float(row[loss_key]) for row in finite[:size])
        end = statistics.median(float(row[loss_key]) for row in finite[-size:])
        if end < start:
            continue
        findings.append(
            Finding(
                stage=stage,
                message=f"Ending {loss_key} did not improve for {group_key}={group!r}.",
                evidence={
                    "kind": "loss_trend",
                    "group": {group_key: group},
                    "start_median": start,
                    "end_median": end,
                    "window": size,
                },
            )
        )
    return findings


def structural_findings(
    *,
    stage: str,
    rows: Iterable[Mapping[str, Any]],
    id_keys: Sequence[str] = (),
    finite_metrics: Sequence[str] = (),
) -> list[Finding]:
    """Report duplicate identifiers and non-finite numeric metrics."""

    materialized = list(rows)
    findings = []
    if id_keys:
        identifiers = [tuple(row.get(key) for key in id_keys) for row in materialized]
        for identifier, count in Counter(identifiers).items():
            if count > 1:
                findings.append(
                    Finding(
                        stage=stage,
                        message=f"Duplicate result identifier appears {count} times.",
                        evidence={
                            "kind": "duplicate_identifier",
                            "identifier": dict(zip(id_keys, identifier)),
                            "count": count,
                        },
                    )
                )
    for index, row in enumerate(materialized):
        for metric in finite_metrics:
            value = row.get(metric)
            if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                findings.append(
                    Finding(
                        stage=stage,
                        message=f"Result row {index} has non-finite {metric}.",
                        evidence={
                            "kind": "non_finite_metric",
                            "row": index,
                            "metric": metric,
                            "value": repr(value),
                        },
                    )
                )
    return findings
