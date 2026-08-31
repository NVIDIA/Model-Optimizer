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

"""Deterministic selection modes for post-MIP candidate sets."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .records import CandidateLedger

__all__ = ["apply_filter", "filter_metric_references", "validate_filter_config"]

_BEST_SELECTION_MODES = frozenset({"individual_best", "best_per_concurrency"})


def _metric_entries(config: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    entries = config.get("metrics") or ()
    if not isinstance(entries, (list, tuple)) or not entries:
        raise ValueError("pareto and aggregate_rank filters require a non-empty metrics list")
    normalized = []
    for entry in entries:
        if not isinstance(entry, Mapping) or not entry.get("metric"):
            raise TypeError("filter metrics must be mappings containing metric")
        unknown = set(entry) - {"metric", "direction", "weight"}
        if unknown:
            raise ValueError(f"unknown filter metric fields: {sorted(unknown)}")
        direction = str(entry.get("direction", "minimize"))
        if direction not in {"minimize", "maximize"}:
            raise ValueError("filter metric direction must be minimize or maximize")
        weight = float(entry.get("weight", 1.0))
        if weight <= 0:
            raise ValueError("aggregate-rank weights must be positive")
        normalized.append(
            {"metric": str(entry["metric"]), "direction": direction, "weight": weight}
        )
    return tuple(normalized)


def validate_filter_config(config: Mapping[str, Any]) -> None:
    mode = str(config.get("mode") or "")
    common = {
        "type",
        "input",
        "model_source",
        "failure_policy",
        "config",
        "mode",
        "require_match",
    }
    allowed = {
        "top_k": common | {"metric", "direction", "top_k", "best_selection_mode"},
        "threshold": common | {"metric", "min", "max"},
        "pareto": common | {"metrics"},
        "aggregate_rank": common | {"metrics", "top_k"},
    }
    if mode not in allowed:
        raise ValueError("filter.mode must be top_k, threshold, pareto, or aggregate_rank")
    if not isinstance(config.get("require_match", False), bool):
        raise TypeError("filter.require_match must be a boolean")
    unknown = set(config) - allowed[mode]
    if unknown:
        raise ValueError(f"unknown {mode} filter fields: {sorted(unknown)}")
    if mode in {"top_k", "threshold"} and not config.get("metric"):
        raise ValueError(f"{mode} filter requires metric")
    if mode == "top_k":
        direction = str(config.get("direction", "minimize"))
        if direction not in {"minimize", "maximize"}:
            raise ValueError("top_k direction must be minimize or maximize")
        best_selection_mode = config.get("best_selection_mode")
        if best_selection_mode is not None:
            if best_selection_mode not in _BEST_SELECTION_MODES:
                raise ValueError(
                    "top_k.best_selection_mode must be individual_best or best_per_concurrency"
                )
            metric = str(config["metric"])
            owner, separator, leaf = metric.partition(".")
            if not separator or not owner or not leaf or owner == "mip":
                raise ValueError(
                    "best_selection_mode requires a node-qualified metric "
                    "such as serving.output_token_throughput"
                )
        top_k = config.get("top_k")
        if best_selection_mode is not None and isinstance(top_k, Mapping):
            raise ValueError("best_selection_mode requires an integer top_k")
        if isinstance(top_k, Mapping):
            if set(top_k) - {"homogeneous", "heterogeneous"}:
                raise ValueError("top_k quotas accept homogeneous and heterogeneous only")
            values = top_k.values()
        else:
            values = (top_k,)
        if any(
            value is None or isinstance(value, bool) or not str(value).isdigit() or int(value) < 1
            for value in values
        ):
            raise ValueError("top_k values must be positive integers")
    elif mode == "threshold":
        if config.get("min") is None and config.get("max") is None:
            raise ValueError("threshold filter requires min or max")
    else:
        _metric_entries(config)
        if mode == "aggregate_rank" and int(config.get("top_k", 1)) < 1:
            raise ValueError("aggregate_rank.top_k must be positive")


def filter_metric_references(config: Mapping[str, Any]) -> tuple[str, ...]:
    mode = str(config.get("mode"))
    if mode in {"top_k", "threshold"}:
        return (str(config["metric"]),)
    return tuple(entry["metric"] for entry in _metric_entries(config))


def _finite_metric(ledger: CandidateLedger, revision_id: str, reference: str) -> float | None:
    value = ledger.resolve_metric(revision_id, reference)
    return value if value is not None and math.isfinite(value) else None


def _origin_kind(ledger: CandidateLedger, revision_id: str) -> str:
    revision = ledger.revisions[revision_id]
    while revision.parent_revision_id is not None:
        revision = ledger.revisions[revision.parent_revision_id]
    return str(revision.artifact.get("kind", "heterogeneous"))


def _ordered_metric_rows(
    rows: Sequence[tuple[float, str]],
    *,
    direction: str,
) -> list[tuple[float, str]]:
    return sorted(
        rows,
        key=lambda row: (row[0], row[1]),
        reverse=direction == "maximize",
    )


def _apply_sweep_top_k(
    ledger: CandidateLedger,
    revision_ids: Sequence[str],
    config: Mapping[str, Any],
) -> tuple[tuple[str, ...], dict[str, str], dict[str, float]]:
    metric = str(config["metric"])
    direction = str(config.get("direction", "minimize"))
    selection_mode = str(config["best_selection_mode"])
    top_k = int(config["top_k"])
    sweeps = {
        revision_id: ledger.resolve_concurrency_metrics(revision_id, metric)
        for revision_id in revision_ids
    }
    concurrencies = sorted({concurrency for values in sweeps.values() for concurrency in values})
    excluded = {}
    complete = {}
    for revision_id, values in sweeps.items():
        missing = [value for value in concurrencies if value not in values]
        if not concurrencies:
            excluded[revision_id] = f"missing or non-finite metric {metric}"
        elif missing:
            excluded[revision_id] = f"incomplete concurrency sweep; missing {missing}"
        else:
            complete[revision_id] = values

    scores: dict[str, float] = {}
    if selection_mode == "individual_best":
        reducer = max if direction == "maximize" else min
        scores = {revision_id: reducer(values.values()) for revision_id, values in complete.items()}
        rows = _ordered_metric_rows(
            [(value, revision_id) for revision_id, value in scores.items()],
            direction=direction,
        )
        selected = tuple(revision_id for _value, revision_id in rows[:top_k])
        for _value, revision_id in rows[top_k:]:
            excluded[revision_id] = "outside top_k"
        return selected, excluded, scores

    selected_ids = set()
    best_ranks = {revision_id: math.inf for revision_id in complete}
    for concurrency in concurrencies:
        rows = _ordered_metric_rows(
            [(values[concurrency], revision_id) for revision_id, values in complete.items()],
            direction=direction,
        )
        for rank, (_value, revision_id) in enumerate(rows, start=1):
            best_ranks[revision_id] = min(best_ranks[revision_id], rank)
            if rank <= top_k:
                selected_ids.add(revision_id)
    scores = {revision_id: float(rank) for revision_id, rank in best_ranks.items()}
    selected = tuple(
        sorted(selected_ids, key=lambda revision_id: (scores[revision_id], revision_id))
    )
    for revision_id in complete:
        if revision_id not in selected_ids:
            excluded[revision_id] = "outside top_k at every concurrency"
    return selected, excluded, scores


def apply_filter(
    ledger: CandidateLedger,
    revision_ids: Sequence[str],
    config: Mapping[str, Any],
) -> tuple[tuple[str, ...], dict[str, str], dict[str, float]]:
    """Select revisions and return exclusions plus filter-produced scores."""

    validate_filter_config(config)
    mode = str(config["mode"])
    excluded: dict[str, str] = {}
    scores: dict[str, float] = {}
    if mode == "top_k":
        if config.get("best_selection_mode") is not None:
            return _apply_sweep_top_k(ledger, revision_ids, config)
        metric = str(config["metric"])
        reverse = str(config.get("direction", "minimize")) == "maximize"
        rows = []
        for revision_id in revision_ids:
            value = _finite_metric(ledger, revision_id, metric)
            if value is None:
                excluded[revision_id] = f"missing or non-finite metric {metric}"
            else:
                rows.append((value, revision_id))
        rows.sort(key=lambda row: (row[0], row[1]), reverse=reverse)
        top_k = config["top_k"]
        if isinstance(top_k, Mapping):
            grouped = defaultdict(list)
            for row in rows:
                grouped[_origin_kind(ledger, row[1])].append(row)
            kept_rows = []
            for kind in ("heterogeneous", "homogeneous"):
                kept_rows.extend(grouped[kind][: int(top_k.get(kind, 0))])
            kept = {revision_id for _value, revision_id in kept_rows}
            selected = tuple(revision_id for _value, revision_id in rows if revision_id in kept)
        else:
            selected = tuple(revision_id for _value, revision_id in rows[: int(top_k)])
        for _value, revision_id in rows:
            if revision_id not in selected:
                excluded[revision_id] = "outside top_k"
        return selected, excluded, scores

    if mode == "threshold":
        metric = str(config["metric"])
        minimum = config.get("min")
        maximum = config.get("max")
        threshold_selected = []
        for revision_id in revision_ids:
            value = _finite_metric(ledger, revision_id, metric)
            if value is None:
                excluded[revision_id] = f"missing or non-finite metric {metric}"
            elif minimum is not None and value < float(minimum):
                excluded[revision_id] = f"{metric} below minimum"
            elif maximum is not None and value > float(maximum):
                excluded[revision_id] = f"{metric} above maximum"
            else:
                threshold_selected.append(revision_id)
        return tuple(threshold_selected), excluded, scores

    entries = _metric_entries(config)
    values = {}
    for revision_id in revision_ids:
        row = tuple(_finite_metric(ledger, revision_id, entry["metric"]) for entry in entries)
        if any(value is None for value in row):
            excluded[revision_id] = "missing one or more required finite metrics"
        else:
            values[revision_id] = tuple(value for value in row if value is not None)
    if mode == "pareto":
        pareto_selected = []
        for revision_id, row in values.items():
            dominated = False
            for other_id, other in values.items():
                if other_id == revision_id:
                    continue
                no_worse = all(
                    other[index] <= row[index]
                    if entry["direction"] == "minimize"
                    else other[index] >= row[index]
                    for index, entry in enumerate(entries)
                )
                strictly_better = any(
                    other[index] < row[index]
                    if entry["direction"] == "minimize"
                    else other[index] > row[index]
                    for index, entry in enumerate(entries)
                )
                if no_worse and strictly_better:
                    dominated = True
                    break
            if dominated:
                excluded[revision_id] = "Pareto dominated"
            else:
                pareto_selected.append(revision_id)
        return tuple(pareto_selected), excluded, scores

    ranks: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for index, entry in enumerate(entries):
        ordered = sorted(
            ((row[index], revision_id) for revision_id, row in values.items()),
            key=lambda item: (item[0], item[1]),
            reverse=entry["direction"] == "maximize",
        )
        start = 0
        while start < len(ordered):
            end = start + 1
            while end < len(ordered) and ordered[end][0] == ordered[start][0]:
                end += 1
            mean_rank = ((start + 1) + end) / 2.0
            for _value, revision_id in ordered[start:end]:
                ranks[revision_id].append((mean_rank, entry["weight"]))
            start = end
    for revision_id, weighted_ranks in ranks.items():
        scores[revision_id] = sum(rank * weight for rank, weight in weighted_ranks) / sum(
            weight for _rank, weight in weighted_ranks
        )
    ordered_ids = sorted(scores, key=lambda revision_id: (scores[revision_id], revision_id))
    selected = tuple(ordered_ids[: int(config.get("top_k", 1))])
    for revision_id in ordered_ids[len(selected) :]:
        excluded[revision_id] = "outside aggregate_rank top_k"
    return selected, excluded, scores
