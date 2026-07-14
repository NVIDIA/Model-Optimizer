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

"""Incremental, single-file Puzzletron campaign report."""

from __future__ import annotations

import html
import json
from pathlib import Path
from statistics import median
from typing import Any

from ..stages.graph import (
    STAGE_SPECS,
    StageSpec,
    selected_parent_stage_ids,
    stage_display_name,
)

_STAGES = tuple(spec.stage_id for spec in STAGE_SPECS)

_ACTIVATION_METRIC_ORDER = (
    "raw_replacement_loss",
    "cosine_embedding_loss_hidden_states",
    "normalized_mse_loss_hidden_states",
    "mse_loss_hidden_states",
    "mae_loss_hidden_states",
    "kl_div",
    "lm_loss",
    "token_accuracy_top_1",
    "token_accuracy_top_1_consistency",
    "token_accuracy_top_5",
    "token_accuracy_top_5_consistency",
    "token_accuracy_top_10",
    "token_accuracy_top_10_consistency",
)

_METRIC_DESCRIPTIONS = {
    "raw_replacement_loss": (
        "The unadjusted hidden-state mean squared error between the sliced candidate and "
        "the full unsliced teacher. No sliced-teacher baseline is subtracted; when hidden "
        "shapes match, it is numerically identical to mse_loss_hidden_states."
    ),
    "cosine_embedding_loss_hidden_states": "One minus cosine similarity of final hidden states.",
    "normalized_mse_loss_hidden_states": "Hidden-state squared error normalized by teacher energy.",
    "mse_loss_hidden_states": "Mean squared error between final hidden states.",
    "mae_loss_hidden_states": "Mean absolute error between final hidden states.",
    "kl_div": "KL divergence between teacher and candidate token distributions.",
    "lm_loss": "Candidate next-token cross-entropy against labels.",
    "token_accuracy_top_1": "True label-based top-1 token accuracy.",
    "token_accuracy_top_1_consistency": "Teacher/candidate top-1 prediction consistency.",
    "token_accuracy_top_5": "True label-based top-5 token accuracy.",
    "token_accuracy_top_5_consistency": "Teacher/candidate top-5 set consistency.",
    "token_accuracy_top_10": "True label-based top-10 token accuracy.",
    "token_accuracy_top_10_consistency": "Teacher/candidate top-10 set consistency.",
}

_ACTIVATION_ROW_METADATA = {
    "axis",
    "layer_idx",
    "ratio",
    "teacher_value",
    "target_value",
    "method",
    "parent_role",
    "selection_basis",
    "ranking_applicable",
    "ranking_reason",
    "solution_id",
    "solution_file",
    "kept_kv_groups",
    "removed_kv_groups",
    "kv_group_order",
    "kept_query_heads_per_group",
    "removed_query_heads_per_group",
    "query_head_order_per_group",
    "kept_units",
    "removed_units",
    "unit_order",
    "changed_layers",
    "num_changed_layers",
}


def _load_optional(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _campaign_options_data(root: Path) -> dict[str, Any]:
    """Load optional-stage choices without making legacy reports depend on them."""

    try:
        payload = _load_optional(root / "manifests" / "campaign_options.json")
    except (OSError, ValueError):
        payload = {}
    optional_stages = payload.get("optional_stages")
    if not isinstance(optional_stages, dict):
        optional_stages = {}
    vllm_enabled = optional_stages.get("vllm_stats")
    return {
        **payload,
        "present": bool(payload),
        "optional_stages": optional_stages,
        "selection_mode": ("latency_verified" if vllm_enabled is True else "parameter_constrained"),
        "latency_verified": vllm_enabled is True,
    }


def _campaign_options_section(data: dict[str, Any]) -> str:
    if not data.get("present"):
        return "<p class='empty'>Campaign option metadata is unavailable for this legacy run.</p>"
    optional = data.get("optional_stages") or {}
    rows = "".join(
        f"<tr><th>{html.escape(str(name).replace('_', ' '))}</th>"
        f"<td>{'enabled' if enabled else 'skipped'}</td></tr>"
        for name, enabled in sorted(optional.items())
    )
    selection = (
        "Latency-verified: vLLM runtime statistics selected the latency-constrained profile."
        if data.get("latency_verified")
        else "Parameter-constrained: no vLLM runtime measurement was run; this campaign makes no latency claim."
    )
    parent = html.escape(str(data.get("search_parent", "unknown")))
    return (
        f"<p class='note'>{html.escape(selection)}</p>"
        f"<p class='note'>Search parent: <code>{parent}</code></p>"
        "<div class='table-wrap'><table><thead><tr><th>Optional stage</th>"
        "<th>Selection</th></tr></thead><tbody>"
        f"{rows}</tbody></table></div>"
    )


def _manifest(root: Path, stage: str) -> dict[str, Any]:
    return _load_optional(root / "manifests" / f"{stage}.json")


def _stage_artifact_present(root: Path, spec: StageSpec) -> bool:
    """Return whether any canonical completion artifact for a stage exists."""

    for pattern in spec.completion_artifacts:
        if any(character in pattern for character in "*?["):
            if any(root.glob(pattern)):
                return True
        elif (root / pattern).exists():
            return True
    return False


def _pipeline_state(root: Path, spec: StageSpec, config: dict[str, Any]) -> str:
    """Return completed, disabled, or pending from artifacts and configuration."""

    if _stage_artifact_present(root, spec):
        return "completed"
    section = config.get(spec.stage_id)
    if (
        not spec.required
        and isinstance(section, dict)
        and section.get("enabled") is False
    ):
        return "disabled"
    return "pending"


def _dag_label_lines(label: str, *, max_chars: int = 22) -> list[str]:
    """Wrap a stage label at word boundaries for a fixed-size SVG node."""

    lines: list[str] = []
    for word in label.split():
        if lines and len(lines[-1]) + len(word) + 1 <= max_chars:
            lines[-1] += f" {word}"
        else:
            lines.append(word)
    return lines


def _stage_granularity(root: Path, stage: str, config: dict[str, Any]) -> str:
    """Resolve one stage's own granularity from artifacts, then configuration."""

    summary = _load_optional(root / "artifacts" / stage / "summary.json")
    granularity = summary.get("granularity")
    if granularity in {"block", "subblock"}:
        return str(granularity)
    if stage == "bypass":
        observations = root / "artifacts" / "bypass" / "dp_observations.jsonl"
        try:
            first = json.loads(next(line for line in observations.read_text().splitlines() if line))
        except (OSError, StopIteration, ValueError):
            first = {}
        rows = first.get("observations") or []
        if rows and rows[0].get("granularity") in {"block", "subblock"}:
            return str(rows[0]["granularity"])
    section = config.get(stage)
    if isinstance(section, dict) and section.get("granularity") in {"block", "subblock"}:
        return str(section["granularity"])
    return "block"


def _latest_merged_config(root: Path) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for stage in _STAGES:
        manifest = _manifest(root, stage)
        candidate = manifest.get("merged_config", manifest.get("config"))
        if isinstance(candidate, dict):
            merged = candidate
    return merged


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.9g}"
    return str(value)


def _teacher_self_metric_is_not_applicable(metric: str) -> bool:
    return (
        metric in {"kl_div", "raw_replacement_loss"}
        or metric.endswith(("_hidden_states", "_consistency"))
        or (metric.startswith("top_") and metric.endswith("_logit_agreement"))
    )


def _sort_table(summary: dict[str, Any]) -> str:
    teacher = summary.get("teacher") or {}
    sorted_teacher = summary.get("sorted_teacher") or {}
    reverse_sorted = summary.get("reverse_sorted") or {}
    if not teacher and not sorted_teacher and not reverse_sorted:
        return "<p class='empty'>Pending sort diagnosis.</p>"
    available = set(teacher) | set(sorted_teacher) | set(reverse_sorted)
    metrics = [metric for metric in _ACTIVATION_METRIC_ORDER if metric in available]
    metrics.extend(sorted(available - set(metrics)))
    include_reverse = bool(reverse_sorted)
    rows = []
    for metric in metrics:
        teacher_value = (
            "N/A"
            if _teacher_self_metric_is_not_applicable(metric)
            else _fmt(teacher.get(metric, "missing"))
        )
        reverse_cell = (
            f"<td>{html.escape(_fmt(reverse_sorted.get(metric, 'missing')))}</td>"
            if include_reverse
            else ""
        )
        rows.append(
            "<tr>"
            f"<th>{html.escape(metric)}</th>"
            f"<td>{html.escape(teacher_value)}</td>"
            f"<td>{html.escape(_fmt(sorted_teacher.get(metric, 'missing')))}</td>"
            f"{reverse_cell}"
            "</tr>"
        )
    gate = "passed" if summary.get("passed") is True else "not passed"
    reverse_header = "<th>Reverse sorted</th>" if include_reverse else ""
    return (
        f"<p class='gate {html.escape(gate.replace(' ', '-'))}'>Equivalence gate: {gate}</p>"
        "<div class='table-wrap'><table><thead><tr><th>Metric</th><th>Teacher</th>"
        f"<th>Sorted teacher</th>{reverse_header}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _result_metric_value(value: Any) -> float | None:
    if isinstance(value, dict):
        value = value.get("avg")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _result_metrics(path: Path) -> dict[str, float]:
    raw = _load_optional(path)
    metrics = {}
    for key in _ACTIVATION_METRIC_ORDER:
        value = raw.get(key)
        number = _result_metric_value(value)
        if number is not None:
            metrics[str(key)] = number
    return metrics


def _sort_summary(root: Path) -> dict[str, Any]:
    summary = _load_optional(root / "artifacts" / "sort_sanity" / "summary.json")
    if summary.get("teacher") or summary.get("sorted_teacher"):
        return summary
    diagnostic = root / "diagnostics" / "sort_sanity"
    validation = diagnostic / "single_sequence_replacement_solutions--validation"
    teacher = _result_metrics(validation / "teacher.json")
    sorted_teacher = _result_metrics(validation / "sliced_teacher.json") or _result_metrics(
        validation / "solution_0.json"
    )
    reverse_sorted = _result_metrics(
        diagnostic
        / "reverse"
        / "single_sequence_replacement_solutions--validation"
        / "sliced_teacher.json"
    )
    if not reverse_sorted:
        reverse_sorted = _result_metrics(
            diagnostic
            / "reverse"
            / "single_sequence_replacement_solutions--validation"
            / "solution_0.json"
        )
    if teacher or sorted_teacher or reverse_sorted:
        summary = {
            **summary,
            "teacher": teacher,
            "sorted_teacher": sorted_teacher,
            "reverse_sorted": reverse_sorted,
        }
    return summary


def _activation_diagnostic_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [dict(row) for row in summary.get("rows", ()) if isinstance(row, dict)]
    hidden = summary.get("hidden_width") or {}
    teacher_width = hidden.get("teacher_hidden_width")
    width = hidden.get("hidden_width")
    for row in hidden.get("rows", ()):
        if not isinstance(row, dict):
            continue
        metrics = row.get("metrics") or {}
        rows.append(
            {
                "axis": "hidden_width",
                "layer_idx": "global",
                "ratio": (float(width) / float(teacher_width) if width and teacher_width else None),
                "teacher_value": teacher_width,
                "target_value": width,
                "method": row.get("role"),
                **metrics,
            }
        )
    return rows


def _activation_diagnostic_summary(root: Path) -> dict[str, Any]:
    width = _load_optional(root / "artifacts" / "width_sanity" / "summary.json")
    slicing = _load_optional(root / "artifacts" / "slicing_sanity" / "summary.json")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for summary in (width, slicing):
        for raw in summary.get("rows", ()):
            if not isinstance(raw, dict):
                continue
            row = dict(raw)
            identity = json.dumps(row, sort_keys=True, default=str)
            if identity not in seen:
                rows.append(row)
                seen.add(identity)
    return {
        "rows": rows,
        "axes": sorted({str(row.get("axis")) for row in rows if row.get("axis")}),
        "width_findings": list(width.get("findings") or ()),
        "slicing_findings": list(slicing.get("findings") or ()),
    }


def _finding_cards(findings: list[dict[str, Any]]) -> str:
    if not findings:
        return ""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for finding in findings:
        evidence = finding.get("evidence") or {}
        group = evidence.get("group") or {}
        key = (str(finding.get("stage") or "sanity"), json.dumps(group, sort_keys=True))
        grouped.setdefault(key, []).append(finding)

    def card(key: tuple[str, str], values: list[dict[str, Any]]) -> str:
        stage, raw_group = key
        group = json.loads(raw_group)
        location = " · ".join(
            f"{name}={value}" for name, value in group.items() if value is not None
        ) or "campaign"
        messages = [
            html.escape(str(value.get("message") or "Measured result needs review."))
            for value in values
        ]
        details = "".join(f"<li>{message}</li>" for message in messages)
        return (
            "<article class='finding warning'><div><strong>Warning</strong>"
            f"<span>{html.escape(stage.replace('_', ' ').title())} · "
            f"{html.escape(location)} · {len(values)} finding(s)</span></div>"
            f"<details><summary>{messages[0]}</summary><ul>{details}</ul></details></article>"
        )

    return (
        "<div class='finding-list'>"
        + "".join(card(key, values) for key, values in grouped.items())
        + "</div>"
    )


_FINDING_METHOD_KEYS = (
    "preferred_method",
    "comparison_method",
    "left_method",
    "right_method",
)


def _finding_is_table_bound(finding: dict[str, Any]) -> bool:
    evidence = finding.get("evidence") or {}
    group = evidence.get("group") or {}
    return (
        all(key in group for key in ("axis", "layer_idx", "target_value"))
        and evidence.get("metric") is not None
        and any(evidence.get(key) is not None for key in _FINDING_METHOD_KEYS)
    )


def _cell_finding_messages(
    findings: list[dict[str, Any]],
    *,
    axis: Any,
    layer_idx: Any,
    target_value: Any,
    metric: str,
    method: str,
) -> list[str]:
    messages = []
    for finding in findings:
        evidence = finding.get("evidence") or {}
        group = evidence.get("group") or {}
        methods = {
            str(evidence[key])
            for key in _FINDING_METHOD_KEYS
            if evidence.get(key) is not None
        }
        if (
            str(group.get("axis")) == str(axis)
            and str(group.get("layer_idx")) == str(layer_idx)
            and str(group.get("target_value")) == str(target_value)
            and str(evidence.get("metric")) == metric
            and method in methods
        ):
            messages.append(str(finding.get("message") or "Measured result needs review."))
    return messages


def _activation_diagnostic_section(summary: dict[str, Any]) -> str:
    rows = _activation_diagnostic_rows(summary)
    if not rows:
        return "<p class='empty'>No width or slicing sanity artifact.</p>"
    axes = sorted({str(row.get("axis")) for row in rows if row.get("axis") is not None})
    available_metrics = {
        key
        for row in rows
        for key, value in row.items()
        if key not in _ACTIVATION_ROW_METADATA
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    }
    metrics = [metric for metric in _ACTIVATION_METRIC_ORDER if metric in available_metrics]
    metrics.extend(sorted(available_metrics - set(metrics)))
    axis_options = "".join(
        f"<option value='{html.escape(axis)}'>{html.escape(axis)}</option>" for axis in axes
    )
    metric_options = "".join(
        f"<option value='{html.escape(metric)}'>{html.escape(metric)}</option>"
        for metric in metrics
    )
    if not axes or not metrics:
        return (
            _finding_cards(
                list(summary.get("width_findings") or ())
                + list(summary.get("slicing_findings") or ())
            )
            + "<p class='empty'>The sanity artifact contains no plottable numeric metrics.</p>"
        )
    first_axis = axes[0]
    first_metric = metrics[0]
    findings = list(summary.get("width_findings") or ()) + list(
        summary.get("slicing_findings") or ()
    )
    cases: dict[tuple[Any, Any, Any], dict[str, dict[str, Any]]] = {}
    for row in rows:
        if str(row.get("axis")) != first_axis:
            continue
        key = (row.get("layer_idx"), row.get("target_value"), row.get("ratio"))
        cases.setdefault(key, {})[str(row.get("method"))] = row
    body = []
    for (layer_idx, target, ratio), methods in sorted(
        cases.items(), key=lambda item: (str(item[0][0]), str(item[0][1]))
    ):
        label = (
            f"global@{float(ratio) * 100:.0f}%"
            if layer_idx == "global" and ratio is not None
            else f"layer_{layer_idx}@{float(ratio) * 100:.0f}%"
            if ratio is not None
            else str(layer_idx)
        )
        method_names = ("sorted", "original", "reverse", "physical")
        cells = []
        for method in method_names:
            value = methods.get(method, {}).get(first_metric)
            messages = _cell_finding_messages(
                findings,
                axis=first_axis,
                layer_idx=layer_idx,
                target_value=target,
                metric=first_metric,
                method=method,
            )
            attributes = (
                " class='warning-cell'"
                if messages
                else ""
            )
            formatted = html.escape(_fmt(value)) if value is not None else "N/A"
            if messages:
                formatted = (
                    "<span class='warning-value' tabindex='0' "
                    f"data-warning='{html.escape(chr(10).join(messages), quote=True)}'>"
                    f"{formatted}</span>"
                )
            cells.append(
                f"<td{attributes}>{formatted}</td>"
            )
        body.append(
            "<tr>"
            f"<th>{html.escape(label)}</th><td>{html.escape(_fmt(target))}</td>"
            + "".join(cells)
            + "</tr>"
        )
    return (
        _finding_cards(
            [finding for finding in findings if not _finding_is_table_bound(finding)]
        )
        + "<p class='note'>Every sliced model is compared with the full, unsliced original teacher. "
        "The original baseline is the teacher channel order sliced to the same target. "
        "Physical is a materialized slice of the sorted checkpoint and is the slicing ground truth.</p>"
        "<label class='selector-label' for='activation-axis-select'>Swept axis</label>"
        f'<select id="activation-axis-select">{axis_options}</select>'
        "<label class='selector-label' for='activation-metric-select'>Metric</label>"
        f'<select id="activation-metric-select">{metric_options}</select>'
        f"<p id='activation-metric-help' class='note'>{html.escape(_METRIC_DESCRIPTIONS.get(first_metric, ''))}</p>"
        "<div class='table-wrap'><table id=\"activation-diagnostic-table\"><thead><tr>"
        "<th>Slice</th><th>Target</th><th>Sorted runtime</th><th>Original runtime</th>"
        "<th>Reverse runtime</th><th>Physical sorted</th>"
        f"</tr></thead><tbody id='activation-diagnostic-body'>{''.join(body)}</tbody></table></div>"
    )


def _bypass_overfit_mode_data(root: Path, mode: str) -> dict[str, Any]:
    history_path = (
        root / "artifacts" / "bypass" / "overfit_probe" / mode / "local_kd_loss_history.json"
    )
    history = _load_optional(history_path)
    if mode == "smallest_fixed" and not history:
        legacy_path = root / "artifacts" / "bypass" / "overfit_probe" / "local_kd_loss_history.json"
        legacy = _load_optional(legacy_path)
        if legacy:
            history = legacy
            history_path = legacy_path
    records = [row for row in history.get("records", ()) if isinstance(row, dict)]
    granularity = str(history.get("granularity") or "").strip().lower()
    if granularity not in {"block", "subblock"}:
        granularity = (
            "subblock" if any(row.get("per_subblock_loss") for row in records) else "block"
        )
    metric_key = "per_subblock_loss" if granularity == "subblock" else "per_layer_loss"

    def unit_sort_key(value: str) -> tuple[Any, ...]:
        layer, *parts = value.split(":")
        try:
            layer_key: tuple[int, Any] = (0, int(layer))
        except ValueError:
            layer_key = (1, layer)
        return (*layer_key, *parts)

    units = sorted(
        {str(unit) for row in records for unit in (row.get(metric_key) or {})},
        key=unit_sort_key,
    )
    summary = dict(history.get("summary") or {})
    window = max(1, int(history.get("trend_window") or (8 if mode == "diverse_resampled" else 4)))
    finite_losses = [
        float(row["loss"]) for row in records if isinstance(row.get("loss"), (int, float))
    ]
    if len(finite_losses) >= 2 * window and not summary.get("loss_trend"):
        first = median(finite_losses[:window])
        last = median(finite_losses[-window:])
        summary["loss_trend"] = {
            "window": window,
            "first_window_median": first,
            "last_window_median": last,
            "last_to_first_ratio": last / first if first else None,
            "hard_gate_passed": last < first,
        }
    if "distinct_structure_count" not in summary and records:
        structures = {
            json.dumps(
                {
                    key: value
                    for key, value in (row.get("elastic_selection") or {}).items()
                    if key != "step"
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            for row in records
        }
        summary["distinct_structure_count"] = len(structures)
    return {
        "mode": mode,
        "path": str(history_path),
        "loss_name": str(history.get("loss_name") or "local_kd_loss"),
        "max_steps": int(history.get("max_steps") or len(records)),
        "source_checkpoint_identity": history.get("source_checkpoint_identity"),
        "trend_window": window,
        "summary": summary,
        "granularity": granularity,
        "metric_key": metric_key,
        "units": units,
        "records": records,
    }


def _bypass_overfit_data(root: Path) -> dict[str, Any]:
    modes = {
        mode: _bypass_overfit_mode_data(root, mode)
        for mode in ("smallest_fixed", "diverse_resampled")
    }
    granularities = {
        data.get("granularity") for data in modes.values() if data.get("records")
    }
    granularity = "subblock" if "subblock" in granularities else "block"
    units = sorted(
        {unit for data in modes.values() for unit in data.get("units", ())},
        key=lambda value: (
            int(value.split(":", 1)[0]),
            value.split(":", 1)[1:] if ":" in value else (),
        ),
    )
    return {"modes": modes, "granularity": granularity, "units": units}


def _overfit_summary_card(mode: str, data: dict[str, Any]) -> str:
    label = "Smallest fixed" if mode == "smallest_fixed" else "Diverse resampled"
    records = data.get("records") or []
    if not records:
        return (
            f"<article class='probe-summary pending'><h3>{html.escape(label)}</h3>"
            "<p>Pending</p></article>"
        )
    summary = data.get("summary") or {}
    trend = summary.get("loss_trend") or {}
    if mode == "diverse_resampled":
        per_width = trend.get("per_hidden_width") or {}
        decreased = (
            all(row.get("decreased") is True for row in per_width.values())
            if per_width
            else trend.get("decreased") is True
        )
        passed = decreased and summary.get("diversity_passed") is True
        gate_label = "Trend + diversity gate"
    else:
        passed = trend.get("hard_gate_passed") is True
        gate_label = "Trend gate"
    ratio = trend.get("last_to_first_ratio")
    return (
        f"<article class='probe-summary {'passed' if passed else 'failed'}'>"
        f"<h3>{html.escape(label)}</h3>"
        f"<dl><dt>Steps</dt><dd>{len(records)}</dd>"
        f"<dt>Distinct structures</dt><dd>{html.escape(_fmt(summary.get('distinct_structure_count', 'N/A')))}</dd>"
        f"<dt>First-window median</dt><dd>{html.escape(_fmt(trend.get('first_window_median', 'N/A')))}</dd>"
        f"<dt>Last-window median</dt><dd>{html.escape(_fmt(trend.get('last_window_median', 'N/A')))}</dd>"
        f"<dt>Last / first</dt><dd>{html.escape(_fmt(ratio if ratio is not None else 'N/A'))}</dd>"
        f"<dt>{gate_label}</dt><dd>{'passed' if passed else 'not passed'}</dd></dl></article>"
    )


def _bypass_overfit_section(data: dict[str, Any]) -> str:
    units = data.get("units") or []
    modes = data.get("modes") or {}
    if not units or not any(mode.get("records") for mode in modes.values()):
        return "<p class='empty'>Pending bypass overfit.</p>"
    options = "".join(
        f"<option value='{html.escape(unit)}'>layer_{html.escape(unit)}</option>"
        for unit in units
    )
    unit_label = "Subblock" if data.get("granularity") == "subblock" else "Layer"
    cards = "".join(
        _overfit_summary_card(mode, modes.get(mode) or {})
        for mode in ("diverse_resampled", "smallest_fixed")
    )
    return (
        "<p class='note'>Both probes independently load the same sorted checkpoint and reuse "
        "one fixed batch. The smallest probe fixes its nested structure; the diverse probe "
        "resamples a complete legal nested structure every optimizer step.</p>"
        f"<div class='probe-summaries'>{cards}</div>"
        f"<label class='selector-label' for='bypass-overfit-unit-select'>{unit_label}</label>"
        f'<select id="bypass-overfit-unit-select">{options}</select>'
        "<div class='probe-plots'>"
        "<article class='probe-plot-panel'><h3>Diverse resampled</h3>"
        "<div id='bypass-diverse-plot' class='plotly-chart' role='img' "
        "aria-label='Diverse resampled bypass overfit loss'></div></article>"
        "<article class='probe-plot-panel'><h3>Smallest fixed</h3>"
        "<div id='bypass-fixed-plot' class='plotly-chart' role='img' "
        "aria-label='Smallest fixed bypass overfit loss'></div></article></div>"
    )


def _nested_bypass_data(root: Path) -> dict[str, Any]:
    history_path = root / "artifacts" / "bypass" / "local_kd_loss_history.json"
    history = _load_optional(history_path)
    records = [row for row in history.get("records", ()) if isinstance(row, dict)]
    observation_path = Path(
        history.get("dp_observation_path") or history_path.with_name("dp_observations.jsonl")
    )
    observations: list[dict[str, Any]] = []
    if observation_path.is_file():
        for line in observation_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            observations.extend(
                item for item in payload.get("observations", ()) if isinstance(item, dict)
            )
    catalog_path = Path(
        history.get("candidate_catalog_path") or history_path.with_name("candidate_catalog.json")
    )
    candidate_catalog = _load_optional(catalog_path)
    layers = sorted(
        {str(layer_idx) for row in records for layer_idx in (row.get("per_layer_loss") or {})},
        key=lambda value: int(value),
    )
    subblocks = sorted(
        {str(subblock) for row in records for subblock in (row.get("per_subblock_loss") or {})},
        key=lambda value: (
            int(value.split(":", 1)[0]),
            value.split(":", 2)[1:],
        ),
    )
    granularity = (
        str(observations[0].get("granularity"))
        if observations
        else "subblock"
        if subblocks
        else "block"
    )
    units = sorted(
        {
            (
                int(item["layer_idx"]),
                str(item.get("subblock_kind") or ""),
                str(item.get("subblock_name") or ""),
            )
            for item in observations
        }
        or (
            {(int(value.split(":", 1)[0]), *value.split(":", 2)[1:]) for value in subblocks}
            if granularity == "subblock"
            else {(int(value), "", "") for value in layers}
        )
    )
    return {
        "path": str(history_path),
        "loss_name": str(history.get("loss_name") or "local_kd_loss"),
        "max_steps": int(history.get("max_steps") or len(records)),
        "layers": layers,
        "subblocks": subblocks,
        "records": records,
        "granularity": granularity,
        "units": [
            {"layer_idx": layer, "subblock_kind": kind or None, "subblock_name": name or None}
            for layer, kind, name in units
        ],
        "observations": observations,
        "observation_path": str(observation_path),
        "candidate_catalog": candidate_catalog,
        "candidate_catalog_path": str(catalog_path),
    }


def _nested_bypass_section(data: dict[str, Any]) -> str:
    records = data.get("records") or []
    units = data.get("units") or []
    observations = data.get("observations") or []
    if not records or not units:
        return "<p class='empty'>Pending nested bypass.</p>"
    granularity = str(data.get("granularity") or "block")
    selection_note = (
        "Exact semantic subblock losses are shown."
        if granularity == "subblock"
        else "Exact block-local losses are shown."
    )
    color_note = (
        " Point color is active parameters normalized by the matching teacher "
        f"{granularity}; blue is 0 and red is 1."
        if observations
        else " Legacy aggregate records do not contain normalized DP-lane parameter colors."
    )
    options = []
    for unit in units:
        layer = int(unit["layer_idx"])
        kind = unit.get("subblock_kind")
        name = unit.get("subblock_name")
        label = f"layer_{layer}" if granularity == "block" else f"layer_{layer}:{kind}:{name}"
        options.append(
            f'<option value="{html.escape(json.dumps(unit, sort_keys=True), quote=True)}">'
            f"{html.escape(label)}</option>"
        )
    selector_label = "Sublayer" if granularity == "subblock" else "Layer"
    return (
        f"<p class='note'>{len(records)} elastic optimizer steps. {selection_note}{color_note}</p>"
        f'<label class="selector-label" for="nested-bypass-unit-select">{selector_label}</label>'
        f'<select id="nested-bypass-unit-select">{"".join(options)}</select>'
        '<label class="selector-label" for="nested-bypass-config-select">Layer configuration</label>'
        '<select id="nested-bypass-config-select"><option value="ALL">ALL</option></select>'
        "<article class='probe-plot-panel'>"
        "<h3 id='nested-bypass-unit-title'></h3>"
        "<div id='nested-bypass-unit-plot' class='plotly-chart' role='img' "
        "aria-label='Nested bypass selected-unit loss'></div>"
        "</article>"
    )


def _all_numeric_metrics(payload: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in payload.items():
        number = _result_metric_value(value)
        if number is not None:
            metrics[str(key)] = number
    return metrics


def _depth_data(root: Path) -> dict[str, Any]:
    output = root / "depth" / "iterative"
    trajectory = _load_optional(output / "trajectory.json")
    selected = [row for row in trajectory.get("selected", ()) if isinstance(row, dict)]
    rows: list[dict[str, Any]] = []
    teacher = _load_optional(output / "teacher.json")
    if teacher:
        rows.append({"removed_count": 0, "removals": [], "metrics": _all_numeric_metrics(teacher)})
    for index, removal in enumerate(selected):
        result = _load_optional(
            output
            / f"iteration_{index:02d}"
            / f"candidate_layer_{int(removal['layer_idx']):03d}_{removal['kind']}.json"
        )
        if not result:
            continue
        rows.append(
            {
                "removed_count": index + 1,
                "removals": selected[: index + 1],
                "metrics": _all_numeric_metrics(result),
            }
        )
    available = {
        metric
        for row in rows
        for metric in (row.get("metrics") or {})
        if metric not in {"iteration", "baseline_lm_loss"}
    }
    metrics = [metric for metric in _ACTIVATION_METRIC_ORDER if metric in available]
    metrics.extend(sorted(available - set(metrics)))
    return {
        "trajectory_path": str(output / "trajectory.json"),
        "status": trajectory.get("status"),
        "max_removals": trajectory.get("max_removals"),
        "metrics": metrics,
        "rows": rows,
    }


def _depth_section(data: dict[str, Any]) -> str:
    rows = data.get("rows") or []
    metrics = data.get("metrics") or []
    if not rows or not metrics:
        return "<p class='empty'>Pending iterative depth pruning.</p>"
    default = "lm_loss" if "lm_loss" in metrics else metrics[0]
    ordered = [default, *(metric for metric in metrics if metric != default)]
    options = "".join(
        f"<option value='{html.escape(metric)}'>{html.escape(metric)}</option>"
        for metric in ordered
    )
    return (
        "<p class='note'>Point 0 is the full scoring parent. Each later point adds the "
        "next sublayer selected by the iterative ranking.</p>"
        "<label class='selector-label' for='depth-metric-select'>Metric</label>"
        f"<select id='depth-metric-select'>{options}</select>"
        "<div id='depth-trajectory-plot' class='plotly-chart depth-chart' role='img' "
        "aria-label='Iterative depth pruning metric trajectory'></div>"
    )


_VLLM_METRICS = (
    "runtime_ms",
    "prefill_runtime_ms",
    "decode_runtime_ms",
    "decode_runtime_ms_per_token",
    "weight_memory_mib",
    "kv_cache_bytes_per_token",
    "state_cache_bytes_per_sequence",
    "prefill_flops",
    "decode_flops",
    "num_params",
    "active_params",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _config_subblocks(block_config: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(value) for value in block_config.get("subblock_configs", ()) if isinstance(value, dict)
    ]


def _block_type_label(kinds: tuple[str, ...]) -> str:
    mixer = next((kind for kind in kinds if kind != "ffn"), kinds[0] if kinds else "block")
    return {
        "attention": "Attention blocks",
        "mamba": "GDN blocks",
        "moe": "MoE blocks",
        "mla": "MLA blocks",
    }.get(mixer, f"{mixer.upper()} blocks")


_AXIS_LABELS = {
    "hidden_width": "Embedding width",
    "ffn_intermediate": "FFN size",
    "kv_groups": "KV groups",
    "q_heads_per_group": "Query heads/group",
    "qk_head_dim": "Q/K head dimension",
    "sliding_window_size": "Attention window",
    "gdn_key_groups": "GDN key groups",
    "gdn_value_heads_per_group": "GDN value heads/group",
    "gdn_key_head_dim": "GDN key-head dimension",
    "gdn_value_head_dim": "GDN value-head dimension",
    "moe_experts": "MoE experts",
    "moe_expert_intermediate": "MoE expert size",
    "moe_shared_expert_intermediate": "Shared-expert size",
    "moe_top_k": "MoE top-k",
    "moe_latent_dim": "MoE latent dimension",
}

_AXIS_ORDER = (
    "hidden_width",
    "kv_groups",
    "q_heads_per_group",
    "qk_head_dim",
    "sliding_window_size",
    "gdn_key_groups",
    "gdn_value_heads_per_group",
    "gdn_key_head_dim",
    "gdn_value_head_dim",
    "moe_experts",
    "moe_top_k",
    "moe_expert_intermediate",
    "moe_shared_expert_intermediate",
    "moe_latent_dim",
    "ffn_intermediate",
)


def _axis_label(axis: str) -> str:
    return _AXIS_LABELS.get(axis, axis.replace("_", " ").title())


def _subblock_axes(subblock: dict[str, Any]) -> dict[str, Any]:
    kind = str(subblock.get("kind", "unknown"))
    axes: dict[str, Any] = {}
    consumed = {"kind", "name", "no_op", "conv_kernel_size"}
    if kind == "ffn" and subblock.get("intermediate_size") is not None:
        axes["ffn_intermediate"] = subblock["intermediate_size"]
        consumed.add("intermediate_size")
    elif kind == "attention":
        kv = subblock.get("num_kv_heads")
        query = subblock.get("num_query_heads")
        if kv is not None:
            axes["kv_groups"] = kv
        if kv and query is not None:
            ratio = float(query) / float(kv)
            axes["q_heads_per_group"] = int(ratio) if ratio.is_integer() else ratio
        for field, axis in (
            ("qk_head_dim", "qk_head_dim"),
            ("sliding_window_size", "sliding_window_size"),
        ):
            if subblock.get(field) is not None:
                axes[axis] = subblock[field]
        consumed.update({"num_kv_heads", "num_query_heads", "qk_head_dim", "sliding_window_size"})
    elif kind == "mamba":
        groups = subblock.get("num_groups")
        heads = subblock.get("num_heads")
        if groups is not None:
            axes["gdn_key_groups"] = groups
        if groups and heads is not None:
            ratio = float(heads) / float(groups)
            axes["gdn_value_heads_per_group"] = int(ratio) if ratio.is_integer() else ratio
        if subblock.get("state_dim") is not None:
            axes["gdn_key_head_dim"] = subblock["state_dim"]
        if subblock.get("head_dim") is not None:
            axes["gdn_value_head_dim"] = subblock["head_dim"]
        consumed.update({"num_groups", "num_heads", "state_dim", "head_dim"})
    elif kind == "moe":
        for field, axis in (
            ("num_experts", "moe_experts"),
            ("expert_intermediate_size", "moe_expert_intermediate"),
            ("shared_expert_intermediate_size", "moe_shared_expert_intermediate"),
            ("top_k", "moe_top_k"),
            ("latent_dim", "moe_latent_dim"),
        ):
            if subblock.get(field) is not None:
                axes[axis] = subblock[field]
            consumed.add(field)
    for field, value in sorted(subblock.items()):
        if field in consumed or value is None or isinstance(value, (dict, list)):
            continue
        if isinstance(value, (int, float, bool, str)):
            axes[f"{kind}_{field}"] = value
    return axes


def _block_axes(block: dict[str, Any]) -> dict[str, Any]:
    axes: dict[str, Any] = {}
    for subblock in _config_subblocks(block):
        axes.update(_subblock_axes(subblock))
    return axes


def _normalized_block(block: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(block)
    normalized["subblock_configs"] = sorted(
        _config_subblocks(block), key=lambda value: (str(value.get("kind")), str(value.get("name")))
    )
    return normalized


def _axis_sort_key(axis: str) -> tuple[int, str]:
    return (_AXIS_ORDER.index(axis) if axis in _AXIS_ORDER else len(_AXIS_ORDER), axis)


def _library_scenario(path: Path) -> dict[str, Any]:
    payload = _load_optional(path)
    entries = [entry for entry in payload.get("entries", ()) if isinstance(entry, dict)]
    by_signature: dict[tuple[str, ...], dict[int, dict[str, dict[str, Any]]]] = {}
    runtime_configs: set[str] = set()
    for entry in entries:
        children = entry.get("child_block_configs") or []
        parents = entry.get("parent_layer_indices") or []
        if not children or not parents or not isinstance(children[0], dict):
            continue
        block = children[0]
        layer = int(parents[0])
        subblocks = _config_subblocks(block)
        # Replacement helpers may move the changed subblock to the end of the tuple.
        # Layer type is a set of subblock kinds, not their serialization order.
        signature = tuple(sorted(str(value.get("kind", "unknown")) for value in subblocks))
        normalized = _normalized_block(block)
        by_signature.setdefault(signature, {}).setdefault(layer, {})[
            _canonical_json(normalized)
        ] = normalized
        runtime_configs.update(_canonical_json(value) for value in subblocks)
    terms: list[dict[str, Any]] = []
    for signature, layers in sorted(by_signature.items(), key=lambda item: item[0]):
        by_option_count: dict[int, int] = {}
        for options in layers.values():
            by_option_count[len(options)] = by_option_count.get(len(options), 0) + 1
        for option_count, layer_count in sorted(by_option_count.items()):
            matching_layers = [
                options for options in layers.values() if len(options) == option_count
            ]
            configs = [block for options in matching_layers for block in options.values()]
            axis_values: dict[str, set[str]] = {}
            decoded_values: dict[str, dict[str, Any]] = {}
            for block in configs:
                for axis, value in _block_axes(block).items():
                    encoded = _canonical_json(value)
                    axis_values.setdefault(axis, set()).add(encoded)
                    decoded_values.setdefault(axis, {})[encoded] = value
            factors = []
            for axis in sorted(axis_values, key=_axis_sort_key):
                values = axis_values[axis]
                if len(values) <= 1:
                    continue
                factors.append(
                    {
                        "axis": axis,
                        "label": _axis_label(axis),
                        "count": len(values),
                        "values": sorted(
                            decoded_values[axis].values(),
                            key=lambda value: (isinstance(value, str), value),
                        ),
                    }
                )
            cartesian_product = 1
            for factor in factors:
                cartesian_product *= int(factor["count"])
            terms.append(
                {
                    "signature": list(signature),
                    "label": _block_type_label(signature),
                    "num_layers": layer_count,
                    "options_per_layer": option_count,
                    "entries": layer_count * option_count,
                    "factors": factors,
                    "cartesian": cartesian_product == option_count,
                }
            )
    return {
        "path": str(path),
        "hidden_width": payload.get("hidden_width"),
        "entries": len(entries),
        "terms": terms,
        "unique_runtime_configs": len(runtime_configs),
    }


def _library_term_formula(term: dict[str, Any]) -> str:
    prefix = f"{term['num_layers']} {str(term['label']).replace('blocks', 'layers')}"
    if not term.get("cartesian"):
        return f"{prefix} × {term['options_per_layer']} valid configurations"

    def plural(label: str) -> str:
        if label.endswith("size"):
            return f"{label}s"
        if label.endswith("dimension"):
            return f"{label}s"
        return label

    factors = " × ".join(
        f"{factor['count']} {plural(str(factor['label'])).lower()}"
        for factor in term.get("factors", ())
    )
    return f"{prefix} × {factors}" if factors else f"{prefix} × 1 configuration"


def _library_term_values(term: dict[str, Any]) -> str:
    if not term.get("factors"):
        return "No varying numeric axes"
    return "; ".join(f"{factor['label']} = {factor['values']}" for factor in term["factors"])


def _library_data(root: Path) -> dict[str, Any]:
    paths = sorted(root.glob("scenarios/width-*/depth-00/replacement_library.json"))
    if not paths and (root / "replacement_library.json").is_file():
        paths = [root / "replacement_library.json"]
    scenarios = [_library_scenario(path) for path in paths]
    total_entries = sum(int(scenario["entries"]) for scenario in scenarios)
    unique_runtime_configs = sum(int(row["unique_runtime_configs"]) for row in scenarios)
    formula = "Pending library creation"
    if scenarios:
        first_terms = scenarios[0]["terms"]
        if all(scenario["terms"] == first_terms for scenario in scenarios):
            inside = " + ".join(_library_term_formula(term) for term in first_terms)
            formula = f"({inside}) × {len(scenarios)} embedding widths = {total_entries}"
        else:
            formula = (
                " + ".join(f"width {row['hidden_width']}: {row['entries']}" for row in scenarios)
                + f" = {total_entries}"
            )
    return {
        "scenarios": scenarios,
        "formula": formula,
        "total_entries": total_entries,
        "num_widths": len(scenarios),
        "unique_runtime_configs": unique_runtime_configs,
    }


def _library_section(data: dict[str, Any]) -> str:
    scenarios = data.get("scenarios") or []
    if not scenarios:
        return "<p class='empty'>Pending block-library creation.</p>"
    rows = []
    for scenario in scenarios:
        for term in scenario.get("terms", ()):
            rows.append(
                "<tr>"
                f"<th>{html.escape(_fmt(scenario.get('hidden_width')))}</th>"
                f"<td>{html.escape(str(term.get('label')))}</td>"
                f"<td>{html.escape(_library_term_formula(term))}</td>"
                f"<td>{html.escape(_library_term_values(term))}</td>"
                f"<td>{int(term.get('entries', 0))}</td>"
                "</tr>"
            )
    return (
        f"<p class='cardinality-formula'>{html.escape(str(data.get('formula')))}</p>"
        f"<p class='note'>The full libraries contain {int(data.get('total_entries', 0))} "
        "layer-specific Cartesian block candidates. Subblock-level vLLM collection "
        f"deduplicates these to {int(data.get('unique_runtime_configs', 0))} width-specific "
        "runtime configurations.</p>"
        "<div class='table-wrap'><table><thead><tr><th>Hidden width</th>"
        "<th>Layer family</th><th>Cartesian decomposition</th><th>Axis values</th>"
        "<th>Library entries</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _compact_runtime_config(config: dict[str, Any]) -> str:
    kind = str(config.get("kind", "unknown"))
    ignored = {"kind", "name", "no_op", "conv_kernel_size"}
    values = ", ".join(
        f"{key.replace('intermediate_size', 'dim').replace('num_', '')}={value}"
        for key, value in sorted(config.items())
        if key not in ignored and value is not None
    )
    return f"{kind}: {values}" if values else kind


def _vllm_data(root: Path, library: dict[str, Any]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    scenarios: list[dict[str, Any]] = []
    runtime_profiles: dict[str, dict[str, Any]] = {}
    for scenario in library.get("scenarios", ()):
        width = scenario.get("hidden_width")
        stats_path = Path(scenario["path"]).with_name("subblock_stats.json")
        payload = json.loads(stats_path.read_text()) if stats_path.is_file() else []
        profiles = (
            [
                row
                for row in payload
                if isinstance(row, dict) and (row.get("args") or {}).get("runtime_stats") is True
            ]
            if isinstance(payload, list)
            else []
        )
        seen: set[str] = set()
        for profile in profiles:
            args = profile.get("args") or {}
            concurrency = args.get("max_num_seqs") or args.get("batch_size")
            profile_spec = {
                "input_sequence_length": args.get("prefill_seq_len"),
                "output_sequence_length": args.get("generation_seq_len"),
                "concurrency": concurrency,
            }
            profile_id = _canonical_json(profile_spec)
            profile_label = (
                f"ISL {profile_spec['input_sequence_length']} · "
                f"OSL {profile_spec['output_sequence_length']} · "
                f"concurrency {profile_spec['concurrency']}"
            )
            runtime_profiles.setdefault(
                profile_id,
                {"id": profile_id, "label": profile_label, **profile_spec},
            )
            profile_seen: set[str] = set()
            for row in profile.get("subblocks", ()):
                config = row.get("subblock_config") or {}
                key = _canonical_json(config)
                if key in profile_seen:
                    continue
                profile_seen.add(key)
                seen.add(key)
                metrics = {
                    metric: float(row[metric])
                    for metric in _VLLM_METRICS
                    if isinstance(row.get(metric), (int, float))
                    and not isinstance(row.get(metric), bool)
                }
                records.append(
                    {
                        "hidden_width": width,
                        "kind": str(config.get("kind", "unknown")),
                        "config": config,
                        "label": _compact_runtime_config(config),
                        "axes": {"hidden_width": width, **_subblock_axes(config)},
                        "metrics": metrics,
                        "profile": args,
                        "profile_id": profile_id,
                        "profile_label": profile_label,
                    }
                )
        expected = int(scenario.get("unique_runtime_configs", 0))
        scenarios.append(
            {
                "hidden_width": width,
                "path": str(stats_path),
                "expected": expected,
                "measured": len(seen),
                "complete": expected > 0 and len(seen) == expected,
                "profiles": len(profiles),
            }
        )
    metrics = [
        metric
        for metric in _VLLM_METRICS
        if any(metric in record.get("metrics", {}) for record in records)
    ]
    observed_axes = {
        axis
        for record in records
        for axis in (record.get("axes") or {})
        if len(
            {
                _canonical_json((candidate.get("axes") or {}).get(axis))
                for candidate in records
                if axis in (candidate.get("axes") or {})
            }
        )
        > 1
    }
    axes = sorted(observed_axes, key=_axis_sort_key)
    return {
        "scenarios": scenarios,
        "records": records,
        "metrics": metrics,
        "axes": axes,
        "axis_labels": {axis: _axis_label(axis) for axis in axes},
        "widths": sorted({record.get("hidden_width") for record in records}),
        "kinds": sorted({str(record.get("kind")) for record in records}),
        "profiles": sorted(runtime_profiles.values(), key=lambda row: str(row["label"])),
    }


def _vllm_section(data: dict[str, Any]) -> str:
    scenarios = data.get("scenarios") or []
    records = data.get("records") or []
    if not records:
        return "<p class='empty'>Pending vLLM statistics.</p>"
    cards = "".join(
        "<article class='probe-summary {}'><h3>Width {}</h3><dl>"
        "<dt>Expected configs</dt><dd>{}</dd><dt>Measured configs</dt><dd>{}</dd>"
        "<dt>Runtime profiles</dt><dd>{}</dd><dt>Coverage</dt><dd>{}</dd>"
        "</dl></article>".format(
            "passed" if scenario.get("complete") else "failed",
            html.escape(_fmt(scenario.get("hidden_width"))),
            int(scenario.get("expected", 0)),
            int(scenario.get("measured", 0)),
            int(scenario.get("profiles", 0)),
            "complete" if scenario.get("complete") else "incomplete",
        )
        for scenario in scenarios
    )
    metric_options = "".join(
        f"<option value='{html.escape(metric)}'>{html.escape(metric)}</option>"
        for metric in data.get("metrics", ())
    )
    axis_options = "".join(
        f"<option value='{html.escape(axis)}'>{html.escape(str((data.get('axis_labels') or {}).get(axis, axis)))}</option>"
        for axis in data.get("axes", ())
    )
    profile_options = "".join(
        f"<option value='{html.escape(str(profile.get('id')))}'>"
        f"{html.escape(str(profile.get('label')))}</option>"
        for profile in data.get("profiles", ())
    )
    return (
        f"<div class='probe-summaries'>{cards}</div>"
        "<div class='vllm-overview-cards'>"
        f"<article><strong>{len(records)}</strong><span>Measured configurations</span></article>"
        f"<article><strong>{len(data.get('widths', ()))}</strong><span>Embedding widths</span></article>"
        f"<article><strong>{len(data.get('axes', ()))}</strong><span>Swept axes</span></article>"
        f"<article><strong>{len(data.get('metrics', ()))}</strong><span>Metrics</span></article>"
        "</div>"
        "<div class='vllm-controls'><label>Runtime profile"
        f"<select id='vllm-profile-select'>{profile_options}</select></label></div>"
        "<article class='probe-plot-panel vllm-panel'><h3>All collected candidates</h3>"
        "<label class='selector-label' for='vllm-overview-metric'>Metric</label>"
        f"<select id='vllm-overview-metric'>{metric_options}</select>"
        "<div id='vllm-overview-plot' class='plotly-chart depth-chart' role='img' "
        "aria-label='All collected vLLM candidates'></div></article>"
        "<article class='probe-plot-panel vllm-panel'><h3>Sweep explorer</h3>"
        "<div class='vllm-controls'>"
        "<label>Metric"
        f"<select id='vllm-metric-select'>{metric_options}</select></label>"
        "<label>Swept axis"
        f"<select id='vllm-axis-select'>{axis_options}</select></label>"
        "<label>Compatible configuration<select id='vllm-config-select'>"
        "<option value='ALL'>ALL</option></select></label></div>"
        "<p id='vllm-config-summary' class='note'></p>"
        "<div id='vllm-stats-plot' class='plotly-chart depth-chart' role='img' "
        "aria-label='vLLM metric sweep across compatible configurations and widths'></div>"
        "</article>"
    )


def _replacement_metrics(payload: dict[str, Any]) -> dict[str, float]:
    ignored = {
        "args",
        "distributed_evaluation",
        "hidden_width",
        "i_solution",
        "observability",
        "puzzle_solution",
        "sliced_teacher_baseline",
    }
    metrics: dict[str, float] = {}
    for key, value in payload.items():
        if key in ignored:
            continue
        number = _result_metric_value(value)
        if number is not None:
            metrics[str(key)] = number
    return metrics


def _compact_block_config(block: dict[str, Any]) -> str:
    return " + ".join(_compact_runtime_config(row) for row in _config_subblocks(block))


def _replacement_data(root: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    scenarios: list[dict[str, Any]] = []
    for scenario_dir in sorted(root.glob("scenarios/width-*/depth-00")):
        manifest = _load_optional(scenario_dir / "scenario_manifest.json")
        width = manifest.get("hidden_width")
        definitions_path = scenario_dir / "single_sequence_replacement_solutions.json"
        definitions = json.loads(definitions_path.read_text()) if definitions_path.is_file() else []
        if not isinstance(definitions, list):
            definitions = []
        result_dir = scenario_dir / "single_sequence_replacement_solutions--validation"
        measured = 0
        for path in sorted(result_dir.glob("solution_*.json")):
            payload = _load_optional(path)
            try:
                index = int(payload.get("i_solution", path.stem.rsplit("_", 1)[-1]))
            except (TypeError, ValueError):
                continue
            if not (0 <= index < len(definitions)):
                continue
            replacement = (definitions[index] or {}).get("single_sequence_replacement") or {}
            parents = replacement.get("parent_layer_indices") or []
            children = replacement.get("child_block_configs") or []
            if not parents or not children or not isinstance(children[0], dict):
                continue
            layer = int(parents[0])
            block = children[0]
            kinds = tuple(
                sorted(str(row.get("kind", "unknown")) for row in _config_subblocks(block))
            )
            metrics = _replacement_metrics(payload)
            records.append(
                {
                    "hidden_width": width,
                    "solution_index": index,
                    "layer_idx": layer,
                    "eval_samples": (payload.get("args") or {}).get("eval_samples"),
                    "family": _block_type_label(kinds),
                    "label": _compact_block_config(block),
                    "config": _normalized_block(block),
                    "axes": {"hidden_width": width, **_block_axes(block)},
                    "metrics": metrics,
                    "sliced_teacher_baseline": _all_numeric_metrics(
                        payload.get("sliced_teacher_baseline") or {}
                    ),
                }
            )
            measured += 1
        expected = len(definitions)
        if expected > 0 or measured > 0:
            scenarios.append(
                {
                    "hidden_width": width,
                    "path": str(result_dir),
                    "expected": expected,
                    "measured": measured,
                    "complete": expected > 0 and measured == expected,
                }
            )
    granularity = "block"
    if not records:
        atomic_manifest = _load_optional(root / "subblock_replacement_manifest.json")
        result_dir = root / "single_subblock_replacement_solutions--validation"
        if atomic_manifest.get("mode") == "replace_one_subblock" and result_dir.is_dir():
            granularity = "subblock"
            for path in sorted(result_dir.glob("solution_*.json")):
                payload = _load_optional(path)
                puzzle_solution = payload.get("puzzle_solution") or {}
                replacement = puzzle_solution.get("single_sequence_replacement") or {}
                marker = puzzle_solution.get("subblock_replacement") or {}
                parents = replacement.get("parent_layer_indices") or []
                children = replacement.get("child_block_configs") or []
                if not parents or not children or not isinstance(children[0], dict):
                    continue
                block = children[0]
                kind = str(marker.get("kind", "unknown"))
                name = str(marker.get("name", kind))
                selected = next(
                    (
                        subblock
                        for subblock in _config_subblocks(block)
                        if str(subblock.get("kind")) == kind
                        and str(subblock.get("name", kind)) == name
                    ),
                    None,
                )
                if selected is None:
                    continue
                index = int(payload.get("i_solution", path.stem.rsplit("_", 1)[-1]))
                records.append(
                    {
                        "hidden_width": atomic_manifest.get("hidden_width"),
                        "solution_index": index,
                        "layer_idx": int(parents[0]),
                        "eval_samples": (payload.get("args") or {}).get("eval_samples"),
                        "family": kind,
                        "label": _compact_runtime_config(selected),
                        "config": _normalized_block(block),
                        "axes": _subblock_axes(selected),
                        "metrics": _replacement_metrics(payload),
                        "sliced_teacher_baseline": _all_numeric_metrics(
                            payload.get("sliced_teacher_baseline") or {}
                        ),
                        "granularity": "subblock",
                        "subblock_kind": kind,
                        "subblock_name": name,
                        "provenance": {
                            "method": "atomic_subblock_measurement",
                            "source_result_id": str(
                                (payload.get("distributed_evaluation") or {}).get(
                                    "request_id", index
                                )
                            ),
                        },
                    }
                )
            expected = int(atomic_manifest.get("subblock_solution_count", 0) or 0)
            scenarios.append(
                {
                    "hidden_width": atomic_manifest.get("hidden_width"),
                    "path": str(result_dir),
                    "expected": expected,
                    "measured": len(records),
                    "complete": expected > 0 and len(records) == expected,
                    "granularity": "subblock",
                }
            )
    available = {metric for record in records for metric in (record.get("metrics") or {})}
    metrics = [metric for metric in _ACTIVATION_METRIC_ORDER if metric in available]
    metrics.extend(sorted(available - set(metrics)))
    observed_axes = {
        axis
        for record in records
        for axis in (record.get("axes") or {})
        if axis != "hidden_width"
        if len(
            {
                _canonical_json((candidate.get("axes") or {}).get(axis))
                for candidate in records
                if axis in (candidate.get("axes") or {})
            }
        )
        > 1
    }
    axes = sorted(observed_axes, key=_axis_sort_key)
    return {
        "granularity": granularity,
        "scenarios": scenarios,
        "records": records,
        "metrics": metrics,
        "axes": axes,
        "axis_labels": {axis: _axis_label(axis) for axis in axes},
        "widths": sorted({record.get("hidden_width") for record in records}),
        "layers": sorted({int(record["layer_idx"]) for record in records}),
        "eval_samples": sorted(
            {
                int(record["eval_samples"])
                for record in records
                if record.get("eval_samples") is not None
            }
        ),
    }


def _replacement_section(data: dict[str, Any]) -> str:
    records = data.get("records") or []
    if not records:
        return "<p class='empty'>Pending replacement scoring.</p>"
    cards = "".join(
        "<article class='probe-summary {}'><h3>Width {}</h3><dl>"
        "<dt>Expected candidates</dt><dd>{}</dd><dt>Scored candidates</dt><dd>{}</dd>"
        "<dt>Coverage</dt><dd>{}</dd></dl></article>".format(
            "passed" if scenario.get("complete") else "failed",
            html.escape(_fmt(scenario.get("hidden_width"))),
            int(scenario.get("expected", 0)),
            int(scenario.get("measured", 0)),
            "complete" if scenario.get("complete") else "incomplete",
        )
        for scenario in data.get("scenarios", ())
    )
    metric_options = "".join(
        f"<option value='{html.escape(metric)}'>{html.escape(metric)}</option>"
        for metric in data.get("metrics", ())
    )
    axis_options = "".join(
        f"<option value='{html.escape(axis)}'>{html.escape(str((data.get('axis_labels') or {}).get(axis, axis)))}</option>"
        for axis in data.get("axes", ())
    )
    width_options = "".join(
        f"<option value='{html.escape(str(width))}'>{html.escape(str(width))}</option>"
        for width in sorted(data.get("widths", ()), reverse=True)
    )
    layer_cells = "".join(
        "<label class='layer-toggle'><input type='checkbox' checked "
        f"data-replacement-layer='{int(layer)}'>Layer {int(layer)}</label>"
        for layer in data.get("layers", ())
    )
    first_layer = min(data.get("layers", ()))
    last_layer = max(data.get("layers", ()))
    return (
        f"<div class='probe-summaries'>{cards}</div>"
        "<div class='vllm-overview-cards'>"
        f"<article><strong>{len(records)}</strong><span>Candidate scores</span></article>"
        f"<article><strong>{len(data.get('widths', ()))}</strong><span>Embedding widths</span></article>"
        f"<article><strong>{len(data.get('layers', ()))}</strong><span>Layers</span></article>"
        f"<article><strong>{html.escape(', '.join(map(str, data.get('eval_samples', ()))))}</strong><span>Samples per candidate</span></article>"
        f"<article><strong>{len(data.get('metrics', ()))}</strong><span>Metrics</span></article>"
        "</div><article class='probe-plot-panel vllm-panel'><h3>Replacement-score explorer</h3>"
        "<p class='note'><strong>Baseline:</strong> the scoring parent sliced to the selected "
        "embedding width, with no block replacement, evaluated against the full-width scoring "
        "target. Candidate points apply one block replacement to that sliced model.</p>"
        "<div class='vllm-controls'><label>Metric"
        f"<select id='replacement-metric-select'>{metric_options}</select></label>"
        "<label>Embedding width"
        f"<select id='replacement-width-select'>{width_options}</select></label>"
        "<label>Swept axis"
        f"<select id='replacement-axis-select'>{axis_options}</select></label>"
        "<label>Compatible configuration<select id='replacement-config-select'>"
        "<option value='ALL'>ALL</option></select></label></div>"
        "<p id='replacement-config-summary' class='note'></p>"
        "<label class='replacement-connect-toggle'><input id='replacement-connect-layers' "
        "type='checkbox' checked> Connect points within each layer</label>"
        "<div class='replacement-layer-toolbar'><label class='replacement-all-toggle'>"
        "<input id='replacement-all-layers' type='checkbox' checked> All layers</label>"
        "<div class='replacement-layer-color-key'>"
        f"<span>Layer {int(first_layer)}</span><i></i><span>Layer {int(last_layer)}</span>"
        "</div></div>"
        f"<div class='replacement-layer-grid'>{layer_cells}</div>"
        "<div id='replacement-score-plot' class='plotly-chart depth-chart' role='img' "
        "aria-label='Replace-one-block scores across layers, widths, and configurations'></div>"
        "</article>"
    )


_MIP_COLUMN_ORDER = (
    "status",
    "sliced_teacher_baseline",
    "solver_objective_sum",
    "num_params",
    "parameter_ratio",
    "memory_mib",
    "active_params",
    "num_experts",
    "top_k",
    "runtime_ms",
    "prefill_runtime_ms",
    "decode_runtime_ms",
    "decode_runtime_ms_per_token",
    "throughput",
    "weight_memory_mib",
    "kv_cache_memory_mib",
    "kv_cache_bytes_per_token",
    "state_cache_bytes_per_sequence",
    "prefill_flops",
    "decode_flops",
    "num_kv_heads",
    "num_query_heads",
    "has_attention",
    "has_mamba",
    "has_ffn",
    "has_moe",
    "not_no_op",
    "chosen_replacement_count",
)


def _mip_outputs(row: dict[str, Any], *, runtime_profile: dict[str, Any]) -> dict[str, Any]:
    outputs: dict[str, Any] = {"status": row.get("status")}
    for key, value in (row.get("total_costs") or {}).items():
        name = str(key).removeprefix("stats.")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            outputs[name] = value
    for key in (
        "sliced_teacher_baseline",
        "solver_objective_sum",
        "parameter_ratio",
        "chosen_replacement_count",
    ):
        value = row.get(key)
        if value is not None:
            outputs[key] = value
    if "throughput" not in outputs and isinstance(outputs.get("runtime_ms"), (int, float)):
        batch = runtime_profile.get("batch_size")
        generation = runtime_profile.get("generation_seq_len")
        if batch and generation and outputs["runtime_ms"]:
            outputs["throughput"] = (
                1000 * float(batch) * float(generation) / float(outputs["runtime_ms"])
            )
    return outputs


def _mip_data(root: Path) -> dict[str, Any]:
    profiles: list[dict[str, Any]] = []
    available_columns: set[str] = set()
    for path in sorted((root / "mip" / "profiles").glob("*/mip_grid.json")):
        payload = _load_optional(path)
        profile = payload.get("profile") or {}
        profile_id = str(profile.get("id") or path.parent.name)
        runtime_profile = payload.get("runtime_profile") or {}
        teacher = dict(payload.get("teacher") or {})
        rows = [
            {
                "label": "Teacher",
                "hidden_width": teacher.get("hidden_width"),
                "removed_sublayers": 0,
                "outputs": _mip_outputs(teacher, runtime_profile=runtime_profile),
            }
        ]
        scenarios = [row for row in payload.get("scenarios", ()) if isinstance(row, dict)]
        for scenario in sorted(
            scenarios,
            key=lambda row: (
                -int(row.get("hidden_width", 0)),
                int(row.get("removed_sublayers", 0)),
            ),
        ):
            width = int(scenario.get("hidden_width"))
            depth = int(scenario.get("removed_sublayers", 0))
            rows.append(
                {
                    "label": f"H={width}, Drop={depth}",
                    "hidden_width": width,
                    "removed_sublayers": depth,
                    "outputs": _mip_outputs(scenario, runtime_profile=runtime_profile),
                    "solution_path": scenario.get("solution_path"),
                }
            )
        # The teacher inventory includes internal per-family accounting fields
        # that the solver does not emit.  Columns are defined by real MIP
        # outputs; the teacher contributes values only where names match.
        for row in rows[1:]:
            available_columns.update((row.get("outputs") or {}).keys())
        expected = int(payload.get("expected_scenario_count") or len(scenarios))
        feasible = sum(row.get("status") == "feasible" for row in scenarios)
        terminal = sum(row.get("status") in {"feasible", "infeasible"} for row in scenarios)
        profiles.append(
            {
                "id": profile_id,
                "label": str(profile.get("label") or profile_id),
                "constraint": profile,
                "runtime_profile": runtime_profile,
                "path": str(path),
                "expected": expected,
                "scenario_count": len(scenarios),
                "feasible_count": feasible,
                "terminal_count": terminal,
                "complete": expected > 0 and len(scenarios) == expected and terminal == expected,
                "rows": rows,
            }
        )
    columns = [name for name in _MIP_COLUMN_ORDER if name in available_columns]
    columns.extend(sorted(available_columns - set(columns)))
    return {"profiles": profiles, "columns": columns}


def _mip_section(data: dict[str, Any]) -> str:
    profiles = data.get("profiles") or []
    if not profiles:
        return "<p class='empty'>Pending MIP optimization.</p>"
    options = "".join(
        f"<option value='{html.escape(str(profile.get('id')))}'>"
        f"{html.escape(str(profile.get('label')))}</option>"
        for profile in profiles
    )

    def constraint_display(profile: dict[str, Any]) -> tuple[str, str]:
        constraint = profile.get("constraint") or {}
        if constraint.get("constraint_type") == "latency_ratio":
            return "Latency limit", f"{_fmt(constraint.get('latency_limit_ms'))} ms"
        if constraint.get("constraint_type") == "none":
            return "Constraint", "None (objective-only)"
        return "Parameter limit", _fmt(constraint.get("parameter_limit"))

    cards = "".join(
        "<article class='probe-summary {}'><h3>{}</h3><dl>"
        "<dt>{}</dt><dd>{}</dd><dt>Scenarios</dt><dd>{}/{}</dd>"
        "<dt>Feasible</dt><dd>{}</dd><dt>Coverage</dt><dd>{}</dd>"
        "</dl></article>".format(
            "passed" if profile.get("complete") else "failed",
            html.escape(str(profile.get("label"))),
            html.escape(constraint_display(profile)[0]),
            html.escape(constraint_display(profile)[1]),
            int(profile.get("scenario_count", 0)),
            int(profile.get("expected", 0)),
            int(profile.get("feasible_count", 0)),
            "complete" if profile.get("complete") else "incomplete",
        )
        for profile in profiles
    )
    return (
        f"<div class='probe-summaries'>{cards}</div>"
        "<p class='note'>Each selected constraint ratio is resolved once against the original "
        "full-width teacher, then reused as one absolute parameter or latency bound for every width and depth. "
        "It counts the entire checkpoint, including fixed embeddings, LM head, vision tower, "
        "vision projector, and MTP tensors; ViT-internal tensors are not pruning candidates. "
        "Changing this selector also updates the shared constraint state used by downstream "
        "evaluation, AIPerf, and KD views. Score is baseline + the sum of each selected "
        "block's raw-score delta from the sliced-teacher baseline; Solver Objective Sum is "
        "the unnormalized sum optimized by MIP. Forced depth removals are hard constraints, "
        "so their separately measured iterative-depth degradation is not added.</p>"
        "<div class='vllm-controls'><label>MIP constraint profile"
        f"<select id='mip-constraint-select'>{options}</select></label></div>"
        "<p id='mip-constraint-summary' class='note'></p>"
        "<div class='table-wrap'><table id='mip-solution-table' class='mip-table'>"
        "<thead id='mip-solution-head'></thead><tbody id='mip-solution-body'></tbody>"
        "</table></div>"
    )


def _profile_registry(root: Path, profile_id: str) -> dict[str, Any]:
    return _load_optional(root / "mip" / "profiles" / profile_id / "selected_solutions.json")


def _evaluation_data(root: Path) -> dict[str, Any]:
    profiles = []
    x_metrics: set[str] = set()
    y_metrics: set[str] = set()
    for path in sorted(
        (root / "artifacts" / "zero_shot_evaluation" / "profiles").glob(
            "*/*/evaluation_summary.json"
        )
    ):
        payload = _load_optional(path)
        profile_id = str(payload.get("profile_id") or path.parents[1].name)
        registry = _profile_registry(root, profile_id)
        styles = {row["solution_id"]: row for row in registry.get("solutions", ())}
        rows = []
        for raw in payload.get("solutions", ()):
            solution_id = str(raw.get("solution_id"))
            style = styles.get(solution_id, {})
            x = {}
            for key, value in (raw.get("total_costs") or {}).items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    x[str(key).removeprefix("stats.")] = float(value)
            for key in ("parameter_count", "parameter_ratio", "score"):
                value = raw.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    x[key] = float(value)
            metrics = {
                str(key): float(value)
                for key, value in (raw.get("metrics") or {}).items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
            x_metrics.update(x)
            y_metrics.update(metrics)
            rows.append(
                {
                    **raw,
                    "label": style.get("label", raw.get("label", solution_id)),
                    "color": style.get("color", raw.get("color", "#4f8cff")),
                    "marker": style.get("marker", raw.get("marker", "circle")),
                    "always_enabled": bool(style.get("always_enabled", False)),
                    "x": x,
                    "metrics": metrics,
                }
            )
        profiles.append(
            {
                "profile_id": profile_id,
                "workload_id": path.parent.name,
                "eval_samples": payload.get("eval_samples"),
                "block_size": payload.get("block_size"),
                "rows": rows,
            }
        )
    x_order = (
        "num_params",
        "parameter_count",
        "parameter_ratio",
        "memory_mib",
        "active_params",
        "runtime_ms",
        "prefill_runtime_ms",
        "decode_runtime_ms",
        "decode_runtime_ms_per_token",
        "throughput",
        "weight_memory_mib",
        "kv_cache_memory_mib",
        "prefill_flops",
        "decode_flops",
        "score",
    )
    y_order = _ACTIVATION_METRIC_ORDER
    return {
        "profiles": profiles,
        "x_metrics": [name for name in x_order if name in x_metrics]
        + sorted(x_metrics - set(x_order)),
        "y_metrics": [name for name in y_order if name in y_metrics]
        + sorted(y_metrics - set(y_order)),
    }


def _evaluation_section(data: dict[str, Any]) -> str:
    if not data.get("profiles"):
        return "<p class='empty'>Pending exact evaluation for selected MIP solutions.</p>"
    x_options = "".join(
        f"<option value='{html.escape(name)}'>{html.escape(name)}</option>"
        for name in data.get("x_metrics", ())
    )
    y_options = "".join(
        f"<option value='{html.escape(name)}'>{html.escape(name)}</option>"
        for name in data.get("y_metrics", ())
    )
    return (
        "<p class='note' id='evaluation-profile-summary'></p>"
        "<div class='vllm-controls'><label>X axis"
        f"<select id='evaluation-x-select'>{x_options}</select></label>"
        "<label>Y axis"
        f"<select id='evaluation-y-select'>{y_options}</select></label></div>"
        "<div id='evaluation-scatter-plot' class='plotly-chart depth-chart' role='img' "
        "aria-label='Exact evaluation metrics against architecture costs'></div>"
    )


def _aiperf_data(root: Path) -> dict[str, Any]:
    profiles = []
    for path in sorted(
        (root / "artifacts" / "aiperf" / "profiles").glob("*/isl-*-osl-*/aiperf_results.json")
    ):
        payload = _load_optional(path)
        profile_id = str(payload.get("profile_id") or path.parents[1].name)
        registry = _profile_registry(root, profile_id)
        styles = {row["solution_id"]: row for row in registry.get("solutions", ())}
        rows = []
        for raw in payload.get("results", ()):
            style = styles.get(str(raw.get("solution_id")), {})
            rows.append(
                {
                    **raw,
                    "label": style.get("label", raw.get("solution_id")),
                    "color": style.get("color", "#4f8cff"),
                    "marker": style.get("marker", "circle"),
                    "always_enabled": bool(style.get("always_enabled", False)),
                }
            )
        workload = payload.get("workload") or {}
        profiles.append(
            {
                "profile_id": profile_id,
                "workload_id": path.parent.name,
                "workload": workload,
                "topologies": payload.get("topologies") or [],
                "solutions": list(styles.values()),
                "rows": rows,
            }
        )
    return {"profiles": profiles}


def _aiperf_section(data: dict[str, Any]) -> str:
    profiles = data.get("profiles") or []
    if not profiles:
        return "<p class='empty'>Pending AIPerf measurements for selected MIP solutions.</p>"
    workloads = sorted({row["workload_id"] for row in profiles})
    workload_options = "".join(
        f'<option value="{html.escape(value)}">{html.escape(value)}</option>' for value in workloads
    )
    topology_ids = sorted(
        {str(row.get("topology_id")) for profile in profiles for row in profile.get("rows", ())}
    )
    topology_options = "".join(
        f'<option value="{html.escape(value)}">{html.escape(value)}</option>'
        for value in topology_ids
    )
    styles = {}
    for profile in profiles:
        for style in profile.get("solutions", ()):
            styles.setdefault(style["solution_id"], style)
    toggles = "".join(
        "<label class='layer-toggle'><input type='checkbox' checked "
        f'data-aiperf-solution="{html.escape(str(style["solution_id"]))}" '
        + ("disabled " if style.get("always_enabled") else "")
        + f'><i style="background:{html.escape(str(style.get("color", "#4f8cff")))}"></i>'
        f"{html.escape(str(style.get('label', style['solution_id'])))}</label>"
        for style in styles.values()
    )
    return (
        "<div class='vllm-controls'><label>ISL / OSL"
        f"<select id='aiperf-workload-select'>{workload_options}</select></label>"
        "<label>TP / PP / DP / EP / CP"
        f"<select id='aiperf-topology-select'>{topology_options}"
        '<option value="PARETO">PARETO</option></select></label>'
        "<label>Latency statistic<select id='aiperf-stat-select'>"
        "<option value='mean'>Mean</option><option value='p95'>P95</option>"
        "<option value='p99'>P99</option></select></label></div>"
        "<div class='replacement-layer-toolbar'><div>"
        "<button id='aiperf-all-solutions' type='button'>ALL</button> "
        "<button id='aiperf-no-solutions' type='button'>NONE</button></div>"
        "<span class='note'>Teacher remains enabled.</span></div>"
        f"<div class='replacement-layer-grid aiperf-solution-grid'>{toggles}</div>"
        "<p id='aiperf-profile-summary' class='note'></p>"
        "<div class='probe-plots'>"
        "<article class='probe-plot-panel'><div id='aiperf-ttft-throughput-plot' "
        "class='plotly-chart'></div></article>"
        "<article class='probe-plot-panel'><div id='aiperf-latency-throughput-plot' "
        "class='plotly-chart'></div></article>"
        "<article class='probe-plot-panel'><div id='aiperf-interactivity-throughput-plot' "
        "class='plotly-chart'></div></article>"
        "<article class='probe-plot-panel'><div id='aiperf-tpot-throughput-plot' "
        "class='plotly-chart'></div></article></div>"
    )


_DISTILLATION_METRIC_ORDER = ("loss", "main_ce", "mtp_ce", "main_kd", "mtp_kd")


def _training_records(path: Path) -> list[dict[str, Any]]:
    """Read an append-only training log without failing on an in-flight tail."""

    records_by_step: dict[int, dict[str, Any]] = {}
    if not path.is_file():
        return []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            step = int(record.get("step", record.get("global_step")))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        records_by_step[step] = record
    return [records_by_step[step] for step in sorted(records_by_step)]


def _distillation_metrics(records: list[dict[str, Any]]) -> list[str]:
    metrics = {
        name
        for record in records
        for name, value in record.items()
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and name in _DISTILLATION_METRIC_ORDER
    }
    return [name for name in _DISTILLATION_METRIC_ORDER if name in metrics] + sorted(
        metrics - set(_DISTILLATION_METRIC_ORDER)
    )


def _live_kd_metadata(solution_dir: Path) -> dict[str, Any]:
    recipe = _load_optional(solution_dir / "global_kd_recipe.yaml")
    scheduler = dict(recipe.get("step_scheduler") or {})
    dataset = dict(recipe.get("dataset") or {})
    scheduler_steps = scheduler.get("max_steps")
    return {
        "max_steps": max(0, int(scheduler_steps) - 1) if scheduler_steps is not None else None,
        "global_batch_size": scheduler.get("global_batch_size"),
        "local_batch_size": scheduler.get("local_batch_size"),
        "sample_count": dataset.get("num_samples"),
        "sequence_length": dataset.get("seq_length"),
    }


def _distillation_overfit_data(root: Path) -> dict[str, Any]:
    profiles_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    finalized_solutions: set[tuple[str, str, str]] = set()
    for path in sorted(
        (root / "artifacts" / "global_distillation_sanity" / "profiles").glob(
            "*/*/global_distillation_sanity_summary.json"
        )
    ):
        payload = _load_optional(path)
        profile_id = str(payload.get("profile_id") or path.parents[1].name)
        workload_id = path.parent.name
        solutions = [dict(row) for row in payload.get("solutions", ())]
        records = [record for solution in solutions for record in solution.get("records", ())]
        profiles_by_key[(profile_id, workload_id)] = {
            **payload,
            "profile_id": profile_id,
            "workload_id": workload_id,
            "partial": False,
            "metrics": _distillation_metrics(records),
        }
        finalized_solutions.update(
            (profile_id, workload_id, str(solution.get("solution_id"))) for solution in solutions
        )

    live_root = root / "artifacts" / "global_distillation_sanity" / "profiles"
    for training_log in sorted(live_root.glob("*/*/*/checkpoints/training.jsonl")):
        solution_dir = training_log.parent.parent
        workload_dir = solution_dir.parent
        profile_dir = workload_dir.parent
        identity = (profile_dir.name, workload_dir.name, solution_dir.name)
        if identity in finalized_solutions:
            continue
        records = _training_records(training_log)
        if not records:
            continue
        registry = _profile_registry(root, profile_dir.name)
        style = next(
            (
                dict(row)
                for row in registry.get("solutions", ())
                if str(row.get("solution_id")) == solution_dir.name
            ),
            {},
        )
        key = (profile_dir.name, workload_dir.name)
        profile = profiles_by_key.setdefault(
            key,
            {
                "profile_id": profile_dir.name,
                "workload_id": workload_dir.name,
                "partial": True,
                "solutions": [],
            },
        )
        metadata = _live_kd_metadata(solution_dir)
        for name, value in metadata.items():
            if value is not None:
                profile.setdefault(name, value)
        profile["partial"] = True
        profile["solutions"].append(
            {
                **style,
                "solution_id": solution_dir.name,
                "label": style.get("label", solution_dir.name),
                "color": style.get("color", "#4f8cff"),
                "partial": True,
                "records": records,
            }
        )
        profile["metrics"] = _distillation_metrics(
            [record for solution in profile["solutions"] for record in solution.get("records", ())]
        )
    profiles = sorted(
        profiles_by_key.values(),
        key=lambda row: (int(row.get("max_steps") or 0), str(row.get("workload_id"))),
        reverse=True,
    )
    return {"profiles": profiles}


def _distillation_overfit_section(data: dict[str, Any]) -> str:
    profiles = data.get("profiles") or []
    if not profiles:
        return "<p class='empty'>Pending frozen-minibatch distillation overfit.</p>"
    profile = profiles[0]
    toggles = "".join(
        "<label class='layer-toggle'><input type='checkbox' checked "
        f'data-distillation-overfit-solution="{html.escape(str(row["solution_id"]))}">'
        f'<i style="background:{html.escape(str(row.get("color", "#4f8cff")))}"></i>'
        f"{html.escape(str(row.get('label', row['solution_id'])))}</label>"
        for row in profile.get("solutions", ())
    )
    return (
        f"<p class='note'>The same frozen {html.escape(str(profile.get('sample_count', 128)))}-sample minibatch is replayed at every optimizer "
        "step. Curves show the independent run from each realized MIP checkpoint.</p>"
        f"<div class='replacement-layer-grid aiperf-solution-grid'>{toggles}</div>"
        "<div id='distillation-overfit-plots' class='probe-plots'></div>"
    )


def _proper_distillation_data(root: Path) -> dict[str, Any]:
    runs = []
    finalized_dirs: set[Path] = set()
    summary_paths = list(
        (root / "artifacts" / "global_distillation" / "profiles").rglob(
            "global_distillation_summary.json"
        )
    ) + list((root / "artifacts" / "distillation" / "profiles").rglob("distillation_summary.json"))
    for path in sorted(summary_paths):
        payload = _load_optional(path)
        records = list(payload.get("records") or ())
        finalized_dirs.add(path.parent)
        profile_id = str(payload.get("profile_id", path.parents[2].name))
        solution_id = str(payload.get("solution_id", path.parent.name))
        before = {}
        candidates = []
        for evaluation_path in (
            root / "artifacts" / "zero_shot_evaluation" / "profiles" / profile_id
        ).glob("*/evaluation_summary.json"):
            evaluation = _load_optional(evaluation_path)
            for row in evaluation.get("solutions", ()):
                if str(row.get("solution_id")) == solution_id:
                    candidates.append(
                        (
                            int(evaluation.get("eval_samples", 0)),
                            dict(row.get("metrics") or {}),
                        )
                    )
        if candidates:
            before = max(candidates, key=lambda item: item[0])[1]
        after_payload = _load_optional(path.parent / "evaluation" / "result.json")
        after = dict(after_payload.get("metrics") or {})
        runs.append(
            {
                **payload,
                "run_id": path.parent.parent.name,
                "partial": False,
                "metrics": _distillation_metrics(records),
                "before_kd_metrics": before,
                "after_kd_metrics": after,
            }
        )

    live_root = root / "artifacts" / "global_distillation" / "profiles"
    for training_log in sorted(live_root.glob("*/*/*/checkpoints/training.jsonl")):
        solution_dir = training_log.parent.parent
        if solution_dir in finalized_dirs:
            continue
        records = _training_records(training_log)
        if not records:
            continue
        workload_dir = solution_dir.parent
        profile_dir = workload_dir.parent
        metadata = _live_kd_metadata(solution_dir)
        registry = _profile_registry(root, profile_dir.name)
        style = next(
            (
                dict(row)
                for row in registry.get("solutions", ())
                if str(row.get("solution_id")) == solution_dir.name
            ),
            {},
        )
        runs.append(
            {
                **metadata,
                **style,
                "profile_id": profile_dir.name,
                "run_id": workload_dir.name,
                "solution_id": solution_dir.name,
                "label": style.get("label", solution_dir.name),
                "color": style.get("color", "#4f8cff"),
                "partial": True,
                "records": records,
                "metrics": _distillation_metrics(records),
                "before_kd_metrics": {},
                "after_kd_metrics": {},
            }
        )
    return {"runs": runs}


def _proper_distillation_section(data: dict[str, Any]) -> str:
    runs = data.get("runs") or []
    if not runs:
        return "<p class='empty'>Pending proper fresh-data global distillation.</p>"
    run = runs[-1]
    progress = ""
    if run.get("partial"):
        latest_step = max(
            (
                int(row.get("step", row.get("global_step", 0)))
                for row in run.get("records", ())
            ),
            default=0,
        )
        progress = (
            "<p class='gate pending'>Partial run: "
            f"{latest_step} / {html.escape(str(run.get('max_steps', 'N/A')))} optimizer steps.</p>"
        )
    return (
        progress
        + "<p class='note'>Fresh shuffled data · "
        f"{html.escape(str(run.get('max_steps', 'N/A')))} optimizer steps · sequence length "
        f"{html.escape(str(run.get('sequence_length', 'N/A')))} · global batch "
        f"{html.escape(str(run.get('global_batch_size', 'N/A')))}.</p>"
        '<div id="proper-distillation-plots" class="probe-plots"></div>'
    )


def _post_distillation_evaluation_section(data: dict[str, Any]) -> str:
    """Render pre/post metrics without duplicating Global Distillation plots."""

    runs = data.get("runs") or []
    if not runs:
        return "<p class='empty'>Post-distillation evaluation is pending.</p>"
    run = runs[-1]
    before = run.get("before_kd_metrics") or {}
    after = run.get("after_kd_metrics") or {}
    names = sorted(set(before) | set(after))
    if not names:
        return "<p class='empty'>Final checkpoint evaluation is pending.</p>"
    table = "".join(
        f"<tr><th>{html.escape(name)}</th><td>{html.escape(_fmt(before.get(name)))}</td>"
        f"<td>{html.escape(_fmt(after.get(name)))}</td></tr>"
        for name in names
    )
    return (
        "<div class='table-wrap'><table><thead><tr><th>Metric</th>"
        "<th>Before KD</th><th>After KD</th></tr></thead>"
        f"<tbody>{table}</tbody></table></div>"
    )


def _report_section(
    section_id: str,
    title: str,
    body: str,
    *,
    expanded: bool = False,
) -> str:
    open_attribute = " open" if expanded else ""
    return (
        f'<details class="report-section" id="{html.escape(section_id)}"{open_attribute}>'
        f"<summary><h2>{html.escape(title)}</h2><span class='collapse-indicator'></span></summary>"
        f"<div class='section-body'>{body}</div></details>"
    )


def _nested_config(config: dict[str, Any], *path: str, default: Any = None) -> Any:
    value: Any = config
    for key in path:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
    return default if value is None else value


def _granularity_data(root: Path, merged_config: dict[str, Any]) -> dict[str, Any]:
    manifest = _load_optional(root / "subblock_replacement_manifest.json")
    scoring_mode = str(manifest.get("mode", ""))
    scoring_granularity = (
        "subblock"
        if scoring_mode == "replace_one_subblock"
        else str(_nested_config(merged_config, "scoring", "granularity", default="block"))
    )
    result_dir = root / (
        "single_subblock_replacement_solutions--validation"
        if scoring_granularity == "subblock"
        else "single_sequence_replacement_solutions--validation"
    )
    expected_scores = int(
        manifest.get(
            "subblock_solution_count"
            if scoring_granularity == "subblock"
            else "canonical_entry_count",
            0,
        )
        or 0
    )
    present_scores = sum(1 for _ in result_dir.glob("solution_*.json"))

    shard_dirs = sorted((root / "runtime_cache" / "shards").glob("*"))
    done_shards = sum(1 for directory in shard_dirs for _ in directory.glob("shard_*.done"))
    shard_indices = []
    for directory in shard_dirs:
        for path in directory.glob("shard_*.*"):
            try:
                shard_indices.append(int(path.stem.rsplit("_", 1)[-1]))
            except ValueError:
                pass
    expected_shards = max(shard_indices, default=-1) + 1
    bypass_enabled = bool(_nested_config(merged_config, "bypass", "enabled", default=False))
    return {
        "depth": str(_nested_config(merged_config, "depth", "granularity", default="subblock")),
        "runtime": str(
            _nested_config(
                merged_config,
                "calc_subblock_stats",
                "runtime_stats",
                "granularity",
                default="block",
            )
        ),
        "scoring": scoring_granularity,
        "bypass": str(_nested_config(merged_config, "bypass", "granularity", default="block")),
        "bypass_enabled": bypass_enabled,
        "canonical_candidates": int(manifest.get("canonical_entry_count", 0) or 0),
        "full_search_space_preserved": bool(manifest.get("full_search_space_preserved", False)),
        "score_expected": expected_scores,
        "score_present": present_scores,
        "score_complete": expected_scores > 0 and present_scores == expected_scores,
        "runtime_shards_expected": expected_shards,
        "runtime_shards_present": done_shards,
        "runtime_complete": expected_shards > 0 and done_shards == expected_shards,
        "composition": (
            "teacher baseline + additive subblock deltas; exact block measurements take precedence"
            if scoring_granularity == "subblock"
            else "exact block measurements"
        ),
    }


def _granularity_section(data: dict[str, Any]) -> str:
    rows = (
        ("Depth", data["depth"]),
        ("vLLM runtime", data["runtime"]),
        ("Replacement scoring", data["scoring"]),
        (
            "Bypass",
            data["bypass"] + ("" if data.get("bypass_enabled") else " (disabled)"),
        ),
    )
    table_rows = "".join(
        f"<tr><th>{html.escape(label)}</th><td><span class='granularity-badge'>"
        f"{html.escape(value)}</span></td></tr>"
        for label, value in rows
    )
    canonical = int(data.get("canonical_candidates", 0))
    score_present = int(data.get("score_present", 0))
    score_expected = int(data.get("score_expected", 0))
    runtime_present = int(data.get("runtime_shards_present", 0))
    runtime_expected = int(data.get("runtime_shards_expected", 0))
    return (
        "<p class='note'>Atomic measurements are inputs to the canonical full-block search; "
        "they do not reduce its candidate space.</p>"
        "<div class='table-wrap'><table><thead><tr><th>Stage</th><th>Granularity</th>"
        f"</tr></thead><tbody>{table_rows}</tbody></table></div>"
        "<div class='vllm-overview-cards'>"
        f"<article><strong>{canonical:,}</strong><span>Canonical block candidates</span></article>"
        f"<article><strong>{score_present:,} / {score_expected:,}</strong><span>Atomic score coverage</span></article>"
        f"<article><strong>{runtime_present:,} / {runtime_expected:,}</strong><span>Runtime shard coverage</span></article>"
        "</div>"
        f"<p class='note'><strong>Score composition:</strong> {html.escape(str(data['composition']))}.</p>"
    )


def _experiment_summary_data(
    merged_config: dict[str, Any], granularity: dict[str, Any]
) -> dict[str, Any]:
    input_tokens = _nested_config(merged_config, "aiperf", "input_tokens")
    output_tokens = _nested_config(merged_config, "aiperf", "output_tokens")
    return {
        "data": " · ".join(
            str(value)
            for value in (
                _nested_config(merged_config, "data", "modality"),
                _nested_config(merged_config, "data", "layout"),
                (
                    f"max {_fmt(_nested_config(merged_config, 'data', 'max_sample_length'))} tokens"
                    if _nested_config(merged_config, "data", "max_sample_length") is not None
                    else None
                ),
            )
            if value is not None and value != ""
        ),
        "granularities": (
            f"depth={granularity['depth']} · runtime={granularity['runtime']} · "
            f"scoring={granularity['scoring']}"
        ),
        "bypass": (
            f"enabled · {granularity['bypass']}"
            if granularity.get("bypass_enabled")
            else f"skipped · {granularity['bypass']}"
        ),
        "workload": (
            f"ISL {_fmt(input_tokens)} · OSL {_fmt(output_tokens)}"
            if input_tokens is not None and output_tokens is not None
            else "not configured"
        ),
        "evaluation_samples": _nested_config(
            merged_config,
            "evaluation",
            "eval_samples",
            default=_nested_config(merged_config, "scoring", "eval_samples", default="unknown"),
        ),
        "width_diagnostic_samples": _nested_config(
            merged_config, "activation_diagnostic", "eval_samples", default="unknown"
        ),
        "aiperf_candidates": _nested_config(
            merged_config, "aiperf", "num_best_to_eval", default="all selected"
        ),
        "distillation_candidates": _nested_config(
            merged_config, "distillation", "num_best_to_distill", default="all selected"
        ),
    }


def _experiment_summary_section(data: dict[str, Any]) -> str:
    labels = (
        ("Dataset", "data"),
        ("Search granularities", "granularities"),
        ("Bypass", "bypass"),
        ("AIPerf workload", "workload"),
        ("Exact-evaluation samples", "evaluation_samples"),
        ("Width-diagnostic samples", "width_diagnostic_samples"),
        ("Best models for AIPerf", "aiperf_candidates"),
        ("Best models for distillation", "distillation_candidates"),
    )
    return "<div class='summary-grid'>" + "".join(
        "<article><span>{}</span><strong>{}</strong></article>".format(
            html.escape(label), html.escape(_fmt(data.get(key) or "unknown"))
        )
        for label, key in labels
    ) + "</div>"


def _stage_dag(
    root: Path,
    merged_config: dict[str, Any],
    statuses: dict[str, str],
    stage_targets: dict[str, str],
) -> str:
    """Render the configured scheduler-neutral stage graph as a navigable SVG."""

    specs = {spec.stage_id: spec for spec in STAGE_SPECS}
    parents = {
        stage_id: tuple(
            parent
            for parent in selected_parent_stage_ids(stage_id, merged_config)
            if parent in specs
        )
        for stage_id in specs
    }
    levels: dict[str, int] = {}
    pending = set(specs)
    while pending:
        ready = [stage_id for stage_id in pending if all(parent in levels for parent in parents[stage_id])]
        if not ready:
            ready = list(pending)
        for stage_id in sorted(ready, key=lambda value: specs[value].topology_order):
            levels[stage_id] = max((levels[parent] + 1 for parent in parents[stage_id]), default=0)
            pending.remove(stage_id)

    by_level: dict[int, list[str]] = {}
    for stage_id in specs:
        by_level.setdefault(levels[stage_id], []).append(stage_id)
    max_rows = max((len(nodes) for nodes in by_level.values()), default=1)
    node_width, node_height = 206, 70
    x_stride, y_stride = 248, 94
    width = (max(by_level, default=0) + 1) * x_stride + 34
    height = max(210, max_rows * y_stride + 24)
    positions: dict[str, tuple[float, float]] = {}
    for level, nodes in by_level.items():
        column_height = (len(nodes) - 1) * y_stride + node_height
        start_y = (height - column_height) / 2
        for row, stage_id in enumerate(nodes):
            positions[stage_id] = (18 + level * x_stride, start_y + row * y_stride)

    edges = []
    for child, child_parents in parents.items():
        child_x, child_y = positions[child]
        for parent in child_parents:
            parent_x, parent_y = positions[parent]
            start_x, start_y = parent_x + node_width, parent_y + node_height / 2
            end_x, end_y = child_x, child_y + node_height / 2
            bend = (start_x + end_x) / 2
            muted = statuses.get(child) == "disabled" or statuses.get(parent) == "disabled"
            edges.append(
                f"<path class='dag-edge{' muted' if muted else ''}' "
                f'data-source="{html.escape(parent)}" data-target="{html.escape(child)}" '
                f"d='M {start_x:.1f} {start_y:.1f} C {bend:.1f} {start_y:.1f}, "
                f"{bend:.1f} {end_y:.1f}, {end_x:.1f} {end_y:.1f}'/>"
            )

    nodes = []
    for stage_id, spec in specs.items():
        x, y = positions[stage_id]
        status = statuses.get(stage_id, "")
        target = stage_targets.get(stage_id)
        granularity = _stage_granularity(root, stage_id, merged_config)
        label = stage_display_name(stage_id, granularity=granularity)
        label_markup = "".join(
            f"<tspan x='26' dy='{'0' if index == 0 else '15'}'>{html.escape(line)}</tspan>"
            for index, line in enumerate(_dag_label_lines(label))
        )
        node_type = "required" if spec.required else "optional"
        status_label = status.title() if status else ""
        content = (
            f'<g class="dag-node {node_type} {html.escape(status)}" '
            f'data-stage="{html.escape(stage_id)}" '
            f'data-status="{html.escape(status)}" transform="translate({x:.1f} {y:.1f})">'
            f"<rect width='{node_width}' height='{node_height}' rx='10'/><circle cx='14' cy='16' r='5'/>"
            f"<text class='dag-label' x='26' y='19'>{label_markup}</text>"
            f"<text class='dag-status' x='14' y='58'>{html.escape(status_label)}</text></g>"
        )
        nodes.append(f"<a href='#{html.escape(target)}'>{content}</a>" if target else content)

    return (
        "<div class='dag-scroll'><svg class='stage-dag' role='img' "
        "aria-label='Puzzletron experiment stage graph' "
        f"viewBox='0 0 {width} {height}' width='{width}' height='{height}'>"
        "<defs><marker id='dag-arrow' markerWidth='7' markerHeight='7' refX='6' refY='3.5' "
        "orient='auto'><path d='M0,0 L7,3.5 L0,7 Z'/></marker></defs>"
        f"{''.join(edges)}{''.join(nodes)}</svg></div>"
        "<div class='dag-legend'><span class='completed'>Completed</span>"
        "<span class='pending'>Pending</span>"
        "<span class='disabled'>Disabled</span></div>"
        "<div class='dag-legend dag-type-legend'><span class='required-node'>Required</span>"
        "<span class='optional-node'>Optional</span></div>"
    )


def generate_campaign_progress_report(
    puzzle_dir: str | Path,
    *,
    model_name: str = "Puzzletron model",
    running_stage: str | None = None,
) -> dict[str, str]:
    """Rewrite the stable campaign HTML using all artifacts available so far."""

    root = Path(puzzle_dir).resolve()
    output = root / "artifacts" / "campaign_report"
    output.mkdir(parents=True, exist_ok=True)
    html_path = output / "campaign_report.html"
    merged_config = _latest_merged_config(root)
    granularity_data = _granularity_data(root, merged_config)
    statuses = {
        spec.stage_id: _pipeline_state(root, spec, merged_config) for spec in STAGE_SPECS
    }
    present = {spec.stage_id: _stage_artifact_present(root, spec) for spec in STAGE_SPECS}
    sort_summary = _sort_summary(root)
    activation_summary = _activation_diagnostic_summary(root)
    bypass_overfit = _bypass_overfit_data(root)
    nested_bypass = _nested_bypass_data(root)
    depth_data = _depth_data(root)
    library_data = _library_data(root)
    vllm_data = _vllm_data(root, library_data)
    replacement_data = _replacement_data(root)
    mip_data = _mip_data(root)
    evaluation_data = _evaluation_data(root)
    aiperf_data = _aiperf_data(root)
    distillation_overfit_data = _distillation_overfit_data(root)
    proper_distillation_data = _proper_distillation_data(root)
    has_sort_diagnosis = present["sort_sanity"]
    has_activation_diagnosis = present["width_sanity"] or present["slicing_sanity"]
    has_bypass_overfit = present["bypass_sanity"]
    has_nested_bypass = present["bypass"]
    has_depth = present["depth_importance"]
    has_library = present["build_library"]
    has_vllm = present["vllm_stats"]
    has_replacement = present["replacement_scoring"]
    has_mip = present["mip"]
    has_evaluation = present["zero_shot_evaluation"]
    has_aiperf = present["aiperf"]
    has_distillation_overfit = present["global_distillation_sanity"]
    has_proper_distillation = bool(proper_distillation_data.get("runs"))
    has_post_distillation_evaluation = present["post_distillation_evaluation"]
    stage_targets: dict[str, str] = {}
    if has_sort_diagnosis:
        stage_targets["sort_sanity"] = "sort-sanity"
    if has_activation_diagnosis:
        if present["width_sanity"]:
            stage_targets["width_sanity"] = "width-sanity"
        if present["slicing_sanity"]:
            stage_targets["slicing_sanity"] = "width-sanity"
    if has_bypass_overfit:
        stage_targets["bypass_sanity"] = "bypass-sanity"
    if has_nested_bypass:
        stage_targets["bypass"] = "bypass"
    if has_depth:
        stage_targets["depth_importance"] = "depth-importance"
    if has_library:
        stage_targets["build_library"] = "block-library"
    if has_vllm:
        stage_targets["vllm_stats"] = "vllm-statistics"
    if has_replacement:
        stage_targets["replacement_scoring"] = "replacement-scoring"
    if has_mip:
        stage_targets["mip"] = "mip-solutions"
    if has_evaluation:
        stage_targets["zero_shot_evaluation"] = "zero-shot-evaluation"
    if has_aiperf:
        stage_targets["aiperf"] = "aiperf-benchmarks"
    if has_distillation_overfit:
        stage_targets["global_distillation_sanity"] = "global-distillation-sanity"
    if has_proper_distillation:
        stage_targets["global_distillation"] = "global-distillation"
    if has_post_distillation_evaluation:
        stage_targets["post_distillation_evaluation"] = "post-distillation-evaluation"
    dag = _stage_dag(root, merged_config, statuses, stage_targets)
    vllm_title = stage_display_name(
        "vllm_stats", granularity=_stage_granularity(root, "vllm_stats", merged_config)
    )
    bypass_title = stage_display_name(
        "bypass", granularity=_stage_granularity(root, "bypass", merged_config)
    )
    replacement_title = stage_display_name(
        "replacement_scoring",
        granularity=_stage_granularity(root, "replacement_scoring", merged_config),
    )
    result_sections = "".join(
        section
        for enabled, section in (
            (
                has_sort_diagnosis,
                _report_section(
                    "sort-sanity", "Sort Sanity Check", _sort_table(sort_summary), expanded=True
                ),
            ),
            (
                has_activation_diagnosis,
                _report_section(
                    "width-sanity",
                    "Width and Slicing Sanity Checks",
                    _activation_diagnostic_section(activation_summary),
                    expanded=True,
                ),
            ),
            (
                has_bypass_overfit,
                _report_section(
                    "bypass-sanity",
                    "Bypass Sanity Check",
                    _bypass_overfit_section(bypass_overfit),
                    expanded=True,
                ),
            ),
            (
                has_nested_bypass,
                _report_section(
                    "bypass", bypass_title, _nested_bypass_section(nested_bypass), expanded=True
                ),
            ),
            (
                has_depth,
                _report_section(
                    "depth-importance",
                    "Depth Importance Estimation",
                    _depth_section(depth_data),
                    expanded=True,
                ),
            ),
            (
                has_library,
                _report_section(
                    "block-library",
                    "Build Block Library",
                    _library_section(library_data),
                    expanded=True,
                ),
            ),
            (
                has_vllm,
                _report_section(
                    "vllm-statistics", vllm_title, _vllm_section(vllm_data), expanded=True
                ),
            ),
            (
                has_replacement,
                _report_section(
                    "replacement-scoring",
                    replacement_title,
                    _replacement_section(replacement_data),
                    expanded=True,
                ),
            ),
            (
                has_mip,
                _report_section(
                    "mip-solutions", "MIP Search", _mip_section(mip_data), expanded=True
                ),
            ),
            (
                has_evaluation,
                _report_section(
                    "zero-shot-evaluation",
                    "Zero-shot Evaluation",
                    _evaluation_section(evaluation_data),
                    expanded=True,
                ),
            ),
            (
                has_aiperf,
                _report_section(
                    "aiperf-benchmarks", "AIPerf", _aiperf_section(aiperf_data), expanded=True
                ),
            ),
            (
                has_distillation_overfit,
                _report_section(
                    "global-distillation-sanity",
                    "Global Distillation Sanity Check",
                    _distillation_overfit_section(distillation_overfit_data),
                    expanded=True,
                ),
            ),
            (
                has_proper_distillation,
                _report_section(
                    "global-distillation",
                    "Global Distillation",
                    _proper_distillation_section(proper_distillation_data),
                    expanded=True,
                ),
            ),
            (
                has_post_distillation_evaluation,
                _report_section(
                    "post-distillation-evaluation",
                    "Post Distillation Evaluation",
                    _post_distillation_evaluation_section(proper_distillation_data),
                    expanded=True,
                ),
            ),
        )
        if enabled
    )
    embedded = json.dumps(
        {
            "model": model_name,
            "root": str(root),
            "stage_status": statuses,
            "merged_config": merged_config,
            "granularity": granularity_data,
            "sort_equivalence": sort_summary,
            "activation_diagnostic": activation_summary,
            "activation_diagnostic_view": {
                "rows": _activation_diagnostic_rows(activation_summary),
                "metric_order": list(_ACTIVATION_METRIC_ORDER),
                "metric_descriptions": _METRIC_DESCRIPTIONS,
                "findings": list(activation_summary.get("width_findings") or ())
                + list(activation_summary.get("slicing_findings") or ()),
            },
            "bypass_overfit": bypass_overfit,
            "nested_bypass": nested_bypass,
            "depth": depth_data,
            "library": library_data,
            "vllm": vllm_data,
            "replacement": replacement_data,
            "mip": mip_data,
            "evaluation": evaluation_data,
            "aiperf": aiperf_data,
            "distillation_overfit": distillation_overfit_data,
            "proper_distillation": proper_distillation_data,
        }
    ).replace("</", "<\\/")
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(model_name)} · Puzzletron campaign</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@dagrejs/dagre@1.1.5/dist/dagre.min.js"></script>
<style>
:root{{--ink:#e7edf7;--muted:#93a4bd;--panel:#101826;--line:#26344a;--blue:#4f8cff;--green:#35d07f;--amber:#ffbd45;--red:#ff6577}}
*{{box-sizing:border-box}} body{{margin:0;background:#08101c;color:var(--ink);font:14px/1.5 Inter,ui-sans-serif,system-ui,sans-serif}}
main{{max-width:1280px;margin:auto;padding:32px}} h1{{font-size:30px;margin:0 0 4px}} h2{{margin:0;font-size:20px}} .subtitle{{color:var(--muted);margin:0 0 28px}}
.report-section{{background:linear-gradient(145deg,#111c2c,#0d1624);border:1px solid var(--line);border-radius:16px;margin:18px 0;box-shadow:0 14px 40px #0004;scroll-margin-top:20px}}
.report-section>summary{{display:flex;align-items:center;justify-content:space-between;gap:16px;padding:22px;cursor:pointer;list-style:none;user-select:none}} .report-section>summary::-webkit-details-marker{{display:none}} .report-section>summary::marker{{display:none}} .section-body{{padding:0 22px 22px}} .collapse-indicator::before{{content:'+';display:grid;place-items:center;width:28px;height:28px;border:1px solid var(--line);border-radius:50%;color:var(--muted);font-size:20px;line-height:1}} .report-section[open] .collapse-indicator::before{{content:'−'}}
.hoverlayer .hovertext path,.hoverlayer .hovertext rect,.hoverlayer .axistext path{{fill-opacity:.32!important}} .cardinality-formula{{padding:14px 16px;border:1px solid #4f8cff66;border-radius:10px;background:#0a1421;color:#cfe0ff;font:600 15px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}}
pre{{max-height:480px;overflow:auto;background:#07101b;border:1px solid #1d2a3d;border-radius:12px;padding:16px;color:#cfe0ff;white-space:pre-wrap}}
.summary-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:10px}} .summary-grid article{{display:flex;min-height:86px;flex-direction:column;justify-content:center;gap:7px;padding:14px 16px;border:1px solid var(--line);border-radius:12px;background:#0a1421}} .summary-grid span{{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.06em}} .summary-grid strong{{color:#cfe0ff;font-size:16px}}
.dag-scroll{{overflow-x:auto;padding:6px 0 12px}} .stage-dag{{display:block;min-width:100%;font-family:Inter,ui-sans-serif,system-ui,sans-serif}} .dag-edge{{fill:none;stroke:#4f8cff88;stroke-width:2;marker-end:url(#dag-arrow)}} .dag-edge.muted{{stroke:#55647a55;stroke-dasharray:5 6}} .dag-node rect{{fill:#0b1421;stroke:#34445e;stroke-width:1.5;transition:fill .15s,stroke .15s}} .dag-node.optional rect{{stroke-dasharray:6 4}} .dag-node circle{{fill:#55647a}} .dag-node .dag-label{{fill:var(--ink);font-size:11px;font-weight:650}} .dag-node .dag-status{{fill:var(--muted);font-size:9px;letter-spacing:.08em}} .dag-node.completed circle{{fill:var(--green)}} .dag-node.completed rect{{stroke:#35d07f99}} .dag-node.pending circle{{fill:var(--amber)}} .dag-node.pending rect{{stroke:#ffbd4577}} .dag-node.disabled{{opacity:.38}} .stage-dag a:hover .dag-node rect{{fill:#101d30;stroke:var(--blue)}} .dag-legend{{display:flex;gap:16px;flex-wrap:wrap;color:var(--muted);font-size:12px;margin-top:5px}} .dag-legend span::before{{content:'';display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;background:#55647a}} .dag-legend .completed::before{{background:var(--green)}} .dag-legend .pending::before{{background:var(--amber)}} .dag-legend .disabled::before{{background:#55647a;opacity:.45}} .dag-type-legend span::before{{width:13px;height:9px;border-radius:2px;background:transparent;border:1.5px solid #93a4bd}} .dag-type-legend .optional-node::before{{border-style:dashed}}
@keyframes pulse{{50%{{transform:scale(1.7);box-shadow:0 0 18px #ffbd45aa}}}} .table-wrap{{overflow:auto}} table{{width:100%;border-collapse:collapse}} th,td{{padding:10px 12px;border-bottom:1px solid var(--line);text-align:right}} th:first-child{{text-align:left}} thead th{{color:#a9bee0;background:#0a1421;position:sticky;top:0}} td.warning-cell{{background:#ffbd4524;color:#ffe3a3;box-shadow:inset 0 0 0 1px #ffbd4570}} .warning-value{{cursor:help;outline-offset:3px}} #warning-tooltip{{position:fixed;z-index:10000;max-width:min(420px,calc(100vw - 24px));padding:9px 11px;border:1px solid #ffbd4588;border-radius:8px;background:#241b0d;color:#ffe3a3;font-size:12px;line-height:1.4;box-shadow:0 10px 30px #0009;pointer-events:none;white-space:pre-wrap}} .gate{{display:inline-block;border-radius:999px;padding:5px 10px;background:#1b2839}} .gate.passed{{color:var(--green)}} .empty,.note{{color:var(--muted)}} select{{margin:0 18px 18px 8px;padding:8px 12px;border:1px solid var(--line);border-radius:8px;background:#0a1421;color:var(--ink)}} .selector-label{{color:var(--muted)}} code{{color:#bcd2ff}} .probe-summaries{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px;margin:16px 0}} .probe-summary{{border:1px solid var(--line);border-radius:12px;padding:14px;background:#0a1421}} .probe-summary.passed{{border-color:#35d07f66}} .probe-summary.failed{{border-color:#ff657766}} .probe-summary h3{{margin:0 0 10px}} .probe-summary dl{{display:grid;grid-template-columns:1fr auto;gap:5px 14px;margin:0}} .probe-summary dt{{color:var(--muted)}} .probe-summary dd{{margin:0;text-align:right}} .probe-plots{{display:grid;grid-template-columns:repeat(auto-fit,minmax(430px,1fr));gap:14px}} .probe-plot-panel{{min-width:0;border:1px solid var(--line);border-radius:12px;padding:12px;background:#091321}} .probe-plot-panel h3{{margin:0 0 4px}} .plotly-chart{{width:100%;height:390px}} .depth-chart{{height:460px}}
.vllm-overview-cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin:16px 0}} .vllm-overview-cards article{{display:flex;flex-direction:column;padding:14px;border:1px solid var(--line);border-radius:10px;background:#0a1421}} .vllm-overview-cards strong{{font-size:24px;color:#cfe0ff}} .vllm-overview-cards span{{color:var(--muted)}} .vllm-panel{{margin:14px 0}} .vllm-controls{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:10px;margin:14px 0}} .vllm-controls label{{display:flex;flex-direction:column;gap:5px;color:var(--muted)}} .vllm-controls select{{margin:0}}
.replacement-layer-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(105px,1fr));gap:6px;margin:12px 0;padding:10px;border:1px solid var(--line);border-radius:10px;background:#091321}} .layer-toggle,.replacement-connect-toggle,.replacement-all-toggle{{display:flex;align-items:center;gap:6px;color:var(--muted);font-size:13px}} .layer-toggle input,.replacement-connect-toggle input,.replacement-all-toggle input{{accent-color:#4f8cff}} .replacement-connect-toggle{{margin:8px 0}} .replacement-layer-toolbar{{display:flex;align-items:center;justify-content:space-between;gap:18px;flex-wrap:wrap;margin:10px 0}} .replacement-all-toggle{{font-weight:600;color:var(--ink)}} .replacement-layer-color-key{{display:flex;align-items:center;gap:8px;color:var(--muted);font-size:12px}} .replacement-layer-color-key i{{display:block;width:min(280px,32vw);height:8px;border-radius:999px;background:linear-gradient(90deg,#ff6577,#4f8cff)}}
.mip-table th:first-child,.mip-table td:first-child{{position:sticky;left:0;z-index:2;background:#0a1421;text-align:left;white-space:nowrap}} .mip-table td{{font-variant-numeric:tabular-nums;white-space:nowrap}} .mip-table .status-feasible,.mip-table .status-reference{{color:var(--green)}} .mip-table .status-infeasible{{color:var(--red)}} .mip-sort-button{{display:inline-flex;align-items:center;gap:5px;padding:0;border:0;background:transparent;color:inherit;font:inherit;font-weight:600;cursor:pointer;white-space:nowrap}} .mip-sort-button:hover{{color:#fff}} .mip-sort-arrow{{display:inline-block;min-width:10px;color:var(--blue)}}
.aiperf-solution-grid .layer-toggle i{{display:inline-block;width:10px;height:10px;border-radius:50%}} .replacement-layer-toolbar button{{border:1px solid var(--line);border-radius:7px;background:#0a1421;color:var(--ink);padding:6px 12px;cursor:pointer}} .replacement-layer-toolbar button:hover{{border-color:var(--blue)}}
.finding-list{{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:8px;margin:12px 0 18px}} .finding{{padding:11px 13px;border:1px solid #ffbd4566;border-radius:10px;background:#221a0d}} .finding>div{{display:flex;gap:10px;align-items:baseline}} .finding strong{{color:var(--amber);white-space:nowrap}} .finding span{{color:#f5dfb5;font-size:12px}} .finding details{{margin-top:7px}} .finding details summary{{cursor:pointer;color:#f5dfb5;font-size:12px}} .finding ul{{margin:8px 0 0;padding-left:18px;color:#d8c49c;font-size:12px}}
</style></head><body><main>
<h1>{html.escape(model_name)}</h1><p class="subtitle">Incremental Puzzletron campaign report</p>
{_report_section("pipeline", "Pipeline", dag, expanded=True)}
{result_sections}
<div id="warning-tooltip" role="tooltip" hidden></div>
<script id="campaign-data" type="application/json">{embedded}</script>
<script>(()=>{{function openTarget(){{if(!location.hash)return;const target=document.querySelector(location.hash);if(!(target instanceof HTMLDetailsElement))return;target.open=true;requestAnimationFrame(()=>target.scrollIntoView({{behavior:'smooth',block:'start'}}));}}document.querySelectorAll('.stage-dag a[href^="#"]').forEach(link=>link.addEventListener('click',()=>requestAnimationFrame(openTarget)));window.addEventListener('hashchange',openTarget);openTarget();}})();</script>
<script>(()=>{{
  const svg=document.querySelector('.stage-dag');
  if(!svg||!window.dagre)return;
  try{{
    const graph=new dagre.graphlib.Graph().setGraph({{rankdir:'LR',ranksep:70,nodesep:28,edgesep:14,marginx:18,marginy:18}}).setDefaultEdgeLabel(()=>({{}}));
    svg.querySelectorAll('.dag-node').forEach(node=>graph.setNode(node.dataset.stage,{{width:206,height:70}}));
    svg.querySelectorAll('.dag-edge').forEach(edge=>graph.setEdge(edge.dataset.source,edge.dataset.target));
    dagre.layout(graph);
    svg.querySelectorAll('.dag-node').forEach(node=>{{const point=graph.node(node.dataset.stage);node.setAttribute('transform',`translate(${{point.x-103}} ${{point.y-35}})`);}});
    svg.querySelectorAll('.dag-edge').forEach(edge=>{{const route=graph.edge(edge.dataset.source,edge.dataset.target);if(!route?.points?.length)return;edge.setAttribute('d',route.points.map((point,index)=>`${{index?'L':'M'}} ${{point.x.toFixed(1)}} ${{point.y.toFixed(1)}}`).join(' '));}});
    const layout=graph.graph();
    svg.setAttribute('viewBox',`0 0 ${{Math.ceil(layout.width)}} ${{Math.ceil(layout.height)}}`);
    svg.setAttribute('width',String(Math.ceil(layout.width)));
    svg.setAttribute('height',String(Math.ceil(layout.height)));
  }}catch(error){{console.warn('Dagre layout failed; retaining static Puzzletron DAG.',error);}}
}})();</script>
<script>(()=>{{
  const tooltip=document.getElementById('warning-tooltip');
  if(!tooltip)return;
  let active=null;
  function hide(){{active=null;tooltip.hidden=true;}}
  function show(target){{
    active=target;
    tooltip.textContent=target.dataset.warning||'';
    tooltip.hidden=false;
    const rect=target.getBoundingClientRect(),tip=tooltip.getBoundingClientRect();
    const left=Math.min(window.innerWidth-tip.width-8,Math.max(8,rect.right-tip.width));
    const above=rect.top-tip.height-8;
    tooltip.style.left=`${{left}}px`;
    tooltip.style.top=`${{above>=8?above:Math.min(window.innerHeight-tip.height-8,rect.bottom+8)}}px`;
  }}
  document.addEventListener('pointerover',event=>{{const target=event.target.closest?.('.warning-value');if(target&&target!==active)show(target);}});
  document.addEventListener('pointerout',event=>{{if(active&&!event.relatedTarget?.closest?.('.warning-value'))hide();}});
  document.addEventListener('focusin',event=>{{const target=event.target.closest?.('.warning-value');if(target)show(target);}});
  document.addEventListener('focusout',event=>{{if(active)hide();}});
  window.addEventListener('scroll',hide,true);
}})();</script>
<script>(()=>{{
  const axis=document.getElementById('activation-axis-select');
  const metric=document.getElementById('activation-metric-select');
  const body=document.getElementById('activation-diagnostic-body');
  const help=document.getElementById('activation-metric-help');
  if(!axis||!metric||!body)return;
  const data=JSON.parse(document.getElementById('campaign-data').textContent).activation_diagnostic_view;
  const esc=v=>String(v).replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
  const fmt=v=>v==null?'N/A':typeof v==='number'?v.toPrecision(9).replace(/0+$/,'').replace(/[.]$/,''):String(v);
  const methodKeys=['preferred_method','comparison_method','left_method','right_method'];
  function warningsFor(layer,target,method){{
    return (data.findings||[]).filter(finding=>{{
      const evidence=finding.evidence||{{}},group=evidence.group||{{}};
      const methods=methodKeys.map(key=>evidence[key]).filter(value=>value!=null).map(String);
      return String(group.axis)===axis.value&&String(group.layer_idx)===String(layer)&&String(group.target_value)===String(target)&&String(evidence.metric)===metric.value&&methods.includes(method);
    }}).map(finding=>finding.message||'Measured result needs review.');
  }}
  function cell(methods,layer,target,method){{
    const warnings=warningsFor(layer,target,method);
    const value=esc(fmt((methods[method]||{{}})[metric.value]));
    if(!warnings.length)return `<td>${{value}}</td>`;
    return `<td class="warning-cell"><span class="warning-value" tabindex="0" data-warning="${{esc(warnings.join(' · '))}}">${{value}}</span></td>`;
  }}
  function render(){{
    const cases=new Map();
    data.rows.filter(r=>String(r.axis)===axis.value).forEach(r=>{{
      const key=JSON.stringify([r.layer_idx,r.target_value,r.ratio]);
      if(!cases.has(key))cases.set(key,{{}});
      cases.get(key)[String(r.method)]=r;
    }});
    const rows=[...cases.entries()].sort((a,b)=>String(JSON.parse(a[0])[0]).localeCompare(String(JSON.parse(b[0])[0]),undefined,{{numeric:true}})).map(([key,methods])=>{{
      const [layer,target,ratio]=JSON.parse(key);
      const label=layer==='global'?(ratio==null?'global':`global@${{(ratio*100).toFixed(0)}}%`):(ratio==null?`layer_${{layer}}`:`layer_${{layer}}@${{(ratio*100).toFixed(0)}}%`);
      return `<tr><th>${{esc(label)}}</th><td>${{esc(fmt(target))}}</td>${{['sorted','original','reverse','physical'].map(method=>cell(methods,layer,target,method)).join('')}}</tr>`;
    }});
    body.innerHTML=rows.join('');
    if(help)help.textContent=(data.metric_descriptions[metric.value]||'')+' '+(metric.value.startsWith('token_accuracy_')?'Higher is better.':'Lower is better.');
  }}
  axis.addEventListener('change',render);
  metric.addEventListener('change',render);
  render();
}})();</script>
<script>(()=>{{const root=document.getElementById('campaign-data');if(!root)return;const report=JSON.parse(root.textContent),theme={{paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'#091321',font:{{color:'#cfe0ff'}},xaxis:{{title:'Iteration',gridcolor:'#26344a',zerolinecolor:'#26344a'}},yaxis:{{title:'Loss',gridcolor:'#26344a',zerolinecolor:'#26344a'}},legend:{{orientation:'h',y:1.12}},margin:{{l:62,r:18,t:42,b:55}},hovermode:'x unified',hoverlabel:{{bgcolor:'rgba(8,16,28,.74)',bordercolor:'rgba(147,164,189,.35)',font:{{size:11,color:'#e7edf7'}},namelength:18}}}},config={{responsive:true,displaylogo:false,modeBarButtonsToRemove:['lasso2d','select2d']}};function stableConfigKey(value){{const normalize=item=>{{if(Array.isArray(item))return item.map(normalize);if(item&&typeof item==='object')return Object.keys(item).sort().reduce((result,key)=>{{result[key]=normalize(item[key]);return result;}},{{}});return item;}};return JSON.stringify(normalize(value));}}function installConfigFocus(element,keys){{if(!element||typeof element.on!=='function'||!keys.length)return;if(typeof element.removeAllListeners==='function'){{element.removeAllListeners('plotly_hover');element.removeAllListeners('plotly_unhover');}}const restore=()=>Plotly.restyle(element,{{'marker.opacity':[keys.map(()=>.82)]}},[0]);element.on('plotly_hover',event=>{{const index=event.points&&event.points[0]&&event.points[0].pointIndex;if(index==null)return;const active=keys[index];Plotly.restyle(element,{{'marker.opacity':[keys.map(key=>key===active?.95:.1)]}},[0]);}});element.on('plotly_unhover',restore);restore();}}function layerSelection(record,unit){{const layer=String(unit).split(':',1)[0];return ((record.elastic_selection||{{}}).layers||[]).find(item=>String(item.layer_idx)===layer)||{{}};}}function renderLossChart(elementId,mode,unit){{const element=document.getElementById(elementId),modeData=(((report.bypass_overfit||{{}}).modes||{{}})[mode]||{{}}),records=modeData.records||[],metricKey=modeData.metric_key||'per_layer_loss';if(!element||!records.length)return;const x=records.map(r=>Number(r.step)),mean=records.map(r=>Number(r.loss)),selected=records.map(r=>Number((r[metricKey]||{{}})[unit])),meanMeta=records.map(r=>[r.hidden_width==null?'N/A':r.hidden_width]),unitMeta=records.map(r=>{{const choice=layerSelection(r,unit);return [r.hidden_width==null?'N/A':r.hidden_width,choice.candidate_id||'N/A',JSON.stringify(choice.changed_axes||{{}})];}}),unitLabel=modeData.granularity==='subblock'?`Subblock ${{unit}}`:`Layer ${{unit}}`,traces=[{{x,y:mean,name:'Mean',mode:'lines+markers',line:{{color:'#ffbd45',width:2}},marker:{{size:4}},customdata:meanMeta,hovertemplate:'#%{{x}} · mean %{{y:.5g}}<br>w=%{{customdata[0]}}<extra></extra>'}},{{x,y:selected,name:unitLabel,mode:'lines+markers',line:{{color:'#4f8cff',width:2}},marker:{{size:4}},customdata:unitMeta,hovertemplate:'#%{{x}} · loss %{{y:.5g}}<br>w=%{{customdata[0]}} · %{{customdata[1]}}<br>%{{customdata[2]}}<extra></extra>'}}];Plotly.react(element,traces,{{...theme,title:{{text:mode==='diverse_resampled'?'Diverse resampled':'Smallest fixed',font:{{size:15}}}}}},config);}}window.PuzzletronCharts={{theme,config,renderLossChart,installConfigFocus,stableConfigKey}};const select=document.getElementById('bypass-overfit-unit-select');if(!select)return;function render(){{renderLossChart('bypass-diverse-plot','diverse_resampled',select.value);renderLossChart('bypass-fixed-plot','smallest_fixed',select.value);}}select.addEventListener('change',render);render();}})();</script>
<script>(()=>{{
  const charts=window.PuzzletronCharts;
  if(!charts)return;
  const nested=JSON.parse(document.getElementById('campaign-data').textContent).nested_bypass||{{}};
  const observations=nested.observations||[],catalog=nested.candidate_catalog||{{}};
  const colorscale=[[0,'#4f8cff'],[1,'#ff6577']];
  const select=document.getElementById('nested-bypass-unit-select');
  const configSelect=document.getElementById('nested-bypass-config-select');
  const element=document.getElementById('nested-bypass-unit-plot');
  const title=document.getElementById('nested-bypass-unit-title');
  if(!select||!configSelect||!element)return;
  const configKey=point=>charts.stableConfigKey({{hidden_width:point.hidden_width,configuration:catalog[point.candidate_id]||{{candidate_id:point.candidate_id}}}});
  function pointsForUnit(unit){{
    const legacy=!observations.length;
    let points=observations.filter(point=>Number(point.layer_idx)===Number(unit.layer_idx));
    if(unit.subblock_kind)points=points.filter(point=>point.subblock_kind===unit.subblock_kind&&point.subblock_name===unit.subblock_name);
    if(legacy){{
      const key=unit.subblock_kind?`${{unit.layer_idx}}:${{unit.subblock_kind}}:${{unit.subblock_name}}`:String(unit.layer_idx);
      points=(nested.records||[]).map(row=>({{step:row.step,dp_lane:0,loss:Number(unit.subblock_kind?(row.per_subblock_loss||{{}})[key]:(row.per_layer_loss||{{}})[key]),parameter_ratio:null,candidate_id:'legacy',hidden_width:row.hidden_width}})).filter(point=>Number.isFinite(point.loss));
    }}
    return points;
  }}
  function configLabel(point){{
    const width=point.hidden_width==null?'N/A':point.hidden_width;
    return `width=${{width}} · ${{JSON.stringify(catalog[point.candidate_id]||{{candidate_id:point.candidate_id}})}}`;
  }}
  function populateConfigSelector(){{
    const unit=JSON.parse(select.value),points=pointsForUnit(unit),previous=configSelect.value;
    const choices=new Map();
    points.forEach(point=>choices.set(configKey(point),configLabel(point)));
    configSelect.replaceChildren(new Option('ALL','ALL'));
    [...choices.entries()].sort((left,right)=>left[1].localeCompare(right[1])).forEach(([key,label])=>configSelect.add(new Option(label,key)));
    configSelect.value=[...configSelect.options].some(option=>option.value===previous)?previous:'ALL';
  }}
  function render(){{
    const unit=JSON.parse(select.value),legacy=!observations.length;
    const label=unit.subblock_kind?`layer_${{unit.layer_idx}}:${{unit.subblock_kind}}:${{unit.subblock_name}}`:`layer_${{unit.layer_idx}}`;
    if(title)title.textContent=label;
    let points=pointsForUnit(unit);
    if(configSelect.value==='ALL'){{/* Keep the complete selected-unit scatter. */}}
    else points=points.filter(point=>configKey(point)===configSelect.value);
    const meta=points.map(point=>[point.dp_lane,point.hidden_width==null?'N/A':point.hidden_width,point.candidate_id,JSON.stringify(catalog[point.candidate_id]||{{}}),point.active_params==null?'N/A':point.active_params,point.teacher_params==null?'N/A':point.teacher_params]);
    const marker={{size:8,opacity:.82,color:points.map(point=>point.parameter_ratio),cmin:0,cmax:1,colorscale,showscale:!legacy,colorbar:{{title:'Active / teacher params',thickness:14}}}};
    Plotly.react(element,[{{x:points.map(point=>Number(point.step)),y:points.map(point=>Number(point.loss)),mode:'markers',type:'scatter',name:'DP observations',marker,customdata:meta,hovertemplate:'step=%{{x}} · loss=%{{y:.6g}}<br>DP lane %{{customdata[0]}} · width=%{{customdata[1]}}<br>ratio=%{{marker.color:.4f}} · params=%{{customdata[4]}}/%{{customdata[5]}}<br>%{{customdata[2]}}<br>%{{customdata[3]}}<extra></extra>'}}],{{...charts.theme,xaxis:{{...charts.theme.xaxis,title:'Optimizer step'}},yaxis:{{...charts.theme.yaxis,title:'Loss'}},hovermode:'closest'}},charts.config);
    const configKeys=points.map(configKey);
    charts.installConfigFocus(element,configKeys);
  }}
  select.addEventListener('change',()=>{{populateConfigSelector();render();}});
  configSelect.addEventListener('change',render);
  populateConfigSelector();
  render();
}})();</script>
<script>(()=>{{const select=document.getElementById('depth-metric-select'),charts=window.PuzzletronCharts;if(!select||!charts)return;const data=JSON.parse(document.getElementById('campaign-data').textContent).depth||{{}},rows=data.rows||[];function render(){{const metric=select.value,valid=rows.filter(row=>Number.isFinite(Number((row.metrics||{{}})[metric]))),x=valid.map(row=>Number(row.removed_count)),y=valid.map(row=>Number(row.metrics[metric])),meta=valid.map(row=>[row.removed_count===0?'Teacher':JSON.stringify(row.removals)]);Plotly.react('depth-trajectory-plot',[{{x,y,mode:'lines+markers',name:metric,line:{{color:'#4f8cff',width:3}},marker:{{size:10,color:x.map(value=>value===0?'#ffbd45':'#4f8cff')}},customdata:meta,hovertemplate:'Removed sublayers %{{x}}<br>'+metric+' %{{y:.7g}}<br>%{{customdata[0]}}<extra></extra>'}}],{{...charts.theme,title:{{text:`${{metric}} by iterative depth`,font:{{size:16}}}},xaxis:{{...charts.theme.xaxis,title:'Sublayers removed',dtick:1}},yaxis:{{...charts.theme.yaxis,title:metric}},hovermode:'closest'}},charts.config);}}select.addEventListener('change',render);render();}})();</script>
<script>
(()=>{{
  const profile=document.getElementById('vllm-profile-select');
  const overviewMetric=document.getElementById('vllm-overview-metric');
  const metric=document.getElementById('vllm-metric-select');
  const axis=document.getElementById('vllm-axis-select');
  const configSelect=document.getElementById('vllm-config-select');
  const summary=document.getElementById('vllm-config-summary');
  const charts=window.PuzzletronCharts;
  if(!profile||!overviewMetric||!metric||!axis||!configSelect||!charts)return;
  // Let Plotly choose its normal vertical legend placement instead of forcing
  // a horizontal legend into the title area. Re-render the already-created
  // overfit charts once so they receive the same layout policy.
  delete charts.theme.legend;
  const bypassUnit=document.getElementById('bypass-overfit-unit-select');
  if(bypassUnit){{
    charts.renderLossChart('bypass-diverse-plot','diverse_resampled',bypassUnit.value);
    charts.renderLossChart('bypass-fixed-plot','smallest_fixed',bypassUnit.value);
  }}
  const data=JSON.parse(document.getElementById('campaign-data').textContent).vllm||{{}};
  const rows=data.records||[],labels=data.axis_labels||{{}};
  const palette=['#4f8cff','#35d07f','#ffbd45','#ff6577','#a78bfa','#22d3ee'];
  const finite=value=>Number.isFinite(Number(value));
  const canonical=value=>{{
    if(value===null||typeof value!=='object')return JSON.stringify(value);
    if(Array.isArray(value))return JSON.stringify(value);
    return JSON.stringify(Object.fromEntries(Object.entries(value).sort(([a],[b])=>a.localeCompare(b))));
  }};
  const metricRows=name=>rows.filter(row=>row.profile_id===profile.value&&finite((row.metrics||{{}})[name]));
  const profileLabel=()=>{{const selected=profile.options[profile.selectedIndex];return selected?selected.textContent:'runtime profile';}};
  const widthColor=width=>{{const index=(data.widths||[]).map(String).indexOf(String(width));return palette[(index<0?0:index)%palette.length];}};
  function renderOverview(){{
    const name=overviewMetric.value,ordered=metricRows(name).slice().sort((a,b)=>Number(a.metrics[name])-Number(b.metrics[name]));
    const rank=new Map(ordered.map((row,index)=>[row,index+1])),groups=new Map();
    ordered.forEach(row=>{{const key=`${{row.hidden_width}} · ${{row.kind}}`;if(!groups.has(key))groups.set(key,[]);groups.get(key).push(row);}});
    const traces=[...groups.entries()].map(([traceName,values])=>({{
      x:values.map(row=>rank.get(row)),y:values.map(row=>Number(row.metrics[name])),mode:'markers',name:traceName,
      marker:{{size:11,color:widthColor(values[0].hidden_width),symbol:values[0].kind==='ffn'?'square':values[0].kind==='attention'?'circle':'diamond'}},
      customdata:values.map(row=>[row.hidden_width,row.label]),
      hovertemplate:'rank=%{{x}} · %{{y:.7g}}<br>w=%{{customdata[0]}} · %{{customdata[1]}}<extra></extra>'
    }}));
    Plotly.react('vllm-overview-plot',traces,{{...charts.theme,title:{{text:`${{name}} · ${{profileLabel()}}`,font:{{size:16}}}},xaxis:{{...charts.theme.xaxis,title:`Candidate rank (ascending ${{name}})`}},yaxis:{{...charts.theme.yaxis,title:name}},hovermode:'closest'}},charts.config);
  }}
  function anchorFor(row,swept){{
    const result={{kind:row.kind}};
    Object.entries(row.axes||{{}}).sort(([a],[b])=>a.localeCompare(b)).forEach(([key,value])=>{{if(key!==swept&&key!=='hidden_width')result[key]=value;}});
    return result;
  }}
  function configLabel(anchor){{
    const parts=[anchor.kind];
    Object.entries(anchor).filter(([key])=>key!=='kind').forEach(([key,value])=>parts.push(`${{labels[key]||key}}=${{value}}`));
    return parts.join(' · ');
  }}
  let compatible=[];
  function rebuildConfigs(){{
    const swept=axis.value,groups=new Map();
    metricRows(metric.value).filter(row=>swept in (row.axes||{{}})).forEach(row=>{{
      const anchor=anchorFor(row,swept),key=canonical(anchor);
      if(!groups.has(key))groups.set(key,{{anchor,values:new Set()}});
      groups.get(key).values.add(canonical(row.axes[swept]));
    }});
    compatible=[...groups.values()].filter(group=>group.values.size>=2).sort((a,b)=>configLabel(a.anchor).localeCompare(configLabel(b.anchor)));
    configSelect.innerHTML='';
    const all=document.createElement('option');all.value='ALL';all.textContent='ALL';configSelect.appendChild(all);
    compatible.forEach((group,index)=>{{const option=document.createElement('option');option.value=String(index);option.textContent=configLabel(group.anchor);configSelect.appendChild(option);}});
    renderSweep();
  }}
  function renderSweep(){{
    const name=metric.value,swept=axis.value,choice=configSelect.value,anchor=choice==='ALL'?null:(compatible[Number(choice)]||{{}}).anchor;
    let selected=metricRows(name).filter(row=>swept in (row.axes||{{}}));
    if(anchor)selected=selected.filter(row=>canonical(anchorFor(row,swept))===canonical(anchor));
    const widths=[...new Set(selected.map(row=>row.hidden_width))].sort((a,b)=>Number(b)-Number(a));
    const traces=widths.map(width=>{{
      const values=selected.filter(row=>String(row.hidden_width)===String(width)).sort((a,b)=>Number(a.axes[swept])-Number(b.axes[swept]));
      return {{x:values.map(row=>row.axes[swept]),y:values.map(row=>Number(row.metrics[name])),mode:choice==='ALL'?'markers':'lines+markers',name:`width ${{width}}`,marker:{{size:11,color:widthColor(width)}},line:{{color:widthColor(width),width:2}},customdata:values.map(row=>[row.label]),hovertemplate:`${{labels[swept]||swept}}=%{{x}} · %{{y:.7g}}<br>%{{customdata[0]}}<extra></extra>`}};
    }});
    if(summary)summary.textContent=`${{profileLabel()}} · `+(choice==='ALL'?'ALL compatible configurations; every width remains visible as its own trace.':configLabel(anchor));
    Plotly.react('vllm-stats-plot',traces,{{...charts.theme,title:{{text:`${{name}} by ${{labels[swept]||swept}}`,font:{{size:16}}}},xaxis:{{...charts.theme.xaxis,title:labels[swept]||swept}},yaxis:{{...charts.theme.yaxis,title:name}},hovermode:'closest'}},charts.config);
  }}
  profile.addEventListener('change',()=>{{renderOverview();rebuildConfigs();}});overviewMetric.addEventListener('change',renderOverview);metric.addEventListener('change',rebuildConfigs);axis.addEventListener('change',rebuildConfigs);configSelect.addEventListener('change',renderSweep);
  renderOverview();rebuildConfigs();
}})();
</script>
<script>
(()=>{{
  const metric=document.getElementById('replacement-metric-select');
  const width=document.getElementById('replacement-width-select');
  const axis=document.getElementById('replacement-axis-select');
  const configSelect=document.getElementById('replacement-config-select');
  const connect=document.getElementById('replacement-connect-layers');
  const allLayers=document.getElementById('replacement-all-layers');
  const summary=document.getElementById('replacement-config-summary');
  const charts=window.PuzzletronCharts;
  if(!metric||!width||!axis||!configSelect||!connect||!allLayers||!charts)return;
  const data=JSON.parse(document.getElementById('campaign-data').textContent).replacement||{{}};
  const rows=data.records||[],labels=data.axis_labels||{{}};
  const toggles=[...document.querySelectorAll('[data-replacement-layer]')];
  const finite=value=>Number.isFinite(Number(value));
  const layerColor=layer=>{{
    const layerValues=(data.layers||[]).map(Number),low=Math.min(...layerValues),high=Math.max(...layerValues),ratio=high===low?0:(Number(layer)-low)/(high-low);
    const red=[255,101,119],blue=[79,140,255],rgb=red.map((value,index)=>Math.round(value+(blue[index]-value)*ratio));
    return `rgb(${{rgb.join(',')}})`;
  }};
  const canonical=value=>{{
    if(value===null||typeof value!=='object')return JSON.stringify(value);
    if(Array.isArray(value))return JSON.stringify(value);
    return JSON.stringify(Object.fromEntries(Object.entries(value).sort(([a],[b])=>a.localeCompare(b))));
  }};
  const metricRows=name=>rows.filter(row=>String(row.hidden_width)===width.value&&finite((row.metrics||{{}})[name]));
  function anchorFor(row,swept){{
    const result={{family:row.family}};
    Object.entries(row.axes||{{}}).sort(([a],[b])=>a.localeCompare(b)).forEach(([key,value])=>{{if(key!==swept&&key!=='hidden_width')result[key]=value;}});
    return result;
  }}
  function configLabel(anchor){{
    const parts=[anchor.family];
    Object.entries(anchor).filter(([key])=>key!=='family').forEach(([key,value])=>parts.push(`${{labels[key]||key}}=${{value}}`));
    return parts.join(' · ');
  }}
  let compatible=[];
  function rebuildConfigs(){{
    const swept=axis.value,groups=new Map();
    metricRows(metric.value).filter(row=>swept in (row.axes||{{}})).forEach(row=>{{
      const anchor=anchorFor(row,swept),key=canonical(anchor);
      if(!groups.has(key))groups.set(key,{{anchor,values:new Set()}});
      groups.get(key).values.add(canonical(row.axes[swept]));
    }});
    compatible=[...groups.values()].filter(group=>group.values.size>=2).sort((a,b)=>configLabel(a.anchor).localeCompare(configLabel(b.anchor)));
    configSelect.innerHTML='<option value="ALL">ALL</option>';
    compatible.forEach((group,index)=>{{const option=document.createElement('option');option.value=String(index);option.textContent=configLabel(group.anchor);configSelect.appendChild(option);}});
    render();
  }}
  function render(){{
    const name=metric.value,swept=axis.value,choice=configSelect.value,anchor=choice==='ALL'?null:(compatible[Number(choice)]||{{}}).anchor;
    const enabled=new Set(toggles.filter(item=>item.checked).map(item=>String(item.dataset.replacementLayer)));
    let selected=metricRows(name).filter(row=>enabled.has(String(row.layer_idx))&&swept in (row.axes||{{}}));
    if(anchor)selected=selected.filter(row=>canonical(anchorFor(row,swept))===canonical(anchor));
    const connectLayers=choice!=='ALL'&&connect.checked;
    connect.disabled=choice==='ALL';
    connect.parentElement.style.opacity=choice==='ALL'?'.5':'1';
    const layers=[...new Set(selected.map(row=>Number(row.layer_idx)))].sort((a,b)=>a-b);
    const traces=layers.map(layer=>{{
      const values=selected.filter(row=>Number(row.layer_idx)===layer).sort((a,b)=>Number(a.axes[swept])-Number(b.axes[swept]));
      return {{
        x:values.map(row=>row.axes[swept]),y:values.map(row=>Number(row.metrics[name])),mode:connectLayers?'lines+markers':'markers',name:`Layer ${{layer}}`,showlegend:false,
        marker:{{size:9,color:layerColor(layer),opacity:.82}},line:{{color:layerColor(layer),width:1.5}},
        customdata:values.map(row=>[row.layer_idx,row.label,(row.sliced_teacher_baseline||{{}})[name]]),
        hovertemplate:`layer %{{customdata[0]}} · ${{labels[swept]||swept}}=%{{x}}<br>${{name}}=%{{y:.7g}}<br>baseline=%{{customdata[2]:.7g}}<br>%{{customdata[1]}}<extra>width ${{width.value}}</extra>`
      }};
    }});
    if(summary)summary.textContent=`width ${{width.value}} · `+(choice==='ALL'?`ALL compatible configurations · ${{selected.length}} visible scores`:`${{configLabel(anchor)}} · ${{selected.length}} visible scores`);
    Plotly.react('replacement-score-plot',traces,{{...charts.theme,title:{{text:`${{name}} by ${{labels[swept]||swept}} · width ${{width.value}}`,font:{{size:16}}}},xaxis:{{...charts.theme.xaxis,title:labels[swept]||swept}},yaxis:{{...charts.theme.yaxis,title:name}},hovermode:'closest',showlegend:false}},charts.config);
  }}
  function syncMaster(){{
    const selected=toggles.filter(toggle=>toggle.checked).length;
    allLayers.checked=selected===toggles.length;
    allLayers.indeterminate=selected>0&&selected<toggles.length;
  }}
  metric.addEventListener('change',rebuildConfigs);width.addEventListener('change',rebuildConfigs);axis.addEventListener('change',rebuildConfigs);configSelect.addEventListener('change',render);connect.addEventListener('change',render);
  allLayers.addEventListener('change',()=>{{toggles.forEach(toggle=>{{toggle.checked=allLayers.checked;}});allLayers.indeterminate=false;render();}});
  toggles.forEach(toggle=>toggle.addEventListener('change',()=>{{syncMaster();render();}}));
  syncMaster();
  rebuildConfigs();
}})();
</script>
<script>
(()=>{{
  const xSelect=document.getElementById('evaluation-x-select');
  const ySelect=document.getElementById('evaluation-y-select');
  const element=document.getElementById('evaluation-scatter-plot');
  const summary=document.getElementById('evaluation-profile-summary');
  const charts=window.PuzzletronCharts;
  if(!xSelect||!ySelect||!element||!charts)return;
  const report=JSON.parse(document.getElementById('campaign-data').textContent);
  const profiles=(report.evaluation||{{}}).profiles||[];
  let profileId=((window.PuzzletronReportState||{{}}).mipConstraintProfile||{{}}).id||profiles[0]?.profile_id;
  const finite=value=>Number.isFinite(Number(value));
  function selectedProfile(){{return profiles.find(row=>String(row.profile_id)===String(profileId))||profiles[0];}}
  function render(){{
    const profile=selectedProfile();if(!profile)return;
    const xName=xSelect.value,yName=ySelect.value;
    const rows=(profile.rows||[]).filter(row=>finite((row.x||{{}})[xName])&&finite((row.metrics||{{}})[yName]));
    const traces=rows.map(row=>({{
      x:[Number(row.x[xName])],y:[Number(row.metrics[yName])],name:row.label,mode:'markers',
      marker:{{color:row.color,symbol:row.marker||'circle',size:row.solution_id==='teacher'?18:13,line:{{color:'#e7edf7',width:row.solution_id==='teacher'?1.5:.5}}}},
      customdata:[[row.solution_id,row.hidden_width,row.removed_sublayers,row.parameter_ratio,row.checkpoint]],
      hovertemplate:`${{row.label}}<br>${{xName}}=%{{x:.7g}}<br>${{yName}}=%{{y:.7g}}<br>width=%{{customdata[1]}} · drop=%{{customdata[2]}}<br>ratio=%{{customdata[3]:.4f}}<extra></extra>`
    }}));
    if(summary)summary.textContent=`${{profile.profile_id}} · ${{profile.workload_id}} · ${{rows.length}} evaluated models`;
    Plotly.react(element,traces,{{...charts.theme,title:{{text:`${{yName}} vs ${{xName}}`,font:{{size:16}}}},xaxis:{{...charts.theme.xaxis,title:xName}},yaxis:{{...charts.theme.yaxis,title:yName}},hovermode:'closest'}},charts.config);
  }}
  xSelect.addEventListener('change',render);ySelect.addEventListener('change',render);
  window.addEventListener('puzzletron:mip-constraint-change',event=>{{profileId=event.detail.profile.id;render();}});
  render();
}})();
</script>
<script>
(()=>{{
  const workload=document.getElementById('aiperf-workload-select');
  const topology=document.getElementById('aiperf-topology-select');
  const statistic=document.getElementById('aiperf-stat-select');
  const allButton=document.getElementById('aiperf-all-solutions');
  const noneButton=document.getElementById('aiperf-no-solutions');
  const summary=document.getElementById('aiperf-profile-summary');
  const charts=window.PuzzletronCharts;
  if(!workload||!topology||!statistic||!charts)return;
  const report=JSON.parse(document.getElementById('campaign-data').textContent);
  const profiles=(report.aiperf||{{}}).profiles||[];
  const toggles=[...document.querySelectorAll('[data-aiperf-solution]')];
  let profileId=((window.PuzzletronReportState||{{}}).mipConstraintProfile||{{}}).id||profiles[0]?.profile_id;
  const finite=value=>Number.isFinite(Number(value));
  const topologySymbols={{'tp2-pp1-dp1-ep1-pcp1-dcp1':'circle','tp1-pp1-dp2-ep1-pcp1-dcp1':'diamond','tp1-pp2-dp1-ep1-pcp1-dcp1':'square'}};
  function selectedProfile(){{return profiles.find(row=>String(row.profile_id)===String(profileId)&&String(row.workload_id)===String(workload.value));}}
  function enabled(){{return new Set(toggles.filter(toggle=>toggle.checked||toggle.disabled).map(toggle=>toggle.dataset.aiperfSolution));}}
  function nondominated(points,yDirection){{return points.filter((point,index)=>!points.some((other,j)=>j!==index&&Number(other.x)>=Number(point.x)&&(yDirection==='max'?Number(other.y)>=Number(point.y):Number(other.y)<=Number(point.y))&&(Number(other.x)>Number(point.x)||(yDirection==='max'?Number(other.y)>Number(point.y):Number(other.y)<Number(point.y)))));}}
  function metricValue(row,key,fallback){{const metrics=row.metrics||{{}},value=metrics[key];return finite(value)?Number(value):(fallback&&finite(metrics[fallback])?Number(metrics[fallback]):null);}}
  function renderPlot(elementId,yBase,title,yDirection='min'){{
    const profile=selectedProfile();if(!profile)return;
    const active=enabled(),stat=statistic.value;
    const yKey=yBase.includes('{{stat}}')?yBase.replace('{{stat}}',stat):yBase;
    const fallback=yBase.includes('{{stat}}')?yBase.replace('{{stat}}','mean'):null;
    let rows=(profile.rows||[]).filter(row=>active.has(row.solution_id)&&finite((row.metrics||{{}}).output_token_throughput));
    if(topology.value!=='PARETO')rows=rows.filter(row=>String(row.topology_id)===topology.value);
    const solutionIds=[...new Set(rows.map(row=>row.solution_id))];
    const traces=solutionIds.map(solutionId=>{{
      const source=rows.filter(row=>row.solution_id===solutionId).map(row=>({{row,x:Number(row.metrics.output_token_throughput),y:metricValue(row,yKey,fallback)}})).filter(point=>finite(point.y));
      const points=(topology.value==='PARETO'?nondominated(source,yDirection):source).sort((a,b)=>Number(a.row.concurrency)-Number(b.row.concurrency));
      const style=points[0]?.row||{{}},teacher=solutionId==='teacher';
      return {{x:points.map(point=>point.x),y:points.map(point=>point.y),name:style.label||solutionId,mode:'lines+markers',
        marker:{{color:style.color||'#4f8cff',size:teacher?15:10,symbol:points.map(point=>teacher?'star':(topology.value==='PARETO'?(topologySymbols[point.row.topology_id]||'circle'):(style.marker||'circle'))),line:{{color:'#e7edf7',width:teacher?1.2:.4}}}},
        line:{{color:style.color||'#4f8cff',width:2}},customdata:points.map(point=>[point.row.concurrency,point.row.topology_id,point.row.solution_id]),
        hovertemplate:`${{style.label||solutionId}}<br>throughput=%{{x:.7g}}<br>${{title}}=%{{y:.7g}}<br>concurrency=%{{customdata[0]}}<br>%{{customdata[1]}}<extra></extra>`}};
    }}).filter(trace=>trace.x.length);
    Plotly.react(elementId,traces,{{...charts.theme,title:{{text:title,font:{{size:15}}}},xaxis:{{...charts.theme.xaxis,title:'Output token throughput (tokens/s)'}},yaxis:{{...charts.theme.yaxis,title}},hovermode:'closest'}},charts.config);
  }}
  function render(){{
    const profile=selectedProfile();
    if(summary)summary.textContent=profile?`${{profile.profile_id}} · ${{profile.workload_id}} · ${{topology.value}} · concurrencies 1, 4, 16, 64`:`No AIPerf data for ${{profileId}} / ${{workload.value}}`;
    renderPlot('aiperf-ttft-throughput-plot','ttft_{{stat}}_ms',`${{statistic.value.toUpperCase()}} TTFT (ms)`,'min');
    renderPlot('aiperf-latency-throughput-plot','request_latency_{{stat}}_ms',`${{statistic.value.toUpperCase()}} request latency (ms)`,'min');
    renderPlot('aiperf-interactivity-throughput-plot','output_token_throughput_per_user_{{stat}}','Interactivity (tokens/s/user)','max');
    renderPlot('aiperf-tpot-throughput-plot','tpot_{{stat}}_ms',`${{statistic.value.toUpperCase()}} TPOT (ms)`,'min');
  }}
  function syncProfiles(){{const available=profiles.filter(row=>String(row.profile_id)===String(profileId)).map(row=>String(row.workload_id));[...workload.options].forEach(option=>option.disabled=!available.includes(option.value));if(!available.includes(workload.value)&&available.length)workload.value=available[0];render();}}
  workload.addEventListener('change',render);topology.addEventListener('change',render);statistic.addEventListener('change',render);toggles.forEach(toggle=>toggle.addEventListener('change',render));
  allButton?.addEventListener('click',()=>{{toggles.forEach(toggle=>toggle.checked=true);render();}});
  noneButton?.addEventListener('click',()=>{{toggles.forEach(toggle=>{{if(!toggle.disabled)toggle.checked=false;}});render();}});
  window.addEventListener('puzzletron:mip-constraint-change',event=>{{profileId=event.detail.profile.id;syncProfiles();}});
  syncProfiles();
}})();
</script>
<script>
(()=>{{
  const select=document.getElementById('mip-constraint-select');
  const head=document.getElementById('mip-solution-head');
  const body=document.getElementById('mip-solution-body');
  const summary=document.getElementById('mip-constraint-summary');
  if(!select||!head||!body)return;
  const report=JSON.parse(document.getElementById('campaign-data').textContent);
  const data=report.mip||{{}},profiles=data.profiles||[],columns=data.columns||[];
  const byId=new Map(profiles.map(profile=>[String(profile.id),profile]));
  const esc=value=>String(value).replace(/[&<>"']/g,char=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[char]));
  const label=name=>name.replaceAll('_',' ').replace(/(^|\\s)([a-z])/g,(_,prefix,char)=>prefix+char.toUpperCase());
  const numeric=new Intl.NumberFormat(undefined,{{maximumSignificantDigits:7}});
  let sortColumn=null,sortDirection='asc';
  function format(name,value){{
    if(value==null||Number.isNaN(value))return 'N/A';
    if(typeof value!=='number')return String(value);
    if(name==='parameter_ratio')return `${{(100*value).toFixed(3).replace(/0+$/,'').replace(/\\.$/,'')}}%`;
    return numeric.format(value);
  }}
  function publish(profile){{
    window.PuzzletronReportState=window.PuzzletronReportState||{{}};
    window.PuzzletronReportState.mipConstraintProfile=profile;
    try{{localStorage.setItem(`puzzletron-mip-profile:${{report.root}}`,String(profile.id));}}catch(error){{}}
    window.dispatchEvent(new CustomEvent('puzzletron:mip-constraint-change',{{detail:{{profile}}}}));
  }}
  function orderedRows(rows){{
    const teacher=rows.find(row=>row.label==='Teacher'),candidates=rows.filter(row=>row!==teacher);
    if(sortColumn)candidates.sort((left,right)=>{{
      const a=(left.outputs||{{}})[sortColumn],b=(right.outputs||{{}})[sortColumn];
      if(a==null&&b==null)return 0;if(a==null)return 1;if(b==null)return -1;
      const comparison=typeof a==='number'&&typeof b==='number'?a-b:String(a).localeCompare(String(b),undefined,{{numeric:true}});
      return sortDirection==='asc'?comparison:-comparison;
    }});
    return teacher?[teacher,...candidates]:candidates;
  }}
  function render(){{
    const profile=byId.get(select.value)||profiles[0];if(!profile)return;
    head.innerHTML=`<tr><th>Solution</th>${{columns.map(name=>{{const active=sortColumn===name,arrow=active?(sortDirection==='asc'?'▲':'▼'):'';return `<th title="${{esc(name)}}" aria-sort="${{active?(sortDirection==='asc'?'ascending':'descending'):'none'}}"><button class="mip-sort-button" data-mip-sort="${{esc(name)}}">${{esc(label(name))}}<span class="mip-sort-arrow">${{arrow}}</span></button></th>`;}}).join('')}}</tr>`;
    body.innerHTML=orderedRows(profile.rows||[]).map(row=>{{
      const outputs=row.outputs||{{}},status=String(outputs.status||'unknown');
      return `<tr><th>${{esc(row.label)}}</th>${{columns.map(name=>`<td class="${{name==='status'?'status-'+status:''}}">${{esc(format(name,outputs[name]))}}</td>`).join('')}}</tr>`;
    }}).join('');
    head.querySelectorAll('[data-mip-sort]').forEach(button=>button.addEventListener('click',()=>{{const name=button.dataset.mipSort;if(sortColumn===name)sortDirection=sortDirection==='asc'?'desc':'asc';else{{sortColumn=name;sortDirection='asc';}}render();}}));
    const constraint=profile.constraint||{{}},runtime=profile.runtime_profile||{{}},concurrency=runtime.max_num_seqs||runtime.batch_size||'N/A';
    const constraintText=constraint.constraint_type==='latency_ratio'
      ? `limit ${{numeric.format(constraint.latency_limit_ms)}} ms · denominator ${{numeric.format(constraint.latency_denominator_ms)}} ms`
      : `limit ${{numeric.format(constraint.parameter_limit)}} parameters · denominator ${{numeric.format(constraint.parameter_denominator)}}`;
    if(summary)summary.textContent=`${{profile.label}} · ${{constraintText}} · ISL ${{runtime.prefill_seq_len??'N/A'}} · OSL ${{runtime.generation_seq_len??'N/A'}} · concurrency ${{concurrency}}`;
    publish(profile);
  }}
  let saved=null;try{{saved=localStorage.getItem(`puzzletron-mip-profile:${{report.root}}`);}}catch(error){{}}
  if(saved&&byId.has(saved))select.value=saved;
  select.addEventListener('change',render);render();
}})();
</script>
<script>
(()=>{{
  const container=document.getElementById('distillation-overfit-plots');
  const charts=window.PuzzletronCharts;
  if(!container||!charts)return;
  const report=JSON.parse(document.getElementById('campaign-data').textContent);
  const profile=((report.distillation_overfit||{{}}).profiles||[])[0];
  if(!profile)return;
  const toggles=[...document.querySelectorAll('[data-distillation-overfit-solution]')];
  const finite=value=>Number.isFinite(Number(value));
  const median=values=>{{const sorted=values.slice().sort((a,b)=>a-b),middle=Math.floor(sorted.length/2);return sorted.length%2?sorted[middle]:(sorted[middle-1]+sorted[middle])/2;}};
  container.innerHTML=(profile.metrics||[]).map((metric,index)=>`<article class="probe-plot-panel"><div id="distillation-overfit-${{index}}" class="plotly-chart"></div></article>`).join('');
  function render(){{
    const enabled=new Set(toggles.filter(toggle=>toggle.checked).map(toggle=>toggle.dataset.distillationOverfitSolution));
    (profile.metrics||[]).forEach((metric,index)=>{{
      const traces=(profile.solutions||[]).filter(solution=>enabled.has(String(solution.solution_id))).map(solution=>{{
        const rows=(solution.records||[]).filter(row=>finite(row[metric]));
        const values=rows.map(row=>Number(row[metric])),first=values.slice(0,4),last=values.slice(-4),down=first.length===4&&last.length===4&&median(last)<median(first);
        return {{x:rows.map((row,i)=>Number(row.step??row.global_step??i)),y:values,name:solution.label||solution.solution_id,mode:'lines+markers',
          line:{{color:solution.color||'#4f8cff',width:2}},marker:{{color:solution.color||'#4f8cff',size:5}},
          customdata:rows.map(()=>[solution.solution_id,down?'decreased':'not decreased']),hovertemplate:`${{solution.label||solution.solution_id}}<br>step=%{{x}}<br>${{metric}}=%{{y:.7g}}<br>%{{customdata[1]}}<extra></extra>`}};
      }});
      Plotly.react(`distillation-overfit-${{index}}`,traces,{{...charts.theme,title:{{text:metric,font:{{size:15}}}},xaxis:{{...charts.theme.xaxis,title:'Optimizer step'}},yaxis:{{...charts.theme.yaxis,title:metric}},hovermode:'closest'}},charts.config);
    }});
  }}
  toggles.forEach(toggle=>toggle.addEventListener('change',render));render();
}})();
</script>
<script>
(()=>{{
  const container=document.getElementById('proper-distillation-plots');
  const charts=window.PuzzletronCharts;
  if(!container||!charts)return;
  const report=JSON.parse(document.getElementById('campaign-data').textContent);
  const runs=((report.proper_distillation||{{}}).runs||[]);
  const run=runs[runs.length-1];if(!run)return;
  const finite=value=>Number.isFinite(Number(value));
  container.innerHTML=(run.metrics||[]).map((metric,index)=>`<article class="probe-plot-panel"><div id="proper-distillation-${{index}}" class="plotly-chart"></div></article>`).join('');
  (run.metrics||[]).forEach((metric,index)=>{{
    const rows=(run.records||[]).filter(row=>finite(row[metric]));
    Plotly.react(`proper-distillation-${{index}}`,[{{
      x:rows.map((row,i)=>Number(row.step??row.global_step??i)),
      y:rows.map(row=>Number(row[metric])),name:run.label||run.solution_id,
      mode:'lines',line:{{color:run.color||'#4f8cff',width:2}},
      hovertemplate:`step=%{{x}}<br>${{metric}}=%{{y:.7g}}<extra></extra>`
    }}],{{...charts.theme,title:{{text:metric,font:{{size:15}}}},xaxis:{{...charts.theme.xaxis,title:'Optimizer step'}},yaxis:{{...charts.theme.yaxis,title:metric}},hovermode:'closest'}},charts.config);
  }});
}})();
</script>
</main></body></html>"""
    temporary = html_path.with_suffix(".tmp")
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(html_path)
    return {"html": str(html_path)}


__all__ = ["generate_campaign_progress_report"]
