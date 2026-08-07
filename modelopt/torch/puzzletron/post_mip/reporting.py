# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTML fragments for reportable post-MIP node types."""

from __future__ import annotations

import hashlib
import html
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

__all__ = [
    "build_post_mip_report_payloads",
    "render_aiperf_report",
    "render_downstream_evaluation_report",
    "render_evaluation_report",
    "render_global_kd_report",
]

_CANDIDATE_COLORS = (
    "#4f8cff",
    "#ffbd45",
    "#35d07f",
    "#ff6577",
    "#a78bfa",
    "#22d3ee",
    "#fb923c",
    "#f472b6",
    "#84cc16",
    "#60a5fa",
    "#facc15",
    "#2dd4bf",
)


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _execution(node_root: Path, summary: Mapping[str, Any]) -> tuple[str, Path | None]:
    identity = str(summary.get("execution_identity") or "")
    if not identity:
        identity = str(_json(node_root / "current.json").get("execution_identity") or "")
    executions = node_root / "executions"
    if identity:
        return identity, executions / identity
    candidates = [path for path in executions.glob("*") if path.is_dir()]
    if not candidates:
        return "", None
    latest = max(candidates, key=lambda path: path.stat().st_mtime_ns)
    return latest.name, latest


def _execution_observations(execution_root: Path | None) -> list[dict[str, Any]]:
    if execution_root is None:
        return []
    observations = _json(execution_root / "observations.json")
    if isinstance(observations, list):
        return [dict(row) for row in observations if isinstance(row, Mapping)]
    rows = []
    for shard_path in sorted((execution_root / "shards").glob("shard_*.json")):
        shard = _json(shard_path)
        if isinstance(shard, list):
            rows.extend(dict(row) for row in shard if isinstance(row, Mapping))
    return rows


def _architecture_id(
    row: Mapping[str, Any],
    revisions: Mapping[str, Mapping[str, Any]],
) -> str:
    if row.get("architecture_id"):
        return str(row["architecture_id"])
    for key in ("input_revision_id", "source_revision_id", "output_revision_id"):
        revision = revisions.get(str(row.get(key) or ""))
        if revision and revision.get("architecture_id"):
            return str(revision["architecture_id"])
    return "architecture_unknown"


def _candidate_color(architecture_id: str) -> str:
    digest = hashlib.sha256(architecture_id.encode()).digest()
    return _CANDIDATE_COLORS[int.from_bytes(digest[:2], "big") % len(_CANDIDATE_COLORS)]


def _candidate_label(
    architecture_id: str,
    revision_id: str,
    revisions: Mapping[str, Mapping[str, Any]],
) -> str:
    current_id = revision_id
    visited = set()
    width = None
    kind = None
    while current_id and current_id not in visited:
        visited.add(current_id)
        revision = revisions.get(current_id) or {}
        artifact = dict(revision.get("artifact") or {})
        width = width if width is not None else artifact.get("hidden_width")
        kind = kind or artifact.get("kind")
        current_id = str(revision.get("parent_revision_id") or "")
    prefix = f"h{width} · " if width is not None else ""
    suffix = architecture_id.removeprefix("architecture_")[:8]
    return f"{prefix}{kind} · {suffix}" if kind else f"{prefix}{suffix}"


def _run_records(path: Path) -> list[dict[str, Any]]:
    records = []
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return records
    for line in lines:
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, Mapping):
            records.append(dict(record))
    return records


def _kd_runs(
    execution_root: Path | None,
    observations: list[dict[str, Any]],
    revisions: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_architecture = {
        _architecture_id(row, revisions): row for row in observations
    }
    architecture_ids = set(by_architecture)
    checkpoint_root = execution_root / "checkpoints" if execution_root else None
    if checkpoint_root and checkpoint_root.is_dir():
        architecture_ids.update(
            path.name for path in checkpoint_root.glob("architecture_*") if path.is_dir()
        )
    runs = []
    for architecture_id in sorted(architecture_ids):
        row = by_architecture.get(architecture_id, {})
        architecture_root = (
            checkpoint_root / architecture_id if checkpoint_root is not None else None
        )
        records = (
            _run_records(architecture_root / "checkpoints" / "training.jsonl")
            if architecture_root is not None
            else []
        )
        revision_id = str(
            row.get("input_revision_id") or row.get("source_revision_id") or ""
        )
        runs.append(
            {
                "architecture_id": architecture_id,
                "label": _candidate_label(architecture_id, revision_id, revisions),
                "color": _candidate_color(architecture_id),
                "status": row.get("status") or ("running" if architecture_root else "pending"),
                "records": records,
                "error": row.get("error"),
            }
        )
    return runs


def build_post_mip_report_payloads(
    root: Path,
    nodes,
) -> dict[str, dict[str, Any]]:
    """Collect current node observations and lineage once for all node renderers."""

    registry = _json(root / "artifacts" / "post_mip" / "candidate_registry.json")
    revisions = dict(registry.get("revisions") or {})
    raw: dict[str, dict[str, Any]] = {}
    selected_by: dict[str, list[str]] = {}
    for node in nodes:
        node_root = root / "artifacts" / "post_mip" / "nodes" / node.node_id
        summary = _json(node_root / "summary.json")
        if not summary:
            summary = _json(node_root / "manual_review.json")
        identity, execution_root = _execution(node_root, summary)
        observations = _execution_observations(execution_root)
        raw[node.stage_id] = {
            **dict(summary),
            "status": summary.get("status")
            or (
                "failed"
                if observations
                and all(row.get("status") in {"failed", "timed_out"} for row in observations)
                else "pending"
            ),
            "execution_identity": identity,
            "execution_root": str(execution_root) if execution_root else "",
            "observations": observations,
        }
        for row in observations:
            if row.get("status") != "selected":
                continue
            for key in ("input_revision_id", "source_revision_id", "output_revision_id"):
                revision_id = str(row.get(key) or "")
                if revision_id:
                    selected_by.setdefault(revision_id, []).append(node.node_id)

    payloads = {}
    for node in nodes:
        payload = raw[node.stage_id]
        observations = []
        for row in payload["observations"]:
            revision_id = str(
                row.get("input_revision_id") or row.get("source_revision_id") or ""
            )
            architecture_id = _architecture_id(row, revisions)
            observations.append(
                {
                    **row,
                    "architecture_id": architecture_id,
                    "label": _candidate_label(architecture_id, revision_id, revisions),
                    "color": _candidate_color(architecture_id),
                    "selected_by": sorted(set(selected_by.get(revision_id, ()))),
                }
            )
        section_id = "post-" + "-".join(
            part.replace("_", "-") for part in (node.flow_id, node.node_id)
        )
        payloads[node.stage_id] = {
            **payload,
            "section_id": section_id,
            "observations": observations,
            "runs": (
                _kd_runs(
                    Path(payload["execution_root"])
                    if payload["execution_root"]
                    else None,
                    observations,
                    revisions,
                )
                if node.node_type == "global_kd"
                else []
            ),
        }
    return payloads


def _text(value: Any) -> str:
    return html.escape(str(value))


def _number(value: Any) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "—"
    number = float(value)
    if not math.isfinite(number):
        return "—"
    return f"{number:.6g}"


def _status_summary(payload: Mapping[str, Any]) -> str:
    observations = list(payload.get("observations") or ())
    counts: dict[str, int] = {}
    for row in observations:
        status = str(row.get("status") or "pending")
        counts[status] = counts.get(status, 0) + 1
    outcomes = " · ".join(
        f"{_text(status.replace('_', ' '))}={count}"
        for status, count in sorted(counts.items())
    )
    status = _text(payload.get("status") or "pending")
    return f"<p>status={status}{f' · {outcomes}' if outcomes else ''}</p>"


def render_evaluation_report(section_id: str, payload: Mapping[str, Any]) -> str:
    """Render one candidate-evaluation node."""

    observations = list(payload.get("observations") or ())
    metric_names = sorted(
        {
            str(metric)
            for row in observations
            for metric, value in dict(row.get("metrics") or {}).items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    )
    rows = []
    for row in observations:
        metrics = dict(row.get("metrics") or {})
        cells = "".join(f"<td>{_number(metrics.get(metric))}</td>" for metric in metric_names)
        rows.append(
            "<tr>"
            f"<td>{_text(row.get('label') or row.get('architecture_id') or 'unknown')}</td>"
            f"<td>{_text(row.get('status') or 'pending')}</td>"
            f"{cells}"
            f"<td>{_text(row.get('error') or '')}</td>"
            "</tr>"
        )
    headings = "".join(f"<th>{_text(metric)}</th>" for metric in metric_names)
    table = (
        "<div class='table-wrap'><table><thead><tr>"
        f"<th>Candidate</th><th>Status</th>{headings}<th>Error</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
        if rows
        else "<p class='empty'>No evaluation observations are available yet.</p>"
    )
    plot = (
        f"<div id='{_text(section_id)}-metrics' class='plotly-chart depth-chart' "
        "role='img' aria-label='Candidate evaluation metrics'></div>"
        if metric_names
        else ""
    )
    return (
        "<h3>Candidate evaluation</h3>"
        f"{_status_summary(payload)}"
        f"{plot}{table}"
    )


def render_aiperf_report(section_id: str, payload: Mapping[str, Any]) -> str:
    """
    Render AIPerf candidate status, performance metrics, selection markers, and errors.
    
    Parameters:
        section_id (str): Identifier used to scope the throughput chart element.
        payload (Mapping[str, Any]): AIPerf observations and status data.
    
    Returns:
        str: HTML fragment containing the status summary, throughput chart placeholder,
            and candidate metrics table.
    """

    observations = list(payload.get("observations") or ())
    rows = []
    for row in observations:
        metrics = dict(row.get("metrics") or {})
        selected = bool(row.get("selected_by"))
        selection = "Selected by Fastest" if selected else ""
        rows.append(
            f"<tr class='{'selected-candidate' if selected else ''}'>"
            f"<td>{_text(row.get('label') or row.get('architecture_id') or 'unknown')}</td>"
            f"<td>{_text(row.get('status') or 'pending')}</td>"
            f"<td>{_text(selection)}</td>"
            f"<td>{_number(metrics.get('request_throughput'))}</td>"
            f"<td>{_number(metrics.get('output_token_throughput'))}</td>"
            f"<td>{_number(metrics.get('ttft_mean_ms'))}</td>"
            f"<td>{_number(metrics.get('tpot_mean_ms'))}</td>"
            f"<td>{_text(row.get('error') or '')}</td>"
            "</tr>"
        )
    table = (
        "<div class='table-wrap'><table><thead><tr>"
        "<th>Candidate</th><th>Status</th><th>Selection</th>"
        "<th>Requests/s</th><th>Output tokens/s</th><th>TTFT mean (ms)</th>"
        "<th>TPOT mean (ms)</th><th>Error</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
        if rows
        else "<p class='empty'>No AIPerf observations are available yet.</p>"
    )
    return (
        "<h3>AIPerf candidate trade-offs</h3>"
        f"{_status_summary(payload)}"
        f"<div id='{_text(section_id)}-throughput' class='plotly-chart depth-chart' "
        "role='img' aria-label='AIPerf throughput trade-offs'></div>"
        f"{table}"
    )


def render_downstream_evaluation_report(section_id: str, payload: Mapping[str, Any]) -> str:
    """Render lmms-eval task metrics for downstream-evaluation nodes."""

    return render_evaluation_report(section_id, payload).replace(
        "<h3>Candidate evaluation</h3>",
        "<h3>Downstream evaluation</h3>",
        1,
    )


def render_global_kd_report(section_id: str, payload: Mapping[str, Any]) -> str:
    """
    Render the Short KD comparison with candidate statuses, loss plots, and run summaries.
    
    Parameters:
        section_id (str): Identifier used to generate unique plot element IDs.
        payload (Mapping[str, Any]): Short KD runs and status data to display.
    
    Returns:
        str: HTML fragment containing the comparison summary, plot placeholders, and run table.
    """

    runs = list(payload.get("runs") or ())
    rows = []
    for run in runs:
        records = list(run.get("records") or ())
        final = records[-1] if records else {}
        rows.append(
            "<tr>"
            f"<td><span class='candidate-swatch' style='background:{_text(run.get('color') or '#8090a8')}'></span>"
            f"{_text(run.get('label') or run.get('architecture_id') or 'unknown')}</td>"
            f"<td>{_text(run.get('status') or 'pending')}</td>"
            f"<td>{len(records)}</td>"
            f"<td>{_number(final.get('loss', final.get('train_loss')))}</td>"
            f"<td>{_number(final.get('kd_loss'))}</td>"
            f"<td>{_text(run.get('error') or '')}</td>"
            "</tr>"
        )
    table = (
        "<div class='table-wrap'><table><thead><tr>"
        "<th>Candidate</th><th>Status</th><th>Logged steps</th>"
        "<th>Final loss</th><th>Final KD loss</th><th>Error</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
        if rows
        else "<p class='empty'>No Short KD runs are available yet.</p>"
    )
    return (
        "<h3>Short KD comparison</h3>"
        f"{_status_summary(payload)}"
        "<div class='probe-plots'>"
        f"<article class='probe-plot-panel'><div id='{_text(section_id)}-loss' "
        "class='plotly-chart' role='img' aria-label='Short KD total loss'></div></article>"
        f"<article class='probe-plot-panel'><div id='{_text(section_id)}-kd-loss' "
        "class='plotly-chart' role='img' aria-label='Short KD loss component'></div></article>"
        f"</div>{table}"
    )
