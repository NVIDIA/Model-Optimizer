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

"""Replaceable human views over one validated Puzzletron result."""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .run_reporting import canonical_json_bytes, result_sha256, validate_result

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["RenderedView", "render_run_html", "render_run_text"]


@dataclass(frozen=True)
class RenderedView:
    """Rendered bytes and their source-bearing derived-view manifest."""

    content: bytes
    manifest: dict[str, Any]


def _value(value: object) -> str:
    return "unknown" if value is None else str(value)


def _compact(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _row(*values: object) -> str:
    return "<tr>" + "".join(f"<td>{html.escape(_value(value))}</td>" for value in values) + "</tr>"


def _measure_text(progress: Mapping[str, Any]) -> str:
    rows = []
    for measure in progress["measures"]:
        total = measure.get("total")
        amount = f"{measure['completed']}" if total is None else f"{measure['completed']}/{total}"
        rows.append(f"{measure['name']}: {amount} {measure['unit']} ({measure['total_kind']})")
    return "; ".join(rows)


def _eta_text(progress: Mapping[str, Any]) -> str:
    eta = progress["eta"]
    if eta.get("qualified") is True and isinstance(eta.get("seconds"), (int, float)):
        return f"{eta['seconds']:.0f}s ({eta.get('method', 'observed')})"
    return f"unavailable ({eta.get('unavailable_reason', 'not_reported')})"


def render_run_text(view: Mapping[str, Any]) -> str:
    """Render a compact status from the authoritative result."""

    validate_result(view)
    run = view["run"]
    status = run["status"]
    freshness = run["freshness"]
    counts = view.get("stage_counts") or {}
    lines = [
        f"Run {run['identity']['run_id']}",
        f"Status: {status['execution']}; controller: {status['attachment']}; evidence: {status['evidence']}",
        f"Freshness: {freshness['state']} as of {_value(freshness.get('as_of'))}",
        "Stages: " + (", ".join(f"{name}={count}" for name, count in counts.items()) or "none"),
    ]
    if freshness.get("reason"):
        lines.append(f"Freshness reason: {freshness['reason']}")
    active = view.get("active_progress") or []
    lines.append("Active progress:")
    lines.extend(
        f"  {item['stage_id']}: {_measure_text(item)}; ETA {_eta_text(item)}" for item in active
    )
    if not active:
        lines.append("  none")
    return "\n".join(lines) + "\n"


def render_run_html(
    result: Mapping[str, Any], *, generated_at: str, renderer_revision: str
) -> RenderedView:
    """Render a self-contained optional HTML summary from exactly one result JSON."""

    if not generated_at or not renderer_revision:
        raise ValueError("generated_at and renderer_revision must be non-empty")
    validate_result(result)
    run = result["run"]
    status = run["status"]
    freshness = run["freshness"]
    subject_rows = (
        "".join(
            _row(
                subject["role"],
                subject["subject_id"],
                _compact(subject["checkpoint"]),
                _compact(subject["architecture"]),
            )
            for subject in result["subjects"]
        )
        or '<tr><td colspan="4">No subjects recorded.</td></tr>'
    )
    stage_rows = (
        "".join(
            _row(
                stage["stage_id"],
                stage["stage_type"],
                stage.get("phase_id"),
                ", ".join(stage["parent_stage_ids"]) or "none",
                ", ".join(stage.get("external_prerequisite_stage_ids", ())) or "none",
                stage["state"],
                len(stage["attempts"]),
                stage.get("elapsed_seconds"),
            )
            for stage in result["stages"]
        )
        or '<tr><td colspan="8">No stages recorded.</td></tr>'
    )
    progress_rows = (
        "".join(
            _row(
                stage["stage_id"],
                stage["progress"]["status"],
                _compact(stage["progress"].get("scope") or {}),
                _measure_text(stage["progress"]),
                _eta_text(stage["progress"]),
            )
            for stage in result["stages"]
        )
        or '<tr><td colspan="5">No progress recorded.</td></tr>'
    )
    metric_rows = (
        "".join(
            _row(
                metric["name"],
                metric["subject_id"],
                metric["checkpoint_id"],
                metric["value"] if metric["value_state"] == "present" else metric["value_state"],
                metric["unit"],
                metric["aggregation"],
                metric["direction"],
                _compact(metric["workload"]),
                _compact(metric.get("dimensions") or {}),
            )
            for metric in result["metrics"]
        )
        or '<tr><td colspan="9">No metrics recorded.</td></tr>'
    )
    comparison_rows = (
        "".join(
            _row(
                item["left_metric_id"],
                item["right_metric_id"],
                item["delta"] if item["comparable"] else "not comparable",
                item["relative_delta"] if item["comparable"] else "not comparable",
                ", ".join(item["exclusion_reasons"]) or "none",
            )
            for item in result.get("comparisons", ())
        )
        or '<tr><td colspan="5">No teacher/candidate comparisons recorded.</td></tr>'
    )
    artifact_rows = (
        "".join(
            _row(
                artifact["role"],
                artifact["path"],
                artifact.get("availability"),
                artifact.get("validation"),
                (artifact.get("digest") or {}).get("value"),
            )
            for artifact in result["artifacts"]
        )
        or '<tr><td colspan="5">No artifacts recorded.</td></tr>'
    )
    limitations = (
        "".join(f"<li>{html.escape(str(item))}</li>" for item in result["limitations"])
        or "<li>No limitations recorded.</li>"
    )
    timing = "".join(
        f"<dt>{html.escape(str(name))}</dt><dd>{html.escape(_value(value))}</dd>"
        for name, value in run["timing"].items()
    )
    provenance = result["provenance"]
    embedded = json.dumps(result, sort_keys=True).replace("<", "\\u003c")
    run_id = html.escape(str(run["identity"]["run_id"]))
    document = (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        f"<title>Puzzletron run {run_id}</title>"
        "<style>body{font-family:system-ui,sans-serif;max-width:72rem;margin:2rem auto;padding:0 1rem}"
        "table{border-collapse:collapse;width:100%;margin-bottom:1.5rem}"
        "th,td{border:1px solid #bbb;padding:.4rem;text-align:left;vertical-align:top}"
        "dt{font-weight:600;float:left;clear:left;margin-right:.5rem}"
        ".stale{background:#fff2cc;padding:.6rem}</style></head><body>"
        f"<h1>Puzzletron run <code>{run_id}</code></h1>"
        + (
            f'<p class="stale">State is stale as of {html.escape(_value(freshness.get("as_of")))}: '
            f"{html.escape(_value(freshness.get('reason')))}</p>"
            if freshness["state"] == "stale"
            else ""
        )
        + f"<p>Execution: {html.escape(status['execution'])}; controller: {html.escape(status['attachment'])}; "
        f"evidence: {html.escape(status['evidence'])}; support: {html.escape(status['support'])}</p>"
        f'<h2>Timing and freshness</h2><dl>{timing}</dl><div style="clear:both"></div>'
        f"<p>Freshness: {html.escape(freshness['state'])}; as of {html.escape(_value(freshness.get('as_of')))}</p>"
        "<h2>Subjects and heterogeneous configurations</h2><table><thead><tr><th>Role</th><th>Subject</th>"
        f"<th>Checkpoint</th><th>Architecture and axes</th></tr></thead><tbody>{subject_rows}</tbody></table>"
        "<h2>DAG stages and phases</h2><table><thead><tr><th>Stage</th><th>Type</th><th>Phase</th>"
        "<th>Parents</th><th>External prerequisites</th><th>State</th><th>Attempts</th>"
        f"<th>Elapsed seconds</th></tr></thead><tbody>{stage_rows}</tbody></table>"
        "<h2>Operational progress</h2><table><thead><tr><th>Stage</th><th>Status</th><th>Scope</th>"
        f"<th>Completed and total work</th><th>Qualified ETA</th></tr></thead><tbody>{progress_rows}</tbody></table>"
        "<h2>Metrics and token contracts</h2><table><thead><tr><th>Metric</th><th>Subject</th><th>Checkpoint</th>"
        "<th>Value</th><th>Unit</th><th>Aggregation</th><th>Direction</th><th>Workload</th><th>Dimensions</th>"
        f"</tr></thead><tbody>{metric_rows}</tbody></table>"
        "<h2>Teacher and candidate comparisons</h2><table><thead><tr><th>Teacher metric</th><th>Candidate metric</th>"
        f"<th>Delta</th><th>Relative delta</th><th>Exclusions</th></tr></thead><tbody>{comparison_rows}</tbody></table>"
        "<h2>Artifact drill-down</h2><table><thead><tr><th>Role</th><th>Path</th><th>Availability</th>"
        f"<th>Validation</th><th>SHA-256</th></tr></thead><tbody>{artifact_rows}</tbody></table>"
        f"<h2>Provenance</h2><pre>{html.escape(json.dumps(provenance, indent=2, sort_keys=True))}</pre>"
        f"<h2>Limitations</h2><ul>{limitations}</ul>"
        "<p>This optional view is generated from result.json and contains no unique evidence.</p>"
        f'<script type="application/json" id="puzzletron-run-result">{embedded}</script>'
        "</body></html>\n"
    ).encode()
    source = canonical_json_bytes(
        {
            key: value
            for key, value in result.items()
            if key
            not in {
                "result_path",
                "result_digest",
                "stage_counts",
                "active_progress",
                "comparisons",
            }
        }
    )
    manifest = {
        "schema": "modelopt.puzzletron.derived-view/v1",
        "view_type": "html",
        "source_run_id": run["identity"]["run_id"],
        "source_result_path": "results/result.json",
        "source_result_digest": result_sha256(source),
        "source_producer_revision": result["provenance"]["resolved_bundle"]["producer_revision"],
        "source_validation": "passed",
        "renderer_revision": renderer_revision,
        "generated_at": generated_at,
        "output_digest": result_sha256(document),
    }
    return RenderedView(content=document, manifest=manifest)
