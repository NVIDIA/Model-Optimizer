# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict, scenario-aware final campaign report generation."""

from __future__ import annotations

import html
import json
import math
import re
from pathlib import Path
from typing import Any

_SCENARIO_RE = re.compile(r"(width-\d+)/(depth-\d+)")
_GRADIENT_GROUPS = ("vision", "projector", "language", "mtp")


def _load(path: Path) -> Any:
    if not path.is_file():
        raise RuntimeError(f"missing required campaign artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _scenario(value: str) -> str:
    match = _SCENARIO_RE.search(value)
    if match is None:
        raise RuntimeError(f"could not infer width/depth scenario from {value}")
    return f"{match.group(1)}/{match.group(2)}"


def _records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"missing global-KD training history: {path}")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    losses = [float(record["loss"]) for record in records]
    if len(losses) < 2 or not all(math.isfinite(loss) for loss in losses):
        raise RuntimeError(f"global-KD history must contain at least two finite losses: {path}")
    return records


def _training_path(checkpoint: str) -> Path:
    path = Path(checkpoint)
    for parent in path.parents:
        if parent.name == "checkpoints":
            return parent / "training.jsonl"
    raise RuntimeError(f"checkpoint is not under a checkpoints directory: {checkpoint}")


def _loss_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    losses = [float(record["loss"]) for record in records]
    gradients = {
        group: [float(record.get(f"gradient_norm_{group}", 0.0)) for record in records]
        for group in _GRADIENT_GROUPS
    }
    missing = [group for group, values in gradients.items() if max(values, default=0.0) <= 0.0]
    if missing:
        raise RuntimeError(f"global-KD has no positive gradients for groups: {missing}")
    return {
        "steps": len(losses),
        "initial": losses[0],
        "final": losses[-1],
        "best": min(losses),
        "improved": losses[-1] < losses[0],
        "values": losses,
        "gradient_norm_max": {group: max(values) for group, values in gradients.items()},
    }


def _metric_rows(payload: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {_scenario(str(row["checkpoint"])): row for row in payload}


def _aiperf_checkpoint_name(item: dict[str, Any]) -> str:
    explicit = item.get("checkpoint_name") or item.get("name")
    if explicit:
        return str(explicit)
    checkpoint = str(item.get("checkpoint_dir", ""))
    if match := _SCENARIO_RE.search(checkpoint):
        return f"{match.group(1)}__{match.group(2)}"
    return Path(checkpoint).name


def _html_pages(root: Path) -> dict[str, str]:
    pages = list(root.rglob("*.html"))
    result: dict[str, str] = {}
    for key, needles in {
        "vllm": ("vllm",),
        "replace_one_block": ("replace", "scoring"),
        "aiperf": ("aiperf",),
    }.items():
        matches = [path for path in pages if any(needle in str(path).lower() for needle in needles)]
        if matches:
            result[key] = str(sorted(matches)[-1])
    return result


def _fmt(value: Any) -> str:
    if value is None:
        return "missing"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _kd_svg(rows: list[dict[str, Any]]) -> str:
    width, height = 760, 300
    series = [row["kd_loss"]["values"] for row in rows]
    ymax = max(max(values) for values in series)
    ymin = min(min(values) for values in series)
    span = max(ymax - ymin, 1e-9)
    colors = ("#2563eb", "#dc2626", "#059669", "#7c3aed", "#ea580c", "#0891b2")
    paths = []
    legend = []
    for index, (row, values) in enumerate(zip(rows, series)):
        color = colors[index % len(colors)]
        denom = max(len(values) - 1, 1)
        points = " ".join(
            f"{45 + 670 * step / denom:.1f},{20 + 225 * (1 - (value - ymin) / span):.1f}"
            for step, value in enumerate(values)
        )
        paths.append(f"<polyline points='{points}' fill='none' stroke='{color}' stroke-width='2'/>")
        legend.append(
            f"<text x='{50 + (index % 3) * 235}' y='{267 + (index // 3) * 16}' fill='{color}'>"
            f"{html.escape(row['scenario'])}</text>"
        )
    return (
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='Global KD loss trends'>"
        "<line x1='45' y1='245' x2='720' y2='245'/><line x1='45' y1='20' x2='45' y2='245'/>"
        + "".join(paths + legend)
        + "</svg>"
    )


def generate_campaign_report(
    puzzle_dir: str | Path,
    *,
    model_name: str = "Puzzletron model",
    expected_kd_scenarios: int,
) -> dict[str, Any]:
    """Generate a strict JSON/HTML report for one completed model campaign."""

    root = Path(puzzle_dir).resolve()
    post_path = root / "artifacts/post_kd_evaluation/evaluation_summary.json"
    post_payload = _load(post_path)
    if len(post_payload) != expected_kd_scenarios:
        raise RuntimeError(
            f"expected {expected_kd_scenarios} post-KD scenarios, found {len(post_payload)}"
        )
    post_by_scenario = _metric_rows(post_payload)
    if len(post_by_scenario) != expected_kd_scenarios:
        raise RuntimeError("post-KD evaluation contains duplicate width/depth scenarios")
    if expected_kd_scenarios >= 5 and not any(name.startswith("width-1024/") for name in post_by_scenario):
        raise RuntimeError("requested width-1024 post-KD reference is missing")

    pre_by_scenario = _metric_rows(_load(root / "artifacts/exact_evaluation/evaluation_summary.json"))
    stage_files = {
        "convert": root / "manifests/convert.json",
        "activation": root / "manifests/activation.json",
        "sort": root / "manifests/sort.json",
        "sort_equivalence": root / "artifacts/sort_equivalence/sort_equivalence_summary.json",
        "activation_diagnostic": root / "artifacts/activation_diagnostic/activation_diagnostic_summary.json",
        "bypass": root / "manifests/bypass.json",
        "bypass_loss": root / "artifacts/bypass/local_kd_loss_history.json",
        "bypass_coverage": root / "artifacts/bypass/nested_axis_coverage.json",
        "build_library": root / "manifests/build_library.json",
        "candidate_library": root / "candidate_library.json",
        "replacement_library": root / "replacement_library.json",
        "vllm_stats": root / "subblock_stats.json",
        "mip_grid": root / "scenarios/mip_grid.json",
        "pre_kd_evaluation": root / "artifacts/exact_evaluation/evaluation_summary.json",
        "post_kd_evaluation": post_path,
        "aiperf": root / "artifacts/aiperf/aiperf_results.json",
    }
    for path in stage_files.values():
        if not path.is_file():
            raise RuntimeError(f"missing required campaign artifact: {path}")
    sort_equivalence = _load(stage_files["sort_equivalence"])
    if sort_equivalence.get("passed") is not True:
        raise RuntimeError("full-width sort equivalence did not pass")

    rows = []
    for scenario, post in sorted(post_by_scenario.items()):
        records = _records(_training_path(str(post["checkpoint"])))
        if int((post.get("observability") or {}).get("vision_forward_count", 0)) <= 0:
            raise RuntimeError(f"post-KD evaluation did not observe ViT forwards: {scenario}")
        pre = pre_by_scenario.get(scenario)
        if pre is None:
            raise RuntimeError(f"missing matching pre-KD evaluation: {scenario}")
        rows.append(
            {
                "scenario": scenario,
                "hidden_width": int(post["hidden_width"]),
                "checkpoint": post["checkpoint"],
                "pre_kd_metrics": pre["metrics"],
                "post_kd_metrics": post["metrics"],
                "metric_delta": {
                    key: float(post["metrics"][key]) - float(pre["metrics"][key])
                    for key in post["metrics"].keys() & pre["metrics"].keys()
                },
                "vision_forward_count": int(post["observability"]["vision_forward_count"]),
                "kd_loss": _loss_summary(records),
                "training_history": str(_training_path(str(post["checkpoint"]))),
            }
        )
    if not all(row["kd_loss"]["improved"] for row in rows):
        raise RuntimeError("one or more selected global-KD runs did not reduce training loss")

    aiperf = _load(stage_files["aiperf"])
    expected_names = {"teacher"} | {row["scenario"].replace("/", "__") for row in rows}
    observed_names = {_aiperf_checkpoint_name(item) for item in aiperf}
    missing_names = sorted(expected_names - observed_names)
    if missing_names:
        raise RuntimeError(f"AIPerf is missing checkpoints: {missing_names}")

    output = root / "artifacts/final_report"
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "campaign_report.json"
    html_path = output / "campaign_report.html"
    report = {
        "status": "complete",
        "model": model_name,
        "root": str(root),
        "stage_coverage": {name: str(path) for name, path in stage_files.items()},
        "sort_equivalence": sort_equivalence,
        "post_kd": rows,
        "aiperf": aiperf,
        "detailed_reports": _html_pages(root),
        "reports": {"json": str(json_path), "html": str(html_path)},
    }
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    columns = ("scenario", "hidden_width", "pre_lm_loss", "post_lm_loss", "lm_delta", "kd_initial", "kd_final")
    table_rows = []
    for row in rows:
        values = {
            "scenario": row["scenario"],
            "hidden_width": row["hidden_width"],
            "pre_lm_loss": row["pre_kd_metrics"].get("lm_loss"),
            "post_lm_loss": row["post_kd_metrics"].get("lm_loss"),
            "lm_delta": row["metric_delta"].get("lm_loss"),
            "kd_initial": row["kd_loss"]["initial"],
            "kd_final": row["kd_loss"]["final"],
        }
        table_rows.append("<tr>" + "".join(f"<td>{html.escape(_fmt(values[key]))}</td>" for key in columns) + "</tr>")
    stage_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>complete</td><td>{html.escape(str(path))}</td></tr>"
        for name, path in stage_files.items()
    )
    links = "".join(
        f"<li>{html.escape(name)}: {html.escape(path)}</li>" for name, path in report["detailed_reports"].items()
    )
    embedded = json.dumps(report).replace("</", "<\\/")
    html_path.write_text(
        "<!doctype html><meta charset='utf-8'><title>Puzzletron campaign report</title>"
        "<style>body{font-family:system-ui,sans-serif;margin:2rem;color:#172033}table{border-collapse:collapse;width:100%;margin:1rem 0 2rem}"
        "th,td{border:1px solid #cbd5e1;padding:.45rem;text-align:right}th:first-child,td:first-child{text-align:left}"
        "svg{max-width:950px;border:1px solid #cbd5e1}svg line{stroke:#64748b}svg text{font-size:11px}</style>"
        f"<h1>{html.escape(model_name)} campaign report</h1><p>Status: complete · selected KD scenarios: {len(rows)}</p>"
        f"<h2>Detailed reports</h2><ul>{links}</ul>"
        f"<h2>Global-KD loss trends</h2>{_kd_svg(rows)}"
        f"<h2>Pre/post-KD exact evaluation</h2><table><tr>{''.join(f'<th>{html.escape(key)}</th>' for key in columns)}</tr>{''.join(table_rows)}</table>"
        f"<h2>Stage coverage</h2><table><tr><th>stage</th><th>status</th><th>artifact</th></tr>{stage_rows}</table>"
        f"<script id='campaign-data' type='application/json'>{embedded}</script>",
        encoding="utf-8",
    )
    return report


__all__ = ["generate_campaign_report"]
