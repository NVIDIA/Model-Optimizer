#!/usr/bin/env python3
"""Render the self-contained Qwen3.6 FP8 granularity study report."""

from __future__ import annotations

import argparse
import html
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "study_manifest.json"
DEFAULT_RESULTS = ROOT / "results"
DEFAULT_OUTPUT = ROOT / "report.html"
DEFAULT_TOY = ROOT / "theory" / "toy_scale_sweep.json"
STUDY_RESULT_SCHEMA = "qwen36-fp8-granularity-study-v1"


METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "logit_mse": ("logit_mse", "mse", "raw_logit_mse"),
    "normalized_mse": (
        "centered_variance_normalized_mse",
        "variance_normalized_logit_mse",
        "variance_normalized_mse",
        "normalized_logit_mse",
        "normalized_mse",
    ),
    "kl_forward": (
        "forward_kl_ref_to_quant",
        "kl_forward",
        "forward_kl",
        "kl_ref_to_quant",
        "kl_divergence",
    ),
    "kl_reverse": (
        "reverse_kl_quant_to_ref",
        "kl_reverse",
        "reverse_kl",
        "kl_quant_to_ref",
    ),
    "js": ("jensen_shannon", "js_divergence", "js"),
    "target_logprob_mse": (
        "target_logprob_squared_error",
        "target_logprob_mse",
        "target_log_prob_mse",
    ),
    "nll_delta": ("nll_delta_quant_minus_ref", "nll_delta", "mean_nll_delta"),
    "top1": ("top1_agreement", "top_1_agreement", "top1"),
    "top5": ("top5_set_overlap", "top5_agreement", "top_5_agreement", "top5"),
}


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            out[name] = child
            out.update(flatten(child, name))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            out.update(flatten(child, f"{prefix}.{index}"))
    return out


def pick(flat: dict[str, Any], aliases: tuple[str, ...]) -> Any:
    for alias in aliases:
        for key, value in flat.items():
            leaf = key.rsplit(".", 1)[-1].lower()
            if leaf == alias and isinstance(value, (int, float)) and not isinstance(value, bool):
                return value
    return None


def pick_text(flat: dict[str, Any], aliases: tuple[str, ...]) -> str | None:
    for alias in aliases:
        for key, value in flat.items():
            if key.rsplit(".", 1)[-1].lower() == alias and isinstance(value, str):
                return value
    return None


def finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def normalize_id(text: str) -> str:
    return "".join(char.lower() for char in text if char.isalnum())


def infer_candidate(path: Path, flat: dict[str, Any], candidates: list[dict[str, Any]]) -> str:
    explicit = pick_text(flat, ("candidate", "candidate_id", "recipe", "quant_format", "format"))
    explicit_normalized = normalize_id(explicit or "")
    for candidate in candidates:
        if explicit_normalized in {
            normalize_id(candidate["id"]),
            normalize_id(candidate["label"]),
        }:
            return candidate["id"]
    haystack = normalize_id(str(path))
    # Prefer the most specific path match so prefix IDs such as ``mxfp8`` do not
    # swallow ``mxfp8_weight_only_control``.
    for candidate in sorted(
        candidates,
        key=lambda item: max(len(normalize_id(item["id"])), len(normalize_id(item["label"]))),
        reverse=True,
    ):
        probes = (candidate["id"], candidate["label"])
        if any(normalize_id(probe) in haystack for probe in probes):
            return candidate["id"]
    return explicit or "unknown"


def infer_model(path: Path, flat: dict[str, Any], models: list[dict[str, Any]]) -> str:
    explicit = pick_text(flat, ("model", "model_name", "model_id", "model_path", "hf_model"))
    haystack = normalize_id(" ".join((str(path), explicit or "")))
    for model in models:
        probes = (model["handle"], model["short_name"])
        if any(normalize_id(probe) in haystack for probe in probes):
            return model["short_name"]
    return explicit or "unknown"


def load_results(
    results_dir: Path, manifest: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in sorted(results_dir.rglob("results.json")) if results_dir.exists() else []:
        try:
            display_path = path.relative_to(ROOT)
        except ValueError:
            display_path = path
        try:
            payload = load_json(path)
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"{display_path}: {error}")
            continue
        if not isinstance(payload, dict) or payload.get("schema_version") != STUDY_RESULT_SCHEMA:
            errors.append(f"{display_path}: unsupported or missing study result schema")
            continue
        flat = flatten(payload)
        record: dict[str, Any] = {
            "path": display_path,
            "model": infer_model(path, flat, manifest["models"]),
            "candidate": infer_candidate(path, flat, manifest["candidates"]),
            "status": pick_text(flat, ("status", "state")) or "result artifact found",
            "reference_hash": mapping(payload.get("reference")).get("signature_hash"),
            "comparable": False,
            "valid_complete": False,
            "quantization_mse": mapping(payload.get("quantization_mse")),
            "quantization": mapping(payload.get("quantization")),
            "phase_walltime_seconds": mapping(payload.get("phase_walltime_seconds")),
            "total_walltime_seconds": payload.get("total_walltime_seconds"),
            "weight_quantizer_names": [],
            "input_quantizer_names": [],
            "normalized_mse_document_bootstrap": {},
        }
        if record["candidate"] not in {item["id"] for item in manifest["candidates"]}:
            errors.append(f"{display_path}: unknown candidate {record['candidate']!r}")
            continue
        if record["model"] not in {item["short_name"] for item in manifest["models"]}:
            errors.append(f"{display_path}: unknown model {record['model']!r}")
            continue
        for metric, aliases in METRIC_ALIASES.items():
            record[metric] = pick(flat, aliases)
        if status_class(record["status"]) == "ok":
            output_similarity = mapping(payload.get("output_similarity"))
            orientation = mapping(output_similarity.get("orientation"))
            aggregates = mapping(output_similarity.get("aggregate_per_token"))
            token_count = output_similarity.get("token_count")
            quantization = mapping(payload.get("quantization"))
            coverage_contract = mapping(quantization.get("coverage_contract"))
            quantization_mse = mapping(payload.get("quantization_mse"))
            weight_coverage = mapping(mapping(quantization_mse.get("weight")).get("coverage"))
            input_coverage = mapping(mapping(quantization_mse.get("input")).get("coverage"))
            weight_cost = mapping(quantization.get("weight_cost_estimate"))
            bootstrap = mapping(output_similarity.get("paired_document_bootstrap"))
            normalized_bootstrap = mapping(
                mapping(bootstrap.get("metrics")).get("variance_normalized_logit_mse")
            )
            missing = []
            if not isinstance(record["reference_hash"], str) or not record["reference_hash"]:
                missing.append("reference signature")
            if not finite_number(token_count) or float(token_count) <= 0:
                missing.append("positive token count")
            required_aggregates = {
                "logit_mse",
                "variance_normalized_logit_mse",
                "forward_kl_ref_to_quant",
                "reverse_kl_quant_to_ref",
                "jensen_shannon",
                "target_logprob_squared_error",
                "nll_delta_quant_minus_ref",
                "top1_agreement",
            }
            invalid_aggregates = sorted(
                name for name in required_aggregates if not finite_number(aggregates.get(name))
            )
            if invalid_aggregates:
                missing.append("finite aggregate metrics: " + ", ".join(invalid_aggregates))
            if orientation.get("forward_kl") != "KL(reference || quantized)":
                missing.append("forward-KL orientation")
            if orientation.get("reverse_kl") != "KL(quantized || reference)":
                missing.append("reverse-KL orientation")
            bootstrap_interval = mapping(normalized_bootstrap.get("percentile_interval"))
            bootstrap_lower = bootstrap_interval.get("lower")
            bootstrap_upper = bootstrap_interval.get("upper")
            if not (
                finite_number(bootstrap_lower)
                and finite_number(bootstrap_upper)
                and float(bootstrap_lower) <= float(bootstrap_upper)
                and finite_number(normalized_bootstrap.get("document_count"))
                and int(normalized_bootstrap["document_count"]) > 0
                and finite_number(normalized_bootstrap.get("resamples"))
                and int(normalized_bootstrap["resamples"]) > 0
            ):
                missing.append("paired-document NMSE bootstrap interval")
            else:
                record["normalized_mse_document_bootstrap"] = normalized_bootstrap
            weight_names = coverage_contract.get("weight_quantizer_names")
            input_names = coverage_contract.get("input_quantizer_names")
            if coverage_contract.get("status") != "passed":
                missing.append("passed quantizer coverage contract")
            if (
                not isinstance(weight_names, list)
                or not weight_names
                or not all(isinstance(name, str) for name in weight_names)
            ):
                missing.append("non-empty weight-quantizer owner set")
            else:
                record["weight_quantizer_names"] = weight_names
            if not isinstance(input_names, list) or not all(
                isinstance(name, str) for name in input_names
            ):
                missing.append("input-quantizer owner set")
            else:
                record["input_quantizer_names"] = input_names
            weight_eligible = weight_coverage.get("eligible_count")
            weight_executed = weight_coverage.get("executed_count")
            if not (
                finite_number(weight_eligible)
                and finite_number(weight_executed)
                and int(weight_eligible) > 0
                and int(weight_executed) == int(weight_eligible)
            ):
                missing.append("complete nonzero weight-MSE coverage")
            candidate_scope = next(
                item["scope"]
                for item in manifest["candidates"]
                if item["id"] == record["candidate"]
            )
            input_eligible = input_coverage.get("eligible_count")
            input_executed = input_coverage.get("executed_count")
            if candidate_scope == "W8A8":
                if not (
                    finite_number(input_eligible)
                    and finite_number(input_executed)
                    and int(input_eligible) > 0
                    and int(input_executed) == int(input_eligible)
                ):
                    missing.append("complete nonzero W8A8 input-MSE coverage")
            elif not (
                finite_number(input_eligible)
                and finite_number(input_executed)
                and int(input_eligible) == 0
                and int(input_executed) == 0
            ):
                missing.append("zero W8A16 input-quantizer coverage")
            if weight_cost.get("unmapped_weight_quantizers") != []:
                missing.append("zero unmapped weight quantizers")
            if missing:
                record["status"] = "rejected invalid complete artifact"
                errors.append(f"{display_path}: missing/invalid {', '.join(missing)}")
            else:
                record["valid_complete"] = True
                record["comparable"] = True
        records.append(record)

    # More than one nominally complete artifact for a model/candidate makes provenance
    # selection ambiguous. Keep both visible, but do not silently pick one for rankings.
    successful_pairs: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        if record["valid_complete"]:
            successful_pairs.setdefault((record["model"], record["candidate"]), []).append(record)
    for (model, candidate), duplicates in successful_pairs.items():
        if len(duplicates) <= 1:
            continue
        errors.append(
            f"{model} / {candidate}: {len(duplicates)} complete artifacts found; "
            "rankings require exactly one"
        )
        for record in duplicates:
            record["comparable"] = False
            record["status"] = "incomparable duplicate artifact"

    # A per-model ranking is valid only when every successful candidate used the
    # same exact BF16 reference signature (which covers revision, tokenizer, held-out
    # samples, fixed shapes, dtype, and runtime provenance).
    for model in {record["model"] for record in records}:
        successful = [
            record for record in records if record["model"] == model and record["comparable"]
        ]
        reference_hashes = {record["reference_hash"] for record in successful}
        if len(reference_hashes) > 1:
            errors.append(
                f"{model}: successful artifacts have mismatched BF16 reference signatures; "
                "charts are suppressed for this model"
            )
            for record in successful:
                record["comparable"] = False
                record["status"] = "incomparable reference signature"

    # A pattern miss must not masquerade as lower distortion. Require identical
    # enabled weight sets across every candidate, and identical input sets across
    # the three W8A8 format-policy bundles.
    candidate_scopes = {item["id"]: item["scope"] for item in manifest["candidates"]}
    for model in {record["model"] for record in records}:
        successful = [
            record for record in records if record["model"] == model and record["comparable"]
        ]
        weight_sets = {tuple(record["weight_quantizer_names"]) for record in successful}
        if len(weight_sets) > 1:
            errors.append(
                f"{model}: enabled weight-quantizer sets differ across candidates; "
                "rankings are suppressed"
            )
            for record in successful:
                record["comparable"] = False
                record["status"] = "incomparable weight coverage"
            continue
        w8a8 = [
            record for record in successful if candidate_scopes.get(record["candidate"]) == "W8A8"
        ]
        input_sets = {tuple(record["input_quantizer_names"]) for record in w8a8}
        if len(input_sets) > 1:
            errors.append(
                f"{model}: enabled W8A8 input-quantizer sets differ across candidates; "
                "rankings are suppressed"
            )
            for record in successful:
                record["comparable"] = False
                record["status"] = "incomparable input coverage"
    return records, errors


def format_number(value: Any, percent: bool = False) -> str:
    if value is None:
        return "—"
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return esc(value)
    scaled = float(value) * 100 if percent else float(value)
    if percent:
        return f"{scaled:.2f}%"
    magnitude = abs(scaled)
    if magnitude == 0:
        return "0"
    if magnitude < 1e-3 or magnitude >= 1e4:
        return f"{scaled:.3e}"
    return f"{scaled:.5g}"


def status_class(status: str) -> str:
    lowered = status.lower()
    if any(
        word in lowered for word in ("fail", "error", "cancel", "invalid", "reject", "incomparable")
    ):
        return "bad"
    if any(word in lowered for word in ("complete", "success", "passed")):
        return "ok"
    return "pending"


def results_table(records: list[dict[str, Any]], candidate_labels: dict[str, str]) -> str:
    if not records:
        return (
            '<div class="empty"><strong>No Qwen3.6 measurement artifact is present.</strong>'
            " The report is intentionally pending. Copy each remote <code>results.json</code>"
            " beneath <code>results/</code> and rerun the renderer.</div>"
        )
    rows = []
    for record in records:
        label = candidate_labels.get(record["candidate"], record["candidate"])
        show_metrics = bool(record.get("valid_complete"))
        bootstrap = mapping(record.get("normalized_mse_document_bootstrap"))
        interval = mapping(bootstrap.get("percentile_interval"))
        bootstrap_display = (
            f"{format_number(bootstrap.get('point_estimate_equal_document_mean'))} "
            f"[{format_number(interval.get('lower'))}, {format_number(interval.get('upper'))}]"
            if show_metrics and bootstrap
            else "—"
        )
        rows.append(
            "<tr>"
            f"<td>{esc(record['model'])}</td><td>{esc(label)}</td>"
            f'<td><span class="tag {status_class(record["status"])}">{esc(record["status"])}</span></td>'
            f"<td>{format_number(record['logit_mse'] if show_metrics else None)}</td>"
            f"<td>{format_number(record['normalized_mse'] if show_metrics else None)}</td>"
            f"<td>{bootstrap_display}</td>"
            f"<td>{format_number(record['kl_forward'] if show_metrics else None)}</td>"
            f"<td>{format_number(record['kl_reverse'] if show_metrics else None)}</td>"
            f"<td>{format_number(record['js'] if show_metrics else None)}</td>"
            f"<td>{format_number(record['target_logprob_mse'] if show_metrics else None)}</td>"
            f"<td>{format_number(record['nll_delta'] if show_metrics else None)}</td>"
            f"<td>{format_number(record['top1'] if show_metrics else None, percent=True)}</td>"
            f"<td><code>{esc(record['path'])}</code></td>"
            "</tr>"
        )
    return (
        '<div class="table-wrap"><table><thead><tr><th>Model</th><th>Candidate</th>'
        "<th>Status</th><th>Logit MSE</th><th>Token NMSE</th>"
        "<th>Equal-doc NMSE [95% bootstrap]</th><th>KL ref→quant</th>"
        "<th>KL quant→ref</th><th>JS</th><th>Target log-p MSE</th><th>ΔNLL</th>"
        "<th>Top-1</th><th>Artifact</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def metric_chart(
    records: list[dict[str, Any]], metric: str, candidate_labels: dict[str, str]
) -> str:
    data = [
        record
        for record in records
        if record.get("comparable") and isinstance(record.get(metric), (int, float))
    ]
    if not data:
        return (
            '<div class="chart-empty">Chart appears when successful metric artifacts arrive.</div>'
        )
    width, row_height, left, right = 900, 42, 255, 80
    height = 52 + row_height * len(data)
    values = [max(float(record[metric]), 1e-16) for record in data]
    lo, hi = min(math.log10(value) for value in values), max(math.log10(value) for value in values)
    if math.isclose(lo, hi):
        lo -= 0.5
        hi += 0.5
    plot_width = width - left - right
    parts = [
        f'<svg role="img" aria-label="{esc(metric)} comparison" viewBox="0 0 {width} {height}">',
        "<style>.axis{stroke:#9ca3af;stroke-width:1}.bar{fill:#76b900}.bar2{fill:#4f46e5}"
        ".lbl{font:13px system-ui;fill:#dbe4ee}.val{font:12px ui-monospace;fill:#dbe4ee}</style>",
        f'<line class="axis" x1="{left}" y1="24" x2="{left}" y2="{height - 18}"/>',
    ]
    for index, record in enumerate(data):
        value = max(float(record[metric]), 1e-16)
        fraction = (math.log10(value) - lo) / (hi - lo)
        bar_width = max(3.0, fraction * plot_width)
        y = 34 + index * row_height
        label = (
            f"{record['model']} · {candidate_labels.get(record['candidate'], record['candidate'])}"
        )
        color_class = "bar" if "35B" in record["model"] else "bar2"
        parts.extend(
            (
                f'<text class="lbl" x="0" y="{y + 15}">{esc(label)}</text>',
                f'<rect class="{color_class}" x="{left}" y="{y}" width="{bar_width:.1f}" height="22" rx="4"/>',
                f'<text class="val" x="{left + bar_width + 8:.1f}" y="{y + 15}">{format_number(value)}</text>',
            )
        )
    parts.append("</svg>")
    return "".join(parts)


def format_duration(value: Any) -> str:
    if not finite_number(value) or float(value) < 0:
        return "—"
    seconds = float(value)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {remainder:.0f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m"


def coverage_cell(section: dict[str, Any]) -> str:
    coverage = mapping(section.get("coverage"))
    eligible = coverage.get("eligible_count")
    executed = coverage.get("executed_count")
    if not finite_number(eligible) or not finite_number(executed):
        return "—"
    eligible_int, executed_int = int(eligible), int(executed)
    percent = 100 * executed_int / eligible_int if eligible_int else 0.0
    missing = coverage.get("missing_quantizers")
    missing_count = (
        len(missing) if isinstance(missing, list) else max(0, eligible_int - executed_int)
    )
    return f"{executed_int}/{eligible_int} ({percent:.1f}%); {missing_count} not executed"


def coverage_and_cost_table(records: list[dict[str, Any]], candidate_labels: dict[str, str]) -> str:
    rows = []
    for record in records:
        if not record.get("valid_complete"):
            continue
        mse = mapping(record.get("quantization_mse"))
        quantization = mapping(record.get("quantization"))
        cost = mapping(quantization.get("weight_cost_estimate"))
        totals = mapping(cost.get("logical_totals"))
        total_bits = totals.get("total_bits")
        scale_bits = totals.get("scale_overhead_bits")
        logical_gib = float(total_bits) / 8 / 2**30 if finite_number(total_bits) else None
        overhead_percent = (
            100 * float(scale_bits) / float(total_bits)
            if finite_number(scale_bits) and finite_number(total_bits) and float(total_bits) > 0
            else None
        )
        label = candidate_labels.get(record["candidate"], record["candidate"])
        rows.append(
            "<tr>"
            f"<td>{esc(record['model'])}</td><td>{esc(label)}</td>"
            f"<td>{esc(coverage_cell(mapping(mse.get('weight'))))}</td>"
            f"<td>{esc(coverage_cell(mapping(mse.get('input'))))}</td>"
            f"<td>{format_number(totals.get('element_count'))}</td>"
            f"<td>{format_number(totals.get('effective_bits_per_weight'))}</td>"
            f"<td>{format_number(overhead_percent, percent=False) + '%' if overhead_percent is not None else '—'}</td>"
            f"<td>{format_number(logical_gib)} GiB</td>"
            f"<td>{format_number(cost.get('unique_parameter_slice_count', cost.get('unique_parameter_tensor_count')))}</td>"
            "</tr>"
        )
    if not rows:
        return '<div class="chart-empty">Coverage and logical-cost summaries appear with successful artifacts.</div>'
    return (
        '<div class="table-wrap"><table><thead><tr><th>Model</th><th>Candidate</th>'
        "<th>Weight MSE coverage</th><th>Input MSE coverage</th><th>Logical weights</th>"
        "<th>Effective bits / weight</th><th>Scale share of bits</th><th>Logical payload</th>"
        "<th>Unique parameter slices</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def quantizer_mse_details(records: list[dict[str, Any]], candidate_labels: dict[str, str]) -> str:
    sections = []
    for record in records:
        if not record.get("valid_complete"):
            continue
        family_rows = []
        layer_rows = []
        inventory = mapping(record.get("quantization")).get("quantizer_inventory")
        families_by_name = (
            {
                item.get("name"): item.get("family", "—")
                for item in inventory
                if isinstance(inventory, list)
                and isinstance(item, dict)
                and isinstance(item.get("name"), str)
            }
            if isinstance(inventory, list)
            else {}
        )
        for role in ("weight", "input"):
            section = mapping(mapping(record.get("quantization_mse")).get(role))
            for family, summary_value in sorted(mapping(section.get("families")).items()):
                summary = mapping(summary_value)
                quantiles = mapping(summary.get("quantiles"))
                family_rows.append(
                    "<tr>"
                    f"<td>{esc(role)}</td><td>{esc(family)}</td>"
                    f"<td>{format_number(summary.get('count'))}</td>"
                    f"<td>{format_number(summary.get('mean'))}</td>"
                    f"<td>{format_number(quantiles.get('p95'))}</td>"
                    f"<td>{format_number(summary.get('max'))}</td>"
                    "</tr>"
                )
            named = mapping(section.get("by_quantizer"))
            ordered = sorted(
                (
                    (str(name), float(value))
                    for name, value in named.items()
                    if finite_number(value)
                ),
                key=lambda item: item[1],
                reverse=True,
            )
            for name, value in ordered[:12]:
                layer_rows.append(
                    "<tr>"
                    f"<td>{esc(role)}</td><td>{esc(families_by_name.get(name, '—'))}</td>"
                    f"<td><code>{esc(name)}</code></td><td>{format_number(value)}</td>"
                    "</tr>"
                )
        family_table = (
            '<div class="table-wrap"><table><thead><tr><th>Role</th><th>Family</th>'
            "<th>Executed quantizers</th><th>Mean call-MSE</th><th>P95</th><th>Max</th>"
            "</tr></thead><tbody>" + "".join(family_rows) + "</tbody></table></div>"
            if family_rows
            else '<div class="chart-empty">No family MSE summary was recorded.</div>'
        )
        layer_table = (
            '<div class="table-wrap"><table><thead><tr><th>Role</th><th>Family</th>'
            "<th>Quantizer</th><th>Mean call-MSE</th></tr></thead><tbody>"
            + "".join(layer_rows)
            + "</tbody></table></div>"
            if layer_rows
            else '<div class="chart-empty">No named quantizer MSE was recorded.</div>'
        )
        label = candidate_labels.get(record["candidate"], record["candidate"])
        sections.append(
            '<details class="diagnostic"><summary>'
            f"{esc(record['model'])} · {esc(label)}</summary>"
            '<p class="small-note">Weight values are direct one-pass per-slice MSE; input values '
            "average quantizer hook invocations. Family summaries weight quantizers equally, not "
            "by tensor elements or token count.</p>"
            f"{family_table}<h4>Highest named quantizer MSE (up to 12 per role)</h4>{layer_table}"
            "</details>"
        )
    if not sections:
        return (
            '<div class="chart-empty">Quantizer diagnostics appear with successful artifacts.</div>'
        )
    return "".join(sections)


def walltime_table(records: list[dict[str, Any]], candidate_labels: dict[str, str]) -> str:
    phases = (
        "initialization",
        "dataset_materialization",
        "reference_logits",
        "quantization",
        "quantizer_mse",
        "output_similarity",
    )
    rows = []
    for record in records:
        if not record.get("valid_complete"):
            continue
        phase_values = mapping(record.get("phase_walltime_seconds"))
        label = candidate_labels.get(record["candidate"], record["candidate"])
        cells = "".join(f"<td>{format_duration(phase_values.get(phase))}</td>" for phase in phases)
        rows.append(
            "<tr>"
            f"<td>{esc(record['model'])}</td><td>{esc(label)}</td>"
            f"<td>{format_duration(record.get('total_walltime_seconds'))}</td>{cells}</tr>"
        )
    if not rows:
        return (
            '<div class="chart-empty">Wall-time breakdowns appear with successful artifacts.</div>'
        )
    headers = "".join(f"<th>{esc(phase.replace('_', ' '))}</th>" for phase in phases)
    return (
        '<div class="table-wrap"><table><thead><tr><th>Model</th><th>Candidate</th>'
        f"<th>Total</th>{headers}</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def ranking_tables(records: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> str:
    candidate_meta = {candidate["id"]: candidate for candidate in candidates}
    metric_specs = (
        ("normalized_mse", "Tokenwise mean NMSE", False),
        ("kl_forward", "Forward KL", False),
        ("js", "Jensen–Shannon", False),
        ("top1", "Top-1", True),
    )
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        if not record.get("comparable"):
            continue
        scope = candidate_meta.get(record["candidate"], {}).get("scope", "unknown scope")
        groups.setdefault((record["model"], scope), []).append(record)
    sections = []
    for (model, scope), group in sorted(groups.items()):
        if len(group) < 2:
            continue
        ranks: dict[str, dict[str, int]] = {}
        for metric, _label, descending in metric_specs:
            ordered = sorted(
                (record for record in group if finite_number(record.get(metric))),
                key=lambda record: float(record[metric]),
                reverse=descending,
            )
            ranks[metric] = {record["candidate"]: index + 1 for index, record in enumerate(ordered)}
        rows = []
        for record in sorted(
            group, key=lambda item: ranks["normalized_mse"].get(item["candidate"], 999)
        ):
            cells = []
            for metric, _label, _descending in metric_specs:
                rank = ranks[metric].get(record["candidate"])
                value = record.get(metric)
                formatted = format_number(value, percent=metric == "top1")
                cells.append(f"<td>{'#' + str(rank) if rank else '—'} · {formatted}</td>")
            label = candidate_meta.get(record["candidate"], {}).get("label", record["candidate"])
            rows.append(f"<tr><td>{esc(label)}</td>{''.join(cells)}</tr>")
        headers = "".join(f"<th>{esc(label)}</th>" for _metric, label, _descending in metric_specs)
        sections.append(
            f"<h4>{esc(model)} · {esc(scope)} ({len(group)} comparable candidates)</h4>"
            '<div class="table-wrap"><table><thead><tr><th>Candidate</th>'
            f"{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
        )
    if not sections:
        return (
            '<div class="chart-empty">Rankings require at least two validated artifacts in the '
            "same model, scope, and BF16-reference cohort.</div>"
        )
    return "".join(sections)


def toy_table(toy: dict[str, Any] | None) -> str:
    if not toy:
        return '<div class="chart-empty">Run <code>scripts/toy_scale_sweep.py</code> to generate the synthetic check.</div>'
    labels = {
        "per_tensor_fp8": "Per-tensor E4M3",
        "block128_full_precision_scale": "Block-128 E4M3 / FP scale",
        "mxfp8_block32_e8m0_scale": "MXFP8 block-32 / E8M0 scale",
    }
    rows = []
    for sweep in toy.get("sweeps", []):
        multiplier = sweep.get("outlier_multiplier")
        for candidate, metrics in sweep.get("metrics", {}).items():
            rows.append(
                "<tr>"
                f"<td>{format_number(multiplier)}</td><td>{esc(labels.get(candidate, candidate))}</td>"
                f"<td>{format_number(metrics.get('mse'))}</td>"
                f"<td>{format_number(metrics.get('normalized_mse'))}</td>"
                f"<td>{format_number(metrics.get('kl_forward'))}</td>"
                f"<td>{format_number(metrics.get('top1_agreement'), percent=True)}</td>"
                "</tr>"
            )
    return (
        '<div class="table-wrap"><table><thead><tr><th>Outlier / σ</th><th>Mapping</th>'
        "<th>MSE</th><th>Normalized MSE</th><th>KL ref→quant</th><th>Top-1</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def toy_chart(toy: dict[str, Any] | None) -> str:
    if not toy or not toy.get("sweeps"):
        return '<div class="chart-empty">Synthetic data is not present.</div>'
    series = {
        "per_tensor_fp8": ("Per-tensor", "#f3b33d"),
        "block128_full_precision_scale": ("Block-128 / FP scale", "#76b900"),
        "mxfp8_block32_e8m0_scale": ("MXFP8 block-32 / E8M0", "#818cf8"),
    }
    width, height, left, right, top, bottom = 900, 390, 74, 32, 36, 62
    raw_points: dict[str, list[tuple[float, float]]] = {key: [] for key in series}
    for sweep in toy["sweeps"]:
        x = float(sweep["outlier_multiplier"])
        for key in series:
            value = sweep.get("metrics", {}).get(key, {}).get("mse")
            if isinstance(value, (int, float)) and value > 0:
                raw_points[key].append((x, float(value)))
    all_points = [point for points in raw_points.values() for point in points]
    if not all_points:
        return '<div class="chart-empty">Synthetic MSE values are not present.</div>'
    x_logs = [math.log10(x) for x, _ in all_points]
    y_logs = [math.log10(y) for _, y in all_points]
    x_lo, x_hi = min(x_logs), max(x_logs)
    y_lo, y_hi = min(y_logs), max(y_logs)
    y_pad = max(0.1, (y_hi - y_lo) * 0.08)
    y_lo, y_hi = y_lo - y_pad, y_hi + y_pad
    plot_w, plot_h = width - left - right, height - top - bottom

    def xy(point: tuple[float, float]) -> tuple[float, float]:
        x, y = point
        px = left + (math.log10(x) - x_lo) / max(x_hi - x_lo, 1e-12) * plot_w
        py = top + (y_hi - math.log10(y)) / max(y_hi - y_lo, 1e-12) * plot_h
        return px, py

    parts = [
        f'<svg role="img" aria-label="Synthetic ModelOpt FP8 MSE outlier sweep" viewBox="0 0 {width} {height}">',
        "<style>.grid{stroke:#273648;stroke-width:1}.axistext{font:12px system-ui;fill:#9eacbd}"
        ".legend{font:12px system-ui;fill:#dbe4ee}.dot{stroke:#081019;stroke-width:2}</style>",
    ]
    for step in range(5):
        fraction = step / 4
        y = top + fraction * plot_h
        value = 10 ** (y_hi - fraction * (y_hi - y_lo))
        parts.append(
            f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}"/>'
        )
        parts.append(
            f'<text class="axistext" x="{left - 8}" y="{y + 4:.1f}" text-anchor="end">{format_number(value)}</text>'
        )
    multipliers = [float(sweep["outlier_multiplier"]) for sweep in toy["sweeps"]]
    for multiplier in multipliers:
        x = left + (math.log10(multiplier) - x_lo) / max(x_hi - x_lo, 1e-12) * plot_w
        parts.append(
            f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}"/>'
        )
        parts.append(
            f'<text class="axistext" x="{x:.1f}" y="{height - bottom + 22}" text-anchor="middle">{format_number(multiplier)}</text>'
        )
    for index, (key, (label, color)) in enumerate(series.items()):
        points = [xy(point) for point in raw_points[key]]
        path = " ".join(
            ("M" if i == 0 else "L") + f" {x:.1f} {y:.1f}" for i, (x, y) in enumerate(points)
        )
        parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="3"/>')
        for x, y in points:
            parts.append(f'<circle class="dot" cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}"/>')
        legend_x = left + index * 235
        parts.append(f'<rect x="{legend_x}" y="8" width="18" height="4" rx="2" fill="{color}"/>')
        parts.append(f'<text class="legend" x="{legend_x + 25}" y="15">{esc(label)}</text>')
    parts.append(
        f'<text class="axistext" x="{left + plot_w / 2:.1f}" y="{height - 8}" text-anchor="middle">Outlier magnitude / σ (log scale)</text>'
    )
    parts.append(
        f'<text class="axistext" transform="translate(16 {top + plot_h / 2:.1f}) rotate(-90)" text-anchor="middle">Element MSE (log scale)</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def toy_readout(toy: dict[str, Any] | None) -> str:
    if not toy or not toy.get("sweeps"):
        return "Synthetic readout pending."
    final = toy["sweeps"][-1]
    metrics = final.get("metrics", {})
    per_tensor = metrics.get("per_tensor_fp8", {}).get("mse")
    block = metrics.get("block128_full_precision_scale", {}).get("mse")
    mx = metrics.get("mxfp8_block32_e8m0_scale", {}).get("mse")
    if not all(isinstance(value, (int, float)) and value > 0 for value in (per_tensor, block, mx)):
        return "Synthetic readout pending."
    return (
        f"At the deliberately extreme {format_number(final['outlier_multiplier'])}σ endpoint, "
        f"per-tensor MSE is {per_tensor / mx:.1f}× MXFP8 MSE and "
        f"{per_tensor / block:.1f}× block-128/full-precision-scale MSE. "
        "This demonstrates the isolation mechanism under stress; it does not estimate Qwen error."
    )


def render(
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
    errors: list[str],
    toy: dict[str, Any] | None,
) -> str:
    candidates = manifest["candidates"]
    candidate_labels = {item["id"]: item["label"] for item in candidates}
    completed = sum(bool(record.get("comparable")) for record in records)
    expected = len(manifest["models"]) * len(candidates)
    failed = sum(status_class(record["status"]) == "bad" for record in records)
    pending = sum(status_class(record["status"]) == "pending" for record in records)
    present_pairs = {(record["model"], record["candidate"]) for record in records}
    missing = max(0, expected - len(present_pairs))
    if completed == expected:
        status = "complete"
        status_note = f"All {expected} expected, comparable candidate artifacts were parsed."
        executive_title = (
            "Full screening matrix complete; measured comparisons are available below."
        )
        executive_body = (
            "Every displayed value comes from a validated study result with a common per-model "
            "BF16 reference signature. Interpret the W8A8 bundle and W8A16 control matrix under "
            "the stated confounding guardrails."
        )
        executive_panel_class = "panel"
    elif records:
        status = "partial_results"
        status_note = (
            f"Parsed {completed} comparable successes, {failed} failed/rejected/incomparable "
            f"artifacts, {pending} pending artifacts, and {missing} expected model/candidate "
            "pairs are still missing."
        )
        executive_title = (
            "Partial measurements are available; cross-candidate conclusions remain provisional."
        )
        executive_body = (
            "The table shows only values read from result artifacts. Charts include successful "
            "artifacts only when their per-model BF16 reference signatures match."
        )
        executive_panel_class = "panel pending-panel"
    else:
        status = manifest["status"]
        status_note = manifest["status_note"]
        executive_title = (
            "Infrastructure and measurement design are ready; numerical conclusions are pending."
        )
        executive_body = (
            "No model-result values have been prefilled. The study will test whether increasingly "
            "local scales reduce outlier coupling enough to outweigh MXFP8's power-of-two scale "
            "rounding, separately for the dense and MoE architectures."
        )
        executive_panel_class = "panel pending-panel"
    model_cards = "".join(
        f'<div class="mini-card"><strong>{esc(model["short_name"])}</strong>'
        f"<span>{esc(model['role'])} · <code>{esc(model['handle'])}</code></span></div>"
        for model in manifest["models"]
    )
    candidate_rows = "".join(
        "<tr>"
        f"<td><strong>{esc(item['label'])}</strong><br><code>{esc(item['id'])}</code></td>"
        f"<td>{esc(item['scope'])}</td><td>{esc(item['value_format'])}</td>"
        f"<td>{esc(item['weight_granularity'])}</td><td>{esc(item['activation_granularity'])}</td>"
        f"<td>{esc(item['scale_format'])}</td><td>{esc(item['calibration'])}</td>"
        f"<td>{esc(item['classification'])}</td><td>{esc(item['effective_bits'])}</td>"
        "</tr>"
        for item in candidates
    )
    guardrails = "".join(f"<li>{esc(item)}</li>" for item in manifest["interpretation_guardrails"])
    errors_html = ""
    if errors:
        errors_html = (
            '<div class="warning"><strong>Rejected or incomparable artifacts</strong><ul>'
            + "".join(f"<li>{esc(error)}</li>" for error in errors)
            + "</ul></div>"
        )
    methodology = manifest["methodology"]
    execution = manifest["execution"]
    historical = manifest["historical_reference"]
    generated = datetime.now().astimezone().isoformat(timespec="seconds")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{esc(manifest["title"])}</title>
  <style>
    :root{{--ink:#e8eef5;--muted:#9eacbd;--panel:#111923;--panel2:#162232;--line:#2a384a;--green:#76b900;--violet:#818cf8;--amber:#f3b33d;--red:#fb7185;--bg:#081019}}
    *{{box-sizing:border-box}} body{{margin:0;background:radial-gradient(circle at 88% 4%,#183126 0,transparent 28%),var(--bg);color:var(--ink);font:15px/1.55 Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
    main{{max-width:1440px;margin:auto;padding:40px 36px 80px}} a{{color:#a7c7ff}} code{{font-family:"SFMono-Regular",Consolas,monospace;font-size:.88em;overflow-wrap:anywhere}}
    .eyebrow{{color:var(--green);font-weight:750;letter-spacing:.12em;text-transform:uppercase;font-size:12px}} h1{{font-size:clamp(34px,5vw,64px);line-height:1.02;max-width:940px;margin:12px 0 20px;letter-spacing:-.045em}} h2{{font-size:28px;margin:64px 0 18px;letter-spacing:-.025em}} h3{{font-size:18px;margin:28px 0 10px}} p{{max-width:930px;color:#c6d0dc}} .lead{{font-size:19px;color:#cbd7e5}}
    .topline{{display:flex;gap:12px;align-items:center;flex-wrap:wrap}} .tag{{display:inline-block;padding:3px 9px;border-radius:999px;background:#233044;color:#ced8e4;font-size:12px;white-space:nowrap}} .tag.ok{{background:#183b27;color:#9ce2b1}} .tag.pending{{background:#493817;color:#ffd98e}} .tag.bad{{background:#4d202a;color:#ffb0bf}}
    .stats{{display:grid;grid-template-columns:repeat(4,minmax(140px,1fr));gap:14px;margin:30px 0}} .stat,.mini-card,.panel{{border:1px solid var(--line);background:linear-gradient(145deg,rgba(22,34,50,.96),rgba(12,21,31,.96));border-radius:14px;box-shadow:0 14px 40px rgba(0,0,0,.15)}} .stat{{padding:18px}} .stat b{{display:block;font-size:28px}} .stat span,.mini-card span{{display:block;color:var(--muted);font-size:13px}}
    .models{{display:grid;grid-template-columns:repeat(2,1fr);gap:14px}} .mini-card{{padding:18px}} .mini-card strong{{display:block;font-size:18px;margin-bottom:4px}} .panel{{padding:22px;margin:16px 0}} .pending-panel{{border-left:4px solid var(--amber)}}
    .table-wrap{{overflow-x:auto;border:1px solid var(--line);border-radius:12px}} table{{border-collapse:collapse;width:100%;min-width:1040px;background:#0e1721;font-size:13px}} th,td{{padding:11px 12px;border-bottom:1px solid var(--line);vertical-align:top;text-align:left}} th{{position:sticky;top:0;background:#172333;color:#dce7f4;font-size:11px;text-transform:uppercase;letter-spacing:.05em}} tr:last-child td{{border-bottom:0}} tr:hover td{{background:#121f2d}}
    .formula-grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:14px}} .formula{{padding:19px;border:1px solid var(--line);border-radius:12px;background:#0c151f}} .formula code{{display:block;color:#c4f092;font-size:14px;margin-bottom:8px}} .formula small{{color:var(--muted)}}
    .flow{{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;counter-reset:step}} .flow div{{border:1px solid var(--line);border-radius:10px;padding:16px;background:#101b28;min-height:115px}} .flow div:before{{counter-increment:step;content:counter(step);display:block;color:var(--green);font-weight:800;margin-bottom:7px}}
    .empty,.warning{{border:1px dashed #846629;border-radius:12px;background:#2d2515;padding:18px;color:#f5dfaf}} .warning{{border-color:#8f3347;background:#301921;color:#ffc3cf}} .chart{{overflow-x:auto;background:#0a121b;border:1px solid var(--line);border-radius:12px;padding:16px}} .chart svg{{min-width:820px;width:100%;height:auto}} .chart-empty{{color:var(--muted);padding:45px;text-align:center}}
    .callout{{border-left:4px solid var(--violet);padding:12px 17px;background:#171a31;border-radius:0 10px 10px 0}} ul{{padding-left:22px;color:#c6d0dc}} details.diagnostic{{border:1px solid var(--line);border-radius:12px;background:#0d1722;margin:12px 0;padding:0 16px 16px}} details.diagnostic summary{{cursor:pointer;padding:15px 0;font-weight:750}} details.diagnostic h4{{margin:20px 0 8px}} .small-note{{font-size:13px;color:var(--muted)}} footer{{margin-top:70px;padding-top:18px;border-top:1px solid var(--line);color:var(--muted);font-size:12px}}
    @media(max-width:850px){{main{{padding:28px 18px 60px}}.stats,.models,.formula-grid{{grid-template-columns:1fr 1fr}}.flow{{grid-template-columns:1fr}}}}
    @media(max-width:560px){{.stats,.models,.formula-grid{{grid-template-columns:1fr}}}}
    @media print{{body{{background:white;color:#111}}.panel,.stat,.mini-card,.formula{{background:white;color:#111;box-shadow:none}}p,ul,.stat span,.mini-card span,.formula small{{color:#333}}}}
  </style>
</head>
<body><main>
  <div class="eyebrow">ModelOpt numerical study · prepared 17 Jul 2026</div>
  <h1>{esc(manifest["title"])}</h1>
  <p class="lead">{esc(manifest["objective"])}</p>
  <div class="topline"><span class="tag {status_class(status)}">{esc(status.replace("_", " "))}</span><span>{esc(status_note)}</span></div>
  <div class="stats">
    <div class="stat"><b>{len(manifest["models"])}</b><span>official Qwen checkpoints</span></div>
    <div class="stat"><b>{len(candidates)}</b><span>FP8 candidates per model</span></div>
    <div class="stat"><b>{completed}/{expected}</b><span>successful result artifacts</span></div>
    <div class="stat"><b>1 × 4</b><span>aws-cmh nodes × GPUs per candidate</span></div>
  </div>
  <div class="models">{model_cards}</div>

  <h2>Executive readout</h2>
  <div class="{executive_panel_class}">
    <strong>{esc(executive_title)}</strong>
    <p>{esc(executive_body)}</p>
  </div>
  <div class="callout"><strong>Two linked screens, not one isolated variable.</strong> The W8A8 screen compares format-level bundles: per-tensor uses a static calibrated activation scale, while block-128 and MXFP8 rescale dynamically per invocation; MXFP8 also changes block size and scale encoding. The W8A16 controls remove activation quantization, but MXFP8 versus block-128 still combines a block-size change with E8M0 versus full-precision scales.</div>

  <h2>Candidate matrix</h2>
  <div class="table-wrap"><table><thead><tr><th>Candidate</th><th>Scope</th><th>Values</th><th>Weight scale domain</th><th>Activation scale domain</th><th>Scale representation</th><th>Calibration</th><th>Support boundary</th><th>Approx. storage</th></tr></thead><tbody>{candidate_rows}</tbody></table></div>

  <h2>Mathematical lens</h2>
  <p>For E4M3, ModelOpt maps values into a finite FP8 grid after scaling. These candidates vary which values share a scale, when the scale is obtained, and whether it is full precision or the E8M0 power-of-two scale used by MXFP8. The primary W8A8 result is therefore a comparison of complete numerical-format policies, not a causal estimate of granularity alone.</p>
  <div class="formula-grid">
    <div class="formula"><code>Q(x) = s · FP8_E4M3(clamp(x / s, −448, 448))</code><small>Per-tensor FP8 uses one calibrated <em>s</em>; block formats recompute or store local scales.</small></div>
    <div class="formula"><code>s_MX = 2^ceil(log₂(amax / 448))</code><small>MXFP8's E8M0 scale protects the local block from clipping, but power-of-two rounding can leave up to roughly 2× range slack.</small></div>
    <div class="formula"><code>NMSE_Qwen = meanₜ [ meanᵥ((q̃ₜ − r̃ₜ)²) / (meanᵥ(r̃ₜ²) + ε) ]</code><small><em>q̃</em> and <em>r̃</em> are centered over vocabulary for each scored token. The report averages tokenwise ratios, so each valid next-token position has equal weight.</small></div>
    <div class="formula"><code>KL(Pᵣ || Pq) = Σᵥ Pᵣ(v) · [log Pᵣ(v) − log Pq(v)]</code><small>Forward KL measures reference probability mass lost by quantization; reverse KL and Jensen–Shannon expose different tail behavior.</small></div>
  </div>
  <h3>Hypotheses to test, not results</h3>
  <ul>
    <li>A sufficiently extreme outlier can inflate a per-tensor scale enough to coarsen or underflow ordinary values that share it.</li>
    <li>128-value/128×128 local scaling should isolate outliers, with modest scale overhead.</li>
    <li>MXFP8's 32-value blocks offer stronger isolation, while E8M0 scale rounding can reduce effective mantissa use compared with a full-precision local scale.</li>
    <li>Output KL need not track weight MSE monotonically: errors in sensitive projections and router-adjacent paths can matter more than aggregate squared error.</li>
  </ul>

  <h3>Synthetic ModelOpt outlier stress test</h3>
  <p>This deterministic 64 × 1,024 tensor check is a mechanism test, <strong>not a Qwen result</strong>. It uses ModelOpt's own <code>FP8QTensor</code> and <code>MXFP8QTensor</code> paths. One signed outlier is injected per row; the high end is intentionally extreme to expose E4M3 underflow/coarsening. Static and dynamic full-precision block scaling have the same instantaneous mapping on one tensor, so the green curve represents both before cross-batch calibration effects. Its “normalized MSE” is a single ratio of global element means; unlike <code>NMSE_Qwen</code>, it is not the mean of per-token ratios and should not be numerically compared with the Qwen screen.</p>
  <div class="chart">{toy_chart(toy)}</div>
  <p class="callout"><strong>Mechanism readout.</strong> {toy_readout(toy)}</p>
  <details><summary>Show synthetic metrics</summary>{toy_table(toy)}</details>

  <h2>Measurement protocol</h2>
  <div class="flow">
    <div><strong>Pin inputs</strong><br>Resolve model revision, tokenizer, dataset sample IDs, padding, sequence length, dtype, and random seeds.</div>
    <div><strong>Cache reference</strong><br>Persist BF16 logits and a matching manifest before any in-place quantization.</div>
    <div><strong>Apply one candidate</strong><br>Use ModelOpt fake QDQ; refresh research block-reshape state per invocation for variable routed-MoE shapes.</div>
    <div><strong>Measure globally</strong><br>Mask padding; compute MSE, normalized MSE, bidirectional KL, JS, target log-p, ΔNLL, top-k agreement, and document bootstrap intervals.</div>
    <div><strong>Localize error</strong><br>Measure all weight/expert slices directly, hook input quantizers, and retain family summaries, provenance, coverage, and wall times.</div>
  </div>
  <div class="panel"><strong>Fixed deterministic screen:</strong> {methodology["calibration_samples"]} packed calibration samples and {methodology["evaluation_samples"]} evaluation documents from row offset {methodology["evaluation_row_offset"]} of the same CNN/DailyMail training split; batch {methodology["batch_size"]}; padded sequence length {methodology["sequence_length"]}. The 32-document evaluation slice is a controlled numerical screen, not evidence of broad task or corpus generalization. <span style="color:var(--muted)">{esc(methodology["fixed_shape_reason"])}</span></div>

  <h2>Qwen3.6 experiment results</h2>
{errors_html}
  {results_table(records, candidate_labels)}
  <p class="small-note">Token NMSE weights every valid next-token position equally. The adjacent 95% interval bootstraps equal-weight document means from the same 32 paired quantized/reference documents; it quantifies screen sampling uncertainty, not corpus or task generalization.</p>
  <h3>Centered / variance-normalized logit MSE</h3>
  <div class="chart">{metric_chart(records, "normalized_mse", candidate_labels)}</div>
  <h3>Forward KL: reference → quantized</h3>
  <div class="chart">{metric_chart(records, "kl_forward", candidate_labels)}</div>

  <h3>Within-scope screen rankings</h3>
  <p>Ranks are computed only among validated artifacts for the same model, W8A8/W8A16 scope, and exact BF16-reference signature. Lower is better for NMSE, KL, and JS; higher is better for top-1 agreement. These are ranks on the deterministic 32-document screen, not general model-quality rankings.</p>
  {ranking_tables(records, candidates)}

  <h2>Quantizer-level diagnostics</h2>
  <p>Weight MSE is measured once directly for every mapped weight and fused-expert slice; input MSE averages ModelOpt hook invocations over 32 packed calibration rows. Every eligible role must reach complete coverage, and cross-candidate owner sets must match before ranking. Logical weight costs describe FP8 payload plus declared scale bits; they are not measured checkpoint size, GPU memory, or throughput.</p>
  <h3>Coverage and logical weight cost</h3>
  {coverage_and_cost_table(records, candidate_labels)}
  <h3>Family and named-quantizer MSE</h3>
  {quantizer_mse_details(records, candidate_labels)}

  <h3>Phase wall times</h3>
  <p>Wall times are operational context, not kernel benchmarks: model-cache reuse, distributed loading, and the research fake-quant path can dominate them.</p>
  {walltime_table(records, candidate_labels)}

  <h2>Interpretation guardrails</h2>
  <ul>{guardrails}</ul>

  <h2>Execution provenance</h2>
  <div class="table-wrap"><table><tbody>
    <tr><th>Cluster</th><td>{esc(execution["cluster"])}</td><th>Slurm account</th><td>{esc(execution["account"])}</td></tr>
    <tr><th>Partition</th><td>{esc(execution["partition"])}</td><th>Requested resources</th><td>1 node, {execution["gpus_per_candidate"]} GPUs per running candidate; task-slot walltime {esc(execution["gpu_task_slot_walltime"])}</td></tr>
    <tr><th>Remote root</th><td colspan="3"><code>{esc(execution["remote_root"])}</code></td></tr>
    <tr><th>Source ref / base</th><td><code>{esc(execution["source_ref"])}</code> from <code>{esc(execution["source_base_commit"])}</code></td><th>Hardware validation</th><td>{esc(execution["hardware_note"])}</td></tr>
  </tbody></table></div>

  <h2>Historical NEL reference</h2>
  <p>The requester supplied this as a prior deployment-control example. It informs job sizing and reporting discipline; it is not a result in the Qwen study.</p>
  <div class="table-wrap"><table><tbody>
    <tr><th>Reference</th><td>{esc(historical["label"])}</td></tr>
    <tr><th>Model</th><td><code>{esc(historical["model"])}</code></td></tr>
    <tr><th>Serving</th><td>{esc(historical["serving"])}</td></tr>
    <tr><th>Tasks</th><td>{esc(historical["tasks"])}</td></tr>
    <tr><th>Sampling</th><td>{esc(historical["sampling"])}</td></tr>
    <tr><th>Boundary</th><td>{esc(historical["important_difference"])}</td></tr>
  </tbody></table></div>

  <footer>Generated {esc(generated)} from <code>study_manifest.json</code> and {len(records)} JSON artifact(s) under <code>results/</code>. This HTML is self-contained; no external scripts, fonts, or chart libraries are loaded.</footer>
</main></body></html>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--toy", type=Path, default=DEFAULT_TOY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_json(args.manifest)
    records, errors = load_results(args.results_dir, manifest)
    toy = load_json(args.toy) if args.toy.exists() else None
    output = render(manifest, records, errors, toy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding="utf-8")
    print(f"Wrote {args.output} with {len(records)} result artifact(s)")


if __name__ == "__main__":
    main()
