"""Conservative importer for legacy Puzzletron JSON evaluation caches."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from .campaign import Campaign
from .schema import EvaluationRequest, EvaluationResult
from .storage import atomic_write_json


def _decode_legacy_key(key: str) -> dict[str, Any] | None:
    try:
        decoded = json.loads(key)
    except (TypeError, json.JSONDecodeError):
        return None
    if isinstance(decoded, dict):
        return decoded
    if isinstance(decoded, list):
        return {"architecture": decoded}
    return None


def import_legacy_cache(
    campaign: Campaign,
    cache_file: str | Path,
    *,
    handler: str = "replace_block",
) -> dict[str, Any]:
    path = Path(cache_file).resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Legacy cache must be a JSON object: {path}")
    report: dict[str, Any] = {
        "source": str(path),
        "imported": 0,
        "duplicates": 0,
        "conflicts": 0,
        "skipped": [],
    }
    for legacy_key, legacy_value in raw.items():
        payload = _decode_legacy_key(legacy_key)
        if payload is None:
            report["skipped"].append({"key": legacy_key, "reason": "key is not JSON"})
            continue
        if not isinstance(legacy_value, dict):
            report["skipped"].append({"key": legacy_key, "reason": "value is not an object"})
            continue
        metrics = legacy_value.get("metrics", legacy_value)
        if not isinstance(metrics, dict):
            report["skipped"].append({"key": legacy_key, "reason": "metrics are not an object"})
            continue
        request = EvaluationRequest(
            campaign_id=campaign.campaign_id,
            handler=handler,
            payload=payload,
            model=campaign.manifest.model,
            data=campaign.manifest.data,
            metrics=campaign.manifest.metrics,
            precision=campaign.manifest.precision,
            evaluator_revision=campaign.manifest.evaluator_revision,
            metadata={"legacy_cache": str(path), "legacy_key": legacy_key},
        )
        result = EvaluationResult(
            request_id=request.request_id,
            campaign_id=campaign.campaign_id,
            metrics=metrics,
            provenance={"imported_from": str(path), "legacy_key": legacy_key},
        )
        try:
            campaign.storage.put_request(request)
            status = campaign.storage.put_result(
                result,
                attempt_id=f"legacy-{uuid.uuid4().hex}",
                atol=campaign.manifest.result_atol,
                rtol=campaign.manifest.result_rtol,
            )
        except (TypeError, ValueError) as error:
            report["skipped"].append({"key": legacy_key, "reason": str(error)})
            continue
        report_key = {
            "written": "imported",
            "duplicate": "duplicates",
            "conflict": "conflicts",
        }[status.value]
        report[report_key] += 1
    report_path = campaign.storage.summaries_dir / f"legacy_import_{path.stem}.json"
    atomic_write_json(report_path, report)
    report["report_path"] = str(report_path)
    return report
