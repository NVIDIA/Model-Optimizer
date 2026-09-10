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

"""Tests for the generated discovery catalog over authoritative and legacy results."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from examples.puzzletron.generate_results_catalog import load_sources
from puzzletron_orchestrator.result_catalog import (
    LEGACY_WRAPPER_SCHEMA,
    CatalogSource,
    build_results_catalog,
    render_catalog_yaml,
)
from puzzletron_orchestrator.run_reporting import (
    RESULT_SCHEMA,
    ResultValidationError,
    canonical_json_bytes,
    result_sha256,
)

NOW = "2026-09-08T02:00:00+00:00"


def _current_result() -> dict:
    return {
        "schema": RESULT_SCHEMA,
        "run": {
            "identity": {
                "run_id": "run-1",
                "model_family": "Qwen",
                "modality": "vlm",
            },
            "status": {
                "execution": "completed",
                "attachment": "not_running",
                "evidence": "complete",
                "support": "supported",
            },
            "timing": {
                "created_at": "2026-09-08T00:00:00+00:00",
                "started_at": None,
                "updated_at": "2026-09-08T01:00:00+00:00",
                "ended_at": "2026-09-08T01:00:00+00:00",
                "finalized_at": "2026-09-08T01:00:00+00:00",
                "last_executor_observation_at": "2026-09-08T01:00:00+00:00",
            },
            "freshness": {
                "as_of": "2026-09-08T01:00:00+00:00",
                "stale_after_seconds": 120,
                "state": "fresh",
                "reason": None,
            },
        },
        "subjects": [
            {
                "subject_id": "subject:candidate",
                "role": "candidate",
                "checkpoint": {"checkpoint_id": "checkpoint:candidate"},
                "architecture": {"architecture_id": "architecture:candidate"},
            }
        ],
        "stages": [],
        "metrics": [
            {
                "metric_id": "metric:accuracy",
                "name": "quality.accuracy",
                "value": 0.7,
                "value_state": "present",
                "missing_reason": None,
                "unit": "ratio",
                "direction": "higher_is_better",
                "subject_id": "subject:candidate",
                "checkpoint_id": "checkpoint:candidate",
                "producer_execution_id": "evaluation-1",
                "workload": {"workload_id": "workload-1"},
                "aggregation": "mean",
                "dimensions": {},
            }
        ],
        "artifacts": [],
        "provenance": {
            "resolved_bundle": {
                "bundle_id": "run-1",
                "producer_revision": "revision-1",
                "manifest": "orchestration/resolved_bundles/run-1/manifest.json",
                "provenance": "orchestration/resolved_bundles/run-1/provenance.json",
            }
        },
        "limitations": [],
    }


def test_catalog_discovers_structured_results_without_copying_metric_values() -> None:
    result = _current_result()
    source = CatalogSource(
        "reports/qwen/vlm/campaign/runs/run-1/result.json",
        result,
        summary_path="reports/qwen/vlm/campaign/runs/run-1/summary.html",
    )

    catalog = build_results_catalog([source], generated_at=NOW, generator_revision="catalog-v1")
    entry = catalog["entries"][0]

    assert entry == {
        "run_id": "run-1",
        "model_family": "Qwen",
        "modality": "vlm",
        "evidence_class": "authoritative_structured_result",
        "execution_status": "completed",
        "evidence_status": "complete",
        "updated_at": "2026-09-08T01:00:00+00:00",
        "result_path": "reports/qwen/vlm/campaign/runs/run-1/result.json",
        "result_digest": result_sha256(canonical_json_bytes(result)),
        "subject_roles": ["candidate"],
        "metric_names": ["quality.accuracy"],
        "summary_path": "reports/qwen/vlm/campaign/runs/run-1/summary.html",
    }
    assert "value: 0.7" not in render_catalog_yaml(catalog)

    with pytest.raises(ResultValidationError, match="duplicate result paths"):
        build_results_catalog([source, source], generated_at=NOW, generator_revision="catalog-v1")


def test_catalog_qualifies_historical_html_without_inventing_evidence() -> None:
    legacy = {
        "schema": LEGACY_WRAPPER_SCHEMA,
        "record_id": "legacy-record",
        "run": {"id": "legacy-run", "recorded_date": "2026-09-01"},
        "campaign": {"modality": "vlm"},
        "model": {"family": "Qwen"},
        "evidence_status": "teacher_only",
        "reproduction_status": "not_reproducible_from_retained_material",
        "legacy_support_status": "historical_only",
        "artifacts": [
            {
                "role": "legacy_summary_html",
                "path": "reports/qwen/vlm/campaign/legacy.html",
            }
        ],
        "limitations": ["The evaluator revision was not retained."],
    }

    entry = build_results_catalog(
        [
            CatalogSource(
                "reports/qwen/vlm/campaign/runs/legacy/result_record.json",
                legacy,
                summary_path="reports/qwen/vlm/campaign/legacy.html",
            )
        ],
        generated_at=NOW,
        generator_revision="catalog-v1",
    )["entries"][0]

    assert entry["evidence_class"] == "qualified_historical"
    assert entry["execution_status"] == "unknown"
    assert entry["evidence_status"] == "teacher_only"
    assert entry["limitation_count"] == 1
    assert "metric_names" not in entry


def test_checked_in_catalog_reproduces_all_structured_leaves() -> None:
    repository_root = Path(__file__).resolve().parents[4]
    catalog_path = repository_root / "examples/puzzletron/reports/catalog.yaml"
    checked_in = yaml.safe_load(catalog_path.read_text())

    generated = build_results_catalog(
        load_sources(),
        generated_at=checked_in["generated_at"],
        generator_revision=checked_in["generator_revision"],
    )

    assert checked_in == generated
