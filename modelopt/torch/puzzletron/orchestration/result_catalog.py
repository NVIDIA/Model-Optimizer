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

"""Deterministic discovery catalog for current and qualified historical results."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

import yaml

from .run_reporting import (
    RESULT_SCHEMA,
    ResultValidationError,
    canonical_json_bytes,
    result_sha256,
    validate_result,
)

__all__ = [
    "CATALOG_SCHEMA",
    "LEGACY_RESULT_RECORD_SCHEMA",
    "LEGACY_WRAPPER_SCHEMA",
    "CatalogSource",
    "build_results_catalog",
    "render_catalog_yaml",
]

CATALOG_SCHEMA = "modelopt.puzzletron.results-catalog/v1"
LEGACY_RESULT_RECORD_SCHEMA = "modelopt.puzzletron-result-record/v1"
LEGACY_WRAPPER_SCHEMA = "modelopt.puzzletron.legacy-result-wrapper/v1"


class _CatalogDumper(yaml.SafeDumper):
    def increase_indent(self, flow: bool = False, indentless: bool = False) -> None:
        return super().increase_indent(flow, False)


_CatalogDumper.add_representer(
    type(None), lambda dumper, _value: dumper.represent_scalar("tag:yaml.org,2002:null", "")
)


@dataclass(frozen=True)
class CatalogSource:
    result_path: str
    result: Mapping[str, Any]
    summary_path: str | None = None


def _standard_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or str(path) != value
        or path.name not in {"result.json", "result_record.json"}
        or "runs" not in path.parts
    ):
        raise ResultValidationError(f"nonstandard result path: {value!r}")
    return value


def _current_entry(source: CatalogSource) -> dict[str, Any]:
    validate_result(source.result)
    run = source.result["run"]
    identity = run["identity"]
    entry = {
        "run_id": identity["run_id"],
        "model_family": identity.get("model_family"),
        "modality": identity.get("modality"),
        "evidence_class": "authoritative_structured_result",
        "execution_status": run["status"]["execution"],
        "evidence_status": run["status"]["evidence"],
        "updated_at": run["timing"].get("updated_at"),
        "result_path": _standard_path(source.result_path),
        "result_digest": result_sha256(canonical_json_bytes(source.result)),
    }
    roles = sorted({str(item["role"]) for item in source.result["subjects"]})
    metric_names = sorted({str(item["name"]) for item in source.result["metrics"]})
    if roles:
        entry["subject_roles"] = roles
    if metric_names:
        entry["metric_names"] = metric_names
    if source.summary_path:
        entry["summary_path"] = source.summary_path
    return {key: value for key, value in entry.items() if value is not None}


def _legacy_entry(source: CatalogSource) -> dict[str, Any]:
    record = source.result
    schema = record.get("schema")
    if schema not in {LEGACY_RESULT_RECORD_SCHEMA, LEGACY_WRAPPER_SCHEMA}:
        raise ResultValidationError(f"unsupported result schema: {schema!r}")
    record_id = record.get("record_id")
    run = record.get("run")
    campaign = record.get("campaign") or {}
    model = record.get("model") or {}
    limitations = record.get("limitations")
    if not isinstance(record_id, str) or not record_id or not isinstance(run, Mapping):
        raise ResultValidationError("qualified historical result requires record_id and run")
    run_id = run.get("id")
    if not isinstance(run_id, str) or not run_id:
        raise ResultValidationError("qualified historical result requires run.id")
    if not isinstance(limitations, list) or not all(
        isinstance(item, str) and item for item in limitations
    ):
        raise ResultValidationError("qualified historical result requires explicit limitations")
    if schema == LEGACY_WRAPPER_SCHEMA:
        for field in ("reproduction_status", "legacy_support_status"):
            if not isinstance(record.get(field), str) or not record[field]:
                raise ResultValidationError(f"legacy wrapper requires {field}")
        artifacts = record.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise ResultValidationError("legacy wrapper requires at least one artifact")
    status = run.get("status")
    entry = {
        "run_id": run_id,
        "record_id": record_id,
        "model_family": model.get("family") or model.get("id") or model.get("repository"),
        "modality": campaign.get("modality") or model.get("modality"),
        "evidence_class": "qualified_historical",
        "execution_status": "completed" if status == "success" else status or "unknown",
        "evidence_status": record.get("evidence_status", "partial"),
        "updated_at": run.get("recorded_at") or run.get("recorded_date") or run.get("collected_on"),
        "result_path": _standard_path(source.result_path),
        "limitation_count": len(limitations),
    }
    if source.summary_path:
        entry["summary_path"] = source.summary_path
    return {key: value for key, value in entry.items() if value is not None}


def build_results_catalog(
    sources: Iterable[CatalogSource], *, generated_at: str, generator_revision: str
) -> dict[str, Any]:
    """Build one discovery-only YAML catalog without copying result evidence."""

    if not generated_at or not generator_revision:
        raise ValueError("generated_at and generator_revision must be non-empty")
    entries = [
        _current_entry(source)
        if source.result.get("schema") == RESULT_SCHEMA
        else _legacy_entry(source)
        for source in sources
    ]
    entries.sort(key=lambda entry: (str(entry["run_id"]), str(entry["result_path"])))
    paths = [str(entry["result_path"]) for entry in entries]
    if len(paths) != len(set(paths)):
        raise ResultValidationError("catalog contains duplicate result paths")
    return {
        "schema": CATALOG_SCHEMA,
        "role": "generated_discovery_index",
        "evidence_source": "structured result.json leaves and qualified historical wrappers",
        "generated_at": generated_at,
        "generator_revision": generator_revision,
        "entries": entries,
    }


def render_catalog_yaml(catalog: Mapping[str, Any]) -> str:
    if catalog.get("schema") != CATALOG_SCHEMA or not isinstance(catalog.get("entries"), list):
        raise ResultValidationError("unsupported results catalog")
    return yaml.dump(
        dict(catalog),
        Dumper=_CatalogDumper,
        allow_unicode=True,
        default_flow_style=False,
        indent=2,
        sort_keys=False,
    )
