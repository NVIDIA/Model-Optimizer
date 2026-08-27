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

"""Runner-backed final campaign report contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from pathlib import Path

from .schema import AttemptSpec, CampaignPlan, CommandSpec, TaskLauncher, TaskTopology

__all__ = [
    "FinalReportResult",
    "build_final_report_attempt",
    "completed_final_report",
    "final_report_paths",
    "record_completed_final_report",
]


@dataclass(frozen=True)
class FinalReportResult:
    """Nonfatal outcome of final campaign report generation."""

    status: str
    path: str | None = None
    manifest_path: str | None = None
    log_paths: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Return fields exposed by the orchestrator result contract."""

        return {
            "report_status": self.status,
            "report_path": self.path,
            "report_manifest_path": self.manifest_path,
            "report_log_paths": list(self.log_paths),
        }


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _report_model_name(config: Mapping[str, Any]) -> str:
    """Return the same stable model identity used by the in-process reporter."""

    model = _mapping(config.get("model"))
    model_info = _mapping(config.get("model_info"))
    return str(
        config.get("display_name")
        or model_info.get("hf_repo")
        or model.get("display_name")
        or model.get("name")
        or model.get("source")
        or "Puzzletron model"
    )


def final_report_paths(plan: CampaignPlan) -> tuple[Path, Path]:
    """Return canonical HTML and manifest paths for a campaign."""

    output_dir = plan.puzzle_dir / "artifacts" / "campaign_report"
    return output_dir / "campaign_report.html", output_dir / "report_manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _completion_path(plan: CampaignPlan) -> Path:
    report_path, _ = final_report_paths(plan)
    return report_path.parent / "completion.json"


def completed_final_report(plan: CampaignPlan) -> FinalReportResult | None:
    """Return a sealed final report when its contract and artifact hashes still match."""

    report_path, manifest_path = final_report_paths(plan)
    try:
        payload = json.loads(_completion_path(plan).read_text(encoding="utf-8"))
        log_paths = payload["log_paths"]
        if (
            payload["schema_version"] != 1
            or payload["contract_hash"] != plan.contract_hash
            or payload["report_sha256"] != _sha256(report_path)
            or payload["manifest_sha256"] != _sha256(manifest_path)
            or not isinstance(log_paths, list)
            or not all(isinstance(path, str) for path in log_paths)
        ):
            return None
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return FinalReportResult(
        status="completed",
        path=str(report_path),
        manifest_path=str(manifest_path),
        log_paths=tuple(log_paths),
    )


def record_completed_final_report(
    plan: CampaignPlan, *, log_paths: tuple[str, ...]
) -> FinalReportResult:
    """Atomically seal a completed final report for idempotent controller resumes."""

    report_path, manifest_path = final_report_paths(plan)
    payload = {
        "schema_version": 1,
        "contract_hash": plan.contract_hash,
        "report_sha256": _sha256(report_path),
        "manifest_sha256": _sha256(manifest_path),
        "log_paths": list(log_paths),
    }
    completion_path = _completion_path(plan)
    temporary = completion_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(completion_path)
    return FinalReportResult(
        status="completed",
        path=str(report_path),
        manifest_path=str(manifest_path),
        log_paths=log_paths,
    )


def build_final_report_attempt(plan: CampaignPlan, *, attempt_id: str) -> AttemptSpec:
    """Build one direct, CPU-only attempt in the campaign runner environment."""

    metadata: dict[str, Any] = {"gpus_per_node": 0}
    if plan.final_report_partition is not None:
        metadata["partition"] = plan.final_report_partition
    log_path = plan.log_dir / f"final_report_{attempt_id}.log"
    return AttemptSpec(
        attempt_id=attempt_id,
        work_id="final_report:0",
        stage_id="final_report",
        command=CommandSpec(
            argv=(
                "python",
                "examples/puzzletron/generate_campaign_progress_report.py",
                "--puzzle-dir",
                str(plan.puzzle_dir),
                "--model-name",
                _report_model_name(plan.experiment_config),
            ),
            cwd=plan.runner.contract.repository,
            log_path=str(log_path),
        ),
        allocation_nodes=1,
        allocation_gpus=0,
        exclusive=False,
        contract_hash=plan.contract_hash,
        metadata=metadata,
        task_topology=TaskTopology(
            task_count=1,
            gpus_per_task=0,
            launcher=TaskLauncher.DIRECT,
        ),
    )
