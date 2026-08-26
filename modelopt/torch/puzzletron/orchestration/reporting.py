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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from pathlib import Path

from .schema import AttemptSpec, CampaignPlan, CommandSpec, TaskLauncher, TaskTopology

__all__ = [
    "FinalReportResult",
    "build_final_report_attempt",
    "final_report_paths",
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
