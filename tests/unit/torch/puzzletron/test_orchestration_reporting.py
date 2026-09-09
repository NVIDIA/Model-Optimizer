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

"""Tests for structured-first orchestration report finalization."""

from __future__ import annotations

from pathlib import Path

from puzzletron_orchestrator.controller import CampaignController
from puzzletron_orchestrator.executors.base import Executor
from puzzletron_orchestrator.reporting import build_final_report_attempt
from puzzletron_orchestrator.schema import (
    AttemptSpec,
    CampaignPlan,
    ExecutionContract,
    RunnerEnvironment,
    SlurmRunnerConfig,
    TaskLauncher,
)


def _plan(
    tmp_path: Path,
    *,
    partition: str | None = "shared",
    final_report_partition: str | None = None,
    log_dir: Path | None = None,
) -> CampaignPlan:
    return CampaignPlan(
        experiment_config_path=str(tmp_path / "experiment.yaml"),
        puzzle_dir=tmp_path / "run",
        experiment_config={
            "display_name": "Qwen production",
            "experiment": {"dir": str(tmp_path / "run")},
        },
        runner=RunnerEnvironment(
            kind="slurm",
            contract=ExecutionContract(
                repository=str(tmp_path / "repo"),
                venv=".venv_new",
            ),
            slurm=SlurmRunnerConfig(
                account="test",
                partition=partition,
                log_dir=str(log_dir) if log_dir is not None else None,
            ),
        ),
        execution_defaults={},
        stages=(),
        contract_hash="contract",
        final_report_partition=final_report_partition,
    )


def test_build_final_report_attempt_uses_configured_cpu_partition(tmp_path: Path):
    plan = _plan(tmp_path, final_report_partition="cpu-a,cpu-b")

    attempt = build_final_report_attempt(plan, attempt_id="report-attempt")

    assert attempt.attempt_id == "report-attempt"
    assert attempt.stage_id == "final_report"
    assert attempt.work_id == "final_report:0"
    assert attempt.command.argv == (
        "python",
        "examples/puzzletron/generate_campaign_progress_report.py",
        "--puzzle-dir",
        str(plan.puzzle_dir),
        "--model-name",
        "Qwen production",
    )
    assert attempt.command.cwd == plan.runner.contract.repository
    assert attempt.command.log_path == str(
        plan.puzzle_dir / "logs" / "final_report_report-attempt.log"
    )
    assert attempt.allocation_nodes == 1
    assert attempt.allocation_gpus == 0
    assert attempt.exclusive is False
    assert attempt.metadata == {"gpus_per_node": 0, "partition": "cpu-a,cpu-b"}
    assert attempt.task_topology.task_count == 1
    assert attempt.task_topology.gpus_per_task == 0
    assert attempt.task_topology.launcher is TaskLauncher.DIRECT


def test_build_final_report_attempt_uses_runner_default_when_unconfigured(tmp_path: Path):
    attempt = build_final_report_attempt(_plan(tmp_path), attempt_id="report-attempt")

    assert attempt.metadata == {"gpus_per_node": 0}


def test_build_final_report_attempt_uses_configured_log_directory(tmp_path: Path):
    log_dir = tmp_path / "shared-logs"

    attempt = build_final_report_attempt(
        _plan(tmp_path, log_dir=log_dir),
        attempt_id="report-attempt",
    )

    assert attempt.command.log_path == str(log_dir / "final_report_report-attempt.log")


class _NoSubmissionExecutor(Executor):
    backend = "fake"

    def submit(self, attempt: AttemptSpec):
        raise AssertionError(f"unexpected executor submission: {attempt.stage_id}")

    def poll(self, handles):
        raise AssertionError(f"unexpected executor poll: {handles}")

    def cancel(self, handles) -> None:
        raise AssertionError(f"unexpected executor cancellation: {handles}")

    def recover(self, handle):
        raise AssertionError(f"unexpected executor recovery: {handle}")


def test_clean_completion_generates_and_returns_final_report(tmp_path: Path):
    plan = _plan(tmp_path)
    executor = _NoSubmissionExecutor()
    controller = CampaignController(plan, executor=executor, poll_interval_seconds=0)

    result = controller.run()

    report_dir = plan.puzzle_dir / "artifacts" / "campaign_report"
    assert result["halted"] is False
    assert result["report_status"] == "completed"
    assert result["report_path"] == str(report_dir / "campaign_report.html")
    assert result["report_manifest_path"] == str(report_dir / "report_manifest.json")
    assert result["report_log_paths"] == []
    assert result["result_path"] == str(plan.puzzle_dir / "results/result.json")


def test_clean_completion_regenerates_the_same_derived_report(tmp_path: Path):
    plan = _plan(tmp_path)
    executor = _NoSubmissionExecutor()

    first = CampaignController(plan, executor=executor, poll_interval_seconds=0).run()
    resumed = CampaignController(plan, executor=executor, poll_interval_seconds=0).run()

    assert resumed == first


def test_clean_completion_regenerates_tampered_final_report(tmp_path: Path):
    plan = _plan(tmp_path)
    executor = _NoSubmissionExecutor()
    CampaignController(plan, executor=executor, poll_interval_seconds=0).run()
    report_path = plan.puzzle_dir / "artifacts/campaign_report/campaign_report.html"
    report_path.write_text("tampered\n")

    result = CampaignController(plan, executor=executor, poll_interval_seconds=0).run()

    assert report_path.read_text().startswith("<!doctype html>")
    assert result["report_status"] == "completed"


def test_optional_html_failure_is_nonfatal(monkeypatch, tmp_path: Path):
    plan = _plan(tmp_path)
    executor = _NoSubmissionExecutor()
    controller = CampaignController(plan, executor=executor, poll_interval_seconds=0)
    monkeypatch.setattr(
        "puzzletron_orchestrator.controller.refresh_run_report",
        lambda _run_root: (_ for _ in ()).throw(OSError("report failed")),
    )

    result = controller.run()

    assert result["halted"] is False
    assert result["failed_stages"] == []
    assert result["report_status"] == "failed"
    assert result["report_path"] is None
    assert result["report_manifest_path"] is None
    assert result["report_log_paths"] == []
    assert Path(result["result_path"]).is_file()


def test_structured_result_failure_is_nonfatal(monkeypatch, tmp_path: Path):
    plan = _plan(tmp_path)
    controller = CampaignController(plan, executor=_NoSubmissionExecutor(), poll_interval_seconds=0)
    monkeypatch.setattr(
        "puzzletron_orchestrator.controller.publish_controller_result",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("result failed")),
    )

    result = controller.run()

    assert result["halted"] is False
    assert result["result_path"] is None
    assert result["result_finalized"] is False
    assert result["report_status"] == "failed"
