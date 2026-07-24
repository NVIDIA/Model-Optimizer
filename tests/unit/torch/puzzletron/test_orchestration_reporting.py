# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the runner-backed orchestration report finalizer."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from puzzletron_orchestrator.controller import CampaignController
from puzzletron_orchestrator.executors.base import Executor
from puzzletron_orchestrator.reporting import build_final_report_attempt
from puzzletron_orchestrator.schema import (
    AttemptSpec,
    CampaignPlan,
    ExecutionContract,
    ExecutionStrategy,
    FailurePolicy,
    JobHandle,
    JobState,
    JobStatus,
    RunnerEnvironment,
    SlurmRunnerConfig,
    StagePlanNode,
    TaskLauncher,
)


def _plan(tmp_path: Path, *, partition_cpu: str | None = "cpu") -> CampaignPlan:
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
                partition_interactive="interactive",
                partition_batch="batch",
                partition_cpu=partition_cpu,
            ),
        ),
        execution_defaults={},
        stages=(),
        contract_hash="contract",
    )


def test_build_final_report_attempt_uses_runner_cpu_task(tmp_path: Path):
    plan = _plan(tmp_path)

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
    assert attempt.metadata == {"gpus_per_node": 0, "partition": "cpu"}
    assert attempt.task_topology.task_count == 1
    assert attempt.task_topology.gpus_per_task == 0
    assert attempt.task_topology.launcher is TaskLauncher.DIRECT


def test_build_final_report_attempt_uses_normal_partition_fallback(tmp_path: Path):
    plan = _plan(tmp_path, partition_cpu=None)

    attempt = build_final_report_attempt(plan, attempt_id="report-attempt")

    assert attempt.metadata == {"gpus_per_node": 0}


class _ReportExecutor(Executor):
    backend = "fake"

    def __init__(self, terminal_state: JobState) -> None:
        self.terminal_state = terminal_state
        self.submitted: list[AttemptSpec] = []

    def submit(self, attempt: AttemptSpec) -> JobHandle:
        self.submitted.append(attempt)
        return JobHandle(
            backend=self.backend,
            handle_id=f"fake-{attempt.attempt_id}",
            attempt_id=attempt.attempt_id,
            metadata={
                "log_paths": (attempt.command.log_path,) if attempt.command.log_path else (),
            },
        )

    def poll(self, handles: Sequence[JobHandle]) -> list[JobStatus]:
        if self.terminal_state is JobState.COMPLETED:
            plan_root = Path(self.submitted[-1].command.argv[3])
            report_dir = plan_root / "artifacts" / "campaign_report"
            report_dir.mkdir(parents=True, exist_ok=True)
            (report_dir / "campaign_report.html").write_text("<html></html>\n")
            (report_dir / "report_manifest.json").write_text("{}\n")
        return [
            JobStatus(
                handle=handle,
                state=self.terminal_state,
                reason="report failed" if self.terminal_state is JobState.FAILED else None,
                log_paths=self.fetch_logs(handle),
            )
            for handle in handles
        ]

    def cancel(self, handles: Sequence[JobHandle]) -> None:
        pass

    def recover(self, handle: JobHandle) -> JobStatus:
        return JobStatus(handle=handle, state=self.terminal_state)


def test_clean_completion_generates_and_returns_final_report(tmp_path: Path):
    plan = _plan(tmp_path)
    executor = _ReportExecutor(JobState.COMPLETED)
    controller = CampaignController(plan, executor=executor, poll_interval_seconds=0)

    result = controller.run()

    report_dir = plan.puzzle_dir / "artifacts" / "campaign_report"
    assert [attempt.stage_id for attempt in executor.submitted] == ["final_report"]
    assert result["halted"] is False
    assert result["report_status"] == "completed"
    assert result["report_path"] == str(report_dir / "campaign_report.html")
    assert result["report_manifest_path"] == str(report_dir / "report_manifest.json")
    assert result["report_log_paths"] == [executor.submitted[0].command.log_path]


def test_final_report_failure_is_nonfatal(tmp_path: Path):
    plan = _plan(tmp_path)
    executor = _ReportExecutor(JobState.FAILED)
    controller = CampaignController(plan, executor=executor, poll_interval_seconds=0)

    result = controller.run()

    assert result["halted"] is False
    assert result["failed_stages"] == []
    assert result["report_status"] == "failed"
    assert result["report_path"] is None
    assert result["report_manifest_path"] is None
    assert result["report_log_paths"] == [executor.submitted[0].command.log_path]


def test_partial_invocation_skips_final_report(tmp_path: Path, monkeypatch):
    base = _plan(tmp_path)
    node = StagePlanNode(
        stage_id="sort_sanity",
        strategy=ExecutionStrategy.SINGLE,
        instances=1,
        failure_policy=FailurePolicy.STRICT,
        mesh={},
        gpus_per_instance=1,
        gpus_per_node=1,
        nodes=1,
        total_gpus=1,
        exclusive=False,
        parents=(),
        distributed=False,
    )
    plan = CampaignPlan(
        experiment_config_path=base.experiment_config_path,
        puzzle_dir=base.puzzle_dir,
        experiment_config=base.experiment_config,
        runner=base.runner,
        execution_defaults=base.execution_defaults,
        stages=(node,),
        contract_hash=base.contract_hash,
    )
    monkeypatch.setattr(
        "puzzletron_orchestrator.controller.stage_is_complete",
        lambda *_args, **_kwargs: False,
    )
    executor = _ReportExecutor(JobState.COMPLETED)
    controller = CampaignController(plan, executor=executor, poll_interval_seconds=0)

    result = controller.run(max_iterations=0)

    assert executor.submitted == []
    assert result["report_status"] == "skipped"
    assert result["report_path"] is None
    assert result["report_manifest_path"] is None
    assert result["report_log_paths"] == []
