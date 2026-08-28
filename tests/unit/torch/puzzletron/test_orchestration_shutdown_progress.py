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

"""Tests for orchestrator shutdown and progress reporting."""

from __future__ import annotations

import json
import signal
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import yaml

from puzzletron_orchestrator.adapters.registry import adapter_for_stage
from puzzletron_orchestrator.adapters.stage_compat import (
    stage_is_complete as artifacts_are_complete,
)
from puzzletron_orchestrator.compiler import (
    compile_campaign_plan,
    load_execution_config,
    load_runner_config,
)
from puzzletron_orchestrator.controller import CampaignController
from puzzletron_orchestrator.executors.base import Executor
from puzzletron_orchestrator.schema import (
    AttemptSpec,
    JobHandle,
    JobState,
    JobStatus,
    ValidatedResult,
)
from puzzletron_orchestrator.state import PersistedAttempt, StageRunRecord

if TYPE_CHECKING:
    from collections.abc import Sequence


class _FakeExecutor(Executor):
    backend = "fake"

    def __init__(self) -> None:
        self.cancelled: list[JobHandle] = []
        self._handles: dict[str, JobHandle] = {}
        self._attempts: dict[str, AttemptSpec] = {}

    def submit(self, attempt: AttemptSpec) -> JobHandle:
        handle = JobHandle(
            backend=self.backend,
            handle_id=f"fake-{attempt.attempt_id}",
            attempt_id=attempt.attempt_id,
            metadata={"log_paths": (attempt.command.log_path,) if attempt.command.log_path else ()},
        )
        self._handles[handle.handle_id] = handle
        self._attempts[handle.handle_id] = attempt
        return handle

    def poll(self, handles: Sequence[JobHandle]) -> list[JobStatus]:
        statuses = []
        for handle in handles:
            attempt = self._attempts[handle.handle_id]
            state = JobState.RUNNING
            if attempt.stage_id == "final_report":
                puzzle_dir = Path(
                    attempt.command.argv[attempt.command.argv.index("--puzzle-dir") + 1]
                )
                report_dir = puzzle_dir / "artifacts" / "campaign_report"
                report_dir.mkdir(parents=True, exist_ok=True)
                (report_dir / "campaign_report.html").write_text("<html></html>\n")
                (report_dir / "report_manifest.json").write_text("{}\n")
                state = JobState.COMPLETED
            statuses.append(
                JobStatus(handle=handle, state=state, log_paths=self.fetch_logs(handle))
            )
        return statuses

    def cancel(self, handles: Sequence[JobHandle]) -> None:
        self.cancelled.extend(handles)

    def recover(self, handle: JobHandle) -> JobStatus:
        return JobStatus(handle=handle, state=JobState.RUNNING, log_paths=self.fetch_logs(handle))


def _write_configs(tmp_path: Path):
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(
        yaml.safe_dump(
            {
                "experiment": {"dir": str(tmp_path / "run")},
                "convert": {"enabled": True},
                "depth_importance": {
                    "enabled": True,
                    "max_removals": 5,
                    "expected_initial_sublayers": 80,
                },
                "vllm_stats": {
                    "enabled": True,
                    "runtime_stats": {
                        "topology": {
                            "tensor_parallel_size": 1,
                            "pipeline_parallel_size": 1,
                            "prefill_context_parallel_size": 1,
                            "gpu_group_size": 1,
                        }
                    },
                },
            }
        )
    )
    runner = tmp_path / "runner.yaml"
    runner.write_text(
        yaml.safe_dump(
            {
                "runner": {
                    "kind": "slurm",
                    "slurm": {"account": "test"},
                    "execution_contract": {
                        "repository": str(tmp_path),
                        "venv": str(tmp_path / ".venv"),
                    },
                }
            }
        )
    )
    execution = tmp_path / "execution.yaml"
    execution.write_text(
        yaml.safe_dump(
            {
                "execution": {
                    "defaults": {"gpus_per_node": 8},
                    "stages": {
                        "convert": {"strategy": "single", "instances": 1},
                        "vllm_stats": {"strategy": "sharded", "instances": 2},
                    },
                }
            }
        )
    )
    return experiment, runner, execution


def _write_sanity_drain_configs(tmp_path: Path):
    run_dir = tmp_path / "run"
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(
        yaml.safe_dump(
            {
                "experiment": {"dir": str(run_dir)},
                "sort_sanity": {"enabled": True},
                "width_sanity": {"enabled": True},
                "bypass_sanity": {"enabled": True},
            }
        )
    )
    runner = tmp_path / "runner.yaml"
    runner.write_text(
        yaml.safe_dump(
            {
                "runner": {
                    "kind": "slurm",
                    "slurm": {"account": "test"},
                    "execution_contract": {
                        "repository": str(tmp_path),
                        "venv": str(tmp_path / ".venv"),
                    },
                }
            }
        )
    )
    execution = tmp_path / "execution.yaml"
    execution.write_text(
        yaml.safe_dump(
            {
                "execution": {
                    "defaults": {"gpus_per_node": 8, "halt_policy": "drain"},
                    "stages": {
                        "sort_sanity": {"strategy": "single", "instances": 1},
                        "width_sanity": {"strategy": "single", "instances": 1},
                        "bypass_sanity": {"strategy": "single", "instances": 1},
                    },
                }
            }
        )
    )
    return experiment, runner, execution, run_dir


def _seed_convert_complete(run_dir: Path, write_terminal_manifest, config: dict) -> None:
    write_terminal_manifest(run_dir, "convert", config=config)
    teacher = run_dir / "ckpts" / "teacher"
    teacher.mkdir(parents=True, exist_ok=True)
    (teacher / "config.json").write_text("{}\n")
    (run_dir / "subblock_library.json").write_text("[]\n")


def _seed_sort_complete(run_dir: Path, write_terminal_manifest, config: dict) -> None:
    write_terminal_manifest(run_dir, "sort", config=config)
    sorted_dir = run_dir / "ckpts" / "sorted_teacher"
    sorted_dir.mkdir(parents=True, exist_ok=True)
    (sorted_dir / "config.json").write_text("{}\n")
    (sorted_dir / "parallel_sort_manifest.json").write_text(
        json.dumps({"status": "complete"}) + "\n"
    )
    (sorted_dir / "sorted_permutations.json").write_text("{}\n")
    (sorted_dir / "model.safetensors").write_text("weights\n")


def _seed_sanity_complete(
    run_dir: Path,
    stage_id: str,
    write_terminal_manifest,
    config: dict,
) -> None:
    write_terminal_manifest(run_dir, stage_id, config=config)
    summary = run_dir / "artifacts" / stage_id / "summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps({"passed": True}) + "\n")


def _seed_vllm_stats_complete(run_dir: Path, write_terminal_manifest, config: dict) -> None:
    write_terminal_manifest(run_dir, "vllm_stats", config=config)
    stats = run_dir / "subblock_stats.json"
    stats.write_text(json.dumps([{"args": {"runtime_stats": True, "n_embd": 1}}]) + "\n")
    summary = run_dir / "artifacts" / "vllm_stats" / "summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text("{}\n")


class _TrackingFakeExecutor(_FakeExecutor):
    def __init__(self) -> None:
        super().__init__()
        self.submitted_stage_ids: list[str] = []

    def submit(self, attempt: AttemptSpec) -> JobHandle:
        self.submitted_stage_ids.append(attempt.work_id.split(":", 1)[0])
        handle = JobHandle(
            backend=self.backend,
            handle_id=f"fake-{attempt.attempt_id}",
            attempt_id=attempt.attempt_id,
            metadata={
                "work_id": attempt.work_id,
                "log_paths": (attempt.command.log_path,) if attempt.command.log_path else (),
            },
        )
        self._handles[handle.handle_id] = handle
        self._attempts[handle.handle_id] = attempt
        return handle

    @staticmethod
    def _stage_id(handle: JobHandle) -> str:
        return str(handle.metadata.get("work_id", "")).split(":", 1)[0]


def _blocked_descendants(plan, failed_stages: set[str]) -> set[str]:
    blocked = set(failed_stages)
    changed = True
    while changed:
        changed = False
        for node in plan.stages:
            if node.stage_id not in blocked and any(parent in blocked for parent in node.parents):
                blocked.add(node.stage_id)
                changed = True
    return blocked


def _compile_test_plan(
    tmp_path: Path,
    *,
    stage_filter: str | None = None,
    overrides: list[str] | None = None,
    execution_defaults: dict | None = None,
):
    experiment, runner_path, execution_path = _write_configs(tmp_path)
    execution = load_execution_config(execution_path)
    execution["defaults"].update(execution_defaults or {})
    return compile_campaign_plan(
        experiment_config_path=experiment,
        runner=load_runner_config(runner_path),
        execution=execution,
        overrides=overrides,
        stage_filter=stage_filter,
    )


def _compile_changed_convert_plans(tmp_path: Path):
    experiment, runner_path, execution_path = _write_configs(tmp_path)
    runner = load_runner_config(runner_path)
    execution = load_execution_config(execution_path)
    baseline = compile_campaign_plan(
        experiment_config_path=experiment,
        runner=runner,
        execution=execution,
        stage_filter="convert",
    )
    config = yaml.safe_load(experiment.read_text())
    config["convert"]["model_path"] = "/models/replacement"
    experiment.write_text(yaml.safe_dump(config))
    changed = compile_campaign_plan(
        experiment_config_path=experiment,
        runner=runner,
        execution=execution,
        stage_filter="convert",
    )
    return baseline, changed


def _record_completed_attempt(
    controller: CampaignController,
    node,
    *,
    handle: JobHandle | None = None,
) -> AttemptSpec:
    adapter = adapter_for_stage(node)
    work_plan = adapter.plan(controller.plan, node)
    item = work_plan.items[0]
    attempt = controller._bind_attempt_to_stage_execution(
        node,
        work_plan,
        adapter.command(
            plan=controller.plan,
            node=node,
            item=item,
            attempt_id="completed-attempt",
            runner=controller.plan.runner,
        ),
    )
    controller.store.save_attempt(attempt, None, JobState.RUNNING.value)
    controller.store.update_attempt_status(
        item.work_id,
        attempt.attempt_id,
        JobStatus(handle=handle, state=JobState.COMPLETED),
    )
    return attempt


def test_controller_shutdown_cancels_active_jobs(tmp_path: Path):
    plan = _compile_test_plan(tmp_path, stage_filter="convert")
    executor = _FakeExecutor()
    controller = CampaignController(plan, executor=executor, poll_interval_seconds=0.01)
    result = controller.run(once=True)
    assert result["halted"] is False
    assert executor.cancelled == []
    assert controller._active

    cancelled = controller.shutdown(reason="test")
    assert cancelled == 1
    assert len(executor.cancelled) == 1
    assert controller._active == {}
    attempts = controller.store.list_attempts("convert")
    assert attempts
    assert attempts[-1]["status"] == JobState.CANCELLED.value


def test_controller_waits_for_parent_job_after_artifact_appears(tmp_path: Path, monkeypatch):
    plan = _compile_test_plan(tmp_path, stage_filter="vllm_stats")
    controller = CampaignController(plan, executor=_FakeExecutor())
    child = next(node for node in plan.stages if node.stage_id == "vllm_stats")
    handle = JobHandle(backend="fake", handle_id="parent", attempt_id="a1")
    controller._active[handle.handle_id] = (handle, "convert:0", "a1")
    monkeypatch.setattr(
        "puzzletron_orchestrator.controller.stage_is_complete",
        lambda _config, _stage_id: True,
    )

    assert not controller._parents_ready(child)
    controller._active.clear()
    assert controller._parents_ready(child)


def test_controller_resubmits_completed_work_when_stage_semantics_change(
    tmp_path: Path, monkeypatch
):
    plan_a, plan_b = _compile_changed_convert_plans(tmp_path)
    executor = _TrackingFakeExecutor()
    controller_a = CampaignController(plan_a, executor=_FakeExecutor())
    controller_b = CampaignController(plan_b, executor=executor)
    identity_a = controller_a._stage_execution_identity(plan_a.stages[0])
    identity_b = controller_b._stage_execution_identity(plan_b.stages[0])
    attempts = [
        {
            "work_id": "convert:0",
            "status": JobState.COMPLETED.value,
            "contract_hash": plan_a.contract_hash,
            "metadata": {"stage_execution_identity": identity_a},
        }
    ]
    monkeypatch.setattr(controller_b.store, "list_attempts", lambda _stage_id=None: attempts)

    assert plan_a.contract_hash == plan_b.contract_hash
    assert identity_a != identity_b
    assert not controller_b._required_work_is_completed(plan_b.stages[0], attempts)
    assert controller_b._submit_stage(plan_b.stages[0])
    assert executor.submitted_stage_ids == ["convert"]
    submitted_attempt = next(iter(executor._attempts.values()))
    assert submitted_attempt.metadata["stage_execution_identity"] == identity_b
    persisted_attempts = CampaignController(plan_b, executor=_FakeExecutor()).store.list_attempts(
        "convert"
    )
    assert persisted_attempts[-1]["metadata"]["stage_execution_identity"] == identity_b


def test_controller_cancels_stale_active_attempt_before_current_resubmission(tmp_path: Path):
    old_plan, plan = _compile_changed_convert_plans(tmp_path)
    old_node = old_plan.stages[0]
    old_controller = CampaignController(old_plan, executor=_FakeExecutor())
    handle = JobHandle(
        backend="fake",
        handle_id="stale-active-handle",
        attempt_id="completed-attempt",
        metadata={"work_id": "convert:0"},
    )
    attempt = _record_completed_attempt(old_controller, old_node, handle=handle)
    old_controller.store.save_attempt(attempt, handle, JobState.RUNNING.value)
    old_controller.store.track_live_job(handle)

    class _FailsIfPolledExecutor(_TrackingFakeExecutor):
        def poll(self, handles):
            raise AssertionError("a stale active attempt must not enter the current poll set")

    executor = _FailsIfPolledExecutor()
    controller = CampaignController(plan, executor=executor)

    with pytest.raises(RuntimeError, match="stale active attempts were cancelled"):
        controller.run(once=True)

    assert executor.cancelled == [handle]
    assert executor.submitted_stage_ids == []
    assert controller._active == {}
    assert controller._failed_stages == set()
    assert controller.store.load_attempt(attempt.work_id, attempt.attempt_id)["status"] == "running"
    assert (
        len(
            list(
                controller.store.events_root.glob(
                    "*_stale_active_attempt_cancellation_requested.json"
                )
            )
        )
        == 1
    )


def test_controller_rejects_overrides_that_differ_from_compiled_plan(tmp_path: Path):
    experiment, runner_path, execution_path = _write_configs(tmp_path)
    compiled_overrides = ["++convert.model_path=/models/compiled"]
    plan = compile_campaign_plan(
        experiment_config_path=experiment,
        runner=load_runner_config(runner_path),
        execution=load_execution_config(execution_path),
        overrides=compiled_overrides,
        stage_filter="convert",
    )
    executor = _TrackingFakeExecutor()
    controller = CampaignController(plan, executor=executor)
    baseline_plan = compile_campaign_plan(
        experiment_config_path=experiment,
        runner=load_runner_config(runner_path),
        execution=load_execution_config(execution_path),
        stage_filter="convert",
    )

    with pytest.raises(ValueError, match="must match the overrides compiled"):
        controller.run(overrides=["++convert.model_path=/models/runtime"], once=True)

    assert executor.submitted_stage_ids == []
    assert controller.store.list_attempts("convert") == []

    result = controller.run(overrides=compiled_overrides, once=True)

    assert result["halted"] is False
    assert executor.submitted_stage_ids == ["convert"]
    submitted_attempt = next(iter(executor._attempts.values()))
    override_index = submitted_attempt.command.argv.index("--override")
    assert submitted_attempt.command.argv[override_index + 1] == compiled_overrides[0]
    assert submitted_attempt.metadata["stage_execution_identity"] == (
        controller._stage_execution_identity(plan.stages[0])
    )
    baseline_identity = CampaignController(
        baseline_plan, executor=_FakeExecutor()
    )._stage_execution_identity(baseline_plan.stages[0])
    assert submitted_attempt.metadata["stage_execution_identity"] != baseline_identity


def test_controller_artifact_settling_deadline_survives_restart(tmp_path: Path, monkeypatch):
    plan = _compile_test_plan(
        tmp_path,
        stage_filter="convert",
        execution_defaults={"artifact_settling_timeout_seconds": 120},
    )
    node = plan.stages[0]
    first_controller = CampaignController(plan, executor=_FakeExecutor())
    now = [1000.0]
    monkeypatch.setattr("puzzletron_orchestrator.controller.time.time", lambda: now[0])
    _record_completed_attempt(first_controller, node)

    now[0] += 120.0
    restarted = CampaignController(plan, executor=_FakeExecutor())

    assert (
        restarted._completed_work_artifact_settling_elapsed(
            node,
            restarted.store.list_attempts(node.stage_id),
        )
        == 120.0
    )


@pytest.mark.parametrize("aggregation_failure", [False, True])
def test_controller_fails_when_completed_work_artifacts_do_not_settle(
    tmp_path: Path, monkeypatch, aggregation_failure: bool
):
    plan = _compile_test_plan(
        tmp_path,
        stage_filter="convert",
        execution_defaults={"artifact_settling_timeout_seconds": 120},
    )
    node = plan.stages[0]
    executor = _TrackingFakeExecutor()
    controller = CampaignController(plan, executor=executor, poll_interval_seconds=60.0)
    now = [1000.0]
    delegate = adapter_for_stage(node)
    monkeypatch.setattr("puzzletron_orchestrator.controller.time.time", lambda: now[0])
    monkeypatch.setattr(
        controller,
        "_interruptible_sleep",
        lambda seconds: now.__setitem__(0, now[0] + seconds),
    )
    _record_completed_attempt(
        controller,
        node,
        handle=JobHandle(
            backend="fake",
            handle_id="completed-handle",
            attempt_id="completed-attempt",
        ),
    )

    class _MissingArtifactsAdapter:
        aggregation_ready = False

        def __getattr__(self, name):
            return getattr(delegate, name)

        def aggregate(self, *, plan, node, work_plan):
            if aggregation_failure and not self.aggregation_ready:
                raise FileNotFoundError("shards are still publishing")
            return {"status": "complete"}

        def validate(self, *, plan, node):
            if aggregation_failure:
                assert self.aggregation_ready
                return ValidatedResult(valid=True, reason="stage outputs present")
            return ValidatedResult(
                valid=False,
                reason="stage outputs missing",
                artifacts=("ckpts/teacher/config.json",),
            )

    missing = _MissingArtifactsAdapter()
    monkeypatch.setattr(
        "puzzletron_orchestrator.controller.adapter_for_stage", lambda _node: missing
    )
    monkeypatch.setattr(
        "puzzletron_orchestrator.controller.stage_is_complete",
        lambda _config, stage_id: controller.store.stage_is_complete(stage_id),
    )

    result = controller.run(max_iterations=3)

    assert result["halted"] is True
    assert result["failed_stages"] == ["convert"]
    assert executor.submitted_stage_ids == []
    assert len(controller.store.list_attempts("convert")) == 1
    phase = "aggregation" if aggregation_failure else "validation"
    event_name = f"stage_{phase}_failed"
    event_paths = list(controller.store.events_root.glob(f"*_{event_name}.json"))
    assert len(event_paths) == 1
    event = json.loads(event_paths[0].read_text())
    assert event["payload"]["stage_id"] == "convert"
    assert event["payload"]["failure_class"] == "timeout_fatal"
    assert event["payload"]["contract_hash"] == plan.contract_hash
    assert event["payload"]["stage_execution_identity"] == controller._stage_execution_identity(
        node
    )
    assert event["payload"]["phase"] == phase
    assert event["payload"]["exception_type"] == (
        "FileNotFoundError" if aggregation_failure else None
    )
    assert event["payload"]["attempt_ids"] == ["completed-attempt"]
    assert event["payload"]["elapsed_seconds"] == 120.0
    assert event["payload"]["timeout_seconds"] == 120.0
    expected_reason = (
        "stage aggregation failed: FileNotFoundError: shards are still publishing"
        if aggregation_failure
        else "stage outputs missing"
    )
    assert event["payload"]["reason"] == expected_reason
    assert event["payload"]["expected_artifacts"] == (
        [] if aggregation_failure else ["ckpts/teacher/config.json"]
    )
    stage_record = controller.store.load_stage_record("convert")
    assert stage_record is not None
    assert stage_record.status == JobState.FAILED.value
    assert stage_record.attempts[0].status == JobState.COMPLETED.value
    assert stage_record.attempts[0].metadata["stage_finalization_failure"]["phase"] == phase

    recovered_executor = _TrackingFakeExecutor()
    missing.aggregation_ready = aggregation_failure
    recovered = CampaignController(plan, executor=recovered_executor)
    recovered_result = recovered.run(once=True)

    assert recovered_result["halted"] is (not aggregation_failure)
    assert recovered_result["failed_stages"] == ([] if aggregation_failure else ["convert"])
    assert recovered_executor.submitted_stage_ids == (
        ["final_report"] if aggregation_failure else []
    )
    assert len(list(controller.store.events_root.glob(f"*_{event_name}.json"))) == 1
    assert recovered.store.stage_is_complete("convert") is aggregation_failure


def test_controller_ignores_failed_record_from_stale_stage_execution(tmp_path: Path):
    old_plan, plan = _compile_changed_convert_plans(tmp_path)
    old_identity = CampaignController(old_plan, executor=_FakeExecutor())._stage_execution_identity(
        old_plan.stages[0]
    )
    executor = _TrackingFakeExecutor()
    controller = CampaignController(plan, executor=executor)
    controller.store.write_stage_record(
        StageRunRecord(
            stage_id="convert",
            status=JobState.FAILED.value,
            attempts=[
                PersistedAttempt(
                    attempt_id="stale-attempt",
                    work_id="convert:0",
                    stage_id="convert",
                    status=JobState.COMPLETED.value,
                    contract_hash=plan.contract_hash,
                    metadata={"stage_execution_identity": old_identity},
                )
            ],
        )
    )

    result = controller.run(once=True)

    assert result["failed_stages"] == []
    assert executor.submitted_stage_ids == ["convert"]


def test_controller_aggregates_completed_work_before_resubmitting(
    tmp_path: Path, monkeypatch, write_terminal_manifest
):
    plan = _compile_test_plan(tmp_path, stage_filter="convert")
    node = plan.stages[0]
    delegate = adapter_for_stage(node)
    item = delegate.plan(plan, node).items[0]
    attempt = delegate.command(
        plan=plan,
        node=node,
        item=item,
        attempt_id="completed-attempt",
        runner=plan.runner,
    )
    executor = _FakeExecutor()
    controller = CampaignController(plan, executor=executor, poll_interval_seconds=0.01)
    attempt = controller._bind_attempt_to_stage_execution(node, delegate.plan(plan, node), attempt)
    controller.store.save_attempt(attempt, None, JobState.COMPLETED.value)

    class _AggregateAdapter:
        def __getattr__(self, name):
            return getattr(delegate, name)

        def aggregate(self, *, plan, node, work_plan):
            _seed_convert_complete(plan.puzzle_dir, write_terminal_manifest, plan.experiment_config)

    aggregate_adapter = _AggregateAdapter()
    monkeypatch.setattr(
        "puzzletron_orchestrator.controller.adapter_for_stage",
        lambda _node: aggregate_adapter,
    )

    result = controller.run(once=True)

    assert result["halted"] is False
    assert not any(attempt.stage_id == "convert" for attempt in executor._attempts.values())
    assert controller.store.stage_is_complete("convert")


def test_controller_preserves_live_job_when_cancel_fails(tmp_path: Path):
    plan = _compile_test_plan(tmp_path, stage_filter="convert")

    class _FailingCancelExecutor(_FakeExecutor):
        def cancel(self, handles: Sequence[JobHandle]) -> None:
            raise RuntimeError("still queued")

    executor = _FailingCancelExecutor()
    controller = CampaignController(plan, executor=executor, poll_interval_seconds=0.01)
    controller.run(once=True)

    assert controller.shutdown(reason="test-failure") == 0
    assert controller.store.list_attempts("convert")[-1]["status"] == JobState.RUNNING.value
    assert len(controller.store.list_live_handles()) == 1


def test_slurm_cancel_batches_job_ids(tmp_path: Path, monkeypatch):
    from puzzletron_orchestrator.executors.slurm import SlurmExecutor
    from puzzletron_orchestrator.schema import (
        ExecutionContract,
        RunnerEnvironment,
        SlurmRunnerConfig,
    )

    calls: list[list[str]] = []

    def _fake_run(argv):
        calls.append(list(argv))

        class _Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return _Result()

    monkeypatch.setattr("puzzletron_orchestrator.executors.slurm._run_command", _fake_run)
    executor = SlurmExecutor(
        RunnerEnvironment(
            kind="slurm",
            slurm=SlurmRunnerConfig(account="test"),
            contract=ExecutionContract(repository=str(tmp_path), venv=str(tmp_path / ".venv")),
        ),
        scripts_dir=tmp_path / "sbatch",
    )
    executor.cancel(
        [
            JobHandle(backend="slurm", handle_id="slurm-11", attempt_id="a", metadata={}),
            JobHandle(
                backend="slurm",
                handle_id="slurm-12",
                attempt_id="b",
                metadata={"job_id": "12"},
            ),
        ]
    )
    assert calls == [
        ["scancel", "11", "12"],
        ["scancel", "--full", "-s", "SIGKILL", "11", "12"],
        ["squeue", "-h", "-o", "%A"],
    ]


def test_slurm_cancel_detects_jobs_that_remain_queued(tmp_path: Path, monkeypatch):
    from puzzletron_orchestrator.executors.slurm import SlurmExecutor
    from puzzletron_orchestrator.schema import (
        ExecutionContract,
        RunnerEnvironment,
        SlurmRunnerConfig,
    )

    def _fake_run(argv):
        class _Result:
            returncode = 0
            stdout = "11\n" if argv[0] == "squeue" else ""
            stderr = ""

        return _Result()

    monkeypatch.setattr("puzzletron_orchestrator.executors.slurm._run_command", _fake_run)
    monkeypatch.setattr("puzzletron_orchestrator.executors.slurm.time.sleep", lambda _: None)
    executor = SlurmExecutor(
        RunnerEnvironment(
            kind="slurm",
            slurm=SlurmRunnerConfig(account="test"),
            contract=ExecutionContract(repository=str(tmp_path), venv=str(tmp_path / ".venv")),
        ),
        scripts_dir=tmp_path / "sbatch",
    )
    handle = JobHandle(backend="slurm", handle_id="slurm-11", attempt_id="a", metadata={})
    with pytest.raises(RuntimeError, match="remained queued"):
        executor.cancel([handle])


def test_controller_keyboard_interrupt_cancels_jobs(tmp_path: Path, monkeypatch):
    plan = _compile_test_plan(tmp_path, stage_filter="convert")
    executor = _FakeExecutor()
    controller = CampaignController(plan, executor=executor, poll_interval_seconds=0.01)

    def _raise_interrupt(_seconds: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(
        "puzzletron_orchestrator.controller.time.sleep",
        _raise_interrupt,
    )
    result = controller.run()
    assert result["cancelled"] is True
    assert result["halted"] is True
    assert len(executor.cancelled) == 1
    assert controller.store.list_attempts("convert")[-1]["status"] == JobState.CANCELLED.value


def test_controller_fatal_failure_drains_without_cancelling_siblings(
    tmp_path: Path, monkeypatch, write_terminal_manifest
):
    experiment, runner_path, execution_path = _write_configs(tmp_path)
    experiment_config = yaml.safe_load(experiment.read_text())
    experiment_config["tokenize_data"] = {"enabled": True}
    experiment.write_text(yaml.safe_dump(experiment_config))
    run_dir = tmp_path / "run"
    _seed_convert_complete(run_dir, write_terminal_manifest, experiment_config)
    plan = compile_campaign_plan(
        experiment_config_path=experiment,
        runner=load_runner_config(runner_path),
        execution=load_execution_config(execution_path),
    )
    plan = replace(
        plan,
        stages=tuple(
            node for node in plan.stages if node.stage_id in {"tokenize_data", "vllm_stats"}
        ),
    )

    class _DrainingExecutor(_FakeExecutor):
        def __init__(self) -> None:
            super().__init__()
            self.poll_count = 0
            self.submitted_work_ids = []

        def submit(self, attempt: AttemptSpec) -> JobHandle:
            self.submitted_work_ids.append(attempt.work_id)
            return super().submit(attempt)

        def poll(self, handles: Sequence[JobHandle]) -> list[JobStatus]:
            self.poll_count += 1
            if self.poll_count == 1:
                statuses = [
                    JobStatus(
                        handle=handle,
                        state=JobState.FAILED if index == 0 else JobState.RUNNING,
                        reason="test failure" if index == 0 else None,
                    )
                    for index, handle in enumerate(handles)
                ]
            else:
                statuses = [
                    JobStatus(handle=handle, state=JobState.COMPLETED) for handle in handles
                ]
            for status in statuses:
                if status.state is JobState.COMPLETED and str(
                    status.handle.metadata.get("work_id", "")
                ).startswith("vllm_stats:"):
                    _seed_vllm_stats_complete(
                        run_dir, write_terminal_manifest, plan.experiment_config
                    )
            return statuses

    executor = _DrainingExecutor()
    monkeypatch.setattr(
        "puzzletron_orchestrator.adapters.sharded._run_slurm_aggregate",
        lambda **_kwargs: "fake-merge",
    )
    controller = CampaignController(plan, executor=executor, poll_interval_seconds=0.01)
    result = controller.run(max_iterations=10)

    assert result["halted"] is True
    assert result["cancelled"] is False
    failed_stage = executor.submitted_work_ids[0].split(":", 1)[0]
    assert result["failed_stages"] == [failed_stage]
    assert executor.cancelled == []
    assert any(
        attempt["status"] == JobState.COMPLETED.value
        and attempt["work_id"].split(":", 1)[0] != failed_stage
        for attempt in controller.store.list_attempts()
    )
    blocked = {failed_stage}
    changed = True
    while changed:
        changed = False
        for node in plan.stages:
            if node.stage_id not in blocked and any(parent in blocked for parent in node.parents):
                blocked.add(node.stage_id)
                changed = True
    attempted_stages = {
        attempt["work_id"].split(":", 1)[0] for attempt in controller.store.list_attempts()
    }
    assert not ((blocked - {failed_stage}) & attempted_stages)


def test_controller_fatal_failure_cancels_other_jobs_in_fail_fast_mode(
    tmp_path: Path, write_terminal_manifest
):
    experiment, runner_path, execution_path = _write_configs(tmp_path)
    run_dir = tmp_path / "run"
    _seed_convert_complete(run_dir, write_terminal_manifest, yaml.safe_load(experiment.read_text()))
    plan = compile_campaign_plan(
        experiment_config_path=experiment,
        runner=load_runner_config(runner_path),
        execution=load_execution_config(execution_path),
    )

    class _FailingExecutor(_FakeExecutor):
        def poll(self, handles: Sequence[JobHandle]) -> list[JobStatus]:
            return [
                JobStatus(
                    handle=handle,
                    state=JobState.FAILED if index == 0 else JobState.RUNNING,
                    reason="test failure" if index == 0 else None,
                )
                for index, handle in enumerate(handles)
            ]

    executor = _FailingExecutor()
    controller = CampaignController(plan, executor=executor, poll_interval_seconds=0.01)
    controller._halt_policy = "fail_fast"
    result = controller.run()

    assert result["halted"] is True
    assert result["cancelled"] is False
    assert executor.cancelled
    assert all(
        attempt["status"] in {JobState.FAILED.value, JobState.CANCELLED.value}
        for attempt in controller.store.list_attempts()
    )


def test_controller_failed_ancestor_blocks_descendant_submit(
    tmp_path: Path, monkeypatch, write_terminal_manifest
):
    experiment, runner_path, execution_path, run_dir = _write_sanity_drain_configs(tmp_path)
    _seed_sort_complete(run_dir, write_terminal_manifest, yaml.safe_load(experiment.read_text()))
    upstream = {"convert", "tokenize_data", "width_importance", "sort"}
    monkeypatch.setattr(
        "puzzletron_orchestrator.controller.stage_is_complete",
        lambda config, stage_id: stage_id in upstream or artifacts_are_complete(config, stage_id),
    )
    plan = compile_campaign_plan(
        experiment_config_path=experiment,
        runner=load_runner_config(runner_path),
        execution=load_execution_config(execution_path),
    )
    plan = replace(
        plan,
        stages=tuple(
            node
            for node in plan.stages
            if node.stage_id in {"sort_sanity", "width_sanity", "bypass_sanity"}
        ),
    )

    class _SanityDrainExecutor(_TrackingFakeExecutor):
        def poll(self, handles: Sequence[JobHandle]) -> list[JobStatus]:
            statuses = [
                JobStatus(
                    handle=handle,
                    state=(
                        JobState.FAILED
                        if self._stage_id(handle) == "sort_sanity"
                        else JobState.COMPLETED
                    ),
                    reason=(
                        "sort sanity worker failure"
                        if self._stage_id(handle) == "sort_sanity"
                        else None
                    ),
                )
                for handle in handles
            ]
            if any(
                status.state is JobState.COMPLETED
                and self._stage_id(status.handle) == "bypass_sanity"
                for status in statuses
            ):
                _seed_sanity_complete(
                    run_dir,
                    "bypass_sanity",
                    write_terminal_manifest,
                    plan.experiment_config,
                )
            return statuses

    executor = _SanityDrainExecutor()
    controller = CampaignController(plan, executor=executor, poll_interval_seconds=0.01)
    result = controller.run()

    assert result["halted"] is True
    assert result["failed_stages"] == ["sort_sanity"]
    assert "bypass_sanity" in executor.submitted_stage_ids
    assert "width_sanity" not in executor.submitted_stage_ids
    blocked = _blocked_descendants(plan, {"sort_sanity"})
    assert "width_sanity" in blocked
    assert "bypass_sanity" not in blocked


def test_controller_sigint_during_recovery_cancels_jobs(tmp_path: Path):
    plan = _compile_test_plan(tmp_path, stage_filter="convert")
    first_executor = _FakeExecutor()
    CampaignController(plan, executor=first_executor, poll_interval_seconds=0.01).run(once=True)

    class _InterruptingExecutor(_FakeExecutor):
        def recover(self, handle: JobHandle) -> JobStatus:
            signal.raise_signal(signal.SIGINT)
            raise AssertionError("SIGINT handler should interrupt recovery")

    executor = _InterruptingExecutor()
    controller = CampaignController(plan, executor=executor, poll_interval_seconds=0.01)
    result = controller.run()

    assert result["cancelled"] is True
    assert result["halted"] is True
    assert len(executor.cancelled) == 1
    assert controller.store.list_attempts("convert")[-1]["status"] == JobState.CANCELLED.value


def test_controller_shutdown_flag_cancels_without_keyboard_interrupt(tmp_path: Path, monkeypatch):
    plan = _compile_test_plan(tmp_path, stage_filter="convert")
    executor = _FakeExecutor()
    controller = CampaignController(plan, executor=executor, poll_interval_seconds=0.05)

    def _trip_shutdown(_seconds: float) -> None:
        controller._shutdown_requested = True

    monkeypatch.setattr(
        "puzzletron_orchestrator.controller.time.sleep",
        _trip_shutdown,
    )
    result = controller.run()
    assert result["cancelled"] is True
    assert len(executor.cancelled) == 1
    assert controller.store.list_attempts("convert")[-1]["status"] == JobState.CANCELLED.value
