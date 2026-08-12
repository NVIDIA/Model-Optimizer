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

"""Durable campaign controller loop."""

from __future__ import annotations

import json
import signal
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from .adapters.base import ExecutionIdentityProjectionUnavailable
from .adapters.post_mip import ManualInputRequired
from .adapters.registry import adapter_for_stage
from .adapters.stage_compat import stage_is_complete
from .compiler import plan_to_dict
from .dashboard import StageView, format_duration, progress_eta, progress_fraction
from .executors import BareMetalSSHExecutor, Executor, LocalExecutor, SlurmExecutor
from .identity import stable_hash
from .logging import OrchestratorLogger
from .progress import summarize_stage_artifacts
from .reporting import FinalReportResult, build_final_report_attempt, final_report_paths
from .schema import (
    AttemptSpec,
    CampaignPlan,
    FailureClass,
    FailurePolicy,
    HaltPolicy,
    JobHandle,
    JobState,
    JobStatus,
    StagePlanNode,
    ValidatedResult,
    WorkPlan,
)
from .stages import semantic_stage_config, stage_display_name
from .state import (
    CampaignStateStore,
    PersistedAttempt,
    StageRunRecord,
    acquire_controller_lease,
    release_controller_lease,
)
from .task_topology import resolve_task_topology
from .terminal import InteractiveControlRequest, ShutdownAction, TerminalControls

__all__ = ["CampaignController", "create_executor", "dry_run_plan"]


_ARTIFACT_SETTLING_TIMEOUT_SECONDS = 300.0


def create_executor(plan: CampaignPlan, *, local: bool = False) -> Executor:
    if local:
        return LocalExecutor(plan.runner)
    if plan.runner.kind == "slurm":
        return SlurmExecutor(plan.runner)
    if plan.runner.kind == "baremetal":
        executor = BareMetalSSHExecutor(plan.runner)
        executor.preflight()
        return executor
    raise ValueError(f"Unsupported runner kind: {plan.runner.kind}")


def _stage_dashboard_display_name(
    config: Mapping[str, Any],
    stage_id: str,
    *,
    granularity: str | None = None,
) -> str:
    if stage_id.startswith("post."):
        parts = stage_id.split(".", 2)
        if len(parts) == 3:
            _prefix, flow_id, node_id = parts
            node = (
                (config.get("post_mip") or {})
                .get("flows", {})
                .get(flow_id, {})
                .get("nodes", {})
                .get(node_id, {})
            )
            node_type = str(node.get("type") or "") if isinstance(node, Mapping) else ""
            if node_type == "downstream_evaluation":
                return "Downstream Evaluation"
    return stage_display_name(stage_id, granularity=granularity)


@dataclass
class DryRunSubmission:
    stage_id: str
    work_id: str
    attempt_id: str
    nodes: int
    gpus: int
    gpus_per_node: int
    task_count: int
    gpus_per_task: int
    tasks_per_group: int
    group_count: int
    task_capacity: int
    unused_gpus: int
    launcher: str
    exclusive: bool
    argv: tuple[str, ...]


@dataclass(frozen=True)
class _FinalizationFailure:
    phase: Literal["aggregation", "validation"]
    reason: str
    artifacts: tuple[str, ...] = ()
    exception_type: str | None = None


def dry_run_plan(
    plan: CampaignPlan,
    *,
    overrides: list[str] | None = None,
) -> list[DryRunSubmission]:
    if overrides is not None and tuple(overrides) != plan.overrides:
        raise ValueError(
            "dry-run overrides must match the overrides compiled into the campaign plan"
        )
    submissions: list[DryRunSubmission] = []
    for node in plan.stages:
        adapter = adapter_for_stage(node)
        work_plan = adapter.plan(plan, node)
        for item in work_plan.items:
            attempt_id = str(uuid.uuid4())
            attempt = adapter.command(
                plan=plan,
                node=node,
                item=item,
                attempt_id=attempt_id,
                runner=plan.runner,
                overrides=list(plan.overrides),
            )
            topology = resolve_task_topology(attempt)
            submissions.append(
                DryRunSubmission(
                    stage_id=node.stage_id,
                    work_id=item.work_id,
                    attempt_id=attempt_id,
                    nodes=attempt.allocation_nodes,
                    gpus=attempt.allocation_gpus,
                    gpus_per_node=topology.gpus_per_node,
                    task_count=topology.task_count,
                    gpus_per_task=topology.gpus_per_task,
                    tasks_per_group=topology.tasks_per_group,
                    group_count=topology.group_count,
                    task_capacity=topology.task_capacity,
                    unused_gpus=topology.unused_gpus,
                    launcher=topology.launcher.value,
                    exclusive=attempt.exclusive,
                    argv=attempt.command.argv,
                )
            )
    return submissions


class CampaignController:
    """Scheduler-neutral durable controller for one campaign plan."""

    def __init__(
        self,
        plan: CampaignPlan,
        *,
        executor: Executor | None = None,
        poll_interval_seconds: float = 5.0,
        local: bool = False,
        logger: OrchestratorLogger | None = None,
        terminal_controls: TerminalControls | None = None,
    ) -> None:
        self.plan = plan
        self.store = CampaignStateStore(plan.puzzle_dir)
        (plan.puzzle_dir / "logs").mkdir(parents=True, exist_ok=True)
        self.executor = executor or create_executor(plan, local=local)
        self.poll_interval_seconds = poll_interval_seconds
        self.logger = logger or OrchestratorLogger()
        self.terminal_controls = terminal_controls or TerminalControls(
            output_stream=self.logger.stream
        )
        self._active: dict[str, tuple[JobHandle, str, str]] = {}
        self._last_states: dict[str, JobState] = {}
        self._last_heartbeat = 0.0
        self._campaign_started_monotonic = time.monotonic()
        self._shutdown_requested = False
        self._shutdown_signal: int | None = None
        self._shutting_down = False
        self._interactive_ready = False
        self._failed_stages: set[str] = set()
        self._finalization_failures: dict[str, _FinalizationFailure] = {}
        self._first_completion_observed: dict[str, float] = {}
        self._manual_waiting: ManualInputRequired | None = None
        defaults = dict(plan.execution_defaults or {})
        self._halt_policy = HaltPolicy(str(defaults.get("halt_policy", HaltPolicy.DRAIN.value)))

    def _recover_active_attempts(self) -> None:
        for attempt in self.store.list_attempts():
            if attempt.get("status") not in {
                JobState.RUNNING.value,
                JobState.PENDING.value,
                JobState.UNKNOWN.value,
            }:
                continue
            handle_payload = attempt.get("handle")
            if not isinstance(handle_payload, dict):
                continue
            handle = JobHandle(
                backend=str(handle_payload["backend"]),
                handle_id=str(handle_payload["handle_id"]),
                attempt_id=str(handle_payload["attempt_id"]),
                metadata=dict(handle_payload.get("metadata") or {}),
            )
            status = self.executor.recover(handle)
            self.store.update_attempt_status(
                str(attempt["work_id"]),
                str(attempt["attempt_id"]),
                status,
            )
            if status.state in {JobState.RUNNING, JobState.PENDING, JobState.UNKNOWN}:
                tracked = JobHandle(
                    backend=handle.backend,
                    handle_id=handle.handle_id,
                    attempt_id=handle.attempt_id,
                    metadata={
                        **dict(handle.metadata),
                        "work_id": str(attempt["work_id"]),
                    },
                )
                self._active[handle.handle_id] = (
                    tracked,
                    str(attempt["work_id"]),
                    str(attempt["attempt_id"]),
                )
                self._last_states[handle.handle_id] = status.state
                self.store.track_live_job(tracked)
                message = (
                    f"recovered {attempt['work_id']} as {status.state.value} [{handle.handle_id}]"
                )
                if status.state is JobState.PENDING:
                    self.logger.pending(message)
                elif status.state is JobState.RUNNING:
                    self.logger.running(message)
                else:
                    self.logger.warning(message)
            else:
                self.store.untrack_live_job(handle.handle_id)
                self.logger.warning(
                    f"recovered {attempt['work_id']} as {status.state.value} [{handle.handle_id}]"
                )

    def _parents_ready(self, node: StagePlanNode) -> bool:
        for parent in node.parents:
            # A worker may publish its completion artifact shortly before Slurm
            # reports terminal success. Do not let that early artifact start a
            # dependent stage while the parent allocation is still active.
            if self._stage_is_active(parent):
                return False
            if parent in self._failed_stages:
                return False
            if not stage_is_complete(self.plan.experiment_config, parent):
                return False
        return True

    def _ancestors_failed(self, stage_id: str, *, visited: set[str] | None = None) -> bool:
        if stage_id in self._failed_stages:
            return True
        visited = visited or set()
        if stage_id in visited:
            return False
        visited.add(stage_id)
        node = next((item for item in self.plan.stages if item.stage_id == stage_id), None)
        if node is None:
            return False
        return any(self._ancestors_failed(parent, visited=visited) for parent in node.parents)

    def _ready_nodes(self) -> list[StagePlanNode]:
        ready: list[StagePlanNode] = []
        for node in self.plan.stages:
            # Artifact completeness is authoritative; stale store records must not
            # hide a convert that still owes subblock_library.json.
            if stage_is_complete(self.plan.experiment_config, node.stage_id):
                continue
            if node.stage_id in self._failed_stages:
                continue
            if self._ancestors_failed(node.stage_id):
                continue
            if not self._parents_ready(node):
                continue
            ready.append(node)
        return ready

    def _should_fail_fast(self) -> bool:
        return HaltPolicy(self._halt_policy) is HaltPolicy.FAIL_FAST

    def _drain_complete(self) -> bool:
        if self._active:
            return False
        if self._ready_nodes():
            return False
        if all(
            stage_is_complete(self.plan.experiment_config, node.stage_id)
            for node in self.plan.stages
        ):
            return True
        return bool(self._failed_stages)

    def _log_completed_stages(self) -> None:
        for node in self.plan.stages:
            if not self._stage_is_active(node.stage_id) and stage_is_complete(
                self.plan.experiment_config, node.stage_id
            ):
                self.logger.skip(f"{node.stage_id}: completion artifacts validated")

    def _required_completed_attempts(
        self,
        node: StagePlanNode,
        attempts: list[dict[str, Any]],
    ) -> list[dict[str, Any]] | None:
        try:
            work_plan = adapter_for_stage(node).plan(self.plan, node)
            stage_execution_identity = self._stage_execution_identity(node, work_plan)
        except ExecutionIdentityProjectionUnavailable:
            return None
        completed: list[dict[str, Any]] = []
        for item in work_plan.items:
            matches = [
                attempt
                for attempt in attempts
                if attempt.get("work_id") == item.work_id
                and attempt.get("status") == JobState.COMPLETED.value
                and attempt.get("contract_hash") == self.plan.contract_hash
                and isinstance(attempt.get("metadata"), Mapping)
                and attempt["metadata"].get("stage_execution_identity") == stage_execution_identity
            ]
            if not matches:
                return None

            def _completion_time(attempt: dict[str, Any]) -> float:
                completed_at = attempt.get("completed_at")
                if isinstance(completed_at, (int, float)):
                    return float(completed_at)
                submitted_at = attempt.get("submitted_at")
                return float(submitted_at) if isinstance(submitted_at, (int, float)) else 0.0

            completed.append(
                max(
                    matches,
                    key=_completion_time,
                )
            )
        return completed

    def _stage_execution_identity(
        self,
        node: StagePlanNode,
        work_plan: WorkPlan | None = None,
    ) -> str:
        adapter = adapter_for_stage(node)
        work_plan = work_plan or adapter.plan(self.plan, node)
        compiled_node = next(
            stage
            for stage in plan_to_dict(self.plan)["stages"]
            if stage["stage_id"] == node.stage_id
        )
        payload = {
            "execution_contract_hash": self.plan.contract_hash,
            "semantic_config": semantic_stage_config(self.plan.experiment_config, node.stage_id),
            "compiled_node": compiled_node,
            "root_overrides": list(self.plan.overrides),
            "work_items": [
                {
                    "work_id": item.work_id,
                    "shard_index": item.shard_index,
                    "shard_count": item.shard_count,
                    "gpus_per_instance": item.gpus_per_instance,
                    "local_gpu_ids": list(item.local_gpu_ids),
                    "metadata": dict(item.metadata),
                }
                for item in work_plan.items
            ],
        }
        adapter_projection = adapter.execution_identity_projection(
            plan=self.plan,
            node=node,
            work_plan=work_plan,
        )
        if adapter_projection:
            payload["adapter_projection"] = dict(adapter_projection)
        return stable_hash(payload, prefix=f"{node.stage_id}_execution")

    def _bind_attempt_to_stage_execution(
        self,
        node: StagePlanNode,
        work_plan: WorkPlan,
        attempt: AttemptSpec,
    ) -> AttemptSpec:
        return replace(
            attempt,
            metadata={
                **dict(attempt.metadata),
                "stage_execution_identity": self._stage_execution_identity(node, work_plan),
            },
        )

    def _required_work_is_completed(
        self,
        node: StagePlanNode,
        attempts: list[dict[str, Any]],
    ) -> bool:
        return self._required_completed_attempts(node, attempts) is not None

    def _legacy_completed_attempts(
        self,
        node: StagePlanNode,
        attempts: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        work_ids = {item.work_id for item in adapter_for_stage(node).plan(self.plan, node).items}
        return [
            attempt
            for attempt in attempts
            if attempt.get("work_id") in work_ids
            and attempt.get("status") == JobState.COMPLETED.value
            and attempt.get("contract_hash") == self.plan.contract_hash
            and (
                not isinstance(attempt.get("metadata"), Mapping)
                or "stage_execution_identity" not in attempt["metadata"]
            )
        ]

    def _completed_work_artifact_settling_elapsed(
        self,
        node: StagePlanNode,
        attempts: list[dict[str, Any]],
    ) -> float | None:
        completed = self._required_completed_attempts(node, attempts)
        if completed is None:
            return None
        completed_at: list[float] = []
        for attempt in completed:
            value = attempt.get("completed_at")
            if not isinstance(value, (int, float)):
                return _ARTIFACT_SETTLING_TIMEOUT_SECONDS
            completed_at.append(float(value))
        now = time.time()
        first_observed = self._first_completion_observed.setdefault(node.stage_id, now)
        return max(0.0, now - max(max(completed_at), first_observed))

    def _policy_allows_retry(self, node: StagePlanNode, failure: FailureClass) -> bool:
        if failure in {FailureClass.SUCCESS, FailureClass.CANCELLED}:
            return False
        if failure is FailureClass.OOM:
            return False
        if node.failure_policy is FailurePolicy.STRICT and failure is FailureClass.APPLICATION:
            return False
        return failure in {
            FailureClass.TRANSIENT,
            FailureClass.TIMEOUT_RESUMABLE,
            FailureClass.UNKNOWN,
        }

    def _stage_has_active_or_completed_work(self, node: StagePlanNode) -> bool:
        if stage_is_complete(self.plan.experiment_config, node.stage_id):
            return True
        attempts = self.store.list_attempts(node.stage_id)
        if not attempts:
            return False
        active_for_stage = any(
            handle_id in self._active and self._active[handle_id][2] == attempt["attempt_id"]
            for handle_id in self._active
            for attempt in attempts
        )
        if active_for_stage:
            return True
        running = any(
            attempt.get("status") in {JobState.RUNNING.value, JobState.PENDING.value}
            for attempt in attempts
        )
        if running:
            return True
        if self._required_work_is_completed(node, attempts):
            # Aggregation is attempted before submission in the controller loop.
            # A scheduler-successful attempt must never overlap with a duplicate
            # while distributed filesystems are still publishing its artifacts.
            # The controller loop either validates those outputs or records a
            # bounded settling failure. Historical records without a completion
            # timestamp fail validation immediately because their settling age
            # cannot be established safely.
            return self._completed_work_artifact_settling_elapsed(node, attempts) is not None
        return bool(self._legacy_completed_attempts(node, attempts))

    def _stage_is_active(self, stage_id: str) -> bool:
        return any(work_id.startswith(f"{stage_id}:") for _, work_id, _ in self._active.values())

    def _recover_failed_stages(self) -> None:
        for node in self.plan.stages:
            record = self.store.load_stage_record(node.stage_id)
            if record is None or record.status != JobState.FAILED.value or not record.attempts:
                continue
            try:
                stage_execution_identity = self._stage_execution_identity(node)
            except ExecutionIdentityProjectionUnavailable:
                continue
            current_failure = bool(record and record.attempts) and all(
                attempt.contract_hash == self.plan.contract_hash
                and (attempt.metadata or {}).get("stage_execution_identity")
                == stage_execution_identity
                for attempt in record.attempts
            )
            legacy_incompatibility = bool(record and record.attempts) and all(
                attempt.contract_hash == self.plan.contract_hash
                and (attempt.metadata or {}).get("stage_execution_identity_incompatible") is True
                for attempt in record.attempts
            )
            if not (current_failure or legacy_incompatibility) or stage_is_complete(
                self.plan.experiment_config,
                node.stage_id,
            ):
                continue
            finalization_failures = [
                (attempt.metadata or {}).get("stage_finalization_failure")
                for attempt in record.attempts
            ]
            if all(
                isinstance(failure, Mapping) and failure.get("phase") == "aggregation"
                for failure in finalization_failures
            ):
                failure = finalization_failures[0]
                assert isinstance(failure, Mapping)
                self._finalization_failures[node.stage_id] = _FinalizationFailure(
                    phase="aggregation",
                    reason=str(failure.get("reason") or "stage aggregation failed"),
                    exception_type=(
                        str(failure["exception_type"])
                        if failure.get("exception_type") is not None
                        else None
                    ),
                )
                self.logger.wait(
                    f"{node.stage_id}: recovered aggregation failure; retrying finalization"
                )
                continue
            self._failed_stages.add(node.stage_id)
            self.logger.error(f"{node.stage_id}: recovered terminal stage validation failure")

    def _fail_legacy_completed_attempts(
        self,
        node: StagePlanNode,
        attempts: list[dict[str, Any]],
    ) -> bool:
        legacy_attempts = self._legacy_completed_attempts(node, attempts)
        if not legacy_attempts:
            return False
        try:
            stage_execution_identity = self._stage_execution_identity(node)
        except ExecutionIdentityProjectionUnavailable:
            return False
        persisted_attempts = [
            PersistedAttempt(
                attempt_id=attempt["attempt_id"],
                work_id=attempt["work_id"],
                stage_id=node.stage_id,
                status=attempt.get("status", JobState.COMPLETED.value),
                contract_hash=attempt["contract_hash"],
                metadata={
                    **dict(attempt.get("metadata") or {}),
                    "stage_execution_identity_incompatible": True,
                },
            )
            for attempt in legacy_attempts
        ]
        reason = (
            "completed attempt metadata predates stage execution identities; "
            "refusing automatic resubmission"
        )
        self._failed_stages.add(node.stage_id)
        self.store.write_stage_record(
            StageRunRecord(
                stage_id=node.stage_id,
                status=JobState.FAILED.value,
                attempts=persisted_attempts,
                aggregated=False,
            )
        )
        self.store.append_event(
            "stage_execution_identity_incompatible",
            {
                "stage_id": node.stage_id,
                "failure_class": FailureClass.CONFIG.value,
                "contract_hash": self.plan.contract_hash,
                "stage_execution_identity": stage_execution_identity,
                "attempt_ids": [attempt.attempt_id for attempt in persisted_attempts],
                "incompatibility": "missing_stage_execution_identity",
                "reason": reason,
            },
        )
        self.logger.error(f"{node.stage_id}: {reason}")
        return True

    def _available_nodes(self) -> int | None:
        slurm = self.plan.runner.slurm
        if slurm is None or slurm.max_nodes is None:
            return None
        active_attempt_ids = {entry[2] for entry in self._active.values()}
        active_nodes = 0
        for attempt in self.store.list_attempts():
            if attempt.get("attempt_id") not in active_attempt_ids:
                continue
            active_nodes += int((attempt.get("allocation") or {}).get("nodes", 0))
        return max(0, slurm.max_nodes - active_nodes)

    def _submit_stage(self, node: StagePlanNode) -> bool:
        if self._stage_has_active_or_completed_work(node):
            return False
        adapter = adapter_for_stage(node)
        adapter.prepare_execution_identity_projection(plan=self.plan, node=node)
        work_plan = adapter.plan(self.plan, node)
        handles: list[tuple[JobHandle, str, str]] = []
        available_nodes = self._available_nodes()
        self.logger.stage(
            f"{node.stage_id}: {node.strategy.value}, {len(work_plan.items)} work item(s), "
            f"{node.gpus_per_instance} GPU(s)/instance"
        )
        for item in work_plan.items:
            prior = [
                attempt
                for attempt in self.store.list_attempts(node.stage_id)
                if attempt.get("work_id") == item.work_id
            ]
            resume = adapter.inspect_resume(
                plan=self.plan,
                node=node,
                item=item,
                attempts=prior,
            )
            if resume.skip and stage_is_complete(self.plan.experiment_config, node.stage_id):
                self.logger.success(f"{item.work_id}: already complete, skipping")
                continue
            attempt_id = str(uuid.uuid4())
            attempt = self._bind_attempt_to_stage_execution(
                node,
                work_plan,
                adapter.command(
                    plan=self.plan,
                    node=node,
                    item=item,
                    attempt_id=attempt_id,
                    runner=self.plan.runner,
                    overrides=list(self.plan.overrides),
                ),
            )
            if available_nodes is not None and attempt.allocation_nodes > available_nodes:
                self.logger.wait(
                    f"{node.stage_id}: node budget exhausted; "
                    "remaining work will be submitted later"
                )
                break
            handle = self.executor.submit(attempt)
            tracked = JobHandle(
                backend=handle.backend,
                handle_id=handle.handle_id,
                attempt_id=handle.attempt_id,
                metadata={**dict(handle.metadata), "work_id": item.work_id},
            )
            self.store.save_attempt(attempt, tracked, JobState.RUNNING.value)
            self.store.track_live_job(tracked)
            self.store.append_event(
                "attempt_submitted",
                {
                    "stage_id": node.stage_id,
                    "work_id": item.work_id,
                    "attempt_id": attempt_id,
                    "handle_id": handle.handle_id,
                },
            )
            handles.append((tracked, item.work_id, attempt_id))
            partition = handle.metadata.get("partition")
            placement = f"{attempt.allocation_nodes} node(s), {attempt.allocation_gpus} GPU(s)"
            if partition:
                placement += f", partition={partition}"
            self.logger.submit(
                f"{item.work_id} → {handle.handle_id} ({placement}); log={attempt.command.log_path}"
            )
            if available_nodes is not None:
                available_nodes -= attempt.allocation_nodes
        for handle, work_id, attempt_id in handles:
            self._active[handle.handle_id] = (handle, work_id, attempt_id)
            self._last_states[handle.handle_id] = JobState.RUNNING
        return bool(handles)

    def _wait_for_manual_input(
        self,
        node: StagePlanNode,
        request: ManualInputRequired,
    ) -> bool:
        self._manual_waiting = request
        self.logger.wait(
            f"{node.stage_id}: manual review is ready; write manual_decision.json "
            "and rerun the controller"
        )
        return False

    def _finalize_stage(self, node: StagePlanNode) -> bool:
        adapter = adapter_for_stage(node)
        work_plan = adapter.plan(self.plan, node)
        self.logger.stage(f"{node.stage_id}: aggregating and validating outputs")
        try:
            aggregate = adapter.aggregate(plan=self.plan, node=node, work_plan=work_plan)
        except ManualInputRequired as request:
            if not self.terminal_controls.enabled:
                return self._wait_for_manual_input(node, request)
            selected = self.terminal_controls.choose_revisions(request.prompt, request.revision_ids)
            decision_path = (
                self.plan.puzzle_dir
                / "artifacts"
                / "post_mip"
                / "nodes"
                / request.node_id
                / "manual_decision.json"
            )
            temporary = Path(str(decision_path) + ".tmp")
            temporary.write_text(
                json.dumps(
                    {
                        "revision_ids": selected,
                        "execution_identity": request.execution_identity,
                    },
                    indent=2,
                )
                + "\n"
            )
            temporary.replace(decision_path)
            try:
                aggregate = adapter.aggregate(plan=self.plan, node=node, work_plan=work_plan)
            except ManualInputRequired as request:
                return self._wait_for_manual_input(node, request)
            except (OSError, ValueError, RuntimeError) as error:
                return self._record_stage_aggregation_failure(
                    node,
                    error,
                )
        except (OSError, ValueError, RuntimeError) as error:
            return self._record_stage_aggregation_failure(
                node,
                error,
            )
        validation = adapter.validate(plan=self.plan, node=node)
        if not validation.valid:
            return self._record_stage_validation_failure(node, validation)
        self._finalization_failures.pop(node.stage_id, None)
        attempts = self._persisted_stage_attempts(node)
        self.store.write_stage_record(
            StageRunRecord(
                stage_id=node.stage_id,
                status=JobState.COMPLETED.value,
                attempts=attempts,
                aggregated=aggregate is not None,
            )
        )
        self.store.append_event("stage_completed", {"stage_id": node.stage_id})
        self.logger.success(
            f"{node.stage_id} complete; artifacts={', '.join(validation.artifacts) or 'validated'}"
        )
        return True

    def _record_stage_validation_failure(
        self,
        node: StagePlanNode,
        validation: ValidatedResult,
    ) -> bool:
        self._finalization_failures[node.stage_id] = _FinalizationFailure(
            phase="validation",
            reason=validation.reason,
            artifacts=validation.artifacts,
        )
        self.logger.warning(f"{node.stage_id}: {validation.reason}")
        return False

    def _record_stage_aggregation_failure(
        self,
        node: StagePlanNode,
        error: OSError | ValueError | RuntimeError,
    ) -> bool:
        reason = f"stage aggregation failed: {type(error).__name__}: {error}"
        self._finalization_failures[node.stage_id] = _FinalizationFailure(
            phase="aggregation",
            reason=reason,
            exception_type=type(error).__name__,
        )
        self.logger.warning(f"{node.stage_id}: {reason}")
        return False

    def _persisted_stage_attempts(self, node: StagePlanNode) -> list[PersistedAttempt]:
        stage_execution_identity = self._stage_execution_identity(node)
        return [
            PersistedAttempt(
                attempt_id=attempt["attempt_id"],
                work_id=attempt["work_id"],
                stage_id=node.stage_id,
                status=attempt.get("status", JobState.COMPLETED.value),
                contract_hash=attempt["contract_hash"],
                metadata=dict(attempt["metadata"]),
            )
            for attempt in self.store.list_attempts(node.stage_id)
            if attempt.get("contract_hash") == self.plan.contract_hash
            and isinstance(attempt.get("metadata"), Mapping)
            and attempt["metadata"].get("stage_execution_identity") == stage_execution_identity
        ]

    def _fail_stage_if_artifacts_did_not_settle(
        self,
        node: StagePlanNode,
        attempts: list[dict[str, Any]],
    ) -> bool:
        if node.stage_id in self._failed_stages:
            return True
        elapsed = self._completed_work_artifact_settling_elapsed(node, attempts)
        if elapsed is None or elapsed < _ARTIFACT_SETTLING_TIMEOUT_SECONDS:
            return False
        failure = self._finalization_failures.get(node.stage_id)
        reason = failure.reason if failure is not None else "required artifacts are incomplete"
        expected_artifacts = list(failure.artifacts) if failure is not None else []
        phase = failure.phase if failure is not None else "validation"
        persisted_attempts = [
            replace(
                attempt,
                metadata={
                    **dict(attempt.metadata or {}),
                    "stage_finalization_failure": {
                        "phase": phase,
                        "reason": reason,
                        "exception_type": (failure.exception_type if failure is not None else None),
                    },
                },
            )
            for attempt in self._persisted_stage_attempts(node)
        ]
        self._failed_stages.add(node.stage_id)
        self.store.write_stage_record(
            StageRunRecord(
                stage_id=node.stage_id,
                status=JobState.FAILED.value,
                attempts=persisted_attempts,
                aggregated=False,
            )
        )
        event_type = (
            "stage_aggregation_failed" if phase == "aggregation" else "stage_validation_failed"
        )
        self.store.append_event(
            event_type,
            {
                "stage_id": node.stage_id,
                "phase": phase,
                "exception_type": failure.exception_type if failure is not None else None,
                "failure_class": FailureClass.TIMEOUT_FATAL.value,
                "contract_hash": self.plan.contract_hash,
                "stage_execution_identity": self._stage_execution_identity(node),
                "attempt_ids": [attempt.attempt_id for attempt in persisted_attempts],
                "elapsed_seconds": elapsed,
                "timeout_seconds": _ARTIFACT_SETTLING_TIMEOUT_SECONDS,
                "reason": reason,
                "expected_artifacts": expected_artifacts,
            },
        )
        self.logger.error(
            f"{node.stage_id}: completed work outputs did not settle within "
            f"{_ARTIFACT_SETTLING_TIMEOUT_SECONDS:g}s: {reason}"
        )
        return True

    def _poll_active(self) -> bool:
        if not self._active:
            return False
        handles = [entry[0] for entry in self._active.values()]
        statuses = self.executor.poll(handles)
        halted = False
        for status in statuses:
            handle = status.handle
            work_id, attempt_id = self._active[handle.handle_id][1:]
            self.store.update_attempt_status(work_id, attempt_id, status)
            previous = self._last_states.get(handle.handle_id)
            if status.state is not previous:
                if status.state is JobState.PENDING:
                    self.logger.pending(f"{work_id} [{handle.handle_id}]")
                elif status.state is JobState.RUNNING:
                    self.logger.running(f"{work_id} [{handle.handle_id}]")
                elif status.state is JobState.COMPLETED:
                    self.logger.success(f"{work_id} finished [{handle.handle_id}]")
                elif status.state is JobState.CANCELLED:
                    self.logger.shutdown(f"{work_id} cancelled [{handle.handle_id}]")
                else:
                    detail = status.reason or f"exit_code={status.exit_code}"
                    self.logger.error(
                        f"{work_id} → {status.state.value} [{handle.handle_id}]: {detail}"
                    )
                self._last_states[handle.handle_id] = status.state
            if status.state in {JobState.RUNNING, JobState.PENDING}:
                continue
            if status.state is JobState.UNKNOWN:
                # Transient squeue/sacct blips must not drop the handle — otherwise
                # Ctrl-C later finds nothing to cancel while Slurm jobs keep running.
                if previous is not JobState.UNKNOWN:
                    self.logger.warning(
                        f"{work_id} state unknown; keeping active [{handle.handle_id}]"
                    )
                self._last_states[handle.handle_id] = JobState.UNKNOWN
                continue
            self._active.pop(handle.handle_id, None)
            self._last_states.pop(handle.handle_id, None)
            self.store.untrack_live_job(handle.handle_id)
            if status.state is JobState.CANCELLED:
                self.store.append_event(
                    "attempt_cancelled",
                    {
                        "attempt_id": attempt_id,
                        "work_id": work_id,
                        "reason": status.reason,
                    },
                )
                continue
            if status.state is not JobState.COMPLETED:
                stage_id = work_id.split(":", 1)[0]
                self._failed_stages.add(stage_id)
                if self._should_fail_fast():
                    halted = True
                self.store.append_event(
                    "attempt_failed",
                    {
                        "attempt_id": attempt_id,
                        "work_id": work_id,
                        "reason": status.reason,
                        "exit_code": status.exit_code,
                    },
                )
        return halted

    def _log_paths_for_active(self) -> dict[str, tuple[str, ...]]:
        paths: dict[str, tuple[str, ...]] = {}
        for handle, work_id, attempt_id in self._active.values():
            record = self.store.load_attempt(work_id, attempt_id)
            log_paths = tuple(record.get("log_paths") or ()) if record else ()
            if not log_paths and record:
                command = record.get("command") or {}
                if command.get("log_path"):
                    log_paths = (str(command["log_path"]),)
            if not log_paths:
                log_paths = self.executor.fetch_logs(handle)
            paths[work_id] = log_paths
        return paths

    def _failed_log_paths(self, failed_stages: set[str]) -> dict[str, list[str]]:
        """Return durable log paths for failed attempts, grouped by stage."""

        paths_by_stage: dict[str, list[str]] = {}
        for node in self.plan.stages:
            if node.stage_id not in failed_stages:
                continue
            stage_paths: list[str] = []
            for attempt in self.store.list_attempts(node.stage_id):
                if attempt.get("status") != JobState.FAILED.value:
                    continue
                log_paths = list(attempt.get("log_paths") or ())
                if not log_paths:
                    command = attempt.get("command")
                    if isinstance(command, Mapping) and command.get("log_path"):
                        log_paths = [str(command["log_path"])]
                for path in log_paths:
                    value = str(path)
                    if value not in stage_paths:
                        stage_paths.append(value)
            if stage_paths:
                paths_by_stage[node.stage_id] = stage_paths
        return paths_by_stage

    def _emit_progress_heartbeat(self) -> None:
        if self._shutdown_requested:
            return
        completed = sum(
            not self._stage_is_active(node.stage_id)
            and stage_is_complete(self.plan.experiment_config, node.stage_id)
            for node in self.plan.stages
        )
        pending = sum(state is JobState.PENDING for state in self._last_states.values())
        running = sum(state is JobState.RUNNING for state in self._last_states.values())
        self.logger.wait(
            f"progress {completed}/{len(self.plan.stages)} stages; "
            f"jobs: {running} running, {pending} pending"
        )
        for view in self._stage_views():
            if self._shutdown_requested:
                return
            if view.status not in {"running", "pending"}:
                continue
            self.logger.progress(
                f"{view.stage_id} {view.nodes}n/{view.tasks}t/{view.gpus}g "
                f"{view.progress}; elapsed={format_duration(view.elapsed_seconds)}, "
                f"eta={format_duration(view.eta_seconds, approximate=True)}"
            )

    def _failed_ancestor_ids(self, stage_id: str) -> list[str]:
        """Return failed ancestors in deterministic plan order."""

        by_id = {node.stage_id: node for node in self.plan.stages}
        failed: set[str] = set()
        visited: set[str] = set()

        def visit(current: str) -> None:
            if current in visited:
                return
            visited.add(current)
            node = by_id.get(current)
            if node is None:
                return
            for parent in node.parents:
                if parent in self._failed_stages:
                    failed.add(parent)
                visit(parent)

        visit(stage_id)
        return [node.stage_id for node in self.plan.stages if node.stage_id in failed]

    def _stage_elapsed(self, stage_id: str, *, active: bool) -> float | None:
        attempts = self.store.list_attempts(stage_id)
        starts = [
            float(attempt["submitted_at"])
            for attempt in attempts
            if isinstance(attempt.get("submitted_at"), (int, float))
        ]
        if not starts:
            return None
        start = min(starts)
        if active:
            return max(0.0, time.time() - start)
        ends = [
            float(attempt["completed_at"])
            for attempt in attempts
            if isinstance(attempt.get("completed_at"), (int, float))
        ]
        return max(0.0, max(ends) - start) if ends else None

    def _stage_views(self) -> list[StageView]:
        """Build dashboard rows from durable state, DAG state, and progress artifacts."""

        active_logs = self._log_paths_for_active()
        views = []
        for node in self.plan.stages:
            stage_config = self.plan.experiment_config.get(node.stage_id)
            granularity = (
                stage_config.get("granularity") if isinstance(stage_config, Mapping) else None
            )
            stage_active = self._stage_is_active(node.stage_id)
            completed = not stage_active and stage_is_complete(
                self.plan.experiment_config, node.stage_id
            )
            failed_ancestors = self._failed_ancestor_ids(node.stage_id)
            attempts = self.store.list_attempts(node.stage_id)
            stage_entries = [
                (handle_id, entry)
                for handle_id, entry in self._active.items()
                if entry[1].startswith(f"{node.stage_id}:")
            ]
            active_states = [
                self._last_states.get(handle_id, JobState.UNKNOWN)
                for handle_id, _entry in stage_entries
            ]
            elapsed = self._stage_elapsed(node.stage_id, active=stage_active)
            progress = ""
            current = total = None
            if completed:
                status = "completed"
                progress = "completed"
            elif node.stage_id in self._failed_stages:
                status = "failed"
                failed_attempts = [
                    attempt
                    for attempt in attempts
                    if attempt.get("status") == JobState.FAILED.value
                ]
                reason = failed_attempts[-1].get("reason") if failed_attempts else None
                progress = f"failed: {reason}" if reason else "failed"
            elif stage_active:
                status = (
                    "running"
                    if any(state is JobState.RUNNING for state in active_states)
                    else "pending"
                )
                stage_logs = [
                    path
                    for _handle_id, (_handle, work_id, _attempt_id) in stage_entries
                    for path in active_logs.get(work_id, ())
                ]
                progress = summarize_stage_artifacts(
                    self.plan.puzzle_dir,
                    node.stage_id,
                    config=self.plan.experiment_config,
                    log_paths=stage_logs,
                ) or ("running" if status == "running" else "queued")
                ratio = progress_fraction(progress)
                if ratio is not None:
                    current, total = ratio
            elif failed_ancestors:
                status = "blocked"
                progress = f"blocked by {', '.join(failed_ancestors)}"
            else:
                waiting_for = [
                    parent
                    for parent in node.parents
                    if self._stage_is_active(parent)
                    or not stage_is_complete(self.plan.experiment_config, parent)
                ]
                if waiting_for:
                    status = "waiting"
                    progress = f"waiting for {', '.join(waiting_for)}"
                else:
                    status = "pending"
                    progress = "ready to submit"
            views.append(
                StageView(
                    stage_id=node.stage_id,
                    display_name=_stage_dashboard_display_name(
                        self.plan.experiment_config,
                        node.stage_id,
                        granularity=str(granularity) if granularity is not None else None,
                    ),
                    status=status,
                    nodes=node.nodes,
                    tasks=node.instances,
                    gpus=node.total_gpus,
                    progress=progress,
                    elapsed_seconds=elapsed,
                    eta_seconds=progress_eta(elapsed, current, total),
                    current=current,
                    total=total,
                )
            )
        return views

    def _refresh_dashboard(self, *, drain_pending: bool = False) -> None:
        if not self.logger.live:
            return
        self.logger.update_dashboard(
            self._stage_views(),
            campaign_elapsed=time.monotonic() - self._campaign_started_monotonic,
            drain_pending=drain_pending,
        )

    def _handles_to_cancel(self) -> list[tuple[JobHandle, str, str]]:
        """Collect live handles from memory, live-job registry, and durable attempts."""

        by_id: dict[str, tuple[JobHandle, str, str]] = dict(self._active)
        for handle in self.store.list_live_handles():
            work_id = str(handle.metadata.get("work_id") or handle.attempt_id)
            by_id.setdefault(handle.handle_id, (handle, work_id, handle.attempt_id))
        terminal = {
            JobState.COMPLETED.value,
            JobState.CANCELLED.value,
            JobState.FAILED.value,
        }
        for attempt in self.store.list_attempts():
            if attempt.get("status") in terminal:
                continue
            handle_payload = attempt.get("handle")
            if not isinstance(handle_payload, dict):
                continue
            handle = JobHandle(
                backend=str(handle_payload["backend"]),
                handle_id=str(handle_payload["handle_id"]),
                attempt_id=str(handle_payload["attempt_id"]),
                metadata=dict(handle_payload.get("metadata") or {}),
            )
            by_id[handle.handle_id] = (
                handle,
                str(attempt["work_id"]),
                str(attempt["attempt_id"]),
            )
        return list(by_id.values())

    def _request_shutdown(self, signum: int | None = None) -> None:
        if self._shutting_down or self._shutdown_requested:
            return
        self._shutdown_requested = True
        self._shutdown_signal = signum

    def _interruptible_sleep(self, seconds: float) -> None:
        """Sleep in short slices so Ctrl-C and ``q`` are noticed within ~200ms."""

        deadline = time.monotonic() + max(0.0, seconds)
        while not self._shutdown_requested:
            if self.terminal_controls.poll_quit():
                raise InteractiveControlRequest
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(0.2, remaining))

    def _generate_final_report(self) -> FinalReportResult:
        """Generate the canonical campaign report through the configured executor."""

        attempt = build_final_report_attempt(self.plan, attempt_id=str(uuid.uuid4()))
        fallback_logs = (attempt.command.log_path,) if attempt.command.log_path is not None else ()
        self.logger.stage("generating final campaign report")
        try:
            handle = self.executor.submit(attempt)
        except Exception as exc:  # noqa: BLE001 - reporting must remain nonfatal
            self.logger.error(f"final campaign report submission failed: {exc}")
            return FinalReportResult(status="failed", log_paths=fallback_logs)
        self.logger.submit(f"final_report:0 [{handle.handle_id}]")
        last_state: JobState | None = None
        while True:
            try:
                status = self.executor.poll([handle])[0]
            except Exception as exc:  # noqa: BLE001 - reporting must remain nonfatal
                self.logger.error(f"final campaign report polling failed: {exc}")
                return FinalReportResult(
                    status="failed",
                    log_paths=self.executor.fetch_logs(handle) or fallback_logs,
                )
            log_paths = status.log_paths or self.executor.fetch_logs(handle) or fallback_logs
            if status.state is not last_state:
                if status.state is JobState.PENDING:
                    self.logger.pending(f"final_report:0 [{handle.handle_id}]")
                elif status.state is JobState.RUNNING:
                    self.logger.running(f"final_report:0 [{handle.handle_id}]")
                elif status.state is JobState.UNKNOWN:
                    self.logger.warning(
                        f"final_report:0 scheduler state unavailable [{handle.handle_id}]"
                    )
                last_state = status.state
            if status.state in {JobState.PENDING, JobState.RUNNING, JobState.UNKNOWN}:
                time.sleep(self.poll_interval_seconds)
                continue
            if status.state is not JobState.COMPLETED:
                detail = f": {status.reason}" if status.reason else ""
                self.logger.error(
                    f"final campaign report {status.state.value} [{handle.handle_id}]{detail}"
                )
                return FinalReportResult(status="failed", log_paths=tuple(log_paths))
            report_path, manifest_path = final_report_paths(self.plan)
            missing = [str(path) for path in (report_path, manifest_path) if not path.is_file()]
            if missing:
                self.logger.error(
                    "final campaign report completed without required artifact(s): "
                    + ", ".join(missing)
                )
                return FinalReportResult(status="failed", log_paths=tuple(log_paths))
            self.logger.success(f"final campaign report: {report_path}")
            return FinalReportResult(
                status="completed",
                path=str(report_path),
                manifest_path=str(manifest_path),
                log_paths=tuple(log_paths),
            )

    def _prompt_shutdown_action(self) -> ShutdownAction:
        """Suspend live rendering and collect one interactive exit decision."""

        self.logger.stop_dashboard()
        try:
            action = self.terminal_controls.choose_shutdown()
        except (InteractiveControlRequest, KeyboardInterrupt):
            action = ShutdownAction.CANCEL
        if action is ShutdownAction.CONTINUE:
            self.logger.banner("continuing campaign")
            self.logger.enable_dashboard()
            self._refresh_dashboard()
        return action

    def shutdown(self, *, reason: str = "keyboard interrupt") -> int:
        """Cancel every live executor handle and mark attempts cancelled."""

        if self._shutting_down and not self._active and not self._handles_to_cancel():
            return 0
        self._shutting_down = True
        self._shutdown_requested = True
        previous_sigint = signal.getsignal(signal.SIGINT)
        previous_sigterm = signal.getsignal(signal.SIGTERM)
        # Ignore nested Ctrl-C while scancel runs so teardown can finish.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            entries = self._handles_to_cancel()
            if not entries:
                self.logger.shutdown("no active jobs to cancel")
                self.store.clear_live_jobs()
                return 0
            handles = [entry[0] for entry in entries]
            labels = ", ".join(f"{work_id}[{handle.handle_id}]" for handle, work_id, _ in entries)
            self.logger.shutdown(f"cancelling {len(handles)} job(s): {labels}")
            try:
                self.executor.cancel(handles)
            except Exception as exc:  # noqa: BLE001 - best-effort teardown
                self.logger.error(f"executor cancel raised: {exc}")
                self.logger.error(
                    "jobs may still be running; their durable handles were preserved for retry"
                )
                return 0
            for handle, work_id, attempt_id in entries:
                status = JobStatus(
                    handle=handle,
                    state=JobState.CANCELLED,
                    reason=reason,
                    log_paths=self.executor.fetch_logs(handle),
                )
                self.store.update_attempt_status(work_id, attempt_id, status)
                self.store.untrack_live_job(handle.handle_id)
                self.store.append_event(
                    "attempt_cancelled",
                    {
                        "attempt_id": attempt_id,
                        "work_id": work_id,
                        "reason": reason,
                        "handle_id": handle.handle_id,
                    },
                )
            self._active.clear()
            self._last_states.clear()
            self.store.clear_live_jobs()
            self.logger.shutdown(f"cancelled {len(entries)} job(s); safe to resume later")
            return len(entries)
        finally:
            signal.signal(signal.SIGINT, previous_sigint)
            signal.signal(signal.SIGTERM, previous_sigterm)

    def run(
        self,
        *,
        overrides: list[str] | None = None,
        once: bool = False,
        max_iterations: int | None = None,
    ) -> dict[str, Any]:
        """Run the controller until all stages complete or a fatal failure occurs."""

        if overrides is not None and tuple(overrides) != self.plan.overrides:
            raise ValueError(
                "runtime overrides must match the overrides compiled into the campaign plan"
            )

        iterations = 0
        halted = False
        cancelled = False
        detached = False
        lease = None
        previous_sigint = signal.getsignal(signal.SIGINT)
        previous_sigterm = signal.getsignal(signal.SIGTERM)

        def _on_signal(signum: int, _frame: object | None) -> None:
            if (
                signum == signal.SIGINT
                and self._interactive_ready
                and self.terminal_controls.enabled
            ):
                raise InteractiveControlRequest
            self._request_shutdown(signum)
            # The whole run lifecycle is protected below, including lease acquisition
            # and durable-job recovery. Raising here makes Ctrl-C interrupt a blocking
            # scheduler query immediately without bypassing teardown.
            raise KeyboardInterrupt

        signal.signal(signal.SIGINT, _on_signal)
        signal.signal(signal.SIGTERM, _on_signal)
        try:
            owner = f"controller-{uuid.uuid4()}"
            lease = acquire_controller_lease(self.store.root, owner)
            if lease is None:
                raise RuntimeError("another controller holds the campaign lease")

            self._campaign_started_monotonic = time.monotonic()
            self.logger.enable_dashboard()
            self.store.write_plan(plan_to_dict(self.plan))
            self.logger.banner(
                f"controller starting in blocking mode; backend={self.executor.backend}, "
                f"poll={self.poll_interval_seconds:g}s"
            )
            self.logger.plan(
                f"{len(self.plan.stages)} stage(s), root={self.plan.puzzle_dir}, "
                f"contract={self.plan.contract_hash[:12]}"
            )
            for node in self.plan.stages:
                parents = ", ".join(node.parents) or "none"
                self.logger.plan(
                    f"{node.stage_id}: parents=[{parents}], strategy={node.strategy.value}, "
                    f"instances={node.instances}, total_gpus={node.total_gpus}"
                )
            self._recover_active_attempts()
            self._recover_failed_stages()
            self._log_completed_stages()
            self._refresh_dashboard()
            self.terminal_controls.start()
            self._interactive_ready = True

            while True:
                try:
                    if self._shutdown_requested:
                        cancelled = True
                        halted = True
                        self.shutdown(reason="signal interrupt")
                        break
                    iterations += 1
                    if max_iterations is not None and iterations > max_iterations:
                        break
                    halted = self._poll_active() or halted
                    if halted and self._should_fail_fast():
                        self._refresh_dashboard()
                        self.shutdown(reason="fatal attempt failure")
                        break
                    for node in self.plan.stages:
                        if stage_is_complete(self.plan.experiment_config, node.stage_id):
                            continue
                        if node.stage_id in self._failed_stages:
                            continue
                        stage_attempts = self.store.list_attempts(node.stage_id)
                        if (
                            stage_attempts
                            and not self._stage_is_active(node.stage_id)
                            and self._parents_ready(node)
                        ):
                            if self._required_work_is_completed(node, stage_attempts):
                                finalized = self._finalize_stage(node)
                                if not finalized and self._manual_waiting is None:
                                    self._fail_stage_if_artifacts_did_not_settle(
                                        node, stage_attempts
                                    )
                                if self._manual_waiting is not None:
                                    break
                            else:
                                self._fail_legacy_completed_attempts(node, stage_attempts)
                    if self._manual_waiting is not None:
                        break
                    if self._failed_stages and self._should_fail_fast():
                        halted = True
                        self._refresh_dashboard()
                        self.shutdown(reason="fatal stage validation failure")
                        break
                    for node in self._ready_nodes():
                        if self._shutdown_requested:
                            break
                        self._submit_stage(node)
                    self._refresh_dashboard(
                        drain_pending=bool(
                            self._failed_stages and (self._active or self._ready_nodes())
                        )
                    )
                    if self._shutdown_requested:
                        cancelled = True
                        halted = True
                        self.shutdown(reason="signal interrupt")
                        break
                    if not self._active and all(
                        stage_is_complete(self.plan.experiment_config, node.stage_id)
                        for node in self.plan.stages
                    ):
                        break
                    if self._drain_complete() and self._failed_stages:
                        halted = True
                        break
                    if halted:
                        break
                    if once:
                        break
                    self.store.write_snapshot(
                        {
                            "running_stages": [
                                {"stage_id": node.stage_id} for node in self._ready_nodes()
                            ],
                            "active_handles": list(self._active.keys()),
                        }
                    )
                    now = time.monotonic()
                    if now - self._last_heartbeat >= 30:
                        self._emit_progress_heartbeat()
                        self._last_heartbeat = now
                    self._interruptible_sleep(self.poll_interval_seconds)
                except InteractiveControlRequest:
                    action = self._prompt_shutdown_action()
                    if action is ShutdownAction.CONTINUE:
                        continue
                    if action is ShutdownAction.DETACH:
                        detached = True
                        break
                    cancelled = True
                    halted = True
                    self.shutdown(reason="interactive user request")
                    break
        except KeyboardInterrupt:
            cancelled = True
            halted = True
            if lease is not None and not self._shutting_down:
                signal_name = (
                    signal.Signals(self._shutdown_signal).name
                    if self._shutdown_signal is not None
                    else "Ctrl-C"
                )
                self.logger.shutdown(f"{signal_name} received; cancelling active jobs")
                self.shutdown(reason="keyboard interrupt")
        finally:
            self._interactive_ready = False
            self.terminal_controls.stop()
            if lease is not None and self._shutdown_requested and not self._shutting_down:
                cancelled = True
                halted = True
                self.shutdown(reason="signal interrupt")
            signal.signal(signal.SIGINT, previous_sigint)
            signal.signal(signal.SIGTERM, previous_sigterm)
            if lease is not None:
                release_controller_lease(lease)
            self._refresh_dashboard(drain_pending=bool(self._failed_stages and self._active))
            self.logger.stop_dashboard()
            if lease is not None:
                self.logger.banner("controller lease released")
        selected_stages_complete = all(
            stage_is_complete(self.plan.experiment_config, node.stage_id)
            for node in self.plan.stages
        )
        clean_completion = (
            selected_stages_complete
            and not self._failed_stages
            and not halted
            and not cancelled
            and not detached
            and self._manual_waiting is None
        )
        report_result = (
            self._generate_final_report()
            if clean_completion
            else FinalReportResult(status="skipped")
        )
        if detached:
            self.logger.shutdown(
                "controller detached; jobs remain active and the same command will recover them"
            )
        elif cancelled:
            self.logger.shutdown("campaign stopped by user; rerun the same command to resume")
        elif halted:
            self.logger.error("campaign halted after a stage failure")
        elif self._manual_waiting is not None:
            self.logger.wait("campaign paused for a durable manual-filter decision")
        else:
            self.logger.success("selected campaign plan completed")
        return {
            "completed": [
                node.stage_id
                for node in self.plan.stages
                if stage_is_complete(self.plan.experiment_config, node.stage_id)
            ],
            "failed_stages": sorted(self._failed_stages),
            "failed_log_paths": self._failed_log_paths(self._failed_stages),
            "halted": halted,
            "cancelled": cancelled,
            "detached": detached,
            "waiting_for_manual_input": (
                self._manual_waiting.node_id if self._manual_waiting is not None else None
            ),
            "iterations": iterations,
            **report_result.as_dict(),
        }
