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

"""One durable Slurm allocation for a complete single-node campaign."""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from .adapters.stage_compat import stage_is_complete
from .compiler import plan_to_dict
from .executors.slurm import SlurmExecutor
from .identity import stable_hash
from .schema import (
    AttemptSpec,
    CampaignPlan,
    CommandSpec,
    ExecutionMode,
    JobHandle,
    JobState,
    JobStatus,
    TaskTopology,
)
from .state import (
    CampaignStateStore,
    acquire_controller_lease,
    release_controller_lease,
    release_matching_controller_lease,
    reusable_controller_owner_prefix,
)

if TYPE_CHECKING:
    from .logging import OrchestratorLogger

__all__ = [
    "REUSABLE_PLAN_IDENTITY_ENV",
    "build_reusable_allocation_attempt",
    "reusable_plan_identity",
    "run_reusable_allocation",
]

REUSABLE_PLAN_IDENTITY_ENV = "PUZZLETRON_REUSABLE_PLAN_IDENTITY"


def reusable_plan_identity(plan: CampaignPlan) -> str:
    """Return the identity that prevents incompatible allocation reattachment."""

    return stable_hash(
        {
            "plan": plan_to_dict(plan),
            "experiment_config": {
                key: value for key, value in plan.experiment_config.items() if key != "_runtime"
            },
        },
        prefix="reusable_allocation",
    )


def build_reusable_allocation_attempt(
    plan: CampaignPlan,
    command: Sequence[str],
    *,
    attempt_id: str | None = None,
) -> AttemptSpec:
    """Build the single Slurm attempt that hosts the local campaign controller."""

    if plan.execution_mode is not ExecutionMode.REUSABLE_ALLOCATION:
        raise ValueError("reusable allocation attempt requires reusable_allocation mode")
    if plan.runner.slurm is None:
        raise ValueError("reusable allocation attempt requires a Slurm runner")
    capacity = int(plan.execution_defaults.get("gpus_per_node", 8))
    attempt_id = attempt_id or reusable_plan_identity(plan).rsplit("_", 1)[-1]
    return AttemptSpec(
        attempt_id=attempt_id,
        work_id="campaign:reusable-allocation",
        stage_id="campaign",
        command=CommandSpec(
            argv=tuple(command),
            env={REUSABLE_PLAN_IDENTITY_ENV: reusable_plan_identity(plan)},
            cwd=plan.runner.contract.repository,
            log_path=str(plan.log_dir / f"campaign_allocation_{attempt_id}.log"),
        ),
        allocation_nodes=1,
        allocation_gpus=capacity,
        exclusive=False,
        contract_hash=plan.contract_hash,
        metadata={
            "gpus_per_node": capacity,
            "partition": plan.execution_defaults.get("partition"),
            "reusable_allocation": True,
            "idempotent_submit": True,
        },
        task_topology=TaskTopology(task_count=1, gpus_per_task=capacity),
    )


def _handle_from_payload(payload: Mapping[str, Any] | None) -> JobHandle | None:
    if not isinstance(payload, Mapping):
        return None
    try:
        return JobHandle(
            backend=str(payload["backend"]),
            handle_id=str(payload["handle_id"]),
            attempt_id=str(payload["attempt_id"]),
            metadata=dict(payload.get("metadata") or {}),
        )
    except KeyError:
        return None


def _result_is_complete(plan: CampaignPlan, result: Mapping[str, Any] | None) -> bool:
    if result is None:
        return False
    return (
        result.get("report_status") == "completed"
        and not result.get("halted")
        and all(stage_is_complete(plan.experiment_config, node.stage_id) for node in plan.stages)
    )


def _result_is_terminal_failure(result: Mapping[str, Any] | None) -> bool:
    """Return whether retrying the same plan would only repeat a strict failure."""

    return bool(
        result
        and result.get("halted")
        and result.get("failed_stages")
        and not result.get("cancelled")
    )


def _detached_result(handle: JobHandle, status: JobStatus) -> dict[str, Any]:
    return {
        "allocation_status": status.state.value,
        "allocation_handle": handle.handle_id,
        "allocation_log_paths": list(status.log_paths),
        "detached": True,
        "halted": False,
    }


def run_reusable_allocation(
    plan: CampaignPlan,
    command: Sequence[str],
    *,
    logger: OrchestratorLogger,
    poll_interval_seconds: float,
    once: bool = False,
) -> dict[str, Any]:
    """Submit or reattach to one durable outer allocation and return its result."""

    if plan.execution_mode is not ExecutionMode.REUSABLE_ALLOCATION:
        raise ValueError("campaign is not configured for reusable allocation execution")
    executor = SlurmExecutor(
        plan.runner,
        scripts_dir=plan.puzzle_dir / "orchestration" / "sbatch",
    )
    store = CampaignStateStore(plan.puzzle_dir)
    plan_identity = reusable_plan_identity(plan)
    lease = acquire_controller_lease(
        store.root / "reusable_allocation",
        f"allocation-launcher-{uuid.uuid4()}",
    )
    if lease is None:
        raise RuntimeError("another process is submitting the reusable allocation; retry shortly")
    terminal_result: dict[str, Any] | None = None
    try:
        lease.start_heartbeat()
        conflicting_attempts = [
            str(attempt.get("handle", {}).get("handle_id") or attempt.get("attempt_id"))
            for attempt in store.list_attempts()
            if attempt.get("status")
            in {JobState.PENDING.value, JobState.RUNNING.value, JobState.UNKNOWN.value}
            and isinstance(attempt.get("handle"), Mapping)
            and attempt["handle"].get("backend") != "local"
        ]
        if conflicting_attempts:
            raise RuntimeError(
                "run root has active work from another execution path; resume it before "
                "starting a reusable allocation: " + ", ".join(conflicting_attempts)
            )
        record = store.load_allocation() or {}
        handle = _handle_from_payload(record.get("handle"))
        status = executor.recover(handle) if handle is not None else None
        active = status is not None and status.state in {
            JobState.PENDING,
            JobState.RUNNING,
            JobState.UNKNOWN,
        }
        if active and record.get("plan_identity") != plan_identity:
            raise RuntimeError(
                "an incompatible reusable allocation is still active for this run root"
            )
        completed = store.load_allocation_result(plan_identity=plan_identity)
        if not active and (
            _result_is_complete(plan, completed) or _result_is_terminal_failure(completed)
        ):
            assert completed is not None
            terminal_result = dict(completed)
        elif not active:
            if completed is not None:
                logger.warning(
                    "previous reusable allocation did not complete; submitting a replacement"
                )
            prior_plan_identity = record.get("plan_identity")
            prior_job_id = handle.metadata.get("job_id") if handle is not None else None
            if isinstance(prior_plan_identity, str) and prior_job_id:
                release_matching_controller_lease(
                    store.root,
                    owner_prefix=reusable_controller_owner_prefix(
                        prior_plan_identity, str(prior_job_id)
                    ),
                )
            store.clear_allocation_result()
            plan.log_dir.mkdir(parents=True, exist_ok=True)
            attempt = build_reusable_allocation_attempt(plan, command)
            store.write_allocation(
                {
                    "schema_version": 1,
                    "plan_identity": plan_identity,
                    "submitting_at": time.time(),
                    "attempt": asdict(attempt),
                    "handle": None,
                }
            )
            handle = executor.submit(attempt)
            status = JobStatus(
                handle=handle,
                state=JobState.PENDING,
                log_paths=executor.fetch_logs(handle),
            )
            store.write_allocation(
                {
                    "schema_version": 1,
                    "plan_identity": plan_identity,
                    "submitted_at": time.time(),
                    "handle": asdict(handle),
                }
            )
            logger.submit(
                f"reusable allocation → {handle.handle_id}; "
                f"log={next(iter(status.log_paths), 'unavailable')}"
            )
        else:
            assert handle is not None and status is not None
            logger.running(
                f"reattached to reusable allocation {handle.handle_id}; "
                f"log={next(iter(status.log_paths), 'unavailable')}"
            )
    finally:
        release_controller_lease(lease)

    if terminal_result is not None:
        if _result_is_complete(plan, terminal_result):
            logger.success("campaign is already complete; no allocation submitted")
        else:
            logger.error("campaign already has a terminal failure; no allocation submitted")
        return terminal_result
    assert handle is not None and status is not None
    if once:
        return _detached_result(handle, status)
    last_state: JobState | None = None
    try:
        while True:
            status = executor.recover(handle)
            if status.state is not last_state:
                if status.state is JobState.PENDING:
                    logger.pending(f"reusable allocation {handle.handle_id}")
                elif status.state is JobState.RUNNING:
                    logger.running(f"reusable allocation {handle.handle_id}")
                last_state = status.state
            if status.state in {JobState.PENDING, JobState.RUNNING, JobState.UNKNOWN}:
                time.sleep(poll_interval_seconds)
                continue
            if status.state is JobState.COMPLETED:
                result = store.load_allocation_result(plan_identity=plan_identity)
                if result is None:
                    raise RuntimeError(
                        f"reusable allocation {handle.handle_id} completed without a result"
                    )
                logger.success(f"reusable allocation {handle.handle_id} completed")
                return result
            result = store.load_allocation_result(plan_identity=plan_identity)
            if result is not None:
                return result
            return {
                "allocation_status": status.state.value,
                "allocation_handle": handle.handle_id,
                "allocation_log_paths": list(status.log_paths),
                "reason": status.reason,
                "halted": True,
            }
    except KeyboardInterrupt:
        logger.shutdown(
            f"detached from reusable allocation {handle.handle_id}; rerun the same command to reattach"
        )
        return _detached_result(handle, status)
