# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Durable orchestration state on shared storage."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .schema import AttemptSpec, JobHandle, JobState, JobStatus

__all__ = [
    "CampaignStateStore",
    "ControllerLease",
    "PersistedAttempt",
    "StageRunRecord",
    "acquire_controller_lease",
    "release_controller_lease",
]


@dataclass
class PersistedAttempt:
    """One durable attempt record."""

    attempt_id: str
    work_id: str
    stage_id: str
    status: str
    contract_hash: str
    handle: dict[str, Any] | None = None
    exit_code: int | None = None
    reason: str | None = None
    log_paths: tuple[str, ...] = ()
    metadata: dict[str, Any] | None = None


@dataclass
class StageRunRecord:
    """Aggregate stage run state."""

    stage_id: str
    status: str
    attempts: list[PersistedAttempt]
    aggregated: bool = False


class ControllerLease:
    """File-based campaign controller lease."""

    def __init__(self, path: Path, owner: str) -> None:
        self.path = path
        self.owner = owner

    def release(self) -> None:
        if self.path.exists():
            try:
                payload = json.loads(self.path.read_text())
            except (OSError, ValueError):
                payload = {}
            if payload.get("owner") == self.owner:
                self.path.unlink(missing_ok=True)


def acquire_controller_lease(
    root: Path,
    owner: str,
    *,
    ttl_seconds: int = 120,
) -> ControllerLease | None:
    """Acquire an exclusive controller lease or return None if held."""

    root.mkdir(parents=True, exist_ok=True)
    lease_path = root / "controller.lock"
    now = time.time()
    if lease_path.exists():
        try:
            payload = json.loads(lease_path.read_text())
        except (OSError, ValueError):
            payload = {}
        expires = float(payload.get("expires", 0))
        if expires > now and payload.get("owner") != owner:
            return None
    lease_path.write_text(
        json.dumps({"owner": owner, "pid": os.getpid(), "expires": now + ttl_seconds}, indent=2)
    )
    return ControllerLease(lease_path, owner)


def release_controller_lease(lease: ControllerLease | None) -> None:
    if lease is not None:
        lease.release()


class CampaignStateStore:
    """Read/write orchestration artifacts under puzzle_dir/orchestration/."""

    def __init__(self, puzzle_dir: Path) -> None:
        self.root = Path(puzzle_dir) / "orchestration"
        self.attempts_root = self.root / "attempts"
        self.events_root = self.root / "events"
        self.root.mkdir(parents=True, exist_ok=True)
        self.attempts_root.mkdir(parents=True, exist_ok=True)
        self.events_root.mkdir(parents=True, exist_ok=True)

    def plan_path(self) -> Path:
        return self.root / "compiled_plan.json"

    def snapshot_path(self) -> Path:
        return self.root / "controller_snapshot.json"

    def stage_record_path(self, stage_id: str) -> Path:
        return self.root / "stages" / f"{stage_id}.json"

    def attempt_dir(self, work_id: str, attempt_id: str) -> Path:
        path = self.attempts_root / work_id / attempt_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_plan(self, payload: Mapping[str, Any]) -> None:
        self.plan_path().write_text(json.dumps(payload, indent=2, default=str))

    def load_plan(self) -> dict[str, Any] | None:
        path = self.plan_path()
        if not path.is_file():
            return None
        return json.loads(path.read_text())

    def append_event(self, event_type: str, payload: Mapping[str, Any]) -> Path:
        stamp = int(time.time() * 1000)
        path = self.events_root / f"{stamp}_{event_type}.json"
        path.write_text(
            json.dumps(
                {"type": event_type, "payload": dict(payload)},
                indent=2,
                default=str,
            )
        )
        return path

    def write_snapshot(self, payload: Mapping[str, Any]) -> None:
        self.snapshot_path().write_text(json.dumps(payload, indent=2, default=str))

    def load_snapshot(self) -> dict[str, Any] | None:
        path = self.snapshot_path()
        if not path.is_file():
            return None
        return json.loads(path.read_text())

    def save_attempt(self, attempt: AttemptSpec, handle: JobHandle | None, status: str) -> Path:
        directory = self.attempt_dir(attempt.work_id, attempt.attempt_id)
        payload = {
            "attempt_id": attempt.attempt_id,
            "work_id": attempt.work_id,
            "stage_id": attempt.stage_id,
            "status": status,
            "submitted_at": time.time(),
            "contract_hash": attempt.contract_hash,
            "command": {
                "argv": list(attempt.command.argv),
                "env": dict(attempt.command.env),
                "cwd": attempt.command.cwd,
                "log_path": attempt.command.log_path,
            },
            "allocation": {
                "nodes": attempt.allocation_nodes,
                "gpus": attempt.allocation_gpus,
                "exclusive": attempt.exclusive,
            },
            "task_topology": {
                "task_count": attempt.task_topology.task_count,
                "gpus_per_task": attempt.task_topology.gpus_per_task,
                "tasks_per_group": attempt.task_topology.tasks_per_group,
                "launcher": attempt.task_topology.launcher.value,
                "placement": attempt.task_topology.placement,
            },
            "handle": asdict(handle) if handle is not None else None,
            "metadata": dict(attempt.metadata),
        }
        path = directory / "attempt.json"
        path.write_text(json.dumps(payload, indent=2, default=str))
        return path

    def load_attempt(self, work_id: str, attempt_id: str) -> dict[str, Any] | None:
        path = self.attempt_dir(work_id, attempt_id) / "attempt.json"
        if not path.is_file():
            return None
        return json.loads(path.read_text())

    def list_attempts(self, stage_id: str | None = None) -> list[dict[str, Any]]:
        attempts: list[dict[str, Any]] = []
        if not self.attempts_root.is_dir():
            return attempts
        for work_dir in sorted(self.attempts_root.iterdir()):
            if not work_dir.is_dir():
                continue
            for attempt_dir in sorted(work_dir.iterdir()):
                record_path = attempt_dir / "attempt.json"
                if not record_path.is_file():
                    continue
                record = json.loads(record_path.read_text())
                if stage_id is None or record.get("stage_id") == stage_id:
                    attempts.append(record)
        return attempts

    def update_attempt_status(self, work_id: str, attempt_id: str, status: JobStatus) -> None:
        record = self.load_attempt(work_id, attempt_id)
        if record is None:
            return
        record["status"] = status.state.value
        record["exit_code"] = status.exit_code
        record["reason"] = status.reason
        record["log_paths"] = list(status.log_paths)
        if status.state in {
            JobState.COMPLETED,
            JobState.FAILED,
            JobState.CANCELLED,
        }:
            record["completed_at"] = time.time()
        if status.handle is not None:
            record["handle"] = asdict(status.handle)
        path = self.attempt_dir(work_id, attempt_id) / "attempt.json"
        path.write_text(json.dumps(record, indent=2, default=str))

    def write_stage_record(self, record: StageRunRecord) -> None:
        path = self.stage_record_path(record.stage_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "stage_id": record.stage_id,
                    "status": record.status,
                    "aggregated": record.aggregated,
                    "attempts": [asdict(item) for item in record.attempts],
                },
                indent=2,
                default=str,
            )
        )

    def load_stage_record(self, stage_id: str) -> StageRunRecord | None:
        path = self.stage_record_path(stage_id)
        if not path.is_file():
            return None
        payload = json.loads(path.read_text())
        attempts = [PersistedAttempt(**item) for item in payload.get("attempts", [])]
        return StageRunRecord(
            stage_id=payload["stage_id"],
            status=payload["status"],
            attempts=attempts,
            aggregated=bool(payload.get("aggregated", False)),
        )

    def stage_is_complete(self, stage_id: str) -> bool:
        record = self.load_stage_record(stage_id)
        return record is not None and record.status == JobState.COMPLETED.value

    def _live_jobs_path(self) -> Path:
        return self.root / "live_jobs.json"

    def track_live_job(self, handle: JobHandle) -> None:
        """Persist a live executor handle so Ctrl-C can cancel even if memory is empty."""

        path = self._live_jobs_path()
        payload = self._read_live_jobs()
        payload[handle.handle_id] = asdict(handle)
        path.write_text(json.dumps(payload, indent=2, default=str))

    def untrack_live_job(self, handle_id: str) -> None:
        path = self._live_jobs_path()
        payload = self._read_live_jobs()
        if handle_id not in payload:
            return
        payload.pop(handle_id, None)
        path.write_text(json.dumps(payload, indent=2, default=str))

    def clear_live_jobs(self) -> None:
        path = self._live_jobs_path()
        if path.is_file():
            path.unlink()

    def list_live_handles(self) -> list[JobHandle]:
        handles: list[JobHandle] = []
        for payload in self._read_live_jobs().values():
            if not isinstance(payload, Mapping):
                continue
            handles.append(
                JobHandle(
                    backend=str(payload["backend"]),
                    handle_id=str(payload["handle_id"]),
                    attempt_id=str(payload["attempt_id"]),
                    metadata=dict(payload.get("metadata") or {}),
                )
            )
        return handles

    def _read_live_jobs(self) -> dict[str, Any]:
        path = self._live_jobs_path()
        if not path.is_file():
            return {}
        try:
            payload = json.loads(path.read_text())
        except (OSError, ValueError):
            return {}
        return payload if isinstance(payload, dict) else {}
