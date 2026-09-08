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

"""Durable orchestration state on shared storage."""

from __future__ import annotations

import fcntl
import json
import os
import threading
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
    "release_matching_controller_lease",
    "release_controller_lease",
    "reusable_controller_owner_prefix",
]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace one shared orchestration JSON document."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str))
    os.replace(temporary, path)


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

    def __init__(self, path: Path, owner: str, ttl_seconds: int = 120) -> None:
        self.path = path
        self.owner = owner
        self.ttl_seconds = ttl_seconds
        self._mutex = threading.Lock()
        self._heartbeat_stop = threading.Event()
        self._heartbeat: threading.Thread | None = None
        self._lost = False

    @staticmethod
    def _same_file(path: Path, stat: os.stat_result) -> bool:
        try:
            current = path.stat()
        except OSError:
            return False
        return (current.st_dev, current.st_ino) == (stat.st_dev, stat.st_ino)

    def renew(self, *, ttl_seconds: int | None = None) -> bool:
        """Extend an owned lease, returning false if ownership was lost."""

        with self._mutex:
            if self._lost:
                return False
            try:
                descriptor = os.open(self.path, os.O_RDWR)
                with os.fdopen(descriptor, "r+", encoding="utf-8") as stream:
                    fcntl.flock(stream, fcntl.LOCK_EX)
                    owned_stat = os.fstat(stream.fileno())
                    payload = json.load(stream)
                    if payload.get("owner") != self.owner:
                        self._lost = True
                        return False
                    duration = self.ttl_seconds if ttl_seconds is None else ttl_seconds
                    stream.seek(0)
                    stream.truncate()
                    json.dump(
                        {
                            "owner": self.owner,
                            "pid": os.getpid(),
                            "expires": time.time() + duration,
                        },
                        stream,
                        indent=2,
                    )
                    stream.flush()
                    os.fsync(stream.fileno())
                    if not self._same_file(self.path, owned_stat):
                        self._lost = True
                        return False
            except (OSError, ValueError):
                self._lost = True
                return False
            return True

    def start_heartbeat(self) -> None:
        """Renew the lease while controller work blocks between loop iterations."""

        if self._heartbeat is not None:
            return
        interval = max(1.0, min(30.0, self.ttl_seconds / 3))

        def _heartbeat() -> None:
            while not self._heartbeat_stop.wait(interval):
                if not self.renew():
                    return

        self._heartbeat = threading.Thread(
            target=_heartbeat,
            name="puzzletron-controller-lease",
            daemon=True,
        )
        self._heartbeat.start()

    def release(self) -> None:
        self._heartbeat_stop.set()
        heartbeat = self._heartbeat
        if heartbeat is not None and heartbeat is not threading.current_thread():
            heartbeat.join()
        with self._mutex:
            try:
                descriptor = os.open(self.path, os.O_RDWR)
                with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
                    fcntl.flock(stream, fcntl.LOCK_EX)
                    owned_stat = os.fstat(stream.fileno())
                    payload = json.load(stream)
                    if payload.get("owner") == self.owner and self._same_file(
                        self.path, owned_stat
                    ):
                        self.path.unlink()
            except (OSError, ValueError):
                pass


def acquire_controller_lease(
    root: Path,
    owner: str,
    *,
    ttl_seconds: int = 120,
) -> ControllerLease | None:
    """Acquire an exclusive controller lease or return None if held."""

    root.mkdir(parents=True, exist_ok=True)
    lease_path = root / "controller.lock"
    for _attempt in range(3):
        now = time.time()
        try:
            descriptor = os.open(lease_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            try:
                descriptor = os.open(lease_path, os.O_RDWR)
                with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
                    fcntl.flock(stream, fcntl.LOCK_EX)
                    stale_stat = os.fstat(stream.fileno())
                    try:
                        payload = json.load(stream)
                        expires = float(payload["expires"])
                    except (KeyError, TypeError, ValueError):
                        payload = {}
                        if now - stale_stat.st_mtime < ttl_seconds:
                            return None
                        expires = 0.0
                    if expires > now and payload.get("owner") != owner:
                        return None
                    if not ControllerLease._same_file(lease_path, stale_stat):
                        continue
                    lease_path.unlink()
            except OSError:
                continue
            continue
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            json.dump(
                {"owner": owner, "pid": os.getpid(), "expires": now + ttl_seconds},
                stream,
                indent=2,
            )
            stream.flush()
            os.fsync(stream.fileno())
        return ControllerLease(lease_path, owner, ttl_seconds)
    return None


def release_controller_lease(lease: ControllerLease | None) -> None:
    if lease is not None:
        lease.release()


def reusable_controller_owner_prefix(plan_identity: str, scheduler_job_id: str) -> str:
    """Return the lease-owner prefix for one reusable scheduler allocation."""

    return f"reusable-controller:{plan_identity}:{scheduler_job_id}:"


def release_matching_controller_lease(root: Path, *, owner_prefix: str) -> bool:
    """Release a controller lease only when its owner has the exact trusted prefix."""

    lease_path = root / "controller.lock"
    try:
        descriptor = os.open(lease_path, os.O_RDWR)
        with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            owned_stat = os.fstat(stream.fileno())
            payload = json.load(stream)
            owner = payload.get("owner")
            if not isinstance(owner, str) or not owner.startswith(owner_prefix):
                return False
            if not ControllerLease._same_file(lease_path, owned_stat):
                return False
            lease_path.unlink()
    except (OSError, ValueError):
        return False
    return True


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

    def allocation_path(self) -> Path:
        return self.root / "reusable_allocation.json"

    def allocation_result_path(self) -> Path:
        return self.root / "reusable_allocation_result.json"

    def stage_record_path(self, stage_id: str) -> Path:
        return self.root / "stages" / f"{stage_id}.json"

    def attempt_dir(self, work_id: str, attempt_id: str) -> Path:
        path = self.attempts_root / work_id / attempt_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_plan(self, payload: Mapping[str, Any]) -> None:
        _write_json(self.plan_path(), payload)

    def load_plan(self) -> dict[str, Any] | None:
        path = self.plan_path()
        if not path.is_file():
            return None
        return json.loads(path.read_text())

    def append_event(self, event_type: str, payload: Mapping[str, Any]) -> Path:
        stamp = int(time.time() * 1000)
        path = self.events_root / f"{stamp}_{event_type}.json"
        _write_json(path, {"type": event_type, "payload": dict(payload)})
        return path

    def write_snapshot(self, payload: Mapping[str, Any]) -> None:
        _write_json(self.snapshot_path(), payload)

    def load_snapshot(self) -> dict[str, Any] | None:
        path = self.snapshot_path()
        if not path.is_file():
            return None
        return json.loads(path.read_text())

    def write_allocation(self, payload: Mapping[str, Any]) -> None:
        _write_json(self.allocation_path(), payload)

    def load_allocation(self) -> dict[str, Any] | None:
        path = self.allocation_path()
        if not path.is_file():
            return None
        return json.loads(path.read_text())

    def write_allocation_result(self, *, plan_identity: str, result: Mapping[str, Any]) -> None:
        _write_json(
            self.allocation_result_path(),
            {"plan_identity": plan_identity, "result": dict(result)},
        )

    def load_allocation_result(self, *, plan_identity: str) -> dict[str, Any] | None:
        path = self.allocation_result_path()
        if not path.is_file():
            return None
        payload = json.loads(path.read_text())
        if payload.get("plan_identity") != plan_identity:
            return None
        result = payload.get("result")
        return dict(result) if isinstance(result, Mapping) else None

    def clear_allocation_result(self) -> None:
        self.allocation_result_path().unlink(missing_ok=True)

    def write_allocation_input(self, *, plan_identity: str, name: str, contents: str) -> Path:
        if not name or Path(name).name != name:
            raise ValueError("reusable allocation input name must be a file name")
        path = self.root / "reusable_inputs" / plan_identity / name
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
        temporary.write_text(contents)
        os.replace(temporary, path)
        return path

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
        _write_json(path, payload)
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
                try:
                    record = json.loads(record_path.read_text())
                except (OSError, ValueError):
                    continue
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
        _write_json(path, record)

    def write_stage_record(self, record: StageRunRecord) -> None:
        path = self.stage_record_path(record.stage_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(
            path,
            {
                "stage_id": record.stage_id,
                "status": record.status,
                "aggregated": record.aggregated,
                "attempts": [asdict(item) for item in record.attempts],
            },
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
        _write_json(path, payload)

    def untrack_live_job(self, handle_id: str) -> None:
        path = self._live_jobs_path()
        payload = self._read_live_jobs()
        if handle_id not in payload:
            return
        payload.pop(handle_id, None)
        _write_json(path, payload)

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
