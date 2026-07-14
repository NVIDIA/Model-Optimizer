"""Shared-filesystem worker discovery and heartbeat records."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from .schema import CampaignManifest, WorkerRecord
from .storage import atomic_write_json, read_json


class WorkerRegistry:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def path_for(self, worker_id: str, boot_id: str) -> Path:
        return self.directory / f"{worker_id}_{boot_id}.json"

    def publish(self, record: WorkerRecord) -> Path:
        path = self.path_for(record.worker_id, record.boot_id)
        atomic_write_json(path, record)
        return path

    def remove(self, worker_id: str, boot_id: str) -> None:
        self.path_for(worker_id, boot_id).unlink(missing_ok=True)

    def list_workers(
        self,
        manifest: CampaignManifest,
        *,
        stale_seconds: float = 45.0,
        include_stale: bool = False,
    ) -> list[WorkerRecord]:
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(seconds=stale_seconds)
        workers: list[WorkerRecord] = []
        for path in sorted(self.directory.glob("*.json")):
            try:
                record = WorkerRecord.model_validate(read_json(path))
            except (OSError, ValueError):
                continue
            if record.campaign_id != manifest.campaign_id:
                continue
            if record.parallelism != manifest.parallelism:
                continue
            if not include_stale and record.heartbeat_at < cutoff:
                continue
            workers.append(record)
        by_worker: dict[str, WorkerRecord] = {}
        for record in workers:
            previous = by_worker.get(record.worker_id)
            if previous is None or record.heartbeat_at > previous.heartbeat_at:
                by_worker[record.worker_id] = record
        return sorted(by_worker.values(), key=lambda record: (record.worker_id, record.boot_id))

    def cleanup_stale(self, *, stale_seconds: float = 45.0) -> list[Path]:
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=stale_seconds)
        removed: list[Path] = []
        for path in self.directory.glob("*.json"):
            try:
                record = WorkerRecord.model_validate(read_json(path))
            except (OSError, ValueError):
                continue
            if record.heartbeat_at < cutoff:
                path.unlink(missing_ok=True)
                removed.append(path)
        return removed
