"""Atomic Lustre-backed persistence for evaluation campaigns."""

from __future__ import annotations

import errno
import fcntl
import json
import math
import os
import secrets
import tempfile
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .identity import canonicalize
from .schema import (
    AttemptRecord,
    CacheWriteStatus,
    CampaignManifest,
    EvaluationRequest,
    EvaluationResult,
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def atomic_write_json(path: Path, value: Any, *, mode: int = 0o644) -> None:
    """Write JSON next to ``path`` and atomically publish it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(canonicalize(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.tmp.{os.getpid()}.",
        suffix=f".{uuid.uuid4().hex}",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError as error:
            if error.errno not in {errno.EINVAL, errno.ENOTSUP}:
                raise
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def file_lock(path: Path, *, blocking: bool = True) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        fcntl.flock(stream.fileno(), flags)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _numbers_close(left: Any, right: Any, *, atol: float, rtol: float) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if math.isnan(float(left)) and math.isnan(float(right)):
            return True
        return math.isclose(float(left), float(right), abs_tol=atol, rel_tol=rtol)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _numbers_close(left[key], right[key], atol=atol, rtol=rtol) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _numbers_close(a, b, atol=atol, rtol=rtol) for a, b in zip(left, right)
        )
    return left == right


class CampaignStorage:
    """Filesystem source of truth for one distributed evaluation campaign."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.manifest_path = self.root / "manifest.json"
        self.token_path = self.root / "auth-token"
        self.coordinator_lock_path = self.root / "coordinator.lock"
        self.registry_dir = self.root / "registry"
        self.request_dir = self.root / "journal" / "requests"
        self.attempt_dir = self.root / "journal" / "attempts"
        self.cancel_dir = self.root / "journal" / "cancellations"
        self.terminal_dir = self.root / "journal" / "terminal"
        self.results_dir = self.root / "results"
        self.conflicts_dir = self.root / "conflicts"
        self.summaries_dir = self.root / "summaries"

    def create(self, manifest: CampaignManifest) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for directory in (
            self.registry_dir,
            self.request_dir,
            self.attempt_dir,
            self.cancel_dir,
            self.terminal_dir,
            self.results_dir,
            self.conflicts_dir,
            self.summaries_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        with file_lock(self.root / ".initialize.lock"):
            if self.manifest_path.exists():
                existing = CampaignManifest.model_validate(read_json(self.manifest_path))
                if existing.campaign_id != manifest.campaign_id:
                    raise ValueError(
                        f"Campaign directory {self.root} already contains "
                        f"{existing.campaign_id}, not {manifest.campaign_id}"
                    )
            else:
                atomic_write_json(self.manifest_path, manifest)
            if not self.token_path.exists():
                token = secrets.token_urlsafe(48)
                self.token_path.write_text(token + "\n", encoding="utf-8")
                os.chmod(self.token_path, 0o600)

    def load_manifest(self) -> CampaignManifest:
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Campaign manifest does not exist: {self.manifest_path}")
        return CampaignManifest.model_validate(read_json(self.manifest_path))

    def read_token(self) -> str:
        return self.token_path.read_text(encoding="utf-8").strip()

    def request_path(self, request_id: str) -> Path:
        return self.request_dir / request_id[:2] / f"{request_id}.json"

    def result_path(self, request_id: str) -> Path:
        return self.results_dir / request_id[:2] / f"{request_id}.json"

    def put_request(self, request: EvaluationRequest) -> None:
        path = self.request_path(request.request_id)
        with file_lock(path.with_suffix(".lock")):
            if path.exists():
                existing = EvaluationRequest.from_wire(read_json(path))
                if existing.request_id != request.request_id:
                    raise ValueError(f"Request collision at {path}")
                return
            atomic_write_json(path, request.to_wire())

    def get_request(self, request_id: str) -> EvaluationRequest | None:
        path = self.request_path(request_id)
        return EvaluationRequest.from_wire(read_json(path)) if path.is_file() else None

    def iter_requests(self) -> Iterator[EvaluationRequest]:
        if not self.request_dir.exists():
            return
        for path in sorted(self.request_dir.glob("*/*.json")):
            yield EvaluationRequest.from_wire(read_json(path))

    def get_result(self, request_id: str) -> EvaluationResult | None:
        path = self.result_path(request_id)
        return EvaluationResult.model_validate(read_json(path)) if path.is_file() else None

    def put_result(
        self,
        result: EvaluationResult,
        *,
        attempt_id: str,
        atol: float,
        rtol: float,
    ) -> CacheWriteStatus:
        path = self.result_path(result.request_id)
        with file_lock(path.with_suffix(".lock")):
            if not path.exists():
                atomic_write_json(path, result)
                return CacheWriteStatus.WRITTEN
            existing = EvaluationResult.model_validate(read_json(path))
            equivalent = _numbers_close(
                existing.metrics,
                result.metrics,
                atol=atol,
                rtol=rtol,
            ) and _numbers_close(
                existing.counts,
                result.counts,
                atol=atol,
                rtol=rtol,
            )
            if equivalent:
                return CacheWriteStatus.DUPLICATE
            conflict = self.conflicts_dir / result.request_id / f"{attempt_id}.json"
            atomic_write_json(conflict, result)
            return CacheWriteStatus.CONFLICT

    def append_attempt(self, attempt: AttemptRecord) -> Path:
        directory = self.attempt_dir / attempt.request_id
        timestamp = attempt.finished_at or attempt.leased_at
        stamp = timestamp.strftime("%Y%m%dT%H%M%S.%fZ")
        path = directory / f"{stamp}_{attempt.attempt_id}_{attempt.status.value}.json"
        atomic_write_json(path, attempt)
        return path

    def mark_cancelled(self, request_id: str, *, reason: str = "cancelled by caller") -> None:
        atomic_write_json(
            self.cancel_dir / f"{request_id}.json",
            {"request_id": request_id, "reason": reason},
        )

    def is_cancelled(self, request_id: str) -> bool:
        return (self.cancel_dir / f"{request_id}.json").is_file()

    def put_terminal_error(self, request_id: str, value: Any) -> None:
        atomic_write_json(self.terminal_dir / f"{request_id}.json", value)

    def get_terminal_error(self, request_id: str) -> dict[str, Any] | None:
        path = self.terminal_dir / f"{request_id}.json"
        return read_json(path) if path.is_file() else None

    def summary(self) -> dict[str, int]:
        requests = sum(1 for _ in self.request_dir.glob("*/*.json"))
        results = sum(1 for _ in self.results_dir.glob("*/*.json"))
        cancellations = sum(1 for _ in self.cancel_dir.glob("*.json"))
        terminal = sum(1 for _ in self.terminal_dir.glob("*.json"))
        conflicts = sum(1 for _ in self.conflicts_dir.glob("*/*.json"))
        return {
            "requests": requests,
            "results": results,
            "pending": max(0, requests - results - cancellations - terminal),
            "cancelled": cancellations,
            "terminal_failures": terminal,
            "conflicts": conflicts,
        }

    def rebuild_summary(self) -> Path:
        path = self.summaries_dir / "status.json"
        atomic_write_json(path, self.summary())
        return path
