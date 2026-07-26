# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import fcntl
import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..identity import cache_key, canonicalize

__all__ = ["EvaluationRequest", "EvaluationResult", "EvaluationCache"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _locked(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(canonicalize(data), indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


@dataclass(frozen=True, kw_only=True)
class EvaluationRequest:
    handler: str
    payload: dict[str, Any]
    settings: dict[str, Any] = field(default_factory=dict)

    @property
    def identity(self) -> str:
        return cache_key(
            "rpc_eval",
            {"handler": self.handler, "payload": self.payload},
            self.settings,
        ).value

    def with_settings(self, **updates: Any) -> "EvaluationRequest":
        settings = dict(self.settings)
        settings.update(updates)
        return EvaluationRequest(handler=self.handler, payload=self.payload, settings=settings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "handler": self.handler,
            "payload": canonicalize(self.payload),
            "settings": canonicalize(self.settings),
            "request_id": self.identity,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationRequest":
        request = cls(
            handler=data["handler"],
            payload=dict(data.get("payload") or {}),
            settings=dict(data.get("settings") or {}),
        )
        request_id = data.get("request_id")
        if request_id is not None and request.identity != request_id:
            raise ValueError(
                f"Stored request id {request_id!r} does not match canonical id "
                f"{request.identity!r}"
            )
        return request


@dataclass(frozen=True, kw_only=True)
class EvaluationResult:
    request_id: str
    metrics: dict[str, Any]
    artifacts: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "metrics": canonicalize(self.metrics),
            "artifacts": canonicalize(self.artifacts),
            "metadata": canonicalize(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationResult":
        return cls(
            request_id=data["request_id"],
            metrics=dict(data.get("metrics") or {}),
            artifacts=dict(data.get("artifacts") or {}),
            metadata=dict(data.get("metadata") or {}),
        )


class EvaluationCache:
    """Durable JSON result cache keyed by :class:`EvaluationRequest` identity.

    The cache stores one JSON file per request identity under ``results/`` and
    updates an ``index.json`` sidecar under the same file lock. The per-result
    files are the source of truth; the index exists only for cheap inspection.
    """

    def __init__(self, root: str | Path):
        root = Path(root)
        if root.suffix == ".json":
            self.root = root.with_suffix("")
            self.index_path = root
            self.lock_path = root.with_suffix(root.suffix + ".lock")
        else:
            self.root = root
            self.index_path = self.root / "index.json"
            self.lock_path = self.root / ".lock"
        self.results_dir = self.root / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, request_id: str) -> Path:
        return self.results_dir / f"{request_id}.json"

    def _legacy_path_for(self, request_id: str) -> Path:
        return self.root / f"{request_id}.json"

    def get(self, request: EvaluationRequest) -> EvaluationResult | None:
        with _locked(self.lock_path):
            path = self.path_for(request.identity)
            if not path.exists():
                path = self._legacy_path_for(request.identity)
            if not path.exists():
                return None
            data = _read_json(path)
            stored_request = data.get("request")
            result_data = data.get("result", data)
            if stored_request is not None:
                request_from_cache = EvaluationRequest.from_dict(stored_request)
                if request_from_cache.identity != request.identity:
                    raise ValueError(
                        f"Cache file {path} is for {request_from_cache.identity}, "
                        f"expected {request.identity}"
                    )
            result = EvaluationResult.from_dict(result_data)
            if result.request_id != request.identity:
                raise ValueError(
                    f"Cache file {path} is for {result.request_id}, expected {request.identity}"
                )
            return result

    def put(self, result: EvaluationResult, request: EvaluationRequest | None = None) -> None:
        if request is not None and result.request_id != request.identity:
            raise ValueError(
                f"Cannot cache result {result.request_id}; expected {request.identity}"
            )
        with _locked(self.lock_path):
            path = self.path_for(result.request_id)
            payload = {"result": result.to_dict()}
            if request is not None:
                payload["request"] = request.to_dict()
            _write_json_atomic(path, payload)
            self._update_index(result, request, path)

    def _update_index(
        self,
        result: EvaluationResult,
        request: EvaluationRequest | None,
        result_path: Path,
    ) -> None:
        index = (
            _read_json(self.index_path)
            if self.index_path.exists()
            else {"version": 1, "results": {}}
        )
        results = dict(index.get("results") or {})
        results[result.request_id] = {
            "request_id": result.request_id,
            "handler": request.handler if request is not None else None,
            "path": str(result_path.relative_to(self.root)),
            "updated_at": _utc_now(),
        }
        index["results"] = results
        _write_json_atomic(self.index_path, index)
