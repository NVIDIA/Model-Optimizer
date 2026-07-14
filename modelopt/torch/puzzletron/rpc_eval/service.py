# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from .cache import EvaluationCache, EvaluationRequest, EvaluationResult
from .metadata import EvaluationCacheSlot, EvaluationMetadataCache

__all__ = [
    "EvaluationBatchHandler",
    "EvaluationHandler",
    "EvaluationService",
    "EvaluationClient",
    "RemoteEvaluationUnsupportedError",
]

EvaluationHandler = Callable[[EvaluationRequest], EvaluationResult | dict]
EvaluationBatchHandler = Callable[[list[EvaluationRequest]], Iterable[EvaluationResult | dict]]


class RemoteEvaluationUnsupportedError(NotImplementedError):
    """Raised when a caller asks this scaffold to open a remote RPC transport."""


class EvaluationService:
    """Cached handler registry used by RPC workers and local smoke runs."""

    def __init__(self, cache_dir: str | Path):
        self.cache = EvaluationCache(cache_dir)
        self.metadata_cache = EvaluationMetadataCache(self.cache.root / "metadata")
        self._handlers: dict[str, EvaluationHandler] = {}
        self._batch_handlers: dict[str, EvaluationBatchHandler] = {}

    def register(
        self,
        name: str,
        handler: EvaluationHandler,
        *,
        batch_handler: EvaluationBatchHandler | None = None,
    ) -> None:
        if name in self._handlers and self._handlers[name] is not handler:
            raise KeyError(f"Evaluation handler {name!r} already registered")
        self._handlers[name] = handler
        if batch_handler is not None:
            if name in self._batch_handlers and self._batch_handlers[name] is not batch_handler:
                raise KeyError(f"Evaluation batch handler {name!r} already registered")
            self._batch_handlers[name] = batch_handler

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        cached = self.cache.get(request)
        if cached is not None:
            return cached
        handler = self._handlers.get(request.handler)
        if handler is None:
            raise KeyError(f"No evaluation handler registered for {request.handler!r}")
        result = self._normalize_result(request, handler(request))
        self.cache.put(result, request)
        return result

    def batch(self, requests: Iterable[EvaluationRequest]) -> list[EvaluationResult]:
        requests = list(requests)
        results: list[EvaluationResult | None] = [None] * len(requests)
        misses_by_handler: dict[str, list[tuple[int, EvaluationRequest]]] = {}
        for index, request in enumerate(requests):
            cached = self.cache.get(request)
            if cached is not None:
                results[index] = cached
            else:
                misses_by_handler.setdefault(request.handler, []).append((index, request))

        for handler_name, misses in misses_by_handler.items():
            handler = self._handlers.get(handler_name)
            if handler is None:
                raise KeyError(f"No evaluation handler registered for {handler_name!r}")
            batch_handler = self._batch_handlers.get(handler_name)
            raw_results = (
                list(batch_handler([request for _, request in misses]))
                if batch_handler is not None
                else [handler(request) for _, request in misses]
            )
            if len(raw_results) != len(misses):
                raise ValueError(
                    f"Evaluation batch handler {handler_name!r} returned "
                    f"{len(raw_results)} results "
                    f"for {len(misses)} requests"
                )
            for (index, request), raw_result in zip(misses, raw_results):
                result = self._normalize_result(request, raw_result)
                self.cache.put(result, request)
                results[index] = result

        return [result for result in results if result is not None]

    def prepare_teacher_hidden_cache(
        self,
        inputs: Any,
        settings: Any | None = None,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> EvaluationCacheSlot:
        return self.metadata_cache.prepare_teacher_hidden(inputs, settings, metadata=metadata)

    def prepare_prefix_cache(
        self,
        inputs: Any,
        settings: Any | None = None,
        *,
        metadata: dict[str, Any] | None = None,
        ttl_seconds: int | None = None,
    ) -> EvaluationCacheSlot:
        return self.metadata_cache.prepare_prefix(
            inputs,
            settings,
            metadata=metadata,
            ttl_seconds=ttl_seconds,
        )

    def evaluate_batch_with_prefix(
        self,
        prefix_inputs: Any,
        requests: Iterable[EvaluationRequest],
        *,
        prefix_settings: Any | None = None,
        metadata: dict[str, Any] | None = None,
        ttl_seconds: int | None = None,
    ) -> list[EvaluationResult]:
        prefix_slot = self.prepare_prefix_cache(
            prefix_inputs,
            prefix_settings,
            metadata=metadata,
            ttl_seconds=ttl_seconds,
        )
        tagged_requests = [
            request.with_settings(prefix_cache=prefix_slot.to_dict())
            for request in requests
        ]
        return self.batch(tagged_requests)

    def _normalize_result(
        self,
        request: EvaluationRequest,
        raw: EvaluationResult | dict,
    ) -> EvaluationResult:
        if isinstance(raw, EvaluationResult):
            result = raw
        elif "metrics" in raw or "artifacts" in raw or "metadata" in raw:
            result = EvaluationResult(
                request_id=str(raw.get("request_id", request.identity)),
                metrics=dict(raw.get("metrics") or {}),
                artifacts=dict(raw.get("artifacts") or {}),
                metadata=dict(raw.get("metadata") or {}),
            )
        else:
            result = EvaluationResult(request_id=request.identity, metrics=dict(raw))
        if result.request_id != request.identity:
            raise ValueError(
                f"Handler {request.handler!r} returned result for {result.request_id}, "
                f"expected {request.identity}"
            )
        return result


class EvaluationClient:
    """Thin client facade with local get-or-submit semantics."""

    def __init__(self, service: EvaluationService):
        self.service = service

    @classmethod
    def local(cls, service: EvaluationService) -> "EvaluationClient":
        return cls(service)

    @classmethod
    def from_registry(cls, registry: str | Path) -> "EvaluationClient":
        raise RemoteEvaluationUnsupportedError(
            "Remote Puzzletron RPC networking is not implemented yet. "
            f"Registry {registry!s} cannot be opened; use EvaluationService with "
            "EvaluationClient.local() "
            "for the in-process fallback."
        )

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        return self.get_or_submit(request)

    def get_or_submit(self, request: EvaluationRequest) -> EvaluationResult:
        return self.service.evaluate(request)

    def batch(self, requests: Iterable[EvaluationRequest]) -> list[EvaluationResult]:
        return self.get_or_submit_batch(requests)

    def get_or_submit_batch(self, requests: Iterable[EvaluationRequest]) -> list[EvaluationResult]:
        return self.service.batch(requests)

    def evaluate_batch_with_prefix(
        self,
        prefix_inputs: Any,
        requests: Iterable[EvaluationRequest],
        *,
        prefix_settings: Any | None = None,
        metadata: dict[str, Any] | None = None,
        ttl_seconds: int | None = None,
    ) -> list[EvaluationResult]:
        return self.service.evaluate_batch_with_prefix(
            prefix_inputs,
            requests,
            prefix_settings=prefix_settings,
            metadata=metadata,
            ttl_seconds=ttl_seconds,
        )
