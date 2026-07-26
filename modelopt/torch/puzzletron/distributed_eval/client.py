"""Async and blocking clients for the optional distributed evaluator."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Iterable
from concurrent.futures import Future
from typing import Any

from .campaign import Campaign
from .coordinator import AsyncEvaluationCoordinator
from .schema import EvaluationHandle, EvaluationRequest, EvaluationResult


class AsyncEvaluationClient:
    def __init__(self, coordinator: AsyncEvaluationCoordinator):
        self.coordinator = coordinator

    @classmethod
    def from_campaign(cls, campaign_dir: str, **kwargs) -> "AsyncEvaluationClient":
        return cls(AsyncEvaluationCoordinator(Campaign.open(campaign_dir), **kwargs))

    async def submit(self, request: EvaluationRequest) -> EvaluationHandle:
        request_id = await self.coordinator.submit(request)
        return EvaluationHandle(campaign_id=request.campaign_id, request_id=request_id)

    async def submit_many(
        self, requests: Iterable[EvaluationRequest]
    ) -> list[EvaluationHandle]:
        return [await self.submit(request) for request in requests]

    async def wait(
        self,
        handle: EvaluationHandle | str,
        *,
        timeout: float | None = None,
    ) -> EvaluationResult:
        request_id = handle.request_id if isinstance(handle, EvaluationHandle) else handle
        return await self.coordinator.wait(request_id, timeout=timeout)

    async def as_completed(
        self,
        handles: Iterable[EvaluationHandle],
    ) -> AsyncIterator[EvaluationResult]:
        tasks = [asyncio.create_task(self.wait(handle)) for handle in handles]
        for future in asyncio.as_completed(tasks):
            yield await future

    def lookup(self, handle: EvaluationHandle | str) -> EvaluationResult | None:
        request_id = handle.request_id if isinstance(handle, EvaluationHandle) else handle
        return self.coordinator.lookup(request_id)

    async def cancel(self, handle: EvaluationHandle | str) -> bool:
        request_id = handle.request_id if isinstance(handle, EvaluationHandle) else handle
        return await self.coordinator.cancel(request_id)

    def status(self) -> dict[str, Any]:
        return self.coordinator.status()

    async def close(self) -> None:
        await self.coordinator.close()

    async def __aenter__(self) -> "AsyncEvaluationClient":
        await self.coordinator.start()
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.close()


class EvaluationClient:
    """Blocking facade backed by one persistent asyncio loop thread."""

    def __init__(self, campaign_dir: str, **kwargs):
        self._campaign_dir = campaign_dir
        self._kwargs = kwargs
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._client = self._call(self._make_client())

    @classmethod
    def from_campaign(cls, campaign_dir: str, **kwargs) -> "EvaluationClient":
        return cls(campaign_dir, **kwargs)

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _make_client(self) -> AsyncEvaluationClient:
        client = AsyncEvaluationClient.from_campaign(self._campaign_dir, **self._kwargs)
        await client.coordinator.start()
        return client

    def _call(self, coroutine, *, timeout: float | None = None):
        future: Future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        return future.result(timeout=timeout)

    def submit(self, request: EvaluationRequest) -> EvaluationHandle:
        return self._call(self._client.submit(request))

    def evaluate(
        self,
        request: EvaluationRequest,
        *,
        timeout: float | None = None,
    ) -> EvaluationResult:
        handle = self.submit(request)
        return self._call(self._client.wait(handle, timeout=timeout), timeout=timeout)

    def batch(
        self,
        requests: Iterable[EvaluationRequest],
        *,
        timeout: float | None = None,
    ) -> list[EvaluationResult]:
        requests = list(requests)

        async def run_batch():
            handles = await self._client.submit_many(requests)
            return await asyncio.gather(
                *(self._client.wait(handle, timeout=timeout) for handle in handles)
            )

        return self._call(run_batch(), timeout=timeout)

    def lookup(self, handle: EvaluationHandle | str) -> EvaluationResult | None:
        return self._client.lookup(handle)

    def cancel(self, handle: EvaluationHandle | str) -> bool:
        return self._call(self._client.cancel(handle))

    def status(self) -> dict[str, Any]:
        return self._client.status()

    def close(self) -> None:
        if not self._loop.is_running():
            return
        self._call(self._client.close())
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=10)
        self._loop.close()

    def __enter__(self) -> "EvaluationClient":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
