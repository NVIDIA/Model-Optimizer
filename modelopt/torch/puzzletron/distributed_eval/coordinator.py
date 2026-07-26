"""Restartable push scheduler for distributed evaluation workers."""

from __future__ import annotations

import asyncio
import logging
import random
import uuid
from datetime import datetime, timezone
from typing import Any

from .campaign import Campaign
from .http_transport import AsyncHttpClient, HttpStatusError
from .schema import (
    AttemptRecord,
    AttemptStatus,
    ErrorKind,
    EvaluationError,
    EvaluationRequest,
    EvaluationResult,
    ExecuteResponse,
    WorkerRecord,
    WorkerState,
)

logger = logging.getLogger(__name__)


class AsyncEvaluationCoordinator:
    """Own the active coordinator lease and dispatch journaled requests."""

    def __init__(
        self,
        campaign: Campaign,
        *,
        stale_seconds: float = 45.0,
        connect_timeout_seconds: float = 10.0,
        task_timeout_seconds: float = 7200.0,
        retry_initial_seconds: float = 5.0,
        retry_max_seconds: float = 60.0,
        poll_seconds: float = 1.0,
    ):
        self.campaign = campaign
        self.stale_seconds = stale_seconds
        self.retry_initial_seconds = retry_initial_seconds
        self.retry_max_seconds = retry_max_seconds
        self.poll_seconds = poll_seconds
        self.http = AsyncHttpClient(
            token=campaign.storage.read_token(),
            connect_timeout_seconds=connect_timeout_seconds,
            task_timeout_seconds=task_timeout_seconds,
        )
        self._pending: dict[str, EvaluationRequest] = {}
        self._retry_count: dict[str, int] = {}
        self._retry_at: dict[str, float] = {}
        self._inflight: dict[str, asyncio.Task] = {}
        self._busy_workers: set[tuple[str, str]] = set()
        self._dispatch_task: asyncio.Task | None = None
        self._wake = asyncio.Event()
        self._closed = False
        self._lease_context = None
        self._cache_hits = 0

    async def start(self) -> None:
        if self._dispatch_task is not None:
            return
        self._lease_context = self.campaign.coordinator_lease()
        self._lease_context.__enter__()
        for request in self.campaign.storage.iter_requests():
            if self._is_unfinished(request.request_id):
                self._pending[request.request_id] = request
        self._dispatch_task = asyncio.create_task(self._dispatch_loop())

    def _is_unfinished(self, request_id: str) -> bool:
        storage = self.campaign.storage
        return (
            storage.get_result(request_id) is None
            and storage.get_terminal_error(request_id) is None
            and not storage.is_cancelled(request_id)
        )

    async def submit(self, request: EvaluationRequest) -> str:
        await self.start()
        self.campaign.validate_request(request)
        self.campaign.storage.put_request(request)
        if self.campaign.storage.get_result(request.request_id) is not None:
            self._cache_hits += 1
            return request.request_id
        if self._is_unfinished(request.request_id):
            self._pending[request.request_id] = request
            self._wake.set()
        return request.request_id

    async def cancel(self, request_id: str) -> bool:
        self.campaign.storage.mark_cancelled(request_id)
        self._pending.pop(request_id, None)
        task = self._inflight.get(request_id)
        if task is not None:
            task.cancel()
        self._wake.set()
        return True

    async def _dispatch_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while not self._closed:
            self._reap_finished()
            workers = self.campaign.registry.list_workers(
                self.campaign.manifest,
                stale_seconds=self.stale_seconds,
            )
            idle = [
                worker
                for worker in workers
                if worker.state == WorkerState.IDLE
                and (worker.worker_id, worker.boot_id) not in self._busy_workers
            ]
            random.shuffle(idle)
            now = loop.time()
            request_ids = [
                request_id
                for request_id in sorted(self._pending)
                if self._retry_at.get(request_id, 0.0) <= now
            ]
            dispatched = False
            for worker, request_id in zip(idle, request_ids):
                request = self._pending.pop(request_id)
                if not self._is_unfinished(request_id):
                    continue
                worker_key = (worker.worker_id, worker.boot_id)
                self._busy_workers.add(worker_key)
                task = asyncio.create_task(self._execute(worker, request))
                self._inflight[request_id] = task
                dispatched = True
            if dispatched:
                await asyncio.sleep(0)
                continue
            self._wake.clear()
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=self.poll_seconds)
            except TimeoutError:
                pass

    def _reap_finished(self) -> None:
        for request_id, task in list(self._inflight.items()):
            if not task.done():
                continue
            self._inflight.pop(request_id, None)
            try:
                task.result()
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("Evaluation dispatch failed for %s", request_id)

    async def _execute(self, worker: WorkerRecord, request: EvaluationRequest) -> None:
        worker_key = (worker.worker_id, worker.boot_id)
        attempt_id = uuid.uuid4().hex
        leased = AttemptRecord(
            request_id=request.request_id,
            attempt_id=attempt_id,
            worker_id=worker.worker_id,
            worker_boot_id=worker.boot_id,
            status=AttemptStatus.LEASED,
        )
        self.campaign.storage.append_attempt(leased)
        error: EvaluationError | None = None
        try:
            payload = {"attempt_id": attempt_id, "request": request.to_wire()}
            data = await self.http.request(
                "POST",
                f"{worker.endpoint}/v1/tasks/{request.request_id}/execute",
                payload=payload,
            )
            response = ExecuteResponse.model_validate(data)
            if response.status == "succeeded":
                result = self.campaign.storage.get_result(request.request_id)
                if result is None and response.result is not None:
                    self.campaign.storage.put_result(
                        response.result,
                        attempt_id=attempt_id,
                        atol=self.campaign.manifest.result_atol,
                        rtol=self.campaign.manifest.result_rtol,
                    )
                    result = self.campaign.storage.get_result(request.request_id)
                if result is None:
                    raise RuntimeError("Worker acknowledged success without a durable result")
                self.campaign.storage.append_attempt(
                    leased.model_copy(
                        update={
                            "status": AttemptStatus.SUCCEEDED,
                            "finished_at": datetime.now(timezone.utc),
                        }
                    )
                )
                self._retry_count.pop(request.request_id, None)
                self._retry_at.pop(request.request_id, None)
                return
            if response.status == "cancelled":
                self.campaign.storage.mark_cancelled(
                    request.request_id,
                    reason=f"worker {worker.worker_id} observed cancellation",
                )
                self.campaign.storage.append_attempt(
                    leased.model_copy(
                        update={
                            "status": AttemptStatus.CANCELLED,
                            "finished_at": datetime.now(timezone.utc),
                        }
                    )
                )
                return
            error = response.error or EvaluationError(
                kind=ErrorKind.INTERNAL,
                message="Worker returned failure without an error record",
            )
        except asyncio.TimeoutError as exception:
            error = EvaluationError(kind=ErrorKind.TIMEOUT, message=str(exception) or "timeout")
        except HttpStatusError as exception:
            embedded = exception.payload.get("error")
            if isinstance(embedded, dict):
                error = EvaluationError.model_validate(embedded)
            else:
                error = EvaluationError(kind=ErrorKind.TRANSPORT, message=str(exception))
        except (OSError, RuntimeError) as exception:
            error = EvaluationError(kind=ErrorKind.TRANSPORT, message=str(exception))
        except Exception as exception:
            error = EvaluationError(
                kind=ErrorKind.INTERNAL,
                message=f"{type(exception).__name__}: {exception}",
            )
        finally:
            self._busy_workers.discard(worker_key)
            self._wake.set()

        assert error is not None
        now = datetime.now(timezone.utc)
        if error.retryable and not self.campaign.storage.is_cancelled(request.request_id):
            retry_count = self._retry_count.get(request.request_id, 0) + 1
            self._retry_count[request.request_id] = retry_count
            delay = min(
                self.retry_max_seconds,
                self.retry_initial_seconds * (2 ** min(retry_count - 1, 16)),
            )
            self._retry_at[request.request_id] = asyncio.get_running_loop().time() + delay
            self._pending[request.request_id] = request
            status = AttemptStatus.RETRY
        else:
            status = AttemptStatus.FAILED
            self.campaign.storage.put_terminal_error(
                request.request_id,
                {"request_id": request.request_id, "attempt_id": attempt_id, "error": error},
            )
        self.campaign.storage.append_attempt(
            leased.model_copy(update={"status": status, "finished_at": now, "error": error})
        )

    async def wait(self, request_id: str, *, timeout: float | None = None) -> EvaluationResult:
        await self.start()

        async def poll() -> EvaluationResult:
            while True:
                result = self.campaign.storage.get_result(request_id)
                if result is not None:
                    return result
                terminal = self.campaign.storage.get_terminal_error(request_id)
                if terminal is not None:
                    error = terminal.get("error", {})
                    raise RuntimeError(error.get("message", f"Evaluation {request_id} failed"))
                if self.campaign.storage.is_cancelled(request_id):
                    raise asyncio.CancelledError(request_id)
                await asyncio.sleep(self.poll_seconds)

        return await asyncio.wait_for(poll(), timeout=timeout) if timeout else await poll()

    def lookup(self, request_id: str) -> EvaluationResult | None:
        return self.campaign.storage.get_result(request_id)

    def status(self) -> dict[str, Any]:
        workers = self.campaign.registry.list_workers(
            self.campaign.manifest,
            stale_seconds=self.stale_seconds,
        )
        return {
            **self.campaign.storage.summary(),
            "queued": len(self._pending),
            "leased": len(self._inflight),
            "active_workers": len(workers),
            "idle_workers": sum(worker.state == WorkerState.IDLE for worker in workers),
            "cache_hits": self._cache_hits,
            "retries": sum(self._retry_count.values()),
        }

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._wake.set()
        if self._dispatch_task is not None:
            await self._dispatch_task
        for task in self._inflight.values():
            task.cancel()
        if self._inflight:
            await asyncio.gather(*self._inflight.values(), return_exceptions=True)
        await self.http.close()
        if self._lease_context is not None:
            self._lease_context.__exit__(None, None, None)
            self._lease_context = None
