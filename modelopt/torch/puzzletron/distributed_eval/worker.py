"""Rank-0 HTTP control server and all-rank worker execution loop."""

from __future__ import annotations

import asyncio
import hmac
import logging
import socket
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from queue import Empty, Queue
from typing import Protocol

from .campaign import Campaign
from .schema import (
    AttemptRecord,
    AttemptStatus,
    CacheWriteStatus,
    ErrorKind,
    EvaluationError,
    EvaluationRequest,
    EvaluationResult,
    ExecuteResponse,
    WorkerRecord,
    WorkerState,
)

logger = logging.getLogger(__name__)


class EvaluationExecutor(Protocol):
    def setup(self) -> None: ...

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult | None: ...

    def close(self) -> None: ...

    def capabilities(self) -> dict: ...


@dataclass
class _Envelope:
    request: EvaluationRequest
    attempt_id: str
    done: threading.Event = field(default_factory=threading.Event)
    response: ExecuteResponse | None = None


def _distributed_info() -> tuple[int, int]:
    try:
        import torch.distributed as torch_dist

        if torch_dist.is_available() and torch_dist.is_initialized():
            return torch_dist.get_rank(), torch_dist.get_world_size()
    except ImportError:
        pass
    return 0, 1


def _broadcast_command(command: dict | None, rank: int, world_size: int) -> dict:
    if world_size == 1:
        if command is None:
            raise RuntimeError("Rank 0 must supply a worker command")
        return command
    import torch.distributed as torch_dist

    values = [command if rank == 0 else None]
    torch_dist.broadcast_object_list(values, src=0)
    if not isinstance(values[0], dict):
        raise RuntimeError(f"Invalid distributed worker command: {values[0]!r}")
    return values[0]


def _gather_outcomes(value: dict, rank: int, world_size: int) -> list[dict]:
    if world_size == 1:
        return [value]
    import torch.distributed as torch_dist

    gathered: list[dict | None] = [None] * world_size
    torch_dist.all_gather_object(gathered, value)
    return [item for item in gathered if item is not None]


def _classify_exception(error: BaseException) -> EvaluationError:
    if isinstance(error, (ValueError, KeyError, TypeError)):
        kind = ErrorKind.INVALID_REQUEST
        retryable = False
    elif isinstance(error, NotImplementedError):
        kind = ErrorKind.UNSUPPORTED
        retryable = False
    elif isinstance(error, MemoryError):
        kind = ErrorKind.RESOURCE_EXHAUSTED
        retryable = False
    else:
        text = f"{type(error).__name__}: {error}"
        lowered = text.lower()
        if any(token in lowered for token in ("nccl", "process group", "connection reset")):
            kind = ErrorKind.PROCESS_GROUP
        else:
            kind = ErrorKind.INTERNAL
        retryable = True
    return EvaluationError(
        kind=kind,
        message=f"{type(error).__name__}: {error}",
        traceback=traceback.format_exc(),
        retryable=retryable,
    )


class _HttpControlServer:
    def __init__(
        self,
        *,
        campaign: Campaign,
        host: str,
        port: int,
        queue: Queue,
        token: str,
        is_busy,
        set_draining,
        capabilities,
    ):
        self.campaign = campaign
        self.host = host
        self.port = port
        self.queue = queue
        self.token = token
        self.is_busy = is_busy
        self.set_draining = set_draining
        self.capabilities = capabilities
        self.ready = threading.Event()
        self.stopped = threading.Event()
        self._thread: threading.Thread | None = None
        self._loop = None
        self._runner = None
        self._admission_lock = threading.Lock()
        self._pending_request = False

    def start(self) -> None:
        self._thread = threading.Thread(target=self._thread_main, daemon=True)
        self._thread.start()
        if not self.ready.wait(timeout=30):
            raise TimeoutError(f"HTTP worker failed to listen on {self.host}:{self.port}")

    def _thread_main(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._start())
            self.ready.set()
            self._loop.run_forever()
        finally:
            if self._runner is not None:
                self._loop.run_until_complete(self._runner.cleanup())
            self._loop.close()
            self.stopped.set()

    async def _start(self) -> None:
        try:
            from aiohttp import web
        except ImportError as error:
            raise RuntimeError(
                "Distributed evaluation worker requires aiohttp; install the "
                "example-local requirements.txt"
            ) from error
        app = web.Application(client_max_size=16 * 1024 * 1024)
        app.router.add_get("/v1/health", self._health)
        app.router.add_get("/v1/capabilities", self._capabilities)
        app.router.add_post("/v1/tasks/{request_id}/execute", self._execute)
        app.router.add_post("/v1/tasks/{request_id}/cancel", self._cancel)
        app.router.add_post("/v1/drain", self._drain)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()

    def _authorized(self, request) -> bool:
        expected = f"Bearer {self.token}"
        return hmac.compare_digest(request.headers.get("Authorization", ""), expected)

    async def _health(self, request):
        from aiohttp import web

        return web.json_response({"status": "ok", "busy": bool(self.is_busy())})

    async def _capabilities(self, request):
        from aiohttp import web

        if not self._authorized(request):
            return web.json_response({"message": "unauthorized"}, status=401)
        return web.json_response(self.capabilities())

    async def _execute(self, request):
        from aiohttp import web

        if not self._authorized(request):
            return web.json_response({"message": "unauthorized"}, status=401)
        with self._admission_lock:
            if self.is_busy() or self._pending_request:
                return web.json_response({"message": "worker is busy or draining"}, status=409)
            self._pending_request = True
        try:
            try:
                body = await request.json()
                evaluation = EvaluationRequest.from_wire(body["request"])
                path_request_id = request.match_info["request_id"]
                if path_request_id != evaluation.request_id:
                    raise ValueError("request id mismatch")
                self.campaign.validate_request(evaluation)
                envelope = _Envelope(request=evaluation, attempt_id=str(body["attempt_id"]))
            except (KeyError, TypeError, ValueError) as error:
                record = EvaluationError(
                    kind=ErrorKind.INVALID_REQUEST,
                    message=f"{type(error).__name__}: {error}",
                    retryable=False,
                )
                return web.json_response(
                    {"message": record.message, "error": record.model_dump(mode="json")},
                    status=400,
                )
            self.queue.put(envelope)
            await asyncio.to_thread(envelope.done.wait)
            assert envelope.response is not None
            return web.json_response(envelope.response.model_dump(mode="json"))
        finally:
            with self._admission_lock:
                self._pending_request = False

    async def _cancel(self, request):
        from aiohttp import web

        if not self._authorized(request):
            return web.json_response({"message": "unauthorized"}, status=401)
        self.campaign.storage.mark_cancelled(request.match_info["request_id"])
        return web.json_response({"status": "cancelled"})

    async def _drain(self, request):
        from aiohttp import web

        if not self._authorized(request):
            return web.json_response({"message": "unauthorized"}, status=401)
        self.set_draining()
        return web.json_response({"status": "draining"})

    def stop(self) -> None:
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=30)


class DistributedEvaluationWorker:
    """Long-lived worker group that executes one distributed request at a time."""

    def __init__(
        self,
        campaign: Campaign,
        executor: EvaluationExecutor,
        *,
        host: str,
        port: int,
        worker_id: str | None = None,
        heartbeat_seconds: float = 10.0,
    ):
        self.campaign = campaign
        self.executor = executor
        self.host = host
        self.port = port
        self.worker_id = worker_id or socket.gethostname()
        self.boot_id = uuid.uuid4().hex
        self.started_at = datetime.now(timezone.utc)
        self.heartbeat_seconds = heartbeat_seconds
        self.rank, self.world_size = _distributed_info()
        self._queue: Queue[_Envelope] = Queue()
        self._state = WorkerState.STARTING
        self._current_request_id: str | None = None
        self._state_lock = threading.Lock()
        self._draining = False
        self._stop_heartbeat = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._http: _HttpControlServer | None = None

    def _set_state(self, state: WorkerState, request_id: str | None = None) -> None:
        with self._state_lock:
            self._state = state
            self._current_request_id = request_id

    def _is_busy(self) -> bool:
        with self._state_lock:
            return self._state != WorkerState.IDLE or self._draining

    def _set_draining(self) -> None:
        with self._state_lock:
            was_idle = self._state == WorkerState.IDLE
            self._draining = True
            if was_idle:
                self._state = WorkerState.DRAINING

    def _worker_record(self) -> WorkerRecord:
        with self._state_lock:
            state = self._state
            current = self._current_request_id
        return WorkerRecord(
            worker_id=self.worker_id,
            boot_id=self.boot_id,
            campaign_id=self.campaign.campaign_id,
            host=self.host,
            port=self.port,
            parallelism=self.campaign.manifest.parallelism,
            capabilities=self.executor.capabilities(),
            state=state,
            current_request_id=current,
            started_at=self.started_at,
            heartbeat_at=datetime.now(timezone.utc),
        )

    def _heartbeat_loop(self) -> None:
        while not self._stop_heartbeat.is_set():
            self.campaign.registry.publish(self._worker_record())
            self._stop_heartbeat.wait(self.heartbeat_seconds)

    def run(self) -> None:
        self.executor.setup()
        if self.rank == 0:
            self._http = _HttpControlServer(
                campaign=self.campaign,
                host=self.host,
                port=self.port,
                queue=self._queue,
                token=self.campaign.storage.read_token(),
                is_busy=self._is_busy,
                set_draining=self._set_draining,
                capabilities=self.executor.capabilities,
            )
            self._http.start()
            self._set_state(WorkerState.IDLE)
            self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            self._heartbeat_thread.start()
        try:
            self._run_loop()
        finally:
            try:
                self.executor.close()
            finally:
                if self.rank == 0:
                    self._stop_heartbeat.set()
                    if self._heartbeat_thread is not None:
                        self._heartbeat_thread.join(timeout=30)
                    self.campaign.registry.remove(self.worker_id, self.boot_id)
                    if self._http is not None:
                        self._http.stop()

    def _next_rank0_command(self) -> tuple[dict, _Envelope | None]:
        if self._draining and self._queue.empty():
            return {"kind": "shutdown"}, None
        try:
            envelope = self._queue.get(timeout=0.5)
        except Empty:
            return {"kind": "idle"}, None
        if self.campaign.storage.is_cancelled(envelope.request.request_id):
            envelope.response = ExecuteResponse(status="cancelled")
            envelope.done.set()
            return {"kind": "idle"}, None
        return {
            "kind": "evaluate",
            "attempt_id": envelope.attempt_id,
            "request": envelope.request.to_wire(),
        }, envelope

    def _run_loop(self) -> None:
        while True:
            envelope = None
            command = None
            if self.rank == 0:
                command, envelope = self._next_rank0_command()
            command = _broadcast_command(command, self.rank, self.world_size)
            if command["kind"] == "shutdown":
                break
            if command["kind"] == "idle":
                continue
            request = EvaluationRequest.from_wire(command["request"])
            if self.rank == 0:
                self._set_state(WorkerState.BUSY, request.request_id)
            local_result = None
            local_error = None
            started = time.perf_counter()
            try:
                local_result = self.executor.evaluate(request)
            except BaseException as error:
                local_error = _classify_exception(error)
                import traceback

                print(
                    "[distributed-eval/worker] "
                    f"rank={self.rank} request={request.request_id} "
                    f"evaluation_exception={type(error).__name__}: {error}\n"
                    f"{''.join(traceback.format_exception(error))}",
                    flush=True,
                )
            outcome = {
                "rank": self.rank,
                "result": local_result.model_dump(mode="json") if local_result else None,
                "error": local_error.model_dump(mode="json") if local_error else None,
            }
            outcomes = _gather_outcomes(outcome, self.rank, self.world_size)
            if self.rank != 0:
                continue
            assert envelope is not None
            errors = [item["error"] for item in outcomes if item.get("error")]
            if errors:
                error = EvaluationError.model_validate(errors[0])
                envelope.response = ExecuteResponse(status="failed", error=error)
                self.campaign.storage.append_attempt(
                    AttemptRecord(
                        request_id=request.request_id,
                        attempt_id=command["attempt_id"],
                        worker_id=self.worker_id,
                        worker_boot_id=self.boot_id,
                        status=AttemptStatus.RETRY if error.retryable else AttemptStatus.FAILED,
                        finished_at=datetime.now(timezone.utc),
                        error=error,
                        metadata={"elapsed_seconds": time.perf_counter() - started},
                    )
                )
            else:
                result_payloads = [item["result"] for item in outcomes if item.get("result")]
                if not result_payloads:
                    error = EvaluationError(
                        kind=ErrorKind.INTERNAL,
                        message="No rank produced an EvaluationResult",
                    )
                    envelope.response = ExecuteResponse(status="failed", error=error)
                else:
                    result = EvaluationResult.model_validate(result_payloads[0])
                    status = self.campaign.storage.put_result(
                        result,
                        attempt_id=command["attempt_id"],
                        atol=self.campaign.manifest.result_atol,
                        rtol=self.campaign.manifest.result_rtol,
                    )
                    if status == CacheWriteStatus.CONFLICT:
                        error = EvaluationError(
                            kind=ErrorKind.INTERNAL,
                            message=f"Conflicting result for {request.request_id}",
                            retryable=False,
                        )
                        envelope.response = ExecuteResponse(status="failed", error=error)
                    else:
                        envelope.response = ExecuteResponse(status="succeeded", result=result)
            envelope.done.set()
            self._set_state(WorkerState.DRAINING if self._draining else WorkerState.IDLE)
