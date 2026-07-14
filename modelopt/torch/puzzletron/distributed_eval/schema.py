"""Versioned wire and persistence schemas for distributed evaluation."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .identity import canonicalize, content_id

SCHEMA_VERSION = 1


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ParallelismSpec(StrictModel):
    tp_size: int = Field(default=1, ge=1)
    ep_size: int = Field(default=1, ge=1)
    cp_size: int = Field(default=1, ge=1)
    pp_size: int = Field(default=1, ge=1)
    dp_size: int | None = Field(default=None, ge=1)
    sequence_parallel: bool = False
    fsdp: bool = True
    distributed_backend: str = "nccl"
    world_size: int = Field(default=1, ge=1)
    gpus_per_task: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_world_and_gpu_ownership(self):
        dp = self.dp_size
        model_parallel = self.tp_size * self.cp_size * self.pp_size * self.ep_size
        if dp is not None and model_parallel * dp != self.world_size:
            raise ValueError(
                f"parallel sizes imply world_size={model_parallel * dp}, "
                f"configured world_size={self.world_size}"
            )
        if self.gpus_per_task is not None and self.gpus_per_task != self.world_size:
            raise ValueError(
                f"gpus_per_task={self.gpus_per_task} must equal worker "
                f"world_size={self.world_size}"
            )
        return self


class CampaignManifest(StrictModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    name: str
    model: dict[str, Any]
    descriptor: str
    force_hf: bool
    parallelism: ParallelismSpec
    precision: dict[str, Any] = Field(default_factory=dict)
    automodel_recipe: dict[str, Any]
    data: dict[str, Any]
    metrics: dict[str, Any]
    evaluator_revision: str
    result_atol: float = Field(default=1e-4, ge=0.0)
    result_rtol: float = Field(default=1e-4, ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def campaign_id(self) -> str:
        return content_id("campaign", self.model_dump(mode="python"))


class WorkerState(str, Enum):
    STARTING = "starting"
    IDLE = "idle"
    BUSY = "busy"
    DRAINING = "draining"
    FAILED = "failed"


class WorkerRecord(StrictModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    worker_id: str
    boot_id: str
    campaign_id: str
    host: str
    port: int = Field(ge=1, le=65535)
    parallelism: ParallelismSpec
    capabilities: dict[str, Any] = Field(default_factory=dict)
    state: WorkerState = WorkerState.STARTING
    current_request_id: str | None = None
    started_at: datetime = Field(default_factory=utc_now)
    heartbeat_at: datetime = Field(default_factory=utc_now)

    @property
    def endpoint(self) -> str:
        return f"http://{self.host}:{self.port}"


class EvaluationRequest(StrictModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    campaign_id: str
    handler: str
    payload: dict[str, Any]
    model: dict[str, Any]
    data: dict[str, Any]
    metrics: dict[str, Any]
    precision: dict[str, Any] = Field(default_factory=dict)
    evaluator_revision: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def request_id(self) -> str:
        return content_id("eval", self.model_dump(mode="python"))

    def to_wire(self) -> dict[str, Any]:
        return {**canonicalize(self), "request_id": self.request_id}

    @classmethod
    def from_wire(cls, payload: dict[str, Any]) -> "EvaluationRequest":
        data = dict(payload)
        stored_id = data.pop("request_id", None)
        request = cls.model_validate(data)
        if stored_id is not None and stored_id != request.request_id:
            raise ValueError(
                f"Request identity mismatch: payload says {stored_id!r}, "
                f"canonical request is {request.request_id!r}"
            )
        return request


class EvaluationResult(StrictModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    request_id: str
    campaign_id: str
    metrics: dict[str, Any]
    counts: dict[str, int | float] = Field(default_factory=dict)
    reduction_state: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    timing: dict[str, float] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    completed_at: datetime = Field(default_factory=utc_now)


class ErrorKind(str, Enum):
    TRANSPORT = "transport"
    TIMEOUT = "timeout"
    WORKER_LOST = "worker_lost"
    PROCESS_GROUP = "process_group"
    INVALID_REQUEST = "invalid_request"
    UNSUPPORTED = "unsupported"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    CANDIDATE = "candidate"
    INTERNAL = "internal"
    CANCELLED = "cancelled"


TRANSIENT_ERROR_KINDS = {
    ErrorKind.TRANSPORT,
    ErrorKind.TIMEOUT,
    ErrorKind.WORKER_LOST,
    ErrorKind.PROCESS_GROUP,
    ErrorKind.INTERNAL,
}


class EvaluationError(StrictModel):
    kind: ErrorKind
    message: str
    traceback: str | None = None
    retryable: bool | None = None

    @model_validator(mode="before")
    @classmethod
    def set_default_retryable(cls, value):
        if isinstance(value, dict) and value.get("retryable") is None and value.get("kind"):
            data = dict(value)
            data["retryable"] = ErrorKind(data["kind"]) in TRANSIENT_ERROR_KINDS
            return data
        return value


class AttemptStatus(str, Enum):
    LEASED = "leased"
    SUCCEEDED = "succeeded"
    RETRY = "retry"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DUPLICATE = "duplicate"
    CONFLICT = "conflict"


class AttemptRecord(StrictModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    request_id: str
    attempt_id: str
    worker_id: str | None = None
    worker_boot_id: str | None = None
    status: AttemptStatus
    leased_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime | None = None
    error: EvaluationError | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationHandle(StrictModel):
    campaign_id: str
    request_id: str


class ExecuteResponse(StrictModel):
    status: Literal["succeeded", "failed", "cancelled"]
    result: EvaluationResult | None = None
    error: EvaluationError | None = None


class CacheWriteStatus(str, Enum):
    WRITTEN = "written"
    DUPLICATE = "duplicate"
    CONFLICT = "conflict"
