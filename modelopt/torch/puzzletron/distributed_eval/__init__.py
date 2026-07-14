"""Optional, fault-tolerant distributed evaluation for Puzzletron.

This package is intentionally isolated from the existing local evaluator and
``rpc_eval`` scaffold.  Import it explicitly when distributed evaluation is
desired; importing :mod:`modelopt.torch.puzzletron` does not start services or
change scoring behavior.
"""

from .campaign import Campaign
from .client import AsyncEvaluationClient, EvaluationClient
from .identity import prefix_cache_id
from .schema import (
    AttemptRecord,
    AttemptStatus,
    CampaignManifest,
    ErrorKind,
    EvaluationError,
    EvaluationHandle,
    EvaluationRequest,
    EvaluationResult,
    ParallelismSpec,
    WorkerRecord,
    WorkerState,
)

__all__ = [
    "AsyncEvaluationClient",
    "AttemptRecord",
    "AttemptStatus",
    "Campaign",
    "CampaignManifest",
    "ErrorKind",
    "EvaluationClient",
    "EvaluationError",
    "EvaluationHandle",
    "EvaluationRequest",
    "EvaluationResult",
    "ParallelismSpec",
    "prefix_cache_id",
    "WorkerRecord",
    "WorkerState",
]
