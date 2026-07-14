# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Normalized measured-runtime result shared by Puzzletron reports."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import Field

from ..distributed_eval.schema import StrictModel

__all__ = ["BenchmarkResult"]


class BenchmarkResult(StrictModel):
    architecture_id: str
    checkpoint_dir: str
    solution_id: str = "unknown"
    profile_id: str = "unknown"
    topology_id: str = "unknown"
    workload_id: str = "unknown"
    gpu_count: int = Field(default=1, ge=1)
    cache_identity: str = "unknown"
    engine: Literal["aiperf"] = "aiperf"
    topology: dict[str, Any]
    workload: dict[str, Any]
    concurrency: int = Field(ge=1)
    metrics: dict[str, float]
    failures: int = Field(default=0, ge=0)
    raw_artifacts: dict[str, str]
    command: tuple[str, ...]
    started_at: datetime
    completed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
