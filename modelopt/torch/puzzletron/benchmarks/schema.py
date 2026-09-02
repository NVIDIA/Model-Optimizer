# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    repetition: int = Field(default=0, ge=0)
    gpu_count: int = Field(default=1, ge=1)
    cache_identity: str = "unknown"
    engine: Literal["aiperf"] = "aiperf"
    topology: dict[str, Any]
    workload: dict[str, Any]
    concurrency: int = Field(ge=1)
    metrics: dict[str, float]
    measurement_contract: dict[str, Any] = Field(default_factory=dict)
    checkpoint_identity: dict[str, Any] = Field(default_factory=dict)
    hardware_identity: dict[str, Any] = Field(default_factory=dict)
    software_identity: dict[str, Any] = Field(default_factory=dict)
    result_fingerprint: str = "unknown"
    failures: int = Field(default=0, ge=0)
    raw_artifacts: dict[str, str]
    raw_artifact_sha256: dict[str, str] = Field(default_factory=dict)
    command: tuple[str, ...]
    started_at: datetime
    completed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
