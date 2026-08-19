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

"""WorkAdapter contract for stage orchestration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

from ..schema import (
    AttemptSpec,
    CampaignPlan,
    CommandSpec,
    FailureClass,
    PublishedOutput,
    ResumeDecision,
    RunnerEnvironment,
    StagePlanNode,
    ValidatedResult,
    WorkItem,
    WorkPlan,
)

__all__ = ["ExecutionIdentityProjectionUnavailable", "WorkAdapter"]


class ExecutionIdentityProjectionUnavailable(RuntimeError):
    """The current upstream state does not yet define an adapter identity projection."""


class WorkAdapter(ABC):
    """Semantic adapter for one stage execution strategy."""

    @abstractmethod
    def plan(self, plan: CampaignPlan, node: StagePlanNode) -> WorkPlan:
        raise NotImplementedError

    @abstractmethod
    def command(
        self,
        *,
        plan: CampaignPlan,
        node: StagePlanNode,
        item: WorkItem,
        attempt_id: str,
        runner: RunnerEnvironment,
        overrides: list[str] | None = None,
    ) -> AttemptSpec:
        raise NotImplementedError

    def inspect_resume(
        self,
        *,
        plan: CampaignPlan,
        node: StagePlanNode,
        item: WorkItem,
        attempts: list[Mapping[str, Any]],
    ) -> ResumeDecision:
        for attempt in attempts:
            if attempt.get("status") == "completed":
                return ResumeDecision(action="skip", reason="attempt already completed", skip=True)
        return ResumeDecision(action="run", reason="no completed attempt")

    @abstractmethod
    def validate(self, *, plan: CampaignPlan, node: StagePlanNode) -> ValidatedResult:
        raise NotImplementedError

    def aggregate(
        self,
        *,
        plan: CampaignPlan,
        node: StagePlanNode,
        work_plan: WorkPlan,
    ) -> PublishedOutput | None:
        return None

    def execution_identity_projection(
        self,
        *,
        plan: CampaignPlan,
        node: StagePlanNode,
        work_plan: WorkPlan,
    ) -> Mapping[str, Any]:
        """Return adapter-owned, currently resolvable execution inputs."""

        return {}

    def prepare_execution_identity_projection(
        self,
        *,
        plan: CampaignPlan,
        node: StagePlanNode,
    ) -> None:
        """Prepare mutable adapter inputs immediately before binding a new attempt.

        Read-only currentness checks call only :meth:`execution_identity_projection`.
        Adapters that require preparation must implement it here rather than mutate
        state while projecting identity.
        """

    def classify_failure(
        self,
        *,
        exit_code: int | None,
        reason: str | None,
        logs: tuple[str, ...],
    ) -> FailureClass:
        if exit_code == 0:
            return FailureClass.SUCCESS
        text = reason or ""
        upper = text.upper()
        if "OUT_OF_MEMORY" in upper or "OOM" in upper:
            return FailureClass.OOM
        if "TIMEOUT" in upper:
            return FailureClass.TIMEOUT_FATAL
        if "CANCEL" in upper:
            return FailureClass.CANCELLED
        if exit_code is None:
            return FailureClass.UNKNOWN
        return FailureClass.APPLICATION
