# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
        """Prepare mutable adapter inputs immediately before a new attempt is bound.

        The default is intentionally empty. Read-only currentness checks call only
        :meth:`execution_identity_projection`; adapters that require preparation
        must implement it here instead of mutating from their projection method.
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
