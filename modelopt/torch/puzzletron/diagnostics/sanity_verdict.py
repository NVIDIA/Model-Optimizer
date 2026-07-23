# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for policy-controlled Puzzletron sanity verdicts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..manifest import StageManifest
from ..stage_runner import StageResult
from ..stages.common import complete_stage

__all__ = ["SanityVerdict", "complete_sanity_stage", "finding_from_message"]


@dataclass
class SanityVerdict:
    """Quality outcome for one sanity stage execution."""

    passed: bool
    findings: list[dict[str, Any]] = field(default_factory=list)
    message: str | None = None


def finding_from_message(
    *,
    stage: str,
    message: str,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "message": message,
        "evidence": evidence or {},
        "severity": "warning",
    }


def complete_sanity_stage(
    config: dict[str, Any],
    manifest: StageManifest,
    *,
    outputs: dict[str, Any] | None = None,
    verdict: SanityVerdict,
    message: str | None = None,
) -> StageResult:
    """Complete a sanity stage according to the global warning policy."""

    merged = dict(outputs or {})
    merged["passed"] = verdict.passed
    merged["findings"] = list(verdict.findings)
    merged["verdict"] = "passed" if verdict.passed else "warning"
    fail_on_warnings = bool((config.get("sanity") or {}).get("fail_on_warnings", False))
    status = "failed" if fail_on_warnings and not verdict.passed else "success"
    return complete_stage(
        config,
        manifest,
        outputs=merged,
        status=status,
        message=message or verdict.message or f"Stage '{manifest.stage}' completed.",
    )
