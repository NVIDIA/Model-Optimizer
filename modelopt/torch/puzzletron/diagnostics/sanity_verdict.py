# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for policy-controlled Puzzletron sanity verdicts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from ..manifest import StageManifest
from ..stage_runner import StageResult
from ..stages.common import complete_stage

__all__ = [
    "SanityVerdict",
    "complete_sanity_stage",
    "finding_from_message",
    "is_correctness_sanity_stage",
]

_CORRECTNESS_STAGES = frozenset({"sort_sanity", "slicing_sanity"})


def is_correctness_sanity_stage(stage: str) -> bool:
    """Return whether a failed stage verdict is a blocking correctness failure."""

    return stage in _CORRECTNESS_STAGES


@dataclass
class SanityVerdict:
    """Observed outcome for one sanity stage execution."""

    passed: bool
    findings: list[dict[str, Any]] = field(default_factory=list)
    blocking: bool = False
    message: str | None = None


def finding_from_message(
    *,
    stage: str,
    message: str,
    evidence: dict[str, Any] | None = None,
    severity: Literal["warning", "error"] = "warning",
) -> dict[str, Any]:
    return {
        "stage": stage,
        "message": message,
        "evidence": evidence or {},
        "severity": severity,
    }


def complete_sanity_stage(
    config: dict[str, Any],
    manifest: StageManifest,
    *,
    outputs: dict[str, Any] | None = None,
    verdict: SanityVerdict,
    message: str | None = None,
) -> StageResult:
    """Complete a sanity stage without downgrading correctness failures."""

    merged = dict(outputs or {})
    correctness_failure = not verdict.passed and (
        verdict.blocking or is_correctness_sanity_stage(manifest.stage)
    )
    findings = []
    for finding in verdict.findings:
        normalized = dict(finding)
        finding_stage = str(normalized.get("stage", manifest.stage))
        if correctness_failure and (
            is_correctness_sanity_stage(manifest.stage)
            or is_correctness_sanity_stage(finding_stage)
        ):
            normalized["severity"] = "error"
        else:
            normalized.setdefault("severity", "warning")
        findings.append(normalized)
    merged["passed"] = verdict.passed
    merged["findings"] = findings
    merged["blocking"] = correctness_failure
    merged["verdict"] = (
        "passed" if verdict.passed else "failed" if correctness_failure else "warning"
    )
    fail_on_warnings = bool((config.get("sanity") or {}).get("fail_on_warnings", False))
    status = (
        "failed"
        if correctness_failure or (fail_on_warnings and not verdict.passed)
        else "success"
    )
    return complete_stage(
        config,
        manifest,
        outputs=merged,
        status=status,
        message=message or verdict.message or f"Stage '{manifest.stage}' completed.",
    )
