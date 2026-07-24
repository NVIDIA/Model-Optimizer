# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Path-addressed cross-section validation for setup v2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .state import WizardState

__all__ = ["ValidationIssue", "validate_state"]


@dataclass(frozen=True)
class ValidationIssue:
    path: str
    message: str


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def validate_state(state: WizardState) -> tuple[ValidationIssue, ...]:
    """Return actionable authoring issues before canonical compilation."""

    issues: list[ValidationIssue] = []
    required = (
        "model.source",
        "data.source",
        "infrastructure.execution_contract.repository",
        "output.result_root",
    )
    for path in required:
        if not state.get_field(path):
            issues.append(ValidationIssue(path, "A value is required."))
    for path, record in state.records().items():
        if record.stale:
            issues.append(
                ValidationIssue(path, record.error or "This answer must be revalidated.")
            )

    profiles = _mapping(state.collection("parallel_profiles"))
    resources = _mapping(state.collection("stage_resources"))
    for stage_id, raw in resources.items():
        profile_name = _mapping(raw).get("profile_name")
        if profile_name and profile_name not in profiles:
            issues.append(
                ValidationIssue(
                    f"stage_resources.{stage_id}.profile_name",
                    f"Unknown parallel profile {profile_name!r}.",
                )
            )

    measurements = _mapping(state.collection("vllm_measurements"))
    mip = _mapping(state.collection("mip_config"))
    workloads = _mapping(mip.get("workloads"))
    for name in workloads:
        if measurements and name not in measurements:
            issues.append(
                ValidationIssue(
                    f"mip.workloads.{name}",
                    "The referenced named vLLM measurement does not exist.",
                )
            )

    for name, raw in measurements.items():
        setting = _mapping(raw)
        for field_name in (
            "prefill_seq_len",
            "generation_seq_len",
            "batch_size",
            "max_num_seqs",
        ):
            try:
                valid = int(setting.get(field_name, 0)) > 0
            except (TypeError, ValueError):
                valid = False
            if not valid:
                issues.append(
                    ValidationIssue(
                        f"vllm.measurements.{name}.{field_name}",
                        "Enter a positive integer.",
                    )
                )
        if int(setting.get("max_num_seqs", 0) or 0) < int(
            setting.get("batch_size", 1) or 1
        ):
            issues.append(
                ValidationIssue(
                    f"vllm.measurements.{name}.max_num_seqs",
                    "max_num_seqs must be at least batch_size.",
                )
            )

    order = {
        "campaign": 0,
        "model": 1,
        "data": 2,
        "infrastructure": 3,
        "pruning": 4,
        "stage_resources": 5,
        "vllm": 6,
        "mip": 7,
        "post_mip": 8,
        "output": 9,
    }
    return tuple(
        sorted(issues, key=lambda item: (order.get(item.path.split(".", 1)[0], 99), item.path))
    )


def validate_sources(state: WizardState) -> tuple[ValidationIssue, ...]:
    """Optional local source existence checks used by review screens."""

    issues = []
    for path in ("model.source", "data.source"):
        source = str(state.get_field(path, ""))
        if source.startswith((".", "/")) and not Path(source).expanduser().exists():
            issues.append(ValidationIssue(path, f"Local path does not exist: {source}"))
    return tuple(issues)
