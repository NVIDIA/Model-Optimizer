# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Path-addressed cross-section validation for setup v2."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .parallel_validation import (
    ParallelCompatibilityIssue,
    validate_automodel_parallelism,
    validate_vllm_parallelism,
)

if TYPE_CHECKING:
    from .state import WizardState

__all__ = ["ValidationIssue", "validate_state"]


@dataclass(frozen=True)
class ValidationIssue:
    """One path-addressed problem in persisted wizard state."""

    path: str
    message: str


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _parallel_issues(
    issues: tuple[ParallelCompatibilityIssue, ...],
    *,
    prefix: str,
) -> list[ValidationIssue]:
    return [
        ValidationIssue(f"{prefix}.{issue.path.rsplit('.', 1)[-1]}", issue.message)
        for issue in issues
    ]


def _post_mip_nodes(state: WizardState) -> dict[str, Mapping[str, Any]]:
    nodes = {}
    for flow_id, raw_flow in _mapping(state.collection("post_mip_flows")).items():
        for node_id, raw_node in _mapping(_mapping(raw_flow).get("nodes")).items():
            nodes[f"post.{flow_id}.{node_id}"] = _mapping(raw_node)
    return nodes


def validate_state(state: WizardState) -> tuple[ValidationIssue, ...]:
    """Return actionable authoring issues before canonical compilation."""
    issues: list[ValidationIssue] = []
    required = (
        "model.source",
        "data.source",
        "infrastructure.execution_contract.repository",
        "output.result_root",
    )
    issues.extend(
        ValidationIssue(path, "A value is required.")
        for path in required
        if not state.get_field(path)
    )
    for path, record in state.records().items():
        if record.stale:
            issues.append(
                ValidationIssue(path, record.error or "This answer must be revalidated.")
            )

    profiles = _mapping(state.collection("parallel_profiles"))
    resources = _mapping(state.collection("stage_resources"))
    inventory = _mapping(state.payload.get("inventory"))
    pruning = _mapping(state.collection("pruning"))
    sequence_length = int(state.get_field("data.sequence_length", 4096))
    post_mip_nodes = _post_mip_nodes(state)
    for stage_id, raw in resources.items():
        profile_name = _mapping(raw).get("profile_name")
        if profile_name and profile_name not in profiles:
            issues.append(
                ValidationIssue(
                    f"stage_resources.{stage_id}.profile_name",
                    f"Unknown parallel profile {profile_name!r}.",
                )
            )
            continue
        if profile_name:
            node_type = _mapping(post_mip_nodes.get(str(stage_id))).get("type")
            parallel = _mapping(profiles.get(str(profile_name)))
            issues.extend(
                _parallel_issues(
                    validate_automodel_parallelism(
                        parallel,
                        inventory,
                        pruning,
                        stage_id=str(stage_id),
                        sequence_length=sequence_length,
                        node_type=str(node_type) if node_type else None,
                    ),
                    prefix=f"stage_resources.{stage_id}",
                )
            )

    for stage_id, node in post_mip_nodes.items():
        node_type = str(node.get("type", ""))
        config = _mapping(node.get("config"))
        if node_type == "aiperf":
            topology = _mapping(config.get("topology"))
            if topology:
                issues.extend(
                    _parallel_issues(
                        validate_vllm_parallelism(
                            topology,
                            inventory,
                            pruning,
                            stage_id=stage_id,
                        ),
                        prefix=f"post_mip_flows.{stage_id}",
                    )
                )
        elif node_type in {"evaluation", "global_kd"}:
            parallel = _mapping(_mapping(config.get("automodel")).get("parallel"))
            resource = _mapping(resources.get(stage_id))
            profile_name = resource.get("profile_name")
            if parallel and not profile_name:
                issues.extend(
                    _parallel_issues(
                        validate_automodel_parallelism(
                            parallel,
                            inventory,
                            pruning,
                            stage_id=stage_id,
                            sequence_length=sequence_length,
                            node_type=node_type,
                        ),
                        prefix=f"post_mip_flows.{stage_id}",
                    )
                )

    measurements = _mapping(state.collection("vllm_measurements"))
    mip = _mapping(state.collection("mip_config"))
    workloads = _mapping(mip.get("workloads"))
    if measurements:
        issues.extend(
            (
                ValidationIssue(
                    f"mip.workloads.{name}",
                    "The referenced named vLLM measurement does not exist.",
                )
            )
            for name in workloads
            if name not in measurements
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
        topology = _mapping(_mapping(setting.get("runtime_stats")).get("topology"))
        if topology:
            issues.extend(
                _parallel_issues(
                    validate_vllm_parallelism(
                        topology,
                        inventory,
                        pruning,
                        stage_id=f"vllm.measurements.{name}",
                    ),
                    prefix=f"vllm.measurements.{name}.runtime_stats.topology",
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
