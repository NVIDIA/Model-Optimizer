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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Path-addressed cross-section validation for setup v2."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from puzzletron_setup import validate_worker_path

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


def _dataset_subset_issues(state: WizardState) -> list[ValidationIssue]:
    selection = _mapping(state.collection("data_subset_selection"))
    if not selection:
        return []
    issues = []
    source = str(selection.get("source", ""))
    revision = str(selection.get("revision", ""))
    selected_source = str(
        state.get_field("data.selected_source", state.get_field("data.source", ""))
    )
    if not source or source != selected_source:
        issues.append(
            ValidationIssue(
                "data.subsets.source",
                "The selected subset source no longer matches the dataset source.",
            )
        )
    cache = _mapping(state.collection("hf_dataset_catalogs"))
    catalog = _mapping(cache.get(f"{source}@{revision}"))
    if not revision or not catalog:
        issues.append(
            ValidationIssue(
                "data.subsets.revision",
                "The revision-locked Hugging Face subset catalog is missing.",
            )
        )
    catalog_entries = {
        str(item.get("name", "")): item
        for item in catalog.get("subsets") or ()
        if isinstance(item, Mapping)
    }
    records = [
        item for item in selection.get("subsets") or () if isinstance(item, Mapping)
    ]
    names = [str(item.get("name", "")) for item in records]
    if not records or any(not name for name in names) or len(names) != len(set(names)):
        issues.append(
            ValidationIssue(
                "data.subsets",
                "Choose at least one unique Hugging Face dataset subset.",
            )
        )
    weights = []
    for record, name in zip(records, names):
        rows = record.get("num_rows")
        size = record.get("num_bytes_original_files")
        weight = record.get("weight")
        if (
            not isinstance(rows, int)
            or isinstance(rows, bool)
            or rows <= 0
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
        ):
            issues.append(
                ValidationIssue(
                    f"data.subsets.{name or 'unknown'}",
                    "Selected subset metadata requires positive rows and a known size.",
                )
            )
        if (
            not isinstance(weight, (int, float))
            or isinstance(weight, bool)
            or not math.isfinite(float(weight))
            or float(weight) < 0
        ):
            issues.append(
                ValidationIssue(
                    f"data.subsets.{name or 'unknown'}.weight",
                    "Selected subset weight must be finite and non-negative.",
                )
            )
        else:
            weights.append(float(weight))
        cached = _mapping(catalog_entries.get(name))
        if not cached:
            issues.append(
                ValidationIssue(
                    f"data.subsets.{name or 'unknown'}",
                    "Selected subset is absent from the revision-locked catalog.",
                )
            )
        elif not bool(cached.get("selectable", True)):
            reason = str(cached.get("disabled_reason") or "unavailable")
            issues.append(
                ValidationIssue(
                    f"data.subsets.{name}",
                    f"Selected subset is unavailable: {reason}.",
                )
            )
        elif any(
            record.get(key) != cached.get(key)
            for key in ("num_rows", "num_bytes_original_files")
        ):
            issues.append(
                ValidationIssue(
                    f"data.subsets.{name}",
                    "Selected subset metadata no longer matches the cached catalog.",
                )
            )
    if len(weights) != len(records) or not math.isclose(
        math.fsum(weights),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        issues.append(
            ValidationIssue(
                "data.subsets.weights",
                "Selected subset weights must sum to 1.0.",
            )
        )
    return issues


def validate_state(state: WizardState) -> tuple[ValidationIssue, ...]:
    """Return actionable authoring issues before canonical compilation."""
    issues: list[ValidationIssue] = []
    required = (
        "model.source",
        "data.source",
        "infrastructure.execution_contract.repository",
        "infrastructure.execution_contract.venv",
        "output.result_root",
    )
    issues.extend(
        ValidationIssue(path, "A value is required.")
        for path in required
        if not state.get_field(path)
    )
    for path in (
        "infrastructure.execution_contract.repository",
        "infrastructure.execution_contract.venv",
    ):
        value = state.get_field(path)
        if not value:
            continue
        verdict = validate_worker_path(str(value))
        if verdict is not True:
            issues.append(ValidationIssue(path, str(verdict)))
    for path, record in state.records().items():
        if record.stale:
            issues.append(ValidationIssue(path, record.error or "This answer must be revalidated."))
    issues.extend(_dataset_subset_issues(state))

    acquisition = _mapping(state.collection("data_acquisition"))
    adapter = str(acquisition.get("adapter", ""))
    if adapter:
        if not acquisition.get("output"):
            issues.append(
                ValidationIssue("data.acquisition.output", "A materialization path is required.")
            )
        if int(acquisition.get("seed", -1)) < 0:
            issues.append(
                ValidationIssue("data.acquisition.seed", "The selection seed cannot be negative.")
            )
        if adapter == "puzzle_kd_v2":
            issues.extend(
                ValidationIssue(
                    f"data.acquisition.{key}",
                    "The requested row count must be positive.",
                )
                for key in ("train_samples", "validation_samples")
                if int(acquisition.get(key, 0)) <= 0
            )
        elif adapter == "nemotron_vlm_v2":
            subsets = [str(item) for item in acquisition.get("subsets") or ()]
            if not subsets or len(subsets) != len(set(subsets)):
                issues.append(
                    ValidationIssue(
                        "data.acquisition.subsets",
                        "Choose at least one unique Nemotron-VLM subset.",
                    )
                )
            issues.extend(
                ValidationIssue(
                    f"data.acquisition.{key}",
                    "The bounded acquisition value must be positive.",
                )
                for key in ("num_samples", "max_shards_per_subset")
                if int(acquisition.get(key, 0)) <= 0
            )
        else:
            issues.append(
                ValidationIssue(
                    "data.acquisition.adapter",
                    f"Unknown first-class dataset adapter {adapter!r}.",
                )
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

    serving_workloads = _mapping(state.collection("serving_workloads"))
    measurements = _mapping(state.collection("vllm_measurements"))
    mip = _mapping(state.collection("mip_config"))
    workloads = _mapping(mip.get("workloads"))
    if not serving_workloads:
        issues.append(
            ValidationIssue(
                "serving_workloads",
                "Define at least one serving workload.",
            )
        )
    for name, raw in serving_workloads.items():
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
                        f"serving_workloads.{name}.{field_name}",
                        "Enter a positive integer.",
                    )
                )

    issues.extend(
        ValidationIssue(
            f"mip.workloads.{name}",
            "The referenced serving workload does not exist.",
        )
        for name in workloads
        if name not in serving_workloads
    )

    for name, raw in measurements.items():
        setting = _mapping(raw)
        serving = _mapping(serving_workloads.get(name))
        if not serving:
            issues.append(
                ValidationIssue(
                    f"vllm.measurements.{name}",
                    "The referenced serving workload does not exist.",
                )
            )
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
            if serving and setting.get(field_name) != serving.get(field_name):
                issues.append(
                    ValidationIssue(
                        f"vllm.measurements.{name}.{field_name}",
                        "The measurement must match its serving workload.",
                    )
                )
        if int(setting.get("max_num_seqs", 0) or 0) < int(setting.get("batch_size", 1) or 1):
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
        "serving_workloads": 6,
        "vllm": 7,
        "mip": 8,
        "post_mip": 9,
        "output": 10,
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
