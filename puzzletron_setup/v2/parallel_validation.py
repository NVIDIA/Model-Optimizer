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

"""Stage-aware model geometry and parallelism compatibility checks."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

from puzzletron_orchestrator import normalize_vllm_topology

__all__ = [
    "ParallelCompatibilityIssue",
    "geometry_scope",
    "validate_automodel_parallelism",
    "validate_vllm_parallelism",
]

GeometryScope = Literal["teacher", "candidate", "none"]

_TEACHER_STAGE_IDS = frozenset(
    {
        "depth_importance",
        "width_importance",
        "sort_sanity",
        "bypass",
        "bypass_sanity",
    }
)
_CANDIDATE_STAGE_IDS = frozenset(
    {
        "width_sanity",
        "slicing_sanity",
        "replacement_scoring",
        "vllm_stats",
    }
)
_CANDIDATE_POST_MIP_TYPES = frozenset({"evaluation", "global_kd", "aiperf"})


@dataclass(frozen=True)
class ParallelCompatibilityIssue:
    """One actionable conflict between a stage topology and model geometry."""

    path: str
    message: str


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _value(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def geometry_scope(stage_id: str, node_type: str | None = None) -> GeometryScope:
    """Return the geometry domain loaded by one static or post-MIP stage."""
    if node_type in _CANDIDATE_POST_MIP_TYPES:
        return "candidate"
    name = str(stage_id).rsplit(".", 1)[-1]
    if name in _TEACHER_STAGE_IDS:
        return "teacher"
    if name in _CANDIDATE_STAGE_IDS:
        return "candidate"
    return "none"


def _facts(inventory: Any) -> Mapping[str, Any]:
    return _mapping(_value(inventory, "facts", {}))


def _axes(inventory: Any) -> dict[str, Any]:
    return {
        str(_value(axis, "axis_id")): axis
        for axis in (_value(inventory, "axes", ()) or ())
        if _value(axis, "axis_id")
    }


def _domain(
    inventory: Any,
    pruning: Mapping[str, Any],
    axis_id: str,
    *,
    scope: GeometryScope,
    fallback: int | None = None,
) -> tuple[int, ...]:
    axis = _axes(inventory).get(axis_id)
    teacher = _value(axis, "teacher_value") if axis is not None else fallback
    values = {int(teacher)} if teacher is not None and int(teacher) > 0 else set()
    if scope == "candidate":
        selected = _mapping(_mapping(pruning).get("axes")).get(axis_id)
        selected = _mapping(selected)
        if selected and bool(selected.get("enabled", True)):
            values.update(int(value) for value in selected.get("values", ()) if int(value) > 0)
    return tuple(sorted(values))


def _geometry(
    inventory: Any,
    pruning: Mapping[str, Any],
    scope: GeometryScope,
) -> dict[str, tuple[int, ...] | tuple[tuple[int, int], ...]]:
    facts = _facts(inventory)
    axes = _axes(inventory)
    query_heads = facts.get("num_attention_heads")
    kv_heads = facts.get("num_key_value_heads")
    teacher_q_per_kv = None
    if query_heads is not None and kv_heads is not None and int(kv_heads) != 0:
        teacher_q_per_kv = int(query_heads) // int(kv_heads)
    kv_axis = "kv_heads" if "kv_heads" in axes else "kv_groups"
    kv_domain = _domain(
        inventory,
        pruning,
        kv_axis,
        scope=scope,
        fallback=int(kv_heads) if kv_heads is not None else None,
    )
    if "query_heads" in axes:
        query_domain = _domain(
            inventory,
            pruning,
            "query_heads",
            scope=scope,
            fallback=int(query_heads) if query_heads is not None else None,
        )
        attention_pairs = tuple(
            sorted(
                (query_count, kv_count)
                for query_count in query_domain
                for kv_count in kv_domain
                if query_count % kv_count == 0
            )
        )
    else:
        q_per_kv_domain = _domain(
            inventory,
            pruning,
            "q_heads_per_group",
            scope=scope,
            fallback=teacher_q_per_kv,
        )
        attention_pairs = tuple(
            sorted(
                {
                    (kv_count * q_per_kv, kv_count)
                    for kv_count in kv_domain
                    for q_per_kv in q_per_kv_domain
                }
            )
        )
        query_domain = tuple(sorted({query for query, _ in attention_pairs}))

    gdn_key_heads = _domain(
        inventory,
        pruning,
        "gdn_key_groups",
        scope=scope,
    )
    gdn_value_per_group = _domain(
        inventory,
        pruning,
        "gdn_value_heads_per_group",
        scope=scope,
    )
    gdn_value_heads = tuple(
        sorted(
            {
                key_heads * value_per_group
                for key_heads in gdn_key_heads
                for value_per_group in gdn_value_per_group
            }
        )
    )
    return {
        "hidden widths": _domain(
            inventory,
            pruning,
            "hidden_width",
            scope=scope,
            fallback=(int(facts["hidden_size"]) if facts.get("hidden_size") is not None else None),
        ),
        "attention_pairs": attention_pairs,
        "query-head counts": query_domain,
        "KV-head counts": kv_domain,
        "PLE widths": _domain(
            inventory,
            pruning,
            "ple_width",
            scope=scope,
        ),
        "FFN intermediate widths": _domain(
            inventory,
            pruning,
            "ffn_intermediate",
            scope=scope,
            fallback=(
                int(facts["intermediate_size"])
                if facts.get("intermediate_size") is not None
                else None
            ),
        ),
        "expert counts": _domain(
            inventory,
            pruning,
            "moe_experts",
            scope=scope,
            fallback=(int(facts["num_experts"]) if facts.get("num_experts") is not None else None),
        ),
        "expert intermediate widths": _domain(
            inventory,
            pruning,
            "moe_expert_intermediate",
            scope=scope,
        ),
        "shared-expert intermediate widths": _domain(
            inventory,
            pruning,
            "moe_shared_expert_intermediate",
            scope=scope,
        ),
        "MoE latent widths": _domain(
            inventory,
            pruning,
            "moe_latent_dim",
            scope=scope,
        ),
        "GDN key-head counts": gdn_key_heads,
        "GDN value-head counts": gdn_value_heads,
        "Mamba head counts": _domain(
            inventory,
            pruning,
            "mamba_heads",
            scope=scope,
        ),
    }


def _valid_divisors(values: tuple[int, ...]) -> list[int]:
    common = 0
    for value in values:
        common = math.gcd(common, int(value))
    return [candidate for candidate in range(1, common + 1) if common % candidate == 0]


def _divisibility_issue(
    *,
    stage_id: str,
    setting: str,
    degree: int,
    label: str,
    values: tuple[int, ...],
) -> ParallelCompatibilityIssue | None:
    if not values or all(value % degree == 0 for value in values):
        return None
    return ParallelCompatibilityIssue(
        f"{stage_id}.{setting}",
        f"{setting.upper()}={degree} is incompatible with {label} {list(values)}; "
        f"valid choices {_valid_divisors(values)}.",
    )


def validate_automodel_parallelism(
    profile: Any,
    inventory: Any,
    pruning: Mapping[str, Any] | None,
    *,
    stage_id: str,
    sequence_length: int,
    node_type: str | None = None,
) -> tuple[ParallelCompatibilityIssue, ...]:
    """Return every AutoModel conflict for the stage's reachable geometries."""
    issues: list[ParallelCompatibilityIssue] = []
    dimensions = {
        "tp": int(_value(profile, "tp", 1)),
        "cp": int(_value(profile, "cp", 1)),
        "pp": int(_value(profile, "pp", 1)),
        "dp_shard": int(_value(profile, "dp_shard", 1)),
        "dp_replicate": int(_value(profile, "dp_replicate", 1)),
        "ep": int(_value(profile, "ep", 1)),
    }
    invalid = {name: value for name, value in dimensions.items() if value < 1}
    if invalid:
        issues.append(
            ParallelCompatibilityIssue(
                stage_id,
                f"Parallel dimensions must be positive: {invalid}.",
            )
        )
        return tuple(issues)
    if dimensions["dp_shard"] % dimensions["ep"]:
        issues.append(
            ParallelCompatibilityIssue(
                f"{stage_id}.ep",
                "DP-shard must be divisible by EP because EP overlays the FSDP "
                f"shard axis; got DP-shard={dimensions['dp_shard']}, "
                f"EP={dimensions['ep']}.",
            )
        )
    if bool(_value(profile, "sequence_parallel", False)) and dimensions["tp"] == 1:
        issues.append(
            ParallelCompatibilityIssue(
                f"{stage_id}.sequence_parallel",
                "Sequence parallelism requires TP greater than one.",
            )
        )
    if int(sequence_length) % dimensions["cp"]:
        issues.append(
            ParallelCompatibilityIssue(
                f"{stage_id}.cp",
                f"CP={dimensions['cp']} does not divide sequence length {int(sequence_length)}.",
            )
        )

    scope = geometry_scope(stage_id, node_type)
    if scope == "none":
        return tuple(issues)
    geometry = _geometry(inventory, _mapping(pruning), scope)
    for label in (
        "hidden widths",
        "query-head counts",
        "KV-head counts",
        "PLE widths",
        "FFN intermediate widths",
        "expert intermediate widths",
        "shared-expert intermediate widths",
        "MoE latent widths",
        "GDN key-head counts",
        "GDN value-head counts",
        "Mamba head counts",
    ):
        issue = _divisibility_issue(
            stage_id=stage_id,
            setting="tp",
            degree=dimensions["tp"],
            label=label,
            values=cast("tuple[int, ...]", geometry[label]),
        )
        if issue is not None:
            issues.append(issue)

    moe = bool(_value(inventory, "moe", False))
    if not moe and dimensions["ep"] != 1:
        issues.append(
            ParallelCompatibilityIssue(
                f"{stage_id}.ep",
                f"Dense models require EP=1; got EP={dimensions['ep']}.",
            )
        )
    elif moe:
        issue = _divisibility_issue(
            stage_id=stage_id,
            setting="ep",
            degree=dimensions["ep"],
            label="expert counts",
            values=cast("tuple[int, ...]", geometry["expert counts"]),
        )
        if issue is not None:
            issues.append(issue)
    return tuple(sorted(issues, key=lambda issue: (issue.path, issue.message)))


def validate_vllm_parallelism(
    topology: Mapping[str, Any],
    inventory: Any,
    pruning: Mapping[str, Any] | None,
    *,
    stage_id: str,
) -> tuple[ParallelCompatibilityIssue, ...]:
    """Return every vLLM conflict for candidate geometries served by a stage."""
    try:
        canonical = normalize_vllm_topology(topology)
    except (TypeError, ValueError) as error:
        return (ParallelCompatibilityIssue(f"{stage_id}.topology", str(error)),)
    geometry = _geometry(inventory, _mapping(pruning), "candidate")
    issues: list[ParallelCompatibilityIssue] = []
    tp_issue = _divisibility_issue(
        stage_id=stage_id,
        setting="tp",
        degree=canonical["tp"],
        label="query-head counts",
        values=cast("tuple[int, ...]", geometry["query-head counts"]),
    )
    if tp_issue is not None:
        issues.append(tp_issue)

    dcp = canonical["decode_cp"]
    if dcp > 1:
        for query_heads, kv_heads in cast(
            "tuple[tuple[int, int], ...]", geometry["attention_pairs"]
        ):
            q_per_kv = query_heads // kv_heads
            if canonical["tp"] <= kv_heads or dcp > canonical["tp"] // kv_heads or q_per_kv % dcp:
                issues.append(
                    ParallelCompatibilityIssue(
                        f"{stage_id}.decode_cp",
                        f"DCP={dcp} is incompatible with query-head count "
                        f"{query_heads}, KV-head count {kv_heads}, and TP="
                        f"{canonical['tp']}.",
                    )
                )

    moe = bool(_value(inventory, "moe", False))
    if canonical["enable_expert_parallel"] and not moe:
        issues.append(
            ParallelCompatibilityIssue(
                f"{stage_id}.enable_expert_parallel",
                "vLLM expert parallelism can be enabled only for an MoE model.",
            )
        )
    elif canonical["enable_expert_parallel"]:
        experts = cast("tuple[int, ...]", geometry["expert counts"])
        if experts and any(value % canonical["effective_ep"] for value in experts):
            issues.append(
                ParallelCompatibilityIssue(
                    f"{stage_id}.enable_expert_parallel",
                    f"effective EP={canonical['effective_ep']} (TP * DP) is "
                    f"incompatible with expert counts {list(experts)}; valid "
                    f"effective EP choices {_valid_divisors(experts)}.",
                )
            )
    return tuple(sorted(issues, key=lambda issue: (issue.path, issue.message)))
