# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Candidate-domain filtering and homogeneous Puzzletron solution ranking."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from copy import deepcopy
from itertools import product
from typing import Any

from ..block_config import AttentionConfig, MambaConfig, MoEConfig
from .utils import get_nested_key

__all__ = ["filter_replacements_by_axes", "rank_homogeneous_solutions"]


_AXIS_ALIASES = {
    "mamba_head_dim": "mamba.head_dim",
    "mamba_num_heads": "mamba.num_heads",
    "moe_intermediate_size": "moe.expert_intermediate_size",
    "moe_shared_expert_intermediate_size": "moe.shared_expert_intermediate_size",
    "n_routed_experts": "moe.num_experts",
    "num_experts_per_tok": "moe.top_k",
    "num_key_value_heads": "attention.num_kv_heads",
    "q_per_group": "attention.q_per_group",
}

_KNOWN_AXES = frozenset(
    {
        *_AXIS_ALIASES.values(),
        "moe.latent_dim",
        "mamba.state_dim",
        "ffn.intermediate_size",
        "mla.q_lora_rank",
        "mla.kv_lora_rank",
    }
)


def _canonical_axis(axis: str) -> str:
    axis = _AXIS_ALIASES.get(axis, axis)
    if axis not in _KNOWN_AXES:
        raise ValueError(f"unknown MIP search-space axis {axis!r}")
    return axis


def _block_axis_values(block_config) -> dict[str, Any]:
    values = {}
    for subblock in block_config.subblock_configs:
        if subblock.no_op:
            continue
        prefix = subblock.kind
        if isinstance(subblock, AttentionConfig):
            if subblock.num_kv_heads is not None:
                values[f"{prefix}.num_kv_heads"] = subblock.num_kv_heads
            if (
                subblock.num_query_heads is not None
                and subblock.num_kv_heads is not None
            ):
                values[f"{prefix}.q_per_group"] = (
                    subblock.num_query_heads // subblock.num_kv_heads
                )
        elif isinstance(subblock, MoEConfig):
            for field in (
                "num_experts",
                "expert_intermediate_size",
                "shared_expert_intermediate_size",
                "top_k",
                "latent_dim",
            ):
                value = getattr(subblock, field)
                if value is not None:
                    values[f"{prefix}.{field}"] = value
        elif isinstance(subblock, MambaConfig):
            for field in ("num_heads", "head_dim", "state_dim"):
                value = getattr(subblock, field)
                if value is not None:
                    values[f"{prefix}.{field}"] = value
        else:
            for field in (
                "intermediate_size",
                "q_lora_rank",
                "kv_lora_rank",
            ):
                value = getattr(subblock, field, None)
                if value is not None:
                    values[f"{prefix}.{field}"] = value
    return values


def _layer_index(replacement: Mapping[str, Any]) -> int:
    layers = replacement.get("parent_layer_indices")
    if layers is None and "block_idx" in replacement:
        layers = [replacement["block_idx"]]
    if not isinstance(layers, (list, tuple)) or len(layers) != 1:
        raise ValueError("search-space filtering requires one-layer replacements")
    return int(layers[0])


def _teacher_axes(replacements: Mapping[Any, Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    teachers = {}
    for replacement in replacements.values():
        if replacement.get("is_teacher", False):
            teachers[_layer_index(replacement)] = _block_axis_values(
                replacement["block_config"]
            )
    missing = sorted({_layer_index(row) for row in replacements.values()} - set(teachers))
    if missing:
        raise ValueError(f"search-space filtering is missing teacher blocks for layers {missing}")
    return teachers


def _selector_accepts(selector: Any, value: Any, teacher_value: Any) -> bool:
    if selector == "all":
        return True
    if selector is None or selector == "teacher":
        return value == teacher_value
    if isinstance(selector, Mapping):
        if set(selector) != {"range"} or len(selector["range"]) != 2:
            raise ValueError("axis selector range must contain [min, max]")
        lower, upper = selector["range"]
        return lower <= value <= upper
    if isinstance(selector, Iterable) and not isinstance(selector, (str, bytes, Mapping)):
        return value in selector
    return value == selector


def filter_replacements_by_axes(
    replacements: Mapping[Any, Mapping[str, Any]],
    *,
    axes_default: str = "all",
    axis_options: Mapping[str, Any] | None = None,
    teacher_replacements: Mapping[Any, Mapping[str, Any]] | None = None,
) -> dict[Any, dict[str, Any]]:
    """Restrict candidate axes while preserving layer and teacher identities."""

    if axes_default not in {"all", "teacher"}:
        raise ValueError("axes_default must be all or teacher")
    options = {
        _canonical_axis(str(axis)): selector
        for axis, selector in dict(axis_options or {}).items()
    }
    if axes_default == "all" and not options:
        return {key: deepcopy(value) for key, value in replacements.items()}
    teachers = _teacher_axes(teacher_replacements or replacements)
    known_axes = {
        axis
        for replacement in replacements.values()
        for axis in _block_axis_values(replacement["block_config"])
    }
    selectors = {}
    if axes_default == "teacher":
        selectors.update(dict.fromkeys(known_axes, "teacher"))
    selectors.update(options)

    filtered = {}
    for replacement_id, replacement in replacements.items():
        layer = _layer_index(replacement)
        candidate = _block_axis_values(replacement["block_config"])
        teacher = teachers[layer]
        accepted = True
        for axis, selector in selectors.items():
            if axis not in candidate:
                continue
            if not _selector_accepts(selector, candidate[axis], teacher.get(axis)):
                accepted = False
                break
        if accepted:
            filtered[replacement_id] = deepcopy(replacement)

    before = {_layer_index(row) for row in replacements.values()}
    after = {_layer_index(row) for row in filtered.values()}
    if before != after:
        raise ValueError(
            "search-space restrictions removed every candidate for layers "
            f"{sorted(before - after)}"
        )
    return filtered


def _constraint_satisfied(total: float, bound: Any) -> bool:
    if isinstance(bound, Iterable) and not isinstance(bound, (str, bytes, Mapping)):
        minimum, maximum = bound
    else:
        minimum, maximum = None, bound
    if minimum is not None and total < minimum and not math.isclose(total, minimum):
        return False
    if maximum is not None and total > maximum and not math.isclose(total, maximum):
        return False
    return True


def _constraint_is_effective(bound: Any) -> bool:
    if isinstance(bound, Iterable) and not isinstance(bound, (str, bytes, Mapping)):
        values = bound
    else:
        values = (bound,)
    return any(value is not None and math.isfinite(float(value)) for value in values)


def rank_homogeneous_solutions(
    replacements: Mapping[Any, Mapping[str, Any]],
    *,
    objective: str,
    constraints: Mapping[str, Any],
    bigger_is_better: bool,
    num_solutions: int,
) -> list[dict[str, Any]]:
    """Rank uniform-per-axis solutions without invoking one MIP per assignment."""

    if num_solutions == 0:
        return []
    if num_solutions < -1:
        raise ValueError("num_solutions must be -1 or non-negative")
    by_layer = defaultdict(list)
    axis_domains = defaultdict(set)
    for replacement_id, replacement in replacements.items():
        layer = _layer_index(replacement)
        values = _block_axis_values(replacement["block_config"])
        objective_value = float(get_nested_key(replacement, objective))
        row = (
            replacement_id,
            replacement,
            values,
            objective_value,
            str(replacement["block_config"]),
        )
        by_layer[layer].append(row)
        for axis, value in values.items():
            axis_domains[axis].add(value)
    axes = tuple(sorted(axis_domains))
    domains = [tuple(sorted(axis_domains[axis], key=str)) for axis in axes]

    def preferred(row, incumbent) -> bool:
        row_key = (row[3], str(row[0]))
        incumbent_key = (incumbent[3], str(incumbent[0]))
        return row_key > incumbent_key if bigger_is_better else row_key < incumbent_key

    # A candidate only constrains axes present in its block type. Pre-index the
    # best candidate for every (present axes, projected values) pair so each
    # Cartesian assignment performs a handful of dictionary lookups instead
    # of rescanning every candidate in every layer.
    indexed_by_layer = {}
    for layer, rows in by_layer.items():
        groups = defaultdict(dict)
        for row in rows:
            if not math.isfinite(row[3]):
                continue
            present_axes = tuple(sorted(row[2]))
            projected = tuple(row[2][axis] for axis in present_axes)
            incumbent = groups[present_axes].get(projected)
            if incumbent is None or preferred(row, incumbent):
                groups[present_axes][projected] = row
        indexed_by_layer[layer] = tuple(groups.items())

    solutions = []
    seen_architectures = set()
    effective_constraints = {
        key: bound for key, bound in constraints.items() if _constraint_is_effective(bound)
    }
    assignment_count = math.prod(len(domain) for domain in domains)
    for assignment_index, selected in enumerate(product(*domains), start=1):
        if (
            assignment_index == 1
            or assignment_index == assignment_count
            or assignment_index % 1000 == 0
        ):
            print(
                "[homogeneous] "
                f"assignments={assignment_index}/{assignment_count} "
                f"feasible_unique={len(solutions)}",
                flush=True,
            )
        assignment = dict(zip(axes, selected))
        chosen = []
        feasible = True
        for layer in sorted(indexed_by_layer):
            candidates = []
            for present_axes, lookup in indexed_by_layer[layer]:
                projected = tuple(assignment[axis] for axis in present_axes)
                candidate = lookup.get(projected)
                if candidate is not None:
                    candidates.append(candidate)
            if not candidates:
                feasible = False
                break
            best = candidates[0]
            for candidate in candidates[1:]:
                if preferred(candidate, best):
                    best = candidate
            chosen.append(best)
        if not feasible:
            continue
        architecture = tuple(row[4] for row in chosen)
        if architecture in seen_architectures:
            continue
        effective_costs = {
            key: sum(float(get_nested_key(row[1], key)) for row in chosen)
            for key in effective_constraints
        }
        if not all(
            _constraint_satisfied(effective_costs[key], bound)
            for key, bound in effective_constraints.items()
        ):
            continue
        seen_architectures.add(architecture)
        solutions.append(
            {
                # Keep shared references while ranking. Only retained results
                # are copied below; top-k search must not clone thousands of
                # discarded full replacement/stat records.
                "chosen_replacements": [row[1] for row in chosen],
                "total_value": sum(row[3] for row in chosen),
                "homogeneous_assignment": assignment,
            }
        )

    solutions.sort(
        key=lambda solution: (
            solution["total_value"],
            str(solution["homogeneous_assignment"]),
        ),
        reverse=bigger_is_better,
    )
    retained = solutions if num_solutions == -1 else solutions[:num_solutions]
    for solution in retained:
        replacements_to_copy = solution["chosen_replacements"]
        solution["total_costs"] = {
            key: sum(
                float(get_nested_key(replacement, key))
                for replacement in replacements_to_copy
            )
            for key in constraints
        }
        solution["chosen_replacements"] = [
            deepcopy(replacement) for replacement in replacements_to_copy
        ]
    return retained
