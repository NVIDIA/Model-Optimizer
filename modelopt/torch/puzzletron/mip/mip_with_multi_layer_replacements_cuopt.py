# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import math
import warnings
from collections import defaultdict
from collections.abc import Hashable, Iterable
from copy import deepcopy
from typing import Any, TypeAlias

from cuopt.linear_programming.problem import INTEGER, MAXIMIZE, MINIMIZE, Problem
from cuopt.linear_programming.solver_settings import SolverSettings

from .utils import get_nested_key, sort_replacements

ReplacementID: TypeAlias = Hashable
Replacement: TypeAlias = dict[str, Any]

__all__ = ["run_mip"]


def _signature(value: Any) -> str:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    return json.dumps(value, sort_keys=True, default=str)


def _layer_signatures(replacement: Replacement) -> dict[int, str]:
    layers = tuple(int(value) for value in replacement["parent_layer_indices"])
    children = replacement.get("child_block_configs")
    if children is None:
        children = replacement.get("block_config")
    if not isinstance(children, (list, tuple)) or len(children) != len(layers):
        children = [children] * len(layers)
    return {layer: _signature(child) for layer, child in zip(layers, children)}


def run_mip(
    replacements: dict[ReplacementID, Replacement],
    objective: str,
    constraints: dict[str, float],
    bigger_is_better: bool,
    max_seconds_per_solution: float | None = None,
    num_solutions: int = 1,
    min_hamming_distance: int = 1,
) -> list[dict[str, Any]]:
    """Solve and enumerate layer-configuration-diverse cuOpt solutions."""

    if num_solutions < 1:
        raise ValueError("num_solutions must be positive")
    if min_hamming_distance < 1:
        raise ValueError("min_hamming_distance must be positive")
    original_count = len(replacements)
    replacements = {
        replacement_id: deepcopy(replacement)
        for replacement_id, replacement in replacements.items()
        if math.isfinite(get_nested_key(replacement, objective))
    }
    if len(replacements) < original_count:
        warnings.warn(
            f"cuOpt MIP removed {original_count - len(replacements)} replacement(s) "
            "with NaN/inf objective value"
        )

    problem = Problem()
    settings = SolverSettings()
    if max_seconds_per_solution is not None:
        settings.set_parameter("time_limit", float(max_seconds_per_solution))
    objective_vars = []
    choice_vars = {}
    constraint_vars = {key: [] for key in constraints}
    choices_by_layer = defaultdict(list)
    for replacement_id, replacement in replacements.items():
        variable = problem.addVariable(lb=0, ub=1, vtype=INTEGER)
        choice_vars[replacement_id] = variable
        for layer in replacement["parent_layer_indices"]:
            choices_by_layer[layer].append(variable)
        objective_vars.append(variable * get_nested_key(replacement, objective))
        for key in constraints:
            constraint_vars[key].append(variable * get_nested_key(replacement, key))

    for variables in choices_by_layer.values():
        problem.addConstraint(sum(variables) == 1)
    for key, bound in constraints.items():
        minimum, maximum = (bound if isinstance(bound, Iterable) else (None, bound))
        if maximum is not None and math.isfinite(maximum):
            problem.addConstraint(sum(constraint_vars[key]) <= maximum)
        if minimum is not None and math.isfinite(minimum):
            problem.addConstraint(sum(constraint_vars[key]) >= minimum)
    problem.setObjective(
        sum(objective_vars), sense=MAXIMIZE if bigger_is_better else MINIMIZE
    )

    layer_signatures = {
        replacement_id: _layer_signatures(replacement)
        for replacement_id, replacement in replacements.items()
    }
    all_layers = tuple(sorted(choices_by_layer))
    solutions = []
    for solution_index in range(num_solutions):
        problem.solve(settings)
        if problem.Status.name not in ("Optimal", "Feasible"):
            break
        selected_ids = [key for key, variable in choice_vars.items() if variable.getValue() >= 0.99]
        chosen = [replacements[key] for key in selected_ids]
        selected_by_layer = {
            layer: signature
            for key in selected_ids
            for layer, signature in layer_signatures[key].items()
        }
        missing = set(all_layers) - set(selected_by_layer)
        assert not missing, f"cuOpt solution is missing layers {sorted(missing)}"
        copied = sort_replacements([deepcopy(row) for row in chosen])
        for row in copied:
            if "block_config" in row:
                row["child_block_configs"] = row["block_config"]
        solutions.append(
            {
                "chosen_replacements": copied,
                "total_value": sum(float(get_nested_key(row, objective)) for row in chosen),
                "total_costs": {
                    key: sum(float(get_nested_key(row, key)) for row in chosen)
                    for key in constraints
                },
                "solution_rank": solution_index,
            }
        )
        if solution_index + 1 >= num_solutions:
            break
        required_difference = min_hamming_distance
        if required_difference > len(all_layers):
            break
        terms = []
        for key, variable in choice_vars.items():
            matches = sum(
                selected_by_layer.get(layer) == signature
                for layer, signature in layer_signatures[key].items()
            )
            if matches:
                terms.append(matches * variable)
        problem.addConstraint(sum(terms) <= len(all_layers) - required_difference)
    return solutions
