# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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
ChosenReplacements: TypeAlias = list[Replacement]

__all__ = ["run_mip"]


def run_mip(
    replacements: dict[ReplacementID, Replacement],
    objective: str,
    constraints: dict[str, float],
    bigger_is_better: bool,
    max_seconds_per_solution: float | None = None,
) -> tuple[ChosenReplacements, float, dict[str, float]]:
    orig_num_replacements = len(replacements)
    replacements = {
        replacement_id: deepcopy(replacement)
        for replacement_id, replacement in replacements.items()
        if math.isfinite(get_nested_key(replacement, objective))
    }
    if len(replacements) < orig_num_replacements:
        warnings.warn(
            f"cuOpt MIP removed {orig_num_replacements - len(replacements)} "
            "replacement(s) with NaN/inf objective value"
        )

    problem = Problem()
    settings = SolverSettings()
    settings.set_parameter("time_limit", float(max_seconds_per_solution or 60.0))

    objective_vars = []
    constraint_vars = {constraint_key: [] for constraint_key in constraints}
    choice_indicators_by_layer = defaultdict(list)
    for replacement_id, replacement in replacements.items():
        is_chosen = problem.addVariable(lb=0, ub=1, vtype=INTEGER)
        replacement["is_chosen"] = is_chosen

        for parent_layer_idx in replacement["parent_layer_indices"]:
            choice_indicators_by_layer[parent_layer_idx].append(is_chosen)

        objective_vars.append(is_chosen * get_nested_key(replacement, objective))
        for constraint_key in constraints:
            constraint_vars[constraint_key].append(
                is_chosen * get_nested_key(replacement, constraint_key)
            )

    for parent_layer_idx, indicators in choice_indicators_by_layer.items():
        problem.addConstraint(sum(indicators) == 1)

    for constraint_key, max_cost in constraints.items():
        min_cost = None
        if isinstance(max_cost, Iterable):
            min_cost, max_cost = max_cost
        if max_cost is not None and math.isfinite(max_cost):
            problem.addConstraint(sum(constraint_vars[constraint_key]) <= max_cost)
        if min_cost is not None and math.isfinite(min_cost):
            problem.addConstraint(sum(constraint_vars[constraint_key]) >= min_cost)

    sense = MAXIMIZE if bigger_is_better else MINIMIZE
    problem.setObjective(sum(objective_vars), sense=sense)
    problem.solve(settings)

    status_name = problem.Status.name
    if status_name not in ("Optimal", "Feasible"):
        return []

    total_value = 0.0
    total_costs = dict.fromkeys(constraints.keys(), 0.0)
    chosen_replacements: ChosenReplacements = []
    chosen_layers = []
    for replacement in replacements.values():
        if replacement["is_chosen"].getValue() < 0.99:
            continue
        chosen_replacements.append(replacement)
        total_value += get_nested_key(replacement, objective)
        for constraint_key in constraints:
            total_costs[constraint_key] += get_nested_key(replacement, constraint_key)
        for parent_layer_idx in replacement["parent_layer_indices"]:
            assert parent_layer_idx not in chosen_layers
            chosen_layers.append(parent_layer_idx)

    missing_layers = set(choice_indicators_by_layer) - set(chosen_layers)
    assert not missing_layers, f"cuOpt MIP did not choose replacements for layers: {missing_layers}"

    for constraint_key, max_cost in constraints.items():
        min_cost = None
        if isinstance(max_cost, Iterable):
            min_cost, max_cost = max_cost
        if max_cost is not None:
            assert total_costs[constraint_key] <= max_cost or math.isclose(
                total_costs[constraint_key],
                max_cost,
                rel_tol=1e-9,
            )
        if min_cost is not None:
            assert total_costs[constraint_key] >= min_cost or math.isclose(
                total_costs[constraint_key],
                min_cost,
                rel_tol=1e-9,
            )

    chosen_replacements = sort_replacements(chosen_replacements)
    for replacement in chosen_replacements:
        del replacement["is_chosen"]
        if "block_config" in replacement:
            replacement["child_block_configs"] = replacement["block_config"]

    return [
        {
            "chosen_replacements": chosen_replacements,
            "total_value": total_value,
            "total_costs": total_costs,
        }
    ]
