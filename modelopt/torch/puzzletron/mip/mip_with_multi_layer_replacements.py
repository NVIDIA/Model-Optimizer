# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Solves multi-layer replacement optimization using Mixed Integer Programming."""

# mypy: ignore-errors
import json
import math
import warnings
from collections import defaultdict
from collections.abc import Hashable, Iterable
from copy import deepcopy
from random import random
from typing import Any, TypeAlias

import pulp

from .utils import consecutive_ngrams, get_nested_key, sort_replacements

__all__ = ["run_mip"]

ReplacementID: TypeAlias = Hashable
Replacement: TypeAlias = dict[str, Any]


def run_mip(
    replacements: dict[ReplacementID, Replacement],
    objective: str,
    constraints: dict[str, float],
    bigger_is_better: bool,
    max_seconds_per_solution: float | None = None,
    num_solutions: int = 1,
    min_hamming_distance: int = 1,
) -> list[dict[str, Any]]:
    if num_solutions < 1:
        raise ValueError("num_solutions must be positive")
    if min_hamming_distance < 1:
        raise ValueError("min_hamming_distance must be positive")
    orig_num_replacements = len(replacements)
    replacements = {
        replacement_id: deepcopy(replacement)
        for replacement_id, replacement in replacements.items()
        if math.isfinite(get_nested_key(replacement, objective))
    }
    if len(replacements) < orig_num_replacements:
        print("\n\n\n")
        warnings.warn(
            f"mip: removed {orig_num_replacements - len(replacements)} replacements "
            "with NaN/inf objective value"
        )
        print("\n\n\n")

    # Create pulp problem with appropriate sense (minimize or maximize)
    sense = pulp.LpMaximize if bigger_is_better else pulp.LpMinimize
    problem = pulp.LpProblem(name="multi_layer_replacement", sense=sense)

    objective_vars = []
    choice_vars = {}
    constraint_vars = {constraint_key: [] for constraint_key in constraints}
    choice_indicators_by_layer = defaultdict(list)
    for i, (replacement_id, replacement) in enumerate(replacements.items()):
        is_chosen = pulp.LpVariable(f"choice_{i}", cat=pulp.LpBinary)
        choice_vars[replacement_id] = is_chosen

        for parent_layer_idx in replacement["parent_layer_indices"]:
            choice_indicators_by_layer[parent_layer_idx].append(is_chosen)

        objective_vars.append(is_chosen * get_nested_key(replacement, objective))

        for constraint_key in constraints:
            constraint_vars[constraint_key].append(
                is_chosen * get_nested_key(replacement, constraint_key)
            )

    # MIP constraints: each parent layer must come from exactly one chosen replacement
    for parent_layer_idx, curr_choice_indicators in choice_indicators_by_layer.items():
        problem += pulp.lpSum(curr_choice_indicators) == 1

    # MIP constraints: the sum of chosen replacement costs must be lower than the max cost
    for constraint_key, max_cost in constraints.items():
        min_cost = None
        if isinstance(max_cost, Iterable):
            min_cost, max_cost = max_cost

        # PuLP is stricter than mip - it doesn't allow NaN/inf in constraints
        if max_cost is not None and math.isfinite(max_cost):
            problem += pulp.lpSum(constraint_vars[constraint_key]) <= max_cost
        if min_cost is not None and math.isfinite(min_cost):
            problem += pulp.lpSum(constraint_vars[constraint_key]) >= min_cost

    # MIP objective
    problem += (pulp.lpSum(objective_vars), "objective")

    # Configure and run solver
    solver = pulp.PULP_CBC_CMD(msg=True, timeLimit=max_seconds_per_solution)
    layer_signatures = {
        replacement_id: _layer_signatures(replacement)
        for replacement_id, replacement in replacements.items()
    }
    all_layers = tuple(sorted(choice_indicators_by_layer))
    solutions = []
    for solution_index in range(num_solutions):
        problem.solve(solver)
        if problem.status != pulp.LpStatusOptimal:
            break
        selected_ids = [
            replacement_id
            for replacement_id, variable in choice_vars.items()
            if variable.varValue is not None and variable.varValue >= 0.99
        ]
        chosen = [replacements[replacement_id] for replacement_id in selected_ids]
        total_value = sum(float(get_nested_key(row, objective)) for row in chosen)
        total_costs = {
            key: sum(float(get_nested_key(row, key)) for row in chosen)
            for key in constraints
        }
        selected_by_layer = {
            layer: signature
            for replacement_id in selected_ids
            for layer, signature in layer_signatures[replacement_id].items()
        }
        missing_layers = set(all_layers) - set(selected_by_layer)
        assert not missing_layers, f"MIP solution is missing layers {sorted(missing_layers)}"
        copied = sort_replacements([deepcopy(row) for row in chosen])
        for row in copied:
            row.pop("is_chosen", None)
            if "block_config" in row:
                row["child_block_configs"] = row["block_config"]
        solutions.append(
            {
                "chosen_replacements": copied,
                "total_value": total_value,
                "total_costs": total_costs,
                "solution_rank": solution_index,
            }
        )
        if solution_index + 1 >= num_solutions:
            break
        required_difference = min_hamming_distance
        if required_difference > len(all_layers):
            break
        matching_terms = []
        for replacement_id, variable in choice_vars.items():
            matches = sum(
                selected_by_layer.get(layer) == signature
                for layer, signature in layer_signatures[replacement_id].items()
            )
            if matches:
                matching_terms.append(matches * variable)
        problem += (
            pulp.lpSum(matching_terms) <= len(all_layers) - required_difference,
            f"diversity_{solution_index}",
        )
    return solutions


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


def usage_example():
    num_layers = 32
    num_options_per_parent_replacement = 5

    replacements = dict()
    for num_layers_in_replacement in (1, 2, 3):
        for i_option in range(num_options_per_parent_replacement):
            for parent_layer_indices in consecutive_ngrams(num_layers, num_layers_in_replacement):
                replacement_id = f"parent layers {parent_layer_indices}  child config {i_option}"
                replacement = {
                    "parent_layer_indices": parent_layer_indices,
                    "metrics": {"loss": random()},
                    "stats": {"memory_mib": random() * 100, "runtime_ms": random() * 10},
                    "replacement_id": replacement_id,
                }
                replacements[replacement_id] = replacement

    constraints = {"stats.memory_mib": num_layers * 15.0, "stats.runtime_ms": num_layers * 1.5}
    (result,) = run_mip(
        replacements,
        objective="metrics.loss",
        constraints=constraints,
        bigger_is_better=False,
    )
    chosen_replacements = result["chosen_replacements"]
    total_value = result["total_value"]
    total_costs = result["total_costs"]

    print()
    print()
    print(f"{total_value=}")
    print(f"{total_costs=}")
    print(f"{constraints=}")
    print("chosen_replacements=")
    print("\n".join([rep["replacement_id"] for rep in chosen_replacements]))


if __name__ == "__main__":
    usage_example()
