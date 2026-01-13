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
import math
import warnings
from collections import defaultdict
from collections.abc import Hashable, Iterable
from copy import deepcopy
from random import random
from typing import Any, TypeAlias

from mip import BINARY, Model, maximize, minimize, xsum

from modelopt.torch.puzzletron.mip.utils import (
    consecutive_ngrams,
    get_nested_key,
    sort_replacements,
)

ReplacementID: TypeAlias = Hashable
Replacement: TypeAlias = dict[str, Any]
ChosenReplacements: TypeAlias = list[Replacement]


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
        print("\n\n\n")
        warnings.warn(
            f"mip: removed {orig_num_replacements - len(replacements)} replacements with NaN/inf objective value"
        )
        print("\n\n\n")

    mip_model = Model()

    objective_vars = []
    constraint_vars = {constraint_key: [] for constraint_key in constraints}
    choice_indicators_by_layer = defaultdict(list)
    for replacement_id, replacement in replacements.items():
        is_chosen = mip_model.add_var(var_type=BINARY)
        replacement["is_chosen"] = is_chosen

        for parent_layer_idx in replacement["parent_layer_indices"]:
            choice_indicators_by_layer[parent_layer_idx].append(is_chosen)

        objective_vars.append(is_chosen * get_nested_key(replacement, objective))

        for constraint_key in constraints:
            constraint_vars[constraint_key].append(
                is_chosen * get_nested_key(replacement, constraint_key)
            )

    # MIP constraints: each parent layer must come from exactly one chosen replacement
    for parent_layer_idx, curr_choice_indicators in choice_indicators_by_layer.items():
        mip_model += xsum(curr_choice_indicators) == 1

    # MIP constraints: the sum of chosen replacement costs must be lower than the max cost
    for constraint_key, max_cost in constraints.items():
        min_cost = None
        if isinstance(max_cost, Iterable):
            min_cost, max_cost = max_cost

        if max_cost is not None:
            mip_model += xsum(constraint_vars[constraint_key]) <= max_cost
        if min_cost is not None:
            mip_model += xsum(constraint_vars[constraint_key]) >= min_cost

    # MIP objective
    mip_model.objective = (
        maximize(xsum(objective_vars)) if bigger_is_better else minimize(xsum(objective_vars))
    )

    if max_seconds_per_solution is not None:
        mip_model.max_seconds = max_seconds_per_solution

    mip_model.optimize()

    if is_chosen.x is None:
        return []
        # raise InfeasibleError()

    # Trust But Verify: calculate total value and costs, and check that all the constraints are filled
    total_value = 0.0
    total_costs = dict.fromkeys(constraints.keys(), 0)
    chosen_replacements: ChosenReplacements = []
    chosen_layers = []
    for replacement_id, replacement in replacements.items():
        is_chosen = replacement["is_chosen"].x >= 0.99
        if is_chosen:
            assert replacement not in chosen_replacements
            chosen_replacements.append(replacement)
            total_value += get_nested_key(replacement, objective)
            for constraint_key in constraints:
                total_costs[constraint_key] += get_nested_key(replacement, constraint_key)
            for parent_layer_idx in replacement["parent_layer_indices"]:
                assert parent_layer_idx not in chosen_layers
                chosen_layers.append(parent_layer_idx)

    missing_layers = set(choice_indicators_by_layer.keys()) - set(chosen_layers)
    assert len(missing_layers) == 0, (
        f"The following layers were not chosen by any replacement:\n{missing_layers=}\n{chosen_replacements}"
    )

    for constraint_key, max_cost in constraints.items():
        min_cost = None
        if isinstance(max_cost, Iterable):
            min_cost, max_cost = max_cost

        if max_cost is not None:
            assert total_costs[constraint_key] < max_cost or math.isclose(
                total_costs[constraint_key], max_cost, rel_tol=1e-9
            ), (
                f"This max_cost was violated {constraint_key} in the solution, sol val={total_costs[constraint_key]} > {max_cost=}"
            )
        if min_cost is not None:
            assert total_costs[constraint_key] > min_cost or math.isclose(
                total_costs[constraint_key], min_cost, rel_tol=1e-9
            ), (
                f"This min_cost was violated {constraint_key} in the solution, sol val={total_costs[constraint_key]} < {min_cost=}"
            )

    chosen_replacements = sort_replacements(chosen_replacements)
    for cr in chosen_replacements:
        del cr["is_chosen"]  # not copyable, will cause errors in deep copy
        if "block_config" in cr:
            cr["child_block_configs"] = cr["block_config"]
        # del cr['block_config'] for now the dump includes both keys (duplicated values) # we might wanna either delete one of them or keep both
        # I prefer keeping block_config and deleting 'child_block_configs' from previous puzzle steps

    return [
        {
            "chosen_replacements": chosen_replacements,
            "total_value": total_value,
            "total_costs": total_costs,
        }
    ]


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
