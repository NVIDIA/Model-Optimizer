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

"""Standard interface to implement a searcher algorithm.

A searcher is useful whenever we want to search/optimize over a set of hyperparameters in the model.
Searchers are usually used in conjunction with a mode, which can define a search space via its
entrypoints, i.e., convert the model into a search space. The searcher then optimizes over this
search space.
"""

import copy
import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any, final
from warnings import warn

import numpy as np
import pulp
import torch
import torch.nn as nn

from modelopt.torch.utils import distributed as dist
from modelopt.torch.utils import no_stdout, run_forward_loop

LimitsTuple = tuple[float, float]
ConstraintsDict = dict[str, str | float | dict | None]
Deployment = dict[str, str]
ForwardLoop = Callable[[nn.Module], None]
ScoreFunc = Callable[[nn.Module], float]

SearchConfig = dict[str, Any]  # config dict for searcher
SearchStateDict = dict[str, Any]  # state dict for searcher

__all__ = ["BaseSearcher"]


class BaseSearcher(ABC):
    """A basic search interface that can be used to search/optimize a model.

    The base interface supports basic features like setting up a search, checkpointing, and
    loading logic and defines a minimal workflow to follow.
    """

    model: nn.Module
    config: SearchConfig
    constraints: ConstraintsDict
    deployment: Deployment | None
    dummy_input: Any | tuple
    forward_loop: ForwardLoop | None

    @final
    def __init__(self) -> None:
        """We don't allow to override __init__ method."""
        super().__init__()

    # TODO: see if we really want to keep all the config here.
    @property
    def default_search_config(self) -> SearchConfig:
        """Get the default config for the searcher."""
        return {
            "checkpoint": None,
            "verbose": dist.is_master(),
            "forward_loop": None,
            "data_loader": None,
            "collect_func": None,
            "max_iter_data_loader": None,
            "score_func": None,
            "loss_func": None,
            "deployment": None,
        }

    @property
    @abstractmethod
    def default_state_dict(self) -> SearchStateDict:
        """Return default state dict."""

    def sanitize_search_config(self, config: SearchConfig | None) -> SearchConfig:
        """Sanitize the search config dict."""
        # supply with defaults (for verbose we wanna make sure it's on master only)
        config = {**self.default_search_config, **(config or {})}
        config["verbose"] = config["verbose"] and self.default_search_config["verbose"]

        # sanity checks
        assert config.keys() == self.default_search_config.keys(), (
            f"Unexpected config keys: {config.keys() - self.default_search_config.keys()}"
        )

        return config

    # TODO: double-check if we want all these args here.
    @final
    def search(
        self,
        model: nn.Module,
        constraints: ConstraintsDict,
        dummy_input: Any | tuple | None = None,
        config: SearchConfig | None = None,
    ) -> SearchStateDict:
        """Search a given prunable model for the best sub-net and return the search model.

        The best sub-net maximizes the score given by ``score_func`` while satisfying the
        ``constraints``.

        Args:
            model: The converted model to be searched.
            constraints: The dictionary from constraint name to upper bound the searched model has
                to satisfy.
            dummy_input: Arguments of ``model.forward()``. This is used for exporting and
                calculating inference-based metrics, such as latency/FLOPs. The format of
                ``dummy_inputs`` follows the convention of the ``args`` argument in
                `torch.onnx.export <https://pytorch.org/docs/stable/onnx.html#torch.onnx.export>`_.
            config: Additional optional arguments to configure the search.

        Returns: A tuple (subnet, state_dict) where
            subnet is the searched subnet (nn.Module), which can be used for subsequent tasks like
            fine-tuning, state_dict contains the history and detailed stats of the search procedure.
        """
        # check model train state
        is_training = model.training

        # reset the search
        self.reset_search()

        # update and initialize searcher
        self.model = model
        self.config = self.sanitize_search_config(config)
        self.deployment = self.config["deployment"]

        self.constraints = constraints
        self.dummy_input = dummy_input
        self.forward_loop = self.construct_forward_loop(silent=not self.config["verbose"])

        # load checkpoint if it exists
        self.load_search_checkpoint()

        # run initial step and sanity checks before the search
        self.before_search()

        # run actual search
        self.run_search()

        # run clean-up steps after search
        self.after_search()

        # make sure model is in original state
        model.train(is_training)

        # return the config for the best result
        return self.best

    def reset_search(self) -> None:
        """Reset search at the beginning."""
        # reset self.best where we store results
        self.best: SearchStateDict = {}

        # reset state dict (do it afterwards in case best is in state_dict)
        for key, val in self.default_state_dict.items():
            setattr(self, key, copy.deepcopy(val))

    def before_search(self) -> None:
        """Optional pre-processing steps before the search."""

    @abstractmethod
    def run_search(self) -> None:
        """Run actual search."""

    def after_search(self) -> None:
        """Optional post-processing steps after the search."""

    @property
    def has_score(self) -> bool:
        """Check if the model has a score function."""
        return self.config["score_func"] is not None

    def eval_score(self, silent=True) -> float:
        """Optionally silent evaluation of the score function."""
        assert self.has_score, "Please provide `score_func`!"
        score_func: ScoreFunc = self.config["score_func"]
        with no_stdout() if silent else nullcontext():
            return float(score_func(self.model))

    def construct_forward_loop(
        self,
        silent=True,
        progress_bar_msg=None,
        max_iter_data_loader=None,
        post_process_fn=False,
    ) -> ForwardLoop | None:
        """Get runnable forward loop on the model using the provided configs."""
        # check config
        data_loader = self.config["data_loader"]
        forward_loop = self.config.get("forward_loop", None)
        assert None in [data_loader, forward_loop], "Only provide `data_loader` or `forward_loop`!"

        # check trivial case
        if not (data_loader or forward_loop):
            return None

        def forward_loop_with_silence_check(m: nn.Module) -> None:
            with no_stdout() if silent else nullcontext():
                if data_loader is not None:
                    run_forward_loop(
                        m,
                        data_loader=data_loader,
                        max_iters=(
                            max_iter_data_loader
                            if max_iter_data_loader is not None
                            else self.config.get("max_iter_data_loader", None)
                        ),
                        collect_func=self.config["collect_func"],
                        progress_bar=progress_bar_msg,
                        post_process=post_process_fn,
                    )
                elif forward_loop is not None:
                    forward_loop(m)

        return forward_loop_with_silence_check

    @final
    def state_dict(self) -> SearchStateDict:
        """The state dictionary that can be stored/loaded."""
        return {key: getattr(self, key) for key in self.default_state_dict}

    def load_search_checkpoint(self) -> bool:
        """Load function for search checkpoint returning indicator whether checkpoint was loaded."""
        # check if checkpoint exists
        checkpoint: str | None = self.config["checkpoint"]
        if checkpoint is None:
            return False
        if not os.path.exists(checkpoint):
            if dist.is_master():
                warn(f"Checkpoint {checkpoint} does not exist! Initializing from scratch.")
            return False

        # iterate through state dict and load keys
        print(f"Loading searcher state from {checkpoint}...")
        state_dict = torch.load(checkpoint, weights_only=False)
        assert state_dict.keys() == self.state_dict().keys(), "Keys in checkpoint don't match!"
        for key, state in state_dict.items():
            setattr(self, key, state)
        return True

    def save_search_checkpoint(self, verbose=False) -> None:
        """Save function for search checkpoint."""
        # check if save requirements are satisfied
        checkpoint: str | None = self.config["checkpoint"]
        if checkpoint is None or not dist.is_master():
            return

        # save state dict
        if verbose:
            print(f"Saving searcher state to {checkpoint}...")
        save_dirname, _ = os.path.split(checkpoint)
        if save_dirname:
            os.makedirs(save_dirname, exist_ok=True)
        torch.save(self.state_dict(), checkpoint)


class LPS:
    """A wrapper on top of PuLP Linear Programming Solver.

    This solver maximizes/minimizes the candidate scores while meeting the cost constraints.
    """

    def __init__(
        self,
        name: str,
        constraints: dict[str, float | tuple[float]],  # upper bound or (lower, upper) bound
        constraints_to_candidate_costs: dict[str, list[list[float]]],
        candidate_scores: list[list[float]],
        objective_type: str = "minimize",
        verbose: bool = False,
    ):
        """Initialize the LPS solver."""
        assert set(constraints.keys()) == set(constraints_to_candidate_costs.keys())
        assert objective_type in ("minimize", "maximize")
        num_candidates_per_layer = list(map(len, candidate_scores))
        for candidate_costs in constraints_to_candidate_costs.values():
            assert list(map(len, candidate_costs)) == num_candidates_per_layer

        self.name = name
        self.constraints = constraints
        self.constraints_to_candidate_costs = constraints_to_candidate_costs
        self.candidate_scores = candidate_scores
        self.objective_type = pulp.LpMinimize if objective_type == "minimize" else pulp.LpMaximize
        self.solver = pulp.PULP_CBC_CMD(msg=verbose)

        self.num_layers = len(self.candidate_scores)
        self.num_candidates_per_layer = list(map(len, self.candidate_scores))

    def _build_selection_vars(self) -> list[list[pulp.LpVariable]]:
        vars = []
        for li in range(self.num_layers):
            num_candidates = self.num_candidates_per_layer[li]
            layer_vars = [
                pulp.LpVariable(f"z{li}_{ci}", lowBound=0, upBound=1, cat=pulp.LpBinary)
                for ci in range(num_candidates)
            ]
            vars.append(layer_vars)
        return vars

    def _build_objective_problem(
        self, selection_vars: list[list[pulp.LpVariable]]
    ) -> pulp.LpProblem:
        problem = pulp.LpProblem(name=self.name, sense=self.objective_type)
        objective_value = 0
        for layer_id, layer_vars in enumerate(selection_vars):
            objective_value += sum(
                [z * a for z, a in zip(layer_vars, self.candidate_scores[layer_id])]
            )
        problem += (objective_value, "L")
        return problem

    def _build_one_hot_constraints(self, selection_vars: list[list[pulp.LpVariable]]) -> list:
        return [sum(layer_vars) == 1 for layer_vars in selection_vars]

    def _build_budget_constraints(self, selection_vars: list[list[pulp.LpVariable]]) -> list:
        budget_constraints = []
        for (
            constraint_name,
            candidate_costs_list,
        ) in self.constraints_to_candidate_costs.items():
            cost = 0
            for layer_vars, candidate_costs in zip(selection_vars, candidate_costs_list):
                cost += sum([z * b for z, b in zip(layer_vars, candidate_costs)])
            if isinstance(self.constraints[constraint_name], tuple):
                lower_bound, upper_bound = self.constraints[constraint_name]  # type: ignore[misc]
            else:
                lower_bound, upper_bound = None, self.constraints[constraint_name]
            if upper_bound is not None:
                budget_constraints.append(cost <= upper_bound)
            if lower_bound is not None:
                budget_constraints.append(cost >= lower_bound)

        return budget_constraints

    def __call__(self) -> tuple[list[int], str]:
        """Run the solver.

        Returns:
            selections: A list of selected candidate indices per layer.
            status: Status of the solver.
        """
        selection_vars = self._build_selection_vars()

        problem = self._build_objective_problem(selection_vars)
        for one_hot_constraint in self._build_one_hot_constraints(selection_vars):
            problem += one_hot_constraint
        for budget_constraint in self._build_budget_constraints(selection_vars):
            problem += budget_constraint

        problem.solve(self.solver)

        selections = [np.argmax([z.varValue for z in layer_vars]) for layer_vars in selection_vars]
        return selections, pulp.LpStatus[problem.status]
