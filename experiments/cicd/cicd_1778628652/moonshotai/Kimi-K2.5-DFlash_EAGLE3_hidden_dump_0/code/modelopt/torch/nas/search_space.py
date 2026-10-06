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

"""An extension of the DynamicSpace to handle searchable units and related utilities."""

import fnmatch
from typing import Any

import torch
import torch.nn as nn

from modelopt.torch.opt import RulesDict
from modelopt.torch.opt.dynamic import DynamicModule, DynamicSpace, _DMRegistryCls
from modelopt.torch.trace import Symbol, SymMap, analyze_symbols
from modelopt.torch.utils import print_rank_0, random
from modelopt.torch.utils.distributed import is_master, rank

from .registry import DMRegistry
from .traced_hp import TracedHp, TracedHpRegistry

__all__ = ["SearchSpace", "generate_search_space"]

SampleFunc = random.FSample | dict[str, random.FSample]


class SearchSpace(DynamicSpace):
    """Stub for search space interface."""

    def __init__(self, model: nn.Module, sym_map: SymMap | None = None) -> None:
        """Initialize the search space from the model and its associated symbolic map."""
        super().__init__(model)
        self._sym_map = sym_map
        if self._sym_map:
            assert self._sym_map.model is self.model, "Model and sym_map's model must match."

    def _sym_to_hp(self, sym: Symbol) -> TracedHp:
        assert isinstance(self._sym_map, SymMap), "Model's sym_map was not provided!"
        for name, sym_ in self._sym_map.named_symbols():
            if sym is sym_:
                mod, _, hp_name = name.rpartition(".")
                mod = self.model.get_submodule(mod)
                assert isinstance(mod, DynamicModule)
                hp = mod.get_hparam(hp_name)
                assert isinstance(hp, TracedHp), f"Expected TracedHp, got {type(hp)}!"
                return hp
        raise KeyError(f"Symbol {sym} not found in sym_map!")

    def _should_be_converted(self, mod: nn.Module) -> bool:
        return self._sym_map is None or mod in self._sym_map

    def generate(self, rules: RulesDict | None = None) -> None:
        """Generate the search space from the model + symbolic map."""
        # sanity checks
        assert isinstance(self._sym_map, SymMap), "Model's sym_map was not provided!"

        # 1. + 2. convert to dynamic module and apply rules
        mods_converted = self.convert_to_dynamic(rules, DMRegistry)

        # 3. iterate through searchable/constant symbols and generate hparams for each
        #    dependency tree.
        sym_to_hp_all: dict[Symbol, TracedHp] = {}
        for name, sym in self._sym_map.named_symbols(searchable=True, constant=True):
            mod_name, _, sym_name = name.rpartition(".")
            mod = self.model.get_submodule(mod_name)
            assert isinstance(mod, DynamicModule), f"Module {mod} must be a DynamicModule!"

            hp = TracedHpRegistry.initialize_from(sym, mod.get_hparam(sym_name))
            setattr(mod, sym_name, hp)
            sym_to_hp = hp.resolve_dependencies(sym, self._sym_to_hp)
            sym_to_hp_all.update(sym_to_hp)

        # 4. sanity check that we cover all symbols and hyperparameters
        # NOTE: symbols and hparams of the non-leaf modules will not be included in the search space
        symbols_all = dict(self._sym_map.named_symbols())
        assert set(sym_to_hp_all.keys()) == set(symbols_all.values()), "Missing symbols!"

        # 5. iterate through all dynamic modules and reassign hparams accordingly
        for mod_name, mod in mods_converted.items():
            for sym_name, sym in self._sym_map[mod].items():
                # set new hp
                setattr(mod, sym_name, sym_to_hp_all[sym])

    def sample(self, sample_func: SampleFunc = random.choice) -> dict[str, Any]:
        """Sample configurable hparams using the provided sample_func and return resulting config.

        Args:
            model: A configurable model that contains one or more DynamicModule(s).
            sample_func: A sampling function for hyperparameters. If a
                dict is provided, the key indicates a Unix shell-style wildcard pattern which is
                used to match the hyperparameter name. The default is random sampling corresponding
                to ``{"*": random.choice}``.

                .. note::

                    Later matches take precedence over earlier ones when a dict with wildcards is
                    provided.

        Returns:
            A dict of ``(parameter_name, choice)`` that specifies an active subnet.
        """
        # wrap sample_func into a dict if it's not already
        if not isinstance(sample_func, dict):
            sample_func = {"*": sample_func}

        # sample configurable hparams
        f_sample: random.FSample | None
        for hp_name, hp in self.named_hparams(configurable=True):
            for pattern, f_sample in sample_func.items():
                if fnmatch.fnmatch(hp_name, pattern):
                    break
            else:
                f_sample = None
            if f_sample:
                hp.active = f_sample(hp.choices)

        return self.config()

    @torch.no_grad()
    def sort_parameters(self, hps_to_sort: set[str] | None = None, verbose: bool = False) -> None:
        """A graph propagation based parameter sorting algorithm.

        Args:
            hps_to_sort: A set of hparam names to sort. If not provided or empty, all hparams will be sorted.
            verbose: Whether to print the search space and hparam importances.
        """
        print_rank_0("\nSorting parameters...")

        # get config and set to max
        config = self.config()
        self.sample(sample_func=max)

        # sort configurable hparams
        for name, hp in self.named_hparams(configurable=True):
            hp_name = name.split(".")[-1]
            if hps_to_sort and hp_name not in hps_to_sort:
                continue
            importance = hp.importance
            # no sorting to be done (maybe because there is nothing to sort or it's unsortable)
            if importance is None:
                continue
            # compute order from importance and enforce it
            # NOTE: use .argsort() instead of torch.argsort() so hp can overwrite the behavior
            order = importance if hp._importance_is_order else importance.argsort(descending=True)
            hp.enforce_order(order)
            if verbose:
                print(
                    f"Sorted {name} for rank {rank()} with "
                    f"{'order' if hp._importance_is_order else 'importance'}={importance}"
                )

        # go back to old config
        self.select(config)

    def export(self, dm_registry: _DMRegistryCls = DMRegistry) -> nn.Module:
        """Export with default registry."""
        return super().export(dm_registry)

    def print_summary(self, skipped_hparams: list[str] = ["kernel_size"]) -> None:
        """Print a summary of the search space."""
        if not is_master():
            return
        print(f"\nSearch Space Summary for rank {rank()}:\n{'-' * 100}")
        hp_visited = set()  # Only highlight configurable hparams once
        for name, hp in self.named_hparams():
            if not any(name.endswith(s) for s in skipped_hparams):
                choices_to_print = (
                    [*hp.choices[:4], "...", *hp.choices[-4:]]
                    if len(hp.choices) > 8
                    else hp.choices
                )
                if hp.is_configurable and hp not in hp_visited:
                    hp_visited.add(hp)
                    prefix = "*"
                else:
                    prefix = " "
                print(f"{prefix} {name:80} {choices_to_print}")

        print("{:-^100}".format(""))


def generate_search_space(model: nn.Module, rules: RulesDict | None = None) -> SearchSpace:
    """Patch model with dynamic units and generate/return the search space.

    Args:
        model: The model to patched for which we want to generate the search space.
        rules: An optional dict specifying the rules used during the search space generation.
            If not provided, the largest possible search space is generated. An example rule:

            .. code-block:: python

                rules = {
                    "nn.Sequential": {"min_depth": 2},
                    "nn.Conv2d": {
                        "*": {
                            "channels_ratio": [1],
                            "channel_divisor": 16,
                            "kernel_size": [],
                        },
                        "backbone.stages.[1-5]*.spatial_conv*": {
                            "channels_ratio": [0.334, 0.5, 0.667, 1],
                        },
                        "det_blocks.*.spatial_conv*": {
                            "channels_ratio": [0.667, 1],
                        },
                    },
                    "nn.Linear": {},  # uses default arguments for Linear
                    "nn.BatchNorm2d": {},
                    "nn.SyncBatchNorm": {},
                }

            .. note::

                When rules are provided, the search space generation is restricted to the rules that
                are provided. Other module types will not be included in the search space.

            .. note::

                Rules with ``None``, e.g., ``{"nn.Sequential": None}``, will also cause the
                conversion to be skipped just like when no rules are provided for the layer type.

            .. note::

                Instead of specifying the module type as a string, the module type can also be
                specified as a key directly, e.g., ``{nn.Sequential: None}``.

    Returns:
        The patched model with dynamic units that describe the resulting search space.

    Generally, the search space generation follows the following outline of steps:

        1. Analyze the model to identify the dynamic units that can be patched via modelopt.torch.trace.
        2. Patch the model with the dynamic units accordingly.
        3. Generate a consistent search space from the patched model by ensuring consistency between
           dynamic units according to the symbolic map obtained from the model analysis.

    Upon return of the model, the search space can be analyzed and modified via the utilities
    provided in ``modelopt.torch.dynamic.SearchSpace``.
    """
    sym_map = analyze_symbols(model)
    search_space = SearchSpace(model, sym_map)
    search_space.generate(rules=rules)
    return search_space
