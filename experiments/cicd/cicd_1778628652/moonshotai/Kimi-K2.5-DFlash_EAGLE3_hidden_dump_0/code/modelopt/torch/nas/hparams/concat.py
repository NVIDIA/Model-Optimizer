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

"""Hparam that represent concat."""

from collections import defaultdict
from collections.abc import Callable, Iterator
from itertools import product
from math import prod
from warnings import warn

import numpy as np
import torch

from modelopt.torch.trace import ConcatSymbol, Symbol

from ..traced_hp import TracedHp, TracedHpRegistry

__all__ = ["ConcatTracedHp"]


@TracedHpRegistry.register(ConcatSymbol)
class ConcatTracedHp(TracedHp):
    """Concat hparam that can stitch together input hparams."""

    _inputs: list[TracedHp]
    _sum_to_combo: dict[int, tuple[int]]
    _hp_start_idx: torch.LongTensor

    @staticmethod
    def _choice_combos(*all_choices) -> Iterator[tuple[int, ...]]:
        # compute total number of combinations
        n_combos = prod(len(c_list) for c_list in all_choices)

        # we don't wanna iterate over more than that to keep it fast
        n_max = 5e7  # takes 4s

        # if we have less than n_max combinations, we can iterate over all of them
        if n_combos <= n_max:
            # use itertools.product and iterate over all combinations
            yield from product(*all_choices)
            return
        else:
            warn(f"ConcatTracedHp: {n_combos=} is larger than {n_max=}. Pruning combinations.")

        # otherwise, we use a pruned set of combinations based on immediate vicinity of each value

        # range of choices
        c_min = min(min(c_list) for c_list in all_choices)
        c_max = max(max(c_list) for c_list in all_choices)

        # figure out for each list what's the closest index for each value in the choice range
        idx_lookup = {
            c: [min(range(len(c_list)), key=lambda i: abs(c_list[i] - c)) for c_list in all_choices]
            for c in range(c_min, c_max + 1)
        }

        # solve equation for r such that n_combos = (c_max - c_min) * (2r+1)^n \approx n_max
        # use r = max(1, r) to avoid edge cases
        r = max(int((n_max / (c_max - c_min)) ** (1 / len(all_choices)) - 1) // 2, 1)

        # yield reduced number of combinations (note that we might have duplicates; hence we filter)
        # each combination is based on the product of choices centered around the value closest to
        # the target value c. We use r to control the size of the neighborhood.
        processed = set()
        for c in range(c_min, c_max + 1):
            all_choices_reduced = []
            for i, c_list in enumerate(all_choices):
                idx_closest = idx_lookup[c][i]
                all_choices_reduced.append(c_list[max(0, idx_closest - r) : idx_closest + r + 1])
            for combo in product(*all_choices_reduced):
                if combo not in processed:
                    processed.add(combo)
                    yield combo

    def _get_sum_to_combo(self) -> dict[int, tuple[int, ...]]:
        """Return dict mapping sum to individual hparam active values."""
        # If two hparams are the same, we need their values to be the same and we need to track that
        hps_unique = list(set(self._inputs))
        idxs_unique = [hps_unique.index(hp) for hp in self._inputs]

        # Keep the combination with the smallest difference between the individual values
        # e.g. for sum 48, prefer [24, 24] over [16, 32] or [32, 16]
        sum_to_combo = {}
        sum_to_cost = defaultdict(lambda: float("inf"))
        for combo in self._choice_combos(*[hp.choices for hp in hps_unique]):
            # get full combo and sum
            combo_full = tuple(combo[idx] for idx in idxs_unique)
            s = sum(combo_full)
            # check cost (sum of absolute differences between average and individual values)
            cost = sum(abs(c - s / len(combo_full)) for c in combo_full)
            # update if better
            if cost < sum_to_cost[s]:
                sum_to_cost[s] = cost
                sum_to_combo[s] = combo_full
        return sum_to_combo

    def _set_hp_start_idx(self) -> None:
        """Compute start indices for input hps."""
        # NOTE: the last index is the length of the concatenated hp
        hp_start_idx = np.concatenate(([0], np.cumsum([hp.max for hp in self._inputs])))
        self._hp_start_idx = torch.asarray(hp_start_idx, dtype=torch.long)

    @property
    def active(self) -> int:
        """Return the sum of active values of all hparams."""
        active = 0
        for hp in self._inputs:
            active += hp.active
        return active

    @active.setter
    def active(self, val: int | None):
        """Set the active value with a sanity check for choices and dynamic hparams."""
        val = self.original if val is None else val
        assert val in self._choices, f"val = {val}, choices = {self.choices}"
        assert val in self._sum_to_combo, f"Invalid value: {val} not in {self._sum_to_combo.keys()}"
        for hp_in, v in zip(self._inputs, self._sum_to_combo[val]):
            assert v in hp_in.choices
            hp_in._active = v

    @property
    def active_slice(self) -> TracedHp.ActiveSlice:
        """Return the currently active sorted indices or slice corresponding to the active value.

        Example:
            hp1(max=4, active=2), hp2(max=4, active=4)
            Then the active indices in the concatenated hp are [0, 1, 4, 5, 6, 7]
        """
        # get vanilla slice first (with sanity check)
        as_simple = super().active_slice
        assert isinstance(as_simple, slice), "ConcatTracedHp does not support its own slice order!"

        # get the list of slices as active indices in the concatenated hp
        as_hps = [hp.active_slice for hp in self._inputs]
        as_hps = [torch.arange(_as.stop)[_as] if isinstance(_as, slice) else _as for _as in as_hps]

        # concatenate the slices with correct offsets
        active_slice = torch.cat([_as + self._hp_start_idx[i] for i, _as in enumerate(as_hps)])

        # check if active_slice corresponds to the vanilla slice
        if torch.equal(active_slice, torch.arange(as_simple.stop)[as_simple]):
            return as_simple

        return active_slice

    def _get_importance(self) -> TracedHp.Importance:
        # compute the regular importance (this hparam + registered dependencies)
        imp_total = super()._get_importance()

        # now we add in the importance from the individual hparams contributing to the concat node
        # concat inputs are not registered as dependencies, so we need to add them here
        # note that we use ``hp.importance`` instead of the private API ``hp._get_importance()``
        # here. Why? We can have the situation where only one of the concat inputs is searchable.
        # We need to account for that and compute the "partial" importances hence!
        imps_cat = [hp.importance if hp.is_configurable else None for hp in self._inputs]

        # clean up importances that are None
        imps_cat = [imp or torch.zeros(hp.max) for hp, imp in zip(self._inputs, imps_cat)]

        # Let's split imp_total
        imps_split = torch.tensor_split(imp_total, self._hp_start_idx[1:-1])

        # Added split imps
        imps = [imp + imp_cat.to(imp) for imp, imp_cat in zip(imps_split, imps_cat)]  # type: ignore[union-attr]

        # We need to aggregate between split importances when the come from the same hparam!
        imps = [
            sum([imp_ for imp_, hp in zip(imps, self._inputs) if hp is self._inputs[i]])
            for i, imp in enumerate(imps)
        ]

        # concatenate and return
        return torch.cat(imps)

    def _split_order(self, order: torch.Tensor) -> list[torch.Tensor]:
        """Split the order tensor into the individual order tensors for each input hparam."""
        return [
            (
                order[torch.logical_and(order >= start, order < start + hp.max)] - start
                if hp.is_configurable and hp.is_sortable
                else torch.arange(hp.max, device=order.device)
            )
            for hp, start in zip(self._inputs, self._hp_start_idx[:-1])
        ]

    @torch.no_grad()
    def _enforce_order(self, order: torch.Tensor | None = None) -> None:
        """Enforcing the order.

        Note that we only support intra-group sorting but **not** inter-group sorting. Inter-group
        sorting would not be compatible with the concat operation and is thus not well-defined.

        E.g., consider a concat operation with multiple inputs from convs. Then a channel that is
        currently assigned to the first group (i.e. first conv) cannot be resorted to appear in any
        other group/conv. Hence, we need to ensure that we maintain inter-group consistency.
        """
        # vanilla case
        if order is None:
            for hp in self._inputs:
                hp._enforce_order(order)  # private API is fine here since it is just None
            return super()._enforce_order(order)

        # we have to group the order by the hparams to enforce order on each group separately with
        # the correct offset subtracted
        orders_split = self._split_order(order)

        # ensure that orders are consistent if they come from the same hparam
        for hp, _order in zip(self._inputs, orders_split):
            assert all(
                torch.equal(_order, order_)
                for hp_, order_ in zip(self._inputs, orders_split)
                if hp is hp_
            )

        # enforce order now ...
        # note that we use the public API here for extra sanity checks
        for hp, partial_order in zip(self._inputs, orders_split):
            if hp.is_configurable and hp.is_sortable:
                hp.enforce_order(partial_order)

        # recombine the order into a single tensor without inter-group sorting for dependencies
        order_intra_only = torch.cat(
            [partial + start for partial, start in zip(orders_split, self._hp_start_idx[:-1])]
        )

        # sort dependencies now
        super()._enforce_order(order_intra_only)
        self._slice_order = None  # we don't store slice_order in a ConcatTracedHp

    def _resolve_dependencies(
        self, sym: Symbol, get_hp: Callable[[Symbol], TracedHp]
    ) -> dict[Symbol, TracedHp]:
        # check that we have the correct symbol
        assert type(sym) is ConcatSymbol, f"Sym should be {type(self)}: {sym}."

        # construct input hparams
        # NOTE: input syms are hidden and corresponding registered symbol should be last dependency!
        # --> if that's not the case we cannot retrieve the hps needed.
        assert all(isinstance(s, ConcatSymbol.Input) for s in sym.input_syms)
        assert all(s._dependencies[-1]._parent == s for s in sym.input_syms)
        self._inputs = [get_hp(s._dependencies[-1]) for s in sym.input_syms]

        def _get_hp_with_input(s: Symbol) -> TracedHp:
            assert type(sym) is ConcatSymbol
            for s_, hp in zip(sym.input_syms, self._inputs):
                if s is s_:
                    return hp
            return get_hp(s)

        # resolve regular dependencies of all input hparams
        mapping: dict[Symbol, TracedHp] = {}
        for hp_in, sym_in in zip(self._inputs, sym.input_syms):
            mapping.update(hp_in._resolve_dependencies(sym_in, _get_hp_with_input))

        # remove input syms from mapping...
        for sym_in in set(sym.input_syms):
            mapping.pop(sym_in)

        # resolve regular dependencies
        mapping.update(super()._resolve_dependencies(sym, get_hp))

        # compute sum_to_combo values and make sure they are consistent with choices
        sum_to_combo = {s: val for s, val in self._get_sum_to_combo().items() if s in self.choices}
        self._sum_to_combo = sum_to_combo
        self.choices = list(self._sum_to_combo)

        # update choices for inputs from sum_to_combo and make them non-configurable
        for i, hp_in in enumerate(self._inputs):
            hp_in.choices = [combo[i] for combo in sum_to_combo.values()]
            hp_in._is_configurable = False

        self._set_hp_start_idx()

        return mapping

    def reset_choices(self) -> None:
        """Reset the choices of the concat hparam.

        Useful if we want to reset choices after input hparam choices are changed during modify().
        """
        self._sum_to_combo = self._get_sum_to_combo()
        self._set_hp_start_idx()
        with self._force_configurable():
            self.choices = list(self._sum_to_combo)


def build_concat_hp(inputs: list[TracedHp]):
    """Initialize a non-configurable concat hparam from a list of input hparams.

    One key difference from ConcatTracedHp via tracing is that in ConcatTracedHp, the input hparams
    are not configurable, and only the concatenated hparam is configurable. In build_concat_hp, its the opposite.

    This is useful for building concat hparams from a list of configurable input hparams instead of
    tracing (e.g. for megatron language model DynamicModule).
    """
    concat_hp = object.__new__(ConcatTracedHp)
    concat_hp._inputs = inputs
    concat_hp._sum_to_combo = concat_hp._get_sum_to_combo()
    concat_hp._set_hp_start_idx()

    choices = list(concat_hp._sum_to_combo)
    concat_hp.__init__(choices)  # type: ignore[misc]
    concat_hp._is_configurable = False
    concat_hp._importance_estimators = None

    return concat_hp
