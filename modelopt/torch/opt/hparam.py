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

"""Standard hyperparameter class for regular symbol."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from typing import Union

import torch

__all__ = ["HPType", "Hparam"]


class CustomHPType(ABC):
    """Custom hyperparameter type base class for user-defined hparam types."""

    @abstractmethod
    def __repr__(self) -> str:
        """Return string representation with relevant properties of the class."""

    @abstractmethod
    def __lt__(self, other) -> bool:
        """Return True if self is less than or equal to other."""

    @abstractmethod
    def __eq__(self, other) -> bool:
        """Return True if self is equal to other."""


HPType = tuple[int, ...] | int | float | CustomHPType


# NOTE: Hparam class should be picklable for distributed broadcasting of hparams for Megatron pruning.
class Hparam:
    """A base hyperparameter of a DynamicModule.

    Keeps track of hyperparameter values and their importance, which can be used for search algorithms.
    """

    Importance = Union[torch.Tensor, None]  # noqa: UP007
    ImportanceEstimator = Callable[[], Importance]
    ActiveSlice = Union[slice, torch.LongTensor]  # noqa: UP007

    def __init__(self, choices: Sequence[HPType], original: HPType | None = None) -> None:
        """Initializes Hparam with original value and choices."""
        self._original = max(choices) if original is None else original
        self._choices = sorted(set(choices) | {self.original})
        self._active = self.original
        self._is_configurable = True  # in case we want to manually overwrite configurability.
        # Additional attributes for hacking megatron dynamic modules to simplify implementation.
        self._importance_is_order = (
            False  # whether the importance is overwritten to be active slice order
        )

        # Callback to compute hparam importance
        self._importance_estimators: list[Hparam.ImportanceEstimator] | None = [
            self._default_get_importance
        ]

        # an optional order to enforce for active_slice
        self._slice_order: torch.LongTensor | None = None

    def __iter__(self) -> Generator[HPType, None, None]:
        """Iterate over choices."""
        yield from self.choices

    @property
    def is_configurable(self):
        """Return whether the hparam is configurable."""
        return self._is_configurable and len(self.choices) > 1

    @contextmanager
    def _force_configurable(self):
        """Context manager to temporarily set hparam to be configurable."""
        original_value = self._is_configurable
        self._is_configurable = True
        yield
        self._is_configurable = original_value

    @property
    def is_sortable(self):
        """Return whether hparam in sortable."""
        return self._importance_estimators is not None

    @property
    def active(self) -> HPType:
        """Return the currently active value."""
        return self._active

    @active.setter
    def active(self, val: HPType | None):
        """Set the active value with a sanity check for choices and dynamic hparams."""
        val = self.original if val is None else val
        assert val in self._choices, f"val = {val}, choices = {self.choices}"
        if self.is_configurable:
            self._active = val
        else:
            assert self._active == val

    @property
    def active_slice(self) -> ActiveSlice:
        """Return the currently active sorted indices or slice corresponding to the active value."""
        assert isinstance(self.active, int), "active_slice only supported for int hparams"
        if self._slice_order is not None:
            return self._slice_order[: self.active]
        return slice(self.active)

    @property
    def choices(self) -> Sequence[HPType]:
        """Return available choices."""
        return self._choices

    @choices.setter
    def choices(self, val: Sequence[HPType]):
        """Set available choices as subset of current choices (original must be preserved!)."""
        # sanity checks
        assert self.original in val, f"Original choice not in choices: {self.original} not in {val}"
        assert self.active in val, f"Active choice not in choices: {self.active} not in {val}"
        curr = set(self.choices)
        val_set = set(val)

        # update local choices
        if self.is_configurable:
            assert val_set <= curr, f"New choices must be a subset: current: {curr}, new: {val_set}"
            self._choices = sorted(val_set)
        else:
            assert val_set == curr, f"Cannot update choices: current {curr}, new: {val_set}"

    def reset_choices(self) -> None:
        """Reset the choices of the hparam."""

    @property
    def min(self) -> HPType:
        """Return min value from among choices."""
        return min(self.choices)

    @property
    def max(self) -> HPType:
        """Return max value from among choices."""
        return max(self.choices)

    @property
    def original(self) -> HPType:
        """Return original value from among choices."""
        return self._original

    @property
    @torch.no_grad()
    def importance(self) -> Importance:
        """Computes and returns the normalized importance among the features the hparam represents.

        Note that the importance is represented as a 1d-tensor with the length equal to the max
        choice (Hparam.max) of the hparam.

        For example, if the hparam represents the number of in_channels to a Conv2d layer, the
        importance should be a 1d-tensor of importance score with length equal to the number of
        in_channels.

        Note that each module should register appropriate importance callbacks to compute the
        actual importance associated with the hparam choices. If there is no notion of importance
        for the hparam, this function returns None.
        """
        # TODO: will we ever need a cycle detector here?
        assert self.is_configurable
        return self._get_importance() if self.is_sortable else None

    # TODO: eventually importance should be registered by the search algorithm and _not_ by the
    # module implementation. This is a temporary solution to get things working...
    def register_importance(self, importance_estimator: ImportanceEstimator):
        """Register importance estimator for the hparam.

        This estimator does not take any arguments and should return a single argument (
        optional 1d-tensor) representing the importance among features the hparam represents.
        If the return argument is a tensor, the length of the tensor must be equal to the max choice
        (Hparam.max) of the hparam.
        """
        if self._importance_estimators is None:  # use instead of self.is_sortable for mypy
            raise RuntimeError("Cannot register importance for unsortable hparams.")
        else:
            self._importance_estimators.append(importance_estimator)

    def _default_get_importance(self) -> Importance:
        """Default importance estimator."""
        return None

    def _get_importance(self) -> Importance:
        """Private method to retrieve importance without sanity checks."""
        # iterate though all importance estimators
        imps_all = []
        for estimator in self._importance_estimators or []:
            imp = estimator()
            if imp is not None:
                assert len(imp) == self.max, "Length of importance must be equal to max choice!"
                imps_all.append(imp)

        if self._importance_is_order:
            assert len(imps_all) == 1, (
                "Only one importance estimator is supported when importance is an order"
            )
            return imps_all[0]

        imps_all = [imp / (imp.max() + 1e-9) for imp in imps_all]  # normalize importance
        return sum(imps_all) if imps_all else None

    @torch.no_grad()
    def enforce_order(self, order: torch.Tensor | None = None) -> None:
        """Store a reference to this order and enforce the order for active_slice.

        This function enables the user to enforce an order how the active_slice is generated.

        Example:
            If the hparam has active value 16 and the max value is 32, the active_slice by default
            will be ``slice(16)``, which is equivalent to ``range(16)`` (although faster).
            When order is set active_slice will instead return ``self._order[:16]``.

        TODO: will we ever need a cycle detector here?
        """
        if order is not None:
            # convert to long (torch's index type)
            order = order.long()

            # check if the order is valid
            assert torch.equal(torch.arange(self.max, device=order.device), torch.sort(order)[0]), (
                "order must be a permutation of range(self.max) to be valid!"
            )

        self._enforce_order(order)

    def _enforce_order(self, order: torch.Tensor | None = None) -> None:
        """Private method to enforce order without sanity checks."""
        if isinstance(order, torch.Tensor):  # Always store the order as a CPU tensor
            order = order.cpu()
        self._slice_order = order

    @property
    def attrs(self) -> list[str]:
        """Return the attributes of the hparam for repr."""
        return ["choices", "active", "original"]

    def __repr__(self) -> str:
        """Return string representation with relevant properties of the class."""
        return f"{type(self).__name__}({', '.join(f'{x}={getattr(self, x)}' for x in self.attrs)})"

    def __iand__(self, hp: "Hparam"):
        """Merge another hparam into self."""
        assert isinstance(hp, (Hparam)), f"Cannot merge {type(hp)} into {type(self)}!"

        # merge choices
        self.choices = sorted(set(self.choices) & set(hp.choices))

        # remove slice order
        self._slice_order = None

        # merge importance estimators
        if isinstance(self._importance_estimators, list) and isinstance(
            hp._importance_estimators, list
        ):
            self._importance_estimators.extend(hp._importance_estimators)
        else:
            self._importance_estimators = None

        return self
