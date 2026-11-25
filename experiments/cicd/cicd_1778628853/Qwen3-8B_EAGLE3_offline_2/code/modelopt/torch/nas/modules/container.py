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

"""Dynamic sequential implementation (variable depth based on torch.nn.modules.container)."""

from collections import OrderedDict
from collections.abc import Callable
from typing import Any

from torch import nn

from modelopt.torch.opt.dynamic import DynamicModule

from ..registry import DMRegistry
from ..traced_hp import TracedHp

__all__ = ["DynamicModuleList", "_DynamicSequential"]


def _activate_depth(func: Callable) -> Callable:
    """A decorator for enabling dynamic depth within methods in _DynamicSequential.

    The decorated method (and methods within the decorated method) will see any attributes with
    dynamic depth enabled.
    """

    def func_with_dynamic_depth(self: "_DynamicSequential", *args, **kwargs) -> Any:
        """Call func and return value wrapped with temporarily enabling dynamic depth."""
        val = self._dynamic_depth
        self._dynamic_depth = True
        ret = func(self, *args, **kwargs)
        self._dynamic_depth = val
        return ret

    return func_with_dynamic_depth


@DMRegistry.register({nn.Sequential: "nn.Sequential"})
class _DynamicSequential(DynamicModule):
    """An ``nn.Sequential`` layer with dynamic hyperparams and variable ``depth``."""

    _dynamic_depth: bool

    @_activate_depth
    def forward(self, input):
        """Forward with activated variable depth."""
        return super().forward(input)

    @_activate_depth
    def export(self) -> nn.Module:
        """Export with activated variable depth."""
        return super().export()

    @_activate_depth
    def __repr__(self):
        """__repr__ with dynamic depth enabled -> str(self) will only show active subnet."""
        return super().__repr__()

    @_activate_depth
    def extra_repr(self):
        """extra_repr with dynamic depth enabled -> str(self) will only show active subnet."""
        return super().extra_repr()

    @staticmethod
    def _get_modules(mod: "_DynamicSequential", modules: OrderedDict) -> OrderedDict:
        if mod.depth < len(modules) and mod._dynamic_depth:
            return OrderedDict(list(modules.items())[: mod.depth])
        return modules

    def _setup(self):
        # register temp attribute to keep track of whether dynamic depth is on
        self._register_temp_attribute("_dynamic_depth", False)

        # register hyperparameters
        self._register_hparam("depth", TracedHp(list(range(len(self) + 1))))

        # register _modules as a dynamic attribute
        self._register_dynamic_attribute("_modules", self._get_modules)

    def modify(self, *, min_depth: int = 0):
        """Modify the dynamic choices of the module according to provided keyword arguments.

        Args:
            min_depth: The minimum depth of the module.
        """
        hp = self.get_hparam("depth")
        hp.choices = [d for d in hp.choices if d >= min_depth]


# NOTE: We provide a parent class since we do not register to DMRegistry and explicitly convert a module if needed.
class DynamicModuleList(DynamicModule, nn.ModuleList):
    """An ``nn.ModuleList`` container with dynamic hyperparams and variable ``depth``.

    Unlike _DynamicSequential, this module supports sorting/reordering of modules based on
    importance in addition to variable depth.
    """

    def _setup(self):
        # register hyperparameters
        self._register_hparam("depth", TracedHp(list(range(1, len(self) + 1))))

        # register _modules as a dynamic attribute
        self._register_dynamic_attribute("_modules", self._get_modules)

    @staticmethod
    def _get_modules(mod: "DynamicModuleList", modules: dict) -> dict:
        """Get modules with dynamic depth and ordering applied based on active_slice."""
        hp = mod.get_hparam("depth")
        active_slice = hp.active_slice

        items = list(modules.items())

        if isinstance(active_slice, slice):
            active_items = items[active_slice]
        else:
            active_items = [items[idx] for idx in active_slice.tolist()]

        # Re-create dict with keys as str(index) from 0 to len(active_items)
        return {str(i): module for i, (_, module) in enumerate(active_items)}
