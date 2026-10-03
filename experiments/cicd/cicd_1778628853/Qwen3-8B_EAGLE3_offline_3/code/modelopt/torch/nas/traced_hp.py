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

"""Module to specify the basic hyperparameter with tracing capabilities."""

from collections.abc import Callable

from modelopt.torch.opt.hparam import Hparam
from modelopt.torch.trace import Symbol

__all__ = ["TracedHp", "TracedHpRegistry"]

TracedHparamType = type["TracedHp"]


class TracedHpRegistry:
    """A simple registry to keep track of different traced hp classes and their symbols."""

    _registry: dict[type[Symbol], TracedHparamType] = {}  # registered classes

    @classmethod
    def register(cls, sym_cls: type[Symbol]) -> Callable[[TracedHparamType], TracedHparamType]:
        """Use this to register a new traced hparam class for the provided symbol class.

        Usage:

        .. code-block:: python

            @TracedHpRegistry.register(MySymbol)
            class MyHparam(TracedHp):
                pass
        """

        def decorator(hp_class: TracedHparamType) -> TracedHparamType:
            """Register hp_class with appropriate sym_class."""
            assert sym_cls not in cls._registry, f"{sym_cls} already registered!"
            cls._registry[sym_cls] = hp_class
            return hp_class

        return decorator

    @classmethod
    def unregister(cls, sym_cls: type[Symbol]) -> None:
        """Unregister a previously registered symbol class.

        It throws a KeyError if the hparam class is not registered.
        """
        if sym_cls not in cls._registry:
            raise KeyError(f"{sym_cls} is not registered!")
        cls._registry.pop(sym_cls)

    @classmethod
    def initialize_from(cls, sym: Symbol, hp: Hparam) -> "TracedHp":
        """Initialize the sym-appropriate hparam from a vanilla hparam."""
        return cls._registry[type(sym)].initialize_from(hp)

    @classmethod
    def get(cls, sym: Symbol) -> TracedHparamType | None:
        """Get Hparam type associated with symbol."""
        return cls._registry.get(type(sym))


@TracedHpRegistry.register(Symbol)
class TracedHp(Hparam):
    """A hparam that exhibits additional functionality required to handle tracing."""

    @classmethod
    def initialize_from(cls, hp: Hparam) -> "TracedHp":
        """Initialize a new hparam from an existing vanilla Trace."""
        # sanity checks
        assert type(hp) is TracedHp, f"Can only initialize from an {TracedHp} object."
        # relegate implementation to child class
        return cls._initialize_from(hp)

    @classmethod
    def _initialize_from(cls, hp: Hparam) -> "TracedHp":
        """Initialize a new hparam from an existing vanilla Hparam."""
        hp.__class__ = cls
        assert isinstance(hp, cls)
        return hp

    def resolve_dependencies(
        self, sym: Symbol, get_hp: Callable[[Symbol], "TracedHp"]
    ) -> dict[Symbol, "TracedHp"]:
        """Resolve dependencies of the hparam via symbolic map.

        This method iterates through the dependency map described in sym and generates an
        appropriate hparam based on the currently assigned hparam and the parent symbol.

        Args:
            sym: The symbol associated with self for which we want to resolve dependencies.
            get_hp: A function that returns the hparam associated with a symbol.

        Returns:
            A mapping describing the hparam that should be associated with each symbol.
        """
        # dependency resolution only works for searchable symbols and must be associated with self
        assert sym.is_searchable or sym.is_constant, f"Symbol {sym} must be searchable or constant!"
        assert get_hp(sym) is self, f"Symbol {sym} must be associated with self to resolve!"
        assert TracedHpRegistry.get(sym) is type(self), "Symbol and Hparam type must match!"

        # relegate implementation to child class
        return self._resolve_dependencies(sym, get_hp)

    def _resolve_dependencies(
        self, sym: Symbol, get_hp: Callable[[Symbol], "TracedHp"]
    ) -> dict[Symbol, "TracedHp"]:
        # check sortability of sym
        if not sym.is_sortable:
            self._importance_estimators = None

        # process dependencies and generate hparams
        to_process = set(sym._dependencies)
        processed: set[Symbol] = set()
        mapping: dict[Symbol, TracedHp] = {sym: self}

        while to_process:
            # get a new symbol
            sym_dep = to_process.pop()
            processed.add(sym_dep)

            # this should never be any other symbol
            assert type(sym_dep) is Symbol, f"Unexpected type {type(sym_dep)} for {sym_dep}!"

            # merge hparam into self and store mapping
            self &= get_hp(sym_dep)
            mapping[sym_dep] = self

            # add dependencies of sym_dep to to_process
            to_process |= set(sym_dep._dependencies) - processed

        # check constant case at the end
        if sym.is_constant:
            self.active = self.original
            self.choices = [self.original]

        return mapping
