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

"""Utilities to describe symbols found in common torch modules."""

from collections import deque
from collections.abc import Callable, Generator, Iterable
from enum import Enum, auto

import torch.nn as nn

from .tracer import RobustTracer

__all__ = ["SymInfo", "SymMap", "Symbol"]


class Symbol:
    """A symbolic parameter (``Symbol``) of a ``SymModule``.

    An example of a Symbol could be the kernel_size of a conv.

    Note that a symbol can have the following states (mutually exclusive):
    - free: the symbol is not bound to any value
    - searchable: the symbol is free and can be searched over
    - constant: the symbol's value cannot be changed
    - dynamic: the symbol's value is determined by its parent symbol

    In addition, a symbol can exhibit properties related to its cross-layer significance:
    - incoming: the symbol depends on the input tensor to the module
    - outgoing: the module's output tensor depends on the symbol
    - none: the symbol is not cross-layer significant (only affects the internals of the module)

    Based on these basic properties, we define a few useful compound properties:
    - is_cross_layer: the symbol is incoming or outgoing
    - is_dangling: the symbol is free and cross_layer
    """

    _extra_repr_attrs = []

    class CLType(Enum):
        """Cross-layer type for the symbol."""

        NONE = auto()
        INCOMING = auto()
        OUTGOING = auto()

        def __eq__(self, other: object) -> bool:
            """Two enums are not equal if they are NONE."""
            if self.value == Symbol.CLType.NONE.value:
                return False
            return super().__eq__(other)

    def __init__(
        self,
        is_searchable: bool = False,
        is_sortable: bool = True,
        cl_type: CLType = CLType.NONE,
        elastic_dims: set[int] | None = None,
    ):
        """Initializes Symbol with tracing-relevant information."""
        self._is_searchable = is_searchable
        self._is_sortable = is_sortable
        self._cl_type = cl_type
        self._elastic_dims = elastic_dims or set()
        self._is_constant = False

        # Parent Symbol for a dynamic symbol
        self._parent: Symbol | None = None

        # list of Symbols that have this symbol as a dependency
        self._dependencies: list[Symbol] = []

    def _reset_state(self) -> None:
        """Reset state of the symbol to a simple "free" symbol.

        This is generally not recommended to be used outside of the Symbol class as it can destroy
        traced dependencies.
        """
        self._is_searchable = False
        self._is_constant = False
        self._parent = None
        self._dependencies = []

    def link_to(self, sp_parent: "Symbol") -> None:
        """Register a parent symbol, i.e., make this symbol dependent on the parent."""
        # sanity checks
        assert self.is_free or self._is_searchable, "Symbol is not free/searchable!"
        assert self not in sp_parent._dependencies, "Symbol already registered!"
        assert sp_parent not in self._dependencies, "Circular dependency!"

        # set up dependency
        self._is_searchable = False
        self._parent = sp_parent
        sp_parent._dependencies.append(self)

    def disable(self, _memo: set["Symbol"] | None = None) -> None:
        """Disable symbol and mark it as constant together with its whole dependency tree via DFS.

        After this call, ``is_constant == True``.
        """
        # check starting condition and recursion case
        if _memo is None:
            _memo = set()
        elif self in _memo:
            return

        # add self to memo
        _memo.add(self)

        # disable parent first
        if self.parent:
            self.parent.disable(_memo)

        # disable dependencies
        for dep in self._dependencies:
            dep.disable(_memo)

        # disable self at the end
        self._parent = None
        self._dependencies.clear()
        self._is_searchable = False
        self._is_sortable = False
        self._is_constant = True

    def _check_sortable(self, _memo: set["Symbol"] | None = None) -> bool:
        """Check for sortability in dependency tree via DFS."""
        # check starting condition and recursion case
        if _memo is None:
            _memo = set()
        elif self in _memo:
            return self._is_sortable

        # add self to memo
        _memo.add(self)

        # collect symbols to check
        all_syms = [self, *self._dependencies]
        if self.parent:
            all_syms.append(self.parent)

        # check sortability of all symbols in dependency tree
        return all(sym._check_sortable(_memo) for sym in all_syms)

    @property
    def parent(self) -> "Symbol | None":
        """Return the parent symbol."""
        return self._parent

    @property
    def is_searchable(self) -> bool:
        """Return indicator whether symbol is searchable."""
        return self._is_searchable

    @property
    def is_sortable(self) -> bool:
        """Return indicator whether symbols in dependency tree are sortable."""
        return self._check_sortable()

    @is_sortable.setter
    def is_sortable(self, value: bool) -> None:
        """Set symbol's sortability."""
        self._is_sortable = value

    @property
    def is_dynamic(self) -> bool:
        """Return indicator whether symbol is dynamic."""
        return self._parent is not None

    @property
    def is_constant(self) -> bool:
        """Return indicator whether symbol is constant."""
        return self._is_constant

    @property
    def is_free(self) -> bool:
        """Return indicator whether symbol is free."""
        return not (self.is_dynamic or self.is_constant or self.is_searchable)

    @property
    def cl_type(self) -> CLType:
        """Return the cross-layer type of the symbol."""
        assert self.is_cross_layer == bool(self._elastic_dims), (
            "Cross-layer symbols need elastic dims and elastic dim only defined for cross-layer!"
        )
        return self._cl_type

    @property
    def is_cross_layer(self) -> bool:
        """Return indicator whether symbol is cross-layer."""
        return self._cl_type in [Symbol.CLType.INCOMING, Symbol.CLType.OUTGOING]

    @property
    def is_incoming(self) -> bool:
        """Return indicator whether symbol is cross-layer incoming."""
        return self._cl_type == Symbol.CLType.INCOMING

    @property
    def is_outgoing(self) -> bool:
        """Return indicator whether symbol is cross-layer outgoing."""
        return self._cl_type == Symbol.CLType.OUTGOING

    @property
    def is_dangling(self) -> bool:
        """Return indicator whether symbol is dangling (cross-layer and free)."""
        return self.is_cross_layer and self.is_free

    @property
    def elastic_dims(self) -> set[int]:
        """Returns the set of tensor dimensions that refer to this symbol.

        Note that this refers to the dimension of the tensor incoming or outgoing from the layer,
        **not** the parameters of the module. E.g. for a Conv2d layer, this refers to the "C" in
        "NCHW" of the incoming/outgoing tensor.

        This must be defined for incoming/outgoing symbols, and must be empty for all others.

        Also note that this is a set of dimension, although only **one** actual dimension per
        Symbol can be elastic. The added flexibility of using a set instead of a single tensor is
        to enable describing the same dimension in different indexing notations (e.g. {1,-3} for
        Conv2d where both 1 and -3 refer to the "C" dimension in "NCHW").
        """
        # sanity check whether we can access cl_type and then return
        _ = self.cl_type
        return self._elastic_dims

    def __repr__(self) -> str:
        """Return string representation with relevant properties of the class."""
        attrs = [
            "is_searchable",
            "is_free",
            "is_dynamic",
            "is_constant",
            "is_incoming",
            "is_outgoing",
            *self._extra_repr_attrs,
        ]
        attrs = [x for x in attrs if getattr(self, x) or not isinstance(getattr(self, x), bool)]
        return f"{type(self).__name__}({', '.join(f'{x}={getattr(self, x)}' for x in attrs)})"


class SymInfo:
    """A simple class to hold relevant information about the symbolic nature of a given module."""

    SymDict = dict[str, Symbol]

    def __init__(self, is_shape_preserving: bool = False, **kwargs: Symbol):
        """Initialize the instance with the given symbolic information."""
        super().__init__()
        self._is_shape_preserving = is_shape_preserving
        self.symbols: SymInfo.SymDict = kwargs

    @property
    def is_shape_preserving(self) -> bool:
        """Return indicator whether module is shape-preserving."""
        return self._is_shape_preserving

    def __repr__(self) -> str:
        attr = [f"{k}={v}" for k, v in self.symbols.items()]
        attr.append(f"is_shape_preserving={self.is_shape_preserving}")
        sep_str = ",\n  "
        return f"{type(self).__name__}({sep_str[1:]}{sep_str.join(attr)}\n)"


class SymMap:
    """A class to hold the symbolic representations of a model."""

    SymRegisterFunc = Callable[[nn.Module], SymInfo]
    _registry: dict[type[nn.Module], SymRegisterFunc] = {}
    _dependent_registry: dict[type[nn.Module], type[nn.Module]] = {}

    # these are the dictionaries that hold the symbolic representations
    # defining their types here to make it easier to keep track of them
    _map: dict[nn.Module, SymInfo]
    _mod_to_name: dict[nn.Module, str]

    @classmethod
    def register(
        cls, nn_cls: type[nn.Module] | list[type[nn.Module]], is_explicit_leaf: bool = True
    ) -> Callable[[SymRegisterFunc], SymRegisterFunc]:
        """Use this to register a function that defines the symbols for a given nn module.

        Args:
            nn_cls: The nn module class for which the function is registered.
            is_explicit_leaf: Whether the module is an explicit leaf, i.e., whether it should be
                treated as leaf during tracing.


        Returns:
            A decorator that registers the given function for the given nn module class.

        An example for registering the symbolic information of a module is shown below:

        .. code-block:: python

            @SymMap.register(nn.Linear)
            def get_linear_sym_info(mod: nn.Linear) -> SymInfo:
                in_features = Symbol(cl_type=Symbol.CLType.INCOMING, elastic_dims={-1})
                out_features = Symbol(
                    is_searchable=True, cl_type=Symbol.CLType.OUTGOING, elastic_dims={-1}
                )
                return SymInfo(in_features=in_features, out_features=out_features)
        """

        def decorator(func: SymMap.SymRegisterFunc) -> SymMap.SymRegisterFunc:
            cls_list = nn_cls if isinstance(nn_cls, Iterable) else [nn_cls]
            for nn_cls_ in cls_list:
                cls._registry[nn_cls_] = func
                if is_explicit_leaf:
                    RobustTracer.register_leaf(nn_cls_)
            return func

        return decorator

    @classmethod
    def unregister(cls, nn_cls: type[nn.Module]) -> None:
        """Unregister module that previously has been registered.

        It throws a KeyError if the module is not registered.
        """

        def _remove_as_leaf(nn_cls_):
            if RobustTracer.is_registered_leaf(nn_cls_):
                RobustTracer.unregister_leaf(nn_cls_)

        # unregister
        cls._registry.pop(nn_cls)
        _remove_as_leaf(nn_cls)

        # unregister dependencies
        for nn_cls_dependent, nn_cls_ in list(cls._dependent_registry.items()):
            if nn_cls_ == nn_cls:
                cls._dependent_registry.pop(nn_cls_dependent)
                _remove_as_leaf(nn_cls_dependent)

    @classmethod
    def _get_from_registry(cls, nn_cls: type[nn.Module]) -> SymRegisterFunc | None:
        """We check the registry whether the given module has been registered and return it.

        If there is no match, we check for approximate matches. Specifically, we check whether a
        module inherits from a registered module _without_ overwriting the forward method. If this
        is the case we return the approximate match.
        """
        # check exact match first
        get_sym_info = cls._registry.get(nn_cls)
        if get_sym_info:
            return get_sym_info

        # check for previously registered approximate matches
        nn_cls_match = cls._dependent_registry.get(nn_cls)
        if nn_cls_match:
            return cls._registry[nn_cls_match]

        # let's try finding an approximate match next (note that we break on the first match)
        for nn_cls_, get_sym_info in cls._registry.items():
            if issubclass(nn_cls, nn_cls_) and nn_cls.forward == nn_cls_.forward:
                nn_cls_match = nn_cls_
                break

        # if it's an approximate match, we register it as dependent match to speed up future lookups
        if nn_cls_match:
            if RobustTracer.is_registered_leaf(nn_cls_match):
                RobustTracer.register_leaf(nn_cls)
            cls._dependent_registry[nn_cls] = nn_cls_match
            return get_sym_info

        return None

    def __init__(self, model) -> None:
        """Initialize with the desired module."""
        super().__init__()
        self.model = model

        # build an internal map next
        self._map = {}
        self._mod_to_name = {}

        # Build internal map from registered modules with sym info
        # Note that we do not want to iterate over module subtrees from modules that are registered
        # as leaf modules. This is because we treat them as leaf modules during tracing and hence
        # we would not record symbolic information for them. If there are internal dependencies,
        # they must be handled on the DynamicModule level.
        queue: deque[tuple[str, nn.Module]] = deque([("", self.model)])
        memo: set[nn.Module] = {self.model}  # to keep track of duplicates
        while queue:
            # grab next module in queue
            name, mod = queue.popleft()

            # register sym info if available
            get_sym_info = self._get_from_registry(type(mod))
            if get_sym_info is not None:
                self._map[mod] = get_sym_info(mod)
                self._mod_to_name[mod] = name

            # add children to queue if not registered as leaf
            if RobustTracer.is_registered_leaf(mod):
                continue
            for c_name, c_mod in mod.named_children():
                if c_mod not in memo:
                    queue.append((f"{name}.{c_name}", c_mod) if name else (c_name, c_mod))
                    memo.add(c_mod)  # store as already queued

    def __len__(self) -> int:
        """Return the number of symbolic modules."""
        return len(self._map)

    def __bool__(self) -> bool:
        """Return whether the symbolic map is empty."""
        return bool(self._map)

    def __getitem__(self, key: nn.Module) -> SymInfo.SymDict:
        """Return the symbolic module for the given module."""
        return self._map[key].symbols

    def __contains__(self, key: nn.Module) -> bool:
        """Return whether the given module has a symbolic representation."""
        return key in self._map

    def __iter__(self) -> Generator[nn.Module, None, None]:
        """Return an iterator over the keys of the dictionary."""
        yield from self._map

    def items(self) -> Generator[tuple[nn.Module, SymInfo.SymDict], None, None]:
        """Return an iterator over the dictionary."""
        for mod, sym_info in self._map.items():
            yield mod, sym_info.symbols

    def pop(self, key: nn.Module) -> SymInfo.SymDict:
        """Remove the given module from the dictionary and return its symbolic representation."""
        self._mod_to_name.pop(key)
        return self._map.pop(key).symbols

    def named_modules(self) -> Generator[tuple[str, nn.Module], None, None]:
        """Yield the name (from self._mod_to_name) and the associated module."""
        for mod, name in self._mod_to_name.items():
            yield name, mod

    def named_sym_dicts(self) -> Generator[tuple[str, SymInfo.SymDict], None, None]:
        """Yield the name (from self._mod_to_name) and the associated symbolic module."""
        for mod, sym_info in self._map.items():
            yield self._mod_to_name[mod], sym_info.symbols

    def named_symbols(
        self,
        key: nn.Module | None = None,
        free: bool | None = None,
        dynamic: bool | None = None,
        searchable: bool | None = None,
        constant: bool | None = None,
    ) -> Generator[tuple[str, Symbol], None, None]:
        """Yield the name and symbol of symbols in either all symbolic modules or a specific one.

        Args:
            key: The module to get symbols from. If not provided, recursive through all modules.
            free: Whether to include free symbols.
            dynamic: Whether to include dynamic symbols.
            searchable: Whether to include searchable symbols.
            constant: Whether to include constant symbols.

        Yields:
            (name, Symbol): Tuple containing the name and symbol.

        Default behavior is to iterate over free, dynamic, searchable, or constant symbols. Set
        args accordingly to only iterate over some. When either ``free``, ``dynamic``,
        ``searchable``, or ``constant`` is set to ``True``, only symbols of that type are
        iterated over. If either of ``free``, ``dynamic``, ``searchable``, or ``constant`` is set to
        ``False``, symbols of that type are skipped over.
        """
        _map = self._map if key is None else {key: self._map[key]}

        args = [free, dynamic, searchable, constant]
        if all(arg is None for arg in args):
            free = dynamic = searchable = constant = True
        if all(arg is not True for arg in args):
            free = True if free is None else free
            dynamic = True if dynamic is None else dynamic
            searchable = True if searchable is None else searchable
            constant = True if constant is None else constant

        for mod, sym_info in _map.items():
            mod_name = self._mod_to_name[mod]
            prefix = f"{mod_name}." if mod_name and not key else ""
            for sym_name, symbol in sym_info.symbols.items():
                if (
                    (symbol.is_free and free)
                    or (symbol.is_dynamic and dynamic)
                    or (symbol.is_searchable and searchable)
                    or (symbol.is_constant and constant)
                ):
                    yield prefix + sym_name, symbol

    def get_symbol(self, mod: nn.Module, name: str) -> Symbol:
        """Get symbol from the given module with the given name."""
        return self._map[mod].symbols[name]

    def set_symbol(self, mod: nn.Module, name: str, symbol: Symbol) -> None:
        """Set symbol from the given module with the given name."""
        self._map[mod].symbols[name] = symbol

    def prune(self) -> None:
        """Prune the map by removing modules with constant-only symbols."""
        # Note that this assumes that constant symbols don't have any children
        # This is enforced though during disabling a symbol
        prunable = []
        for mod, sym_info in self._map.items():
            if all(sym.is_constant for sym in sym_info.symbols.values()):
                prunable.append(mod)
        for mod in prunable:
            del self._map[mod]
            del self._mod_to_name[mod]

    def is_shape_preserving(self, key: nn.Module) -> bool:
        """Return whether the symbolic module is shape preserving."""
        return self._map[key].is_shape_preserving

    def add_sym_info(self, key: nn.Module, sym_info: SymInfo) -> None:
        """Manually add a model's module's sym_info."""
        for name, mod in self.model.named_modules():
            if mod is not key:
                continue
            self._mod_to_name[mod] = name
            self._map[mod] = sym_info
            return
        raise ValueError(f"Module {key} not found in model.")

    def __repr__(self):
        lines = [f"{type(self).__name__}(model={type(self.model).__name__}"]
        for name, mod in self.named_modules():
            name_str = f"  ({name}): "
            empty_str = " " * len(name_str)
            lines.append(name_str + str(mod))
            lines.append("\n".join(empty_str + x for x in str(self._map[mod]).split("\n")))
        lines.append(")")
        return "\n".join(lines)
