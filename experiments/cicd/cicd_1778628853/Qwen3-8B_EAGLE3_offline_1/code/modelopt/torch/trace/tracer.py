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

"""Tracing module for building a set of partial Graphs describing the model."""

import inspect
from collections import deque
from collections.abc import Callable, Generator
from typing import Any

import torch
import torch.nn as nn
from torch.fx import Graph, Tracer
from torch.fx.proxy import TraceError

from modelopt.torch.utils.graph import NodeTarget

__all__ = ["GraphCollection", "RobustTracer", "recursive_trace"]


class RobustTracer:
    """A robust graph tracer for tracing dependencies between layers.

    Note that this tracer wraps unsupported ops into its parent module as a whole.
    """

    _leaves: set[type[nn.Module]] = set()

    @classmethod
    def is_registered_leaf(cls, m: nn.Module | type[nn.Module]) -> bool:
        """Check if module is registered as extra leaf module."""
        if isinstance(m, nn.Module):
            m = type(m)
        return m in cls._leaves

    @classmethod
    def register_leaf(cls, m: type[nn.Module]) -> None:
        """Register a module as extra leaf module."""
        cls._leaves.add(m)

    @classmethod
    def unregister_leaf(cls, m: type[nn.Module]) -> None:
        """Unregister a module as extra leaf module."""
        cls._leaves.remove(m)

    class _FxTracerPlus(Tracer):
        """An augmented version of the standard fx tracer in PyTorch.

        Note that although the Tracer is a class it is really not meant to be persistent or re-used
        multiple times to repeat tracing the same or different models! It is meant to be used once!
        """

        def __init__(self, outer_tracer: "RobustTracer", *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._outer_tracer = outer_tracer

        @property
        def _expected_tracing_errors(self) -> tuple[type[Exception], ...]:
            return (TraceError, TypeError)

        def trace(
            self,
            root: torch.nn.Module | Callable[..., Any],
            concrete_args: dict[str, Any] | None = None,
        ) -> Graph:
            """Trace like regular tracer, however, we need special treatment for root == leaf.

            Unlike the regular tracer we don't want to trace into the leaf module even if the leaf
            module is the root. In the regular tracer, a leaf module is traced into when it is the
            root.

            .. note::

                ``concrete_args`` is experimental and not backwards compatible!!!
            """
            if not (isinstance(root, nn.Module) and self.is_leaf_module(root, ".")):
                # If root is not a leaf module, we can just trace it normally.
                return super().trace(root, concrete_args)

            # There are leaf modules that aren't traceable at all because their forward method
            # is invalid (e.g. nn.ModuleList). Hence we check if we run into an expected tracing
            # error or a more general exception. Regular tracing errors are then handled.
            try:
                temp_tracer = Tracer()
                temp_tracer.trace(root, concrete_args)
            except self._expected_tracing_errors:
                pass

            # Here we handle the graph construction ourselves following a very simple setup:
            # 1. Initialize the tracer instance for graph tracing.
            # 2. Record the input nodes via self.create_args_for_root.
            # 3. Record the call_module node for the root node itself.
            # 4. Record the output node.
            # 5. Return the graph.

            # 1.
            self.root = root
            self.graph = Graph(tracer_cls=getattr(self, "__class__", None))

            # 2.
            fn = getattr(type(root), self.traced_func_name)
            fn, args = self.create_args_for_root(
                fn, isinstance(root, torch.nn.Module), concrete_args
            )

            # 3.
            def fake_forward(*args, **kwargs):
                raise RuntimeError("This should not be called since root is a leaf!")

            proxy = self.call_module(root, fake_forward, tuple(args[1:]), {})

            # 4.
            self.create_node(
                "output",
                "output",
                (self.create_arg(proxy),),
                {},
                type_expr=fn.__annotations__.get("return", None),
            )

            # 5.
            return self.graph

        def call_module(self, m, forward, args, kwargs) -> Any:
            """Method specifying behavior when encountering a call to an nn.Module instance.

            On top of the default behavior, this method wraps the whole module when encountering
            untraceable operation.
            """
            module_qualified_name = self.path_of_module(m)
            e = None
            try:
                proxy = super().call_module(m, forward, args, kwargs)
            except self._expected_tracing_errors as _e:
                e = _e
                proxy = self.create_proxy("call_module", module_qualified_name, args, kwargs)
            self._outer_tracer.record_call_module(m, e)
            return proxy

        def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
            # any failed or registered module is also a leaf module
            return (
                super().is_leaf_module(m, module_qualified_name)
                or self._outer_tracer.is_registered_leaf(m)
                or self._outer_tracer.is_failed(m)
            )

    def __init__(self):
        """Initialize the tracer."""
        super().__init__()
        # some properties
        self.root = None
        self.graph = None

        # tracing related
        self._leaf_added = True
        self._tracer_failed: dict[int, str] = {}
        self._tracer_visited: set[int] = set()

    def record_call_module(self, m: nn.Module, e: Exception | None = None) -> None:
        """Record the call_module operation."""
        self._tracer_visited.add(id(m))  # mark as visited
        if e is not None:
            self._tracer_failed[id(m)] = f"{type(e).__name__}: {e!s}"  # mark as failed as well
            self._leaf_added = True  # add a flag to indicate a leaf node is added

    def is_failed(self, m: NodeTarget) -> bool:
        """Check if the module is failed."""
        return id(m) in self._tracer_failed

    def failure_msg(self, m: NodeTarget) -> str | None:
        """Get the failure message of the module."""
        return self._tracer_failed.get(id(m))

    def is_unvisited(self, m: NodeTarget) -> bool:
        """Check if the node target is unvisited."""
        return id(m) not in self._tracer_visited

    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        """Check if the given module is a leaf module."""
        return self._FxTracerPlus(self).is_leaf_module(m, module_qualified_name)

    def trace(self, root: nn.Module, concrete_args: dict[str, Any] | None = None) -> Graph:
        """Trace root using the fx tracer.

        Unlike the standard tracer in fx, this tracer adds the following functionalities:

        1. Only torch.Tensors in concrete_args are converted to Proxy objects.
        2. The tracer doesn't fail when encountering unsupported operations. Instead it will wrap
           the module containing the unsupported operation into a leaf node and repeat the tracing.
        3. The tracer doesn't trace into leaf modules even if the leaf module is the root.
        """
        self.root = root
        self.graph = None

        # get graph
        e = None
        try:
            full_kwargs = self._get_full_kwargs(root, concrete_args=concrete_args)
            self._leaf_added = True
            while self._leaf_added:
                self._leaf_added = False
                self.graph = self._FxTracerPlus(self).trace(root, concrete_args=full_kwargs)
        except Exception as _e:
            e = _e
            self.graph = Graph()
        self.record_call_module(root, e)

        return self.graph

    def _get_full_kwargs(
        self, root: NodeTarget, concrete_args: dict[str, Any] | None = None
    ) -> Graph:
        # extract function to be traced
        fn = type(root).forward if isinstance(root, nn.Module) else root

        # supplement concrete args with default values if they are not empty or tensor
        full_kwargs = {}
        for k, p in inspect.signature(fn).parameters.items():
            if not (p.default is inspect.Parameter.empty or isinstance(p.default, torch.Tensor)):
                full_kwargs[k] = p.default
        full_kwargs.update(concrete_args or {})

        return full_kwargs


class GraphCollection:
    """A collection of fx graphs and tracing results."""

    def __init__(self) -> None:
        """Initialize the graph collection."""
        self._graphs: dict[NodeTarget, Graph] = {}
        self._tracer = RobustTracer()

    def __iter__(self) -> Generator[NodeTarget, None, None]:
        """Return an iterator over the modules with graph representations."""
        yield from self._graphs

    def __getitem__(self, key: NodeTarget) -> Graph:
        """Return the fx graph associated with the given key."""
        return self._graphs[key]

    def is_failed(self, m: NodeTarget) -> bool:
        """Check if node target failed during tracing."""
        return self._tracer.is_failed(m)

    def failure_msg(self, m: NodeTarget) -> str | None:
        """Get the failure message of the module if it failed."""
        return self._tracer.failure_msg(m)

    def is_unvisited(self, m: NodeTarget) -> bool:
        """Check if the node target is unvisited."""
        return self._tracer.is_unvisited(m)

    def recursive_trace(
        self, model: nn.Module, concrete_args: dict[str, Any] | None = None
    ) -> None:
        """Recursively trace the model and all failing sub-modules and store graph collection.

        Args:
            model: The model to be traced.
            concrete_args: [EXPERIMENTAL; not backwards compatible!] The concrete arguments to be
                used for tracing of the root module (i.e. the model's forward method). Note that
                if/when the tracing recurses, no concrete args are passed on! Defaults to None.

        Note that the graph collection contains the graph of the root module as well as all failing
        submodules. For failing modules, the graph will contain the failing module as a leaf and
        then separate graphs for each submodule within the failing module to maximize the coverage
        of the tracing results.
        """
        # reinitialize the collector
        self._graphs = {}
        self._tracer = RobustTracer()

        # trace model as well as failing sub-modules
        mods_to_trace = deque([model])
        mods_traced_all = {model}
        while mods_to_trace:
            # get module
            mod = mods_to_trace.popleft()

            # run tracer
            graph = self._tracer.trace(mod, concrete_args=concrete_args if mod is model else None)

            # record results
            self._graphs[mod] = graph

            # add children of failing modules to the queue
            # use of set guarantees that each module is only added once
            for m in mod.modules():
                if self.is_failed(m):
                    for child in m.children():
                        if child not in mods_traced_all:
                            mods_to_trace.append(child)
                            mods_traced_all.add(child)


def recursive_trace(
    model: nn.Module, concrete_args: dict[str, Any] | None = None
) -> GraphCollection:
    """Recursively trace the model and all failing sub-modules and return graph collection.

    Args:
        model: The model to be traced.
        concrete_args: [EXPERIMENTAL; not backwards compatible!] The concrete arguments to be used
            for tracing of the root module (i.e. the model's forward method). Note that if/when the
            tracing recurses, no concrete args are passed on! Defaults to None.

    Returns:
        The collection of fx graphs resulting from the tracing.

    Note that the graph collection contains the graph of the root module as well as all failing
    submodules. For failing modules, the graph will contain the failing module as a leaf and then
    separate graphs for each submodule within the failing module to maximize the coverage of the
    tracing results.
    """
    gc = GraphCollection()
    gc.recursive_trace(model, concrete_args)
    return gc
