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

"""Analyzer to look for layer-wise dependencies to identify connected symbols."""

import inspect
import itertools
import operator
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from collections.abc import Callable, Generator, Iterable, Iterator
from contextlib import contextmanager
from typing import Any, TypeVar

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console
from rich.table import Table
from torch.fx import Node

from modelopt.torch.utils.graph import NodeTarget, _get_node_target

from .modules.nn import SymDepth
from .symbols import Symbol, SymMap
from .tracer import GraphCollection, RobustTracer, recursive_trace

__all__ = ["analyze_symbols"]


class Map:
    """A mapping of the dependencies that continuously updates/resolve recursive mappings."""

    Dependency = dict[str, Node | None | NodeTarget | int | bool]

    def __init__(self):
        # this is the collection of root dependencies
        self._root_dependencies: dict[Node, Map.Dependency] = {}

        # this is the mapping of nodes to their root nodes
        self._node_mappings: dict[Node, Node] = {}

    def is_root(self, node: Node) -> bool:
        """Return True if node is root."""
        return node in self and node in self._root_dependencies

    def is_dependent(self, node: Node) -> bool:
        """Return True if node is dependent."""
        return node in self and not self.is_root(node)

    def root(self, node: Node) -> Node:
        """Return root node of node."""
        return self._get_dependency(node)["root"]

    def target(self, node: Node) -> NodeTarget:
        """Return root target of node."""
        return self._get_dependency(node)["target"]

    def id(self, node: Node) -> int:
        """Return root id of node."""
        id = self._get_dependency(node)["id"]
        assert isinstance(id, int)
        return id

    def is_free(self, node: Node) -> bool:
        """Check if root node is free."""
        is_free = self._get_dependency(node)["is_free"]
        assert isinstance(is_free, bool)
        return is_free

    def set_free(self, node: Node, is_free: bool):
        """Set is_free tag of associated root node."""
        assert isinstance(is_free, bool)
        self._get_dependency(node)["is_free"] = is_free

    def priority(self, node: Node) -> int:
        """Return priority of the root node."""
        priority = self._get_dependency(node)["priority"]
        assert isinstance(priority, int)
        return priority

    def __contains__(self, node: Node) -> bool:
        """Return True if node is in mapping."""
        return node in self._node_mappings

    def __iter__(self) -> Iterator[Node]:
        """Return iterator over nodes in mapping."""
        return iter(self._node_mappings)

    def _get_dependency(self, node: Node) -> "Map.Dependency":
        """Return root dependency of node."""
        if node not in self:
            raise ValueError(f"Node {node} is not in mapping.")
        return self._root_dependencies[self._node_mappings[node]]

    def link_nodes(self, node: Node, other_node: Node) -> None:
        """Link dependency of node to the dependency of other_node."""
        # get other root node
        other_root = self.root(other_node)

        # in this case we have a new dependent node.
        if node not in self:
            self._node_mappings[node] = other_root
            return

        # retrieve root dependency of node (this one needs re-assignment)
        root = self.root(node)

        # check if we simply try to re-link
        if root == other_root:
            return

        # in this case a root node is now becoming dependent on other_root
        del self._root_dependencies[root]
        self._node_mappings[root] = other_root

        # find all nodes that previously depended on root and update them
        for n, r in self._node_mappings.items():
            if r == root:
                self._node_mappings[n] = other_root

    def create_root(
        self, node: Node, target: NodeTarget, id: int, is_free: bool, priority: int = 0
    ) -> None:
        """Create new root dependency."""
        if node in self:
            raise ValueError(f"Cannot create root for node {node} that is already in mapping.")

        # create new root dependency
        dep: Map.Dependency = {
            "root": node,
            "target": target,
            "id": id,
            "is_free": is_free,
            "priority": priority,  # higher number means higher priority
        }
        self._root_dependencies[node] = dep
        self._node_mappings[node] = node

    def clear(self):
        """Clear mapping."""
        self._root_dependencies.clear()
        self._node_mappings.clear()


class NodeProcessor(ABC):
    """Abstract class for handling tracing for special nodes like concat.

    To register a processor, use `@GraphDependencyProcessor.register_node_processor` decorator.
    """

    SymIterator = Iterator[tuple[str, Symbol]]

    def __init__(
        self, model: nn.Module, gc: GraphCollection, sym_map: SymMap, dependency_map: Map
    ) -> None:
        super().__init__()
        self._model = model
        self._gc = gc
        self._sym_map = sym_map
        self._dependency_map = dependency_map

    # TODO: consider including input_nodes here since we already sometimes use it...
    @abstractmethod
    def is_special_node(self, node: Node, target: NodeTarget) -> bool:
        """Check if node is this type of special node."""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset processor."""

    @abstractmethod
    def process(self, node: Node, id: int, input_nodes: list[Node]) -> None:
        """Handle processing for the fx graph tracing."""
        raise NotImplementedError

    def post_process(self) -> None:
        """Handle post-processing at the end of the entire fx graph tracing."""

    def _filtered_named_symbols(
        self, target: NodeTarget, filter: Callable[[Symbol], bool]
    ) -> Iterator[tuple[str, Symbol]]:
        """Yield symbols of current target filtered by filter."""
        # get (fake) iterator over hyperparameters of current target
        sym_gen = self._sym_map.named_symbols(key=target) if target in self._sym_map else {}.items()
        for sym_name, sym in sym_gen:
            if filter(sym):
                yield sym_name, sym

    def named_in_symbols(self, target: NodeTarget) -> SymIterator:
        """Yield incoming symbols of current target."""
        yield from self._filtered_named_symbols(target, lambda x: x.is_incoming)

    def named_out_symbols(self, target: NodeTarget) -> SymIterator:
        """Yield outgoing symbols of current target."""
        yield from self._filtered_named_symbols(target, lambda x: x.is_outgoing)

    def named_searchable_out_symbols(self, target: NodeTarget) -> SymIterator:
        """Yield searchable outgoing symbols of current target.

        NOTE: For concat, incoming symbols are also searchable but we do not want to use that here.
        """
        yield from self._filtered_named_symbols(target, lambda x: x.is_searchable and x.is_outgoing)

    def named_dangling_symbols(self, target: NodeTarget) -> SymIterator:
        """Yield dangling (cross layer but free) symbols of current target."""
        yield from self._filtered_named_symbols(target, lambda x: x.is_dangling)

    def named_cross_layer_symbols(self, target: NodeTarget) -> SymIterator:
        """Yield all cross-layer symbols of current target."""
        yield from self._filtered_named_symbols(target, lambda x: x.is_cross_layer)

    @classmethod
    def _is_from_node_list(
        cls, target: NodeTarget, node_list: list, gc: GraphCollection, check_failed=True
    ) -> bool:
        """Check if target is from node_list."""
        is_on_list = type(target) in node_list or target in node_list
        return is_on_list and not (check_failed and gc.is_failed(target))

    def _get_node_target(self, node: Node) -> NodeTarget | None:
        return _get_node_target(node, root=self._model)

    def _get_root_target(self, node: Node) -> NodeTarget:
        return self._dependency_map.target(node)

    def _identify_in_out_nodes(self, node: Node) -> tuple[list[Node], Node | None]:
        """Return list of input nodes w/o mutable out node and out node.

        For methods within torch and torch.Tensor, the user can specify "out" to store the result
        in the given tensor reference --> thus we need to specially treat this node.
        """
        # check if node target is from torch, torch.Tensor
        target = self._get_node_target(node)
        all_torch_ops = itertools.chain(inspect.getmembers(torch), inspect.getmembers(torch.Tensor))
        is_torch_ops = any(target == meth for _, meth in all_torch_ops)

        # check if out is given as kwarg and if it's a node for targets from torch or torch.Tensor
        out_node = node.kwargs.get("out") if is_torch_ops else None
        out_node = out_node if isinstance(out_node, Node) else None

        # identify primary input node, denoted by "input" in torch_ops if it exists
        # --> will be either first arg or in kwargs
        primary_input = node.all_input_nodes[0] if len(node.all_input_nodes) > 0 else None
        primary_input = node.kwargs.get("input", primary_input) if is_torch_ops else primary_input

        # check if input_node is an actual graph node (instead of just any kwarg)
        # Note that constant tensors are traced as nodes, native python input is fine (by virtue of
        # broadcasting), and other inputs would fail anyway.
        primary_input = primary_input if isinstance(primary_input, Node) else None

        # filter out from input_nodes
        input_nodes = [n for n in node.all_input_nodes if n != out_node]

        # sort to have primary input first
        input_nodes.sort(key=lambda x: x != primary_input)

        # return normalized input nodes and out node separately
        return input_nodes, out_node

    def _disable_node(self, node: Node):
        """Disable node and its corresponding searchable outgoing hparams."""
        self._dependency_map.set_free(node, False)
        node_target = self._get_root_target(node)
        for _, sym in self.named_searchable_out_symbols(node_target):
            sym.disable()

    @staticmethod
    def build_sym_matching(syms_other: list[Symbol], syms2: list[Symbol]) -> dict[Symbol, Symbol]:
        """Build 1-on-1 mapping from syms to syms2 based on the syms's and syms2's elastic_dims.

        This method returns a mapping where each symbol in syms is matched to a symbol in
        syms2. An empty mapping is returned if there is no perfect matching available. If the
        perfect matching is not unique, the first match is returned.

        Note that we utilize a brute-force approach for building a perfect matching in a bipartite
        graph with at most 2x2 symbols! Consider
        https://en.wikipedia.org/wiki/Maximum_cardinality_matching for more sophisticated approaches
        in the future if needed.
        """
        if len(syms_other) != len(syms2) or not syms_other:
            return {}

        # build an adjacency matrix where each entry is a set of matching elastic_dims
        _get_match = np.vectorize(lambda s1, s2: s1.elastic_dims & s2.elastic_dims)
        matches = _get_match(*np.meshgrid(syms_other, syms2))

        # make sure there exists at least a match for each hparam
        if tuple(matches.astype(bool).sum(a).astype(bool).sum() for a in [1, 0]) != matches.shape:
            return {}

        # build perfect matching if it exists
        assert len(syms_other) <= 2, "Only 1-on-1 or 2-on-2 matching supported!"

        # 1-on-1 matching
        if len(syms_other) == 1:
            return {syms_other[0]: syms2[0]}

        # 2-on-2 matching
        mapping = [(0, 0), (1, 1)] if matches[0, 0] else [(0, 1), (1, 0)]
        return {syms_other[i]: syms2[j] for i, j in mapping}

    def _get_root_nodes(self, nodes: list[Node]) -> list[Node]:
        """Return root nodes of nodes according to dependency map."""
        return [self._dependency_map.root(n) for n in nodes]

    def _synchronize_nodes(self, nodes: list[Node], disable: bool = False) -> Node | None:
        """Synchronize list of nodes if possible, otherwise disable symbols within nodes.

        Synchronizing hereby means linking the symbols of the provided list of nodes to each
        other by identifying the first node in the DAG order and linking the hparams of all of the
        other nodes to it.

        Args:
            nodes: List of nodes to synchronize
            disable: If True, disable hparams within nodes instead of synchronizing

        Returns:
            Free node (1st node in DAG order) if synchronization was successful, None if symbols
            were disabled instead.
        """
        # we need root nodes for each node
        nodes = self._get_root_nodes(nodes)

        if len(nodes) < 1:
            return None

        # find 1st node in DAG order which will constitute the free node and distinguish from others
        first_node = min(
            nodes, key=lambda n: (-self._dependency_map.priority(n), self._dependency_map.id(n))
        )
        other_input_nodes = [in_node for in_node in nodes if in_node != first_node]

        # retrieve a couple of attributes from the first input node
        target_first = self._get_root_target(first_node)
        syms_first = [hp for _, hp in self.named_searchable_out_symbols(target_first)]

        # reference for hp mappings
        mappings = {}

        # check if dependencies of input nodes are dynamic and hparams are compatible,
        # otherwise we cannot have variable inputs.
        for in_node in nodes:
            target_in = self._get_root_target(in_node)
            is_free_in = self._dependency_map.is_free(in_node)
            syms_in = [sp for _, sp in self.named_searchable_out_symbols(target_in)]
            mappings[in_node] = self.build_sym_matching(syms_first, syms_in)
            if not (is_free_in and mappings[in_node]):
                # if the above condition does not hold, we cannot synchronize the nodes
                disable = True
                break

        if disable:
            for node in nodes:
                self._disable_node(node)
            return None

        if len(other_input_nodes) == 0:
            return first_node

        # sanity check --> should be true at this point.
        assert target_first in self._sym_map, "target_first not in sym_map!"

        # sync symbols for other input nodes with first node
        for node in other_input_nodes:
            for sym_first, sym_other in mappings[node].items():
                sym_other.link_to(sym_first)

        # update dependency maps
        for node_dep in other_input_nodes:
            self._dependency_map.link_nodes(node_dep, first_node)

        # everything successful
        return first_node

    def _create_root(self, node: Node, id: int, is_free: bool, priority: int = 0) -> None:
        """Generate a new root dependency entry."""
        self._dependency_map.create_root(
            node=node,
            target=self._get_node_target(node),
            id=id,
            is_free=is_free,
            priority=priority,
        )

    def _process_passthrough(self, node: Node, id: int, input_nodes: list[Node]) -> None:
        """Enforce passthrough constraint on node i.e. the link node to primary input node."""
        self._dependency_map.link_nodes(node, input_nodes[0])

    def _process_boundary(self, node: Node, id: int, input_nodes: list[Node]) -> None:
        """Enforce boundary constraint on node.

        Note that if the node has internal symbols in sub-modules they will be traced in a separate
        attempt.
        """
        # disable all searchable outgoing symbols for all input nodes
        self._synchronize_nodes(input_nodes, disable=True)

        # generate new root dependency
        self._create_root(node, id, is_free=False)


class BoundaryNodeProcessor(NodeProcessor):
    """Processor for handling boundary nodes.

    NOTE: This processor should NOT be registered to GraphDependencyProcessor.
    It is used as the default processor for all nodes that are not handled by any other processors.
    """

    def is_special_node(self, node: Node, target: NodeTarget) -> bool:
        """Return True if node is a boundary node."""
        return False  # not used

    def process(self, node: Node, id: int, input_nodes: list[Node]) -> None:
        """Enforce boundary constraint on node."""
        self._process_boundary(node, id, input_nodes)


class GraphDependencyProcessor:
    """A processor to enforce hparams constraints/dependencies of a given graph/model."""

    T = TypeVar("T", bound=NodeProcessor)
    _node_processor_classes: list[type[NodeProcessor]] = []

    def __init__(self, model: nn.Module, gc: GraphCollection, sym_map: SymMap):
        self._model = model
        self._sym_map = sym_map  # "global" instance re-used for all recursive traces!
        self._gc = gc  # "global" instance re-used for all recursive traces!
        self._dependency_map = Map()
        np_args = (self._model, self._gc, self._sym_map, self._dependency_map)
        self._node_processors = [
            np_cls(*np_args) for np_cls in GraphDependencyProcessor._node_processor_classes
        ]
        self._default_node_processor = BoundaryNodeProcessor(*np_args)

    @classmethod
    def register_node_processor(cls, node_processor_cls: type[T]) -> type[T]:
        """A decorator to register a node processor.

        For example:

        .. code-block:: python

            @GraphDependencyProcessor.register_node_processor
            class MyNodeProcessor(NodeProcessor):
                pass
        """
        assert node_processor_cls is not BoundaryNodeProcessor
        cls._node_processor_classes.append(node_processor_cls)
        return node_processor_cls

    def reset(self):
        self._dependency_map.clear()

        for node_processor in self._node_processors:
            node_processor.reset()

    def process(self):
        """Process all dependencies for a given graph and model."""
        self.reset()

        for node_id, node in enumerate(self._gc[self._model].nodes):
            # identify in and out nodes
            input_nodes, out_node = self._default_node_processor._identify_in_out_nodes(node)

            # identify constraint func and process constraint
            process_constraint = self._get_constraint_func(node)
            process_constraint(node, node_id, input_nodes)

            # check and update out node
            if out_node:
                self._dependency_map.link_nodes(out_node, node)

        # post-process model
        for node_processor in self._node_processors:
            node_processor.post_process()
        self._default_node_processor.post_process()

    def _get_constraint_func(self, node: Node) -> Callable[[Node, int, list[Node]], None]:
        """Return the constraint function for the type of the given node target."""
        target = _get_node_target(node, root=self._model)
        for node_processor in self._node_processors:
            if node_processor.is_special_node(node, target):
                return node_processor.process

        # node is boundary, i.e., blocks any dynamic input
        return self._default_node_processor.process


@GraphDependencyProcessor.register_node_processor
class FeatureAdaptiveNodeProcessor(NodeProcessor):
    """Processor for handling feature adaptive nodes.

    Such nodes are a collection of possible univariate modules that are adaptive w.r.t. the feature
    dimension. In general, such modules are pass-through nodes and do not impose constraints.
    However, they do need to be treated separately since they need to adapt the feature dimension.
    input can be provided as first arg or kwarg --> remaining args, kwargs are irrelevant.
    These functions are part of registered modules and do not have in-place equivalents.

    For feature adaptive nodes, constraint will be passed on but hparams need to be updated.
    """

    def _link_feature_adaptive_node(self, node: Node, id: int, input_nodes: list[Node]) -> Node:
        """Link feature-adaptive node to its primary input node.

        We also check if the target of the node is re-used and, if so, we sync incoming hparams of
        the primary input node across usages.

        Note this method does not update the dependency map; only the hparam dependencies of target.
        Returns the node associated with the first-time usage of the target.
        """
        # get target of this node
        target = self._get_node_target(node)

        # get all nodes that have been processed and share the same target module
        target_sharing_nodes = []
        if isinstance(target, nn.Module):
            # retrieve all nodes that share the same target
            target_sharing_nodes = [
                node for node in self._dependency_map if self._get_node_target(node) == target
            ]
        target_sharing_nodes.append(node)

        # retrieve the primary input node for each target sharing node
        # this list constistutes all inputs to the target module found so far (Note that a module
        # can be used multiple times during a forward pass / graph traversal)
        primary_inputs = [self._identify_in_out_nodes(nde)[0][0] for nde in target_sharing_nodes]

        # synchronize all primary inputs to the "first" primary input (similar to multivariate!)
        self._synchronize_nodes(primary_inputs)

        # no further linkage of target required if it's not the first usage
        if len(target_sharing_nodes) > 1:
            return target_sharing_nodes[0]

        # link feature-adaptive target to root target if it's the first usage
        root_target = self._get_root_target(input_nodes[0])

        # build map
        syms_dangling = [sym for _, sym in self.named_dangling_symbols(target)]
        syms_input = [sym for _, sym in self.named_searchable_out_symbols(root_target)]
        sym_match = self.build_sym_matching(syms_input, syms_dangling)

        if sym_match:
            # Link an adaptive node according to the dependency of its primary input node.
            for sym_in, sym_dangling in sym_match.items():
                # we can link those two hparams together
                sym_dangling.link_to(sym_in)
        else:
            # disable all if no mapping is found, e.g., for a CV module followed by NLP module
            for sym_dangling in syms_dangling:
                sym_dangling.disable()
            self._synchronize_nodes(primary_inputs, disable=True)

        return target_sharing_nodes[0]

    def is_special_node(self, node: Node, target: NodeTarget) -> bool:
        """Return True if node is feature-adaptive."""
        return (
            any(self.named_in_symbols(target))
            and not any(self.named_searchable_out_symbols(target))
            and not self._gc.is_failed(target)
        )

    def _process_if_first_usage_node(self, node: Node, id: int, input_nodes: list[Node]) -> None:
        """Process node if it's the first usage of the target."""
        # enforce as passthrough constraint since it's the first usage
        self._process_passthrough(node, id, input_nodes)

    def process(self, node: Node, id: int, input_nodes: list[Node]) -> None:
        """Enforce feature-adaptive constraint on node."""
        first_usage_node = self._link_feature_adaptive_node(node, id, input_nodes)

        if node == first_usage_node:
            self._process_if_first_usage_node(node, id, input_nodes)
        else:
            # simply link the current node to the node that shares the same target
            self._dependency_map.link_nodes(node, first_usage_node)


@GraphDependencyProcessor.register_node_processor
class FreeNodeProcessor(FeatureAdaptiveNodeProcessor):
    """Processor for handling free nodes.

    Free node has no constraints and symbols are searchable/free.
    """

    def is_special_node(self, node: Node, target: NodeTarget) -> bool:
        """Return True if node is free."""
        return (
            any(self.named_in_symbols(target))
            and any(self.named_searchable_out_symbols(target))
            and not self._gc.is_failed(target)
        )

    def _process_if_first_usage_node(self, node: Node, id: int, input_nodes: list[Node]) -> None:
        """Process node if it's the first usage of the target."""
        for _, sym in self.named_out_symbols(self._get_node_target(node)):
            assert sym.is_searchable
        self._create_root(node, id, is_free=True)


@GraphDependencyProcessor.register_node_processor
class PassthroughUnivariateNodeProcessor(NodeProcessor):
    """Processor for handling pass-through univariate channel preserving nodes.

    Pass-through node constraint will be passed on. Note that pass-through nodes might also have
    only specific elastic dimensions similar to SymModules! Unlike SymModules, however, pass-through
    nodes

        * can support multiple elastic dimensions simultaneously since they are
          mere operators instead of stateful modules;
        * have elastic dimensions referring to different tensor dimensions, whereas Symbols can
          only support one elastic dimension referred to in different indexing schemes.
    """

    _passthrough_targets: dict[NodeTarget, set[int]] | None = None

    # fmt: off

    # * a collection of possible univariate operators that preserve various dimensions
    # * In general, such operators are pass-through nodes when the elastic dimensions are compatible
    #   and do not impose constraints. If incompatible, they will be boundary nodes
    # * The "out" keyword argument that will store the result in the provided tensor needs to be
    #   treated specially. Specifically, its dependency map must be updated as other nodes might be
    #   using it directly instead of the op-node itself. By the properties of the DAG no subsequent
    #   nodes will need the previous dependency map of the out node.
    # * These functions can have the following signatures:
    #   - (input, *, out: None, **kwargs) --> input is first arg, out is in kwargs
    #   - (input, *, **kwargs) --> input is first arg
    #   - (*, out: None, input: torch.Tensor, **kwargs) --> input is in kwargs, out might be in
    #     kwargs
    # * To extract "out" we can check for "out" in kwargs and then use "==" to remove it from
    #   all_input_nodes.
    # * These operators can be in the torch, torch.Tensor, and the native operator module and can
    #   also appear as in-place operator with trailing underscore.
    # * In-place operators do not need special treatment since they input already has the correct
    #   dependency.

    # * a collection of possible univariate operators that preserve all tensor dimensions.
    PASSSTHROUGH_OPS = dict.fromkeys([
        # univariate pointwise ops
        # gathered from https://pytorch.org/docs/stable/torch.html#pointwise-ops
        "abs", "absolute", "acos", "arccos", "acosh", "arccosh", "angle", "asin", "arcsin", "asinh",
        "arcsinh", "atan", "arctan", "atanh", "arctanh", "atan2", "arctan2", "bitwise_not",
        "bitwise_and", "bitwise_or", "bitwise_xor", "bitwise_left_shift", "bitwise_right_shift",
        "ceil", "conj_physical", "cos", "cosh", "deg2rad", "digamma", "erf", "erfc", "erfinv",
        "exp", "exp2", "expm1", "fix", "floor", "frac", "imag", "lgamma", "log", "log10", "log1p",
        "log2", "logical_not", "logit", "i0", "mvlgamma", "nan_to_num", "neg", "negative",
        "positive", "rad2deg", "real", "reciprocal", "relu", "round", "rsqrt", "sigmoid", "sign",
        "sgn", "signbit", "sin", "sinc", "sinh", "sqrt", "square", "tan", "tanh",
        # univariate comparison ops
        # gathered from https://pytorch.org/docs/stable/torch.html#comparison-ops
        "isfinite", "isinf", "isneginf", "isnan", "isreal",
        # Other ops
        # gathered from https://pytorch.org/docs/stable/torch.html#other-operations
        "clone", "cumprod", "cumsum", "detach", "logcumsumexp", "renorm", "tril", "triu",
        # built-in ops (with different names than torch ops)
        "pos", "invert",
        # misc ops
        "double", "float", "long", "int", "type"
    ], None)

    # * a collection of possible univariate functional that preserve all tensor dimensions.
    PASSTHROUGH_FUNCTIONALS = dict.fromkeys([
        # univariate pointwise functionals (activation functionals)
        "threshold", "relu", "hardtanh", "hardswish", "relu6", "elu", "selu", "celu", "leaky_relu",
        "prelu", "rrelu", "glu", "gelu", "logsigmoid", "hardshrink", "tanhshrink", "softsign",
        "softplus", "softmin", "softmax", "softshrink", "gumbel_softmax", "log_softmax", "tanh",
        "sigmoid", "hardsigmoid", "silu", "mish", "normalize",
        # univariate pointwise functionals (dropout functionals)
        "dropout", "alpha_dropout", "feature_alpha_dropout", "dropout2d", "dropout3d",
    ], None)

    # * a collection of possible univariate functional that preserve the channel dimension.
    # * In general, such operators are pass-through nodes and do not impose constraints.
    # * input can be provided as first arg or kwarg --> remaining args, kwargs are irrelevant.
    # * These functions are part of torch.nn.functional and can also appear as in-place operators.
    # * In-place operators do not need special treatment since they input already has the correct
    #   dependency.
    PASSTHROUGH_FUNCTIONALS.update({
        # univariate feature preserving functionals (pooling functionals)
        **{key: {0, 1, -2} for key in ["avg_pool1d", "max_pool1d", "max_unpool1d", "lp_pool1d",
                                       "adaptive_max_pool1d", "adaptive_avg_pool1d", "fractional_max_pool1d"]},
        **{key: {0, 1, -3} for key in ["avg_pool2d", "max_pool2d", "max_unpool2d", "lp_pool2d",
                                       "adaptive_max_pool2d", "adaptive_avg_pool2d", "fractional_max_pool2d"]},
        **{key: {0, 1, -4} for key in ["avg_pool3d", "max_pool3d", "max_unpool3d", "adaptive_max_pool3d",
                                       "adaptive_avg_pool3d", "fractional_max_pool3d"]},
        # univariate feature preserving functionals (vision functionals)
        **{key: {0, 1} for key in ["pad", "interpolate", "upsample", "upsample_nearest",
                                   "upsample_bilinear", "upsample_bicubic", "grid_sample"]},
    })

    # * a collection of possible univariate modules that preserve all tensor dimensions.
    PASSTHROUGH_MODULES = dict.fromkeys([
        # univariate pointwise modules (activation layers)
        "ELU", "Hardshrink", "Hardsigmoid", "Hardtanh", "Hardswish", "LeakyReLU", "LogSigmoid",
        "PReLU", "ReLU", "ReLU6", "RReLU", "SELU", "CELU", "GELU", "Sigmoid", "SiLU", "Mish",
        "Softplus", "Softshrink", "Softsign", "Tanh", "Tanhshrink", "Threshold", "GLU",
        # univariate feature-preserving modules (activation layers)
        "Softmin", "Softmax", "Softmax2d", "LogSoftmax",
        # univariate feature-preserving modules (linear layers)
        "Identity",
        # univariate feature-preserving modules (dropout layers)
        "Dropout", "Dropout2d", "Dropout3d", "AlphaDropout", "FeatureAlphaDropout",
    ], None)

    # * a collection of possible univariate modules that preserve the channel dimension.
    # * In general, such modules are pass-through nodes and do not impose constraints.
    # * input can be provided as first arg or kwarg --> remaining args, kwargs are irrelevant.
    # * These functions are part of torch.nn and do not have in-place equivalents.
    PASSTHROUGH_MODULES.update({
        # univariate feature preserving modules (pooling+padding layers)
        **{key: {0, 1, -2} for key in ["MaxPool1d", "MaxUnpool1d", "AvgPool1d", "FractionalMaxPool1d", "LPPool1d",
                                       "AdaptiveMaxPool1d", "AdaptiveAvgPool1d", "ReflectionPad1d", "ReplicationPad1d",
                                       "ZeroPad1d", "ConstantPad1d"]},
        **{key: {0, 1, -3} for key in ["MaxPool2d", "MaxUnpool2d", "AvgPool2d", "FractionalMaxPool2d", "LPPool2d",
                                       "AdaptiveMaxPool2d", "AdaptiveAvgPool2d", "ReflectionPad2d", "ReplicationPad2d",
                                       "ZeroPad2d", "ConstantPad2d"]},
        **{key: {0, 1, -4} for key in ["MaxPool3d", "MaxUnpool3d", "AvgPool3d", "FractionalMaxPool3d", "LPPool3d",
                                       "AdaptiveMaxPool3d", "AdaptiveAvgPool3d", "ReflectionPad3d", "ReplicationPad3d",
                                       "ZeroPad3d", "ConstantPad3d"]},
        # univariate feature-preserving modules (upsample layers)
        **{key: {1} for key in ["Upsample", "UpsamplingNearest2d", "UpsamplingBilinear2d"]},
    })

    # fmt: on

    @classmethod
    def _get_passthrough_targets(cls) -> dict[NodeTarget, set[int]]:
        """Compile and return list of passthrough node targets."""
        if cls._passthrough_targets is not None:
            return cls._passthrough_targets

        targets: dict[NodeTarget, set[int]] = {}

        # gather all ops
        ops_modules = [torch, torch.Tensor, operator]
        for op, edim in cls.PASSSTHROUGH_OPS.items():
            for suffix in ["", "_"]:
                op_candidate = op + suffix
                for module in ops_modules:
                    if hasattr(module, op_candidate):
                        targets[getattr(module, op_candidate)] = edim

        # gather all functionals
        for fn, edim in cls.PASSTHROUGH_FUNCTIONALS.items():
            for suffix in ["", "_"]:
                func_candidate = fn + suffix
                if hasattr(F, func_candidate):
                    targets[getattr(F, func_candidate)] = edim

        # gather all modules
        for mod, edim in cls.PASSTHROUGH_MODULES.items():
            if hasattr(nn, mod):
                targets[getattr(nn, mod)] = edim

        cls._passthrough_targets = targets
        return targets

    @classmethod
    def _get_filtered_list(cls, filt_fn: Callable[[set[int] | None], bool]) -> list[NodeTarget]:
        """Return list of nodes that satisfy filter function."""
        return [n for n, edim in cls._get_passthrough_targets().items() if filt_fn(edim)]

    @classmethod
    def is_shape_preserving(cls, target: NodeTarget, gc: GraphCollection) -> bool:
        """Return True if as pass-through node is fully shape-preserving (elastic dims is None)."""
        return cls._is_from_node_list(target, cls._get_filtered_list(lambda edim: edim is None), gc)

    def is_special_node(self, node: Node, target: NodeTarget) -> bool:
        """Check if node is a pass-through node (elastic dim is any)."""
        return self._is_from_node_list(target, self._get_filtered_list(lambda _: True), self._gc)

    def process(self, node: Node, id: int, input_nodes: list[Node]) -> None:
        """Enforce passthrough constraint on node i.e. the link node to primary input node."""
        # check for edim of current node
        # note that we assume that the target is present in the passthrough targets!
        # (otherwise process() should have never been called)
        pt_targets = self._get_passthrough_targets()
        target = self._get_node_target(node)
        edim = pt_targets[target] if target in pt_targets else pt_targets[type(target)]

        # if edim is None, we can simply process it since it will always be compatible
        if edim is None:
            return self._process_passthrough(node, id, input_nodes)

        # get edims of root target of primary input node
        t_root = self._get_root_target(input_nodes[0])
        edims_input = [sym.elastic_dims for _, sym in self.named_searchable_out_symbols(t_root)]

        # if edim is compatible with edims_input, we can process it as well
        # Note that all() returns True for empty iterators, which is desired here!
        if all(edim & e_in for e_in in edims_input):
            return self._process_passthrough(node, id, input_nodes)

        # otherwise we need to handle it as boundary
        self._process_boundary(node, id, input_nodes)


@GraphDependencyProcessor.register_node_processor
class MultivariateDimensionPreservingNodeProcessor(NodeProcessor):
    """Processor for handling multivariate feature dimension preserving nodes."""

    _targets: list[NodeTarget] | None = None

    # fmt: off

    # * a collection of possible multivariate operators that preserve the feature dimension
    # * In general, those operators require input nodes to be "synced" in the sense that all input
    #   nodes need to have the same shape. As such they impose a constraint on the input nodes to
    #   have the same number of features in the feature dimension. Specifically:
    #   - if all inputs are dynamic modules we can sync the feature dimension via hparams
    #   - if any input is _not_ a dynamic module we need to fix the feature dimension of all dynamic
    #     modules to the original input shape.
    #   - Note that if the user provides a specialization for one of the args (e.g. a constant),
    #     then the specialization will not be part of the input nodes in the DAG. This is quite a
    #     tricky case and might lead to unexpected behavior, e.g., if the specialization is a tensor
    #     with a fixed dimension. **This is an inherit limitation of the symbolic tracing.**
    # * The "out" keyword argument that will store the result in the provided tensor needs to be
    #   treated specially. Also see discussion of "out" keyword above.
    # * These operators can be in the torch, torch.Tensor, and the native operator module and can
    #   also appear as in-place operator with trailing underscore.
    # * In-place operators do not need special treatment. However, we should consider that the
    #   hparam infrastructure needs to be recursive at runtime (e.g. dynamic lookups of active).
    OPS = [
        # multivariate pointwise ops
        # gathered from https://pytorch.org/docs/stable/torch.html#pointwise-ops
        "add", "addcdiv", "addcmul", "clamp", "clip", "copysign", "div", "divide", "float_power",
        "floor_divide", "fmod", "ldexp", "lerp", "logaddexp", "logaddexp2", "logical_and",
        "logical_or", "logical_xor", "hypot", "igamma", "igammac", "mul", "multiply", "nextafter",
        "pow", "remainder", "sub", "subtract", "true_divide", "xlogy",
        # multivariate comparison ops
        # gathered from https://pytorch.org/docs/stable/torch.html#comparison-ops
        "eq", "ge", "greater_equal", "gt", "greater", "isclose", "isin", "le", "less_equal", "lt",
        "less", "maximum", "minimum", "fmax", "fmin", "ne", "not_equal",
        # Other ops
        # gathered from https://pytorch.org/docs/stable/torch.html#other-operations
        "gcd", "lcm",
        # built-in ops (with different names than torch ops)
        "and_", "floordiv", "mod", "or", "truediv", "xor", "iadd", "iand", "ifloordiv", "imod",
        "imul", "ior", "ipow", "isub", "itruediv", "ixor",
    ]

    # fmt: on

    @classmethod
    def _get_supported_targets(cls) -> list:
        """Compile and return list of feature-dimension preserving nodes."""
        if cls._targets is not None:
            return cls._targets

        targets = []
        # gather all ops
        ops_modules = [torch, torch.Tensor, operator]
        for op in cls.OPS:
            for suffix in ["", "_"]:
                op_candidate = op + suffix
                for module in ops_modules:
                    if hasattr(module, op_candidate):
                        targets.append(getattr(module, op_candidate))  # noqa: PERF401

        cls._targets = targets
        return targets

    def is_special_node(self, node: Node, target: NodeTarget) -> bool:
        """Return True if node is a constraint."""
        return self._is_from_node_list(target, self._get_supported_targets(), self._gc)

    def process(self, node: Node, id: int, input_nodes: list[Node]) -> None:
        """Enforce multivariate dimension preserving constraint on node."""
        # synchronize input nodes
        first_node = self._synchronize_nodes(input_nodes)

        if first_node is None:
            # in this case synchronization failed and we have to handle node as boundary
            self._process_boundary(node, id, input_nodes)
        else:
            # link node to first_node of synchronization process if successful
            self._dependency_map.link_nodes(node, first_node)


def is_in_slice_range(slice_obj: slice, num: int) -> bool:
    """Check if a number is in the range of a slice (ignoring `step` field)."""
    if slice_obj.start is not None and num < slice_obj.start:
        return False
    return not (slice_obj.stop is not None and num >= slice_obj.stop)


@GraphDependencyProcessor.register_node_processor
class SpecialPassthroughNodeProcessor(NodeProcessor):
    """Processor for handling of particular nodes as passthrough.

    Currently supported:
    * getattr and getitem nodes related to basic cases.
    * torch.mean

    NOTE: we only support tensor.shape, tensor.size(), tensor.dtype, or tensor.device.
    For getitem, we support cases where input is any of these nodes and `searchable_tensor_dim`
    is not accessed.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the processor."""
        super().__init__(*args, **kwargs)
        self._node_to_elastic_dims: dict[Node, set[int]] = {}

    def reset(self) -> None:
        """Reset the processor."""
        self._node_to_elastic_dims.clear()

    def is_special_node(self, node: Node, target: NodeTarget) -> bool:
        """Check if the node is to be supported by this processor."""
        # tensor.shape or tensor.size() or tensor.dtype
        if (
            target == getattr
            and len(node.args) == 2
            and node.args[1] in ["shape", "dtype", "device"]
        ) or (target == torch.Tensor.size and len(node.args) == 1 and len(node.kwargs) == 0):
            in_target = self._get_root_target(node.args[0])
            if in_target in self._sym_map:
                syms_out = [sym for _, sym in self.named_out_symbols(in_target)]
                assert len(syms_out) == 1, "Expected exactly one symbol out of node."
                self._node_to_elastic_dims[node] = syms_out[0].elastic_dims
                return True
            return False

        # getitem on above nodes
        if target == operator.getitem and node.args[0] in self._node_to_elastic_dims:
            elastic_dims = self._node_to_elastic_dims[node.args[0]]
            if isinstance(node.args[1], int):
                return node.args[1] not in elastic_dims
            elif isinstance(node.args[1], slice):
                # all searchable dims semantically mean the same e.g. {1, -3},
                # we need to check that one of the dims is not in the slice range.
                return any(not is_in_slice_range(node.args[1], d) for d in elastic_dims)

        # torch.Tensor.mean or torch.mean handling
        if target in [torch.Tensor.mean, torch.mean]:
            primary_input = self._identify_in_out_nodes(node)[0][0]
            in_target = self._get_root_target(primary_input)
            syms_out = [sym for _, sym in self.named_out_symbols(in_target)]
            if len(syms_out) == 0:
                return False
            elif len(syms_out) > 1:
                raise RuntimeError("Expected exactly one symbol out of node.")
            # check for dims in args/kwargs
            if "dim" in node.kwargs:
                mean_dims = node.kwargs["dim"]
            elif len(node.args) > 1:
                mean_dims = node.args[1]
            else:
                # in this case it's mean over all dims and it's definitely not pass-through
                return False
            mean_dims = set(mean_dims) if isinstance(mean_dims, Iterable) else {mean_dims}
            return all(d not in mean_dims for d in syms_out[0].elastic_dims)

        return False

    def process(self, node: Node, id: int, input_nodes: list[Node]) -> None:
        """Process like a pass-through node."""
        self._process_passthrough(node, id, input_nodes)


class DanglingSymProcessor:
    @staticmethod
    def process(
        gc: GraphCollection,
        sym_map: SymMap,
        has_concrete_args: bool,
        _strict: bool,
    ):
        """Identify and process dangling hparams in the provided model."""
        # retrieve model
        model = sym_map.model

        # collect failing modules
        mods_fail = {n: gc.failure_msg(m) for n, m in model.named_modules() if gc.is_failed(m)}

        # collect set of unvisited modules
        mods_unvisited = [n for n, m in model.named_modules() if gc.is_unvisited(m)]

        # collect modules with dangling hparams
        mods_dangling = {
            sym_name.rpartition(".")[0]: model.get_submodule(sym_name.rpartition(".")[0])
            for sym_name, sym in sym_map.named_symbols()
            if sym.is_dangling
        }

        # get visited failing sub modules with and without dangling hparams
        visited_fail_dangling = [
            m for m in mods_fail if m not in mods_unvisited and m in mods_dangling
        ]
        visited_fail_no_dangling = [
            m for m in mods_fail if m not in mods_unvisited and m not in mods_dangling
        ]

        # deactivate cross-layer hparams of dangling models either when module was unvisited or
        # _strict==False. This might lead to unexpected behaviour though but it's best we can do.
        cross_hparams_dangling_unvisited = defaultdict(list)
        cross_hparams_dangling_visited_names = []

        for name, sym in sym_map.named_symbols():
            if not sym.is_cross_layer:
                continue
            mod_name, _, sym_name = name.rpartition(".")
            if mod_name not in mods_dangling:
                continue
            is_unvisited = gc.is_unvisited(mods_dangling[mod_name])
            if is_unvisited:
                cross_hparams_dangling_unvisited[mod_name].append(sym_name)
            else:
                cross_hparams_dangling_visited_names.append(name)
            if not _strict or is_unvisited:
                sym.disable()

        def mod_str(mod_name: str | Iterable) -> str | list[str]:
            def name_or_root(name: str) -> str:
                return name if name else '""'

            if isinstance(mod_name, str):
                return name_or_root(mod_name)
            return [name_or_root(n) for n in mod_name]

        # process warning for failed sub-modules without hparams
        if visited_fail_no_dangling:
            warn_msg = (
                "\nFound modules that cannot be traced (e.g. due to conditional branching"
                " based on tensor inputs in its forward method). Note that these modules were"
                " skipped and instead only their child modules have been traced.\nFor detailed"
                " warning including a stack trace of where it occurs, please run"
                " `torch.fx.symbolic_trace(model.get_submodule(<sub_module_name>))`. You may address"
                " these warnings if you wish to improve the search space of the model."
            )
            if has_concrete_args:
                warn_msg += (
                    "\nNote that the children were traced *without* the provided `concrete_args`, "
                    "which may lead to unexpected behavior!"
                )

            # map failure message to a list of submodules
            fail_msg_to_mods = defaultdict(list)
            for mod_name in visited_fail_no_dangling:
                fail_msg_to_mods[mods_fail[mod_name]].append(mod_str(mod_name))

            table = Table(
                "Submodule Names", "Warning", title=warn_msg, title_justify="left", show_lines=True
            )
            for fail_msg, mod_names in fail_msg_to_mods.items():
                table.add_row(", ".join(mod_names), fail_msg)  # type: ignore[arg-type]

            console = Console()
            with console.capture() as capture:
                console.print(table)
            warn_msg = capture.get()
            warnings.warn(warn_msg)

        # process warning for unvisited submodules and corresponding hparams
        if cross_hparams_dangling_unvisited:
            warn_msg = (
                "\nFound unvisited modules in model (e.g. due to conditional branching in its "
                "forward method). Note that these modules were skipped and cannot be traced. "
                "\nNote that when the skipped modules are activated in subsequent forward calls the"
                " model consistency is not guaranteed, which may lead to unexpected behavior!"
                "\nWithin unvisited modules, below hparams were disabled:"
            )
            table = Table(
                "Submodule Name",
                "Disabled Symparams",
                title=warn_msg,
                title_justify="left",
                show_lines=True,
            )
            for mod_name, hp_names in cross_hparams_dangling_unvisited.items():
                table.add_row(mod_name, ", ".join(mod_str(hp_names)))

            console = Console()
            with console.capture() as capture:
                console.print(table)
            warn_msg = capture.get()
            warnings.warn(warn_msg)

        # process potential errors with our tracing logic instead of the model
        error_msg = ""
        if visited_fail_dangling:
            found_msg = (
                "\nFound failing modules with symbolic sub-modules (likely due to tracing error)."
                " Note that sub-modules within failing modules cannot be traced:"
            )
            error_msg += "\n\t".join(
                [found_msg] + [f"{mod_str(k)}: {mods_fail[k]}" for k in visited_fail_dangling]
            )
        if cross_hparams_dangling_visited_names:
            found_msg = (
                "\nWithin visited modules, found unaccounted symbols of the model"
                " (likely due to tracing error):"
            )
            error_msg += "\n\t".join([found_msg, *mod_str(cross_hparams_dangling_visited_names)])
        if error_msg:
            if _strict:
                error_msg += (
                    "\nCannot process layer dependencies! \nConsider modifying the model"
                    " definition and forward method of failing sub-modules."
                    "\nAlternatively, only parts of the model can be traced."
                )
                raise RuntimeError(error_msg)
            else:
                error_msg += (
                    "\nFailing modules w/ dangling symbols are disabled (_strict==False)"
                    ", however, note that this might lead to unexpected behavior in the model!"
                )
                warnings.warn(error_msg)


class SequentialDepthProcessor:
    """A processor to check the depth choices of all sequential modules in a given model."""

    def __init__(
        self, gc: GraphCollection, sym_map: SymMap, concrete_args: dict[str, Any] | None = None
    ) -> None:
        """Initialize the processor."""
        self._gc = gc
        self._sym_map = sym_map
        self._model = sym_map.model
        self._concrete_args = concrete_args

    def process(self) -> None:
        self._process()
        self._post_process()

    def _process(self) -> None:
        """Process all sequentials in the provided model."""
        # For each sequential module, we go through each of its submodules and check whether
        # it is a skippable block. We then adjust the depth choices accordingly. The checks
        # are based on some heuristics, so they are fast but not perfect.
        for module, sym_dict in self._sym_map.items():
            dsym = sym_dict.get("depth")
            if not isinstance(dsym, SymDepth):
                continue

            # check through each block in the sequential and see if it's skippable
            for idx, mod in enumerate(module):
                dsym.set_skippable(idx, self._is_skippable(mod))

        # Initial check on the min-depth subnet (this check is necessary but not sufficient;
        # however, we do not run further checks for efficiency reasons).
        if self._check_min_depth_subnet():
            return

        # If the min-depth subnet check fails, that means skipping some blocks could cause
        # errors during the (fake) forward pass. In this case, we need to verify all depth
        # choices of each sequential module. Specifically, for each sequential module, we
        # keep removing the smallest depth choice until the check passes (locally). This
        # process could be very time-consuming, which is why we need pre-checks.
        for name, mod in self._sym_map.named_modules():
            dsym = self._sym_map[mod].get("depth")
            if not isinstance(dsym, SymDepth):
                continue
            skippable_idxs = dsym.skippable_idxs
            if not any(skippable_idxs):
                continue
            errors_max = self._get_tracing_errors()

            for idx in skippable_idxs:
                with self._set_depth({name: [idx]}):
                    error_idx = self._get_tracing_errors()
                if error_idx != errors_max:
                    dsym.set_skippable(idx, False)

        # Final check on the min-depth subnet (same as above, necessary but not sufficient).
        if self._check_min_depth_subnet():
            return

        # This should ideally be unreachable, but who knows. In this unexpected corner case,
        # we will remove all the other depth choices except the largest one.
        warnings.warn(
            "All symbolic depth choices are being removed due to unexpected tracing errors!"
        )
        for _, sym in self._sym_map.named_symbols():
            if isinstance(sym, SymDepth):
                sym.disable()

    def _post_process(self) -> None:
        # ensuring that all symbols that don't have variable depth are fully disabled --> instead of
        # showing up as non-constant symbol which would be trivial with no skippable blocks
        for _, sym in self._sym_map.named_symbols():
            if isinstance(sym, SymDepth) and not sym.is_constant and not any(sym.skippable_idxs):
                sym.disable()

    def _is_skippable(self, module: nn.Module) -> bool:
        """Check whether the module is skippable.

        This check pre-filters the unskippable block based on some heuristics. It is supposed
        to be fast but not perfect, necessary but not sufficient.
        """
        # TODO: we should add more heuristics-based checks here.
        return self._is_shape_preserving_residual_block(module)

    def _is_shape_preserving(self, target: NodeTarget | None) -> bool:
        """Check whether the target is shape-preserving.

        A target can be shape-preserving either if it's a shape-preserving pass-through node or a
        shape-preserving dynamic module.

        Note that by convention a non-existing target (``None``) is considered shape-preserving.
        """
        return (
            target is None
            or PassthroughUnivariateNodeProcessor.is_shape_preserving(target, self._gc)
            or (target in self._sym_map and self._sym_map.is_shape_preserving(target))
        )

    def _is_shape_preserving_residual_block(self, module: nn.Module) -> bool:
        """Check whether the module is a shape-preserving residual block."""
        graph = RobustTracer().trace(module, None)

        # Ensure the module is traceable.
        if not graph.nodes:
            return False

        inputs = [node for node in graph.nodes if node.op == "placeholder"]
        outputs = [node for node in graph.nodes if node.op == "output"]

        # TODO: support blocks with multiple inputs or outputs.
        if len(inputs) != 1 and len(outputs) != 1:
            return False

        # Breadth-first search from output to input to find a path with residual connection
        # and only univariate shape-preserving operators.
        queue = deque([(x, False) for x in outputs])
        visited = set(outputs)
        while queue:
            x, residual = queue.popleft()
            if x in inputs and residual:
                return True
            for y in x.all_input_nodes:
                if y in visited:
                    continue
                target = _get_node_target(y, root=module)
                if target in [operator.add, torch.add, torch.Tensor.add, torch.Tensor.add_]:
                    residual = True
                elif not self._is_shape_preserving(target):
                    continue

                queue.append((y, residual))
                visited.add(y)
        return False

    def _get_tracing_errors(self) -> list[str]:
        """Get all tracing errors in the current model."""
        gc = recursive_trace(self._model, self._concrete_args)
        return sorted(gc._tracer._tracer_failed.values())

    # TODO: here we assume that setting depth implies using an identity module instead. This should
    # also be reflected in the future dynamic module infrastructure because that can make a
    # difference in the forward pass (e.g. DepthModel2 in test_model.py).
    @contextmanager
    def _set_depth(self, config: dict[str, list[int]]) -> Generator[None, None, None]:
        """Set the indices of specified depth modules to identities."""
        # temporarily replace underlying members of the sequential modules with identities
        # NOTE: we might have "nested" sequential modules, so we need to correctly handle the
        # order in which we set the depths. Specifically, we need to modify submodules
        # first and only then parent modules.
        # We can achieve this by simply sorting by the keys (submodules will appear before parent!)
        module_storage = {}
        for name in sorted(config, reverse=True):
            module = self._model.get_submodule(name)
            module_storage[name] = {}
            for idx in config[name]:
                module_storage[name][idx] = module[idx]
                module[idx] = nn.Identity()

        try:
            yield
        finally:
            # restore the original model
            # note that we need to restore the modules in the reverse order of removal above
            # --> parent module before submodule (use sorting as by above)
            for name in sorted(config):
                module = self._model.get_submodule(name)
                submods = module_storage.pop(name)
                for idx, submod in submods.items():
                    module[idx] = submod
            assert not module_storage, f"Some modules are not restored: {module_storage}!"

    def _check_min_depth_subnet(self) -> bool:
        """Min-depth subnet check.

        This check is designed to verify that the min-depth subnet is runnable (since it
        is the hardest one within the search space?). As we cannot run a real forward pass,
        we instead run a symbolic tracing pass and compare its error with the original
        symbolic tracing over the full model (i.e., max-depth subnet). The check will pass
        as long as there are no additional errors caused by the change of depth.

        Note that passing this check does not necessarily mean that the min-depth subnet is
        runnable. There could still be new errors introduced at the module level where there
        is a tracing error. In this case, these new errors will not be raised and therefore
        not discovered.
        """
        # collect tracing error for regular network
        error_max = self._get_tracing_errors()

        # collect tracing for min-depth network
        config_min_depth = {}
        for name, sym_dict in self._sym_map.named_sym_dicts():
            sym_depth = sym_dict.get("depth")
            if not isinstance(sym_depth, SymDepth):
                continue
            config_min_depth[name] = sym_depth.skippable_idxs
        with self._set_depth(config_min_depth):
            error_min = self._get_tracing_errors()

        # return error comparison result
        return error_max == error_min


def analyze_symbols(
    model: nn.Module, concrete_args: dict[str, Any] | None = None, _strict: bool = True
) -> SymMap:
    """Trace layer dependencies and return a valid symbolic mapping.

    The tracing process happens in-place.

    Args:
        model: Model to be traced.
        concrete_args: [FOR DEBUG ONLY!] Concrete arguments to model.forward() passed to the
            symbolic tracer.
        _strict: [FOR DEBUG ONLY!] Whether to enforce strict dependency tracing.
    """
    # initialize symbolic map
    sym_map = SymMap(model)

    # no need to trace dependencies for models with no symbolic layers
    if not sym_map:
        return sym_map

    # trace the model recursively and run the graph dependency tracer
    graph_collection = recursive_trace(model, concrete_args)

    # analyze dependencies for each captured graph
    for mod in graph_collection:
        GraphDependencyProcessor(mod, graph_collection, sym_map).process()

    # finalize dependency processing and do some clean-ups and sanity checks
    DanglingSymProcessor.process(graph_collection, sym_map, concrete_args is not None, _strict)

    # process dynamic-sequentials with depth choices
    SequentialDepthProcessor(graph_collection, sym_map, concrete_args).process()

    # clean up symbolic map (i.e. remove unused/constant modules)
    sym_map.prune()

    # return symbolic map with trace results
    return sym_map
