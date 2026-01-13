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

# mypy: ignore-errors

from __future__ import annotations

from abc import ABC
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal, Optional, Sequence, Union

from typing_extensions import override

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import torch
import torch.distributed
import torch.nn as nn

from .passage import (
    InputArgs,
    OutputValue,
    Passage,
    PassageInputAdapter,
    PassageInputOverrides,
    PassageOutputAdapter,
    PassageOutputOverrides,
    Predicate,
    always_false_predicate,
)
from .utils import distributed_isend_obj, distributed_recv_obj, dynamo_skip

InputAdapter = Callable[[InputArgs], InputArgs]
OutputAdapter = Callable[..., OutputValue]


def default_input_adapter_fn(input_values: InputArgs) -> InputArgs:
    return input_values


def default_output_adapter_fn(v: OutputValue) -> OutputValue:
    return v


@dataclass
class IOReducer:
    pass


def default_input_reducer_fn(acc: InputArgs, input_override: InputArgs, *args):
    return acc + input_override


@dataclass
class InputReducer(IOReducer):
    reducer_fn: Callable[[InputArgs, InputArgs, InputArgs, int, list[InputArgs]], InputArgs] = (
        default_input_reducer_fn
    )

    def __call__(
        self,
        acc: InputArgs,
        input_override: InputArgs,
        original_input: InputArgs,
        index: int,
        all_input_overrides: list[InputArgs],
    ) -> InputArgs:
        result = self.reducer_fn(acc, input_override, original_input, index, all_input_overrides)
        return result

    @classmethod
    def default(cls) -> InputReducer:
        return InputReducer()


def default_output_reducer_fn(acc: OutputValue, input_override: OutputValue, *args):
    return input_override


@dataclass
class OutputReducer(IOReducer):
    reducer_fn: Callable[
        [OutputValue, OutputValue, Optional[OutputValue], int, list[OutputValue]], OutputValue
    ] = default_output_reducer_fn
    requires_original_output: bool = False

    def __call__(
        self,
        acc: OutputValue,
        output_override: OutputValue,
        original_output: Optional[OutputValue],
        index: int,
        all_output_overrides: list[OutputValue],
    ) -> InputArgs:
        result = self.reducer_fn(acc, output_override, original_output, index, all_output_overrides)
        return result

    @classmethod
    def default(cls) -> OutputReducer:
        return OutputReducer()


class Singleton(type):
    _instances = {}

    @override
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


@dataclass
class Target:
    @override
    def __hash__(self) -> int:
        return id(self)


@dataclass
class TargetWithInput(Target):
    @override
    def __hash__(self) -> int:
        return super().__hash__()

    def input(
        self,
        adapter: InputAdapter = default_input_adapter_fn,
        reducer: InputReducer = InputReducer.default(),
    ) -> InputDescriptor:
        result = InputDescriptor(self, input_name="", input_adapter=adapter, reducer=reducer)
        return result


@dataclass
class TargetWithNamedInputs(Target):
    @override
    def __hash__(self) -> int:
        return super().__hash__()

    def input(
        self,
        name: str,
        adapter: InputAdapter = default_input_adapter_fn,
        reducer: InputReducer = InputReducer.default(),
    ) -> InputDescriptor:
        result = InputDescriptor(self, input_name=name, input_adapter=adapter, reducer=reducer)
        return result


@dataclass
class TargetWithOutput(Target):
    @override
    def __hash__(self) -> int:
        return super().__hash__()

    def output(
        self,
        adapter: OutputAdapter = default_output_adapter_fn,
        reducer: OutputReducer = OutputReducer.default(),
    ) -> OutputDescriptor:
        result = OutputDescriptor(self, output_name="", output_adapter=adapter, reducer=reducer)
        return result


@dataclass
class TargetWithNamedOutputs(Target):
    @override
    def __hash__(self) -> int:
        return super().__hash__()

    def output(
        self,
        name: str,
        adapter: OutputAdapter = default_output_adapter_fn,
        reducer: OutputReducer = OutputReducer.default(),
    ) -> OutputDescriptor:
        result = OutputDescriptor(self, output_name=name, output_adapter=adapter, reducer=reducer)
        return result


@dataclass
class ExternalTarget(TargetWithNamedInputs, TargetWithNamedOutputs, metaclass=Singleton):
    """External target for stitched modules."""

    @override
    def __hash__(self) -> int:
        return super().__hash__()


@dataclass
class ConstantTarget(TargetWithOutput):
    name: str
    value: Any

    @override
    def __hash__(self) -> int:
        return super().__hash__()


@dataclass
class FunctionTarget(TargetWithInput, TargetWithOutput):
    name: str
    function: Callable[..., Any]

    @override
    def __hash__(self) -> int:
        return super().__hash__()


@dataclass
class ModuleTarget(TargetWithNamedInputs, TargetWithNamedOutputs):
    name: str
    module: nn.Module

    @override
    def __str__(self) -> str:
        return f"ModuleTarget({self.name})"

    @override
    def __repr__(self) -> str:
        return str(self)

    @override
    def __hash__(self) -> int:
        return super().__hash__()


@dataclass
class RemoteTarget(Target):
    peer_rank: Union[int, Sequence[int]]
    process_group: Optional[torch.distributed.ProcessGroup] = None
    blocking: bool = True

    @override
    def __hash__(self) -> int:
        return super().__hash__()

    def value(
        self,
        name: str,
        adapter: OutputAdapter = default_output_adapter_fn,
        reducer: OutputReducer = OutputReducer.default(),
    ) -> OutputDescriptor:
        result = OutputDescriptor(self, output_name=name, output_adapter=adapter, reducer=reducer)
        return result


@dataclass(frozen=True, eq=True)
class RemoteDataDescriptor(ABC):
    key: str


@dataclass(frozen=True, eq=True)
class RemoteTensorDataDescriptor(RemoteDataDescriptor):
    device: Literal["cuda", "cpu"]
    dtype: torch.dtype
    shape: torch.Size


@dataclass(frozen=True, eq=True)
class RemotePythonDataDescriptor(RemoteDataDescriptor):
    value: Any


@dataclass
class Node:
    target: Target
    stitches_to: list[StitchDescriptor] = field(default_factory=list)
    stitches_from: list[StitchDescriptor] = field(default_factory=list)

    @override
    def __hash__(self) -> int:
        return id(self)


@dataclass
class InputDescriptor:
    target: Target
    input_name: str = ""
    input_adapter: InputAdapter = field(default=default_input_adapter_fn)
    reducer: InputReducer = field(default_factory=InputReducer.default)

    @override
    def __hash__(self) -> int:
        return id(self)


@dataclass
class OutputDescriptor:
    target: Target
    output_name: str = ""
    output_adapter: OutputAdapter = field(default=default_output_adapter_fn)
    reducer: OutputReducer = field(default_factory=OutputReducer.default)

    @override
    def __hash__(self) -> int:
        return id(self)


IODescriptor = Union[InputDescriptor, OutputDescriptor]


@dataclass
class StitchDescriptor:
    source_descriptor: IODescriptor
    destination_descriptor: IODescriptor

    @override
    def __hash__(self) -> int:
        return id(self)


@dataclass
class StitchedModuleOutput:
    captured_inputs: dict[str, InputArgs]
    captured_outputs: dict[str, Any]


class StitchedModuleException(Exception):
    pass


class CantResolveNodeDependenciesException(StitchedModuleException):
    pass


class StitchedModule(nn.Module):
    def __init__(
        self,
        nodes: dict[Target, Node],
        capture_cache_outputs_predicate: Predicate = always_false_predicate,
        early_exit=True,
        ignore_extra_overrides=False,
    ) -> None:
        super().__init__()
        self.nodes = nodes
        self.ignore_extra_overrides = ignore_extra_overrides
        external_nodes = [n for n in nodes.values() if isinstance(n.target, ExternalTarget)]
        remote_nodes = [n for n in nodes.values() if isinstance(n.target, RemoteTarget)]
        assert len(external_nodes) <= 1
        assert len(remote_nodes) + len(external_nodes) > 0
        self.external_node = external_nodes[0] if len(external_nodes) > 0 else None
        self.internal_nodes = [
            n for n in nodes.values() if not isinstance(n.target, ExternalTarget)
        ]
        self.values_from_node: dict[Node, dict[IODescriptor, Any]] = defaultdict(dict)
        self.values_to_node: dict[Node, dict[IODescriptor, Any]] = defaultdict(dict)

        self.node_passages: dict[Node, Passage] = {
            node: Passage.create(
                module=node.target.module,
                inputs_to_capture=set(
                    s.source_descriptor.input_name
                    for s in node.stitches_from
                    if isinstance(s.source_descriptor, InputDescriptor)
                ),
                outputs_to_capture=set(
                    s.source_descriptor.output_name
                    for s in node.stitches_from
                    if isinstance(s.source_descriptor, OutputDescriptor)
                ),
                capture_cache_outputs_predicate=capture_cache_outputs_predicate,
                early_exit=early_exit,
                name=getattr(node.target, "name", None),
            )
            for node in self.internal_nodes
            if isinstance(node.target, ModuleTarget)
        }

        self.passage_modules = nn.ModuleDict(
            {
                f"node_{node_index}": self.node_passages[node]
                for node_index, node in enumerate(nodes.values())
                if node in self.node_passages
            }
        )
        self.adapter_modules = nn.ModuleDict(
            {
                f"node_{node_index}__stitch_{stitch_index}__{descriptor_name}": adapter
                for node_index, node in enumerate(nodes.values())
                for stitch_index, stitch in enumerate(node.stitches_from + node.stitches_to)
                for descriptor_name, descriptor in (
                    ("source", stitch.source_descriptor),
                    ("destination", stitch.destination_descriptor),
                )
                for adapter in [
                    descriptor.input_adapter
                    if isinstance(descriptor, InputDescriptor)
                    else descriptor.output_adapter
                ]
                if isinstance(adapter, nn.Module)
            }
        )

    def create_input_overrides(
        self, values_to_node: dict[IODescriptor, Any]
    ) -> PassageInputOverrides:
        input_descriptors_by_group = defaultdict[str, list[InputDescriptor]](list)
        for io_descriptor in values_to_node.keys():
            if isinstance(io_descriptor, InputDescriptor):
                input_descriptors_by_group[io_descriptor.input_name].append(io_descriptor)

        input_overrides = PassageInputOverrides()
        for group, input_descriptors in input_descriptors_by_group.items():
            reducers = [d.reducer for d in input_descriptors]

            def create_reducer(input_descriptors=input_descriptors, reducers=reducers):
                inputs = [values_to_node[d] for d in input_descriptors]

                def reducer_fn(
                    original_input: InputArgs,
                    module_name: Optional[str],
                    module: Optional[nn.Module],
                ) -> InputArgs:
                    acc = InputArgs()
                    for i, (input_, reducer) in enumerate(zip(inputs, reducers)):
                        acc = reducer(acc, input_, original_input, i, inputs)
                    return acc

                return reducer_fn

            input_override = PassageInputAdapter(create_reducer())
            input_overrides[group] = input_override

        return input_overrides

    def create_output_overrides(
        self, values_to_node: dict[IODescriptor, Any]
    ) -> PassageOutputOverrides:
        output_descriptors_by_group = defaultdict[str, list[OutputDescriptor]](list)
        for io_descriptor in values_to_node.keys():
            if isinstance(io_descriptor, OutputDescriptor):
                output_descriptors_by_group[io_descriptor.output_name].append(io_descriptor)

        output_overrides = PassageOutputOverrides()
        for group, output_descriptors in output_descriptors_by_group.items():
            reducers = [d.reducer for d in output_descriptors]
            requires_original_output = any(r.requires_original_output for r in reducers)

            def create_reducer(reducers=reducers):
                outputs = [values_to_node[d] for d in output_descriptors]

                def reducer_fn(
                    original_output: Optional[OutputValue],
                    module_name: Optional[str],
                    module: Optional[nn.Module],
                ) -> OutputValue:
                    acc = None
                    for i, (output, reducer) in enumerate(zip(outputs, reducers)):
                        acc = reducer(acc, output, original_output, i, outputs)
                    return acc

                return reducer_fn

            reducer_fn = create_reducer()
            if requires_original_output:
                output_override = PassageOutputAdapter(reducer_fn)
            else:
                output_override = reducer_fn(None, None, None)

            output_overrides[group] = output_override

        return output_overrides

    @override
    def __call__(
        self,
        input_overrides: dict[str, Any],
        output_overrides: dict[str, Any],
        *args,
        **kwargs,
    ) -> StitchedModuleOutput:
        return super().__call__(input_overrides, output_overrides, *args, **kwargs)

    @override
    @dynamo_skip
    def forward(
        self,
        input_overrides: dict[str, Any],
        output_overrides: dict[str, Any],
        *args,
        **kwargs,
    ) -> StitchedModuleOutput:
        input_overrides = {k: InputArgs.from_value(v) for k, v in input_overrides.items()}

        self.values_from_node.clear()
        self.values_to_node.clear()

        unresolved_count: int = 0
        nodes_stack: list[Node] = (
            [] if self.external_node is None else [self.external_node]
        ) + self.internal_nodes
        while len(nodes_stack) > 0:
            node = nodes_stack.pop(0)
            values_from_node = self.values_from_node[node]
            values_to_node = self.values_to_node[node]

            if isinstance(node.target, ExternalTarget):
                assert self.external_node is not None

                if not self.ignore_extra_overrides:
                    input_override_names = set(input_overrides.keys())
                    external_node_input_names = set(
                        s.source_descriptor.input_name
                        for s in self.external_node.stitches_from
                        if isinstance(s.source_descriptor, InputDescriptor)
                    )
                    assert input_override_names == external_node_input_names
                    output_override_names = set(output_overrides.keys())
                    external_node_output_names = set(
                        s.source_descriptor.output_name
                        for s in self.external_node.stitches_from
                        if isinstance(s.source_descriptor, OutputDescriptor)
                    )
                    assert output_override_names == external_node_output_names

                for stitch in self.external_node.stitches_from:
                    if isinstance(stitch.source_descriptor, InputDescriptor):
                        orig_input_override = input_overrides[stitch.source_descriptor.input_name]
                        input_override = stitch.source_descriptor.input_adapter(orig_input_override)
                        values_from_node[stitch.source_descriptor] = input_override
                    elif isinstance(stitch.source_descriptor, OutputDescriptor):
                        orig_output_override = output_overrides[
                            stitch.source_descriptor.output_name
                        ]
                        output_override = stitch.source_descriptor.output_adapter(
                            orig_output_override
                        )
                        values_from_node[stitch.source_descriptor] = output_override
                    else:
                        raise RuntimeError("Shouldn't happen")

            else:
                if len(values_to_node) < len(node.stitches_to):
                    nodes_stack.append(node)
                    unresolved_count += 1
                    if unresolved_count >= len(nodes_stack):
                        raise CantResolveNodeDependenciesException(
                            "Can't resolve nodes dependencies"
                        )
                    continue

                if isinstance(node.target, ConstantTarget):
                    assert len(values_to_node) == 0

                    output_value = node.target.value

                    for stitch in node.stitches_from:
                        assert isinstance(stitch.source_descriptor, OutputDescriptor)
                        assert stitch.source_descriptor.output_name == ""
                        value = stitch.source_descriptor.output_adapter(output_value)
                        values_from_node[stitch.source_descriptor] = value

                elif isinstance(node.target, FunctionTarget):
                    assert all(
                        isinstance(v, InputDescriptor) and v.input_name == ""
                        for v in values_to_node
                    )

                    function_input_overrides = self.create_input_overrides(values_to_node)[""]

                    if isinstance(function_input_overrides, InputArgs):
                        input_args = function_input_overrides
                    else:
                        input_args = function_input_overrides(InputArgs(), None, None)

                    function_output = node.target.function(*input_args.args, **input_args.kwargs)

                    for stitch in node.stitches_from:
                        assert isinstance(stitch.source_descriptor, OutputDescriptor)
                        assert stitch.source_descriptor.output_name == ""
                        value = stitch.source_descriptor.output_adapter(function_output)
                        values_from_node[stitch.source_descriptor] = value

                elif isinstance(node.target, ModuleTarget):
                    passage = self.node_passages[node]
                    passage.input_overrides = self.create_input_overrides(values_to_node)
                    passage.output_overrides = self.create_output_overrides(values_to_node)
                    passage_output = passage(*args, **kwargs)

                    for stitch in node.stitches_from:
                        if isinstance(stitch.source_descriptor, InputDescriptor):
                            captured_input = passage_output.captured_inputs[
                                stitch.source_descriptor.input_name
                            ]
                            value = stitch.source_descriptor.input_adapter(captured_input)
                            values_from_node[stitch.source_descriptor] = value
                        elif isinstance(stitch.source_descriptor, OutputDescriptor):
                            captured_output = passage_output.captured_outputs[
                                stitch.source_descriptor.output_name
                            ]
                            value = stitch.source_descriptor.output_adapter(captured_output)
                            values_from_node[stitch.source_descriptor] = value
                        else:
                            raise RuntimeError("Shouldn't happen")

                elif isinstance(node.target, RemoteTarget):
                    assert all(
                        isinstance(v, OutputDescriptor) and v.output_name != ""
                        for v in values_from_node
                    )
                    assert all(
                        isinstance(v, OutputDescriptor) and v.output_name != ""
                        for v in values_to_node
                    )

                    process_group = node.target.process_group
                    peers = node.target.peer_rank
                    if not isinstance(peers, Sequence):
                        peers = [peers]

                    if len(values_to_node) > 0:
                        items_to_send = list(self.create_output_overrides(values_to_node).items())

                        data_descriptors: list[RemoteDataDescriptor] = []
                        tensors_to_send: list[torch.Tensor] = []

                        for key, value in items_to_send:
                            if isinstance(value, torch.Tensor):
                                if value.is_cuda:
                                    tensor_device = "cuda"
                                elif value.is_cpu:
                                    tensor_device = "cpu"
                                else:
                                    raise RuntimeError(
                                        f"Invalid tensor device to send to remote target: {value.device}"
                                    )

                                data_descriptor = RemoteTensorDataDescriptor(
                                    key=key,
                                    device=tensor_device,
                                    dtype=value.dtype,
                                    shape=value.shape,
                                )
                                tensors_to_send.append(value)

                            else:
                                data_descriptor = RemotePythonDataDescriptor(
                                    key=key,
                                    value=value,
                                )

                            data_descriptors.append(data_descriptor)

                        works: list[Optional[torch.distributed.Work]] = []
                        for peer in peers:
                            if process_group is not None:
                                peer = torch.distributed.get_global_rank(process_group, peer)

                            peer_works = distributed_isend_obj(data_descriptors, dst=peer)
                            works.extend(peer_works)

                            for tensor in tensors_to_send:
                                work = torch.distributed.isend(tensor, dst=peer)
                                works.append(work)

                        if node.target.blocking:
                            for work in works:
                                if work is not None:
                                    work.wait()

                        pass

                    if len(node.stitches_from) > 0:
                        assert len(peers) == 1, (
                            f"Cannot use multiple peers when using RemoteTarget as a source ({peers=})"
                        )
                        (peer,) = peers

                        if process_group is not None:
                            peer = torch.distributed.get_global_rank(process_group, peer)

                        data_descriptors = distributed_recv_obj(src=peer)
                        assert isinstance(data_descriptors, list)

                        tensors_to_recv: list[torch.Tensor] = []
                        received_values: dict[str, Any] = {}
                        for data_descriptor in data_descriptors:
                            if isinstance(data_descriptor, RemoteTensorDataDescriptor):
                                tensor = torch.empty(
                                    data_descriptor.shape,
                                    dtype=data_descriptor.dtype,
                                    device=data_descriptor.device,
                                )
                                tensors_to_recv.append(tensor)
                                received_values[data_descriptor.key] = tensor
                            elif isinstance(data_descriptor, RemotePythonDataDescriptor):
                                received_values[data_descriptor.key] = data_descriptor.value
                            else:
                                raise RuntimeError(
                                    f"Received invalid data descriptor from remote peer: {data_descriptor}"
                                )

                        works: list[Optional[torch.distributed.Work]] = []
                        for tensor in tensors_to_recv:
                            work = torch.distributed.irecv(tensor, src=peer)
                            works.append(work)

                        for work in works:
                            if work is not None:
                                work.wait()

                        for stitch in node.stitches_from:
                            if isinstance(stitch.source_descriptor, OutputDescriptor):
                                remote_output = received_values[
                                    stitch.source_descriptor.output_name
                                ]
                                value = stitch.source_descriptor.output_adapter(remote_output)
                                values_from_node[stitch.source_descriptor] = value
                            else:
                                raise RuntimeError("Shouldn't happen")
                else:
                    raise RuntimeError("Shouldn't happen")

            for stitch in node.stitches_from:
                dst_node = self.nodes[stitch.destination_descriptor.target]
                value = values_from_node[stitch.source_descriptor]

                if isinstance(stitch.destination_descriptor, InputDescriptor):
                    value = stitch.destination_descriptor.input_adapter(value)
                elif isinstance(stitch.destination_descriptor, OutputDescriptor):
                    value = stitch.destination_descriptor.output_adapter(value)
                else:
                    raise RuntimeError("Shouldn't happen")

                self.values_to_node[dst_node][stitch.destination_descriptor] = value

            unresolved_count = 0

        values_to_external_node = (
            {} if self.external_node is None else self.values_to_node[self.external_node]
        )
        output = StitchedModuleOutput(
            captured_inputs={
                k.input_name: v
                for k, v in values_to_external_node.items()
                if isinstance(k, InputDescriptor)
            },
            captured_outputs={
                k.output_name: v
                for k, v in values_to_external_node.items()
                if isinstance(k, OutputDescriptor)
            },
        )

        self.values_from_node.clear()
        self.values_to_node.clear()

        return output


class KnotException(Exception):
    pass


class LoopFoundException(KnotException):
    pass


class InputsLoopFoundException(LoopFoundException):
    pass


class OutputsLoopFoundException(LoopFoundException):
    pass


class MultipleExternalNodesException(KnotException):
    pass


class OnlyInternalNodesException(KnotException):
    pass


class Needle:
    def __init__(self) -> None:
        self.nodes = dict[Target, Node]()

    def get_node_for_target(self, target: Target) -> Node:
        if target not in self.nodes:
            node = Node(target=target)
            self.nodes[target] = node
        else:
            node = self.nodes[target]

        return node

    def stitch(self, src: IODescriptor, dst: IODescriptor) -> Self:
        descriptor = StitchDescriptor(source_descriptor=src, destination_descriptor=dst)

        src_node = self.get_node_for_target(descriptor.source_descriptor.target)
        dst_node = self.get_node_for_target(descriptor.destination_descriptor.target)

        if descriptor not in src_node.stitches_from:
            src_node.stitches_from.append(descriptor)

        if descriptor not in dst_node.stitches_to:
            dst_node.stitches_to.append(descriptor)

        return self

    def _search_loops(
        self,
        node: Node,
        expand_fn: Callable[[Node], Iterable[IODescriptor]],
        traversed_nodes: Optional[set[Node]] = None,
    ) -> bool:
        if isinstance(node.target, ExternalTarget):
            return False

        if traversed_nodes is None:
            traversed_nodes = set()

        if node in traversed_nodes:
            found_loop = True
        else:
            traversed_nodes = traversed_nodes | {node}
            found_loop = False
            descriptors = expand_fn(node)
            for descriptor in descriptors:
                stitch_node = self.get_node_for_target(descriptor.target)
                found_loop |= self._search_loops(stitch_node, expand_fn, traversed_nodes)

        return found_loop

    def _validate_nodes(self):
        # internal_nodes = [n for n in self.nodes.values() if not isinstance(n.target, (ExternalTarget, RemoteTarget))]
        external_nodes = [n for n in self.nodes.values() if isinstance(n.target, ExternalTarget)]
        remote_nodes = [n for n in self.nodes.values() if isinstance(n.target, RemoteTarget)]

        if len(external_nodes) + len(remote_nodes) == 0:
            raise OnlyInternalNodesException(f"Has only internal nodes")

        if len(external_nodes) > 1:
            raise MultipleExternalNodesException(
                f"Expected no more than 1 external node, found {len(external_nodes)}"
            )

        for i, node in enumerate(self.nodes.values()):
            found_inputs_loop = self._search_loops(
                node, lambda n: [s.source_descriptor for s in n.stitches_to]
            )
            if found_inputs_loop:
                raise InputsLoopFoundException(f"Found a loop in inputs of node {i}: {node}")

            found_outputs_loop = self._search_loops(
                node, lambda n: [s.destination_descriptor for s in n.stitches_from]
            )
            if found_outputs_loop:
                raise OutputsLoopFoundException(f"Found a loop in outputs of node {i}: {node}")

    def knot(
        self,
        capture_cache_outputs_predicate=always_false_predicate,
        early_exit=True,
        ignore_extra_overrides=False,
    ) -> StitchedModule:
        self._validate_nodes()

        module = StitchedModule(
            nodes=self.nodes,
            capture_cache_outputs_predicate=capture_cache_outputs_predicate,
            early_exit=early_exit,
            ignore_extra_overrides=ignore_extra_overrides,
        )

        return module
