# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Analyze shared native layout metadata and commit real data/interface adapters."""

from typing import TYPE_CHECKING

import numpy as np
from onnxscript import GraphBuilder, ir

from ..common import constant_array, static_shape

if TYPE_CHECKING:
    from ..context import DlaGraph

__all__ = ["LayoutBoundaryCommit", "SplitAxisLayout", "snapshot"]


_ALL_DATA_INPUTS = {
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Pow",
    "Max",
    "Min",
    "Sum",
    "Concat",
    "Where",
    "And",
    "Or",
    "Xor",
    "Equal",
    "Greater",
    "Less",
    "GreaterOrEqual",
    "LessOrEqual",
    "MatMul",
}
_PREPARED_OPERATORS = _ALL_DATA_INPUTS | {
    "Abs",
    "ArgMax",
    "AveragePool",
    "Cast",
    "Clip",
    "Conv",
    "ConvTranspose",
    "Cos",
    "DepthToSpace",
    "DequantizeLinear",
    "Expand",
    "Flatten",
    "Gather",
    "GatherElements",
    "Gelu",
    "GlobalAveragePool",
    "GridSample",
    "GroupQueryAttention",
    "Identity",
    "InstanceNormalization",
    "LayerNormalization",
    "LeakyRelu",
    "LogSoftmax",
    "LpNormalization",
    "MaxPool",
    "Neg",
    "Not",
    "QuantizeLinear",
    "ReduceMax",
    "ReduceMean",
    "ReduceMin",
    "ReduceSum",
    "Relu",
    "Reshape",
    "Resize",
    "Sigmoid",
    "Sin",
    "Slice",
    "Softmax",
    "SpaceToDepth",
    "Split",
    "Sqrt",
    "Squeeze",
    "Tanh",
    "Tile",
    "TopK",
    "Transpose",
    "Unsqueeze",
}


def data_ports(node: ir.Node) -> tuple[int, ...]:
    """Separate activation/index data from shape, quantization and weight controls."""
    if node.domain:
        if (
            node.domain != "com.microsoft"
            or node.op_type
            not in {"QuantizeLinear", "DequantizeLinear", "GroupQueryAttention", "Gelu"}
            or node.graph is None
            or node.graph.opset_imports.get("com.microsoft") != 1
        ):
            return ()
    if node.op_type in _ALL_DATA_INPUTS:
        return tuple(range(len(node.inputs)))
    if node.op_type in {"GatherElements", "GridSample"}:
        return (0, 1)
    if node.op_type == "GroupQueryAttention":
        return (0, 1, 2)
    return (0,) if node.op_type in _PREPARED_OPERATORS else ()


def canonical_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Use leading singleton promotion and the supported singleton 5D collapse."""
    if len(shape) <= 4:
        return (1,) * (4 - len(shape)) + shape
    if len(shape) == 5 and 1 in shape:
        axis = shape.index(1)
        return shape[:axis] + shape[axis + 1 :]
    raise NotImplementedError(f"TODO: additional DLA layout transform for shape {shape}")


def snapshot(model: ir.Model):
    """Freeze native dimensions and trace live data backward from public outputs.

    Control-only view chains and fixed Conv weights stay native. A shared value
    reached as data is planned once; boundary commit preserves its control users.
    """
    native = {
        node: (
            tuple(static_shape(value) for value in node.inputs),
            tuple(static_shape(value) for value in node.outputs),
        )
        for node in model.graph
    }
    planned = {}
    pending = [(value, False) for value in model.graph.outputs]
    # Gather indices are native controls, but their TopK score computation is data.
    pending.extend(
        (node.inputs[0], True)
        for node in model.graph
        if node.op_type == "TopK" and data_ports(node)
    )
    seen = set()
    while pending:
        value, required = pending.pop()
        if (value, required) in seen:
            continue
        seen.add((value, required))
        shape = static_shape(value)
        producer = value.producer()
        ports = data_ports(producer) if producer is not None else ()
        if (
            not required
            and shape is not None
            and len(shape) == 1
            and producer is not None
            and producer.op_type == "Identity"
            and constant_array(producer.inputs[0]) is not None
        ):
            continue
        if producer is not None and not ports:
            if required:
                if shape is None or (shape and shape != canonical_shape(shape)):
                    raise NotImplementedError(
                        f"TODO: additional DLA layout transform for native {producer.op_type} output {value.name!r}"
                    )
            continue
        if shape is None:
            raise NotImplementedError(f"TODO: static DLA data dimensions for {value.name!r}")
        if not shape and (
            producer is None
            or producer.op_type
            not in {"ArgMax", "ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin"}
        ):
            continue
        planned[value.name] = canonical_shape(shape)
        if producer is not None:
            for port in ports:
                source = producer.inputs[port]
                if source is not None:
                    pending.append((source, True))
    return native, planned


def _mapped_axis(axis, source, target):
    if source is None or target is None:
        raise ValueError("TODO: provide static shapes for DLA axis conversion")
    if not -len(source) <= axis < len(source):
        raise ValueError("Operator axis is outside its native data rank")
    if len(source) <= len(target):
        return axis + len(target) - len(source) if axis >= 0 else axis
    removed = [
        index
        for index, dim in enumerate(source)
        if dim == 1 and source[:index] + source[index + 1 :] == tuple(target)
    ]
    if not removed or axis % len(source) == removed[0]:
        raise ValueError("TODO: define an unambiguous axis map for this collapsed tensor")
    mapped = axis % len(source) - int(axis % len(source) > removed[0])
    return mapped - len(target) if axis < 0 else mapped


def _axis_output_shapes(
    node, native_outputs, input_shape, native_axis, mapped_axis, planned_shapes
):
    physical, targets = [], []
    for value, native in zip(node.outputs, native_outputs, strict=True):
        if native is None:
            raise ValueError("TODO: static axis-operator output dimensions for layout conversion")
        dims = list(input_shape)
        dims[mapped_axis] = native[native_axis]
        physical.append(tuple(dims))
        targets.append(tuple(planned_shapes.get(value.name, dims)))
    return tuple(physical), tuple(targets)


class SplitAxisLayout(ir.passes.InPlacePass):
    """Remap Split's axis and output views without limiting its variadic output arity.

    For example, a rank-two Split on axis 1 becomes axis 3 in its leading-one
    4D view. A direct pass preserves every output; SDK patterns fix their arity.
    """

    def __init__(self, graph: "DlaGraph"):
        """Use the current run's immutable native shapes and active 4D plan."""
        self.graph = graph
        self.transformed_nodes = 0

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        """Update matched Split axes while retaining all original outputs."""
        self.transformed_nodes = 0
        builder = GraphBuilder(model.graph)
        for node, (inputs, outputs) in self.graph.native_shapes.items():
            if node.graph is not model.graph or node.domain or node.op_type != "Split":
                continue
            target = self.graph.planned_shapes.get(node.inputs[0].name)
            if target is None or not any(
                value.name in self.graph.planned_shapes for value in node.outputs
            ):
                continue
            axis = node.attributes.get_int("axis", 0)
            mapped = _mapped_axis(axis, inputs[0], tuple(target))
            physical, targets = _axis_output_shapes(
                node, outputs, target, axis, mapped, self.graph.planned_shapes
            )
            if mapped == axis and physical == targets:
                continue
            attrs = dict(node.attributes)
            attrs["axis"] = ir.AttrInt64("axis", mapped)
            replacement = ir.Node(
                "", "Split", node.inputs, attrs.values(), num_outputs=len(node.outputs)
            )
            replacement.meta["dla_native_inputs"] = inputs
            model.graph.insert_before(node, replacement)
            for original, result, dims, output_target in zip(
                node.outputs, replacement.outputs, physical, targets, strict=True
            ):
                result.type, result.shape = original.type, ir.Shape(dims)
                if dims != output_target:
                    result = builder.op.Reshape(
                        result,
                        self.graph.add_initializer(
                            builder, np.array(output_target, np.int64), "split_output_shape"
                        ),
                    )
                    result.type, result.shape = original.type, ir.Shape(output_target)
                result.name = original.name
                original.replace_all_uses_with(result, replace_graph_outputs=True)
            model.graph.remove(node, safe=True)
            self.transformed_nodes += 1
        return ir.passes.PassResult(model, self.transformed_nodes > 0)


class LayoutBoundaryCommit(ir.passes.InPlacePass):
    """Materialize planned 4D data views and restore native public endpoints.

    Shared controls and weights keep their original view. For instance a public
    [N,C] input receives one shared Unsqueeze for data consumers, while output
    Squeeze adapters restore the API's original rank and tensor names.
    """

    def __init__(self, graph: "DlaGraph"):
        """Bind the shared layout plan and initializer naming mechanism."""
        self.graph = graph
        self.adapter_count = 0

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        """Commit real adapters, metadata and the saved public interface."""
        self.adapter_count, modified = _commit(
            model,
            native_shapes=self.graph.native_shapes,
            planned_shapes=self.graph.planned_shapes,
            public_outputs=self.graph.public_outputs,
            add=self.graph.add_initializer,
        )
        return ir.passes.PassResult(model, modified)


def _commit(model: ir.Model, *, native_shapes: dict, planned_shapes: dict, public_outputs, add):
    """Promote data values once, share typed initializer views, and restore endpoints."""
    graph = model.graph
    builder = GraphBuilder(graph)
    reserved = {value.name for node in graph for value in node.outputs}
    reserved.update(value.name for value in (*graph.inputs, *graph.initializers.values()))
    count = 0
    modified = False
    views = {}

    def private_name(name):
        candidate, suffix = name + "_4d", 0
        while candidate in reserved:
            suffix += 1
            candidate = f"{name}_4d_{suffix}"
        reserved.add(candidate)
        return candidate

    def reshape(value, dims):
        nonlocal count
        key = value, tuple(dims)
        if key not in views:
            result = builder.op.Reshape(
                value, add(builder, np.array(dims, np.int64), "layout_view")
            )
            result.type, result.shape = value.type, ir.Shape(dims)
            views[key] = result
            count += 1
        return views[key]

    # Initializers and public inputs retain native ownership for control users.
    sources = dict.fromkeys((*graph.initializers.values(), *graph.inputs))
    for value in sources:
        target = planned_shapes.get(value.name)
        source = static_shape(value)
        if target is None or source == target:
            continue
        uses = [(node, port) for node, port in value.uses() if port in data_ports(node)]
        if not uses:
            continue
        if value.const_value is not None and not value.is_graph_input():
            result = add(
                builder, np.asarray(value.const_value.numpy()).reshape(target), "data_view"
            )
            count += 1
        elif source is not None and len(source) < 4:
            axes = add(builder, np.arange(4 - len(source), dtype=np.int64), "input_axes")
            result = builder.op.Unsqueeze(value, axes)
            result.type, result.shape = value.type, ir.Shape(target)
            count += 1
        else:
            result = reshape(value, target)
        for node, port in uses:
            node.replace_input_with(port, result)

    # Rewritten terminals retain logical names; new internal values already have real shapes.
    for node in tuple(graph):
        for value in node.outputs:
            target = planned_shapes.get(value.name)
            if target is not None and static_shape(value) != tuple(target):
                value.shape = ir.Shape(target)
                modified = True

    # A shared tensor may also supply a native shape/index/weight port.
    for node in tuple(graph):
        # Same-operator replacements retain native control contracts outside the eligibility snapshot.
        inputs = (
            native_shapes[node][0]
            if node in native_shapes
            else node.meta.get("dla_native_inputs", ())
        )
        promoted = "dla_native_inputs" in node.meta or any(
            value.name in planned_shapes for value in node.outputs
        )
        for port, source in enumerate(inputs):
            value = node.inputs[port]
            if (
                value is not None
                and (not promoted or port not in data_ports(node))
                and source is not None
                and static_shape(value) != source
            ):
                node.replace_input_with(port, reshape(value, source))

    for index, (name, source, dtype) in enumerate(public_outputs):
        value = graph.outputs[index]
        target = static_shape(value)
        if source is None or target == source:
            modified |= value.name != name
            value.name = name
            continue
        value.name = private_name(name)
        if len(source) < 4:
            axes = add(builder, np.arange(4 - len(source), dtype=np.int64), "output_axes")
            result = builder.op.Squeeze(value, axes)
            result.type, result.shape = dtype, ir.Shape(source)
            count += 1
        else:
            result = reshape(value, source)
        result.name = name
        graph.outputs[index] = result
    unused = ir.passes.common.RemoveUnusedNodesPass()(model)
    ordered = ir.passes.common.TopologicalSortPass()(model)
    return count, modified or count > 0 or unused.modified or ordered.modified
