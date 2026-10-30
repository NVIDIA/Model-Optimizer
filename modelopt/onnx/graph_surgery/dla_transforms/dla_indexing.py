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

"""DLA indexing, reduction and attention passes with deferred rank promotion."""

import math

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from ..utils.graph_utils import (
    GraphCache,
    _attribute,
    _shape_before_boundary_promotion,
    add_initializer,
    batch_replace_nodes,
    pad4d,
)
from .dla_constants_to_initializers import _add_legacy_initializer_inputs
from .dla_graph_cleanup import _RANK_AGNOSTIC_OPS, _subgraph_captures
from .dla_matmul_to_transpose_conv_transpose import _apply_matmul_to_transpose_conv_transpose
from .dla_model_boundaries import _BoundaryPreflightError, _has_native_symbolic_data, _native_conv1d

__all__ = []


def _insert_support_nodes(graph, nodes):
    """Insert generated support nodes without copying existing protobuf payloads."""
    positions = {id(node): index for index, node in enumerate(graph.node)}
    batch = {}
    pending = []
    for node in nodes:
        index = positions.get(id(node))
        if index is None:
            pending.append(node)
        elif pending:
            batch[index] = ([], pending)
            pending = []
    if pending:
        batch[len(graph.node)] = ([], pending)
    batch_replace_nodes(graph, batch)


def _operator_nodes(model, operators, domains=("", "ai.onnx")):
    """Select supported schemas without interpreting opaque model-local functions."""
    functions = {(fn.domain, fn.name, fn.overload) for fn in model.functions}
    return [
        node
        for node in model.graph.node
        if node.op_type in operators
        and node.domain in domains
        and not node.overload
        and (node.domain, node.op_type, node.overload) not in functions
    ]


def _known_rank(cache, name):
    """Read tensor rank even when individual dimensions are symbolic."""
    shape = cache.get_shape(name)
    if shape is not None:
        return len(shape)
    value = cache.vi_map.get(name)
    if value is not None and value.type.tensor_type.HasField("shape"):
        return len(value.type.tensor_type.shape.dim)
    return None


def _gather_flatten_count(cache, node, index_shape):
    """Return the flattened count only when leading padding preserves the native output layout."""
    shape = cache.get_shape(node.input[0])
    output = cache.get_shape(node.output[0])
    if (
        index_shape is None
        or shape is None
        or len(shape) not in (2, 3, 4)
        or output is None
        or len(output) > 4
    ):
        return None
    axis = _attribute(node, "axis")
    value = axis.i if axis is not None else 0
    if not -len(shape) <= value < len(shape):
        return None
    value = value % len(shape) + 4 - len(shape)
    prepared = pad4d(shape)
    count = math.prod(index_shape)
    return count if pad4d(output) == [*prepared[:value], count, *prepared[value + 1 :]] else None


def _gather_index_needs_view(cache, name, layout_nodes):
    """Find native index layouts that later rank stages would otherwise promote."""
    seen = set()
    producer = cache.get_producer(name)
    while (
        producer is not None
        and id(producer) in layout_nodes
        and producer.op_type in {"Cast", "Identity"}
    ):
        if name in seen:
            raise ValueError("Cyclic Gather index producers are unsupported")
        seen.add(name)
        name = producer.input[0]
        producer = cache.get_producer(name)
    return producer is not None and id(producer) in layout_nodes


def _gather_layout_reinterpreted(cache, name, supported, captured):
    """Prove that fixed four-dimensional reshapes erase a private Gather layout difference."""
    pending, seen = [name], set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        if current in cache.graph_output_names | captured:
            return False
        for consumer, port in cache.get_consumers(current):
            if id(consumer) not in supported or port != 0:
                return False
            if consumer.op_type in _RANK_AGNOSTIC_OPS and len(consumer.input) == 1:
                pending.extend(consumer.output)
                continue
            if (
                consumer.op_type != "Reshape"
                or consumer.input[1] in cache.graph_input_names
                or cache.get_shape(consumer.input[1]) != [4]
            ):
                return False
            target = cache.get_init_array(consumer.input[1])
            if target is None or np.any((target <= 0) & (target != -1)):
                return False
    return True


def _prepared_index_data(
    model, cache, prepared_ops, data_names, *, before_boundary=False, restoring_outputs=False
):
    """Prove leading-4D data from actual layouts and registered rank stages without mutation."""
    standard = {id(node) for node in _operator_nodes(model, {n.op_type for n in model.graph.node})}
    if any(op.domain == "com.microsoft" and op.version == 1 for op in model.opset_import):
        standard.update(
            id(node)
            for node in _operator_nodes(
                model,
                {"QuantizeLinear", "DequantizeLinear", "Gelu", "GroupQueryAttention"},
                ("com.microsoft",),
            )
        )
    local_functions = {(fn.domain, fn.name, fn.overload) for fn in model.functions}
    captured = _subgraph_captures(model.graph)
    matrices = set()
    if {"MatMulToTransposeConvTranspose", "RemoveIntermediarySqueezeAndUnsqueeze"} <= set(
        prepared_ops
    ):
        _apply_matmul_to_transpose_conv_transpose(
            model, defer_rank_conversion=True, validate_only=True, planned_outputs=matrices
        )
    expand_ops = {"Add", "Sub", "Mul", "Div", "LSTM"}

    def unsqueezed_rank(node):
        """Read adapter rank from metadata without decoding its fixed axes payload."""
        source_rank = _known_rank(cache, node.input[0])
        axes = cache.get_init(node.input[1]) if len(node.input) > 1 else None
        return (
            source_rank + math.prod(axes.dims)
            if axes is not None and source_rank is not None
            else None
        )

    def matrix_rank(name):
        """Prove that an operand is a matrix even when a generated adapter has no value info."""
        rank = _known_rank(cache, name)
        producer = cache.get_producer(name)
        if (
            rank is None
            and producer is not None
            and id(producer) in standard
            and producer.op_type == "Unsqueeze"
        ):
            rank = unsqueezed_rank(producer)
        return rank

    def classify(name):
        """Return a proven layout or the rank-preserving parents needed to establish it."""
        shape = cache.get_shape(name)
        if name in matrices or (shape is not None and len(shape) == 4):
            return True
        if (
            before_boundary
            and name in cache.graph_input_names
            and shape is not None
            and len(shape) in (2, 3)
        ):
            return True
        tensor = cache.get_init(name)
        if tensor is not None and len(tensor.dims) == 4:
            return True
        if name in cache.graph_input_names and (model.ir_version > 3 or tensor is None):
            return False
        if tensor is not None:
            users = [node for node, _ in cache.get_consumers(name)]
            if (
                "Non4dInitializers" not in prepared_ops
                or name in captured | cache.graph_output_names
                or any(
                    id(node) not in standard
                    or node.op_type in {"Gemm", "Unsqueeze", "GroupQueryAttention"}
                    or _native_conv1d(node, cache)
                    for node in users
                )
            ):
                return False
            operations = {node.op_type for node in users}
            return (
                len(tensor.dims) == 3
                or (
                    len(tensor.dims) == 2
                    and bool(operations & (expand_ops | {"Gather", "Where", "Split"}))
                )
                or (len(tensor.dims) == 1 and bool(operations & expand_ops))
            )
        node = cache.get_producer(name)
        if node is None or id(node) not in standard:
            return False
        if node.op_type == "Reshape":
            target = cache.get_shape(node.input[1])
            return target == [4] or (
                "Reshape" in prepared_ops
                and target is not None
                and len(target) == 1
                and target[0] in (1, 2, 3, 5)
                and not _has_native_symbolic_data(cache, node.input[0], local_functions)
            )
        if node.op_type == "Expand":
            target = cache.get_shape(node.input[1])
            return target == [4] or (
                "Expand" in prepared_ops
                and target is not None
                and len(target) == 1
                and 0 < target[0] < 4
                and not _has_native_symbolic_data(cache, node.input[0], local_functions)
            )
        if node.op_type == "Flatten":
            return node.op_type in prepared_ops
        if node.op_type in {"ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin"}:
            return [node.input[0]] if node.op_type in prepared_ops else False
        if node.op_type == "InstanceNormalization":
            return (
                "FixInstanceNormChannelMismatch" in prepared_ops
                or "RemoveIntermediarySqueezeAndUnsqueeze" in prepared_ops
            )
        if node.op_type == "GroupQueryAttention":
            return "GroupQueryAttention" in prepared_ops
        if node.op_type == "Conv":
            weight = cache.get_init(node.input[1])
            if weight is not None:
                return len(weight.dims) == 4
            kernel_rank = _known_rank(cache, node.input[1])
            if kernel_rank is not None:
                return kernel_rank == 4
            return [node.input[1]]
        if node.op_type == "MatMul" and (
            not before_boundary
            or restoring_outputs
            or {"MatMulToTransposeConvTranspose", "RemoveIntermediarySqueezeAndUnsqueeze"}
            <= set(prepared_ops)
        ):
            if not restoring_outputs and name in cache.graph_output_names | captured:
                return False
            ranks = [matrix_rank(value) for value in node.input[:2]]
            return (
                list(node.input[:2])
                if all(rank is not None and rank >= 2 for rank in ranks)
                else False
            )
        if node.op_type == "Transpose":
            perm = _attribute(node, "perm")
            if perm is not None and sorted(perm.ints) == [0, 1, 2, 3]:
                return True
        if node.op_type in {"Squeeze", "Unsqueeze"}:
            if node.op_type == "Unsqueeze":
                if unsqueezed_rank(node) == 4:
                    return True
            if "RemoveIntermediarySqueezeAndUnsqueeze" not in prepared_ops:
                return False
            source = node.input[0]
            if source in captured or (
                not restoring_outputs and name in captured | cache.graph_output_names
            ):
                return False
            if source in cache.graph_input_names:
                original = cache.get_shape(source)
                if original is None or len(original) not in (2, 3):
                    return False
                if node.op_type == "Squeeze":
                    return "Gather" not in node.name
                return shape is not None and (
                    pad4d(original)[-len(shape) :] == shape or "Reshape" in prepared_ops
                )
            return source not in cache.init_names | cache.graph_output_names
        if node.op_type in {"QuantizeLinear", "DequantizeLinear"}:
            return (
                [node.input[0]]
                if not before_boundary
                or cache.get_shape(node.input[1]) == []
                or node.op_type in prepared_ops
                else False
            )
        if node.op_type in {"Softmax", "LogSoftmax", "LayerNormalization"}:
            axis = _attribute(node, "axis")
            opset = next(
                (op.version for op in model.opset_import if op.domain in {"", "ai.onnx"}), 1
            )
            value = axis.i if axis is not None else (-1 if opset >= 13 else 1)
            return (
                [node.input[0]]
                if not before_boundary
                or value < 0
                or "axis_updated" in node.name
                or "RemoveIntermediarySqueezeAndUnsqueeze" in prepared_ops
                or node.op_type in prepared_ops
                or node.input[0] in cache.graph_input_names
                else False
            )
        if node.op_type in _RANK_AGNOSTIC_OPS | {"Gelu"}:
            return [value for value in node.input if value]
        if node.op_type == "Gather" or node.op_type in prepared_ops:
            return [node.input[0]]
        return False

    resolved, dependencies, active = {}, {}, set()
    pending = [(name, False) for name in data_names]
    while pending:
        name, expanded = pending.pop()
        if name in resolved:
            continue
        if expanded:
            resolved[name] = any(resolved[parent] for parent in dependencies[name])
            active.remove(name)
            continue
        if name in active:
            raise ValueError("Cyclic Gather data producers are unsupported")
        result = classify(name)
        if isinstance(result, bool):
            resolved[name] = result
            continue
        dependencies[name] = result
        active.add(name)
        pending.append((name, True))
        pending.extend((parent, False) for parent in result)
    return resolved


def _apply_gather(model, *, prepared_data_ops=()):
    """Prepare Gather indices for the subsequent ordered data-rank promotion."""
    graph = model.graph
    cache = GraphCache(graph)
    nodes = []
    squeezed_indices = {}
    promoted_data = _prepared_index_data(
        model,
        cache,
        prepared_data_ops,
        [node.input[0] for node in _operator_nodes(model, {"Gather"})],
        before_boundary=True,
    )
    selected = [
        node
        for node in _operator_nodes(model, {"Gather"})
        if (shape := cache.get_shape(node.input[0])) is None or len(shape) >= 2
    ]
    layout = _operator_nodes(model, {"Cast", "Identity", "Reshape", "Squeeze", "Unsqueeze"})
    layout_nodes = {id(node) for node in layout}
    casts = {id(node) for node in layout if node.op_type == "Cast"}
    flattened = set()
    index_views = {}
    captured = _subgraph_captures(graph)
    for candidate in _operator_nodes(model, {"Gather"}):
        data_shape = cache.get_shape(candidate.input[0])
        if data_shape is not None and len(data_shape) == 1 and promoted_data[candidate.input[0]]:
            raise ValueError(
                f"{candidate.name!r}: native vector Gather data has a later rank promotion"
            )
    shape_readers = _operator_nodes(model, _RANK_AGNOSTIC_OPS | {"Reshape"})
    shape_reader_ids = {id(node) for node in shape_readers}
    for candidate in selected:
        data_shape = cache.get_shape(candidate.input[0])
        if data_shape is None:
            raise ValueError(f"{candidate.name!r}: Gather requires a static data shape")
        if len(data_shape) in (2, 3) and not promoted_data[candidate.input[0]]:
            raise ValueError(f"{candidate.name!r}: Gather data requires rank preparation")
        original = candidate.input[1]
        index_shape = cache.get_shape(original)
        rank = _known_rank(cache, original)
        producer = cache.get_producer(original)
        fixed = cache.get_init(original)
        if rank is None:
            raise ValueError(f"{candidate.name!r}: Gather indices require a known rank")
        if rank > 1 and original in captured:
            raise ValueError(
                f"{candidate.name!r}: captured Gather indices with rank > 1 are unsupported"
            )
        if rank > 1 and (
            original in cache.graph_input_names
            or fixed is not None
            or (producer is not None and id(producer) in casts)
            or index_shape is None
            or 1 in index_shape
            or len(data_shape) in (2, 3)
        ):
            count = _gather_flatten_count(cache, candidate, index_shape)
            if count is None:
                raise ValueError(
                    f"{candidate.name!r}: Gather indices with rank > 1 are unsupported"
                )
            flattened.add(original)
            index_views[original] = 1
        if rank <= 1 and original not in index_views:
            index_views[original] = (
                rank if _gather_index_needs_view(cache, original, layout_nodes) else None
            )
        if (
            rank == 0
            and fixed is not None
            and original not in cache.graph_input_names
            and _gather_flatten_count(cache, candidate, []) is None
            and not _gather_layout_reinterpreted(
                cache, candidate.output[0], shape_reader_ids, captured
            )
        ):
            raise ValueError(
                f"{candidate.name!r}: fixed scalar Gather indices change the padded output layout"
            )
        if fixed is None and cache.get_dtype(original) not in (
            TensorProto.INT32,
            TensorProto.INT64,
        ):
            if producer is None or id(producer) not in casts:
                raise ValueError(
                    f"{candidate.name!r}: dynamic Gather indices require an integer Cast"
                )
    selected_ids = {id(node) for node in selected}
    for node in graph.node:
        if id(node) not in selected_ids:
            nodes.append(node)
            continue
        indices = cache.get_init(node.input[1])
        if indices is not None and node.input[1] not in cache.graph_input_names:
            if indices.name not in squeezed_indices:
                values = numpy_helper.to_array(indices).reshape(-1).astype(np.int32)
                squeezed_indices[indices.name] = add_initializer(
                    graph,
                    cache.unique_name(indices.name + "_gather_indices"),
                    values,
                    dtype=TensorProto.INT32,
                    with_value_info=False,
                )
            node.input[1] = squeezed_indices[indices.name]
        else:
            original = node.input[1]
            rank = _known_rank(cache, original)
            if rank is not None and (rank <= 1 or original in flattened):
                if original not in squeezed_indices:
                    producer = cache.get_producer(original)
                    if (
                        original in flattened
                        and producer is not None
                        and id(producer) in casts
                        and len(cache.get_consumers(original)) == 1
                        and original not in cache.graph_output_names | captured
                    ):
                        _attribute(producer, "to").i = TensorProto.INT32
                        for value in graph.value_info:
                            if value.name == original:
                                value.type.tensor_type.elem_type = TensorProto.INT32
                        prepared = original
                    else:
                        prepared = cache.unique_name(original + "_gather_int32")
                        nodes.append(
                            helper.make_node(
                                "Cast", [original], [prepared], name=prepared, to=TensorProto.INT32
                            )
                        )
                    if index_views.get(original) is not None:
                        target = add_initializer(
                            graph,
                            cache.unique_name(original + "_gather_shape"),
                            np.asarray([] if index_views[original] == 0 else [-1], np.int64),
                            dtype=TensorProto.INT64,
                            with_value_info=False,
                        )
                        view = cache.unique_name(original + "_gather_view")
                        nodes.append(
                            helper.make_node("Reshape", [prepared, target], [view], name=view)
                        )
                        prepared = view
                    squeezed_indices[original] = prepared
            if original in squeezed_indices:
                node.input[1] = squeezed_indices[original]
                axis = _attribute(node, "axis")
                shape = cache.get_shape(node.input[0])
                if axis is None:
                    node.attribute.append(helper.make_attribute("axis", 0))
                    axis = _attribute(node, "axis")
                if axis.i >= 0 and shape:
                    axis.i += {2: 2, 3: 1}.get(len(shape), 0)
                node.name = cache.unique_name((node.name or node.output[0]) + "_axis_updated")
                nodes.append(node)
                continue
            squeezed = cache.unique_name(original + "_squeezed")
            nodes.append(
                helper.make_node(
                    "Squeeze",
                    [original],
                    [squeezed],
                    name=cache.unique_name((node.name or node.output[0]) + "_squeeze"),
                )
            )
            node.input[1] = squeezed
            squeezed_indices[original] = squeezed
            graph.value_info.append(
                helper.make_tensor_value_info(
                    squeezed, cache.get_dtype(original) or TensorProto.FLOAT, None
                )
            )
        if _attribute(node, "axis") is None:
            node.attribute.append(helper.make_attribute("axis", 0))
        axis = _attribute(node, "axis")
        shape = cache.get_shape(node.input[0])
        if axis is not None:
            if axis.i >= 0 and shape:
                axis.i += {2: 2, 3: 1}.get(len(shape), 0)
            node.name = cache.unique_name((node.name or node.output[0]) + "_axis_updated")
        nodes.append(node)
    _insert_support_nodes(graph, nodes)
    live = {name for node in graph.node for name in node.input}
    live |= cache.graph_input_names | cache.graph_output_names | captured
    for original in squeezed_indices:
        tensor = cache.get_init(original)
        if tensor is not None and original not in live:
            graph.initializer.remove(tensor)
    _add_legacy_initializer_inputs(model)
    return model


def _apply_gather_elements(model, *, prepared_data_ops=(), validate_only=False):
    """Prepare matching index ranks and preserve native axis and runtime-value contracts."""
    graph = model.graph
    cache = GraphCache(graph)
    converted_indices = {}
    replacements = {}
    selected_ids = {id(node) for node in _operator_nodes(model, {"GatherElements"})}
    prepared = _prepared_index_data(
        model,
        cache,
        prepared_data_ops,
        [node.input[0] for node in graph.node if id(node) in selected_ids],
        before_boundary=validate_only,
    )
    internal_indices = [
        node.input[1]
        for node in graph.node
        if id(node) in selected_ids
        and prepared[node.input[0]]
        and node.input[1] not in cache.graph_input_names | cache.init_names
        and _known_rank(cache, node.input[1]) in (None, 2, 3)
    ]
    index_layouts = (
        _prepared_index_data(
            model, cache, prepared_data_ops, internal_indices, before_boundary=validate_only
        )
        if internal_indices
        else {}
    )
    for name in internal_indices:
        if not index_layouts[name]:
            raise ValueError(f"{name!r}: GatherElements indices require rank preparation")
    for node in graph.node:
        if (
            id(node) in selected_ids
            and cache.get_shape(node.input[0]) is not None
            and _known_rank(cache, node.input[0]) in (2, 3)
            and not prepared[node.input[0]]
        ):
            raise ValueError(f"{node.name!r}: GatherElements data requires rank preparation")
    if validate_only:
        return model
    for index, node in enumerate(graph.node):
        if id(node) not in selected_ids:
            continue
        original = node.input[1]
        tensor = cache.get_init(original)
        rank = len(tensor.dims) if tensor is not None else _known_rank(cache, original)
        data_rank = _known_rank(cache, node.input[0])
        if rank in (2, 3) and data_rank != 4 and cache.get_shape(node.input[0]) is None:
            # Boundary promotion can leave symbolic data native; retain matching native indices.
            continue
        added = []
        if original in converted_indices:
            prepared, rank = converted_indices[original]
        elif tensor is not None and original not in cache.graph_input_names:
            array = numpy_helper.to_array(tensor).astype(np.int32)
            if rank in (2, 3):
                array = array.reshape([*([1] * (4 - rank)), *array.shape])
            elif rank == 0:
                array = array.reshape(1)
            prepared = add_initializer(
                graph,
                cache.unique_name(tensor.name + "_gatherelements_indices"),
                array,
                dtype=TensorProto.INT32,
                with_value_info=False,
            )
        else:
            prepared = cache.unique_name(original + "_cast_to_int32")
            added.append(
                helper.make_node(
                    "Cast",
                    [original],
                    [prepared],
                    name=cache.unique_name(original + "_cast_to_int32_node"),
                    to=TensorProto.INT32,
                )
            )
            # Other intermediate producers are promoted by the subsequent rank stages.
            if rank in (2, 3) and original in cache.graph_input_names:
                axes = add_initializer(
                    graph,
                    cache.unique_name(original + "_gatherelements_axes"),
                    np.arange(4 - rank, dtype=np.int64),
                    dtype=TensorProto.INT64,
                    with_value_info=False,
                )
                promoted = cache.unique_name(original + "_gatherelements_indices")
                added.append(
                    helper.make_node(
                        "Unsqueeze",
                        [prepared, axes],
                        [promoted],
                        name=cache.unique_name(original + "_indices_unsqueeze"),
                    )
                )
                prepared = promoted
        converted_indices[original] = (prepared, rank)
        node.input[1] = prepared
        if rank in (2, 3):
            axis = _attribute(node, "axis")
            if axis is None:
                node.attribute.append(helper.make_attribute("axis", 0))
                axis = _attribute(node, "axis")
            if axis.i >= 0:
                axis.i += 4 - rank
            node.name = cache.unique_name((node.name or node.output[0]) + "_axis_updated")
        restored = []
        output = node.output[0]
        if (
            data_rank == 4
            and output in cache.graph_output_names
            and rank in (2, 3)
            and _known_rank(cache, output) == rank
        ):
            # The earlier output stage cannot plan adapters from concrete shapes for symbolic outputs.
            internal = cache.unique_name(output + "_gatherelements_4d")
            node.output[0] = internal
            axes = add_initializer(
                graph,
                cache.unique_name(output + "_gatherelements_squeeze_axes"),
                np.arange(4 - rank, dtype=np.int64),
                dtype=TensorProto.INT64,
                with_value_info=False,
            )
            restored.append(
                helper.make_node(
                    "Squeeze",
                    [internal, axes],
                    [output],
                    name=cache.unique_name(output + "_restore_rank"),
                )
            )
        replacements[index] = ([node], [*added, node, *restored])
    batch_replace_nodes(graph, replacements)
    live = {name for node in graph.node for name in node.input}
    live.update(cache.graph_input_names | cache.graph_output_names | _subgraph_captures(graph))
    for original in converted_indices.keys() - live:
        tensor = cache.get_init(original)
        if tensor is not None:
            graph.initializer.remove(tensor)
    _add_legacy_initializer_inputs(model)
    return model


def _apply_standalone_reduce_operations(model, *, prepared_data_ops=(), validate_only=False):
    """Plan native reduction axes and preserve their output layouts during rank promotion."""
    graph = model.graph
    selected = _operator_nodes(model, {"ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin"})
    if not selected:
        return model
    cache = GraphCache(graph)
    nodes = []
    count = 0
    plans = {}
    prepared = _prepared_index_data(
        model,
        cache,
        prepared_data_ops,
        [node.input[0] for node in selected],
        before_boundary=validate_only,
    )
    shapes = {id(node): _shape_before_boundary_promotion(cache, node.input[0]) for node in selected}
    constant_controls = (
        {name for node in _operator_nodes(model, {"Constant"}) for name in node.output}
        if validate_only and model.ir_version > 3 and "ConstantsToInitializers" in prepared_data_ops
        else set()
    )
    for node in selected:
        tensor = cache.get_init(node.input[1]) if len(node.input) > 1 else None
        if (
            len(node.input) > 1
            and node.input[1] in constant_controls
            and node.input[1] not in cache.graph_input_names
        ):
            continue
        if (
            len(node.input) > 1
            and (tensor is None or tensor.name in cache.graph_input_names)
            and (
                (validate_only and prepared[node.input[0]])
                or shapes[id(node)]
                or (
                    validate_only
                    and (keepdims := _attribute(node, "keepdims")) is not None
                    and keepdims.i == 0
                )
            )
        ):
            error = (
                _BoundaryPreflightError if node.input[1] in cache.graph_input_names else ValueError
            )
            raise error(f"Dynamic reduction axes are unsupported for {node.name!r}")
    for node in selected:
        shape = shapes[id(node)]
        tensor = cache.get_init(node.input[1]) if len(node.input) > 1 else None
        if not shape:
            if validate_only:
                continue
            keepdims = _attribute(node, "keepdims")
            if keepdims is not None and keepdims.i == 0:
                raise ValueError(f"Missing static reduction input shape for {node.name!r}")
            continue  # Already rank-preserving, including generated 4D reduction branches.
        if len(shape) in (2, 3) and not prepared[node.input[0]]:
            raise ValueError(f"{node.name!r}: reduction data requires rank preparation")
        if validate_only:
            continue
        attribute = _attribute(node, "axes")
        raw_axes = (
            numpy_helper.to_array(tensor).reshape(-1).tolist()
            if tensor is not None
            else list(attribute.ints)
            if attribute is not None
            else list(range(len(shape)))
        )
        reduced = {value % len(shape) for value in raw_axes or range(len(shape))}
        output_shape = cache.get_shape(node.output[0])
        if output_shape is None and node.output[0].endswith("_squeeze_input"):
            output_shape = cache.get_shape(node.output[0].removesuffix("_squeeze_input"))
        if output_shape is None:
            output_shape = [dim for index, dim in enumerate(shape) if index not in reduced]
        plans[id(node)] = (shape, raw_axes, output_shape)
    if validate_only:
        return model
    for node in graph.node:
        nodes.append(node)
        if id(node) not in plans:
            continue
        shape, raw_axes, output_shape = plans[id(node)]
        keepdims = _attribute(node, "keepdims")
        reshape = keepdims is not None and keepdims.i == 0
        if keepdims is not None:
            keepdims.i = 1
        delta = {2: 2, 3: 1}.get(len(shape), 0)
        values = [value + delta if value >= 0 else value for value in raw_axes]
        if len(node.input) > 1:
            node.input[1] = add_initializer(
                graph,
                cache.unique_name((node.name or node.output[0]) + "_axes"),
                np.asarray(values, dtype=np.int64),
                dtype=TensorProto.INT64,
                with_value_info=False,
            )
        else:
            attribute = _attribute(node, "axes")
            if attribute is not None:
                del attribute.ints[:]
                attribute.ints.extend(values)
        if reshape:
            target = pad4d(output_shape)
            name = add_initializer(
                graph,
                cache.unique_name(f"reshape_shape_{count}"),
                np.asarray(target, dtype=np.int64),
                dtype=TensorProto.INT64,
                with_value_info=False,
            )
            original = node.output[0]
            node.output[0] = cache.unique_name(original + "_temp")
            nodes.append(
                helper.make_node(
                    "Reshape",
                    [node.output[0], name],
                    [original],
                    name=cache.unique_name((node.name or original) + "_reshape"),
                )
            )
            count += 1
    _insert_support_nodes(graph, nodes)
    _add_legacy_initializer_inputs(model)
    return model


def _apply_group_query_attention(model, *, prepared_data_ops=()):
    """Restore native attention ranks across internal tensors while retaining public boundaries."""
    graph = model.graph
    cache = GraphCache(graph)
    nodes = []
    protected_outputs = cache.graph_output_names | _subgraph_captures(graph)
    boundaries = cache.graph_input_names | cache.init_names | cache.graph_output_names
    squeezed_inputs = {}
    axes_names = {}

    def axes_for(name):
        """Reuse a private axes control for adapters of the same attention tensor."""
        if name not in axes_names:
            axes_names[name] = add_initializer(
                graph,
                cache.unique_name(name + "_axis"),
                np.asarray([0], dtype=np.int64),
                dtype=TensorProto.INT64,
                with_value_info=False,
            )
        return axes_names[name]

    selected_ids = {
        id(node) for node in _operator_nodes(model, {"GroupQueryAttention"}, ("com.microsoft",))
    }
    data_names = [
        name
        for node in graph.node
        if id(node) in selected_ids
        for name in node.input[:3]
        if name and name not in boundaries
    ]
    prepared = (
        _prepared_index_data(
            model, cache, set(prepared_data_ops) | {"GroupQueryAttention"}, data_names
        )
        if data_names
        else {}
    )
    for node in graph.node:
        if id(node) not in selected_ids:
            nodes.append(node)
            continue
        for index, name in enumerate(node.input[:3]):
            if name and name not in boundaries:
                shape = cache.get_shape(name)
                if not prepared[name] or (shape is not None and len(shape) == 4 and shape[0] != 1):
                    continue
                if name in squeezed_inputs:
                    node.input[index] = squeezed_inputs[name]
                    continue
                axes = axes_for(name)
                squeezed = cache.unique_name(name + "_squeezed")
                squeezed_inputs[name] = squeezed
                nodes.append(
                    helper.make_node(
                        "Squeeze",
                        [name, axes],
                        [squeezed],
                        name=cache.unique_name(node.name + "_" + name + "_squeeze"),
                    )
                )
                node.input[index] = squeezed
        nodes.append(node)
        for index, name in enumerate(node.output[:1]):
            if name and name not in protected_outputs:
                axes = axes_for(name)
                node.output[index] = cache.unique_name(name + "_squeezed")
                nodes.append(
                    helper.make_node(
                        "Unsqueeze",
                        [node.output[index], axes],
                        [name],
                        name=cache.unique_name(node.name + "_" + name + "_unsqueeze"),
                    )
                )
    _insert_support_nodes(graph, nodes)
    _add_legacy_initializer_inputs(model)
    return model
