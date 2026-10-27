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

"""Ordered rank and axis passes sharing the preceding native shape snapshot.

Do not infer shapes between stages: original axes precede upstream rank promotion.
"""

import math
from functools import partial

import numpy as np
from onnx import TensorProto, defs, helper, numpy_helper

from ..utils.graph_utils import (
    GraphCache,
    _attribute,
    _shape_before_boundary_promotion,
    add_initializer,
    pad4d,
)
from .dla_constants_to_initializers import _add_legacy_initializer_inputs
from .dla_graph_cleanup import _RANK_AGNOSTIC_OPS, _subgraph_captures
from .dla_indexing import _insert_support_nodes, _known_rank, _operator_nodes
from .dla_model_boundaries import _has_native_symbolic_data

__all__ = []


def _runtime_input_names(model):
    """Exclude fixed initializer declarations required by legacy ONNX IR."""
    names = {value.name for value in model.graph.input}
    if model.ir_version <= 3:
        names.difference_update(tensor.name for tensor in model.graph.initializer)
    return names


def _promoted_input_shape(cache, name, local_functions, native_sources):
    """Distinguish native symbolic paths from inputs prepared for four-dimensional data."""
    shape = _shape_before_boundary_promotion(cache, name)
    rank = len(shape) if shape is not None else _known_rank(cache, name)
    native = (
        rank is not None
        and rank < 4
        and _has_native_symbolic_data(cache, name, local_functions, native_sources)
    )
    return shape, native


def _apply_non4d_slice_axis(model):
    """Remap fixed Slice axes before their data rank is promoted."""
    cache = GraphCache(model.graph)
    local_functions = {(fn.domain, fn.name, fn.overload) for fn in model.functions}
    native_sources = {}
    runtime_inputs = _runtime_input_names(model)
    plans = []
    for node in _operator_nodes(model, {"Slice"}):
        shape, native = _promoted_input_shape(cache, node.input[0], local_functions, native_sources)
        if native:
            continue
        if shape is None or len(shape) not in (2, 3):
            continue
        if len(node.input) < 4 or not node.input[3]:
            starts_shape = cache.get_shape(node.input[1])
            if starts_shape is None:
                raise ValueError(
                    f"{node.name!r}: implicit Slice axes require a known starts length"
                )
            values = np.arange(math.prod(starts_shape), dtype=np.int64) + 4 - len(shape)
            plans.append((node, node.input[1], values))
            continue
        axes = cache.get_init(node.input[3])
        if axes is None or axes.name in runtime_inputs:
            raise ValueError(f"{node.name!r}: Slice axes must be fixed")
        values = numpy_helper.to_array(axes).reshape(-1)
        if values.size and np.any(values >= 0):
            plans.append((node, axes.name, np.where(values >= 0, values + 4 - len(shape), values)))
    views = {}
    for node, name, values in plans:
        key = (name, tuple(values))
        if key not in views:
            views[key] = add_initializer(
                model.graph,
                cache.unique_name(name + "_axes_only"),
                np.asarray(values, dtype=np.int64),
                dtype=TensorProto.INT64,
                with_value_info=False,
            )
        if len(node.input) < 4:
            node.input.append(views[key])
        else:
            node.input[3] = views[key]
    _add_legacy_initializer_inputs(model)
    return model


def _apply_non4d_reshape(model):
    """Prepare fixed Reshape targets for the deferred four-dimensional layout."""
    cache = GraphCache(model.graph)
    local_functions = {(fn.domain, fn.name, fn.overload) for fn in model.functions}
    native_sources = {}
    runtime_inputs = _runtime_input_names(model)
    reshapes = _operator_nodes(model, {"Reshape"})
    gathers = {id(node) for node in _operator_nodes(model, {"Gather"})}
    squeezes = {id(node) for node in _operator_nodes(model, {"Squeeze"})}
    replacements = {}
    users = {}
    captured = _subgraph_captures(model.graph)
    for node in reshapes:
        if any(
            index == 1 and id(user) in gathers
            for user, index in cache.get_consumers(node.output[0])
        ):
            continue
        tensor = cache.get_init(node.input[1])
        target_shape = cache.get_shape(node.input[1])
        native_input = _shape_before_boundary_promotion(cache, node.input[0])
        if target_shape == [4] and native_input is not None and len(native_input) == 4:
            continue
        rank = len(native_input) if native_input is not None else _known_rank(cache, node.input[0])
        if rank is None:
            raise ValueError(f"{node.name!r}: Reshape input rank must be known")
        if rank < 4 and _has_native_symbolic_data(
            cache, node.input[0], local_functions, native_sources
        ):
            continue
        if tensor is None or tensor.name in runtime_inputs:
            raise ValueError(f"{node.name!r}: Reshape target must be fixed")
        values = numpy_helper.to_array(tensor).reshape(-1).tolist()
        if len(values) < 4:
            native_rank = len(values)
            values = pad4d(values)
            output_shape = cache.get_shape(node.output[0])
            if output_shape is None:
                aliases = [
                    cache.get_shape(user.output[0])
                    for user, index in cache.get_consumers(node.output[0])
                    if index == 0
                    and id(user) in squeezes
                    and user.output[0] in cache.graph_output_names
                ]
                aliases = [
                    shape for shape in aliases if shape is not None and len(shape) == native_rank
                ]
                if aliases and all(shape == aliases[0] for shape in aliases):
                    output_shape = aliases[0]
            if output_shape:
                output_shape = pad4d(output_shape)
                values = [
                    output_shape[i] if value == 0 else value for i, value in enumerate(values)
                ]
            if 0 in values:
                raise ValueError(f"{node.name!r}: cannot resolve 0 in Reshape target")
        elif len(values) == 5:
            if 0 in values:
                output_shape = cache.get_shape(node.output[0])
                if output_shape is None or len(output_shape) != 5:
                    raise ValueError(f"{node.name!r}: cannot resolve 0 in 5D Reshape target")
                values = [
                    output_shape[i] if value == 0 else value for i, value in enumerate(values)
                ]
            if 1 not in values:
                raise ValueError(f"{node.name!r}: 5D Reshape shape has no unary dimension")
            values.remove(1)
        elif len(values) == 4 and 0 in values:
            output_shape = cache.get_shape(node.output[0])
            if output_shape is None or len(output_shape) != 4:
                raise ValueError(f"{node.name!r}: cannot resolve 0 in Reshape target")
            values = [output_shape[i] if value == 0 else value for i, value in enumerate(values)]
            if 0 in values:
                raise ValueError(f"{node.name!r}: cannot resolve 0 in Reshape target")
        else:
            continue
        replacement = numpy_helper.from_array(np.asarray(values, dtype=np.int64), tensor.name)
        if tensor.name in replacements and replacements[tensor.name] != replacement:
            raise ValueError(f"Shared Reshape target {tensor.name!r} requires different shapes")
        replacements[tensor.name] = replacement
        users.setdefault(tensor.name, []).append(node)
    for name, replacement in replacements.items():
        converted = {id(user) for user in users[name]}
        private = name in cache.graph_output_names | cache.graph_input_names | captured or any(
            index != 1 or id(user) not in converted for user, index in cache.get_consumers(name)
        )
        if private:
            target = add_initializer(
                model.graph,
                cache.unique_name(name + "_reshape_shape"),
                numpy_helper.to_array(replacement),
                dtype=TensorProto.INT64,
                with_value_info=False,
            )
            for user in users[name]:
                user.input[1] = target
        else:
            tensor = cache.get_init(name)
            assert tensor is not None
            tensor.CopyFrom(replacement)
    _add_legacy_initializer_inputs(model)
    return model


def _apply_non4d_initializers(model):
    """Promote fixed weight layouts while preserving runtime input defaults."""
    expand, skip, front, avoid = set(), set(), set(), set()
    supported = _operator_nodes(model, {node.op_type for node in model.graph.node})
    if any(op.domain == "com.microsoft" and op.version == 1 for op in model.opset_import):
        supported += _operator_nodes(
            model, {"QuantizeLinear", "DequantizeLinear", "GroupQueryAttention"}, ("com.microsoft",)
        )
    selected = {id(node) for node in supported}
    protected = _subgraph_captures(model.graph) | {value.name for value in model.graph.output}
    protected.update(
        name for node in model.graph.node if id(node) not in selected for name in node.input
    )
    used = (
        {name for node in supported for name in node.input}
        if any(tensor.name in protected for tensor in model.graph.initializer)
        else set()
    )
    for node in supported:
        if node.op_type in {"Div", "Sub", "Mul", "Add", "LSTM"}:
            expand.update(node.input)
        if node.op_type == "MatMul":
            skip.update(node.input)
        if node.op_type in {"Gemm", "Where", "Gather", "Split"}:
            front.update(node.input)
        if node.op_type in {"Unsqueeze", "GroupQueryAttention"}:
            avoid.update(node.input)
    updates = []
    avoid.update(_runtime_input_names(model))
    expand_or_front = expand | front
    for tensor in model.graph.initializer:
        if tensor.name in avoid:
            continue
        rank = len(tensor.dims)
        promote = (
            (rank == 1 and tensor.name in expand)
            or (rank == 2 and tensor.name in expand_or_front)
            or rank == 3
        )
        if tensor.name in protected:
            if tensor.name in used and promote:
                raise ValueError(
                    f"Initializer {tensor.name!r} has incompatible public, opaque or captured usage"
                )
            continue
        if promote:
            shape = pad4d(list(tensor.dims))
            updates.append((tensor, shape))
        elif rank == 2 and tensor.name not in skip:
            raise ValueError(f"Initializer {tensor.name!r} has unsupported rank 2 usage")
    value_info = {value.name for value in model.graph.value_info}
    native_inputs = {
        node.input[0]
        for node in model.graph.node
        if node.input and node.op_type not in _RANK_AGNOSTIC_OPS
    }
    legacy_inputs = (
        {value.name: value for value in model.graph.input} if model.ir_version <= 3 else {}
    )
    for tensor, shape in updates:
        if tensor.name in native_inputs and tensor.name not in value_info:
            # Later axis stages read native ranks from the same deferred metadata snapshot.
            model.graph.value_info.append(
                helper.make_tensor_value_info(tensor.name, tensor.data_type, list(tensor.dims))
            )
        del tensor.dims[:]
        tensor.dims.extend(shape)
        if tensor.name in legacy_inputs:
            legacy_inputs[tensor.name].type.tensor_type.shape.CopyFrom(
                helper.make_tensor_value_info(
                    tensor.name, tensor.data_type, shape
                ).type.tensor_type.shape
            )
    return model


def _pad_shape_inputs(model, op_type, suffix):
    """Preserve native rank-four layouts and plan fixed control padding otherwise."""
    cache = GraphCache(model.graph)
    local_functions = {(fn.domain, fn.name, fn.overload) for fn in model.functions}
    native_sources = {}
    runtime_inputs = _runtime_input_names(model)
    plans = []
    for node in _operator_nodes(model, {op_type}):
        tensor = cache.get_init(node.input[1])
        native = _shape_before_boundary_promotion(cache, node.input[0])
        rank = len(native) if native is not None else _known_rank(cache, node.input[0])
        if rank == 4 and cache.get_shape(node.input[1]) == [4]:
            continue
        if rank is None:
            raise ValueError(f"{node.name!r}: {op_type} input rank must be known")
        if rank < 4 and _has_native_symbolic_data(
            cache, node.input[0], local_functions, native_sources
        ):
            continue
        if tensor is None or tensor.name in runtime_inputs:
            raise ValueError(f"{node.name!r}: {op_type} target must be fixed")
        values = numpy_helper.to_array(tensor).reshape(-1).tolist()
        if len(values) < 4:
            plans.append((node, tensor.name, pad4d(values)))
        elif len(values) > 4:
            raise ValueError(f"{node.name!r}: {op_type} target has rank {len(values)}")
    views = {}
    for node, name, values in plans:
        if name not in views:
            views[name] = add_initializer(
                model.graph,
                cache.unique_name(name + suffix),
                np.asarray(values, dtype=np.int64),
                dtype=TensorProto.INT64,
                with_value_info=False,
            )
        node.input[1] = views[name]
    _add_legacy_initializer_inputs(model)
    return model


_apply_non4d_expand = partial(_pad_shape_inputs, op_type="Expand", suffix="_expand_shape")
_apply_non4d_tile = partial(_pad_shape_inputs, op_type="Tile", suffix="_tile_shape")


def _collapsed_reshape_axes(cache, name, shape, reshapes):
    """Identify singleton axes consistent with an actually collapsed native Reshape."""
    producer = cache.get_producer(name)
    target = (
        cache.get_init_array(producer.input[1])
        if producer is not None and id(producer) in reshapes
        else None
    )
    if target is None or target.shape != (4,):
        return []
    values = target.tolist()
    return [
        axis
        for axis, dim in enumerate(shape)
        if dim == 1
        and all(
            value in (-1, native) for value, native in zip(values, shape[:axis] + shape[axis + 1 :])
        )
    ]


def _apply_non4d_transpose(model):
    """Remap transpose permutations around added or removed unary axes."""
    cache = GraphCache(model.graph)
    local_functions = {(fn.domain, fn.name, fn.overload) for fn in model.functions}
    native_sources = {}
    plans = []
    reshapes = {id(node) for node in _operator_nodes(model, {"Reshape"})}
    for node in _operator_nodes(model, {"Transpose"}):
        perm = _attribute(node, "perm")
        shape, native = _promoted_input_shape(cache, node.input[0], local_functions, native_sources)
        if native:
            continue
        if perm is None and shape is None:
            raise ValueError(f"{node.name!r}: default Transpose requires a known input rank")
        values = list(perm.ints) if perm is not None else list(reversed(range(len(shape))))
        if len(values) in (2, 3):
            delta = 4 - len(values)
            values = list(range(delta)) + [v + delta for v in values]
        elif len(values) == 5:
            if shape is None or len(shape) != 5 or 1 not in shape:
                raise ValueError(f"{node.name!r}: 5D Transpose requires a known unary dimension")
            removed_axes = _collapsed_reshape_axes(cache, node.input[0], shape, reshapes)
            if not removed_axes:
                continue
            if len(removed_axes) != 1:
                raise ValueError(f"{node.name!r}: ambiguous collapsed 5D Transpose layout")
            removed = removed_axes[0]
            values = [v - int(v > removed) for v in values if v != removed]
        else:
            continue
        plans.append((node, perm, values))
    for node, perm, values in plans:
        if perm is None:
            node.attribute.append(helper.make_attribute("perm", values))
        else:
            del perm.ints[:]
            perm.ints.extend(values)
    return model


def _specialized_collapsed_axis(cache, node, shape, value, reshapes):
    """Map a specialized operator axis only when its native singleton was actually collapsed."""
    removed = _collapsed_reshape_axes(cache, node.input[0], shape, reshapes)
    if not removed:
        return value
    if len(removed) != 1 or value % 5 in removed:
        raise ValueError(
            f"{node.name!r}: cannot map axis {value} through an ambiguous collapsed Reshape"
        )
    native = value % 5
    return native - int(native > removed[0]) - (4 if value < 0 else 0)


def _apply_non4d_lpnorm(model):
    """Preserve normalization axes across leading-dimension promotion."""
    cache = GraphCache(model.graph)
    local_functions = {(fn.domain, fn.name, fn.overload) for fn in model.functions}
    native_sources = {}
    reshapes = {id(node) for node in _operator_nodes(model, {"Reshape"})}
    plans = []
    for node in _operator_nodes(model, {"LpNormalization"}):
        axis = _attribute(node, "axis")
        shape, native = _promoted_input_shape(cache, node.input[0], local_functions, native_sources)
        if native:
            continue
        value = axis.i if axis is not None else -1
        if shape and len(shape) == 5:
            value = _specialized_collapsed_axis(cache, node, shape, value, reshapes)
        elif value >= 0 and shape and len(shape) in (2, 3):
            value = -1 if value == len(shape) - 1 else value + 4 - len(shape)
        plans.append((node, axis, value))
    for node, axis, value in plans:
        if axis is None:
            node.attribute.append(helper.make_attribute("axis", value))
        else:
            axis.i = value
        node.name += "_axis_updated"
    return model


def _apply_flatten(model):
    """Replace Flatten with its four-dimensional equivalent shape."""
    graph = model.graph
    cache = GraphCache(graph)
    plans = []
    for node in _operator_nodes(model, {"Flatten"}):
        shape = _shape_before_boundary_promotion(cache, node.input[0])
        axis = _attribute(node, "axis")
        axis = axis.i if axis is not None else 1
        if shape is None or any(d <= 0 for d in shape):
            raise ValueError(f"{node.name!r}: Flatten requires a known input shape")
        if not -len(shape) <= axis <= len(shape):
            raise ValueError(f"{node.name!r}: invalid Flatten axis {axis}")
        if axis < 0:
            axis += len(shape)
        plans.append((node, math.prod(shape[:axis])))
    targets = {}
    for node, batch in plans:
        if batch not in targets:
            hint = "flatten_reshape_shape" if batch == 1 else f"{node.name}_reshape_shape"
            targets[batch] = add_initializer(
                graph,
                cache.unique_name(hint),
                np.asarray([1, 1, batch, -1], dtype=np.int64),
                dtype=TensorProto.INT64,
                with_value_info=False,
            )
        node.CopyFrom(
            helper.make_node(
                "Reshape",
                [node.input[0], targets[batch]],
                list(node.output),
                name=cache.unique_name((node.name or node.output[0]) + "_reshape"),
            )
        )
    _add_legacy_initializer_inputs(model)
    return model


def _apply_argmax(model):
    """Plan native ArgMax layouts before preparing private output restoration views."""
    cache = GraphCache(model.graph)
    local_functions = {(fn.domain, fn.name, fn.overload) for fn in model.functions}
    native_sources = {}
    plans = {}
    reshapes = {id(node) for node in _operator_nodes(model, {"Reshape"})}
    for node in _operator_nodes(model, {"ArgMax"}):
        shape, native = _promoted_input_shape(cache, node.input[0], local_functions, native_sources)
        if native:
            continue
        if not shape or any(dim <= 0 for dim in shape):
            raise ValueError(f"{node.name!r}: ArgMax requires a static input shape")
        axis = _attribute(node, "axis")
        value = axis.i if axis is not None else 0
        if not -len(shape) <= value < len(shape):
            raise ValueError(f"{node.name!r}: invalid ArgMax axis {value}")
        output_shape = [dim for index, dim in enumerate(shape) if index != value % len(shape)]
        if len(shape) == 5:
            value = _specialized_collapsed_axis(cache, node, shape, value, reshapes)
        elif value >= 0:
            value += {2: 2, 3: 1}.get(len(shape), 0)
        plans[id(node)] = (value, output_shape)
    nodes = []
    for node in model.graph.node:
        nodes.append(node)
        if id(node) not in plans:
            continue
        value, output_shape = plans[id(node)]
        axis = _attribute(node, "axis")
        if axis is None:
            node.attribute.append(helper.make_attribute("axis", value))
        else:
            axis.i = value
        keepdims = _attribute(node, "keepdims")
        restore_shape = keepdims is not None and keepdims.i == 0
        if keepdims is not None:
            keepdims.i = 1
        original = node.output[0]
        node.name = cache.unique_name((node.name or original) + "_axis_updated")
        if restore_shape:
            target = add_initializer(
                model.graph,
                cache.unique_name(node.name + "_reshape_shape"),
                np.asarray(pad4d(output_shape), dtype=np.int64),
                dtype=TensorProto.INT64,
                with_value_info=False,
            )
            node.output[0] = cache.unique_name(original + "_keepdims")
            nodes.append(
                helper.make_node(
                    "Reshape",
                    [node.output[0], target],
                    [original],
                    name=cache.unique_name(node.name + "_reshape"),
                )
            )
    _insert_support_nodes(model.graph, nodes)
    _add_legacy_initializer_inputs(model)
    return model


def _apply_update_axis(model):
    """Remap supported operation axes from their native input ranks."""
    quantizers = {"QuantizeLinear", "DequantizeLinear"}
    update = {"LogSoftmax", "Softmax", "TopK", "Concat", "Split"} | quantizers
    skip = {"Gather", "GatherElements", "LpNormalization", "ArgMax", "Slice", "Flatten"}
    skip |= {"ReduceMax", "ReduceMin", "ReduceSum", "ReduceMean"}
    cache = GraphCache(model.graph)
    local_functions = {(fn.domain, fn.name, fn.overload) for fn in model.functions}
    native_sources = {}
    plans = []
    reshapes = {id(node) for node in _operator_nodes(model, {"Reshape"})}
    selected = _operator_nodes(model, {node.op_type for node in model.graph.node})
    if any(op.domain == "com.microsoft" and op.version == 1 for op in model.opset_import):
        selected += _operator_nodes(model, quantizers, ("com.microsoft",))
    versions = {op.domain: op.version for op in model.opset_import}
    for node in selected:
        axis = _attribute(node, "axis")
        shape, native = (
            _promoted_input_shape(cache, node.input[0], local_functions, native_sources)
            if node.input
            else (None, False)
        )
        if native:
            continue
        if (
            axis is None
            and node.op_type in update
            and node.domain in {"", "ai.onnx"}
            and shape
            and (
                len(shape) == 5
                or (
                    len(shape) in (2, 3)
                    and (node.op_type not in quantizers or cache.get_shape(node.input[1]) != [])
                )
            )
        ):
            schema = defs.get_schema(
                node.op_type, versions.get(node.domain, versions.get("", 1)), node.domain
            )
            default = schema.attributes.get("axis")
            if default is not None and default.default_value.type:
                value = default.default_value.i
            else:
                continue
        elif axis is None or node.op_type in skip or (axis.i < 0 and node.op_type not in update):
            continue
        else:
            value = axis.i
        if node.op_type not in update:
            if shape is not None and len(shape) not in (2, 3, 5):
                continue
            raise ValueError(f"{node.op_type} is unsupported for DLA axis transformation")
        if "axis_updated" in node.name:
            continue
        rank = len(shape) if shape else 0
        original = value
        if shape and rank == 5:
            removed_axes = _collapsed_reshape_axes(cache, node.input[0], shape, reshapes)
            if not -5 <= value < 5 or not removed_axes or value % 5 in removed_axes:
                raise ValueError(
                    f"{node.name!r}: cannot map 5D axis {value} without an unambiguous collapsed Reshape"
                )
            native_axis = value % 5
            value = native_axis - int(native_axis > removed_axes[0])
            if original < 0:
                value -= 4
        elif value >= 0:
            value += {2: 2, 3: 1}.get(rank, 0)
        if value != original:
            plans.append((node, axis, value))
    for node, axis, value in plans:
        if axis is None:
            node.attribute.append(helper.make_attribute("axis", value))
        else:
            axis.i = value
        node.name += "_axis_updated"
    return model
