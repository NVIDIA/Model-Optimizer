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

"""Individual rank and axis passes in the logged DLA transform sequence.

These passes share the shape snapshot produced by the preceding shape-inference
stage. Running inference between passes would change how original axes are
mapped after upstream tensors have been promoted to four dimensions.
"""

import math

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from ..utils.graph_utils import (
    GraphCache,
    _attribute,
    _shape_before_boundary_promotion,
    add_initializer,
    pad4d,
)

__all__ = []


def _apply_non4d_slice_axis(model):
    """Remap fixed Slice axes before their data rank is promoted."""
    cache = GraphCache(model.graph)
    for node in model.graph.node:
        if node.op_type != "Slice" or len(node.input) < 4:
            continue
        shape = cache.get_shape(node.input[0])
        axes = cache.get_init(node.input[3])
        if shape is None or len(shape) not in (2, 3) or axes is None:
            continue
        values = numpy_helper.to_array(axes).reshape(-1)
        if values.size and np.any(values >= 0):
            node.input[3] = add_initializer(
                model.graph,
                axes.name + "_axes_only",
                np.asarray(np.where(values >= 0, values + 4 - len(shape), values), dtype=np.int64),
                dtype=TensorProto.INT64,
                with_value_info=False,
            )
    return model


def _apply_non4d_reshape(model):
    """Prepare fixed Reshape targets for the deferred four-dimensional layout."""
    cache = GraphCache(model.graph)
    replacements = {}
    for node in model.graph.node:
        if node.op_type != "Reshape" or len(node.input) < 2:
            continue
        tensor = cache.get_init(node.input[1])
        if tensor is None:
            continue
        values = numpy_helper.to_array(tensor).reshape(-1).tolist()
        if len(values) < 4:
            values = pad4d(values)
            output_shape = cache.get_shape(node.output[0])
            if output_shape:
                output_shape = pad4d(output_shape)
                values = [
                    output_shape[i] if value == 0 else value for i, value in enumerate(values)
                ]
        elif len(values) == 5:
            if 1 not in values:
                raise ValueError(f"{node.name!r}: 5D Reshape shape has no unary dimension")
            values.remove(1)
        else:
            continue
        replacement = numpy_helper.from_array(np.asarray(values, dtype=np.int64), tensor.name)
        if tensor.name in replacements and replacements[tensor.name] != replacement:
            raise ValueError(f"Shared Reshape target {tensor.name!r} requires different shapes")
        replacements[tensor.name] = replacement
    for tensor in model.graph.initializer:
        if tensor.name in replacements:
            tensor.CopyFrom(replacements[tensor.name])
    return model


def _apply_non4d_initializers(model):
    """Promote fixed weight layouts while preserving runtime input defaults."""
    expand, skip, front, avoid = set(), set(), set(), set()
    for node in model.graph.node:
        if node.op_type in {"Div", "Sub", "Mul", "Add", "LSTM"}:
            expand.update(node.input)
        if node.op_type == "MatMul":
            skip.update(node.input)
        if node.op_type in {"Gemm", "Where", "Gather", "Split"}:
            front.update(node.input)
        if node.op_type in {"Unsqueeze", "GroupQueryAttention"}:
            avoid.update(node.input)
    updates = []
    avoid.update(value.name for value in model.graph.input)
    expand_or_front = expand | front
    for tensor in model.graph.initializer:
        if tensor.name in avoid:
            continue
        rank = len(tensor.dims)
        if (
            (rank == 1 and tensor.name in expand)
            or (rank == 2 and tensor.name in expand_or_front)
            or rank == 3
        ):
            shape = pad4d(list(tensor.dims))
            updates.append((tensor, shape))
        elif rank == 2 and tensor.name not in skip:
            raise ValueError(f"Initializer {tensor.name!r} has unsupported rank 2 usage")
    for tensor, shape in updates:
        del tensor.dims[:]
        tensor.dims.extend(shape)
    return model


def _apply_non4d_expand(model):
    """Prepare fixed Expand targets for four-dimensional data."""
    return _pad_shape_inputs(model, "Expand", "_expand_shape")


def _apply_non4d_tile(model):
    """Prepare fixed Tile repeats for four-dimensional data."""
    return _pad_shape_inputs(model, "Tile", "_tile_shape")


def _pad_shape_inputs(model, op_type, suffix):
    """Pad fixed structural controls for the requested rank stage."""
    cache = GraphCache(model.graph)
    for node in model.graph.node:
        if node.op_type != op_type or len(node.input) < 2:
            continue
        tensor = cache.get_init(node.input[1])
        if tensor is None:
            continue
        values = numpy_helper.to_array(tensor).reshape(-1).tolist()
        if len(values) < 4:
            node.input[1] = add_initializer(
                model.graph,
                tensor.name + suffix,
                np.asarray(pad4d(values), dtype=np.int64),
                dtype=TensorProto.INT64,
                with_value_info=False,
            )
        elif len(values) > 4:
            raise ValueError(f"{node.name!r}: {op_type} target has rank {len(values)}")
    return model


def _apply_non4d_transpose(model):
    """Remap transpose permutations around added or removed unary axes."""
    cache = GraphCache(model.graph)
    for node in model.graph.node:
        if node.op_type != "Transpose":
            continue
        perm = _attribute(node, "perm")
        if perm is None:
            continue
        values = list(perm.ints)
        if len(values) in (2, 3):
            delta = 4 - len(values)
            values = list(range(delta)) + [v + delta for v in values]
        elif len(values) == 5:
            shape = cache.get_shape(node.input[0])
            if shape is None or len(shape) != 5 or 1 not in shape:
                raise ValueError(f"{node.name!r}: 5D Transpose requires a known unary dimension")
            removed = shape.index(1)
            values = [v - int(v > removed) for v in values if v != removed]
        else:
            continue
        del perm.ints[:]
        perm.ints.extend(values)
    return model


def _apply_non4d_lpnorm(model):
    """Preserve normalization axes across leading-dimension promotion."""
    cache = GraphCache(model.graph)
    for node in model.graph.node:
        if node.op_type == "LpNormalization":
            axis = _attribute(node, "axis")
            shape = _shape_before_boundary_promotion(cache, node.input[0])
            if axis is not None and axis.i >= 0 and shape and len(shape) in (2, 3):
                axis.i = -1 if axis.i == len(shape) - 1 else axis.i + 4 - len(shape)
            elif axis is None:
                node.attribute.append(helper.make_attribute("axis", -1))
            node.name += "_axis_updated"
    return model


def _apply_flatten(model):
    """Replace Flatten with its four-dimensional equivalent shape."""
    graph = model.graph
    cache = GraphCache(graph)
    for node in graph.node:
        if node.op_type == "Flatten":
            shape = _shape_before_boundary_promotion(cache, node.input[0])
            axis = _attribute(node, "axis")
            axis = axis.i if axis is not None else 1
            if shape is None or any(d <= 0 for d in shape):
                raise ValueError(f"{node.name!r}: Flatten requires a known input shape")
            if axis < 0:
                axis += len(shape)
            batch = math.prod(shape[:axis])
            name = add_initializer(
                graph,
                "flatten_reshape_shape" if batch == 1 else f"{node.name}_reshape_shape",
                np.asarray([1, 1, batch, -1], dtype=np.int64),
                dtype=TensorProto.INT64,
                with_value_info=False,
            )
            node.CopyFrom(
                helper.make_node(
                    "Reshape", [node.input[0], name], list(node.output), name=node.name + "_reshape"
                )
            )
    return model


def _apply_argmax(model):
    """Preserve ArgMax axes and restore its declared output layout."""
    cache = GraphCache(model.graph)
    nodes = []
    for node in model.graph.node:
        nodes.append(node)
        if node.op_type != "ArgMax":
            continue
        shape = _shape_before_boundary_promotion(cache, node.input[0])
        if not shape:
            continue
        axis = _attribute(node, "axis")
        if axis is None:
            node.attribute.append(helper.make_attribute("axis", 0))
            axis = _attribute(node, "axis")
        if axis.i >= 0 and len(shape) in (2, 3):
            axis.i += 4 - len(shape)
        keepdims = _attribute(node, "keepdims")
        restore_shape = keepdims is not None and keepdims.i == 0
        if keepdims is not None:
            keepdims.i = 1
        node.name += "_axis_updated"
        if restore_shape:
            original = node.output[0]
            output_shape = cache.get_shape(original)
            if output_shape is None and original.endswith("_squeeze_input"):
                output_shape = cache.get_shape(original.removesuffix("_squeeze_input"))
            if output_shape is None:
                raise ValueError(f"Unknown original ArgMax output shape for {node.name!r}")
            target = add_initializer(
                model.graph,
                node.name + "_reshape_shape",
                np.asarray(pad4d(output_shape), dtype=np.int64),
                dtype=TensorProto.INT64,
                with_value_info=False,
            )
            node.output[0] = original + "_keepdims"
            nodes.append(
                helper.make_node(
                    "Reshape", [node.output[0], target], [original], name=node.name + "_reshape"
                )
            )
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    return model


def _apply_update_axis(model):
    """Remap supported operation axes from their native input ranks."""
    update = {
        "LogSoftmax",
        "Softmax",
        "DequantizeLinear",
        "TopK",
        "Concat",
        "Split",
        "QuantizeLinear",
    }
    skip = {
        "Gather",
        "GatherElements",
        "LpNormalization",
        "ArgMax",
        "ReduceMax",
        "ReduceMin",
        "ReduceSum",
        "ReduceMean",
        "Slice",
        "Flatten",
    }
    cache = GraphCache(model.graph)
    for node in model.graph.node:
        axis = _attribute(node, "axis")
        if axis is None or axis.i < 0 or node.op_type in skip:
            continue
        if node.op_type not in update:
            raise ValueError(f"{node.op_type} is unsupported for DLA axis transformation")
        if "axis_updated" in node.name:
            continue
        shape = _shape_before_boundary_promotion(cache, node.input[0])
        delta = {2: 2, 3: 1, 5: -1}.get(len(shape) if shape else 0, 0)
        if delta:
            axis.i += delta
            node.name += "_axis_updated"
    return model
