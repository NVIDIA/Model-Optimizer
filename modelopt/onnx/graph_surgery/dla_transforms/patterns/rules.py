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
"""Connected graph patterns lowered into DLA-compatible operations."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from onnxscript import ir
from onnxscript.rewriter import pattern

from ..common import constant_array, static_shape

if TYPE_CHECKING:
    from ..context import DlaGraph

__all__ = ["ReshapeQDQReduceSum", "ReshapeReduceSum", "ReshapeTransposeReshape", "TopKVectorGather"]


class ReshapeTransposeReshape(pattern.RewriteRuleClassBase):
    """Replace a leading-axis-preserving 5D permutation with 4D slice branches.

    For example, Reshape([B,a,b,c,d]) -> Transpose([0,2,1,3,4]) -> Reshape
    becomes one Slice/4D Reshape/Transpose per B, followed by Concat/Reshape.
    Fixed positive dimensions and a permutation retaining the leading axis are required.
    """

    def __init__(self, graph: DlaGraph):
        """Bind the conversion graph and shared initializer creation."""
        super().__init__(name=type(self).__name__)
        self.graph = graph

    def pattern(self, op, x, shape1, shape2):
        """Capture a Reshape/Transpose/Reshape chain."""
        first = op.Reshape(x, shape1, _outputs=["first"])
        middle = op.Transpose(first, _outputs=["middle"])
        return op.Reshape(middle, shape2, _outputs=["result"])

    def check(self, context, shape1, **bindings):
        """Select fixed five-dimensional intermediate shapes."""
        return _candidate(context, shape1, **bindings)

    def rewrite(self, op, x, shape1, shape2, first, middle, result):
        """Emit leading-axis slices with four-dimensional permutation branches."""
        source = static_shape(x)
        dims = _resolved_shape(
            shape1, source, allowzero=first.producer().attributes.get_int("allowzero", 0)
        )
        perm = middle.producer().attributes.get_ints("perm", None)
        if perm is None or sorted(perm) != list(range(5)) or perm[0] != 0:
            raise ValueError("TODO: support a five-dimensional Transpose moving the leading axis")
        if source is None or source[0] != dims[0]:
            raise ValueError("TODO: support reshape/transpose patterns that split the leading axis")
        final = _resolved_shape(
            shape2,
            [dims[index] for index in perm],
            allowzero=result.producer().attributes.get_int("allowzero", 0),
        )
        axes = self.graph.add_initializer(op, np.array([0], np.int64), "reshape_slice_axis")
        inner = self.graph.add_initializer(op, np.array(dims[1:], np.int64), "reshape_inner_shape")
        inner_perm = [index - 1 for index in perm[1:]]
        moved = [dims[index] for index in perm[1:]]
        branches = []
        for index in range(dims[0]):
            start = self.graph.add_initializer(
                op, np.array([index], np.int64), "reshape_slice_start"
            )
            end = self.graph.add_initializer(
                op, np.array([index + 1], np.int64), "reshape_slice_end"
            )
            sliced = _emit(op.Slice, [x, start, end, axes], [1, *source[1:]])
            reshaped = _emit(op.Reshape, [sliced, inner], dims[1:])
            branches.append(_emit(op.Transpose, [reshaped], moved, perm=inner_perm))
        joined = _emit(op.Concat, branches, [dims[0] * moved[0], *moved[1:]], axis=0)
        return _emit(
            op.Reshape,
            [joined, self.graph.add_initializer(op, np.array(final, np.int64), "reshape_target")],
            final,
        )


class ReshapeReduceSum(pattern.RewriteRuleClassBase):
    """Lower Reshape-to-5D/ReduceSum into sliced 4D reductions.

    A fixed non-leading reduction axis is merged into its preceding dimension;
    Slice/ReduceSum branches are concatenated, preserving keepdims with Unsqueeze.
    This avoids materializing the five-dimensional view.
    """

    def __init__(self, graph: DlaGraph):
        """Bind the conversion graph and shared initializer creation."""
        super().__init__(name=type(self).__name__)
        self.graph = graph

    quantized = False

    def pattern(self, op, x, shape1, axes):
        """Capture Reshape/ReduceSum with the variant's optional Q/DQ."""
        first = op.Reshape(x, shape1, _outputs=["first"])
        if self.quantized:
            quantized_value = op.QuantizeLinear(
                first, _allow_other_inputs=True, _outputs=["quantized"]
            )
            first = op.DequantizeLinear(
                quantized_value, _allow_other_inputs=True, _outputs=["restored"]
            )
        return op.ReduceSum(first, axes, _outputs=["result"])

    def check(self, context, shape1, **bindings):
        """Select fixed five-dimensional intermediate shapes."""
        return _candidate(context, shape1, **bindings)

    def rewrite(self, op, x, shape1, axes, first, result, **bindings):
        """Reduce independent four-dimensional slices and restore output rank."""
        source = static_shape(x)
        dims = _resolved_shape(
            shape1, source, allowzero=first.producer().attributes.get_int("allowzero", 0)
        )
        assert source is not None
        reduced = constant_array(axes)
        if reduced is None or reduced.shape != (1,) or not -5 <= int(reduced[0]) < 5:
            raise ValueError("TODO: support dynamic or multiple axes in reshape/reduction patterns")
        axis = int(reduced[0]) % 5
        if axis == 0:
            raise ValueError("TODO: support leading-axis reshape/reduction patterns")
        middleware = [
            bindings[key].producer() for key in ("quantized", "restored") if key in bindings
        ]
        for node in middleware:
            controls = [constant_array(value) for value in node.inputs[1:] if value is not None]
            if node.attributes.get_int("block_size", 0) or any(
                value is None or value.ndim != 0 for value in controls
            ):
                raise ValueError("TODO: support per-axis QDQ in reshape/reduction patterns")
        split_axis = axis - 1
        count, width = dims[split_axis : axis + 1]
        merged = [*dims[:split_axis], count * width, *dims[axis + 1 :]]
        data = (
            x
            if list(source) == merged
            else _emit(
                op.Reshape,
                [
                    x,
                    self.graph.add_initializer(
                        op, np.array(merged, np.int64), "reduce_merged_shape"
                    ),
                ],
                merged,
            )
        )
        slice_axis = self.graph.add_initializer(
            op, np.array([split_axis], np.int64), "reduce_slice_axis"
        )
        mapped_axes = np.array([split_axis - 4], np.int64)
        reduce_axis = (
            axes
            if np.array_equal(reduced, mapped_axes)
            else self.graph.add_initializer(op, mapped_axes, "reduce_axis")
        )
        branches = []
        for index in range(count):
            start = self.graph.add_initializer(
                op, np.array([index * width], np.int64), "reduce_slice_start"
            )
            end = self.graph.add_initializer(
                op, np.array([(index + 1) * width], np.int64), "reduce_slice_end"
            )
            branch_shape = merged.copy()
            branch_shape[split_axis] = width
            branch = _emit(op.Slice, [data, start, end, slice_axis], branch_shape)
            for node in middleware:
                branch = _emit(
                    getattr(op, node.op_type),
                    [branch, *node.inputs[1:]],
                    branch_shape,
                    **dict(node.attributes),
                )
                branch.type = node.outputs[0].type
            branch_shape[split_axis] = 1
            branches.append(_emit(op.ReduceSum, [branch, reduce_axis], branch_shape))
        shape = [*dims[:axis], *dims[axis + 1 :]]
        joined = _emit(op.Concat, branches, shape, axis=split_axis)
        if result.producer().attributes.get_int("keepdims", 1):
            shape.insert(axis, 1)
            return _emit(
                op.Unsqueeze,
                [
                    joined,
                    self.graph.add_initializer(
                        op, np.array([axis], np.int64), "reduce_restore_axis"
                    ),
                ],
                shape,
            )
        return joined


class ReshapeQDQReduceSum(ReshapeReduceSum):
    """Preserve scalar Q/DQ while splitting a five-dimensional reduction.

    Reshape -> QuantizeLinear -> DequantizeLinear -> ReduceSum becomes 4D
    Slice -> Q/DQ -> ReduceSum branches followed by the original layout restoration.
    Per-axis and blocked quantization require a separate lowering.
    """

    quantized = True


class TopKVectorGather(pattern.RewriteRuleClassBase):
    """Replace private tiled TopK indices and GatherElements with vector Gather.

    TopK -> Cast -> Reshape -> Tile -> [integer Casts] -> GatherElements
    becomes a channel-axis TopK, an INT32 index vector and Gather. Scores must
    have singleton dimensions before their final axis; index views must be private.
    """

    def __init__(self, graph: DlaGraph):
        """Bind the conversion graph and shared initializer creation."""
        super().__init__(name=type(self).__name__)
        self.graph = graph

    def pattern(self, op, data, indices):
        """Anchor private index-chain discovery at GatherElements."""
        return op.GatherElements(data, indices, _outputs=["result"])

    def check(self, context, **bindings):
        """Prove the supported score layout and private integer index ownership."""
        return _topk_plan(context.root) is not None

    def rewrite(self, op, data, indices, result):
        """Select channel-wise indices and gather with their INT32 vector."""
        topk, n, k, axis = _topk_plan(result.producer())
        shape = self.graph.add_initializer(op, np.array([1, n, 1, 1], np.int64), "topk_score_shape")
        scores = op.Reshape(topk.inputs[0], shape)
        scores.type, scores.shape = topk.inputs[0].type, ir.Shape([1, n, 1, 1])
        attrs = dict(topk.attributes)
        attrs["axis"] = 1
        values, selected = op.TopK(scores, topk.inputs[1], _outputs=2, **attrs)
        values.type, values.shape = scores.type, ir.Shape([1, k, 1, 1])
        selected.type, selected.shape = ir.TensorType(ir.DataType.INT64), ir.Shape([1, k, 1, 1])
        selected = op.Cast(selected, to=ir.DataType.INT32)
        selected.type, selected.shape = ir.TensorType(ir.DataType.INT32), ir.Shape([1, k, 1, 1])
        axes = self.graph.add_initializer(op, np.array([0, 2, 3], np.int64), "topk_vector_axes")
        vector = op.Squeeze(selected, axes)
        vector.type, vector.shape = ir.TensorType(ir.DataType.INT32), ir.Shape([k])
        gathered = op.Gather(data, vector, axis=axis)
        gathered.type, gathered.shape = result.type, result.shape
        return gathered


def _resolved_shape(value, source, *, allowzero=0):
    target = constant_array(value)
    if target is None or target.ndim != 1 or source is None:
        raise ValueError("TODO: support dynamic Reshape controls and dimensions")
    result = [int(dim) for dim in target]
    if allowzero or any(dim < -1 for dim in result) or result.count(-1) > 1:
        raise ValueError("TODO: support literal-zero Reshape dimensions")
    result = [source[index] if dim == 0 else dim for index, dim in enumerate(result)]
    if -1 in result:
        known = math.prod(dim for dim in result if dim != -1)
        if known <= 0 or math.prod(source) % known:
            raise ValueError("Reshape target does not preserve the element count")
        result[result.index(-1)] = math.prod(source) // known
    if any(dim <= 0 for dim in result) or math.prod(result) != math.prod(source):
        raise ValueError("Reshape target must have positive dimensions and preserve size")
    return result


def _candidate(ctx, shape1, **_):
    target = constant_array(shape1)
    return target is not None and target.shape == (5,)


def _emit(operator, inputs, shape, **attrs):
    result = operator(*inputs, **attrs)
    result.type = inputs[0].type
    result.shape = ir.Shape(shape)
    return result


def _private_producer(value, operation):
    node = value.producer()
    if value.is_graph_output() or len(value.uses()) != 1 or node is None:
        return None
    return node if node.op_type == operation and not node.domain and not node.overload else None


def _topk_plan(gather):
    tail = gather.inputs[1]
    casts = []
    while (cast := _private_producer(tail, "Cast")) is not None:
        if cast.attributes.get_int("to", 0) not in (ir.DataType.INT32, ir.DataType.INT64):
            return None
        casts.append(cast)
        tail = cast.inputs[0]
    tile = _private_producer(tail, "Tile")
    reshape = _private_producer(tile.inputs[0], "Reshape") if tile else None
    cast = _private_producer(reshape.inputs[0], "Cast") if reshape else None
    topk = _private_producer(cast.inputs[0], "TopK") if cast else None
    if topk is None or cast is None or reshape is None or tile is None:
        return None
    if cast.inputs[0] is not topk.outputs[1]:
        return None
    scores = static_shape(topk.inputs[0])
    data = static_shape(gather.inputs[0])
    if (
        topk.outputs[0].is_graph_output()
        or topk.outputs[0].uses()
        or scores is None
        or len(scores) not in (2, 4)
        or any(dim != 1 for dim in scores[:-1])
        or data is None
        or len(data) not in ((4,) if len(scores) == 4 else (3, 4))
        or cast.attributes.get_int("to", 0) not in (ir.DataType.INT32, ir.DataType.INT64)
        or (casts and len(scores) != 4)
    ):
        return None
    n = scores[-1]
    axis = 2 if len(scores) == 4 else 1
    if (
        not 1 < n <= np.iinfo(np.int32).max
        or topk.attributes.get_int("axis", -1) % len(scores) != len(scores) - 1
        or gather.attributes.get_int("axis", 0) % len(data) != axis
        or data[axis] != n
    ):
        return None
    k_array, repeats = constant_array(topk.inputs[1]), constant_array(tile.inputs[1])
    if k_array is None or k_array.shape != (1,) or k_array.dtype != np.int64:
        return None
    k = int(k_array[0])
    if not 0 < k <= n or repeats is None:
        return None
    required = [1] * len(data)
    required[axis] = k
    tiled = list(data)
    tiled[axis] = 1
    if (
        _resolved_shape(
            reshape.inputs[1],
            [*scores[:-1], k],
            allowzero=reshape.attributes.get_int("allowzero", 0),
        )
        != required
        or repeats.tolist() != tiled
    ):
        return None
    return topk, n, k, axis
