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
"""Fixed operator rewrites into DLA-compatible primitives."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from onnxscript import ir
from onnxscript.rewriter import pattern

from ..common import constant_array, static_shape

if TYPE_CHECKING:
    from ..context import DlaGraph

__all__ = [
    "INDEXING_RULES",
    "MATRIX_RULES",
    "GatherElementsIndices",
    "GatherIndices",
    "GemmBiasToConv",
    "GemmToConv",
    "MatMulToConv",
    "weight_eligible",
]


class _OperatorRule(pattern.RewriteRuleClassBase):
    def __init__(self, graph: DlaGraph, views=None):
        super().__init__(name=type(self).__name__)
        self.graph = graph
        self.views = {} if views is None else views


class _IndexRule(_OperatorRule):
    def check(self, context, data, indices, result):
        if context.root not in self.graph.batch_nodes or indices.dtype == ir.DataType.INT32:
            return False
        if indices.dtype != ir.DataType.INT64:
            raise ValueError("DLA indexing requires integer indices; need an additional transform")
        shape = static_shape(data)
        axis = result.producer().attributes.get_int("axis", 0)
        if shape is None or not shape or not -len(shape) <= axis < len(shape):
            raise ValueError("DLA indexing requires a static, valid data axis")
        if shape[axis] > np.iinfo(np.int32).max:
            raise ValueError("DLA indexing axes larger than INT32 need an additional transform")
        return True

    def rewrite(self, op, data, indices, result):
        if indices not in self.views:
            array = constant_array(indices)
            self.views[indices] = (
                self.graph.add_initializer(op, array.astype(np.int32), "index_int32")
                if array is not None
                else op.Cast(indices, to=ir.DataType.INT32)
            )
            self.views[indices].type, self.views[indices].shape = (
                ir.TensorType(ir.DataType.INT32),
                indices.shape,
            )
        node = result.producer()
        return op.call_op(node.op_type, [data, self.views[indices]], dict(node.attributes))


class GatherIndices(_IndexRule):
    """Use a private INT32 Gather index view without changing the shared owner.

    Gather(data, INT64 indices) becomes Gather(data, Cast(indices, INT32)); fixed
    indices use an INT32 initializer instead. Data rank and axis are preserved,
    and a static axis fitting INT32 is required.
    """

    def pattern(self, op, data, indices):
        """Capture a Gather and its native index owner."""
        return op.Gather(data, indices, _outputs=["result"])


class GatherElementsIndices(_IndexRule):
    """Use private INT32 GatherElements indices while retaining native index rank.

    Runtime indices receive a Cast; fixed indices receive a converted initializer.
    Both indexing variants share their view when they consume the same owner.
    """

    def pattern(self, op, data, indices):
        """Capture GatherElements without imposing a layout on indices."""
        return op.GatherElements(data, indices, _outputs=["result"])


class _MatrixRule(_OperatorRule):
    def __init__(self, graph: DlaGraph, views=None):
        super().__init__(graph, views)
        self.weights = self.views.setdefault("weights", {})
        self.activations = self.views.setdefault("activations", {})

    def check(self, context, x, weight, result, **bindings):
        return context.root in self.graph.batch_nodes and weight_eligible(result.producer(), weight)

    def rewrite(self, op, x, weight, result, **bindings):
        node = result.producer()
        shape = static_shape(x)
        assert shape is not None
        dq = weight.producer()
        dq = (
            dq
            if dq is not None
            and dq.op_type == "DequantizeLinear"
            and dq.domain in ("", "com.microsoft")
            else None
        )
        source = dq.inputs[0] if dq is not None else weight
        raw = constant_array(source)
        assert raw is not None
        attrs = node.attributes
        is_gemm = node.op_type == "Gemm"
        transpose = not is_gemm or not attrs.get_int("transB", 0)
        kernel = (raw.transpose() if transpose else raw).reshape(
            raw.shape[-1] if transpose else raw.shape[-2], -1, 1, 1
        )
        axis = None
        if dq is not None:
            scale_shape = static_shape(dq.inputs[1])
            if "axis" in dq.attributes or scale_shape:
                old_axis = dq.attributes.get_int("axis", 1) % raw.ndim
                axis = raw.ndim - 1 - old_axis if transpose else old_axis
        output_shape = static_shape(result)
        output_rank = len(output_shape) if output_shape is not None else max(len(shape), raw.ndim)
        padded_shape = (1,) * (4 - len(shape)) + shape
        key = (
            source,
            weight if dq is not None else None,
            tuple(kernel.shape),
            kernel.dtype.str,
            bool(node.op_type == "MatMul" or not node.attributes.get_int("transB", 0)),
            axis,
        )
        if key not in self.weights:
            value = self.graph.add_initializer(op, kernel.copy(), "matrix_weight")
            if dq is not None:
                attrs = {name: attr for name, attr in dq.attributes.items() if name != "axis"}
                if axis is not None:
                    attrs["axis"] = axis
                value = op.call_op(
                    "DequantizeLinear", [value, *dq.inputs[1:]], attrs, domain=dq.domain
                )
                value.type, value.shape = weight.type, ir.Shape(kernel.shape)
            self.weights[key] = value
        if x not in self.activations:
            self.activations[x] = (
                op.Unsqueeze(
                    x,
                    self.graph.add_initializer(
                        op, np.arange(4 - len(shape), dtype=np.int64), "matrix_axes"
                    ),
                )
                if len(shape) < 4
                else x
            )
            self.activations[x].type = x.type
            self.activations[x].shape = ir.Shape(padded_shape)
        trans_a = node.attributes.get_int("transA", 0) if node.op_type == "Gemm" else 0
        permutation = [0, 2, 3, 1] if trans_a else [0, 3, 2, 1]
        activation = op.Transpose(self.activations[x], perm=permutation)
        activation_shape = tuple(padded_shape[index] for index in permutation)
        activation.type, activation.shape = x.type, ir.Shape(activation_shape)
        operands = [activation, self.weights[key]]
        if "bias" in bindings:
            operands.append(bindings["bias"])
        convolved = op.Conv(*operands)
        conv_shape = (activation_shape[0], kernel.shape[0], *activation_shape[2:])
        convolved.type, convolved.shape = result.type, ir.Shape(conv_shape)
        value = op.Transpose(convolved, perm=[0, 3, 2, 1])
        output_shape = (conv_shape[0], conv_shape[3], conv_shape[2], conv_shape[1])
        value.type, value.shape = result.type, ir.Shape(output_shape)
        if output_rank < 4:
            value = op.Squeeze(
                value,
                self.graph.add_initializer(
                    op, np.arange(4 - output_rank, dtype=np.int64), "matrix_axes"
                ),
            )
            value.type, value.shape = result.type, ir.Shape(output_shape[4 - output_rank :])
        return value


class MatMulToConv(_MatrixRule):
    """Lower a fixed-weight MatMul to a pointwise Conv with rank adapters.

    MatMul(x, W) becomes Unsqueeze/Transpose -> Conv(W reshaped to OI11) ->
    Transpose/Squeeze. Unary weight batch dimensions and supported per-axis Q/DQ
    weights are retained as private views; runtime weights remain unchanged.
    """

    def pattern(self, op, x, weight):
        """Capture a fixed-weight candidate before layout promotion."""
        return op.MatMul(x, weight, _outputs=["result"])


class GemmToConv(_MatrixRule):
    """Lower a fixed-weight, unit-coefficient Gemm into a pointwise Conv.

    Handles transA/transB through activation and kernel views, then restores the
    native rank-two result. Non-unit alpha/beta or unsupported bias forms remain.
    """

    def pattern(self, op, x, weight):
        """Capture the two-input Gemm form."""
        return op.Gemm(x, weight, _outputs=["result"])


class GemmBiasToConv(_MatrixRule):
    """Preserve a vector Gemm bias as the bias input of the lowered Conv.

    The same rank and quantized-weight handling as GemmToConv applies, with a
    fixed channel-vector bias and unit alpha/beta.
    """

    def pattern(self, op, x, weight, bias):
        """Capture the Gemm form with a bias operand."""
        return op.Gemm(x, weight, bias, _outputs=["result"])


def weight_eligible(node: ir.Node, weight: ir.Value) -> bool:
    """Check whether a matrix's native fixed weight can use the Conv lowering.

    Preprocessing uses this predicate to retain eligible quantized weight chains.
    It neither changes the graph nor creates a transformed weight allocation.
    """
    activation = node.inputs[0]
    if activation is None:
        return False
    shape = static_shape(activation)
    if shape is None or len(shape) not in (2, 3, 4) or min(shape) <= 0:
        return False
    if len(shape) == 4 and shape[0] != 1:
        return False
    if activation.dtype not in (ir.DataType.FLOAT, ir.DataType.FLOAT16, ir.DataType.DOUBLE):
        return False
    dq = weight.producer()
    dq = (
        dq
        if dq is not None
        and dq.op_type == "DequantizeLinear"
        and dq.domain in ("", "com.microsoft")
        else None
    )
    source = dq.inputs[0] if dq is not None else weight
    raw = constant_array(source)
    if (
        raw is None
        or raw.ndim not in (2, 3, 4)
        or any(d != 1 for d in raw.shape[:-2])
        or not raw.size
    ):
        return False
    if dq is not None:
        scale_shape = static_shape(dq.inputs[1])
        if dq.attributes.get_int("block_size", 0) or scale_shape is None or len(scale_shape) > 1:
            return False
    if node.op_type == "Gemm":
        if len(shape) != 2 or raw.ndim != 2:
            return False
        if any(node.attributes.get_float(key, 1.0) != 1.0 for key in ("alpha", "beta")):
            return False
        channels = raw.shape[-2] if node.attributes.get_int("transB", 0) else raw.shape[-1]
        if len(node.inputs) > 2 and node.inputs[2] is not None:
            if static_shape(node.inputs[2]) != (channels,):
                return False
    return True


INDEXING_RULES = (GatherIndices, GatherElementsIndices)

MATRIX_RULES = (MatMulToConv, GemmToConv, GemmBiasToConv)
