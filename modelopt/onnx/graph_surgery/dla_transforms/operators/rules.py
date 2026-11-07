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
    "LOGICAL_RULES",
    "MATRIX_RULES",
    "RECURRENT_RULES",
    "GatherElementsIndices",
    "GatherIndices",
    "GemmBiasToConv",
    "GemmToConv",
    "GreaterFloat",
    "LogicalNot",
    "Lstm1",
    "Lstm2",
    "Lstm3",
    "LstmSequenceLayout",
    "MatMulToConv",
    "WhereZero",
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


class _LogicalRule(_OperatorRule):
    def __init__(self, graph: DlaGraph, views=None):
        super().__init__(graph, views)
        self.zeros = self.views.setdefault("zeros", {})
        self.masks = self.views.setdefault("masks", {})
        self.floats = self.views.setdefault("floats", {})

    def _float_view(self, op, value):
        if value.dtype == ir.DataType.FLOAT:
            return value
        if value not in self.floats:
            array = constant_array(value)
            self.floats[value] = (
                self.graph.add_initializer(op, array.astype(np.float32), "logical_float")
                if array is not None
                else op.Cast(value, to=ir.DataType.FLOAT)
            )
            self.floats[value].type, self.floats[value].shape = (
                ir.TensorType(ir.DataType.FLOAT),
                value.shape,
            )
        return self.floats[value]


class WhereZero(_LogicalRule):
    """Replace a finite FP32 zero-branch Where with multiplication by a float mask.

    Where(c, x, 0) becomes x * Cast(c, FLOAT); Where(c, 0, y) uses 1-mask.
    The zero branch must not add broadcast dimensions. NaN/Inf behavior and the
    sign of zero are outside this finite-data lowering contract.
    """

    def pattern(self, op, condition, x, y):
        """Capture both branch orientations of Where."""
        return op.Where(condition, x, y, _outputs=["result"])

    def check(self, context, condition, x, y, result):
        """Require one zero branch and unchanged broadcast extent."""
        for index, value in enumerate((x, y)):
            if value not in self.zeros:
                array = constant_array(value)
                self.zeros[value] = array is not None and not np.count_nonzero(array)
            if self.zeros[value]:
                other = (y, x)[index]
                shapes = [
                    shape for v in (condition, x, y) if (shape := static_shape(v)) is not None
                ]
                if other.dtype != ir.DataType.FLOAT or len(shapes) != 3:
                    return False
                if np.broadcast_shapes(*shapes) != np.broadcast_shapes(
                    shapes[0], shapes[2 - index]
                ):
                    return False
                self.zeros[result] = index
                return True
        return False

    def rewrite(self, op, condition, x, y, result):
        """Multiply the nonzero branch by its shared float mask."""
        zero_index = self.zeros[result]
        key = (condition, zero_index)
        if key not in self.masks:
            array = constant_array(condition)
            if array is not None and condition.dtype == ir.DataType.BOOL:
                mask = array.astype(np.float32)
                if zero_index == 0:
                    mask = 1.0 - mask
                self.masks[key] = self.graph.add_initializer(op, mask, "where_mask")
            else:
                mask = self._float_view(op, condition)
                if zero_index == 0:
                    mask = op.Sub(
                        self.graph.add_initializer(op, np.array(1, np.float32), "where_one"), mask
                    )
                    mask.type, mask.shape = ir.TensorType(ir.DataType.FLOAT), condition.shape
                self.masks[key] = mask
        return op.Mul(y if zero_index == 0 else x, self.masks[key])


class LogicalNot(_LogicalRule):
    """Express a private binary Cast-to-BOOL/Not region as float arithmetic.

    Not(Cast(x, BOOL)) becomes 1 - Clip(Cast(x, FLOAT), 0, 1) when x is BOOL
    or fixed 0/1 data and every output consumer is a non-string Cast. Public
    boolean outputs and unproven binary sources keep the original operators.
    """

    def pattern(self, op, x):
        """Capture the private Cast-to-BOOL/Not region."""
        return op.Not(op.Cast(x, to=int(ir.DataType.BOOL)), _outputs=["result"])

    def check(self, context, x, result):
        """Prove binary input values and numeric-only downstream uses."""
        if context.root not in self.graph.batch_nodes or result.is_graph_output():
            return False
        if any(
            user.domain
            or user.op_type != "Cast"
            or user.attributes.get_int("to", 0) == ir.DataType.STRING
            for user, _ in result.uses()
        ):
            return False
        array = constant_array(x)
        return x.dtype == ir.DataType.BOOL or (
            array is not None and np.all((array == 0) | (array == 1))
        )

    def rewrite(self, op, x, result):
        """Emit the bounded float complement and update internal output typing."""
        minimum = self.graph.add_initializer(op, np.array(0, np.float32), "not_min")
        maximum = self.graph.add_initializer(op, np.array(1, np.float32), "not_max")
        one = self.graph.add_initializer(op, np.array(1, np.float32), "not_one")
        # SDK replacement carries this internal value's metadata to the new output.
        result.type = ir.TensorType(ir.DataType.FLOAT)
        cast = op.Cast(x, to=ir.DataType.FLOAT)
        cast.type, cast.shape = result.type, x.shape
        clipped = op.Clip(cast, minimum, maximum)
        clipped.type, clipped.shape = result.type, x.shape
        return op.Sub(one, clipped)


class GreaterFloat(_LogicalRule):
    """Give Greater private FP32 operands when the ordering conversion is exact.

    Narrow numeric inputs use float views; INT32/INT64 cases require a fixed
    operand strictly within the exact FP32 integer range. Existing FLOAT/FLOAT
    comparisons stay unchanged, so an emitted Greater is already canonical.
    """

    def pattern(self, op, x, y):
        """Capture Greater operands independently of their numeric type."""
        return op.Greater(x, y, _outputs=["result"])

    def check(self, context, x, y, result):
        """Require conversion to preserve the comparison boundary."""
        if context.root not in self.graph.batch_nodes or x.dtype == y.dtype == ir.DataType.FLOAT:
            return False
        exact = {
            ir.DataType.FLOAT,
            ir.DataType.FLOAT16,
            ir.DataType.INT8,
            ir.DataType.UINT8,
            ir.DataType.INT16,
            ir.DataType.UINT16,
        }
        if {x.dtype, y.dtype} <= exact:
            return True
        if not {x.dtype, y.dtype} <= exact | {ir.DataType.INT32, ir.DataType.INT64}:
            return False
        return any(
            (array := constant_array(v)) is not None
            and (not array.size or (array.min() > -(2**24) and array.max() < 2**24))
            for v in (x, y)
        )

    def rewrite(self, op, x, y, result):
        """Reuse private float views without changing the shared operands."""
        return op.Greater(self._float_view(op, x), self._float_view(op, y))


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


class _LstmRule(_OperatorRule):
    def __init__(self, graph: DlaGraph, views=None):
        super().__init__(graph, views)
        self.name = f"LSTM{self.output_count}"

    def check(self, context, **bindings):
        if (
            context.root not in self.graph.batch_nodes
            or len(context.root.outputs) != self.output_count
        ):
            return False
        _validate_lstm(context.root)
        return True

    def rewrite(self, op, x, w, r, sequence, **_):
        node = sequence.producer()
        _capture_lstm_sequence_layout(node)
        shape = static_shape(x)
        assert shape is not None
        steps, _, features = shape
        hidden = node.attributes.get_int("hidden_size", 0)
        direction = node.attributes.get_string("direction", "forward")
        directions = 2 if direction == "bidirectional" else 1
        weights, recurrent = constant_array(w), constant_array(r)
        assert weights is not None and recurrent is not None
        dtype = weights.dtype

        def constant(array, label):
            return self.graph.add_initializer(op, np.asarray(array), label)

        def emit(operator, inputs, dims, **attrs):
            value = operator(*inputs, **attrs)
            value.type, value.shape = x.type, ir.Shape(dims)
            return value

        axes = constant(np.array([0], np.int64), "lstm_input_axis")
        promoted = emit(op.Unsqueeze, [x, axes], [1, steps, 1, features])
        channels = emit(op.Transpose, [promoted], [1, features, 1, steps], perm=[0, 3, 2, 1])
        gate_sizes = constant(np.full(4, hidden, np.int64), "lstm_gate_sizes")
        time_sizes = constant(np.ones(steps, np.int64), "lstm_time_sizes")
        state_cache = {}

        def state(index, branch):
            source = _parameter(node, index)
            array = constant_array(source)
            if source is None:
                return np.zeros(hidden, dtype=dtype)
            if array is not None:
                return array[branch].reshape(hidden)
            if index not in state_cache:
                promoted_state = emit(op.Unsqueeze, [source, axes], [1, directions, 1, hidden])
                moved = emit(
                    op.Transpose,
                    [promoted_state],
                    [1, hidden, 1, directions],
                    perm=[0, 3, 2, 1],
                )
                if directions == 2:
                    moved = op.Split(
                        moved,
                        constant(np.ones(2, np.int64), "lstm_direction_sizes"),
                        axis=3,
                        _outputs=2,
                    )
                    for value in moved:
                        value.type, value.shape = x.type, ir.Shape([1, hidden, 1, 1])
                else:
                    moved = [moved]
                state_cache[index] = moved
            return state_cache[index][branch]

        outputs = []
        for branch in range(directions):
            reverse = direction == "reverse" or (direction == "bidirectional" and branch == 1)
            projected = emit(
                op.Conv,
                [
                    channels,
                    constant(weights[branch].reshape(4 * hidden, features, 1, 1), "lstm_weights"),
                ],
                [1, 4 * hidden, 1, steps],
            )
            bias = np.zeros(8 * hidden, dtype=dtype)
            if _parameter(node, 3) is not None:
                bias_array = constant_array(node.inputs[3])
                assert bias_array is not None
                bias = bias_array[branch]
            folded = (bias[: 4 * hidden].astype(np.float64) + bias[4 * hidden :]).astype(dtype)
            projected = emit(
                op.Add,
                [constant(folded.reshape(1, -1, 1, 1), "lstm_bias"), projected],
                [1, 4 * hidden, 1, steps],
            )
            gates = op.Split(projected, gate_sizes, axis=1, _outputs=4)
            for gate in gates:
                gate.type, gate.shape = x.type, ir.Shape([1, hidden, 1, steps])
            slices = [op.Split(gate, time_sizes, axis=3, _outputs=steps) for gate in gates]
            if steps == 1:
                slices = [[value] for value in slices]
            for parts in slices:
                for value in parts:
                    value.type, value.shape = x.type, ir.Shape([1, hidden, 1, 1])
            rec = recurrent[branch].reshape(4, hidden, hidden)
            rec_weights = [
                constant(value.reshape(hidden, hidden, 1, 1), "lstm_recurrent_weights")
                for value in rec
            ]
            initial, cell = state(5, branch), state(6, branch)
            if isinstance(cell, np.ndarray):
                cell = constant(cell.reshape(1, hidden, 1, 1), "lstm_initial_cell")
            hidden_value = initial
            history = []
            for ordinal, step in enumerate(reversed(range(steps)) if reverse else range(steps)):
                activated = {}
                for gate in (0, 2, 1, 3):
                    if ordinal == 0 and isinstance(initial, np.ndarray):
                        values = []
                        for row in rec[gate]:
                            total = 0.0
                            for weight, value in zip(row, initial, strict=True):
                                total += float(weight) * float(value)
                            values.append(total)
                        recurrent_value = constant(
                            np.asarray(values, dtype=dtype).reshape(1, hidden, 1, 1),
                            "lstm_initial_gate",
                        )
                    else:
                        recurrent_value = emit(
                            op.Conv, [hidden_value, rec_weights[gate]], [1, hidden, 1, 1]
                        )
                    summed = emit(op.Add, [recurrent_value, slices[gate][step]], [1, hidden, 1, 1])
                    activated[gate] = emit(
                        op.Tanh if gate == 3 else op.Sigmoid, [summed], [1, hidden, 1, 1]
                    )
                forget = emit(op.Mul, [activated[2], cell], [1, hidden, 1, 1])
                update = emit(op.Mul, [activated[0], activated[3]], [1, hidden, 1, 1])
                cell = emit(op.Add, [forget, update], [1, hidden, 1, 1])
                hidden_value = emit(
                    op.Mul,
                    [activated[1], emit(op.Tanh, [cell], [1, hidden, 1, 1])],
                    [1, hidden, 1, 1],
                )
                history.append(hidden_value)
            history = history[::-1] if reverse else history
            joined = emit(op.Concat, history, [1, hidden, 1, steps], axis=3)
            outputs.append([joined, hidden_value, cell])
        results = []
        for index in range(self.output_count):
            values = [branch[index] for branch in outputs]
            merged = (
                emit(
                    op.Concat,
                    values,
                    [1, hidden, directions, steps if index == 0 else 1],
                    axis=2,
                )
                if directions == 2
                else values[0]
            )
            arranged = emit(
                op.Transpose,
                [merged],
                [steps if index == 0 else 1, directions, 1, hidden],
                perm=[3, 2, 0, 1],
            )
            if index:
                arranged = emit(
                    op.Squeeze,
                    [arranged, constant(np.array([0], np.int64), "lstm_output_axis")],
                    [directions, 1, hidden],
                )
            results.append(arranged)
        return results[0] if self.output_count == 1 else results


class Lstm1(_LstmRule):
    """Unroll a fixed-length FP32, batch-one LSTM with its sequence output.

    Each time step becomes pointwise Conv gate projections, Sigmoid/Tanh and
    elementwise state updates. Forward, reverse and bidirectional layout-0 forms
    support fixed W/R/bias and fixed or runtime native initial states; custom
    activations, sequence lengths and peepholes require another lowering.
    """

    output_count = 1

    def pattern(self, op, x, w, r):
        """Capture the sequence-only LSTM output form."""
        return op.LSTM(
            x,
            w,
            r,
            _outputs=["sequence", "hidden", "cell"][: self.output_count],
            _allow_other_inputs=True,
        )


class Lstm2(_LstmRule):
    """Unroll a fixed-length LSTM while preserving sequence and final hidden outputs.

    Each time step becomes pointwise Conv gate projections, Sigmoid/Tanh and
    elementwise state updates. Forward, reverse and bidirectional layout-0 forms
    support fixed W/R/bias and fixed or runtime native initial states; custom
    activations, sequence lengths and peepholes require another lowering.
    """

    output_count = 2

    def pattern(self, op, x, w, r):
        """Capture the sequence and hidden-state output form."""
        return op.LSTM(
            x,
            w,
            r,
            _outputs=["sequence", "hidden", "cell"][: self.output_count],
            _allow_other_inputs=True,
        )


class Lstm3(_LstmRule):
    """Unroll a fixed-length LSTM preserving sequence, hidden and cell outputs.

    Each time step becomes pointwise Conv gate projections, Sigmoid/Tanh and
    elementwise state updates. Forward, reverse and bidirectional layout-0 forms
    support fixed W/R/bias and fixed or runtime native initial states; custom
    activations, sequence lengths and peepholes require another lowering.
    """

    output_count = 3

    def pattern(self, op, x, w, r):
        """Capture all three LSTM outputs."""
        return op.LSTM(
            x,
            w,
            r,
            _outputs=["sequence", "hidden", "cell"][: self.output_count],
            _allow_other_inputs=True,
        )


class LstmSequenceLayout(_OperatorRule):
    """Resolve singleton-only sequence permutations after an LSTM is unrolled.

    A native LSTM sequence Transpose -> lower-rank Reshape becomes an identity
    Transpose -> Reshape with copied dimensions resolved before the old layout
    disappears. Shared shape controls retain their source metadata for layout.
    """

    def __init__(self, graph: DlaGraph, views=None):
        """Bind shared sequence controls and retain the established SDK rule name."""
        super().__init__(graph, views)
        self.name = "LSTMSequenceLayout"

    def pattern(self, op, x, shape):
        """Capture a sequence permutation followed by a Reshape."""
        middle = op.Transpose(x, _outputs=["middle"])
        return op.Reshape(middle, shape, _outputs=["result"])

    def check(self, context, **bindings):
        """Require native consumer metadata saved by LSTM unrolling."""
        return "dla_lstm_sequence_layout" in context.root.meta

    def rewrite(self, op, x, shape, result, **_):
        """Resolve native copied dimensions and retain shape-control provenance."""
        node = result.producer()
        source, resolved = node.meta["dla_lstm_sequence_layout"]
        viewed = op.Transpose(x, perm=list(range(len(source))))
        viewed.type, viewed.shape = x.type, ir.Shape(source)
        key = shape, resolved
        if key not in self.views:
            self.views[key] = self.graph.add_initializer(
                op, np.array(resolved, np.int64), "lstm_sequence_shape"
            )
            # Retain shared shape ownership for later layout views; IR metadata is not serialized.
            self.views[key].meta["dla_shape_source"] = shape
        value = op.Reshape(viewed, self.views[key], **dict(node.attributes))
        value.type, value.shape = result.type, result.shape
        return value


def _parameter(node, index):
    return node.inputs[index] if index < len(node.inputs) else None


def _validate_lstm(node):
    dims = static_shape(node.inputs[0])
    if dims is None or len(dims) != 3 or any(dim <= 0 for dim in dims) or dims[1] != 1:
        raise ValueError("TODO: support dynamic or multi-batch LSTM inputs")
    if node.inputs[0].dtype != ir.DataType.FLOAT:
        raise ValueError("TODO: support non-FP32 LSTM inputs before strong typing")
    if (
        node.attributes.get_int("layout", 0) != 0
        or node.attributes.get_int("input_forget", 0)
        or node.attributes.get_string("direction", "forward")
        not in {"forward", "reverse", "bidirectional"}
        or any(
            key in node.attributes
            for key in ("clip", "activations", "activation_alpha", "activation_beta")
        )
        or _parameter(node, 4) is not None
        or _parameter(node, 7) is not None
    ):
        raise ValueError(
            "TODO: support LSTM sequence lengths, peepholes and custom activation options"
        )
    directions = 2 if node.attributes.get_string("direction", "forward") == "bidirectional" else 1
    hidden = node.attributes.get_int("hidden_size", 0)
    expected = [(directions, 4 * hidden, dims[2]), (directions, 4 * hidden, hidden)]
    for index, shape in zip((1, 2), expected, strict=True):
        value = constant_array(_parameter(node, index))
        if value is None or value.shape != shape or hidden <= 0:
            raise ValueError(
                "TODO: support runtime LSTM weights; static weights must match hidden_size"
            )
    bias = _parameter(node, 3)
    if bias is not None and (
        constant_array(bias) is None or static_shape(bias) != (directions, 8 * hidden)
    ):
        raise ValueError("TODO: support runtime LSTM bias")
    for index in (5, 6):
        state = _parameter(node, index)
        if state is not None and static_shape(state) != (directions, 1, hidden):
            raise ValueError(
                "LSTM initial states must match direction, batch and hidden dimensions"
            )


def _capture_lstm_sequence_layout(lstm):
    source = static_shape(lstm.outputs[0])
    if source is None:
        return
    for transpose, port in lstm.outputs[0].uses():
        if transpose.op_type != "Transpose" or transpose.domain or port != 0:
            continue
        perm = transpose.attributes.get_ints("perm", ())
        if sorted(perm) != list(range(len(source))) or [
            axis for axis in perm if source[axis] != 1
        ] != [axis for axis, dim in enumerate(source) if dim != 1]:
            continue
        moved = tuple(source[axis] for axis in perm)
        for reshape, port in transpose.outputs[0].uses():
            if reshape.op_type != "Reshape" or reshape.domain or port != 0:
                continue
            control_values = constant_array(reshape.inputs[1])
            output = static_shape(reshape.outputs[0])
            if (
                control_values is None
                or control_values.ndim != 1
                or output is None
                or len(output) >= 4
                or reshape.attributes.get_int("allowzero", 0)
            ):
                continue
            resolved = tuple(
                moved[index] if dim == 0 else int(dim) for index, dim in enumerate(control_values)
            )
            reshape.meta["dla_lstm_sequence_layout"] = source, resolved


INDEXING_RULES = (GatherIndices, GatherElementsIndices)
RECURRENT_RULES = (Lstm1, Lstm2, Lstm3, LstmSequenceLayout)
MATRIX_RULES = (MatMulToConv, GemmToConv, GemmBiasToConv)
LOGICAL_RULES = (WhereZero, LogicalNot, GreaterFloat)
