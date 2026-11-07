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
"""Fixed rewrite rules adapting native data operations to planned 4D layouts."""

import math
from typing import TYPE_CHECKING

import numpy as np
from onnx import defs
from onnxscript import ir
from onnxscript.rewriter import pattern

from ..common import constant_array, static_shape
from .passes import _axis_output_shapes, _mapped_axis

if TYPE_CHECKING:
    from ..context import DlaGraph

__all__ = [
    "AXIS_RULES",
    "INDEXING_RULES",
    "NATIVE_RULES",
    "NORMALIZATION_RULES",
    "RANK_ADAPTER_RULES",
    "SHAPE_RULES",
    "ArgMaxLayout",
    "ConcatAxisLayout",
    "ContribDequantizeLinearAxisLayout",
    "ContribQuantizeLinearAxisLayout",
    "DequantizeLinearAxisLayout",
    "ExpandLayout",
    "FlattenLayout",
    "GatherElementsLayout",
    "GatherLayout",
    "GroupQueryAttentionLayout1",
    "GroupQueryAttentionLayout2",
    "GroupQueryAttentionLayout3",
    "InstanceNormalizationLayout",
    "LogSoftmaxAxisLayout",
    "LpNormalizationLayout",
    "QuantizeLinearAxisLayout",
    "ReduceMaxLayout",
    "ReduceMeanLayout",
    "ReduceMinLayout",
    "ReduceSumLayout",
    "ReshapeLayout",
    "SliceLayout",
    "SoftmaxAxisLayout",
    "SqueezeLayout",
    "TileLayout",
    "TopKAxisLayout",
    "TransposeLayout",
    "UnsqueezeLayout",
]


class _LayoutRule(pattern.RewriteRuleClassBase):
    def __init__(self, graph: "DlaGraph"):
        super().__init__(name=getattr(self, "rule_name", type(self).__name__))
        self.graph = graph

    def _planned(self, node, result):
        return node in self.graph.native_shapes and result.name in self.graph.planned_shapes

    def _reshape(self, op, value, dims, label):
        if dims is None:
            raise ValueError("TODO: provide static dimensions for native layout adapters")
        target = self.graph.add_initializer(op, np.array(dims, np.int64), label)
        result = op.Reshape(value, target)
        result.type, result.shape = value.type, ir.Shape(dims)
        return result


class _ShapeRule(_LayoutRule):
    def __init__(self, graph: "DlaGraph", controls: dict):
        super().__init__(graph)
        self.controls = controls

    def _control(self, op, source, values, label, dtype=np.int64):
        array = np.asarray(values, dtype=dtype)
        owner = source.meta.get("dla_shape_source", source) if source is not None else None
        key = (owner, array.dtype, tuple(array.shape), tuple(array.reshape(-1)))
        if key not in self.controls:
            self.controls[key] = self.graph.add_initializer(op, array, label)
        return self.controls[key]

    def _shapes(self, node):
        inputs, outputs = self.graph.native_shapes[node]
        shape, output = inputs[0], outputs[0]
        if shape is None or output is None:
            raise ValueError(f"{node.op_type}: static shapes need an additional transform")
        return shape, output


class _ShapeControl(_ShapeRule):
    def check(self, context, result, **_):
        node = context.root
        if not self._planned(node, result):
            return False
        _, output = self._shapes(node)
        fixed = constant_array(node.inputs[1])
        if fixed is None:
            raise ValueError(f"{node.op_type}: dynamic shape controls need an additional transform")
        values = fixed.reshape(-1).tolist()
        if node.op_type == "Reshape":
            values = [output[i] if value == 0 else value for i, value in enumerate(values)]
            if len(values) == 5:
                if 1 not in output:
                    raise ValueError("Reshape: 5D targets need a unary dimension")
                values.pop(output.index(1))
        if len(values) > 4:
            raise ValueError(
                f"{node.op_type}: targets above four dimensions need an additional transform"
            )
        self.values = [1] * (4 - len(values)) + values
        return self.values != fixed.reshape(-1).tolist()

    def rewrite(self, op, result, **_):
        node = result.producer()
        operands = list(node.inputs)
        operands[1] = self._control(op, operands[1], self.values, "layout_shape")
        return op.call_op(node.op_type, operands, dict(node.attributes))


class ReshapeLayout(_ShapeControl):
    """Promote fixed Reshape targets to 4D, resolving copied zero dimensions.

    For example [0,-1] becomes [1,1,N,-1]. A five-dimensional target may
    collapse one singleton; runtime shape controls are outside this rule's scope.
    """

    def pattern(self, op, x):
        """Capture a standard Reshape and its original output."""
        return op.Reshape(x, _allow_other_inputs=True, _outputs=["result"])


class ExpandLayout(_ShapeControl):
    """Promote a fixed Expand target with leading ones, e.g. [N,C] to [1,1,N,C]."""

    def pattern(self, op, x):
        """Capture Expand before its data and target are promoted."""
        return op.Expand(x, _allow_other_inputs=True, _outputs=["result"])


class TileLayout(_ShapeControl):
    """Pad fixed Tile repetitions with ones to match the promoted data rank."""

    def pattern(self, op, x):
        """Capture Tile with all original repetition controls."""
        return op.Tile(x, _allow_other_inputs=True, _outputs=["result"])


class TransposeLayout(_ShapeRule):
    """Remap Transpose permutations for leading-one promotion or singleton collapse.

    A rank-two [1,0] permutation becomes [0,1,3,2]. Five-dimensional input
    requires one unambiguous singleton axis that can be removed from its permutation.
    """

    def pattern(self, op, x):
        """Capture a native Transpose, including an implicit reverse permutation."""
        return op.Transpose(x, _outputs=["result"])

    def check(self, context, result, **_):
        """Validate the saved rank and prepare its promoted permutation."""
        node = context.root
        if not self._planned(node, result):
            return False
        shape, _ = self._shapes(node)
        if len(shape) == 4:
            return False
        perm = node.attributes.get_ints("perm", list(reversed(range(len(shape)))))
        if len(shape) == 5:
            if shape.count(1) != 1:
                raise ValueError("Transpose: ambiguous 5D collapse needs an additional transform")
            removed = shape.index(1)
            self.perm = [axis - (axis > removed) for axis in perm if axis != removed]
        else:
            delta = max(0, 4 - len(shape))
            self.perm = [*range(delta), *(axis + delta for axis in perm)]
        return True

    def rewrite(self, op, result, **_):
        """Emit Transpose with the remapped permutation and original attributes."""
        node = result.producer()
        attrs = dict(node.attributes)
        attrs["perm"] = self.perm
        return op.Transpose(*node.inputs, **attrs)


class FlattenLayout(_ShapeRule):
    """Replace Flatten with a fixed 4D Reshape preserving its two logical dimensions.

    For example Flatten([N,C,H], axis=2) becomes Reshape([1,1,N*C,-1]).
    """

    def pattern(self, op, x):
        """Capture a native Flatten and its logical two-dimensional result."""
        return op.Flatten(x, _outputs=["result"])

    def check(self, context, result, **_):
        """Validate the native flattening axis and compute its fixed target."""
        node = context.root
        if not self._planned(node, result):
            return False
        shape, _ = self._shapes(node)
        axis = node.attributes.get_int("axis", 1)
        if not -len(shape) <= axis <= len(shape):
            raise ValueError("Flatten: invalid native axis")
        if axis < 0:
            axis += len(shape)
        self.values = [1, 1, math.prod(shape[:axis]), -1]
        return True

    def rewrite(self, op, result, **_):
        """Replace Flatten with the equivalent planned 4D Reshape."""
        return op.Reshape(
            result.producer().inputs[0], self._control(op, None, self.values, "flatten_shape")
        )


class ArgMaxLayout(_ShapeRule):
    """Map ArgMax's axis to 4D and restore a planned view when keepdims is false.

    The reduction keeps its axis internally; a trailing Reshape reproduces the
    original logical output with leading singleton promotion.
    """

    def pattern(self, op, x):
        """Capture ArgMax before axis and output-rank adaptation."""
        return op.ArgMax(x, _outputs=["result"])

    def check(self, context, result, **_):
        """Select planned ArgMax nodes requiring promotion or output restoration."""
        node = context.root
        if not self._planned(node, result):
            return False
        shape, _ = self._shapes(node)
        if len(shape) > 4:
            raise ValueError("ArgMax: collapsed 5D axes need an additional transform")
        return len(shape) != 4 or not node.attributes.get_int("keepdims", 1)

    def rewrite(self, op, result, **_):
        """Map the reduction axis and restore its planned output view."""
        node = result.producer()
        shape, _ = self._shapes(node)
        attrs = dict(node.attributes)
        axis = node.attributes.get_int("axis", 0)
        attrs["axis"] = axis + 4 - len(shape) if axis >= 0 else axis
        restore = node.attributes.get_int("keepdims", 1) == 0
        if restore:
            attrs["keepdims"] = 1
        value = op.ArgMax(*node.inputs, **attrs)
        if restore:
            current = [1] * (4 - len(shape)) + list(shape)
            current[int(attrs["axis"]) % len(current)] = 1
            value.shape, value.type = ir.Shape(current), result.type
            value = op.Reshape(
                value,
                self._control(op, None, self.graph.planned_shapes[result.name], "argmax_shape"),
            )
        return value


class SliceLayout(_ShapeRule):
    """Shift nonnegative Slice axes for leading-one promotion; retain starts/ends/steps.

    Implicit axes are made explicit from the static starts length. Existing
    negative axes already refer to trailing dimensions and need no adjustment.
    """

    def pattern(self, op, x):
        """Capture Slice while retaining all optional operands."""
        return op.Slice(x, _allow_other_inputs=True, _outputs=["result"])

    def check(self, context, result, **_):
        """Resolve static axes and shift nonnegative references into the 4D view."""
        node = context.root
        if not self._planned(node, result):
            return False
        shape, _ = self._shapes(node)
        if len(shape) > 4:
            raise ValueError("Slice: collapsed 5D axes need an additional transform")
        delta = 4 - len(shape)
        if not delta:
            return False
        if len(node.inputs) > 3 and node.inputs[3] is not None:
            fixed = constant_array(node.inputs[3])
            if fixed is None:
                raise ValueError("Slice: runtime axes need an additional transform")
            values = fixed.reshape(-1).tolist()
            if all(axis < 0 for axis in values):
                return False
        else:
            starts_shape = self.graph.native_shapes[node][0][1]
            if starts_shape is None:
                raise ValueError("Slice: implicit axes need a known starts length")
            values = list(range(math.prod(starts_shape)))
        self.values = [axis + delta if axis >= 0 else axis for axis in values]
        return True

    def rewrite(self, op, result, **_):
        """Replace only the axes control, preserving slicing bounds and steps."""
        node = result.producer()
        operands = list(node.inputs)
        source = operands[3] if len(operands) > 3 else operands[1]
        axes = self._control(op, source, self.values, "slice_axes", dtype=operands[1].dtype.numpy())
        if len(operands) > 3:
            operands[3] = axes
        else:
            operands.append(axes)
        return op.Slice(*operands, **dict(node.attributes))


SHAPE_RULES = (
    ArgMaxLayout,
    ExpandLayout,
    FlattenLayout,
    ReshapeLayout,
    SliceLayout,
    TileLayout,
    TransposeLayout,
)


class _IndexingRule(_LayoutRule):
    def __init__(self, graph: "DlaGraph", controls: dict, views: dict):
        super().__init__(graph)
        self.controls = controls
        self.views = views

    def _control(self, op, source, values, label):
        array = np.asarray(values, dtype=np.int64)
        key = (source, tuple(array.shape), tuple(array.reshape(-1)))
        if key not in self.controls:
            self.controls[key] = self.graph.add_initializer(op, array, label)
        return self.controls[key]

    def _source_shape(self, node):
        shape = self.graph.native_shapes[node][0][0]
        if shape is None or len(shape) > 4:
            raise ValueError(
                f"{node.op_type}: a static data rank up to four needs an additional transform"
            )
        return shape

    def _index_view(self, op, source, native, kind):
        desired = (
            (math.prod(native),) if kind == "Gather" else (1,) * (4 - len(native)) + tuple(native)
        )
        promoted = source.name in self.graph.planned_shapes
        if tuple(native) == desired and not (kind == "Gather" and promoted):
            return source
        if kind == "GatherElements" and promoted:
            return source
        key = (source, desired)
        if key not in self.views:
            array = constant_array(source)
            if array is not None:
                self.views[key] = self.graph.add_initializer(
                    op, array.reshape(desired).copy(), "layout_indices"
                )
            elif kind == "Gather":
                self.views[key] = op.Reshape(source, self._control(op, None, [-1], "gather_shape"))
            else:
                self.views[key] = op.Unsqueeze(
                    source, self._control(op, None, list(range(4 - len(native))), "index_axes")
                )
            self.views[key].shape = ir.Shape(desired)
            self.views[key].type = source.type
        return self.views[key]


class _GatherLayout(_IndexingRule):
    def check(self, context, result, **_):
        node = context.root
        if not self._planned(node, result):
            return False
        shape = self._source_shape(node)
        indices = self.graph.native_shapes[node][0][1]
        if indices is None:
            raise ValueError(f"{node.op_type}: a static index shape needs an additional transform")
        axis = node.attributes.get_int("axis", 0)
        if not -len(shape) <= axis < len(shape):
            raise ValueError(f"{node.op_type}: invalid native axis")
        delta = 4 - len(shape)
        if node.op_type == "Gather":
            data_shape = [1] * delta + list(shape)
            axis4 = axis % len(shape) + delta
            gathered = (*data_shape[:axis4], math.prod(indices), *data_shape[axis4 + 1 :])
            if len(indices) != 1 and tuple(self.graph.planned_shapes[result.name]) != gathered:
                restored = result.uses() and all(
                    port == 0
                    and user.op_type == "Reshape"
                    and not user.domain
                    and constant_array(user.inputs[1]) is not None
                    for user, port in result.uses()
                )
                if result.is_graph_output() or not restored:
                    raise ValueError(
                        "Gather: index flattening changes the canonical layout; need an additional transform"
                    )
                self.graph.planned_shapes[result.name] = gathered
        elif len(indices) != len(shape):
            raise ValueError("GatherElements: native data and indices require matching ranks")
        elif not delta:
            return False
        return True

    def rewrite(self, op, result, **_):
        node = result.producer()
        inputs, _ = self.graph.native_shapes[node]
        axis = node.attributes.get_int("axis", 0)
        attrs = dict(node.attributes)
        attrs["axis"] = axis + 4 - len(inputs[0]) if axis >= 0 else axis
        operands = list(node.inputs)
        operands[1] = self._index_view(op, operands[1], inputs[1], node.op_type)
        return op.call_op(node.op_type, operands, attrs)


class GatherLayout(_GatherLayout):
    """Map Gather axes to 4D and flatten scalar/multidimensional indices privately.

    A noncanonical flattened result is accepted only when fixed private Reshapes
    restore its logical layout; public results must already match the 4D plan.
    """

    def pattern(self, op, x):
        """Capture Gather and its native index input."""
        return op.Gather(x, _allow_other_inputs=True, _outputs=["result"])


class GatherElementsLayout(_GatherLayout):
    """Promote GatherElements indices to the same 4D rank as their data input.

    Constant indices use a shared initializer view; runtime indices use one
    Unsqueeze unless another data consumer already requires their promoted view.
    """

    def pattern(self, op, x):
        """Capture GatherElements before matching index and data ranks."""
        return op.GatherElements(x, _allow_other_inputs=True, _outputs=["result"])


class _ReductionLayout(_IndexingRule):
    def check(self, context, result, **_):
        node = context.root
        if not self._planned(node, result):
            return False
        shape = self._source_shape(node)
        delta = 4 - len(shape)
        if not delta and node.attributes.get_int("keepdims", 1):
            return False
        values = None
        if len(node.inputs) > 1 and node.inputs[1] is not None:
            fixed = constant_array(node.inputs[1])
            if fixed is None:
                raise ValueError(
                    f"{node.op_type}: runtime reduction axes need an additional transform"
                )
            values = fixed.reshape(-1).tolist()
        elif "axes" in node.attributes:
            values = list(node.attributes.get_ints("axes"))
        if values == [] and node.attributes.get_int("noop_with_empty_axes", 0):
            return False
        self.values = (
            [axis + delta if axis >= 0 else axis for axis in values] if values is not None else None
        )
        return True

    def rewrite(self, op, result, **_):
        node = result.producer()
        shape = self._source_shape(node)
        attrs, operands = dict(node.attributes), list(node.inputs)
        if self.values is not None and "axes" in attrs:
            attrs["axes"] = self.values
        restore = node.attributes.get_int("keepdims", 1) == 0
        if restore:
            attrs["keepdims"] = 1
        if len(operands) > 1 and operands[1] is not None:
            operands[1] = self.graph.add_initializer(
                op, np.asarray(self.values, np.int64), "reduction_axes"
            )
        value = op.call_op(node.op_type, operands, attrs)
        if restore:
            current = [1] * (4 - len(shape)) + list(shape)
            for axis in self.values or range(4):
                current[axis] = 1
            value.shape, value.type = ir.Shape(current), result.type
            value = op.Reshape(
                value,
                self._control(op, None, self.graph.planned_shapes[result.name], "reduction_shape"),
            )
        return value


class ReduceMaxLayout(_ReductionLayout):
    """Shift fixed ReduceMax axes and restore planned rank after keepdims=0."""

    def pattern(self, op, x):
        """Capture ReduceMax with either supported axes representation."""
        return op.ReduceMax(x, _allow_other_inputs=True, _outputs=["result"])


class ReduceMeanLayout(_ReductionLayout):
    """Shift fixed ReduceMean axes while retaining both attribute/input schema forms."""

    def pattern(self, op, x):
        """Capture ReduceMean while retaining optional axes inputs."""
        return op.ReduceMean(x, _allow_other_inputs=True, _outputs=["result"])


class ReduceMinLayout(_ReductionLayout):
    """Keep ReduceMin's logical result when its input is promoted to 4D."""

    def pattern(self, op, x):
        """Capture ReduceMin and its original output view."""
        return op.ReduceMin(x, _allow_other_inputs=True, _outputs=["result"])


class ReduceSumLayout(_ReductionLayout):
    """Shift fixed ReduceSum axes, using keepdims plus Reshape for removed axes.

    For example rank-three axis 1 becomes axis 2 in [1,N,C,H]. Empty axes
    with noop_with_empty_axes=1 retain their existing no-op semantics.
    """

    def pattern(self, op, x):
        """Capture ReduceSum with optional axes input and attributes."""
        return op.ReduceSum(x, _allow_other_inputs=True, _outputs=["result"])


INDEXING_RULES = (
    GatherLayout,
    GatherElementsLayout,
    ReduceMaxLayout,
    ReduceMeanLayout,
    ReduceMinLayout,
    ReduceSumLayout,
)


class _AxisLayout(_LayoutRule):
    output_count = 1

    def check(self, context, result0, **_):
        node = context.root
        if node not in self.graph.native_shapes or len(node.outputs) != self.output_count:
            return False
        if node.domain and self.graph.model.graph.opset_imports.get(node.domain) != 1:
            return False
        target = self.graph.planned_shapes.get(node.inputs[0].name)
        if target is None or not any(
            value.name in self.graph.planned_shapes for value in node.outputs
        ):
            return False
        quantizer = node.op_type in {"QuantizeLinear", "DequantizeLinear"}
        attr = node.attributes.get("axis")
        if attr is None and quantizer and static_shape(node.inputs[1]) == ():
            return False
        if attr is not None:
            axis = int(attr.value)
        elif node.domain:
            axis = 1
        else:
            schema = defs.get_schema(node.op_type, self.graph.model.graph.opset_imports[""])
            default = schema.attributes.get("axis")
            if default is None or not default.default_value.type:
                raise ValueError(f"{node.op_type} requires an axis attribute")
            axis = int(default.default_value.i)
        self.axis = _mapped_axis(axis, self.graph.native_shapes[node][0][0], tuple(target))
        self.output_shapes, self.output_targets = _axis_output_shapes(
            node,
            self.graph.native_shapes[node][1],
            target,
            axis,
            self.axis,
            self.graph.planned_shapes,
        )
        return self.axis != axis or self.output_shapes != self.output_targets

    def rewrite(self, op, result0, **_):
        node = result0.producer()
        attrs = dict(node.attributes)
        attrs["axis"] = self.axis
        outputs = op.call_op(
            node.op_type, list(node.inputs), attrs, domain=node.domain, outputs=self.output_count
        )
        outputs = [outputs] if self.output_count == 1 else outputs
        outputs[0].producer().meta["dla_native_inputs"] = self.graph.native_shapes[node][0]
        results = []
        for result, original, dims, target in zip(
            outputs, node.outputs, self.output_shapes, self.output_targets, strict=True
        ):
            result.type, result.shape = original.type, ir.Shape(dims)
            if dims != target:
                result = op.Reshape(
                    result,
                    self.graph.add_initializer(op, np.array(target, np.int64), "axis_output_shape"),
                )
                result.type = original.type
            # SDK replacement carries terminal metadata from the matched values.
            original.shape = result.shape = ir.Shape(target)
            results.append(result)
        return results[0] if self.output_count == 1 else results


class SoftmaxAxisLayout(_AxisLayout):
    """Map Softmax's native axis into its promoted or singleton-collapsed data view."""

    rule_name = "SoftmaxAxis1"

    def pattern(self, op, x):
        """Capture standard Softmax without assuming an explicit axis."""
        return op.Softmax(x, _outputs=["result0"])


class LogSoftmaxAxisLayout(_AxisLayout):
    """Map LogSoftmax axes; negative trailing-axis references remain valid after promotion."""

    rule_name = "LogSoftmaxAxis1"

    def pattern(self, op, x):
        """Capture standard LogSoftmax and its original axis semantics."""
        return op.LogSoftmax(x, _outputs=["result0"])


class ConcatAxisLayout(_AxisLayout):
    """Retain every Concat operand while mapping its shared native axis to 4D."""

    rule_name = "ConcatAxis1"

    def pattern(self, op, x):
        """Capture Concat while allowing any number of data operands."""
        return op.Concat(x, _allow_other_inputs=True, _outputs=["result0"])


class TopKAxisLayout(_AxisLayout):
    """Map TopK's axis and preserve both values and indices outputs and their dtypes."""

    output_count = 2
    rule_name = "TopKAxis2"

    def pattern(self, op, x):
        """Capture TopK together with both declared outputs."""
        return op.TopK(x, _allow_other_inputs=True, _outputs=["result0", "result1"])


class QuantizeLinearAxisLayout(_AxisLayout):
    """Map standard per-axis quantization while leaving scalar quantization unchanged."""

    rule_name = "QuantizeLinearAxis1"

    def pattern(self, op, x):
        """Capture standard quantization with all optional controls."""
        return op.QuantizeLinear(x, _allow_other_inputs=True, _outputs=["result0"])


class DequantizeLinearAxisLayout(_AxisLayout):
    """Map standard per-axis dequantization without changing scales or zero points."""

    rule_name = "DequantizeLinearAxis1"

    def pattern(self, op, x):
        """Capture standard dequantization with all optional controls."""
        return op.DequantizeLinear(x, _allow_other_inputs=True, _outputs=["result0"])


class ContribQuantizeLinearAxisLayout(_AxisLayout):
    """Map com.microsoft version-1 per-axis quantization using its native axis default."""

    rule_name = "QuantizeLinearAxis1"

    def pattern(self, op, x):
        """Capture contrib quantization independently of standard-domain rules."""
        return op.QuantizeLinear(
            x, _domain="com.microsoft", _allow_other_inputs=True, _outputs=["result0"]
        )


class ContribDequantizeLinearAxisLayout(_AxisLayout):
    """Map com.microsoft version-1 dequantization while preserving its contrib domain."""

    rule_name = "DequantizeLinearAxis1"

    def pattern(self, op, x):
        """Capture contrib dequantization independently of standard-domain rules."""
        return op.DequantizeLinear(
            x, _domain="com.microsoft", _allow_other_inputs=True, _outputs=["result0"]
        )


AXIS_RULES = (
    ConcatAxisLayout,
    DequantizeLinearAxisLayout,
    ContribDequantizeLinearAxisLayout,
    LogSoftmaxAxisLayout,
    QuantizeLinearAxisLayout,
    ContribQuantizeLinearAxisLayout,
    SoftmaxAxisLayout,
    TopKAxisLayout,
)


class LpNormalizationLayout(_LayoutRule):
    """Map LpNormalization axes and optionally lower scaled last-axis L2 for FP16.

    The FP16 form is x / Max(Sqrt(ReduceSum(x*x)), 2**-14), only for p=2
    followed by one constant-scaled Mul. Existing square/sum overflow behavior
    and the minimum normal FP16 denominator are retained.
    """

    rule_name = "LpNormalization4D"

    def pattern(self, op, x):
        """Capture standard LpNormalization and its consumer-visible output."""
        return op.LpNormalization(x, _outputs=["result"])

    def check(self, context, result, **_):
        """Choose axis remapping or the existing supported FP16 L2 decomposition."""
        node = context.root
        if node not in self.graph.native_shapes:
            return False
        axis = node.attributes.get_int("axis", -1)
        source = self.graph.native_shapes[node][0][0]
        promoted = self.graph.planned_shapes.get(node.inputs[0].name, source)
        if source is None or promoted is None:
            raise ValueError("TODO: provide static LpNormalization data dimensions")
        mapped = _mapped_axis(axis, source, tuple(promoted))
        if (
            len(source) < 4
            and node.inputs[0].name in self.graph.planned_shapes
            and axis >= 0
            and mapped == len(promoted) - 1
        ):
            mapped = -1
        uses = node.outputs[0].uses()
        scaled = False
        if len(uses) == 1:
            user, _ = next(iter(uses))
            scaled = (
                user.op_type == "Mul"
                and not user.domain
                and any(
                    value is not node.outputs[0] and constant_array(value) is not None
                    for value in user.inputs
                )
            )
        self.decompose = (
            self.graph.force_fp16_activations
            and node.attributes.get_int("p", 2) == 2
            and axis % len(source) == len(source) - 1
            and scaled
        )
        self.axis, self.target_shape = -1 if self.decompose else mapped, promoted
        return self.decompose or mapped != axis

    def rewrite(self, op, x, result):
        """Emit remapped normalization or its explicit FP16 arithmetic graph."""
        node = result.producer()
        if not self.decompose:
            attrs = dict(node.attributes)
            attrs["axis"] = self.axis
            output = op.LpNormalization(x, **attrs)
            output.type, output.shape = x.type, ir.Shape(self.target_shape)
            return output

        def emit(operator, inputs, shape, **attrs):
            value = operator(*inputs, **attrs)
            value.type, value.shape = x.type, ir.Shape(shape)
            return value

        squared = emit(op.Mul, [x, x], self.target_shape)
        axes = self.graph.add_initializer(op, np.array([self.axis], np.int64), "l2_axes")
        reduced_shape = list(self.target_shape)
        reduced_shape[self.axis] = 1
        summed = emit(op.ReduceSum, [squared, axes], reduced_shape, keepdims=1)
        norm = emit(op.Sqrt, [summed], reduced_shape)
        floor = self.graph.add_initializer(op, np.array(2**-14, np.float32), "l2_floor")
        safe = emit(op.Max, [norm, floor], reduced_shape)
        return emit(op.Div, [x, safe], self.target_shape)


NORMALIZATION_RULES = (LpNormalizationLayout,)


class InstanceNormalizationLayout(_LayoutRule):
    """Preserve rank-three InstanceNormalization channel semantics in a 4D graph.

    [N,C,L] is normalized through [N,C,L,1], then reshaped to its planned
    [1,N,C,L] view. Leading-one promotion alone would move the channel axis.
    """

    rule_name = "InstanceNormalizationNative1"

    def pattern(self, op, x):
        """Capture InstanceNormalization without binding its scale or bias."""
        return op.InstanceNormalization(x, _allow_other_inputs=True, _outputs=["result0"])

    def check(self, context, **_):
        """Select rank-three normalization whose data participates in the 4D plan."""
        node = context.root
        if node not in self.graph.native_shapes:
            return False
        shape = self.graph.native_shapes[node][0][0]
        if shape is None:
            raise ValueError("TODO: support symbolic InstanceNormalization dimensions")
        if len(shape) not in (3, 4):
            raise ValueError("TODO: support InstanceNormalization outside ranks three and four")
        return len(shape) == 3 and (
            node.inputs[0].name in self.graph.planned_shapes
            or node.outputs[0].name in self.graph.planned_shapes
        )

    def rewrite(self, op, result0, **_):
        """Normalize through the native-channel spatial view and restore its output."""
        node = result0.producer()
        inputs, outputs = self.graph.native_shapes[node]
        args = list(node.inputs)
        spatial = [*inputs[0], 1]
        args[0] = self._reshape(op, args[0], spatial, "instance_spatial_shape")
        result = op.InstanceNormalization(*args, **dict(node.attributes))
        result.type, result.shape = node.outputs[0].type, ir.Shape(spatial)
        output = self.graph.planned_shapes.get(node.outputs[0].name, outputs[0])
        return self._reshape(op, result, output, "instance_output_shape")


class _GroupQueryAttentionLayout(_LayoutRule):
    def check(self, context, **_):
        node = context.root
        if node not in self.graph.native_shapes or len(node.outputs) != self.output_count:
            return False
        if self.graph.model.graph.opset_imports.get("com.microsoft") != 1:
            raise ValueError("TODO: support this GroupQueryAttention contrib version")
        inputs, outputs = self.graph.native_shapes[node]
        return any(
            value is not None
            and value.name in self.graph.planned_shapes
            and tuple(self.graph.planned_shapes[value.name]) != shape
            for value, shape in zip(node.inputs, inputs, strict=True)
        ) or any(
            value.name in self.graph.planned_shapes
            and tuple(self.graph.planned_shapes[value.name]) != shape
            for value, shape in zip(node.outputs, outputs, strict=True)
        )

    def rewrite(self, op, result0, **_):
        node = result0.producer()
        inputs, outputs = self.graph.native_shapes[node]
        args = list(node.inputs)
        for index, (value, source) in enumerate(zip(args, inputs, strict=True)):
            if (
                value is not None
                and value.name in self.graph.planned_shapes
                and tuple(self.graph.planned_shapes[value.name]) != source
            ):
                args[index] = self._reshape(op, value, source, "attention_native_input")
        results = op.GroupQueryAttention(
            *args, _domain="com.microsoft", _outputs=self.output_count, **dict(node.attributes)
        )
        results = [results] if self.output_count == 1 else list(results)
        for index, (result, original, shape) in enumerate(
            zip(results, node.outputs, outputs, strict=True)
        ):
            result.type = original.type
            result.shape = ir.Shape(shape) if shape is not None else original.shape
            target = self.graph.planned_shapes.get(original.name)
            if target is not None and tuple(target) != shape:
                results[index] = self._reshape(op, result, target, "attention_output_shape")
        return results[0] if self.output_count == 1 else results


class GroupQueryAttentionLayout1(_GroupQueryAttentionLayout):
    """Wrap one-output contrib GQA with native-input and planned-output Reshapes.

    A promoted [1,B,S,H] query is restored to [B,S,H] for GQA. Original
    optional operands/attributes survive; emitted GQA nodes are absent from the
    layout-entry snapshot and therefore cannot acquire repeated adapters.
    """

    output_count = 1
    rule_name = "GroupQueryAttentionNative1"

    def pattern(self, op, x):
        """Capture exactly the one-output GQA pattern; check verifies actual arity."""
        return op.GroupQueryAttention(
            x, _domain="com.microsoft", _allow_other_inputs=True, _outputs=["result0"]
        )


class GroupQueryAttentionLayout2(_GroupQueryAttentionLayout):
    """Adapt two-output contrib GQA while preserving each declared output's native shape."""

    output_count = 2
    rule_name = "GroupQueryAttentionNative2"

    def pattern(self, op, x):
        """Capture the two-output GQA variant and retain optional operands."""
        return op.GroupQueryAttention(
            x, _domain="com.microsoft", _allow_other_inputs=True, _outputs=["result0", "result1"]
        )


class GroupQueryAttentionLayout3(_GroupQueryAttentionLayout):
    """Adapt GQA with attention, present-key and present-value outputs independently.

    Only outputs whose 4D plan differs from their native view receive adapters;
    four-dimensional key/value caches retain their shapes and connections.
    """

    output_count = 3
    rule_name = "GroupQueryAttentionNative3"

    def pattern(self, op, x):
        """Capture GQA with attention and both present-cache outputs."""
        return op.GroupQueryAttention(
            x,
            _domain="com.microsoft",
            _allow_other_inputs=True,
            _outputs=["result0", "result1", "result2"],
        )


NATIVE_RULES = (
    GroupQueryAttentionLayout1,
    GroupQueryAttentionLayout2,
    GroupQueryAttentionLayout3,
    InstanceNormalizationLayout,
)


class _RankAdapterLayout(_LayoutRule):
    def check(self, context, source, result, **_):
        return (
            context.root in self.graph.batch_nodes
            and source.name in self.graph.planned_shapes
            and result.name in self.graph.planned_shapes
        )

    def rewrite(self, op, source, result, **_):
        planned = self.graph.planned_shapes
        if planned[source.name] != planned[result.name]:
            shape = self.graph.add_initializer(
                op, np.array(planned[result.name], np.int64), "rank_view"
            )
            return op.Reshape(source, shape)
        return op.Identity(source) if result.is_graph_output() else source


class SqueezeLayout(_RankAdapterLayout):
    """Fold native Squeeze into planned 4D views, preserving public restored shapes.

    Internal redundant adapters disappear or become Reshape. A public Squeeze
    absorbs leading promotion into its axes so the original output rank survives.
    Setup/cleanup preserve original tensor names and input/initializer metadata
    across the whole Squeeze/Unsqueeze SDK batch.
    """

    rule_name = "DlaSqueezeLayout"

    def setup(self):
        """Prepare public Squeeze axes and save metadata for the whole rank batch."""
        self.public_axes = {}
        for node in self.graph.model.graph:
            if (
                node.op_type == "Squeeze"
                and node.outputs[0].is_graph_output()
                and node.inputs[0].name in self.graph.planned_shapes
            ):
                shape = static_shape(node.inputs[0])
                if shape is not None and 0 < len(shape) <= 4:
                    control = node.inputs[1] if len(node.inputs) > 1 else None
                    fixed = constant_array(control)
                    if control is not None and fixed is None:
                        raise ValueError(
                            "TODO: runtime axes for a public Squeeze need an additional transform"
                        )
                    axes = (
                        fixed.reshape(-1).tolist()
                        if fixed is not None
                        else [i for i, dim in enumerate(shape) if dim == 1]
                    )
                    shift = 4 - len(shape)
                    self.graph.planned_shapes.pop(node.outputs[0].name, None)
                    if shift:
                        self.public_axes[node] = [
                            *range(shift),
                            *(axis % len(shape) + shift for axis in axes),
                        ]
        graph = self.graph.model.graph
        self.names = {value: value.name for node in graph for value in node.outputs}
        self.names.update((value, value.name) for value in graph.inputs)
        self.metadata = {
            value: (value.type, value.shape)
            for value in (*graph.inputs, *graph.initializers.values())
        }

    def cleanup(self):
        """Restore original names and source metadata after both rank rules finish."""
        for value, name in self.names.items():
            value.name = name
        for value, (dtype, shape) in self.metadata.items():
            value.type, value.shape = dtype, shape

    def pattern(self, op, source):
        """Capture a standard Squeeze with optional axes input."""
        return op.Squeeze(source, _allow_other_inputs=True, _outputs=["result"])

    def check(self, context, source, result, **_):
        """Select public restoration or an original adapter joining planned data views."""
        return context.root in self.public_axes or super().check(context, source, result)

    def rewrite(self, op, source, result, **_):
        """Absorb public promotion into axes or emit the internal planned view."""
        node = result.producer()
        if node in self.public_axes:
            axes = self.graph.add_initializer(
                op, np.array(self.public_axes[node], np.int64), "output_axes"
            )
            return op.Squeeze(source, axes)
        return super().rewrite(op, source, result)


class UnsqueezeLayout(_RankAdapterLayout):
    """Replace native Unsqueeze with the 4D plan's shared view or necessary Reshape."""

    rule_name = "DlaUnsqueezeLayout"

    def pattern(self, op, source):
        """Capture native Unsqueeze before redundant rank changes are removed."""
        return op.Unsqueeze(source, _allow_other_inputs=True, _outputs=["result"])


RANK_ADAPTER_RULES = (SqueezeLayout, UnsqueezeLayout)
