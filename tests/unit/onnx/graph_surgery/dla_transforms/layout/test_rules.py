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

"""Check fixed layout patterns against native shapes and coordinated 4D plans."""

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator
from onnxscript import ir

from modelopt.onnx.graph_surgery.dla_transforms.common import static_shape
from modelopt.onnx.graph_surgery.dla_transforms.context import DlaGraph
from modelopt.onnx.graph_surgery.dla_transforms.layout.rules import SHAPE_RULES


def _apply_rules(model, *, native_shapes, planned_shapes, group, force_fp16_activations=False):
    graph = DlaGraph(model, force_fp16_activations=force_fp16_activations)
    graph.native_shapes = native_shapes
    graph.planned_shapes = planned_shapes
    if group is SHAPE_RULES:
        controls = {}
        rules = [rule.rule(graph, controls) for rule in group]
    else:
        rules = [rule.rule(graph) for rule in group]
    count = graph.apply_patterns(rules)
    return (graph, rules, count)


def _lower_shapes(model, planned):
    working = ir.serde.deserialize_model(onnx.shape_inference.infer_shapes(model, strict_mode=True))
    native = {
        n: (tuple(static_shape(v) for v in n.inputs), tuple(static_shape(v) for v in n.outputs))
        for n in working.graph
    }
    _, _, count = _apply_rules(
        working, native_shapes=native, planned_shapes=planned, group=SHAPE_RULES
    )
    value = working.graph.inputs[0]
    value.shape = ir.Shape((1,) * (4 - len(value.shape)) + tuple(value.shape))
    for node in working.graph:
        for value in node.outputs:
            if value.name in planned:
                value.shape = ir.Shape(planned[value.name])
    ir.passes.common.RemoveUnusedNodesPass()(working)
    result = onnx.shape_inference.infer_shapes(ir.serde.serialize_model(working), strict_mode=True)
    onnx.checker.check_model(result, full_check=True)
    return result, count


@pytest.mark.parametrize(
    ("kind", "shape", "controls", "attrs"),
    [
        ("Reshape", [2, 3], [[3, 2]], {}),
        ("Tile", [2, 3], [[1, 2]], {}),
        ("Expand", [1, 3], [[2, 3]], {}),
        ("Transpose", [2, 3], [], {"perm": [1, 0]}),
        ("Transpose", [2, 3], [], {}),
        ("Flatten", [2, 3, 4], [], {"axis": 2}),
        ("ArgMax", [2, 3, 4], [], {"axis": 1, "keepdims": 0}),
        ("Slice", [2, 3, 4], [[0], [1]], {}),
        ("Slice", [2, 3, 4], [[0], [4], [2], [2]], {}),
    ],
)
def test_shape_controls_and_native_shapes(kind, shape, controls, attrs):
    """Every shape family fires and retains native values under leading-one promotion."""
    inits = [
        numpy_helper.from_array(np.asarray(a, np.int64), f"control{i}")
        for i, a in enumerate(controls)
    ]
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node(kind, ["x", *[t.name for t in inits]], ["y"], **attrs)],
            "shape_rule",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
            [
                helper.make_tensor_value_info(
                    "y", TensorProto.INT64 if kind == "ArgMax" else TensorProto.FLOAT, None
                )
            ],
            inits,
        ),
        opset_imports=[helper.make_opsetid("", 21)],
    )
    values = (np.arange(np.prod(shape)).reshape(shape) % 7 - 3).astype(np.float32)
    expected = ReferenceEvaluator(model).run(None, {"x": values})[0]
    target = (1,) * (4 - expected.ndim) + expected.shape
    transformed, count = _lower_shapes(model, {"y": target})
    assert count == 1
    actual = ReferenceEvaluator(transformed).run(
        None, {"x": values.reshape((1,) * (4 - values.ndim) + values.shape)}
    )[0]
    np.testing.assert_array_equal(actual, expected.reshape(target))


def test_five_dimensional_reshape_transpose_uses_original_snapshot():
    """A later handler retains original shapes after the preceding Reshape is replaced."""
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Reshape", ["x", "target"], ["reshaped"]),
                helper.make_node("Transpose", ["reshaped"], ["y"], perm=[0, 2, 1, 3, 4]),
            ],
            "collapsed_layout",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 4])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
            [numpy_helper.from_array(np.array([2, 1, 3, 2, 2], np.int64), "target")],
        ),
        opset_imports=[helper.make_opsetid("", 21)],
    )
    transformed, count = _lower_shapes(model, {"reshaped": (2, 3, 2, 2), "y": (2, 3, 2, 2)})
    assert count == 2
    feeds = {"x": np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4)}
    expected = ReferenceEvaluator(model).run(None, feeds)[0].reshape(2, 3, 2, 2)
    np.testing.assert_array_equal(ReferenceEvaluator(transformed).run(None, feeds)[0], expected)
