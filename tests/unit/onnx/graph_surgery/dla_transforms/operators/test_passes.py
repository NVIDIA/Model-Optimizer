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

"""Check schema-driven private float consumer views."""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator
from onnxscript import ir

from modelopt.onnx.graph_surgery.dla_transforms.context import DlaGraph
from modelopt.onnx.graph_surgery.dla_transforms.operators import FloatConsumerViews


def _model(nodes, inputs, outputs, initializers=()):
    return helper.make_model(
        helper.make_graph(nodes, "logical", inputs, outputs, initializers),
        opset_imports=[helper.make_opsetid("", 21)],
    )


def test_unsqueeze_float_consumer_view():
    """A FLOAT-bound Mul gets a cast while the exposed integer Unsqueeze stays INT64."""
    model = _model(
        [
            helper.make_node("Unsqueeze", ["x", "axes"], ["expanded"]),
            helper.make_node("Mul", ["expanded", "weight"], ["y"]),
            helper.make_node("Add", ["expanded", "weight"], ["sum"]),
            helper.make_node("Add", ["expanded", "integer"], ["native"]),
        ],
        [helper.make_tensor_value_info("x", TensorProto.INT64, [2])],
        [
            helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2]),
            helper.make_tensor_value_info("expanded", TensorProto.INT64, [1, 2]),
            helper.make_tensor_value_info("sum", TensorProto.FLOAT, [1, 2]),
            helper.make_tensor_value_info("native", TensorProto.INT64, [1, 2]),
        ],
        [
            numpy_helper.from_array(np.array([0], np.int64), "axes"),
            numpy_helper.from_array(np.array([[0.5, 0.25]], np.float32), "weight"),
            numpy_helper.from_array(np.array([[1, 2]], np.int64), "integer"),
        ],
    )
    # This stage repairs the binding after integer-producing layout preparation.
    working = ir.serde.deserialize_model(model)
    transform = FloatConsumerViews()
    assert DlaGraph(working).apply_pass(transform).modified
    assert transform.transformed_nodes == 2
    assert sum(node.op_type == "Cast" for node in working.graph) == 1
    transformed = onnx.shape_inference.infer_shapes(ir.serde.serialize_model(working))
    onnx.checker.check_model(transformed, full_check=True)
    feeds = {"x": np.array([2, 4], np.int64)}
    result = ReferenceEvaluator(transformed).run(None, feeds)
    np.testing.assert_array_equal(result[0], np.array([[1, 1]], np.float32))
    np.testing.assert_array_equal(result[1], np.array([[2, 4]], np.int64))
    np.testing.assert_array_equal(result[2], np.array([[2.5, 4.25]], np.float32))
    np.testing.assert_array_equal(result[3], np.array([[3, 6]], np.int64))


def test_unknown_consumer_schema_keeps_integer_view():
    """A standalone pass leaves an unknown operator's input ownership unchanged."""
    working = ir.serde.deserialize_model(
        _model(
            [helper.make_node("FutureOperator", ["x"], ["y"])],
            [helper.make_tensor_value_info("x", TensorProto.INT64, [2])],
            [helper.make_tensor_value_info("y", TensorProto.INT64, [2])],
        )
    )
    source = working.graph.inputs[0]
    transform = FloatConsumerViews()
    assert not DlaGraph(working).apply_pass(transform).modified
    assert transform.transformed_nodes == 0
    assert working.graph[0].inputs[0] is source
