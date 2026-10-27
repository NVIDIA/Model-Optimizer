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

"""Keep native integer Unsqueeze contracts across data and structural consumers."""

import copy

import numpy as np
import onnx
import pytest
from onnx import TensorProto as T
from onnx import helper as h
from onnx import numpy_helper as n
from onnx.reference import ReferenceEvaluator

from modelopt.onnx.graph_surgery.dla_transforms.dla_unsqueeze import _apply_unsqueeze

ort = pytest.importorskip("onnxruntime")


def _model(operation):
    """Build a checked native integer graph with values too large for FLOAT precision."""
    inputs = [h.make_tensor_value_info("x", T.INT64, [2])]
    initializers = [n.from_array(np.asarray([0], np.int64), "axes")]
    nodes = [h.make_node("Unsqueeze", ["x", "axes"], ["expanded"], name="expand")]
    dtype, shape = T.INT64, [1, 2]
    if operation == "Cast":
        nodes.append(h.make_node("Cast", ["expanded"], ["y"], to=T.FLOAT))
        dtype = T.FLOAT
    elif operation == "Double":
        nodes.append(h.make_node("Cast", ["expanded"], ["y"], to=T.DOUBLE))
        dtype = T.DOUBLE
    elif operation == "Add":
        initializers.append(n.from_array(np.asarray([[1, 2]], np.int64), "other"))
        nodes.append(h.make_node("Add", ["expanded", "other"], ["y"]))
    elif operation == "Shape":
        nodes.append(h.make_node("Shape", ["expanded"], ["y"]))
        shape = [2]
    elif operation == "GatherND":
        initializers.append(n.from_array(np.arange(8, dtype=np.float32), "data"))
        nodes.append(h.make_node("GatherND", ["data", "expanded"], ["y"]))
        inputs[0] = h.make_tensor_value_info("x", T.INT64, [1])
        dtype, shape = T.FLOAT, [1]
    else:
        nodes.append(h.make_node("Identity", ["expanded"], ["y"]))
    model = h.make_model(
        h.make_graph(
            nodes,
            "integer_unsqueeze",
            inputs,
            [h.make_tensor_value_info("y", dtype, shape)],
            initializers,
        ),
        opset_imports=[h.make_opsetid("", 17)],
    )
    model.ir_version = 9
    return onnx.shape_inference.infer_shapes(model, strict_mode=True)


@pytest.mark.parametrize("operation", ["Identity", "Add", "Shape", "Double", "Cast", "GatherND"])
def test_unsqueeze_preserves_native_consumer_type_contracts(operation):
    """Keep integer owners and controls, and let explicit casts perform their own conversion."""
    model = _model(operation)
    data = (
        np.asarray([2**24 + 1, 2**24 + 3], np.int64)
        if operation != "GatherND"
        else np.asarray([3], np.int64)
    )
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    onnx.checker.check_model(model, full_check=True)
    expected = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    ).run(None, {"x": data})[0]
    before = model.SerializeToString()
    changed = _apply_unsqueeze(copy.deepcopy(model))
    assert changed.SerializeToString() == before
    onnx.checker.check_model(changed, full_check=True)
    actual = ort.InferenceSession(
        changed.SerializeToString(), options, providers=["CPUExecutionProvider"]
    ).run(None, {"x": data})[0]
    np.testing.assert_array_equal(actual, expected)


def test_opaque_unsqueeze_producer_stays_native():
    """Preserve an opaque integer function despite its standard-looking operator name."""
    model = _model("Identity")
    model.graph.node[0].domain = "opaque"
    model.functions.append(
        h.make_function(
            "opaque",
            "Unsqueeze",
            ["x", "axes"],
            ["out"],
            [h.make_node("Unsqueeze", ["x", "axes"], ["out"])],
            [h.make_opsetid("", 17)],
        )
    )
    model.opset_import.append(h.make_opsetid("opaque", 1))
    data = np.asarray([2**24 + 1, 2**24 + 3], np.int64)
    expected = ReferenceEvaluator(model).run(None, {"x": data})[0]
    before = model.SerializeToString()
    changed = _apply_unsqueeze(model)
    assert changed.SerializeToString() == before
    onnx.checker.check_model(changed, full_check=True)
    np.testing.assert_array_equal(ReferenceEvaluator(changed).run(None, {"x": data})[0], expected)


@pytest.mark.parametrize("operation", ["ConstantOfShape", "ScatterND"])
def test_unsqueeze_keeps_structural_indices_for_additional_schemas(operation):
    """Derive integer-only ports from schemas beyond the previous manual list."""
    input_shape = [] if operation == "ConstantOfShape" else [1]
    indices = (
        np.asarray(3, np.int64) if operation == "ConstantOfShape" else np.asarray([2], np.int64)
    )
    initializers = [n.from_array(np.asarray([0], np.int64), "axes")]
    nodes = [h.make_node("Unsqueeze", ["x", "axes"], ["expanded"])]
    if operation == "ConstantOfShape":
        nodes.append(h.make_node(operation, ["expanded"], ["y"]))
        output_shape = [3]
    else:
        initializers.extend(
            [
                n.from_array(np.arange(4, dtype=np.float32), "data"),
                n.from_array(np.asarray([99], np.float32), "updates"),
            ]
        )
        nodes.append(h.make_node(operation, ["data", "expanded", "updates"], ["y"]))
        output_shape = [4]
    model = h.make_model(
        h.make_graph(
            nodes,
            "integer_controls",
            [h.make_tensor_value_info("x", T.INT64, input_shape)],
            [h.make_tensor_value_info("y", T.FLOAT, output_shape)],
            initializers,
        ),
        opset_imports=[h.make_opsetid("", 17)],
    )
    model.ir_version = 9
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    before = model.SerializeToString()
    expected = ReferenceEvaluator(model).run(None, {"x": indices})[0]
    changed = _apply_unsqueeze(model)
    assert changed.SerializeToString() == before
    onnx.checker.check_model(changed, full_check=True)
    np.testing.assert_array_equal(
        ReferenceEvaluator(changed).run(None, {"x": indices})[0], expected
    )


def test_unsqueeze_does_not_rewire_opaque_consumer():
    """Keep opaque integer consumers independent of standard schema lookup."""
    model = _model("Identity")
    model.graph.node[1].domain = "opaque"
    model.functions.append(
        h.make_function(
            "opaque",
            "Identity",
            ["x"],
            ["out"],
            [h.make_node("Neg", ["x"], ["out"])],
            [h.make_opsetid("", 17)],
        )
    )
    model.opset_import.append(h.make_opsetid("opaque", 1))
    before = model.SerializeToString()
    data = np.asarray([2**24 + 1, 2**24 + 3], np.int64)
    expected = ReferenceEvaluator(model).run(None, {"x": data})[0]
    changed = _apply_unsqueeze(model)
    assert changed.SerializeToString() == before
    onnx.checker.check_model(changed, full_check=True)
    np.testing.assert_array_equal(ReferenceEvaluator(changed).run(None, {"x": data})[0], expected)


@pytest.mark.parametrize("operation", ["Add", "Resize"])
def test_unsqueeze_repairs_prepared_float_bindings_with_private_views(operation):
    """Repair an intermediate FLOAT binding against an executable explicit-cast reference."""
    scalar = operation == "Resize"
    inputs = [h.make_tensor_value_info("x", T.INT64, [] if scalar else [2])]
    initializers = [n.from_array(np.asarray([0], np.int64), "axes")]
    nodes = [
        h.make_node("Unsqueeze", ["x", "axes"], ["expanded"]),
        h.make_node("Cast", ["expanded"], ["prepared_float"], to=T.FLOAT),
    ]
    if scalar:
        inputs.append(h.make_tensor_value_info("data", T.FLOAT, [2]))
        nodes.append(h.make_node("Resize", ["data", "", "prepared_float"], ["y"]))
        out_shape = [4]
        feeds = {"x": np.asarray(2, np.int64), "data": np.asarray([3, 7], np.float32)}
    else:
        initializers.append(n.from_array(np.asarray([[1, 2]], np.float32), "other"))
        nodes.append(h.make_node("Add", ["prepared_float", "other"], ["y"]))
        out_shape = [1, 2]
        feeds = {"x": np.asarray([2**24 + 1, 2**24 + 3], np.int64)}
    reference = h.make_model(
        h.make_graph(
            nodes,
            "prepared_float_binding",
            inputs,
            [
                h.make_tensor_value_info("y", T.FLOAT, out_shape),
                h.make_tensor_value_info("expanded", T.INT64, [1] if scalar else [1, 2]),
            ],
            initializers,
        ),
        opset_imports=[h.make_opsetid("", 17)],
    )
    reference.ir_version = 9
    reference = onnx.shape_inference.infer_shapes(reference, strict_mode=True)
    onnx.checker.check_model(reference, full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    expected = ort.InferenceSession(
        reference.SerializeToString(), options, providers=["CPUExecutionProvider"]
    ).run(None, feeds)
    prepared = copy.deepcopy(reference)
    del prepared.graph.node[1]
    consumer = prepared.graph.node[1]
    for port, name in enumerate(consumer.input):
        if name == "prepared_float":
            consumer.input[port] = "expanded"
    # Earlier preparation stages can leave integer views on an already FLOAT-bound port.
    changed = _apply_unsqueeze(prepared)
    onnx.checker.check_model(changed, full_check=True)
    casts = [node for node in changed.graph.node if node.op_type == "Cast"]
    assert len(casts) == 1 and consumer.input[2 if scalar else 0] == casts[0].output[0]
    actual = ort.InferenceSession(
        changed.SerializeToString(), options, providers=["CPUExecutionProvider"]
    ).run(None, feeds)
    for got, wanted in zip(actual, expected):
        np.testing.assert_array_equal(got, wanted)
