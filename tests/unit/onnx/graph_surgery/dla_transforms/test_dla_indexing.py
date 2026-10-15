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

"""Independent numerical and boundary-contract checks for indexing passes."""

import copy

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from modelopt.onnx.graph_surgery.dla_transforms.dla_indexing import (
    _apply_gather,
    _apply_gather_elements,
    _apply_group_query_attention,
    _apply_standalone_reduce_operations,
)
from modelopt.onnx.graph_surgery.dla_transforms.dla_model_boundaries import (
    _apply_non4d_model_inputs,
    _apply_non4d_model_outputs,
)


@pytest.mark.parametrize("operation", ["Gather", "GatherElements"])
def test_index_dtype_and_rank_conversion_preserves_selected_values(operation):
    if operation == "Gather":
        shape, output_shape, axis = [5, 4], [4], 0
        indices = np.int64(2)
        transform = _apply_gather
    else:
        shape, output_shape, axis = [2, 3, 4], [2, 3, 4], 1
        indices = np.ones(shape, dtype=np.int64)
        transform = _apply_gather_elements
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node(operation, ["x", "indices"], ["y"], name="index", axis=axis)],
            "indexing",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, output_shape)],
            initializer=[numpy_helper.from_array(np.asarray(indices), "indices")],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 9
    candidate = transform(copy.deepcopy(model))
    candidate = _apply_non4d_model_inputs(candidate)
    candidate = _apply_non4d_model_outputs(candidate)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    expected = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    ).run(None, {"x": data})
    actual = ort.InferenceSession(
        candidate.SerializeToString(), options, providers=["CPUExecutionProvider"]
    ).run(None, {"x": data})
    np.testing.assert_array_equal(actual[0], expected[0])


@pytest.mark.parametrize("direct_output", [False, True])
def test_attention_preserves_graph_boundaries_and_wraps_internal_tensors(direct_output):
    output = "attention" if direct_output else "y"
    nodes = [
        helper.make_node("Unsqueeze", ["x", "axes"], ["q4"]),
        helper.make_node(
            "GroupQueryAttention",
            ["q4", "x", "x", "", ""],
            ["attention"],
            name="attention",
            domain="com.microsoft",
            num_heads=1,
            kv_num_heads=1,
        ),
    ]
    if not direct_output:
        nodes.append(helper.make_node("Identity", ["attention"], ["y"]))
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "attention",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, 2, 4])],
            [
                helper.make_tensor_value_info(
                    output, TensorProto.FLOAT16, [1, 2, 4] if direct_output else [1, 1, 2, 4]
                )
            ],
            initializer=[numpy_helper.from_array(np.array([0], dtype=np.int64), "axes")],
        ),
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)],
    )
    candidate = _apply_group_query_attention(model)
    onnx.checker.check_model(candidate)
    attention = next(n for n in candidate.graph.node if n.op_type == "GroupQueryAttention")
    producer = {name: node for node in candidate.graph.node for name in node.output}
    assert producer[attention.input[0]].op_type == "Squeeze"
    assert list(attention.input[1:]) == ["x", "x", "", ""]
    assert producer["attention"].op_type == (
        "GroupQueryAttention" if direct_output else "Unsqueeze"
    )


def test_late_refusal_preserves_the_original_model():
    nodes = [
        helper.make_node("ReduceSum", ["x"], ["first"], keepdims=0),
        helper.make_node("ReduceSum", ["x", "dynamic_axes"], ["y"], keepdims=0),
    ]
    inputs = [
        helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3]),
        helper.make_tensor_value_info("dynamic_axes", TensorProto.INT64, [1]),
    ]
    outputs = [helper.make_tensor_value_info("y", TensorProto.FLOAT, [3])]
    model = helper.make_model(
        helper.make_graph(nodes, "refusal", inputs, outputs),
        opset_imports=[helper.make_opsetid("", 21)],
    )
    before = model.SerializeToString()
    with pytest.raises(ValueError):
        _apply_standalone_reduce_operations(model)
    assert model.SerializeToString() == before


@pytest.mark.parametrize("axes", [[-2], [-1, 0], [0, -1]])
def test_reduction_preserves_negative_and_mixed_axes(axes):
    from onnx.reference import ReferenceEvaluator

    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("ReduceSum", ["x", "axes"], ["y"], keepdims=int(len(axes) > 1))],
            "reduce",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
            [numpy_helper.from_array(np.array(axes, dtype=np.int64), "axes")],
        ),
        opset_imports=[helper.make_opsetid("", 21)],
    )
    model = onnx.shape_inference.infer_shapes(model)
    data = np.arange(6, dtype=np.float32).reshape(2, 3)
    expected = ReferenceEvaluator(model).run(None, {"x": data})[0]
    candidate = _apply_non4d_model_inputs(copy.deepcopy(model))
    candidate = _apply_non4d_model_outputs(candidate)
    candidate = _apply_standalone_reduce_operations(candidate)
    onnx.checker.check_model(candidate)
    actual = ReferenceEvaluator(candidate).run(None, {"x": data})[0]
    np.testing.assert_array_equal(actual, expected)
