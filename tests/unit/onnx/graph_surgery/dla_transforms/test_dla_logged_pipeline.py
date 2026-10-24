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

"""Protect logged pass order and numerical behavior at promoted boundaries."""

import copy

import numpy as np
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from modelopt.onnx.graph_surgery.make_dla_compatible import (
    DLA_TRANSFORMS,
    _transform_make_dla_compatible,
)


def test_logged_pass_order():
    # Recorded by the packaged C++ transform.log files, including both inference stages.
    expected = [
        "RemoveQDQ",
        "ConstantsToInitializers",
        "Cast",
        "RemoveDeqLin",
        "ShapeInference",
        "ReshapeTransposeReshape",
        "ReshapeQDQReduceSum",
        "ReshapeReduceSum",
        "Non4dSliceAxis",
        "Gather",
        "DecomposeLstm",
        "MatMulToTransposeConvTranspose",
        "Non4dModelInputs",
        "Non4dModelOutputs",
        "Non4DReshape",
        "ArgMax",
        "StandaloneReduceOperations",
        "RemoveIntermediarySqueezeAndUnsqueeze",
        "GroupQueryAttention",
        "GatherElements",
        "Non4dInitializers",
        "Non4DExpand",
        "Non4DTile",
        "Flatten",
        "Non4DTranspose",
        "Non4DLpNorm",
        "FixInstanceNormChannelMismatch",
        "UpdateAxis",
        "Where",
        "Not",
        "Unsqueeze",
        "Greater",
        "ShapeInference",
        "TopK",
        "RemoveReshapes",
        "RemoveUnusedInitializers",
    ]
    assert [name for name, _ in DLA_TRANSFORMS] == expected


def _run(model, feeds):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    ).run(None, feeds)


@pytest.mark.parametrize("operation", ["ArgMax", "ReduceSum", "Concat", "Flatten"])
def test_boundary_axis_conversion_preserves_values_and_shapes(operation):
    input_shape = [2, 3, 4]
    feeds = {"x": np.arange(24, dtype=np.float32).reshape(input_shape)}
    inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)]
    initializers = []
    output_type = TensorProto.FLOAT
    if operation == "ArgMax":
        node = helper.make_node(operation, ["x"], ["y"], axis=2, keepdims=0)
        output_shape, output_type = [2, 3], TensorProto.INT64
        expected = feeds["x"].argmax(axis=2)
    elif operation == "ReduceSum":
        node = helper.make_node(operation, ["x", "axes"], ["y"], keepdims=0)
        initializers.append(numpy_helper.from_array(np.array([2], dtype=np.int64), "axes"))
        output_shape = [2, 3]
        expected = feeds["x"].sum(axis=2)
    elif operation == "Concat":
        inputs.append(helper.make_tensor_value_info("other", TensorProto.FLOAT, input_shape))
        feeds["other"] = feeds["x"] * 2
        node = helper.make_node(operation, ["x", "other"], ["y"], axis=2)
        output_shape = [2, 3, 8]
        expected = np.concatenate([feeds["x"], feeds["other"]], axis=2)
    else:
        node = helper.make_node(operation, ["x"], ["y"], axis=1)
        output_shape = [2, 12]
        expected = feeds["x"].reshape(output_shape)
    model = helper.make_model(
        helper.make_graph(
            [node],
            "boundaries",
            inputs,
            [helper.make_tensor_value_info("y", output_type, output_shape)],
            initializers,
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 9
    transformed = _transform_make_dla_compatible(copy.deepcopy(model), verbose=False)
    actual = _run(transformed, feeds)
    assert actual[0].shape == tuple(output_shape)
    np.testing.assert_array_equal(actual[0], expected)


def test_remove_all_qdq_rejects_conflicting_dtype_filter():
    with pytest.raises(ValueError, match="not both"):
        _transform_make_dla_compatible(
            None, remove_all_qdq=True, qdq_quantized_dtypes=[TensorProto.INT8]
        )
