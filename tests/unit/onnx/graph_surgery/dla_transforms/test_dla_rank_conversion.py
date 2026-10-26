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

"""Numerical checks for independently reviewable rank and boundary passes."""

import copy

import numpy as np
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from modelopt.onnx.graph_surgery.dla_transforms.dla_model_boundaries import (
    _apply_non4d_model_inputs,
    _apply_non4d_model_outputs,
)
from modelopt.onnx.graph_surgery.dla_transforms.dla_rank_conversion import (
    _apply_non4d_expand,
    _apply_non4d_reshape,
    _apply_non4d_tile,
    _apply_non4d_transpose,
)


@pytest.mark.parametrize("operation", ["Transpose", "Reshape", "Tile", "Expand"])
def test_rank_conversion_preserves_operator_values(operation):
    """Execute native operators across explicitly deferred boundary and rank stages."""
    shape = [1, 3] if operation == "Expand" else [2, 3]
    inputs, initializers, attributes = ["x"], [], {}
    if operation == "Transpose":
        attributes["perm"] = [1, 0]
        output_shape = [3, 2]
        transform = _apply_non4d_transpose
    else:
        values = {"Reshape": [3, 2], "Tile": [2, 1], "Expand": [2, 3]}[operation]
        output_shape = {"Reshape": [3, 2], "Tile": [4, 3], "Expand": [2, 3]}[operation]
        inputs.append("parameters")
        initializers.append(numpy_helper.from_array(np.array(values, dtype=np.int64), "parameters"))
        transform = {
            "Reshape": _apply_non4d_reshape,
            "Tile": _apply_non4d_tile,
            "Expand": _apply_non4d_expand,
        }[operation]
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node(operation, inputs, ["y"], name="operator", **attributes)],
            "rank",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, output_shape)],
            initializers,
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 9
    transformed = copy.deepcopy(model)
    for apply in (_apply_non4d_model_inputs, _apply_non4d_model_outputs, transform):
        options = {"defer_rank_conversion": True} if apply is _apply_non4d_model_outputs else {}
        transformed = apply(transformed, **options)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    expected = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    ).run(None, {"x": data})
    actual = ort.InferenceSession(
        transformed.SerializeToString(), options, providers=["CPUExecutionProvider"]
    ).run(None, {"x": data})
    np.testing.assert_array_equal(actual[0], expected[0])
