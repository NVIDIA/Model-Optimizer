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

"""DLA preprocessing must preserve unrelated custom operator functions."""

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from modelopt.onnx.graph_surgery.dla_transforms.dla_constants_to_initializers import (
    _constants_to_initializers_from_model,
)
from modelopt.onnx.graph_surgery.dla_transforms.dla_handle_qdq import _apply_handle_qdq
from modelopt.onnx.graph_surgery.dla_transforms.dla_remove_qdq import _remove_qdq_from_model

ort = pytest.importorskip(
    "onnxruntime", reason="onnxruntime required for custom-function execution"
)


@pytest.mark.parametrize("stage", ["constants", "handle_q", "handle_dq", "remove_qdq"])
def test_preprocessing_keeps_custom_functions_with_standard_names(stage):
    """Custom functions retain their semantics even when they reuse recognized operator names."""
    functions, initializers, value_info = [], [], []
    shape = [2, 3]
    inputs = [helper.make_tensor_value_info("X", TensorProto.FLOAT, shape)]
    if stage == "constants":
        body = helper.make_node("Constant", [], ["value"])
        attribute = helper.make_attribute_ref("value", onnx.AttributeProto.TENSOR)
        attribute.ref_attr_name = "value"
        body.attribute.append(attribute)
        functions.append(
            helper.make_function(
                "local",
                "Constant",
                [],
                ["Y"],
                [body, helper.make_node("Neg", ["value"], ["Y"])],
                [helper.make_opsetid("", 21)],
                attributes=["value"],
            )
        )
        nodes = [
            helper.make_node(
                "Constant",
                [],
                ["Y"],
                domain="local",
                value=numpy_helper.from_array(np.ones(shape, dtype=np.float32), "seed"),
            )
        ]
        inputs, feeds = [], {}
    else:
        initializers = [
            numpy_helper.from_array(np.float32(0.25), "scale"),
            numpy_helper.from_array(np.uint16(0), "zero"),
        ]
        feeds = {"X": np.arange(6, dtype=np.float32).reshape(shape)}
        if stage == "remove_qdq":
            functions = [
                helper.make_function(
                    "local",
                    "QuantizeLinear",
                    ["X", "S", "Z"],
                    ["Y"],
                    [
                        helper.make_node(
                            "Constant", [], ["one"], value=numpy_helper.from_array(np.float32(1))
                        ),
                        helper.make_node("Add", ["X", "one"], ["Y"]),
                    ],
                    [helper.make_opsetid("", 21)],
                ),
                helper.make_function(
                    "local",
                    "DequantizeLinear",
                    ["X", "S", "Z"],
                    ["Y"],
                    [helper.make_node("Identity", ["X"], ["Y"])],
                    [helper.make_opsetid("", 21)],
                ),
            ]
            nodes = [
                helper.make_node("QuantizeLinear", ["X", "scale", "zero"], ["q"], domain="local"),
                helper.make_node("DequantizeLinear", ["q", "scale", "zero"], ["Y"], domain="local"),
            ]
            value_info = [helper.make_tensor_value_info("q", TensorProto.FLOAT, shape)]
        else:
            op_type = "QuantizeLinear" if stage == "handle_q" else "DequantizeLinear"
            functions = [
                helper.make_function(
                    "local",
                    op_type,
                    ["X", "S", "Z"],
                    ["Y"],
                    [helper.make_node("Transpose", ["X"], ["Y"], perm=[1, 0])],
                    [helper.make_opsetid("", 21)],
                )
            ]
            nodes = [helper.make_node(op_type, ["X", "scale", "zero"], ["Y"], domain="local")]
            shape = [3, 2]
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "custom",
            inputs,
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, shape)],
            initializer=initializers,
            value_info=value_info,
        ),
        functions=functions,
        opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("local", 1)],
        ir_version=9,
    )
    onnx.checker.check_model(model, full_check=True)
    expected = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    ).run(None, feeds)[0]
    before = model.SerializeToString()
    if stage == "constants":
        _constants_to_initializers_from_model(model, verbose=False)
    elif stage == "remove_qdq":
        _remove_qdq_from_model(
            model,
            keep_clip_after_inputs=False,
            max_chained_qdq_pairs=5,
            quant_types_to_remove=frozenset([TensorProto.UINT16]),
        )
    else:
        _apply_handle_qdq(model)
    onnx.checker.check_model(model, full_check=True)
    actual = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    ).run(None, feeds)[0]
    np.testing.assert_array_equal(actual, expected)
    assert model.SerializeToString() == before
