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

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

from modelopt.onnx.autocast import convert_to_f16
from modelopt.onnx.export import NVFP4QuantExporter


@pytest.mark.parametrize("dtype", [TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.BFLOAT16])
@pytest.mark.parametrize("op_type", ["MatMul", "Gemm"])
def test_nvfp4_preserves_compute_dtype(dtype, op_type):
    np_dtype = helper.tensor_dtype_to_np_dtype(dtype)
    weight = np.full((16, 16), 0.25, dtype=np_dtype)
    bias = np.full(16, 0.5, dtype=np_dtype)
    nodes = [
        helper.make_node(
            "TRT_FP4QDQ", ["weight"], ["dq_weight"], name="weight_qdq", domain="trt", block_size=16
        )
    ]
    if op_type == "Gemm":
        nodes.append(helper.make_node("Gemm", ["x", "dq_weight", "bias"], ["y"], name="linear"))
    else:
        nodes.extend(
            [
                helper.make_node(
                    "Transpose", ["dq_weight"], ["transposed_weight"], name="transpose", perm=[1, 0]
                ),
                helper.make_node("MatMul", ["x", "transposed_weight"], ["product"], name="linear"),
                helper.make_node("Add", ["product", "bias"], ["y"], name="bias"),
            ]
        )
    graph = helper.make_graph(
        nodes,
        "nvfp4_compute_dtype",
        [helper.make_tensor_value_info("x", dtype, [2, 16])],
        [helper.make_tensor_value_info("y", dtype, [2, 16])],
        initializer=[
            numpy_helper.from_array(weight, "weight"),
            numpy_helper.from_array(bias, "bias"),
        ],
        value_info=[
            helper.make_tensor_value_info(name, dtype, [16, 16])
            for name in ["dq_weight", "transposed_weight"]
        ]
        + [helper.make_tensor_value_info("product", dtype, [2, 16])],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 23), helper.make_opsetid("trt", 1)]
    )

    model = NVFP4QuantExporter.process_model(model)

    onnx.checker.check_model(model, full_check=True)
    assert [value.name for value in model.graph.input] == ["x"]
    assert any(t.data_type == TensorProto.FLOAT4E2M1 for t in model.graph.initializer)
    packed_weights = {
        value.name: (value.data_type, value.raw_data)
        for value in model.graph.initializer
        if value.data_type in (TensorProto.FLOAT4E2M1, TensorProto.FLOAT8E4M3FN)
    }
    if dtype == TensorProto.FLOAT:
        inputs = np.full((2, 16), 0.5, dtype=np_dtype)
        (actual,) = ReferenceEvaluator(model).run(None, {"x": inputs})
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual, inputs @ weight + bias, rtol=0.01)
    converted = convert_to_f16(
        model,
        low_precision_type="bf16" if dtype == TensorProto.BFLOAT16 else "fp16",
        keep_io_types=False,
    )
    onnx.checker.check_model(converted, full_check=True)
    assert {
        value.name: (value.data_type, value.raw_data)
        for value in converted.graph.initializer
        if value.data_type in (TensorProto.FLOAT4E2M1, TensorProto.FLOAT8E4M3FN)
    } == packed_weights
