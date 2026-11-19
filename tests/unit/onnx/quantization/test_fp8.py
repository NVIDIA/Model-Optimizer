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
from onnxruntime.quantization.quant_utils import compute_data_quant_params

from modelopt.onnx.quantization.fp8 import int8_to_fp8


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("per_channel", [False, True])
@pytest.mark.parametrize("constant_scale", [False, True])
def test_int8_to_fp8_preserves_representable_values(dtype, per_channel, constant_scale):
    # Exact E4M3 values, including subnormals that an oversized scale would round to zero.
    values = np.array([-448, -1, -(2**-9), 0, 2**-9, 1, 448], dtype=dtype)
    data = values[:, None] * np.array([1, 2], dtype=dtype) if per_channel else values
    if per_channel:
        params = [
            compute_data_quant_params(data[:, i], TensorProto.INT8, symmetric=True)
            for i in range(data.shape[1])
        ]
        zero_point, scale = (np.asarray(p) for p in zip(*params))
    else:
        zero_point, scale = compute_data_quant_params(data, TensorProto.INT8, symmetric=True)

    scale_tensor = numpy_helper.from_array(scale, "scale")
    initializers = [numpy_helper.from_array(zero_point, "zero_point")]
    nodes = []
    if constant_scale:
        nodes.append(helper.make_node("Constant", [], ["scale"], value=scale_tensor))
    else:
        initializers.append(scale_tensor)
    nodes.extend(
        [
            helper.make_node("QuantizeLinear", ["input", "scale", "zero_point"], ["q"], axis=1),
            helper.make_node("DequantizeLinear", ["q", "scale", "zero_point"], ["output"], axis=1),
        ]
    )
    tensor_type = TensorProto.FLOAT16 if dtype == np.float16 else TensorProto.FLOAT
    graph = helper.make_graph(
        nodes,
        "fp8_scale_conversion",
        [helper.make_tensor_value_info("input", tensor_type, data.shape)],
        [helper.make_tensor_value_info("output", tensor_type, data.shape)],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)], ir_version=10)

    int8_to_fp8(model)
    onnx.checker.check_model(model)
    quantized, restored = ReferenceEvaluator(model).run(["q", "output"], {"input": data})

    assert restored.dtype == data.dtype
    np.testing.assert_array_equal(np.max(np.abs(quantized.astype(np.float32)), axis=0), 448)
    np.testing.assert_allclose(restored, data, rtol=1e-7, atol=0)
