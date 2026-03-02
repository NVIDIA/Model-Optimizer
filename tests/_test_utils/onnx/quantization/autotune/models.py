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

"""
Shared test ONNX models for autotuner unit tests.

Model creation functions live here; tests import and call them directly.
"""

import onnx
from onnx import helper


def _create_simple_conv_onnx_model():
    """Build ONNX model: Input -> Conv -> Relu -> Output (minimal for autotuner tests)."""
    input_tensor = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [32, 3, 224, 224])
    output_tensor = helper.make_tensor_value_info(
        "output", onnx.TensorProto.FLOAT, [32, 64, 224, 224]
    )
    conv_node = helper.make_node(
        "Conv", inputs=["input", "conv_weight"], outputs=["conv_out"], name="conv"
    )
    relu_node = helper.make_node("Relu", inputs=["conv_out"], outputs=["output"], name="relu")
    graph = helper.make_graph(
        [conv_node, relu_node],
        "simple_conv",
        [input_tensor],
        [output_tensor],
        initializer=[
            helper.make_tensor(
                "conv_weight", onnx.TensorProto.FLOAT, [64, 3, 3, 3], [0.1] * (64 * 3 * 3 * 3)
            )
        ],
    )
    return helper.make_model(graph, producer_name="test")
