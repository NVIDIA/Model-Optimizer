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
import torch
import torch.nn as nn
from onnx import helper


def _create_simple_conv_onnx_model():
    """Build ONNX model: Input -> Conv -> Relu -> Output (minimal for autotuner tests)."""
    input_tensor = helper.make_tensor_value_info(
        "input", onnx.TensorProto.FLOAT, [64, 32, 224, 224]
    )
    output_tensor = helper.make_tensor_value_info(
        "output", onnx.TensorProto.FLOAT, [64, 64, 224, 224]
    )
    conv_node = helper.make_node(
        "Conv",
        inputs=["input", "conv_weight"],
        outputs=["conv_out"],
        name="conv",
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
    )
    relu_node = helper.make_node("Relu", inputs=["conv_out"], outputs=["output"], name="relu")
    graph = helper.make_graph(
        [conv_node, relu_node],
        "simple_conv",
        [input_tensor],
        [output_tensor],
        initializer=[
            helper.make_tensor(
                "conv_weight", onnx.TensorProto.FLOAT, [64, 32, 3, 3], [0.1] * (64 * 32 * 3 * 3)
            )
        ],
    )
    return helper.make_model(graph, producer_name="test")


def _create_simple_resnet18_model():
    """Build a ResNet-18 subgraph (stem + layer1) for MOQ + Autotuner integration tests.

    Architecture:
        Conv(3→64, 7×7, stride=2) → ReLU → MaxPool(3×3, stride=2)
        → BasicBlock(64→64) → BasicBlock(64→64)

    Input shape: [1, 3, 1024, 1024], output shape: [1, 64, 256, 256].
    """

    class _BasicBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(64, 64, 3, padding=1, bias=True)
            self.act1 = nn.ReLU()
            self.conv2 = nn.Conv2d(64, 64, 3, padding=1, bias=True)
            self.act2 = nn.ReLU()

        def forward(self, x):
            return self.act2(self.conv2(self.act1(self.conv1(x))) + x)

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=True)
            self.act1 = nn.ReLU()
            self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
            self.layer1 = nn.Sequential(_BasicBlock(), _BasicBlock())

        def forward(self, x):
            return self.layer1(self.maxpool(self.act1(self.conv1(x))))

    torch.manual_seed(42)
    model = _Model().eval()
    input_tensor = torch.zeros(1, 3, 1024, 1024)

    return model, input_tensor
