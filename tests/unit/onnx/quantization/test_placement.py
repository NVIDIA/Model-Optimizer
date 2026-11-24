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

import io

import onnx
import torch
import torch.nn.functional as F
from torch import nn

from modelopt.onnx.quantization.placement import find_qdq_placement


class _SEResidualBlock(nn.Module):
    def __init__(self, channels=16):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.se1 = nn.Conv2d(channels, 4, 1)
        self.se2 = nn.Conv2d(4, channels, 1)

    def forward(self, x):
        y = self.conv2(F.silu(self.conv1(x)))
        scale = torch.sigmoid(self.se2(F.silu(self.se1(F.adaptive_avg_pool2d(y, 1)))))
        return F.relu(y * scale + x)


def test_find_qdq_placement_shares_quantizers():
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), _SEResidualBlock(), _SEResidualBlock()
    ).eval()
    buffer = io.BytesIO()
    torch.onnx.export(model, (torch.randn(1, 3, 32, 32),), buffer, dynamo=False, opset_version=17)
    onnx_model = onnx.load_from_string(buffer.getvalue())

    placement = find_qdq_placement(onnx_model, log_level="WARNING")

    def readers(node_name):
        return {
            t.name: set(t.consumers)
            for t in placement.tensors.values()
            if any(node == node_name for node, _ in t.consumers)
        }

    # The residual skip and the next block's Conv input share one quantizer.
    assert readers("/3/Add") == {
        "/3/Mul_2_output_0": {("/3/Add", 0)},
        "/2/Relu_output_0": {("/3/conv1/Conv", 0), ("/3/Add", 1)},
    }
    # So do the pooling input and the squeeze-and-excitation scale multiply.
    assert readers("/3/GlobalAveragePool") == {
        "/3/conv2/Conv_output_0": {("/3/GlobalAveragePool", 0), ("/3/Mul_2", 0)}
    }
    for tensor in placement.tensors.values():
        assert tensor.axis == (0 if tensor.is_constant else None), tensor
