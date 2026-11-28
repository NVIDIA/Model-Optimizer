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
from _test_utils.onnx.lib_test_models import build_se_residual_model

from modelopt.onnx.quantization.placement import find_qdq_placement


def test_find_qdq_placement_shares_quantizers():
    model, example = build_se_residual_model()
    buffer = io.BytesIO()
    torch.onnx.export(model, (example,), buffer, dynamo=False, opset_version=17)
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
