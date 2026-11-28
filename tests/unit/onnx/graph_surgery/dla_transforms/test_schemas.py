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

"""Tests for preparing standard Q/DQ schemas before DLA conversion."""

import onnx
import pytest
from onnx import TensorProto, helper

from modelopt.onnx.graph_surgery.dla_transforms.schemas import upgrade_quantizers


@pytest.mark.parametrize(
    ("opset", "with_qdq", "expected_opset"),
    [(17, True, 19), (18, True, 19), (18, False, 18), (19, True, 19), (21, True, 21)],
)
def test_upgrade_quantizers(opset, with_qdq, expected_opset):
    """Upgrade older Q/DQ graphs while preserving caller-owned input models."""
    if with_qdq:
        nodes = [
            helper.make_node("QuantizeLinear", ["x", "scale"], ["q"]),
            helper.make_node("DequantizeLinear", ["q", "scale"], ["y"]),
        ]
        initializers = [helper.make_tensor("scale", TensorProto.FLOAT, [], [0.5])]
    else:
        nodes = [helper.make_node("Identity", ["x"], ["y"])]
        initializers = []
    graph = helper.make_graph(
        nodes,
        "quantizer_schema",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)], ir_version=10)
    original = model.SerializeToString()

    result = upgrade_quantizers(model)

    assert model.SerializeToString() == original
    assert result.opset_import[0].version == expected_opset
    onnx.checker.check_model(result, full_check=True)
