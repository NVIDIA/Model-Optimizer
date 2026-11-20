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

"""Tests for the DLA input opset contract."""

import pytest
from onnx import TensorProto, helper

from modelopt.onnx.graph_surgery.dla_transforms.validation import validate


@pytest.mark.parametrize(
    ("opset", "accepted"),
    [(16, False), (17, True), (18, True), (19, True), (20, True), (21, True), (22, False)],
)
def test_validate_input_opset(opset, accepted):
    """Check the supported input range on a valid ONNX graph."""
    graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"])],
        "opset_validation",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])

    if accepted:
        validate(model)
    else:
        with pytest.raises(ValueError):
            validate(model)
