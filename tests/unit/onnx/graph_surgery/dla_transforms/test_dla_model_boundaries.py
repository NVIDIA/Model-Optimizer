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

"""Boundary promotion refuses unsupported exposed adapters before changing the graph."""

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from modelopt.onnx.graph_surgery.dla_transforms.dla_model_boundaries import (
    _apply_non4d_model_inputs,
)


def test_input_promotion_refusal_leaves_graph_unchanged():
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Identity", ["x"], ["first"]),
                helper.make_node("Unsqueeze", ["x", "axes"], ["y"], name="exposed"),
            ],
            "boundary",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 2, 3])],
            [numpy_helper.from_array(np.array([0, 1], dtype=np.int64), "axes")],
        ),
        opset_imports=[helper.make_opsetid("", 21)],
    )
    before = model.SerializeToString()
    with pytest.raises(ValueError):
        _apply_non4d_model_inputs(model)
    assert model.SerializeToString() == before
