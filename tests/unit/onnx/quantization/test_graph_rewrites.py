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

import onnx_graphsurgeon as gs
import pytest
from onnx import TensorProto, helper

from modelopt.onnx.quantization.graph_rewrites import convert_fp16_io, remove_output_initializers


def _model_with_initializer_outputs(names):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])
    initializers = [helper.make_tensor(name, TensorProto.FLOAT, [2], [1.0, 2.0]) for name in names]
    outputs = [y] + [helper.make_tensor_value_info(name, TensorProto.FLOAT, [2]) for name in names]
    graph = helper.make_graph(
        [helper.make_node("Relu", ["x"], ["y"])], "g", [x], outputs, initializer=initializers
    )
    return helper.make_model(graph)


@pytest.mark.parametrize("num_initializer_outputs", [1, 2, 3])
def test_remove_output_initializers_removes_adjacent_outputs(num_initializer_outputs):
    names = [f"const_{i}" for i in range(num_initializer_outputs)]
    model = _model_with_initializer_outputs(names)
    graph = gs.import_onnx(model)

    remove_output_initializers(graph, model.graph.initializer)

    assert [tensor.name for tensor in graph.outputs] == ["y"]
    # A Constant left in the outputs has a read-only dtype, so this raised AttributeError.
    convert_fp16_io(graph)
