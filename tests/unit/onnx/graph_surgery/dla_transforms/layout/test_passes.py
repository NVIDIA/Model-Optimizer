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

"""Check variadic axis passes and coordinated native/public layout boundaries."""

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator
from onnxscript import ir

from modelopt.onnx.graph_surgery.dla_transforms.context import DlaGraph
from modelopt.onnx.graph_surgery.dla_transforms.layout.passes import (
    LayoutBoundaryCommit,
    SplitAxisLayout,
)


@pytest.mark.parametrize("output_count", [2, 4])
@pytest.mark.parametrize(
    ("shape", "axis", "mapped_axis"),
    [((2, 3), 1, 3), ((1, 1, 3, 2, 3), 2, 1), ((2, 3, 1, 1, 4), -4, -3)],
)
def test_split_axis_layout(output_count, shape, axis, mapped_axis):
    """Split remaps positive/negative axes through promotion and first-singleton collapse."""
    names = [f"y{i}" for i in range(output_count)]
    output_shape = list(shape)
    shape = list(shape)
    shape[axis] *= output_count
    source = helper.make_model(
        helper.make_graph(
            [helper.make_node("Split", ["x", "sizes"], names, axis=axis)],
            "split_layout",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
            [
                helper.make_tensor_value_info(name, TensorProto.FLOAT, output_shape)
                for name in names
            ],
            [numpy_helper.from_array(np.full(output_count, 3, np.int64), "sizes")],
        ),
        opset_imports=[helper.make_opsetid("", 21)],
        ir_version=10,
    )
    context = DlaGraph(ir.serde.deserialize_model(source))
    context.begin_layout()
    transform = SplitAxisLayout(context)
    assert context.apply_pass(transform).modified
    assert transform.transformed_nodes == 1
    assert not context.apply_pass(transform).modified
    context.apply_pass(LayoutBoundaryCommit(context))
    transformed = ir.serde.serialize_model(context.model)
    context.end_layout()
    onnx.checker.check_model(transformed, full_check=True)
    split = next(node for node in transformed.graph.node if node.op_type == "Split")
    assert len(split.output) == output_count
    assert next(attr.i for attr in split.attribute if attr.name == "axis") == mapped_axis
    feeds = {"x": np.arange(np.prod(shape), dtype=np.float32).reshape(shape)}
    for expected, actual in zip(
        ReferenceEvaluator(source).run(None, feeds),
        ReferenceEvaluator(transformed).run(None, feeds),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
