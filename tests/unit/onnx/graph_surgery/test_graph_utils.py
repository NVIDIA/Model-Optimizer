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

"""Shared initializer contracts."""

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from modelopt.onnx.graph_surgery.utils.graph_utils import add_initializer


@pytest.mark.parametrize(
    ("source_dtype", "requested_dtype"),
    [
        (np.float32, None),
        (np.float32, TensorProto.FLOAT16),
        (np.float64, TensorProto.FLOAT),
        (np.int64, TensorProto.INT32),
    ],
)
def test_add_initializer_honors_requested_dtype(source_dtype, requested_dtype):
    """Store and reuse initializers using the requested dtype and matching metadata."""
    dtype = TensorProto.FLOAT16 if requested_dtype is None else requested_dtype
    data = np.array([1, 2], dtype=source_dtype)
    graph = helper.make_graph([], "initializer_dtype", [], [])
    options = {} if requested_dtype is None else {"dtype": requested_dtype}
    assert add_initializer(graph, "weight", data, **options) == "weight"
    assert graph.initializer[0].data_type == dtype
    assert graph.value_info[0].type.tensor_type.elem_type == dtype
    np.testing.assert_array_equal(
        numpy_helper.to_array(graph.initializer[0]),
        data.astype(helper.tensor_dtype_to_np_dtype(dtype)),
    )
    before = graph.SerializeToString()
    assert add_initializer(graph, "weight", data, **options) == "weight"
    assert graph.SerializeToString() == before


def test_add_initializer_preserves_bfloat16_raw_bits():
    """Preserve BFLOAT16 raw bits when adding and reusing an initializer."""
    data = np.array([0x3F80, 0x8000, 0x7FC1], dtype=np.uint16).view(np.int16)
    graph = helper.make_graph([], "bfloat16_bits", [], [])
    add_initializer(graph, "w", data, TensorProto.BFLOAT16)
    assert graph.initializer[0].data_type == TensorProto.BFLOAT16
    assert graph.initializer[0].raw_data == data.tobytes()
    before = graph.SerializeToString()
    add_initializer(graph, "w", data, TensorProto.BFLOAT16)
    assert graph.SerializeToString() == before


@pytest.mark.parametrize(
    ("values", "dtype", "onnx_dtype"),
    [([3], np.int64, TensorProto.INT64), ([2], np.int32, TensorProto.INT32)],
)
def test_generated_initializer_collision_leaves_graph_unchanged(values, dtype, onnx_dtype):
    """Reject conflicting initializer values or types without mutating the graph."""
    graph = helper.make_graph([], "collision", [], [])
    graph.initializer.append(numpy_helper.from_array(np.array([2], dtype=np.int64), "axes"))
    before = graph.SerializeToString()
    with pytest.raises(ValueError, match="Conflicting generated initializer"):
        add_initializer(
            graph, "axes", np.asarray(values, dtype=dtype), onnx_dtype, with_value_info=False
        )
    assert graph.SerializeToString() == before


def test_generated_initializer_collision_preserves_signed_zero():
    """Reject initializer reuse when zero values have different sign bits."""
    graph = helper.make_graph(
        [],
        "signed_zero",
        [],
        [],
        [numpy_helper.from_array(np.array([-0.0], dtype=np.float32), "a")],
    )
    with pytest.raises(ValueError, match="Conflicting generated initializer"):
        add_initializer(
            graph,
            "a",
            np.asarray([0.0], dtype=np.float32),
            TensorProto.FLOAT,
            with_value_info=False,
        )
