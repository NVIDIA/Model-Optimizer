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

"""Native rank and payload contracts for the final reshape cleanup."""

import copy

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from modelopt.onnx.graph_surgery.dla_transforms.dla_graph_cleanup import _remove_unused_initializers
from modelopt.onnx.graph_surgery.dla_transforms.dla_remove_reshapes import _apply_remove_reshapes


def _execute(model, feeds):
    """Check and execute the actual CPU graph with native public outputs."""
    onnx.checker.check_model(model, full_check=True)
    return ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"]).run(
        None, feeds
    )


@pytest.mark.parametrize("ir_version", [3, 9])
def test_fixed_chain_controls_respect_legacy_initializer_declarations(ir_version):
    """Collapse legacy fixed targets, declare new controls, and retain modern runtime defaults."""
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Reshape", ["x", "first_shape"], ["middle"]),
                helper.make_node("Reshape", ["middle", "last_shape"], ["y"]),
            ],
            "legacy_chain",
            [
                helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 2, 3]),
                helper.make_tensor_value_info("first_shape", TensorProto.INT64, [2]),
                helper.make_tensor_value_info("last_shape", TensorProto.INT64, [3]),
                helper.make_tensor_value_info("unused_default", TensorProto.INT64, [1]),
            ],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 1, 2])],
            [
                numpy_helper.from_array(np.array([2, 3], np.int64), "first_shape"),
                numpy_helper.from_array(np.array([3, 1, 2], np.int64), "last_shape"),
                numpy_helper.from_array(np.array([1], np.int64), "unused_default"),
            ],
            value_info=[helper.make_tensor_value_info("middle", TensorProto.FLOAT, [2, 3])],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = ir_version
    feeds = {"x": np.arange(6, dtype=np.float32).reshape(1, 1, 2, 3)}
    expected = _execute(model, feeds)
    converted = _apply_remove_reshapes(copy.deepcopy(model))
    assert len(converted.graph.node) == (1 if ir_version == 3 else 2)
    converted = _remove_unused_initializers(converted)
    input_names = {value.name for value in converted.graph.input}
    assert {"first_shape", "last_shape"}.isdisjoint(input_names) == (ir_version == 3)
    assert ("unused_default" in input_names) == (ir_version >= 4)
    assert {value.name for value in converted.graph.initializer} >= input_names - {"x"}
    for actual, reference in zip(_execute(converted, feeds), expected):
        np.testing.assert_array_equal(actual, reference)


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("cast", [False, True])
def test_native_gather_index_views_are_not_bypassed(legacy, cast):
    """Keep the native vector view required by the Gather index port."""
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Reshape", ["source", "shape"], ["indices"]),
                helper.make_node("Gather", ["x", "indices"], ["y"], axis=2),
            ],
            "index_cleanup",
            [
                helper.make_tensor_value_info("source", TensorProto.INT64, [1, 1, 1, 3]),
                helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 5, 4]),
            ],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 3, 4])],
            [numpy_helper.from_array(np.asarray([-1], np.int64), "shape")],
            value_info=[helper.make_tensor_value_info("indices", TensorProto.INT64, [3])],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    if cast:
        model.graph.node.insert(
            1, helper.make_node("Cast", ["indices"], ["cast_indices"], to=TensorProto.INT64)
        )
        model.graph.node[2].input[1] = "cast_indices"
    model.ir_version = 3 if legacy else 9
    if legacy:
        model.graph.input.append(helper.make_tensor_value_info("shape", TensorProto.INT64, [1]))
    feeds = {
        "source": np.asarray([0, 2, 4], np.int64).reshape(1, 1, 1, 3),
        "x": np.arange(20, dtype=np.float32).reshape(1, 1, 5, 4),
    }
    expected = _execute(model, feeds)[0]
    converted = _apply_remove_reshapes(copy.deepcopy(model))
    np.testing.assert_array_equal(_execute(converted, feeds)[0], expected)


def test_reshape_cleanup_retains_untouched_constant_protobuf_payloads():
    """Leave an unrelated payload object in place while removing an identity Reshape."""
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "Constant",
                    [],
                    ["constant"],
                    value=numpy_helper.from_array(np.arange(8, dtype=np.float32)),
                ),
                helper.make_node("Reshape", ["x", "shape"], ["y"]),
            ],
            "payload_cleanup",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 2, 3])],
            [
                helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 2, 3]),
                helper.make_tensor_value_info("constant", TensorProto.FLOAT, [8]),
            ],
            [numpy_helper.from_array(np.asarray([1, 1, 2, 3], np.int64), "shape")],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 9
    feeds = {"x": np.arange(6, dtype=np.float32).reshape(1, 1, 2, 3)}
    expected = _execute(model, feeds)
    constant = model.graph.node[0]
    converted = _apply_remove_reshapes(model)
    assert converted.graph.node[0] is constant
    for actual, reference in zip(_execute(converted, feeds), expected):
        np.testing.assert_array_equal(actual, reference)


def test_unnamed_public_identity_reshapes_allocate_distinct_node_names():
    """Preserve both public values when multiple unnamed shapes become Identity nodes."""
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Reshape", ["x", "first_shape"], ["first"]),
                helper.make_node("Reshape", ["z", "second_shape"], ["second"]),
            ],
            "public_names",
            [
                helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 2, 3]),
                helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 1, 3, 2]),
            ],
            [
                helper.make_tensor_value_info("first", TensorProto.FLOAT, [1, 1, 2, 3]),
                helper.make_tensor_value_info("second", TensorProto.FLOAT, [1, 1, 3, 2]),
            ],
            [
                numpy_helper.from_array(np.asarray([1, 1, 2, 3], np.int64), "first_shape"),
                numpy_helper.from_array(np.asarray([1, 1, 3, 2], np.int64), "second_shape"),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 9
    feeds = {
        "x": np.arange(6, dtype=np.float32).reshape(1, 1, 2, 3),
        "z": np.arange(6, dtype=np.float32).reshape(1, 1, 3, 2),
    }
    expected = _execute(model, feeds)
    converted = _apply_remove_reshapes(copy.deepcopy(model))
    names = [node.name for node in converted.graph.node]
    assert all(names)
    assert len(set(names)) == len(names)
    for actual, reference in zip(_execute(converted, feeds), expected):
        np.testing.assert_array_equal(actual, reference)


def test_nested_capture_prevents_reshape_chain_collapse():
    """Keep the intermediate native layout referenced inside a nested branch."""
    branch = helper.make_graph(
        [helper.make_node("Identity", ["middle"], ["out"])],
        "branch",
        [],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [2, 3])],
    )
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Reshape", ["x", "short_shape"], ["middle"]),
                helper.make_node("Reshape", ["middle", "long_shape"], ["y"]),
                helper.make_node(
                    "If", ["condition"], ["view"], then_branch=branch, else_branch=branch
                ),
            ],
            "captured_shape",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 2, 3])],
            [
                helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 2, 3]),
                helper.make_tensor_value_info("view", TensorProto.FLOAT, [2, 3]),
            ],
            [
                numpy_helper.from_array(np.asarray([2, 3], np.int64), "short_shape"),
                numpy_helper.from_array(np.asarray([1, 1, 2, 3], np.int64), "long_shape"),
                numpy_helper.from_array(np.asarray(True), "condition"),
            ],
            value_info=[helper.make_tensor_value_info("middle", TensorProto.FLOAT, [2, 3])],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 9
    feeds = {"x": np.arange(6, dtype=np.float32).reshape(1, 1, 2, 3)}
    expected = _execute(model, feeds)
    converted = _apply_remove_reshapes(copy.deepcopy(model))
    for actual, reference in zip(_execute(converted, feeds), expected):
        np.testing.assert_array_equal(actual, reference)


def test_padding_view_before_public_unary_output_keeps_its_rank():
    """Retain a real rank change even when its next consumer is rank-agnostic."""
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Reshape", ["x", "shape"], ["padded"]),
                helper.make_node("Relu", ["padded"], ["y"]),
            ],
            "padding_view",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [3])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 1, 3])],
            [numpy_helper.from_array(np.asarray([1, 1, 1, 3], np.int64), "shape")],
            value_info=[helper.make_tensor_value_info("padded", TensorProto.FLOAT, [1, 1, 1, 3])],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 9
    feeds = {"x": np.asarray([-1, 0, 1], np.float32)}
    expected = _execute(model, feeds)[0]
    converted = _apply_remove_reshapes(copy.deepcopy(model))
    np.testing.assert_array_equal(_execute(converted, feeds)[0], expected)


@pytest.mark.parametrize("identity", [False, True])
@pytest.mark.parametrize("public_middle", [False, True])
def test_interleaved_reshape_chains_follow_data_producers(identity, public_middle):
    """Collapse independent interleaved branches without crossing a public intermediate."""
    final_shape = [2, 3] if identity else [6]
    outputs = [
        helper.make_tensor_value_info(name, TensorProto.FLOAT, final_shape) for name in ["y", "w"]
    ]
    if public_middle:
        outputs.append(helper.make_tensor_value_info("a", TensorProto.FLOAT, [3, 2]))
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Reshape", ["x", "a_shape"], ["a"]),
                helper.make_node("Reshape", ["z", "b_shape"], ["b"]),
                helper.make_node("Reshape", ["a", "final_shape"], ["y"]),
                helper.make_node("Reshape", ["b", "final_shape"], ["w"]),
            ],
            "interleaved_chains",
            [helper.make_tensor_value_info(name, TensorProto.FLOAT, [2, 3]) for name in ["x", "z"]],
            outputs,
            [
                numpy_helper.from_array(np.array(values, np.int64), name)
                for name, values in [
                    ("a_shape", [3, 2]),
                    ("b_shape", [1, 6]),
                    ("final_shape", final_shape),
                ]
            ],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=9,
    )
    model = onnx.shape_inference.infer_shapes(model)
    feeds = {
        "x": np.arange(6, dtype=np.float32).reshape(2, 3),
        "z": np.arange(10, 16, dtype=np.float32).reshape(2, 3),
    }
    expected = _execute(model, feeds)
    changed = _apply_remove_reshapes(copy.deepcopy(model))
    assert len(changed.graph.node) == (3 if public_middle else 2)
    for actual, reference in zip(_execute(changed, feeds), expected):
        np.testing.assert_array_equal(actual, reference)
