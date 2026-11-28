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

"""Check connected pattern rewrites, ownership and numerical equivalence."""

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator
from onnxscript import ir

from modelopt.onnx.graph_surgery.dla_transforms.context import DlaGraph
from modelopt.onnx.graph_surgery.dla_transforms.patterns import (
    ReshapeQDQReduceSum,
    ReshapeReduceSum,
    ReshapeTransposeReshape,
    TopKVectorGather,
)
from modelopt.onnx.graph_surgery.make_dla_compatible import _transform_make_dla_compatible


def _model(nodes, tensors, output_shape):
    graph = helper.make_graph(
        nodes,
        "reshape_patterns",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 2, 2, 8])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, output_shape)],
        [numpy_helper.from_array(np.asarray(array), name) for name, array in tensors.items()],
    )
    return onnx.shape_inference.infer_shapes(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)], ir_version=10)
    )


def _check(original, rules):
    model = ir.serde.deserialize_model(original)
    graph = DlaGraph(model)
    assert graph.apply_pattern(rules.rule(graph)) == 1
    ir.passes.common.RemoveUnusedNodesPass()(model)
    ir.passes.common.TopologicalSortPass()(model)
    result = onnx.shape_inference.infer_shapes(ir.serde.serialize_model(model), strict_mode=True)
    onnx.checker.check_model(result, full_check=True)
    x = np.arange(64, dtype=np.float32).reshape(2, 2, 2, 8) / 16
    np.testing.assert_array_equal(
        ReferenceEvaluator(original).run(None, {"x": x})[0],
        ReferenceEvaluator(result).run(None, {"x": x})[0],
    )
    complete = _transform_make_dla_compatible(original, verbose=False)
    np.testing.assert_array_equal(
        ReferenceEvaluator(original).run(None, {"x": x})[0],
        ReferenceEvaluator(complete).run(None, {"x": x})[0],
    )
    assert graph.apply_pattern(rules.rule(graph)) == 0
    return result


@pytest.mark.parametrize(
    ("shape1", "allowzero"),
    [
        ([2, 2, 2, 2, 4], 0),
        ([0, 2, 2, 2, -1], 0),
        pytest.param([2, 2, 2, 2, 4], 1, id="positive_allowzero"),
    ],
)
def test_reshape_transpose_reshape(shape1, allowzero):
    model = _model(
        [
            helper.make_node("Reshape", ["x", "s1"], ["r"], allowzero=allowzero),
            helper.make_node("Transpose", ["r"], ["t"], perm=[0, 1, 3, 2, 4]),
            helper.make_node("Reshape", ["t", "s2"], ["y"], allowzero=allowzero),
        ],
        {"s1": np.array(shape1, np.int64), "s2": np.array([2, 4, 8], np.int64)},
        [2, 4, 8],
    )
    result = _check(model, ReshapeTransposeReshape)
    assert [node.op_type for node in result.graph.node].count("Slice") == 2
    assert all(len(value.type.tensor_type.shape.dim) <= 4 for value in result.graph.value_info)


@pytest.mark.parametrize(
    ("quantized", "keepdims", "allowzero"),
    [
        (False, 0, 0),
        (True, 0, 0),
        (False, 1, 0),
        pytest.param(False, 0, 1, id="positive_allowzero"),
    ],
)
def test_reshape_reduce_sum(quantized, keepdims, allowzero):
    nodes = [helper.make_node("Reshape", ["x", "s1"], ["r"], allowzero=allowzero)]
    tensors = {"s1": np.array([2, 2, 2, 2, 4], np.int64), "axes": np.array([-1], np.int64)}
    if quantized:
        nodes.extend(
            [
                helper.make_node("QuantizeLinear", ["r", "scale", "zero"], ["q"]),
                helper.make_node("DequantizeLinear", ["q", "scale", "zero"], ["dq"]),
            ]
        )
        tensors.update(scale=np.array(0.125, np.float32), zero=np.array(0, np.uint8))
    nodes.append(
        helper.make_node(
            "ReduceSum", ["dq" if quantized else "r", "axes"], ["y"], keepdims=keepdims
        )
    )
    result = _check(
        _model(nodes, tensors, [2, 2, 2, 2, 1] if keepdims else [2, 2, 2, 2]),
        ReshapeQDQReduceSum if quantized else ReshapeReduceSum,
    )
    assert [node.op_type for node in result.graph.node] == (
        ["Slice", "QuantizeLinear", "DequantizeLinear", "ReduceSum"]
        if quantized
        else ["Slice", "ReduceSum"]
    ) * 2 + ["Concat"] + (["Unsqueeze"] if keepdims else [])
    reductions = [node for node in result.graph.node if node.op_type == "ReduceSum"]
    assert all(node.input[1] == "axes" and not node.attribute for node in reductions)


@pytest.mark.parametrize(
    ("rank", "case"),
    [
        (2, "supported"),
        (4, "supported"),
        (2, "allowzero"),
        (4, "allowzero"),
        (4, "public_indices"),
        (4, "narrow_tail"),
        (4, "values_indices"),
        (4, "public_values"),
        (4, "different_extent"),
        (4, "runtime_k"),
        (4, "runtime_repeats"),
        (4, "different_tiling"),
        (4, "runtime_indices"),
    ],
)
def test_topk_vector_gather(rank, case):
    scores_shape = [1, 5] if rank == 2 else [1, 1, 1, 5]
    data_shape = [2, 5, 3] if rank == 2 else [1, 2, 5, 3]
    target_shape = [1, 2, 1] if rank == 2 else [1, 1, 2, 1]
    repeats = [2, 1, 3] if rank == 2 else [1, 2, 1, 3]
    if case == "different_extent":
        data_shape[2] = 6
    if case == "different_tiling":
        target_shape, repeats = [1, 2, 1, 1], [1, 1, 2, 3]
    axis = 1 if rank == 2 else 2
    nodes = [
        helper.make_node("TopK", ["scores", "k"], ["values", "indices"], axis=-1),
        helper.make_node("Cast", ["indices"], ["cast"], to=TensorProto.INT64),
        helper.make_node(
            "Reshape", ["cast", "shape"], ["reshaped"], allowzero=int(case == "allowzero")
        ),
        helper.make_node("Tile", ["reshaped", "repeats"], ["tiled"]),
    ]
    if rank == 4:
        if case == "narrow_tail":
            nodes.append(helper.make_node("Cast", ["tiled"], ["narrow"], to=TensorProto.INT16))
        nodes.append(helper.make_node("Cast", ["tiled"], ["tail"], to=TensorProto.INT64))
        if case == "narrow_tail":
            nodes[-1].input[0] = "narrow"
    if case == "values_indices":
        nodes[1].input[0] = "values"
    nodes.append(
        helper.make_node(
            "GatherElements", ["data", "tiled" if rank == 2 else "tail"], ["y"], axis=axis
        )
    )
    output_shape = data_shape.copy()
    output_shape[axis] = 2
    outputs = [helper.make_tensor_value_info("y", TensorProto.FLOAT, output_shape)]
    if case == "public_indices":
        outputs.append(helper.make_tensor_value_info("indices", TensorProto.INT64, [1, 1, 1, 2]))
    if case == "public_values":
        outputs.append(helper.make_tensor_value_info("values", TensorProto.FLOAT, [1, 1, 1, 2]))
    graph = helper.make_graph(
        nodes,
        "topk",
        [
            helper.make_tensor_value_info("scores", TensorProto.FLOAT, scores_shape),
            helper.make_tensor_value_info("data", TensorProto.FLOAT, data_shape),
        ],
        outputs,
        [
            numpy_helper.from_array(np.array(array, np.int64), name)
            for name, array in {
                "k": [2],
                "shape": target_shape,
                "repeats": repeats,
            }.items()
        ],
    )
    runtime_controls = {}
    for name, data in [("k", [2]), ("repeats", repeats)]:
        if case == "runtime_" + name:
            initializer = next(value for value in graph.initializer if value.name == name)
            graph.initializer.remove(initializer)
            graph.input.append(helper.make_tensor_value_info(name, TensorProto.INT64, [len(data)]))
            runtime_controls[name] = np.asarray(data, np.int64)
    if case == "runtime_indices":
        del graph.node[:-1]
        graph.node[0].input[1] = "input_indices"
        graph.input.append(
            helper.make_tensor_value_info("input_indices", TensorProto.INT64, output_shape)
        )
        runtime_controls["input_indices"] = np.zeros(output_shape, np.int64)
    if rank == 2:
        graph.input[0].CopyFrom(
            helper.make_tensor_value_info("features", TensorProto.FLOAT, [1, 5, 2])
        )
        graph.initializer.extend(
            [
                numpy_helper.from_array(
                    np.array([[1, 0.75, 0.5], [0.25, 0.125, 0.0625]], np.float32), "score_weight"
                ),
                numpy_helper.from_array(np.array([0, 0.125, -0.125], np.float32), "score_bias"),
            ]
        )
        for node in reversed(
            [
                helper.make_node("MatMul", ["features", "score_weight"], ["projected"]),
                helper.make_node("Add", ["projected", "score_bias"], ["biased"]),
                helper.make_node("ReduceMax", ["biased"], ["scores"], axes=[2], keepdims=0),
            ]
        ):
            graph.node.insert(0, node)
    source = onnx.shape_inference.infer_shapes(
        helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 17 if rank == 2 else 21)], ir_version=10
        )
    )
    model = ir.serde.deserialize_model(source)
    graph = DlaGraph(model)
    supported = case in ("supported", "allowzero")
    assert graph.apply_pattern(TopKVectorGather.rule(graph)) == int(supported)
    ir.passes.common.RemoveUnusedNodesPass()(model)
    assert all(
        value.type is not None and value.shape is not None
        for node in model.graph
        for value in node.outputs
    )
    result = onnx.shape_inference.infer_shapes(ir.serde.serialize_model(model), strict_mode=True)
    onnx.checker.check_model(result, full_check=True)
    feeds = {
        "scores": np.array([1, 5, 3, 2, 4], np.float32).reshape(scores_shape),
        "data": np.arange(np.prod(data_shape), dtype=np.float32).reshape(data_shape),
    }
    if case == "values_indices":
        feeds["scores"] /= 8
    feeds.update(runtime_controls)
    if rank == 2:
        scores = feeds.pop("scores").reshape(5)
        feeds["features"] = np.stack([scores, scores / 2], axis=-1).reshape(1, 5, 2)
    for expected, actual in zip(
        ReferenceEvaluator(source).run(None, feeds),
        ReferenceEvaluator(result).run(None, feeds),
        strict=True,
    ):
        np.testing.assert_array_equal(expected, actual)
    if not supported:
        return
    complete = _transform_make_dla_compatible(source, verbose=False)
    for expected, actual in zip(
        ReferenceEvaluator(source).run(None, feeds),
        ReferenceEvaluator(complete).run(None, feeds),
        strict=True,
    ):
        np.testing.assert_array_equal(expected, actual)
    if rank == 2:
        reduced = next(node for node in complete.graph.node if node.op_type == "ReduceMax")
        assert {attr.name: helper.get_attribute_value(attr) for attr in reduced.attribute} == {
            "axes": [3],
            "keepdims": 1,
        }
    if supported:
        assert [node.op_type for node in result.graph.node] == (
            ["MatMul", "Add", "ReduceMax"] if rank == 2 else []
        ) + [
            "Reshape",
            "TopK",
            "Cast",
            "Squeeze",
            "Gather",
        ]
