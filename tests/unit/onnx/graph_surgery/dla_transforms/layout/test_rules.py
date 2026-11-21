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

"""Check fixed layout patterns against native shapes and coordinated 4D plans."""

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator
from onnxscript import ir

from modelopt.onnx.graph_surgery.dla_transforms.common import static_shape
from modelopt.onnx.graph_surgery.dla_transforms.context import DlaGraph
from modelopt.onnx.graph_surgery.dla_transforms.layout.passes import snapshot
from modelopt.onnx.graph_surgery.dla_transforms.layout.rules import (
    AXIS_RULES,
    INDEXING_RULES,
    NATIVE_RULES,
    NORMALIZATION_RULES,
    SHAPE_RULES,
)


def _apply_rules(model, *, native_shapes, planned_shapes, group, force_fp16_activations=False):
    graph = DlaGraph(model, force_fp16_activations=force_fp16_activations)
    graph.native_shapes = native_shapes
    graph.planned_shapes = planned_shapes
    if group is SHAPE_RULES:
        controls = {}
        rules = [rule.rule(graph, controls) for rule in group]
    elif group is INDEXING_RULES:
        controls, views = {}, {}
        rules = [rule.rule(graph, controls, views) for rule in group]
    else:
        rules = [rule.rule(graph) for rule in group]
    count = graph.apply_patterns(rules)
    return graph, rules, count


@pytest.mark.parametrize("kind", ["Softmax", "LogSoftmax", "Concat", "TopK", "QuantizeLinear"])
def test_axis_promotion(kind):
    shape = [2, 6]
    inputs = ["x"]
    outputs = ["y"]
    controls = {}
    if kind == "Concat":
        inputs.append("x")
    elif kind == "TopK":
        inputs.append("control")
        controls["control"] = np.array([3], np.int64)
        outputs.append("z")
    elif kind == "QuantizeLinear":
        inputs.extend(["scale", "zero"])
        controls.update(scale=np.full(6, 0.125, np.float32), zero=np.zeros(6, np.uint8))
    output_shape = [2, 12 if kind == "Concat" else 3 if kind == "TopK" else 6]
    graph = helper.make_graph(
        [
            helper.make_node(
                kind, inputs, outputs, **({} if kind == "QuantizeLinear" else {"axis": 1})
            )
        ],
        "axes",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
        [
            helper.make_tensor_value_info(
                name,
                TensorProto.INT64
                if kind == "TopK" and name == "z"
                else TensorProto.UINT8
                if kind == "QuantizeLinear"
                else TensorProto.FLOAT,
                output_shape,
            )
            for name in outputs
        ],
        [numpy_helper.from_array(value, name) for name, value in controls.items()],
    )
    source = onnx.shape_inference.infer_shapes(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)], ir_version=10)
    )
    model = ir.serde.deserialize_model(source)
    native = {
        node: (
            tuple(static_shape(value) for value in node.inputs),
            tuple(static_shape(value) for value in node.outputs),
        )
        for node in model.graph
    }
    planned = {"x": (1, 1, *shape), **dict.fromkeys(outputs, (1, 1, *output_shape))}
    context, rules, count = _apply_rules(
        model, native_shapes=native, planned_shapes=planned, group=AXIS_RULES
    )
    assert count == 1
    model.graph.inputs[0].shape = ir.Shape(planned["x"])
    for node in model.graph:
        for value in node.outputs:
            if value.name in planned:
                value.shape = ir.Shape(planned[value.name])
    result = onnx.shape_inference.infer_shapes(ir.serde.serialize_model(model), strict_mode=True)
    onnx.checker.check_model(result, full_check=True)
    x = np.arange(12, dtype=np.float32).reshape(shape) / 16
    expected = ReferenceEvaluator(source).run(None, {"x": x})
    actual = ReferenceEvaluator(result).run(None, {"x": x.reshape(planned["x"])})
    for before, after in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(before, after.reshape(before.shape))
    assert next(iter(model.graph)).attributes["axis"].value == 3
    assert context.apply_patterns(rules) == 0


@pytest.mark.parametrize("version", [1, 2])
def test_contrib_quantizer_default_axis_and_version_boundary(version):
    """Only the supported contrib schema gets its default per-channel axis remapped."""
    source = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "QuantizeLinear", ["x", "scale", "zero"], ["y"], domain="com.microsoft"
                )
            ],
            "contrib_axes",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
            [helper.make_tensor_value_info("y", TensorProto.UINT8, [2, 3])],
            [
                numpy_helper.from_array(np.full(3, 0.125, np.float32), "scale"),
                numpy_helper.from_array(np.zeros(3, np.uint8), "zero"),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", version)],
        ir_version=10,
    )
    model = ir.serde.deserialize_model(source)
    native, planned = snapshot(model)
    if version != 1:
        assert not planned
        planned = {"x": (1, 1, 2, 3), "y": (1, 1, 2, 3)}
    _, _, count = _apply_rules(
        model, native_shapes=native, planned_shapes=planned, group=AXIS_RULES
    )
    assert count == int(version == 1)
    if version != 1:
        assert not next(iter(model.graph)).attributes
        return
    model.graph.inputs[0].shape = ir.Shape(planned["x"])
    model.graph.outputs[0].shape = ir.Shape(planned["y"])
    result = ir.serde.serialize_model(model)
    onnx.checker.check_model(result, full_check=True)
    values = np.arange(6, dtype=np.float32).reshape(2, 3) / 8
    np.testing.assert_array_equal(
        _execute(result, {"x": values.reshape(1, 1, 2, 3)}, True).reshape(2, 3),
        _execute(source, {"x": values}, True),
    )


def _lower_shapes(model, planned):
    working = ir.serde.deserialize_model(onnx.shape_inference.infer_shapes(model, strict_mode=True))
    native = {
        n: (tuple(static_shape(v) for v in n.inputs), tuple(static_shape(v) for v in n.outputs))
        for n in working.graph
    }
    _, _, count = _apply_rules(
        working, native_shapes=native, planned_shapes=planned, group=SHAPE_RULES
    )
    value = working.graph.inputs[0]
    value.shape = ir.Shape((1,) * (4 - len(value.shape)) + tuple(value.shape))
    for node in working.graph:
        for value in node.outputs:
            if value.name in planned:
                value.shape = ir.Shape(planned[value.name])
    ir.passes.common.RemoveUnusedNodesPass()(working)
    result = onnx.shape_inference.infer_shapes(ir.serde.serialize_model(working), strict_mode=True)
    onnx.checker.check_model(result, full_check=True)
    return result, count


@pytest.mark.parametrize(
    ("kind", "shape", "controls", "attrs", "control_dtype"),
    [
        ("Reshape", [2, 3], [[3, 2]], {}, np.int64),
        ("Tile", [2, 3], [[1, 2]], {}, np.int64),
        ("Expand", [1, 3], [[2, 3]], {}, np.int64),
        ("Transpose", [2, 3], [], {"perm": [1, 0]}, np.int64),
        ("Transpose", [2, 3], [], {}, np.int64),
        ("Flatten", [2, 3, 4], [], {"axis": 2}, np.int64),
        ("ArgMax", [2, 3, 4], [], {"axis": 1, "keepdims": 0}, np.int64),
        ("Slice", [2, 3, 4], [[0], [1]], {}, np.int64),
        ("Slice", [2, 3, 4], [[0], [4], [2], [2]], {}, np.int64),
        ("Slice", [2, 3, 4], [[0], [1]], {}, np.int32),
        ("Slice", [2, 3, 4], [[0], [4], [2], [2]], {}, np.int32),
    ],
)
def test_shape_controls_and_native_shapes(kind, shape, controls, attrs, control_dtype):
    """Every shape family fires and retains native values under leading-one promotion."""
    inits = [
        numpy_helper.from_array(np.asarray(a, control_dtype), f"control{i}")
        for i, a in enumerate(controls)
    ]
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node(kind, ["x", *[t.name for t in inits]], ["y"], **attrs)],
            "shape_rule",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
            [
                helper.make_tensor_value_info(
                    "y", TensorProto.INT64 if kind == "ArgMax" else TensorProto.FLOAT, None
                )
            ],
            inits,
        ),
        opset_imports=[helper.make_opsetid("", 21)],
    )
    values = (np.arange(np.prod(shape)).reshape(shape) % 7 - 3).astype(np.float32)
    expected = ReferenceEvaluator(model).run(None, {"x": values})[0]
    target = (1,) * (4 - expected.ndim) + expected.shape
    transformed, count = _lower_shapes(model, {"y": target})
    assert count == 1
    actual = ReferenceEvaluator(transformed).run(
        None, {"x": values.reshape((1,) * (4 - values.ndim) + values.shape)}
    )[0]
    np.testing.assert_array_equal(actual, expected.reshape(target))
    if kind == "Slice":
        initializers = {tensor.name: tensor for tensor in transformed.graph.initializer}
        node = transformed.graph.node[0]
        assert initializers[node.input[3]].data_type == initializers[node.input[1]].data_type


def test_slice_controls_with_distinct_dtypes_are_not_shared():
    """Equal generated axes retain each Slice's control dtype when axes were omitted."""
    nodes = []
    inits = []
    for suffix, dtype in (("32", np.int32), ("64", np.int64)):
        names = [f"{name}{suffix}" for name in ("starts", "ends", "steps")]
        inits.extend(
            numpy_helper.from_array(np.asarray([value], dtype), name)
            for name, value in zip(names, (0, 1, 1))
        )
        nodes.append(
            helper.make_node("Slice", ["x", names[0], names[1], "", names[2]], [f"y{suffix}"])
        )
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "mixed_slice_controls",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4])],
            [
                helper.make_tensor_value_info(f"y{suffix}", TensorProto.FLOAT, None)
                for suffix in ("32", "64")
            ],
            inits,
        ),
        opset_imports=[helper.make_opsetid("", 21)],
    )
    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    expected = ReferenceEvaluator(model).run(None, {"x": values})
    target = (1, 1, 3, 4)
    transformed, count = _lower_shapes(model, {"y32": target, "y64": target})
    assert count == 2
    initializers = {tensor.name: tensor for tensor in transformed.graph.initializer}
    for node in transformed.graph.node:
        assert initializers[node.input[3]].data_type == initializers[node.input[1]].data_type
    actual = ReferenceEvaluator(transformed).run(None, {"x": values.reshape(1, 2, 3, 4)})
    for result, reference in zip(actual, expected):
        np.testing.assert_array_equal(result, reference.reshape(target))


@pytest.mark.parametrize(
    ("target", "transpose", "planned_shape"),
    [
        ([2, 1, 3, 2, 2], True, (2, 3, 2, 2)),
        ([2, -1, 3, 2, 2], True, (2, 3, 2, 2)),
        ([-1, 2, 1, 3, 4], False, (2, 1, 3, 4)),
        ([2, 1, 3, -1, 2], True, (2, 3, 2, 2)),
    ],
)
def test_five_dimensional_shape_rules_use_original_snapshot(target, transpose, planned_shape):
    """Singleton collapse follows resolved shapes while later rules retain their snapshot."""
    nodes = [helper.make_node("Reshape", ["x", "target"], ["reshaped"])]
    planned = {"reshaped": planned_shape}
    output = "reshaped"
    if transpose:
        nodes.append(helper.make_node("Transpose", ["reshaped"], ["y"], perm=[0, 2, 1, 3, 4]))
        planned["y"] = planned_shape
        output = "y"
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "collapsed_layout",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 4])],
            [helper.make_tensor_value_info(output, TensorProto.FLOAT, None)],
            [numpy_helper.from_array(np.array(target, np.int64), "target")],
        ),
        opset_imports=[helper.make_opsetid("", 21)],
    )
    transformed, count = _lower_shapes(model, planned)
    assert count == len(nodes)
    feeds = {"x": np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4)}
    expected = ReferenceEvaluator(model).run(None, feeds)[0].reshape(planned_shape)
    np.testing.assert_array_equal(ReferenceEvaluator(transformed).run(None, feeds)[0], expected)


def _execute(model, feeds, use_ort):
    if not use_ort:
        return ReferenceEvaluator(model).run(None, feeds)[0]
    ort = pytest.importorskip("onnxruntime")
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    return ort.InferenceSession(
        model.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"]
    ).run(None, feeds)[0]


def _lower_indexing(model, planned):
    working = ir.serde.deserialize_model(onnx.shape_inference.infer_shapes(model, strict_mode=True))
    native = {
        n: (tuple(static_shape(v) for v in n.inputs), tuple(static_shape(v) for v in n.outputs))
        for n in working.graph
    }
    _, _, count = _apply_rules(
        working, native_shapes=native, planned_shapes=planned, group=INDEXING_RULES
    )
    x = working.graph.inputs[0]
    x.shape = ir.Shape((1,) * (4 - len(x.shape)) + tuple(x.shape))
    for value in [*working.graph.inputs, *[v for n in working.graph for v in n.outputs]]:
        if value.name in planned:
            value.shape = ir.Shape(planned[value.name])
    ir.passes.common.RemoveUnusedNodesPass()(working)
    result = onnx.shape_inference.infer_shapes(ir.serde.serialize_model(working), strict_mode=True)
    onnx.checker.check_model(result, full_check=True)
    return result, count


@pytest.mark.parametrize(
    ("kind", "index_values", "axis", "runtime", "promote_indices"),
    [
        ("Gather", [2, 0], 1, False, False),
        ("Gather", [[0, 1]], 0, False, False),
        ("Gather", 0, 0, False, False),
        ("GatherElements", [[2, 0], [1, 2]], -1, False, False),
        ("GatherElements", [[2, 0], [1, 2]], 1, True, False),
        ("GatherElements", [[2, 0], [1, 2]], 1, True, True),
    ],
)
def test_index_ranks(kind, index_values, axis, runtime, promote_indices):
    """Gather keeps native controls; GatherElements receives one matching index view."""
    indices = np.asarray(index_values, np.int32)
    shape = [1, 3] if indices.ndim == 0 else [2, 3]
    inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)]
    inits = []
    if runtime:
        inputs.append(helper.make_tensor_value_info("indices", TensorProto.INT32, indices.shape))
    else:
        inits.append(numpy_helper.from_array(indices, "indices"))
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node(kind, ["x", "indices"], ["y"], axis=axis)],
            "index_layout",
            inputs,
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
            inits,
        ),
        opset_imports=[helper.make_opsetid("", 21)],
        ir_version=10,
    )
    feeds = {"x": np.arange(np.prod(shape), dtype=np.float32).reshape(shape)}
    if runtime:
        feeds["indices"] = indices
    # ReferenceEvaluator 1.23.2 does not normalize negative GatherElements axes.
    use_ort = kind == "GatherElements"
    expected = _execute(model, feeds, use_ort)
    target = (1,) * (4 - expected.ndim) + expected.shape
    planned = {"y": target}
    if promote_indices:
        planned["indices"] = (1, 1, *indices.shape)
    transformed, count = _lower_indexing(model, planned)
    assert count == 1
    feeds["x"] = feeds["x"].reshape(1, 1, *shape)
    if promote_indices:
        feeds["indices"] = indices.reshape(1, 1, *indices.shape)
    np.testing.assert_array_equal(_execute(transformed, feeds, use_ort), expected.reshape(target))


@pytest.mark.parametrize(
    ("kind", "version", "axes", "keepdims"),
    [
        ("ReduceSum", 17, [1], 0),
        ("ReduceMean", 17, [-1], 1),
        ("ReduceMax", 21, [0], 0),
        ("ReduceMin", 21, None, 0),
    ],
)
def test_reduction_axes_and_output_restore(kind, version, axes, keepdims):
    """Reductions preserve values and output rank across supported schema forms."""
    operands, inits, attrs = ["x"], [], {"keepdims": keepdims}
    if axes is not None:
        if kind == "ReduceSum" or version >= 18:
            operands.append("axes")
            inits.append(numpy_helper.from_array(np.asarray(axes, np.int64), "axes"))
        else:
            attrs["axes"] = axes
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node(kind, operands, ["y"], **attrs)],
            "reduction_layout",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
            inits,
        ),
        opset_imports=[helper.make_opsetid("", version)],
    )
    x = np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 8
    expected = ReferenceEvaluator(model).run(None, {"x": x})[0]
    target = (1,) * (4 - expected.ndim) + expected.shape
    transformed, count = _lower_indexing(model, {"y": target})
    assert count == 1
    actual = ReferenceEvaluator(transformed).run(None, {"x": x.reshape(1, 2, 3, 4)})[0]
    np.testing.assert_array_equal(actual, expected.reshape(target))


def test_shared_reduction_axes_get_separate_live_owners():
    """Rewritten reductions each retain their own axes while sharing data safely."""
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("ReduceSum", ["x", "axes"], [name], keepdims=1)
                for name in ("y", "z")
            ],
            "shared_axes",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4])],
            [
                helper.make_tensor_value_info(name, TensorProto.FLOAT, [2, 1, 4])
                for name in ("y", "z")
            ],
            [numpy_helper.from_array(np.array([1], np.int64), "axes")],
        ),
        opset_imports=[helper.make_opsetid("", 21)],
    )
    transformed, count = _lower_indexing(model, {"y": (1, 2, 1, 4), "z": (1, 2, 1, 4)})
    assert count == 2
    axes = [node.input[1] for node in transformed.graph.node if node.op_type == "ReduceSum"]
    assert len(set(axes)) == 2
    x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    for actual, expected in zip(
        ReferenceEvaluator(transformed).run(None, {"x": x.reshape(1, 2, 3, 4)}),
        ReferenceEvaluator(model).run(None, {"x": x}),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected.reshape(1, 2, 1, 4))


def _native_shape_snapshot(model):
    return {
        node: (
            tuple(static_shape(value) for value in node.inputs),
            tuple(static_shape(value) for value in node.outputs),
        )
        for node in model.graph
    }


def test_instance_normalization_channel_layout():
    graph = helper.make_graph(
        [
            helper.make_node("InstanceNormalization", ["x", "scale", "bias"], ["y"]),
        ],
        "instance",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3, 4])],
        [
            numpy_helper.from_array(np.array([1, 2, 3], np.float32), "scale"),
            numpy_helper.from_array(np.array([0.1, -0.1, 0.2], np.float32), "bias"),
        ],
    )
    source = onnx.shape_inference.infer_shapes(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)], ir_version=10)
    )
    model = ir.serde.deserialize_model(source)
    planned = {"x": (1, 2, 3, 4), "y": (1, 2, 3, 4)}
    _, _, count = _apply_rules(
        model,
        native_shapes=_native_shape_snapshot(model),
        planned_shapes=planned,
        group=NATIVE_RULES,
    )
    assert count == 1
    model.graph.inputs[0].shape = ir.Shape(planned["x"])
    model.graph.outputs[0].shape = ir.Shape(planned["y"])
    result = onnx.shape_inference.infer_shapes(ir.serde.serialize_model(model), strict_mode=True)
    onnx.checker.check_model(result, full_check=True)
    x = (np.arange(24, dtype=np.float32).reshape(2, 3, 4) % 7) / 8
    np.testing.assert_allclose(
        ReferenceEvaluator(source).run(None, {"x": x})[0],
        ReferenceEvaluator(result).run(None, {"x": x.reshape(1, 2, 3, 4)})[0].reshape(2, 3, 4),
        rtol=2e-6,
    )
    assert [node.op_type for node in result.graph.node] == [
        "Reshape",
        "InstanceNormalization",
        "Reshape",
    ]
    middle = list(model.graph)[1]
    assert static_shape(middle.inputs[0]) == (2, 3, 4, 1)


@pytest.mark.parametrize(
    ("output_count", "packed"), [(1, False), (2, False), (3, False), (3, True)]
)
def test_attention_native_ports(output_count, packed):
    shape = [1, 2, 8]
    inputs = (
        [helper.make_tensor_value_info("q", TensorProto.FLOAT, [1, 2, 24])]
        if packed
        else [
            helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
            for name in ("q", "k", "v")
        ]
    )
    inputs.extend(
        [
            helper.make_tensor_value_info("lengths", TensorProto.INT32, [1]),
            helper.make_tensor_value_info("total", TensorProto.INT32, []),
        ]
    )
    output_shapes = [[1, 2, 8], [1, 2, 2, 4], [1, 2, 2, 4]]
    graph = helper.make_graph(
        [
            helper.make_node(
                "GroupQueryAttention",
                ["q", "" if packed else "k", "" if packed else "v", "", "", "lengths", "total"],
                ["y", "present_k", "present_v"][:output_count],
                domain="com.microsoft",
                num_heads=2,
                kv_num_heads=2,
            ),
        ],
        "attention",
        inputs,
        [
            helper.make_tensor_value_info(name, TensorProto.FLOAT, dims)
            for name, dims in zip(
                ("y", "present_k", "present_v")[:output_count],
                output_shapes[:output_count],
                strict=True,
            )
        ],
    )
    source = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
        ir_version=10,
    )
    model = ir.serde.deserialize_model(source)
    native, planned = snapshot(model)
    context, rules, count = _apply_rules(
        model, native_shapes=native, planned_shapes=planned, group=NATIVE_RULES
    )
    assert count == 1
    assert context.apply_patterns(rules) == 0
    for value in model.graph.inputs:
        if value.name in planned:
            value.shape = ir.Shape(planned[value.name])
    model.graph.outputs[0].shape = ir.Shape(planned["y"])
    result = ir.serde.serialize_model(model)
    onnx.checker.check_model(result, full_check=True)
    attention = next(node for node in model.graph if node.op_type == "GroupQueryAttention")
    assert [static_shape(value) for value in attention.inputs[:3]] == (
        [(1, 2, 24), None, None] if packed else [tuple(shape)] * 3
    )
    assert attention.inputs[3:5] == (None, None)
    assert [value.name for value in attention.inputs[5:]] == ["lengths", "total"]
    assert [static_shape(value) for value in attention.outputs] == [
        tuple(dims) for dims in output_shapes[:output_count]
    ]
    assert [static_shape(value) for value in model.graph.outputs] == [
        (1, 1, 2, 8),
        (1, 2, 2, 4),
        (1, 2, 2, 4),
    ][:output_count]


@pytest.mark.parametrize("strong", [False, True])
def test_normalization_promotion(strong):
    graph = helper.make_graph(
        [
            helper.make_node("LpNormalization", ["x"], ["norm"], axis=1, p=2),
            helper.make_node("Mul", ["norm", "gamma"], ["y"]),
        ],
        "normalization",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4])],
        [numpy_helper.from_array(np.array(1.25, np.float32), "gamma")],
    )
    source = onnx.shape_inference.infer_shapes(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)], ir_version=10)
    )
    model = ir.serde.deserialize_model(source)
    native = {
        node: (
            tuple(static_shape(value) for value in node.inputs),
            tuple(static_shape(value) for value in node.outputs),
        )
        for node in model.graph
    }
    planned = dict.fromkeys(("x", "norm", "y"), (1, 1, 2, 4))
    _, _, count = _apply_rules(
        model,
        native_shapes=native,
        planned_shapes=planned,
        group=NORMALIZATION_RULES,
        force_fp16_activations=strong,
    )
    assert count == 1
    model.graph.inputs[0].shape = ir.Shape(planned["x"])
    model.graph.outputs[0].shape = ir.Shape(planned["y"])
    for node in model.graph:
        for value in node.outputs:
            if value.name in planned:
                value.shape = ir.Shape(planned[value.name])
    result = onnx.shape_inference.infer_shapes(ir.serde.serialize_model(model), strict_mode=True)
    onnx.checker.check_model(result, full_check=True)
    x = np.arange(1, 9, dtype=np.float32).reshape(2, 4) / 8
    np.testing.assert_allclose(
        ReferenceEvaluator(source).run(None, {"x": x})[0],
        ReferenceEvaluator(result).run(None, {"x": x.reshape(1, 1, 2, 4)})[0].reshape(2, 4),
        rtol=2e-6,
    )
    if strong:
        assert [node.op_type for node in result.graph.node] == [
            "Mul",
            "ReduceSum",
            "Sqrt",
            "Max",
            "Div",
            "Mul",
        ]
        assert any(
            np.array_equal(numpy_helper.to_array(value), np.array(2**-14, np.float32))
            for value in result.graph.initializer
        )
    else:
        assert next(iter(model.graph)).attributes["axis"].value == -1
