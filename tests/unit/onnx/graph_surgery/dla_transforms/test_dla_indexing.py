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

"""Independent numerical and boundary-contract checks for indexing passes."""

import copy

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

from modelopt.onnx.graph_surgery.dla_transforms.dla_indexing import (
    _apply_gather,
    _apply_gather_elements,
    _apply_group_query_attention,
    _apply_standalone_reduce_operations,
)
from modelopt.onnx.graph_surgery.dla_transforms.dla_model_boundaries import (
    _apply_non4d_model_inputs,
    _apply_non4d_model_outputs,
)
from modelopt.onnx.graph_surgery.make_dla_compatible import _transform_make_dla_compatible


@pytest.mark.parametrize("operation", ["Gather", "GatherElements"])
def test_index_dtype_and_rank_conversion_preserves_selected_values(operation):
    """Index dtype and rank conversion preserves selected values."""
    if operation == "Gather":
        shape, output_shape, axis = [5, 4], [4], 0
        indices = np.int64(2)
        transform = _apply_gather
    else:
        shape, output_shape, axis = [2, 3, 4], [2, 3, 4], 1
        indices = np.ones(shape, dtype=np.int64)
        transform = _apply_gather_elements
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node(operation, ["x", "indices"], ["y"], name="index", axis=axis)],
            "indexing",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, output_shape)],
            initializer=[numpy_helper.from_array(np.asarray(indices), "indices")],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 9
    candidate = transform(copy.deepcopy(model))
    candidate = _apply_non4d_model_inputs(candidate)
    candidate = _apply_non4d_model_outputs(candidate, defer_rank_conversion=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    expected = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    ).run(None, {"x": data})
    actual = ort.InferenceSession(
        candidate.SerializeToString(), options, providers=["CPUExecutionProvider"]
    ).run(None, {"x": data})
    np.testing.assert_array_equal(actual[0], expected[0])


@pytest.mark.parametrize("direct_output", [False, True])
@pytest.mark.parametrize("legacy_ir", [False, True])
def test_attention_preserves_graph_boundaries_and_wraps_internal_tensors(direct_output, legacy_ir):
    """Attention preserves graph boundaries and wraps internal tensors."""
    output = "attention" if direct_output else "y"
    nodes = [
        helper.make_node("Unsqueeze", ["x", "axes"], ["q4"]),
        helper.make_node(
            "GroupQueryAttention",
            ["q4", "x", "x", "", ""],
            ["attention"],
            name="attention",
            domain="com.microsoft",
            num_heads=1,
            kv_num_heads=1,
        ),
    ]
    if not direct_output:
        nodes.append(helper.make_node("Identity", ["attention"], ["y"]))
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "attention",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, 2, 4])],
            [
                helper.make_tensor_value_info(
                    output, TensorProto.FLOAT16, [1, 2, 4] if direct_output else [1, 1, 2, 4]
                )
            ],
            initializer=[numpy_helper.from_array(np.array([0], dtype=np.int64), "axes")],
        ),
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)],
    )
    if legacy_ir:
        model.ir_version = 3
        for tensor in model.graph.initializer:
            model.graph.input.append(
                helper.make_tensor_value_info(tensor.name, tensor.data_type, list(tensor.dims))
            )
    candidate = _apply_group_query_attention(model)
    onnx.checker.check_model(candidate)
    attention = next(n for n in candidate.graph.node if n.op_type == "GroupQueryAttention")
    producer = {name: node for node in candidate.graph.node for name in node.output}
    assert producer[attention.input[0]].op_type == "Squeeze"
    assert list(attention.input[1:]) == ["x", "x", "", ""]
    assert producer["attention"].op_type == (
        "GroupQueryAttention" if direct_output else "Unsqueeze"
    )


@pytest.mark.parametrize("axes_default", [False, True])
def test_late_refusal_preserves_the_original_model(axes_default):
    """Late refusal preserves the original model."""
    nodes = [
        helper.make_node("ReduceSum", ["x"], ["first"], keepdims=0),
        helper.make_node("ReduceSum", ["x", "dynamic_axes"], ["y"], keepdims=0),
    ]
    inputs = [
        helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3]),
        helper.make_tensor_value_info("dynamic_axes", TensorProto.INT64, [1]),
    ]
    outputs = [helper.make_tensor_value_info("y", TensorProto.FLOAT, [3])]
    model = helper.make_model(
        helper.make_graph(nodes, "refusal", inputs, outputs),
        opset_imports=[helper.make_opsetid("", 21)],
    )
    if axes_default:
        model.graph.initializer.append(
            numpy_helper.from_array(np.asarray([0], np.int64), "dynamic_axes")
        )
    before = model.SerializeToString()
    with pytest.raises(ValueError):
        _apply_standalone_reduce_operations(model)
    assert model.SerializeToString() == before


@pytest.mark.parametrize("axes", [[-2], [-1, 0], [0, -1]])
def test_reduction_preserves_negative_and_mixed_axes(axes):
    """Reduction preserves negative and mixed axes."""
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("ReduceSum", ["x", "axes"], ["y"], keepdims=int(len(axes) > 1))],
            "reduce",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
            [numpy_helper.from_array(np.array(axes, dtype=np.int64), "axes")],
        ),
        opset_imports=[helper.make_opsetid("", 21)],
    )
    model = onnx.shape_inference.infer_shapes(model)
    data = np.arange(6, dtype=np.float32).reshape(2, 3)
    expected = ReferenceEvaluator(model).run(None, {"x": data})[0]
    candidate = _apply_non4d_model_inputs(copy.deepcopy(model))
    candidate = _apply_non4d_model_outputs(candidate, defer_rank_conversion=True)
    candidate = _apply_standalone_reduce_operations(candidate)
    onnx.checker.check_model(candidate)
    actual = ReferenceEvaluator(candidate).run(None, {"x": data})[0]
    np.testing.assert_array_equal(actual, expected)


def _model(nodes, inputs, outputs, initializers=(), extra_opsets=()):
    """Build and fully check a small indexing model with typed public interfaces."""
    model = helper.make_model(
        helper.make_graph(nodes, "index_contract", inputs, outputs, initializer=initializers),
        opset_imports=[helper.make_opsetid("", 21), *extra_opsets],
    )
    model.ir_version = 9
    onnx.checker.check_model(model, full_check=True)
    return model


def _execute(model, feeds):
    """Run CPU graph semantics without backend fusion changing the comparison."""
    onnx.checker.check_model(model, full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    ).run(None, feeds)


@pytest.mark.parametrize("axis", [None, 0, 1, 2, -1, -2, -3])
@pytest.mark.parametrize("kind", ["constant", "dynamic", "overridable"])
def test_gather_elements_rank_three_preserves_axes_and_runtime_indices(axis, kind):
    """Preserve positive, negative and default axes with fixed or runtime indices."""
    shape = [2, 3, 4]
    normalized = (axis or 0) % 3
    indices = (np.arange(24).reshape(shape) % shape[normalized]).astype(np.int64)
    inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)]
    initializers = []
    feeds = {"x": np.arange(24, dtype=np.float32).reshape(shape)}
    if kind != "constant":
        inputs.append(helper.make_tensor_value_info("indices", TensorProto.INT64, shape))
        feeds["indices"] = (indices + 1) % shape[normalized]
    if kind != "dynamic":
        initializers.append(numpy_helper.from_array(indices, "indices"))
    attributes = {} if axis is None else {"axis": axis}
    model = _model(
        [helper.make_node("GatherElements", ["x", "indices"], ["y"], **attributes)],
        inputs,
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)],
        initializers,
    )
    expected = _execute(model, feeds)[0]
    transformed = _transform_make_dla_compatible(copy.deepcopy(model), verbose=False)
    np.testing.assert_array_equal(_execute(transformed, feeds)[0], expected)


@pytest.mark.parametrize("axis", [None, -2])
@pytest.mark.parametrize("dynamic", [False, True])
def test_gather_elements_rank_two_preserves_selected_values(axis, dynamic):
    """Keep data and index ranks aligned when two leading dimensions are introduced."""
    shape = [2, 3]
    indices = (np.arange(6).reshape(shape) % 2).astype(np.int64)
    inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)]
    initializers = []
    feeds = {"x": np.arange(6, dtype=np.float32).reshape(shape)}
    if dynamic:
        inputs.append(helper.make_tensor_value_info("indices", TensorProto.INT64, shape))
        feeds["indices"] = indices
    else:
        initializers.append(numpy_helper.from_array(indices, "indices"))
    attrs = {} if axis is None else {"axis": axis}
    model = _model(
        [helper.make_node("GatherElements", ["x", "indices"], ["y"], **attrs)],
        inputs,
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)],
        initializers,
    )
    expected = _execute(model, feeds)[0]
    transformed = _transform_make_dla_compatible(copy.deepcopy(model), verbose=False)
    np.testing.assert_array_equal(_execute(transformed, feeds)[0], expected)


def test_shared_dynamic_gather_elements_indices_have_one_preparation():
    """Reuse a valid dynamic index view across two GatherElements consumers."""
    shape = [2, 3, 4]
    inputs = [
        helper.make_tensor_value_info(name, TensorProto.FLOAT, shape) for name in ["a", "b"]
    ] + [helper.make_tensor_value_info("indices", TensorProto.INT64, shape)]
    outputs = [
        helper.make_tensor_value_info(name, TensorProto.FLOAT, shape) for name in ["a_out", "b_out"]
    ]
    model = _model(
        [
            helper.make_node("GatherElements", ["a", "indices"], ["a_out"], axis=1),
            helper.make_node("GatherElements", ["b", "indices"], ["b_out"], axis=1),
        ],
        inputs,
        outputs,
    )
    feeds = {
        "a": np.arange(24, dtype=np.float32).reshape(shape),
        "b": -np.arange(24, dtype=np.float32).reshape(shape),
        "indices": (np.arange(24).reshape(shape) % 3).astype(np.int64),
    }
    expected = _execute(model, feeds)
    transformed = _transform_make_dla_compatible(copy.deepcopy(model), verbose=False)
    for actual, reference in zip(_execute(transformed, feeds), expected):
        np.testing.assert_array_equal(actual, reference)


@pytest.mark.parametrize("keepdims", [False, True])
def test_unnamed_reductions_keep_distinct_axes(keepdims):
    """Allocate different controls for unnamed reductions with different axes."""
    shape = [2, 3, 4]
    data = np.arange(24, dtype=np.float32).reshape(shape)
    outputs = [
        helper.make_tensor_value_info(
            "a", TensorProto.FLOAT, list(data.sum(axis=0, keepdims=keepdims).shape)
        ),
        helper.make_tensor_value_info(
            "b", TensorProto.FLOAT, list(data.sum(axis=1, keepdims=keepdims).shape)
        ),
    ]
    model = _model(
        [
            helper.make_node("ReduceSum", ["x", "axes_a"], ["a"], keepdims=int(keepdims)),
            helper.make_node("ReduceSum", ["x", "axes_b"], ["b"], keepdims=int(keepdims)),
        ],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
        outputs,
        [
            numpy_helper.from_array(np.asarray([0], np.int64), "axes_a"),
            numpy_helper.from_array(np.asarray([1], np.int64), "axes_b"),
        ],
    )
    expected = _execute(model, {"x": data})
    transformed = _transform_make_dla_compatible(copy.deepcopy(model), verbose=False)
    for actual, reference in zip(_execute(transformed, {"x": data}), expected):
        np.testing.assert_array_equal(actual, reference)


def test_reduction_generated_shape_preserves_existing_public_initializer():
    """Reserve reshape targets outside an unrelated public initializer namespace."""
    model = _model(
        [helper.make_node("ReduceSum", ["x", "axes"], ["y"], keepdims=0)],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
        [
            helper.make_tensor_value_info("y", TensorProto.FLOAT, [3]),
            helper.make_tensor_value_info("reshape_shape_0", TensorProto.INT64, [1]),
        ],
        [
            numpy_helper.from_array(np.asarray([0], np.int64), "axes"),
            numpy_helper.from_array(np.asarray([97], np.int64), "reshape_shape_0"),
        ],
    )
    feeds = {"x": np.arange(6, dtype=np.float32).reshape(2, 3)}
    expected = _execute(model, feeds)
    transformed = _transform_make_dla_compatible(copy.deepcopy(model), verbose=False)
    for actual, reference in zip(_execute(transformed, feeds), expected):
        np.testing.assert_array_equal(actual, reference)


@pytest.mark.parametrize("collision", [False, True])
@pytest.mark.parametrize("cache_views", [False, True])
def test_chained_attention_preserves_native_kernel_inputs_and_values(collision, cache_views):
    """Run two real CPU attention kernels across the internal rank adapters."""
    shape = [1, 2, 8]
    nodes = [
        helper.make_node(
            "GroupQueryAttention",
            ["q", "k", "v", "past_key", "past_value", "lengths", "total"],
            ["middle", "pk1", "pv1"],
            domain="com.microsoft",
            name="first",
            num_heads=1,
            kv_num_heads=1,
        ),
        helper.make_node(
            "GroupQueryAttention",
            ["middle", "k", "v", "past_key", "past_value", "lengths", "total"],
            ["y", "pk2", "pv2"],
            domain="com.microsoft",
            name="second",
            num_heads=1,
            kv_num_heads=1,
        ),
    ]
    model = _model(
        nodes,
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, shape) for name in ["q", "k", "v"]],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)],
        [
            numpy_helper.from_array(np.asarray([1], np.int32), "lengths"),
            numpy_helper.from_array(np.int32(2), "total"),
            numpy_helper.from_array(np.zeros((1, 1, 2, 8), np.float32), "past_key"),
            numpy_helper.from_array(np.zeros((1, 1, 2, 8), np.float32), "past_value"),
        ],
        [helper.make_opsetid("com.microsoft", 1)],
    )
    rng = np.random.default_rng(15)
    feeds = {name: rng.standard_normal(shape).astype(np.float32) for name in ["q", "k", "v"]}
    if collision:
        model.graph.initializer.append(
            numpy_helper.from_array(np.asarray([2, 3], np.int64), "middle_axis")
        )
        model.graph.output.append(
            helper.make_tensor_value_info("middle_axis", TensorProto.INT64, [2])
        )
    if cache_views:
        attention = [node for node in model.graph.node if node.op_type == "GroupQueryAttention"]
        for name in ["past_key", "past_value"]:
            model.graph.input.append(
                helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 1, 2, 8])
            )
            feeds[name] = np.arange(16, dtype=np.float32).reshape(1, 1, 2, 8) / 16
            model.graph.node.insert(0, helper.make_node("Identity", [name], [name + "_internal"]))
        for node in attention:
            node.input[3] = "past_key_internal"
            node.input[4] = "past_value_internal"
        next(value for value in model.graph.input if value.name == "q").CopyFrom(
            helper.make_tensor_value_info("q", TensorProto.FLOAT, [1, *shape])
        )
        feeds["q"] = feeds["q"].reshape(1, *shape)
        model.graph.initializer.append(
            numpy_helper.from_array(np.asarray([0], np.int64), "public_q_axes")
        )
        model.graph.node.insert(
            0, helper.make_node("Squeeze", ["q", "public_q_axes"], ["public_q"])
        )
        model.graph.output.append(
            helper.make_tensor_value_info("public_q", TensorProto.FLOAT, shape)
        )
        attention[0].input[0] = "public_q"
        for name in ["pk1", "pv1"]:
            model.graph.node.append(helper.make_node("Identity", [name], [name + "_public"]))
            model.graph.output.append(
                helper.make_tensor_value_info(name + "_public", TensorProto.FLOAT, [1, 1, 2, 8])
            )
    expected = _execute(model, feeds)
    if cache_views:
        assert expected[-1].shape == expected[-2].shape == (1, 1, 2, 8)
    transformed = _transform_make_dla_compatible(copy.deepcopy(model), verbose=False)
    for actual, reference in zip(_execute(transformed, feeds), expected):
        np.testing.assert_allclose(actual, reference, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "transform",
    [
        _apply_gather,
        _apply_gather_elements,
        _apply_standalone_reduce_operations,
        _apply_group_query_attention,
    ],
)
def test_indexing_keeps_untouched_constant_payload_in_place(transform):
    """Avoid copying a Constant payload when the stage only changes other operations."""
    model = _model(
        [
            helper.make_node(
                "Constant",
                [],
                ["constant"],
                value=numpy_helper.from_array(np.arange(8, dtype=np.float32)),
            ),
            helper.make_node("Identity", ["x"], ["y"]),
        ],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 4])],
        [
            helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 3, 4]),
            helper.make_tensor_value_info("constant", TensorProto.FLOAT, [8]),
        ],
    )
    constant = model.graph.node[0]
    feeds = {"x": np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4)}
    expected = _execute(model, feeds)
    transform(model)
    assert model.graph.node[0] is constant
    for actual, reference in zip(_execute(model, feeds), expected):
        np.testing.assert_array_equal(actual, reference)


@pytest.mark.parametrize("producer", ["Cast", "Identity"])
def test_intermediate_indices_keep_rank_when_upstream_boundary_is_promoted(producer):
    """Preserve index ranks when a shape-preserving producer already receives promoted data."""
    shape = [2, 3, 4]
    first = helper.make_node(
        producer,
        ["raw_indices"],
        ["indices"],
        **({"to": TensorProto.INT64} if producer == "Cast" else {}),
    )
    model = _model(
        [first, helper.make_node("GatherElements", ["x", "indices"], ["y"], axis=1)],
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, shape),
            helper.make_tensor_value_info("raw_indices", TensorProto.INT64, shape),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)],
    )
    feeds = {
        "x": np.arange(24, dtype=np.float32).reshape(shape),
        "raw_indices": (np.arange(24).reshape(shape) % 3).astype(np.int64),
    }
    expected = _execute(model, feeds)[0]
    transformed = _transform_make_dla_compatible(copy.deepcopy(model), verbose=False)
    np.testing.assert_array_equal(_execute(transformed, feeds)[0], expected)


@pytest.mark.parametrize(
    ("operator", "transform"),
    [
        ("Gather", _apply_gather),
        ("GatherElements", _apply_gather_elements),
        ("ReduceSum", _apply_standalone_reduce_operations),
        ("GroupQueryAttention", _apply_group_query_attention),
    ],
)
def test_indexing_preserves_opaque_model_local_operators(operator, transform):
    """Leave custom functions intact when their names match indexing-stage operators."""
    shape = [1, 2, 3]
    attributes = {"keepdims": 0} if operator == "ReduceSum" else {}
    model = _model(
        [
            helper.make_node("Identity", ["x"], ["middle"]),
            helper.make_node(operator, ["middle"], ["y"], domain="custom", **attributes),
        ],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)],
        extra_opsets=[helper.make_opsetid("custom", 1)],
    )
    model.functions.append(
        helper.make_function(
            "custom",
            operator,
            ["X"],
            ["Y"],
            [helper.make_node("Neg", ["X"], ["Y"])],
            [helper.make_opsetid("", 17)],
            attributes=list(attributes),
        )
    )
    model = onnx.shape_inference.infer_shapes(model)
    feeds = {"x": np.arange(6, dtype=np.float32).reshape(shape)}
    # Use the independent interpreter for opaque custom-function semantics.
    expected = ReferenceEvaluator(model).run(None, feeds)
    before = model.SerializeToString()
    transform(model)
    assert model.SerializeToString() == before
    onnx.checker.check_model(model, full_check=True)
    np.testing.assert_array_equal(ReferenceEvaluator(model).run(None, feeds)[0], expected[0])


@pytest.mark.parametrize(
    "indices", [np.int64(1), np.array([1], np.int64), np.array([0, 2], np.int64)]
)
@pytest.mark.parametrize("producer", [False, True])
def test_gather_preserves_overridable_scalar_and_vector_indices(indices, producer):
    """Execute runtime Gather overrides while retaining scalar and singleton-vector ranks."""
    indices = np.asarray(indices)
    model = _model(
        [helper.make_node("Gather", ["x", "indices"], ["y"], axis=0)],
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [3, 4]),
            helper.make_tensor_value_info("indices", TensorProto.INT64, list(indices.shape)),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [*indices.shape, 4])],
        [numpy_helper.from_array(indices, "indices")],
    )
    if producer:
        model.graph.node.insert(
            0, helper.make_node("Cast", ["indices"], ["cast_indices"], to=TensorProto.INT64)
        )
        model.graph.node[1].input[1] = "cast_indices"
    feeds = {
        "x": np.arange(12, dtype=np.float32).reshape(3, 4),
        "indices": np.asarray((indices + 1) % 3, dtype=np.int64),
    }
    expected = _execute(model, feeds)[0]
    transformed = _transform_make_dla_compatible(copy.deepcopy(model), verbose=False)
    np.testing.assert_array_equal(_execute(transformed, feeds)[0], expected)


def test_reduction_registers_legacy_initializer_inputs():
    """Keep IR3 controls declared while preserving reduction output values and ranks."""
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("ReduceMean", ["x"], ["y"], axes=[2], keepdims=0)],
            "legacy_reduce",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 2, 3])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 3])],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 3
    feeds = {"x": np.arange(6, dtype=np.float32).reshape(1, 1, 2, 3)}
    expected = _execute(model, feeds)[0]
    transformed = _apply_non4d_model_outputs(copy.deepcopy(model), defer_rank_conversion=True)
    transformed = _apply_standalone_reduce_operations(transformed)
    np.testing.assert_array_equal(_execute(transformed, feeds)[0], expected)


def test_gather_preserves_existing_public_initializer_name():
    """Allocate private Gather controls without overwriting a public initializer."""
    model = _model(
        [helper.make_node("Gather", ["x", "indices"], ["y"], axis=0)],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [3, 4])],
        [
            helper.make_tensor_value_info("y", TensorProto.FLOAT, [4]),
            helper.make_tensor_value_info("indices_gather_indices", TensorProto.INT64, [1]),
        ],
        [
            numpy_helper.from_array(np.int64(1), "indices"),
            numpy_helper.from_array(np.asarray([17], np.int64), "indices_gather_indices"),
        ],
    )
    feeds = {"x": np.arange(12, dtype=np.float32).reshape(3, 4)}
    expected = _execute(model, feeds)
    transformed = _transform_make_dla_compatible(copy.deepcopy(model), verbose=False)
    for actual, reference in zip(_execute(transformed, feeds), expected):
        np.testing.assert_array_equal(actual, reference)
