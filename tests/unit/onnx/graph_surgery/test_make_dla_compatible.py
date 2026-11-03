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

"""Exercise complete public DLA pipelines on small, independently built graphs."""

import copy
import sys

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto as T
from onnx import helper as h
from onnx import numpy_helper as nh
from onnx.reference import ReferenceEvaluator
from onnxscript.testing import assert_isomorphic_graph

from modelopt.onnx.graph_surgery.__main__ import main as cli_main
from modelopt.onnx.graph_surgery.make_dla_compatible import (
    _transform_make_dla_compatible,
    dla_make_dla_compatible,
)


def _matrix_model():
    """Compose matrix lowering, axis promotion and a public native-rank endpoint."""
    graph = h.make_graph(
        [h.make_node("MatMul", ["x", "w"], ["m"]), h.make_node("Softmax", ["m"], ["y"], axis=1)],
        "matrix_pipeline",
        [h.make_tensor_value_info("x", T.FLOAT, [2, 3])],
        [h.make_tensor_value_info("y", T.FLOAT, [2, 4])],
        [nh.from_array(np.arange(12, dtype=np.float32).reshape(3, 4) / 16, "w")],
    )
    return h.make_model(graph, opset_imports=[h.make_opsetid("", 18)], ir_version=10)


def _expected_matrix_graph():
    """Specify the complete FP32 result independently of any production rewrite."""
    return h.make_graph(
        [
            h.make_node("Unsqueeze", ["x", "axes"], ["promoted"]),
            h.make_node("Transpose", ["promoted"], ["channels"], perm=[0, 3, 2, 1]),
            h.make_node("Conv", ["channels", "kernel"], ["convolved"]),
            h.make_node("Transpose", ["convolved"], ["matrix"], perm=[0, 3, 2, 1]),
            h.make_node("Softmax", ["matrix"], ["normalized"], axis=3),
            h.make_node("Squeeze", ["normalized", "axes"], ["y"]),
        ],
        "expected",
        [h.make_tensor_value_info("x", T.FLOAT, [2, 3])],
        [h.make_tensor_value_info("y", T.FLOAT, [2, 4])],
        [
            nh.from_array(np.array([0, 1], np.int64), "axes"),
            nh.from_array(
                np.arange(12, dtype=np.float32).reshape(3, 4).T.reshape(4, 3, 1, 1) / 16, "kernel"
            ),
        ],
    )


def _constants_graph(graph):
    """Expose initializer payloads to the SDK's graph-isomorphism assertion."""
    graph = copy.deepcopy(graph)
    nodes = [h.make_node("Constant", [], [value.name], value=value) for value in graph.initializer]
    nodes.extend(graph.node)
    graph.ClearField("node")
    graph.node.extend(nodes)
    graph.ClearField("initializer")
    return graph


@pytest.mark.parametrize("fp16", [False, True])
@pytest.mark.parametrize("entry", ["python", "cli"])
def test_complete_matrix_pipeline(tmp_path, monkeypatch, fp16, entry):
    """The file API preserves endpoints and executes the coordinated Conv/axis graph."""
    original = _matrix_model()
    source, target = tmp_path / "source.onnx", tmp_path / "converted.onnx"
    onnx.save(original, source)
    if entry == "cli":
        arguments = [
            "graph-surgery",
            "make-dla-compatible",
            "--input",
            str(source),
            "--output",
            str(target),
            "--no-external-data",
            "--quiet",
        ]
        if fp16:
            arguments.append("--force-fp16-activations")
        monkeypatch.setattr(sys, "argv", arguments)
        cli_main()
        actual = onnx.load(target)
    else:
        actual = dla_make_dla_compatible(
            str(source),
            str(target),
            use_external_data=False,
            force_fp16_activations=fp16,
            verbose=False,
        )
    onnx.checker.check_model(onnx.load(target), full_check=True)
    assert [node.op_type for node in actual.graph.node].count("Conv") == 1
    assert not any(node.op_type == "MatMul" for node in actual.graph.node)
    assert actual.graph.input[0] == original.graph.input[0]
    assert actual.graph.output[0] == original.graph.output[0]
    if not fp16:
        assert_isomorphic_graph(
            _constants_graph(actual.graph), _constants_graph(_expected_matrix_graph())
        )
    feeds = {"x": np.arange(6, dtype=np.float32).reshape(2, 3) / 8}
    expected = ReferenceEvaluator(original).run(None, feeds)[0]
    result = ReferenceEvaluator(actual).run(None, feeds)[0]
    np.testing.assert_allclose(result, expected, rtol=2e-3 if fp16 else 1e-6, atol=1e-6)


def test_shared_data_and_control_initializer():
    """A shape tensor's native owner survives promotion of its arithmetic use."""
    original = h.make_model(
        h.make_graph(
            [
                h.make_node("Reshape", ["x", "dims"], ["reshaped"]),
                h.make_node("Add", ["dims", "offset"], ["added"]),
                h.make_node("Identity", ["offset"], ["exposed"]),
            ],
            "shared_initializer",
            [h.make_tensor_value_info("x", T.FLOAT, [6])],
            [
                h.make_tensor_value_info("reshaped", T.FLOAT, [2, 3]),
                h.make_tensor_value_info("added", T.INT64, [2]),
                h.make_tensor_value_info("exposed", T.INT64, [2]),
            ],
            [
                nh.from_array(np.array([2, 3], np.int64), "dims"),
                nh.from_array(np.array([1, 1], np.int64), "offset"),
            ],
        ),
        opset_imports=[h.make_opsetid("", 18)],
        ir_version=10,
    )
    actual = _transform_make_dla_compatible(original, verbose=False)
    onnx.checker.check_model(actual, full_check=True)
    feeds = {"x": np.arange(6, dtype=np.float32)}
    for expected, result in zip(
        ReferenceEvaluator(original).run(None, feeds),
        ReferenceEvaluator(actual).run(None, feeds),
        strict=True,
    ):
        np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("mixed", [False, True])
def test_standard_domain_alias_preserves_input(mixed):
    """Canonicalize standard domains before inference without mutating the caller's model."""
    original = _matrix_model()
    original.opset_import[0].domain = "ai.onnx"
    original.graph.node[0].domain = "ai.onnx"
    if mixed:
        original.opset_import.append(h.make_opsetid("", 18))
    else:
        original.graph.node[1].domain = "ai.onnx"
    saved = original.SerializeToString()
    actual = _transform_make_dla_compatible(original, verbose=False)
    assert original.SerializeToString() == saved
    assert [(item.domain, item.version) for item in actual.opset_import] == [("", 18)]
    feeds = {"x": np.arange(6, dtype=np.float32).reshape(2, 3) / 8}
    np.testing.assert_array_equal(
        ReferenceEvaluator(actual).run(None, feeds)[0],
        ReferenceEvaluator(_matrix_model()).run(None, feeds)[0],
    )


@pytest.mark.parametrize(("first", "second"), [(18, 22), (22, 18), (17, 18)])
def test_conflicting_standard_domain_versions_refuse_without_mutation(first, second):
    """Aliased domain imports cannot silently choose a different standard schema."""
    original = _matrix_model()
    original.opset_import[0].version = first
    original.opset_import.append(h.make_opsetid("ai.onnx", second))
    saved = original.SerializeToString()
    with pytest.raises(ValueError):
        _transform_make_dla_compatible(original, verbose=False)
    assert original.SerializeToString() == saved


@pytest.mark.parametrize(
    ("opset", "options", "error"),
    [
        (22, {}, ValueError),
        (18, {"qdq_quantized_dtypes": "INT16"}, TypeError),
        (18, {"qdq_quantized_dtypes": b"INT16"}, TypeError),
        (18, {"qdq_quantized_dtypes": ["unknown"]}, ValueError),
        (18, {"qdq_quantized_dtypes": ["999"]}, ValueError),
        (18, {"qdq_quantized_dtypes": [1.5]}, ValueError),
        (18, {"qdq_quantized_dtypes": [], "remove_all_qdq": True}, ValueError),
    ],
)
def test_invalid_pipeline_options_refuse_without_mutation(opset, options, error):
    """Invalid schemas or quantization selections leave the supplied model intact."""
    original = _matrix_model()
    original.opset_import[0].version = opset
    saved = original.SerializeToString()
    with pytest.raises(error):
        _transform_make_dla_compatible(original, verbose=False, **options)
    assert original.SerializeToString() == saved


def test_rank_adapter_preserves_broadcast_layout():
    """A trailing Unsqueeze becomes a 4D Reshape before a broadcast multiply."""
    original = h.make_model(
        h.make_graph(
            [
                h.make_node("Unsqueeze", ["mask", "axes"], ["expanded"]),
                h.make_node("Mul", ["expanded", "x"], ["y"]),
                h.make_node("Squeeze", ["expanded", "axes"], ["restored"]),
            ],
            "rank_adapter",
            [
                h.make_tensor_value_info("mask", T.FLOAT, [2, 3]),
                h.make_tensor_value_info("x", T.FLOAT, [2, 3, 4]),
            ],
            [
                h.make_tensor_value_info("y", T.FLOAT, [2, 3, 4]),
                h.make_tensor_value_info("restored", T.FLOAT, [2, 3]),
            ],
            [nh.from_array(np.array([2], np.int64), "axes")],
        ),
        opset_imports=[h.make_opsetid("", 18)],
        ir_version=10,
    )
    actual = _transform_make_dla_compatible(original, verbose=False)
    onnx.checker.check_model(actual, full_check=True)
    assert any(node.op_type == "Reshape" for node in actual.graph.node)
    feeds = {
        "mask": np.arange(6, dtype=np.float32).reshape(2, 3),
        "x": np.arange(24, dtype=np.float32).reshape(2, 3, 4),
    }
    endpoint = next(node for node in actual.graph.node if list(node.output) == ["restored"])
    assert endpoint.op_type == "Squeeze"
    controls = {value.name: nh.to_array(value) for value in actual.graph.initializer}
    np.testing.assert_array_equal(controls[endpoint.input[1]], [0, 3])
    for result, expected in zip(
        ReferenceEvaluator(actual).run(None, feeds),
        ReferenceEvaluator(original).run(None, feeds),
        strict=True,
    ):
        np.testing.assert_array_equal(result, expected)


def test_recurrent_output_reaches_matrix_and_layout_rewrites():
    """A native LSTM decomposition feeds later matrix lowering and public axis handling."""
    original = h.make_model(
        h.make_graph(
            [
                h.make_node("LSTM", ["x", "w", "r"], ["sequence"], hidden_size=2),
                h.make_node("Transpose", ["sequence"], ["transposed"], perm=[0, 2, 1, 3]),
                h.make_node("Reshape", ["transposed", "dims"], ["matrix"]),
                h.make_node("MatMul", ["matrix", "projection"], ["projected"]),
                h.make_node("Softmax", ["projected"], ["y"], axis=1),
            ],
            "recurrent_pipeline",
            [h.make_tensor_value_info("x", T.FLOAT, [2, 1, 2])],
            [h.make_tensor_value_info("y", T.FLOAT, [2, 3])],
            [
                nh.from_array(np.arange(16, dtype=np.float32).reshape(1, 8, 2) / 64, "w"),
                nh.from_array(np.arange(16, dtype=np.float32).reshape(1, 8, 2) / 96, "r"),
                nh.from_array(np.array([-1, 2], np.int64), "dims"),
                nh.from_array(np.arange(6, dtype=np.float32).reshape(2, 3) / 8, "projection"),
            ],
        ),
        opset_imports=[h.make_opsetid("", 18)],
        ir_version=10,
    )
    actual = _transform_make_dla_compatible(original, verbose=False)
    assert not {"LSTM", "MatMul"} & {node.op_type for node in actual.graph.node}
    feeds = {"x": np.arange(4, dtype=np.float32).reshape(2, 1, 2) / 8}
    np.testing.assert_allclose(
        ReferenceEvaluator(actual).run(None, feeds)[0],
        ReferenceEvaluator(original).run(None, feeds)[0],
        rtol=1e-6,
        atol=1e-7,
    )


@pytest.mark.parametrize("fp16", [False, True])
def test_shared_multi_branch_pipeline(fp16):
    """Shared operands survive lowering, native-rank boundaries and optional FP16."""
    nodes = [
        h.make_node("Slice", ["x", "starts", "ends", "slice_axes"], ["sliced"]),
        h.make_node("Flatten", ["sliced"], ["flat"], axis=-1),
        h.make_node("MatMul", ["flat", "weight"], ["matrix_a"]),
        h.make_node("MatMul", ["flat", "weight"], ["matrix_b"]),
        h.make_node("Add", ["matrix_a", "matrix_b"], ["matrix"]),
        h.make_node("Gather", ["x", "indices"], ["gather_a"], axis=0),
        h.make_node("Gather", ["x", "indices"], ["gather_b"], axis=0),
        h.make_node("Gather", ["x", "other_indices"], ["gather_c"], axis=0),
        h.make_node("Add", ["gather_a", "gather_b"], ["gather_sum"]),
        h.make_node("Add", ["gather_sum", "gather_c"], ["gathered"]),
        h.make_node("Where", ["condition", "x", "zero"], ["masked_a"]),
        h.make_node("Where", ["condition", "x", "zero"], ["masked_b"]),
        h.make_node("Where", ["condition", "zero", "x"], ["inverse"]),
        h.make_node("Add", ["masked_a", "masked_b"], ["masked_sum"]),
        h.make_node("Add", ["masked_sum", "inverse"], ["masked"]),
        h.make_node("Greater", ["bytes", "byte_zero"], ["positive"]),
        h.make_node("ArgMax", ["x"], ["maximum"], axis=-1, keepdims=1),
        h.make_node("ReduceSum", ["x", "empty_axes"], ["unchanged"], noop_with_empty_axes=1),
        h.make_node("LpNormalization", ["x"], ["normalized"], axis=0),
        h.make_node("Mul", ["normalized", "gain"], ["scaled"]),
        h.make_node("Add", ["normalized", "scaled"], ["normalization"]),
        h.make_node("DequantizeLinear", ["bias_q", "scale", "zero_q"], ["bias_a"]),
        h.make_node("DequantizeLinear", ["bias_q", "scale", "zero_q"], ["bias_b"]),
        h.make_node("Add", ["bias_a", "bias_b"], ["bias"]),
        h.make_node("Add", ["x", "bias"], ["biased"]),
        h.make_node("Unsqueeze", ["x", "promote_axes"], ["conv_input"]),
        h.make_node("GatherElements", ["conv_input", "element_indices"], ["elements"], axis=3),
        h.make_node("DequantizeLinear", ["kernel_q", "scale", "zero_q"], ["kernel"]),
        h.make_node("Conv", ["conv_input", "kernel"], ["convolved"]),
        h.make_node("Add", ["convolved", "y_fp16"], ["y"]),
        h.make_node("Relu", ["volume"], ["relu"]),
        h.make_node("Split", ["relu"], ["volume_a", "volume_b"], axis=0, num_outputs=2),
        h.make_node(
            "LSTM",
            ["sequence", "w", "r", "", "", "h", "c"],
            ["sequence_y", "h_y", "c_y"],
            direction="bidirectional",
            hidden_size=2,
        ),
        h.make_node("Cast", ["integers"], ["floating"], to=T.FLOAT),
        h.make_node("Concat", ["floating", "floats"], ["joined"], axis=3),
        h.make_node("Squeeze", ["joined", "squeeze_axes"], ["terminal"]),
        h.make_node("Resize", ["conv_input", "", "resize_scales"], ["resized"]),
        h.make_node("Gelu", ["conv_input"], ["gelu"], domain="com.microsoft"),
        h.make_node("ConstantOfShape", ["constant_shape"], ["generated"]),
    ]
    feeds = {
        "x": np.arange(6, dtype=np.float32).reshape(2, 3) / 8 + 0.25,
        "indices": np.array([[1, 0]], np.int64),
        "other_indices": np.array([[0, 1]], np.int64),
        "element_indices": np.array([[[[2, 0, 1], [1, 2, 0]]]], np.int64),
        "condition": np.array([[True, False, True], [False, True, False]]),
        "bytes": np.arange(6, dtype=np.uint8).reshape(2, 3),
        "volume": (np.arange(24, dtype=np.float32).reshape(2, 1, 3, 2, 2) - 12) / 8,
        "sequence": np.array([[[0.25, -0.25]]], np.float32),
        "h": np.full((2, 1, 2), 0.125, np.float32),
        "c": np.full((2, 1, 2), -0.125, np.float32),
        "integers": np.array([[[[0, 1]]]], np.int32),
        "floats": np.array([[[[0.25, 0.5]]]], np.float32),
    }
    arrays = {
        "weight": np.arange(6, dtype=np.float32).reshape(3, 2) / 16,
        "starts": np.array([0], np.int64),
        "ends": np.array([3], np.int64),
        "slice_axes": np.array([-1], np.int64),
        "zero": np.array(0, np.float32),
        "byte_zero": np.array(0, np.uint8),
        "empty_axes": np.array([], np.int64),
        "gain": np.array(0.5, np.float32),
        "bias_q": np.array(2, np.int8),
        "scale": np.array(0.125, np.float32),
        "zero_q": np.array(0, np.int8),
        "promote_axes": np.array([0, 1], np.int64),
        "kernel_q": np.array([1, 2], np.int8).reshape(2, 1, 1, 1),
        "y_fp16": np.array(0.125, np.float32),
        "w": np.arange(32, dtype=np.float32).reshape(2, 8, 2) / 128,
        "r": np.arange(32, dtype=np.float32).reshape(2, 8, 2) / 256,
        "squeeze_axes": np.array([0], np.int64),
        "resize_scales": np.array([1, 1, 2, 2], np.float32),
        "constant_shape": np.array([1, 1, 2, 3], np.int64),
    }
    outputs = [
        ("matrix", T.FLOAT, [2, 2]),
        ("gathered", T.FLOAT, [1, 2, 3]),
        ("masked", T.FLOAT, [2, 3]),
        ("positive", T.BOOL, [2, 3]),
        ("maximum", T.INT64, [2, 1]),
        ("unchanged", T.FLOAT, [2, 3]),
        ("normalization", T.FLOAT, [2, 3]),
        ("biased", T.FLOAT, [2, 3]),
        ("elements", T.FLOAT, [1, 1, 2, 3]),
        ("y", T.FLOAT, [1, 2, 2, 3]),
        ("volume_a", T.FLOAT, [1, 1, 3, 2, 2]),
        ("volume_b", T.FLOAT, [1, 1, 3, 2, 2]),
        ("sequence_y", T.FLOAT, [1, 2, 1, 2]),
        ("h_y", T.FLOAT, [2, 1, 2]),
        ("c_y", T.FLOAT, [2, 1, 2]),
        ("terminal", T.FLOAT, [1, 1, 4]),
        ("resized", T.FLOAT, [1, 1, 4, 6]),
        ("gelu", T.FLOAT, [1, 1, 2, 3]),
        ("generated", T.FLOAT, [1, 1, 2, 3]),
    ]
    original = h.make_model(
        h.make_graph(
            nodes,
            "shared_multi_branch_pipeline",
            [
                h.make_tensor_value_info(name, h.np_dtype_to_tensor_dtype(data.dtype), data.shape)
                for name, data in feeds.items()
            ],
            [h.make_tensor_value_info(*output) for output in outputs],
            [nh.from_array(data, name) for name, data in arrays.items()],
        ),
        opset_imports=[h.make_opsetid("", 21), h.make_opsetid("com.microsoft", 1)],
        ir_version=10,
    )
    saved = original.SerializeToString()
    actual = _transform_make_dla_compatible(
        original,
        force_fp16_activations=fp16,
        qdq_quantized_dtypes=["UINT16", "5", T.INT16],
        verbose=False,
    )
    assert original.SerializeToString() == saved
    assert actual.graph.input == original.graph.input
    assert actual.graph.output == original.graph.output
    assert not {"LSTM", "MatMul", "Where"} & {node.op_type for node in actual.graph.node}
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    expected = ort.InferenceSession(original.SerializeToString(), options).run(None, feeds)
    results = ort.InferenceSession(actual.SerializeToString(), options).run(None, feeds)
    for (name, _, _), result, reference in zip(outputs, results, expected, strict=True):
        np.testing.assert_allclose(
            result, reference, rtol=3e-3 if fp16 else 1e-6, atol=1e-6, err_msg=name
        )


@pytest.mark.parametrize("opset", [17, 18])
@pytest.mark.parametrize("fp16", [False, True])
@pytest.mark.parametrize("remove_qdq", [False, True])
def test_fp16_qdq_schema_migration_keeps_reduce_and_split_valid(opset, fp16, remove_qdq):
    reduce_inputs = ["dq", "axes"] if opset == 18 else ["dq"]
    attributes = {} if opset == 18 else {"axes": [1]}
    nodes = [
        h.make_node("QuantizeLinear", ["x", "scale", "zero"], ["quantized"]),
        h.make_node("DequantizeLinear", ["quantized", "scale", "zero"], ["dq"]),
        h.make_node("ReduceMean", reduce_inputs, ["reduced"], keepdims=1, **attributes),
        h.make_node(
            "Split", ["dq"], ["a", "b"], axis=1, **({"num_outputs": 2} if opset == 18 else {})
        ),
        h.make_node("Split", ["dq", "lengths"], ["c", "d"], axis=1),
    ]
    initializers = [
        nh.from_array(np.asarray(0.5, np.float32), "scale"),
        nh.from_array(np.asarray(0, np.uint8), "zero"),
        nh.from_array(np.asarray([1, 1], np.int64), "lengths"),
    ]
    if opset == 18:
        initializers.append(nh.from_array(np.asarray([1], np.int64), "axes"))
    model = h.make_model(
        h.make_graph(
            nodes,
            "schema_migration",
            [h.make_tensor_value_info("x", T.FLOAT, [1, 2])],
            [
                h.make_tensor_value_info("reduced", T.FLOAT, [1, 1]),
                h.make_tensor_value_info("a", T.FLOAT, [1, 1]),
                h.make_tensor_value_info("b", T.FLOAT, [1, 1]),
                h.make_tensor_value_info("c", T.FLOAT, [1, 1]),
                h.make_tensor_value_info("d", T.FLOAT, [1, 1]),
            ],
            initializers,
        ),
        opset_imports=[h.make_opsetid("", opset)],
        ir_version=10,
    )
    model.opset_import.append(h.make_opsetid("com.microsoft", 1))
    saved = model.SerializeToString()
    result = _transform_make_dla_compatible(
        model,
        remove_all_qdq=remove_qdq,
        force_fp16_activations=fp16,
        verbose=False,
    )
    assert model.SerializeToString() == saved
    assert {item.domain: item.version for item in result.opset_import} == {
        "": 19 if fp16 else opset,
        "com.microsoft": 1,
    }
    reduction = next(node for node in result.graph.node if node.op_type == "ReduceMean")
    assert len(reduction.input) == (2 if fp16 or opset >= 18 else 1)
    assert any(attribute.name == "axes" for attribute in reduction.attribute) == (
        not fp16 and opset == 17
    )
    explicit = next(
        node for node in result.graph.node if node.op_type == "Split" and len(node.input) == 2
    )
    assert all(attribute.name != "num_outputs" for attribute in explicit.attribute)
    feeds = {"x": np.asarray([[1, 2]], np.float32)}
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    expected = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    ).run(None, feeds)
    actual = ort.InferenceSession(
        result.SerializeToString(), options, providers=["CPUExecutionProvider"]
    ).run(None, feeds)
    for original, transformed in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(original, transformed)
