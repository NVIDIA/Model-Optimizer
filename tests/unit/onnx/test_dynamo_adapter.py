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

"""Behavior-level tests for the private Dynamo ONNX graph adapter."""

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from modelopt.onnx.export._dynamo_adapter import normalize_dynamo_weight_paths


def _tensor(name, values, dtype=np.float32):
    return numpy_helper.from_array(np.asarray(values, dtype=dtype), name)


def _model(
    nodes,
    *,
    initializers=(),
    inputs=(),
    outputs=(),
    value_info=(),
    opset=23,
    trt_opset=None,
):
    graph = helper.make_graph(
        nodes,
        "dynamo_adapter",
        list(inputs),
        list(outputs),
        list(initializers),
        value_info=list(value_info),
    )
    imports = [helper.make_opsetid("", opset)]
    if trt_opset is not None:
        imports.append(helper.make_opsetid("trt", trt_opset))
    return helper.make_model(graph, opset_imports=imports)


def _value(name, dtype=TensorProto.FLOAT, shape=(2, 4)):
    return helper.make_tensor_value_info(name, dtype, list(shape))


def _nvfp4_marker(weight="weight", output="weight_dq"):
    return helper.make_node("TRT_FP4QDQ", [weight], [output], domain="trt", block_size=16)


@pytest.mark.parametrize(
    ("quantize_op", "dequantize_op", "domain"),
    [
        ("TRT_FP8QuantizeLinear", "TRT_FP8DequantizeLinear", "trt"),
        ("QuantizeLinear", "DequantizeLinear", ""),
    ],
)
def test_normalize_accepts_canonical_qdq_weight_path(quantize_op, dequantize_op, domain):
    weight = _tensor("weight", np.ones((4, 4)))
    blocked_shape = _tensor("blocked_shape", [2, 2, 4], np.int64)
    restored_shape = _tensor("restored_shape", [4, 4], np.int64)
    scale = _tensor("scale", 0.25)
    zero = _tensor("zero", 0, np.uint8)
    model = _model(
        [
            helper.make_node("Reshape", ["weight", "blocked_shape"], ["blocked"]),
            helper.make_node(
                quantize_op, ["blocked", "scale", "zero"], ["weight_q"], domain=domain
            ),
            helper.make_node(
                dequantize_op, ["weight_q", "scale", "zero"], ["weight_dq"], domain=domain
            ),
            helper.make_node("Reshape", ["weight_dq", "restored_shape"], ["restored"]),
            helper.make_node("MatMul", ["input", "restored"], ["output"]),
        ],
        initializers=[weight, blocked_shape, restored_shape, scale, zero],
        inputs=[_value("input", shape=(2, 4))],
        outputs=[_value("output", shape=(2, 4))],
        trt_opset=1 if domain else None,
    )

    assert normalize_dynamo_weight_paths(model) is model
    assert [node.op_type for node in model.graph.node] == [
        quantize_op,
        dequantize_op,
        "Reshape",
        "MatMul",
    ]
    quantizer = model.graph.node[0]
    assert quantizer.input[0] == "weight"
    assert list(next(item for item in model.graph.initializer if item.name == "weight").dims) == [
        2,
        2,
        4,
    ]


def test_normalize_rejects_mismatched_qdq_parameters():
    model = _model(
        [
            helper.make_node("QuantizeLinear", ["weight", "scale", "zero"], ["weight_q"]),
            helper.make_node(
                "DequantizeLinear", ["weight_q", "other_scale", "zero"], ["weight_dq"]
            ),
            helper.make_node("MatMul", ["input", "weight_dq"], ["output"]),
        ],
        initializers=[
            _tensor("weight", np.ones((4, 4))),
            _tensor("scale", 0.25),
            _tensor("other_scale", 0.5),
            _tensor("zero", 0, np.uint8),
        ],
        inputs=[_value("input")],
        outputs=[_value("output")],
    )

    with pytest.raises(NotImplementedError, match="scale and zero-point inputs must match"):
        normalize_dynamo_weight_paths(model)


@pytest.mark.parametrize("view_order", ["reshape_cast", "cast_reshape"])
def test_normalize_canonical_nvfp4_path_removes_export_only_views(view_order):
    weight = _tensor("weight", np.ones((2, 4)))
    blocked_shape = _tensor("blocked_shape", [1, 2, 4], np.int64)
    restored_shape = _tensor("restored_shape", [2, 4], np.int64)
    views = {
        "reshape_cast": [
            helper.make_node("Reshape", ["identity", "restored_shape"], ["restored"]),
            helper.make_node("Cast", ["restored"], ["cast"], to=TensorProto.FLOAT16),
        ],
        "cast_reshape": [
            helper.make_node("Cast", ["identity"], ["cast"], to=TensorProto.FLOAT16),
            helper.make_node("Reshape", ["cast", "restored_shape"], ["restored"]),
        ],
    }[view_order]
    view_output = views[-1].output[0]
    model = _model(
        [
            helper.make_node("Reshape", ["weight", "blocked_shape"], ["blocked"]),
            _nvfp4_marker("blocked", "marked"),
            helper.make_node("Identity", ["marked"], ["identity"]),
            *views,
            helper.make_node("Transpose", [view_output], ["transposed"], perm=[1, 0]),
            helper.make_node("Gemm", ["input", "transposed"], ["output"]),
        ],
        initializers=[weight, blocked_shape, restored_shape],
        inputs=[_value("input", TensorProto.FLOAT16, (1, 4))],
        outputs=[_value("output", TensorProto.FLOAT16, (1, 2))],
        trt_opset=1,
    )

    normalize_dynamo_weight_paths(model)

    assert [node.op_type for node in model.graph.node] == ["TRT_FP4QDQ", "Gemm"]
    marker, gemm = model.graph.node
    assert marker.input[0] == "weight"
    assert gemm.input[1] == marker.output[0]
    assert (
        helper.get_attribute_value(next(attr for attr in gemm.attribute if attr.name == "transB"))
        == 1
    )


@pytest.mark.parametrize("view_op", ["Slice", "Transpose", "Expand", "Gather"])
def test_normalize_rejects_static_view_before_quantizer(view_op):
    if view_op == "Slice":
        view = helper.make_node("Slice", ["weight"], ["view"])
    elif view_op == "Transpose":
        view = helper.make_node("Transpose", ["weight"], ["view"], perm=[1, 0])
    elif view_op == "Expand":
        view = helper.make_node("Expand", ["weight", "expanded_shape"], ["view"])
    else:
        view = helper.make_node("Gather", ["weight", "runtime_indices"], ["view"])
    inputs = [_value("input")]
    if view_op == "Gather":
        inputs.append(_value("runtime_indices", TensorProto.INT64, (4,)))
    model = _model(
        [
            view,
            helper.make_node("QuantizeLinear", ["view", "scale", "zero"], ["weight_q"]),
            helper.make_node("DequantizeLinear", ["weight_q", "scale", "zero"], ["weight_dq"]),
            helper.make_node("MatMul", ["input", "weight_dq"], ["output"]),
        ],
        initializers=[
            _tensor("weight", np.ones((4, 4))),
            _tensor("scale", 0.25),
            _tensor("zero", 0, np.uint8),
            _tensor("expanded_shape", [4, 4], np.int64),
        ],
        inputs=inputs,
        outputs=[_value("output")],
    )

    with pytest.raises(NotImplementedError, match="before the quantization marker"):
        normalize_dynamo_weight_paths(model)


def test_normalize_rejects_static_quantizer_without_matching_dequantizer():
    model = _model(
        [
            helper.make_node("QuantizeLinear", ["weight", "scale", "zero"], ["weight_q"]),
            helper.make_node("Identity", ["weight_q"], ["weight_dq"]),
            helper.make_node("MatMul", ["input", "weight_dq"], ["output"]),
        ],
        initializers=[
            _tensor("weight", np.ones((4, 4))),
            _tensor("scale", 0.25),
            _tensor("zero", 0, np.uint8),
        ],
        inputs=[_value("input")],
        outputs=[_value("output")],
    )

    with pytest.raises(NotImplementedError, match="matching DequantizeLinear"):
        normalize_dynamo_weight_paths(model)


def test_normalize_rejects_wrong_marker_domain():
    model = _model(
        [
            helper.make_node(
                "QuantizeLinear", ["weight", "scale", "zero"], ["weight_q"], domain="custom"
            ),
            helper.make_node(
                "DequantizeLinear",
                ["weight_q", "scale", "zero"],
                ["weight_dq"],
                domain="custom",
            ),
            helper.make_node("MatMul", ["input", "weight_dq"], ["output"]),
        ],
        initializers=[
            _tensor("weight", np.ones((4, 4))),
            _tensor("scale", 0.25),
            _tensor("zero", 0, np.uint8),
        ],
        inputs=[_value("input")],
        outputs=[_value("output")],
    )

    with pytest.raises(NotImplementedError, match="unexpected domain"):
        normalize_dynamo_weight_paths(model)


def test_normalize_rejects_mismatched_post_marker_reshape():
    model = _model(
        [
            helper.make_node("QuantizeLinear", ["weight", "scale", "zero"], ["weight_q"]),
            helper.make_node("DequantizeLinear", ["weight_q", "scale", "zero"], ["weight_dq"]),
            helper.make_node("Reshape", ["weight_dq", "wrong_shape"], ["restored"]),
            helper.make_node("MatMul", ["input", "restored"], ["output"]),
        ],
        initializers=[
            _tensor("weight", np.ones((4, 4))),
            _tensor("scale", 0.25),
            _tensor("zero", 0, np.uint8),
            _tensor("wrong_shape", [2, 8], np.int64),
        ],
        inputs=[_value("input")],
        outputs=[_value("output")],
    )

    with pytest.raises(NotImplementedError, match="restoring Reshape does not match"):
        normalize_dynamo_weight_paths(model)


def test_normalize_materializes_mxfp8_prefix_reshape():
    model = _model(
        [
            helper.make_node("Reshape", ["linear.weight", "blocked_shape"], ["blocked"]),
            helper.make_node(
                "TRT_MXFP8DequantizeLinear",
                ["blocked", "scale"],
                ["weight_dq"],
                domain="trt",
                axis=-1,
                block_size=32,
                output_dtype=TensorProto.FLOAT,
            ),
            helper.make_node("Reshape", ["weight_dq", "restored_shape"], ["restored"]),
            helper.make_node("MatMul", ["input", "restored"], ["output"]),
        ],
        initializers=[
            _tensor("linear.weight", np.ones((2, 64))),
            _tensor("blocked_shape", [4, 32], np.int64),
            _tensor("restored_shape", [2, 64], np.int64),
            _tensor("scale", 1.0),
        ],
        inputs=[_value("input", shape=(1, 2))],
        outputs=[_value("output", shape=(1, 64))],
        trt_opset=1,
    )

    normalize_dynamo_weight_paths(model)

    marker = next(node for node in model.graph.node if node.op_type.startswith("TRT_MXFP8"))
    assert marker.input[0] == "linear.weight"
    weight = next(item for item in model.graph.initializer if item.name == "linear.weight")
    assert list(weight.dims) == [4, 32]


def _unsupported_nvfp4_model(kind):
    weight = _tensor("weight", np.ones((2, 4)))
    marker = _nvfp4_marker()
    common = {
        "initializers": [weight],
        "inputs": [_value("input")],
        "outputs": [_value("output")],
        "trt_opset": 1,
    }
    if kind == "shared weight":
        nodes = [
            marker,
            helper.make_node("Identity", ["weight"], ["other_weight_use"]),
            helper.make_node("MatMul", ["input", "weight_dq"], ["output"]),
        ]
    elif kind == "fanout":
        nodes = [
            marker,
            helper.make_node("MatMul", ["input", "weight_dq"], ["output"]),
            helper.make_node("Identity", ["weight_dq"], ["other_output"]),
        ]
    elif kind == "Slice":
        nodes = [
            marker,
            helper.make_node("Slice", ["weight_dq"], ["view"]),
            helper.make_node("MatMul", ["input", "view"], ["output"]),
        ]
    else:
        nodes = [marker, helper.make_node("Relu", ["weight_dq"], ["output"])]
    return _model(nodes, **common)


@pytest.mark.parametrize("kind", ["shared weight", "fanout", "Slice", "non-compute"])
def test_normalize_rejects_unsupported_weight_topologies(kind):
    with pytest.raises(NotImplementedError, match="Unsupported Dynamo quantized weight topology"):
        normalize_dynamo_weight_paths(_unsupported_nvfp4_model(kind))


def test_normalize_rejects_unmatched_int4_restoring_reshape():
    weight = _tensor("weight", np.ones((2, 4)))
    scale = _tensor("scale", np.ones((1, 2)))
    blocked_shape = _tensor("blocked_shape", [2, 2, 2], np.int64)
    wrong_shape = _tensor("wrong_shape", [4, 2], np.int64)
    model = _model(
        [
            helper.make_node("Reshape", ["weight", "blocked_shape"], ["blocked"]),
            helper.make_node(
                "DequantizeLinear",
                ["blocked", "scale"],
                ["weight_dq"],
                axis=-1,
                block_size=2,
            ),
            helper.make_node("Reshape", ["weight_dq", "wrong_shape"], ["restored"]),
            helper.make_node("MatMul", ["input", "restored"], ["output"]),
        ],
        initializers=[weight, scale, blocked_shape, wrong_shape],
        inputs=[_value("input")],
        outputs=[_value("output")],
    )

    with pytest.raises(NotImplementedError, match="restoring Reshape does not match"):
        normalize_dynamo_weight_paths(model)


def test_normalize_rejects_unmatched_nvfp4_restoring_reshape():
    weight = _tensor("weight", np.ones((2, 4)))
    blocked_shape = _tensor("blocked_shape", [1, 2, 4], np.int64)
    wrong_shape = _tensor("wrong_shape", [4, 2], np.int64)
    model = _model(
        [
            helper.make_node("Reshape", ["weight", "blocked_shape"], ["blocked"]),
            _nvfp4_marker("blocked", "marked"),
            helper.make_node("Reshape", ["marked", "wrong_shape"], ["restored"]),
            helper.make_node("MatMul", ["input", "restored"], ["output"]),
        ],
        initializers=[weight, blocked_shape, wrong_shape],
        inputs=[_value("input")],
        outputs=[_value("output")],
        trt_opset=1,
    )

    with pytest.raises(NotImplementedError, match="restoring Reshape does not match"):
        normalize_dynamo_weight_paths(model)


def test_normalize_rejects_unpaired_nvfp4_restoring_reshape():
    model = _model(
        [
            _nvfp4_marker(),
            helper.make_node("Reshape", ["weight_dq", "view_shape"], ["view"]),
            helper.make_node("MatMul", ["input", "view"], ["output"]),
        ],
        initializers=[
            _tensor("weight", np.ones((2, 4))),
            _tensor("view_shape", [4, 2], np.int64),
        ],
        inputs=[_value("input")],
        outputs=[_value("output")],
        trt_opset=1,
    )

    with pytest.raises(NotImplementedError, match="restoring Reshape requires a pre-Reshape"):
        normalize_dynamo_weight_paths(model)


def test_normalize_splits_shared_int4_scale_without_mutating_source():
    source_scale = np.asarray([[0.25, 0.5], [0.75, 1.0]], dtype=np.float32)
    initializers = [
        _tensor("weight_a", np.ones((2, 4))),
        _tensor("weight_b", np.ones((2, 4))),
        _tensor("shared_scale", source_scale),
    ]
    nodes = []
    for suffix in ("a", "b"):
        nodes.extend(
            [
                helper.make_node(
                    "DequantizeLinear",
                    [f"weight_{suffix}", "shared_scale"],
                    [f"weight_{suffix}_dq"],
                    axis=-1,
                    block_size=2,
                ),
                helper.make_node(
                    "MatMul", [f"input_{suffix}", f"weight_{suffix}_dq"], [f"output_{suffix}"]
                ),
            ]
        )
    model = _model(
        nodes,
        initializers=initializers,
        inputs=[_value("input_a"), _value("input_b")],
        outputs=[_value("output_a"), _value("output_b")],
    )

    normalize_dynamo_weight_paths(model)

    dequantizers = [node for node in model.graph.node if node.op_type == "DequantizeLinear"]
    split_names = [node.input[1] for node in dequantizers]
    assert len(set(split_names)) == 2
    assert "shared_scale" not in split_names
    tensors = {tensor.name: tensor for tensor in model.graph.initializer}
    np.testing.assert_array_equal(numpy_helper.to_array(tensors["shared_scale"]), source_scale)
    for name in split_names:
        np.testing.assert_array_equal(
            numpy_helper.to_array(tensors[name]), source_scale.reshape(-1, 1)
        )


def test_normalize_leaves_standard_int8_conv_weight_qdq_intact():
    model = _model(
        [
            helper.make_node("QuantizeLinear", ["weight", "scale", "zero"], ["weight_q"]),
            helper.make_node("DequantizeLinear", ["weight_q", "scale", "zero"], ["weight_dq"]),
            helper.make_node("Conv", ["input", "weight_dq"], ["output"]),
        ],
        initializers=[
            _tensor("weight", np.ones((4, 3, 1, 1))),
            _tensor("scale", np.ones(4)),
            _tensor("zero", np.zeros(4), np.int8),
        ],
        inputs=[_value("input", shape=(1, 3, 2, 2))],
        outputs=[_value("output", shape=(1, 4, 2, 2))],
    )

    normalize_dynamo_weight_paths(model)

    assert [node.op_type for node in model.graph.node] == [
        "QuantizeLinear",
        "DequantizeLinear",
        "Conv",
    ]
