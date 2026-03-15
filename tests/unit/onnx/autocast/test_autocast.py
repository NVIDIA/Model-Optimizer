# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
from pathlib import Path

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import pytest
from _test_utils.onnx.lib_test_models import build_conv_isinf_model, build_conv_resize_model

import modelopt.onnx.autocast.utils as utils
import modelopt.onnx.utils as onnx_utils
from modelopt.onnx.autocast import convert_to_mixed_precision
from modelopt.onnx.autocast.__main__ import get_parser, main
from modelopt.onnx.autocast.logging_config import configure_logging

configure_logging("DEBUG")


@pytest.fixture
def simple_model():
    # Create a model with multiple nodes and initializers
    input_shape = [1, 5]
    x = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, input_shape)
    y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 3])

    # Create initializers
    gemm_weight = np.random.randn(5, 3).astype(np.float32)
    gemm_bias = np.random.randn(3).astype(np.float32)
    add_const = np.random.randn(1, 3).astype(np.float32)

    gemm_w_init = onnx.numpy_helper.from_array(gemm_weight, name="gemm_weight")
    gemm_b_init = onnx.numpy_helper.from_array(gemm_bias, name="gemm_bias")
    add_init = onnx.numpy_helper.from_array(add_const, name="add_const")

    # Create nodes
    gemm_node = onnx.helper.make_node(
        "Gemm", inputs=["X", "gemm_weight", "gemm_bias"], outputs=["gemm_out"], name="gemm_node"
    )

    add_node = onnx.helper.make_node(
        "Add", inputs=["gemm_out", "add_const"], outputs=["add_out"], name="add_node"
    )

    relu_node = onnx.helper.make_node("Relu", inputs=["add_out"], outputs=["Y"], name="relu_node")

    graph = onnx.helper.make_graph(
        [gemm_node, add_node, relu_node],
        "test_complex_graph",
        [x],
        [y],
        [gemm_w_init, gemm_b_init, add_init],
    )

    model = onnx.helper.make_model(graph)
    model.opset_import[0].version = 20
    model.ir_version = 10
    return model


@pytest.fixture
def temp_model_path(tmp_path, simple_model):
    model_path = tmp_path / "model.onnx"
    onnx.save(simple_model, str(model_path))
    return str(model_path)


@pytest.fixture
def temp_output_path(tmp_path):
    return str(tmp_path / "output_model.onnx")


def test_invalid_model_path():
    with pytest.raises(FileNotFoundError):
        convert_to_mixed_precision(onnx_path="nonexistent.onnx")


def test_setup_mappings(simple_model):
    simple_model = onnx_utils.infer_shapes(simple_model)
    value_info_map, initializer_map, node_to_init_map = utils.setup_mappings(simple_model)

    # Test value_info_map
    assert len(value_info_map) == 4  # X, Y, gemm_out, add_out
    assert all(name in value_info_map for name in ["X", "Y", "gemm_out", "add_out"])
    assert value_info_map["X"].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    assert value_info_map["Y"].type.tensor_type.elem_type == onnx.TensorProto.FLOAT

    # Test initializer_map
    assert len(initializer_map) == 3  # gemm_weight, gemm_bias, add_const
    assert all(name in initializer_map for name in ["gemm_weight", "gemm_bias", "add_const"])
    assert initializer_map["gemm_weight"].data_type == onnx.TensorProto.FLOAT
    assert initializer_map["gemm_bias"].data_type == onnx.TensorProto.FLOAT
    assert initializer_map["add_const"].data_type == onnx.TensorProto.FLOAT

    # Test node_to_init_map
    assert len(node_to_init_map) == 3  # gemm_node, add_node, relu_node
    assert "gemm_node" in node_to_init_map
    assert "add_node" in node_to_init_map
    assert "relu_node" in node_to_init_map
    assert len(node_to_init_map["gemm_node"]) == 2  # gemm_weight, gemm_bias
    assert len(node_to_init_map["add_node"]) == 1  # add_const
    assert len(node_to_init_map["relu_node"]) == 0  # no initializers


@pytest.mark.parametrize("keep_io_types", [True, False])
def test_convert_simple_model(temp_model_path, temp_output_path, keep_io_types):
    # Convert the model
    converted_model = convert_to_mixed_precision(
        onnx_path=temp_model_path, keep_io_types=keep_io_types
    )

    # Save the converted model
    onnx.save(converted_model, temp_output_path)

    assert isinstance(converted_model, onnx.ModelProto)
    assert Path(temp_output_path).exists()

    # Load and verify the saved model
    loaded_model = onnx.load(temp_output_path)
    if keep_io_types:
        assert len(loaded_model.graph.node) == 5
        assert loaded_model.graph.node[0].op_type == "Cast"
        assert loaded_model.graph.node[1].op_type == "Gemm"
        assert loaded_model.graph.node[2].op_type == "Add"
        assert loaded_model.graph.node[3].op_type == "Relu"
        assert loaded_model.graph.node[4].op_type == "Cast"
    else:
        assert len(loaded_model.graph.node) == 3
        assert loaded_model.graph.node[0].op_type == "Gemm"
        assert loaded_model.graph.node[1].op_type == "Add"
        assert loaded_model.graph.node[2].op_type == "Relu"

    # Verify input/output types
    expected_io_type = onnx.TensorProto.FLOAT if keep_io_types else onnx.TensorProto.FLOAT16
    assert loaded_model.graph.input[0].type.tensor_type.elem_type == expected_io_type
    assert loaded_model.graph.output[0].type.tensor_type.elem_type == expected_io_type

    onnx.checker.check_model(loaded_model)


def assert_input_precision(nodes, dtype="float16"):
    for node in nodes:
        for inp in node.inputs:
            assert inp.dtype == dtype, (
                f"Node of type {node.op} has type {inp.dtype} but should have type {dtype}"
            )
    return True


@pytest.mark.parametrize("opset_version", [13, 21])
def test_conv_isinf_conversion(tmp_path, opset_version):
    onnx_model = build_conv_isinf_model(opset_version)
    onnx_path = os.path.join(tmp_path, f"conv_isinf_model_opset{opset_version}.onnx")
    onnx.save(onnx_model, onnx_path)

    # Convert the model
    converted_model = convert_to_mixed_precision(onnx_path=onnx_path, keep_io_types=True)

    # Output model should be produced in the same tmp_path
    output_onnx_path = onnx_path.replace(".onnx", ".fp16.onnx")
    onnx.save(converted_model, output_onnx_path)

    # Load the output model
    graph = gs.import_onnx(converted_model)

    # Check that Conv is converted
    conv_nodes = [n for n in graph.nodes if "Conv" in n.op]
    assert assert_input_precision(conv_nodes)

    # Check that IsInf is running in the lowest supported precision:
    # - FP32 if opset < 20, or
    # - FP16 if opset >= 20
    isinf_nodes = [n for n in graph.nodes if n.op == "IsInf"]
    opset_version = onnx_utils.get_opset_version(converted_model)
    supported_dtype = "float32" if opset_version < 20 else "float16"
    assert assert_input_precision(isinf_nodes, dtype=supported_dtype)


def test_conv_resize_conversion(tmp_path):
    onnx_model = build_conv_resize_model()
    onnx_path = os.path.join(tmp_path, "conv_resize_model.onnx")
    onnx.save(onnx_model, onnx_path)

    # Convert the model
    converted_model = convert_to_mixed_precision(onnx_path=onnx_path)

    # Output model should be produced in the same tmp_path
    output_onnx_path = onnx_path.replace(".onnx", ".fp16.onnx")
    onnx.save(converted_model, output_onnx_path)

    # Load the output model
    graph = gs.import_onnx(converted_model)

    # Check that Resize is correctly converted:
    # - Data and ROI inputs (indices 0 and 1) should be FP16
    # - The remaining inputs (scales/sizes) should be kept in their original precisions
    resize_node = next(n for n in graph.nodes if n.op == "Resize")
    assert all(inp.dtype == np.float16 for inp in resize_node.inputs[0:2]), (
        "Resize data and ROI inputs should be FP16"
    )


@pytest.mark.parametrize("target_opset", [13, 17, 19, 21])
def test_opset_parameter(temp_model_path, target_opset):
    """Test that the opset parameter correctly sets the output model's opset version."""
    # Convert with specific opset
    converted_model = convert_to_mixed_precision(
        onnx_path=temp_model_path, low_precision_type="fp16", opset=target_opset
    )

    # Verify the output model has the correct opset
    output_opset = onnx_utils.get_opset_version(converted_model)
    assert output_opset >= target_opset, f"Expected opset >= {target_opset}, got {output_opset}"

    # Validate the model
    onnx.checker.check_model(converted_model)


def test_opset_fp16_warning(temp_model_path, caplog):
    """Test that a warning is issued when using fp16 with opset < 13."""
    # Convert with fp16 and very low opset
    converted_model = convert_to_mixed_precision(
        onnx_path=temp_model_path, low_precision_type="fp16", opset=11
    )

    # Check that a warning was logged
    assert "limited FP16 support" in caplog.text, (
        "Expected warning about FP16 support with low opset"
    )
    assert "Recommended minimum opset is 13" in caplog.text

    # Model should still be created
    assert isinstance(converted_model, onnx.ModelProto)


def test_opset_bf16_warning(temp_model_path, caplog):
    """Test that a warning is issued when using bf16 with opset < 22."""
    # Convert with bf16 and low opset
    converted_model = convert_to_mixed_precision(
        onnx_path=temp_model_path, low_precision_type="bf16", opset=13
    )

    # Check that a warning was logged
    assert "limited BF16 support" in caplog.text, (
        "Expected warning about BF16 support with low opset"
    )
    assert "Recommended minimum opset is 22" in caplog.text

    # Model should still be created
    assert isinstance(converted_model, onnx.ModelProto)


def test_opset_downgrade_warning(temp_model_path, caplog):
    """Test that a warning is issued when specified opset is lower than original model's opset."""
    # temp_model_path fixture creates a model with opset 20
    # Convert with lower opset
    converted_model = convert_to_mixed_precision(
        onnx_path=temp_model_path, low_precision_type="fp16", opset=13
    )

    # Check that a warning was logged about downgrading
    assert "lower than the original model's opset" in caplog.text, (
        "Expected warning about downgrading opset"
    )

    # Model should still be created
    assert isinstance(converted_model, onnx.ModelProto)


def test_opset_cli_argument(temp_model_path, tmp_path):
    """Test that the --opset CLI argument is properly parsed and used."""
    # Test the CLI with opset argument
    output_path = tmp_path / "test_output.onnx"
    args = [
        "--onnx_path",
        temp_model_path,
        "--output_path",
        str(output_path),
        "--opset",
        "21",
        "--low_precision_type",
        "fp16",
    ]

    result_model = main(args)

    # Verify the output model has the correct opset
    output_opset = onnx_utils.get_opset_version(result_model)
    assert output_opset >= 21, f"Expected opset >= 21, got {output_opset}"

    # Verify the file was created
    assert output_path.exists()

    # Load and validate the saved model
    saved_model = onnx.load(str(output_path))
    onnx.checker.check_model(saved_model)


def test_opset_parser_argument():
    """Test that the parser correctly accepts the --opset argument."""
    parser = get_parser()

    # Test parsing with opset
    args = parser.parse_args(["--onnx_path", "test.onnx", "--opset", "19"])
    assert args.opset == 19

    # Test parsing without opset (should be None)
    args = parser.parse_args(["--onnx_path", "test.onnx"])
    assert args.opset is None
