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

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

import modelopt.onnx.autocast.utils as utils
import modelopt.onnx.utils as onnx_utils
from modelopt.onnx.autocast.logging_config import configure_logging
from modelopt.onnx.autocast.precisionconverter import PrecisionConverter

configure_logging("DEBUG")

LATEST_IR_VERSION_SUPPORTED_BY_ORT = 10


def low_precision_onnx_type(low_precision_type_str):
    return TensorProto.FLOAT16 if low_precision_type_str == "fp16" else TensorProto.BFLOAT16


####################################################################################################
# Testing with a basic GEMM->Add->Relu graph
####################################################################################################
@pytest.fixture
def simple_model():
    # Create a simple model with a GEMM->Add->Relu chain
    input_shape = [1, 5]
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape)
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])
    gemm_init_numpy = np.random.randn(5, 3).astype(np.float32)
    add_init_numpy = np.random.randn(1, 3).astype(np.float32)
    gemm_init = numpy_helper.from_array(gemm_init_numpy, name="gemm_init")
    add_init = numpy_helper.from_array(add_init_numpy, name="add_init")
    gemm_node = helper.make_node("MatMul", ["X", "gemm_init"], ["gemm_output"], name="gemm")
    add_node = helper.make_node("Add", ["gemm_output", "add_init"], ["add_output"], name="add")
    relu_node = helper.make_node("Relu", ["add_output"], ["Y"], name="relu")
    graph = helper.make_graph(
        [gemm_node, add_node, relu_node], "model_base", [x], [y], [gemm_init, add_init]
    )
    model = helper.make_model(graph, producer_name="model_base")
    model.opset_import[0].version = 20
    model.ir_version = 10
    onnx.checker.check_model(model)

    model = onnx_utils.infer_shapes(model)
    value_info_map, initializer_map, node_to_init_map = utils.setup_mappings(model)

    return model, value_info_map, initializer_map, node_to_init_map


def test_graph_converter_init(simple_model):
    model, value_info_map, initializer_map, node_to_init_map = simple_model
    converter = PrecisionConverter(
        model, value_info_map, initializer_map, node_to_init_map, keep_io_types=True
    )
    assert converter.model == model
    assert converter.value_info_map == value_info_map
    assert converter.initializer_map == initializer_map
    assert converter.keep_io_types


@pytest.mark.parametrize("keep_io_types", [True, False])
@pytest.mark.parametrize("low_precision_type", ["fp16", "bf16"])
def test_simple_convert(simple_model, keep_io_types, low_precision_type):
    model, value_info_map, initializer_map, node_to_init_map = simple_model
    converter = PrecisionConverter(
        model,
        value_info_map,
        initializer_map,
        node_to_init_map,
        keep_io_types=keep_io_types,
        low_precision_type=low_precision_type,
    )

    # Convert add node to fp16, keep mul in fp32
    converted_model = converter.convert(
        high_precision_nodes=["gemm", "add"], low_precision_nodes=["relu"]
    )

    # Verify input type changed to fp16
    expected_io_type = (
        TensorProto.FLOAT if keep_io_types else low_precision_onnx_type(low_precision_type)
    )
    assert converted_model.graph.input[0].type.tensor_type.elem_type == expected_io_type
    assert converted_model.graph.output[0].type.tensor_type.elem_type == expected_io_type

    # Verify cast nodes were added in the expected locations
    assert len(converted_model.graph.node) == 5
    if keep_io_types:
        assert converted_model.graph.node[0].op_type == "MatMul"
        assert converted_model.graph.node[1].op_type == "Add"
        assert converted_model.graph.node[2].op_type == "Cast"
        assert converted_model.graph.node[3].op_type == "Relu"
        assert converted_model.graph.node[4].op_type == "Cast"
    else:
        assert converted_model.graph.node[0].op_type == "Cast"
        assert converted_model.graph.node[1].op_type == "MatMul"
        assert converted_model.graph.node[2].op_type == "Add"
        assert converted_model.graph.node[3].op_type == "Cast"
        assert converted_model.graph.node[4].op_type == "Relu"

    # Verify that the model is valid
    onnx.checker.check_model(converted_model)


@pytest.mark.parametrize("low_precision_type", ["int8", "fp8", "fp4", "fp32", "typo"])
def test_unsupported_precision_type(simple_model, low_precision_type):
    model, value_info_map, initializer_map, node_to_init_map = simple_model
    with pytest.raises(ValueError, match=f"Unsupported precision type: {low_precision_type}"):
        PrecisionConverter(
            model,
            value_info_map,
            initializer_map,
            node_to_init_map,
            keep_io_types=False,
            low_precision_type=low_precision_type,
        )


@pytest.mark.parametrize("keep_io_types", [True, False])
@pytest.mark.parametrize("low_precision_type", ["fp16", "bf16"])
def test_convert_no_disabled_nodes(simple_model, keep_io_types, low_precision_type):
    model, value_info_map, initializer_map, node_to_init_map = simple_model
    converter = PrecisionConverter(
        model,
        value_info_map,
        initializer_map,
        node_to_init_map,
        keep_io_types=keep_io_types,
        low_precision_type=low_precision_type,
    )

    # Convert all nodes to fp16
    converted_model = converter.convert(
        high_precision_nodes=[], low_precision_nodes=["gemm", "add", "relu"]
    )

    # Verify input type changed to fp16
    expected_io_type = (
        TensorProto.FLOAT if keep_io_types else low_precision_onnx_type(low_precision_type)
    )
    assert converted_model.graph.input[0].type.tensor_type.elem_type == expected_io_type
    assert converted_model.graph.output[0].type.tensor_type.elem_type == expected_io_type

    # Verify cast nodes were added only for keep_io_types=True
    cast_nodes = [n for n in converted_model.graph.node if n.op_type == "Cast"]
    expected_cast_nodes = 2 if keep_io_types else 0
    assert len(cast_nodes) == expected_cast_nodes

    # Verify that the model is valid
    onnx.checker.check_model(converted_model)


@pytest.mark.parametrize("keep_io_types", [True, False])
@pytest.mark.parametrize("low_precision_type", ["fp16", "bf16"])
def test_get_tensors_to_cast(simple_model, keep_io_types, low_precision_type):
    model, value_info_map, initializer_map, node_to_init_map = simple_model
    converter = PrecisionConverter(
        model,
        value_info_map,
        initializer_map,
        node_to_init_map,
        keep_io_types=keep_io_types,
        low_precision_type=low_precision_type,
    )

    # Test when relu node is in low precision
    cast_down, cast_up, _ = converter._get_tensors_to_cast(["relu"])
    assert "add_output" in cast_down  # Input to relu should be cast down
    assert "Y" in cast_up  # Output of relu should be cast up
    if not keep_io_types:
        assert (
            "X" in cast_up
        )  # Input to gemm should be cast up, because network input are converted to FP16

    # Test when add node is in low precision
    cast_down, cast_up, _ = converter._get_tensors_to_cast(["add"])
    assert "gemm_output" in cast_down  # Input to add should be cast down
    assert "add_init" not in cast_down  # Initializer should not be in cast list
    assert "add_output" in cast_up  # Output of add should be cast up


@pytest.mark.parametrize("keep_io_types", [True, False])
@pytest.mark.parametrize("low_precision_type", ["fp16", "bf16"])
def test_keep_io_names(simple_model, keep_io_types, low_precision_type):
    model, value_info_map, initializer_map, node_to_init_map = simple_model
    converter = PrecisionConverter(
        model,
        value_info_map,
        initializer_map,
        node_to_init_map,
        keep_io_types=keep_io_types,
        low_precision_type=low_precision_type,
    )

    # Convert all nodes to low precision
    converted_model = converter.convert(
        high_precision_nodes=["gemm", "add"], low_precision_nodes=["relu"]
    )

    # Verify that the input and output names are the same as in the original model
    for i in range(len(model.graph.input)):
        assert converted_model.graph.input[i].name == model.graph.input[i].name
    for i in range(len(model.graph.output)):
        assert converted_model.graph.output[i].name == model.graph.output[i].name


####################################################################################################
# Graph with multiple consumers for a single input and multiple consumers for a single initializer
####################################################################################################
@pytest.fixture
def model_with_multiple_consumers():
    # Create a model where a single input and a single initializer are consumed by multiple nodes
    input_shape = [1, 5]
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape)
    y1 = helper.make_tensor_value_info("Y1", TensorProto.FLOAT, [1, 3])
    y2 = helper.make_tensor_value_info("Y2", TensorProto.FLOAT, [1, 3])
    y3 = helper.make_tensor_value_info("Y3", TensorProto.FLOAT, [1, 3])

    # Create initializers
    gemm1_init_numpy = np.random.randn(5, 3).astype(np.float32)
    add_init_numpy = np.random.randn(1, 3).astype(np.float32)
    gemm2_init_numpy = np.random.randn(5, 3).astype(np.float32)
    gemm1_init = numpy_helper.from_array(gemm1_init_numpy, name="gemm1_init")
    add_init = numpy_helper.from_array(add_init_numpy, name="add_init")
    gemm2_init = numpy_helper.from_array(gemm2_init_numpy, name="gemm2_init")

    # Create nodes where X is used by both gemm1 and gemm2
    gemm1_node = helper.make_node("MatMul", ["X", "gemm1_init"], ["gemm1_output"], name="gemm1")
    gemm2_node = helper.make_node("MatMul", ["X", "gemm2_init"], ["gemm2_output"], name="gemm2")
    add1_node = helper.make_node("Add", ["gemm1_output", "add_init"], ["Y1"], name="add1")
    add2_node = helper.make_node("Add", ["gemm2_output", "add_init"], ["Y2"], name="add2")
    add3_node = helper.make_node("Add", ["gemm1_output", "add_init"], ["Y3"], name="add3")

    graph = helper.make_graph(
        [gemm1_node, gemm2_node, add1_node, add2_node, add3_node],
        "model_multi_consumer",
        [x],
        [y1, y2, y3],
        [gemm1_init, add_init, gemm2_init],
    )
    model = helper.make_model(graph, producer_name="model_multi_consumer")
    model.opset_import[0].version = 20
    model.ir_version = 10
    onnx.checker.check_model(model)

    model = onnx_utils.infer_shapes(model)
    value_info_map, initializer_map, node_to_init_map = utils.setup_mappings(model)

    return model, value_info_map, initializer_map, node_to_init_map


@pytest.mark.parametrize("keep_io_types", [True, False])
@pytest.mark.parametrize("low_precision_type", ["fp16", "bf16"])
def test_convert_with_multiple_consumers(
    model_with_multiple_consumers, keep_io_types, low_precision_type
):
    model, value_info_map, initializer_map, node_to_init_map = model_with_multiple_consumers
    converter = PrecisionConverter(
        model,
        value_info_map,
        initializer_map,
        node_to_init_map,
        keep_io_types=keep_io_types,
        low_precision_type=low_precision_type,
    )

    # Only gemm1 and add1 are converted to fp32, gemm2 and add2 are fp16
    converted_model = converter.convert(
        high_precision_nodes=["gemm1", "add1"], low_precision_nodes=["gemm2", "add2"]
    )

    # Verify input type changed to low precision according to keep_io_types
    expected_input_type = (
        TensorProto.FLOAT if keep_io_types else low_precision_onnx_type(low_precision_type)
    )
    assert converted_model.graph.input[0].type.tensor_type.elem_type == expected_input_type

    # Verify cast nodes were added
    cast_nodes = [n for n in converted_model.graph.node if n.op_type == "Cast"]
    assert len(cast_nodes) > 0

    # Verify that the model is valid
    onnx.checker.check_model(converted_model)


@pytest.mark.parametrize("keep_io_types", [True, False])
@pytest.mark.parametrize("low_precision_type", ["fp16", "bf16"])
def test_get_tensors_to_cast_multiple_consumers(
    model_with_multiple_consumers, keep_io_types, low_precision_type
):
    model, value_info_map, initializer_map, node_to_init_map = model_with_multiple_consumers
    converter = PrecisionConverter(
        model,
        value_info_map,
        initializer_map,
        node_to_init_map,
        keep_io_types=keep_io_types,
        low_precision_type=low_precision_type,
    )

    # Test when gemm2 and add1 nodes are in low precision
    cast_down, cast_up, _ = converter._get_tensors_to_cast(["gemm2", "add1"])
    assert "X" in cast_down  # Input to gemm2 should be cast down
    assert "gemm2_output" in cast_up  # Output of gemm2 should be cast up
    assert "Y1" in cast_up  # Output of add1 should be cast up

    # Test when all nodes except gemm1 are in low precision
    cast_down, cast_up, _ = converter._get_tensors_to_cast(["gemm2", "add1", "add2"])
    assert "gemm1_output" in cast_down  # Input to gemm2 should be cast down
    assert "Y1" in cast_up  # Output of add1 should be cast up
    assert "Y2" in cast_up  # Output of add2 should be cast up


@pytest.mark.parametrize("low_precision_type", ["fp16", "bf16"])
def test_convert_initializers(model_with_multiple_consumers, low_precision_type):
    model, value_info_map, initializer_map, node_to_init_map = model_with_multiple_consumers
    converter = PrecisionConverter(
        model,
        value_info_map,
        initializer_map,
        node_to_init_map,
        low_precision_type=low_precision_type,
    )

    # Test successful cast, add1 and add2 share add_init and operate in different precisions
    add1_node = next(n for n in converter.model.graph.node if n.name == "add1")
    add2_node = next(n for n in converter.model.graph.node if n.name == "add2")
    add3_node = next(n for n in converter.model.graph.node if n.name == "add3")
    assert add1_node.input[1] == add2_node.input[1] == add3_node.input[1]

    converter._convert_initializers(
        low_precision_nodes=["add1"], high_precision_nodes=["add2", "add3"]
    )

    assert add1_node.input[1] != add2_node.input[1]  # add1 and add2 have different initializers
    assert add2_node.input[1] == add3_node.input[1]  # add2 and add3 share the same initializer
    init_names = [init.name for init in converter.model.graph.initializer]
    assert len(init_names) == len(set(init_names))  # no duplicate initializers
    assert "add_init" in init_names
    assert f"add_init_{low_precision_type}" in init_names

    # Test successful cast, add1, add2, add3 share add_init.and all operate in FP16
    converter2 = PrecisionConverter(
        model,
        value_info_map,
        initializer_map,
        node_to_init_map,
        low_precision_type=low_precision_type,
    )
    add1_node = next(n for n in converter2.model.graph.node if n.name == "add1")
    add2_node = next(n for n in converter2.model.graph.node if n.name == "add2")
    add3_node = next(n for n in converter2.model.graph.node if n.name == "add3")
    assert add1_node.input[1] == add2_node.input[1] == add3_node.input[1]

    converter2._convert_initializers(
        low_precision_nodes=["add1", "add2", "add3"], high_precision_nodes=[]
    )

    assert add1_node.input[1] == add2_node.input[1] == add3_node.input[1]
    init_names = [init.name for init in converter2.model.graph.initializer]
    assert len(init_names) == len(set(init_names))  # no duplicate initializers
    assert "add_init" in init_names
    assert f"add_init_{low_precision_type}" not in init_names

    # Test successful cast, add1 and add2 share add_init and both operate in FP16, 'add3' is kept in FP32
    converter3 = PrecisionConverter(
        model,
        value_info_map,
        initializer_map,
        node_to_init_map,
        low_precision_type=low_precision_type,
    )
    add1_node = next(n for n in converter3.model.graph.node if n.name == "add1")
    add2_node = next(n for n in converter3.model.graph.node if n.name == "add2")
    add3_node = next(n for n in converter3.model.graph.node if n.name == "add3")
    assert add1_node.input[1] == add2_node.input[1] == add3_node.input[1]

    converter3._convert_initializers(
        low_precision_nodes=["add1", "add2"], high_precision_nodes=["add3"]
    )

    assert (
        add1_node.input[1] == add2_node.input[1]
    )  # after cast, add1 and add2 have the same initializer
    assert add1_node.input[1] != add3_node.input[1]  # add1 and add3 have different initializers
    init_names = [init.name for init in converter3.model.graph.initializer]
    assert len(init_names) == len(set(init_names))  # no duplicate initializers
    assert "add_init" in init_names
    assert f"add_init_{low_precision_type}" in init_names


def test_clamping_fp16_initializers_out_of_range(model_with_multiple_consumers):
    model, value_info_map, initializer_map, node_to_init_map = model_with_multiple_consumers

    # Initializer is out of FP16 range, node is converted to FP16
    add_init_out_of_range = np.array([[-70000.0, 70000.0]], dtype=np.float32)
    add_init = numpy_helper.from_array(add_init_out_of_range, name="add_init")
    model.graph.initializer[1].CopyFrom(add_init)

    converter = PrecisionConverter(model, value_info_map, initializer_map, node_to_init_map)
    converter._convert_initializers(low_precision_nodes=["add1", "add2"], high_precision_nodes=[])

    # Verify initializer is clamped
    add_init_converted = [
        init for init in converter.model.graph.initializer if init.name == "add_init"
    ]
    assert len(add_init_converted) == 1
    add_init_converted_array = numpy_helper.to_array(add_init_converted[0])
    assert add_init_converted_array.dtype == np.float16
    assert add_init_converted_array.shape == (1, 2)
    assert add_init_converted_array[0, 0] == np.finfo(np.float16).min
    assert add_init_converted_array[0, 1] == np.finfo(np.float16).max

    # Initializer is out of FP16 range, node is kept in FP32
    converter2 = PrecisionConverter(model, value_info_map, initializer_map, node_to_init_map)
    converter2._convert_initializers(low_precision_nodes=[], high_precision_nodes=["add1", "add2"])

    # Verify initializer is not clamped
    add_init_converted = [
        init for init in converter2.model.graph.initializer if init.name == "add_init"
    ]
    assert len(add_init_converted) == 1
    add_init_converted_array = numpy_helper.to_array(add_init_converted[0])
    assert add_init_converted_array.dtype == np.float32
    assert add_init_converted_array.shape == (1, 2)
    assert np.all(add_init_converted_array == add_init_out_of_range)

    # Initializer is out of FP16 range, one consumer is converted to FP16, the other is kept in FP32
    converter3 = PrecisionConverter(model, value_info_map, initializer_map, node_to_init_map)
    converter3._convert_initializers(low_precision_nodes=["add1"], high_precision_nodes=["add2"])

    # Verify initializer is duplicated, and the FP16 copy is clamped
    add_init_fp16 = [
        init for init in converter3.model.graph.initializer if init.name == "add_init_fp16"
    ]
    add_init_fp32 = [init for init in converter3.model.graph.initializer if init.name == "add_init"]
    assert len(add_init_fp16) == 1
    assert len(add_init_fp32) == 1
    add_init_fp16_array = numpy_helper.to_array(add_init_fp16[0])
    add_init_fp32_array = numpy_helper.to_array(add_init_fp32[0])
    assert add_init_fp16_array.dtype == np.float16
    assert add_init_fp32_array.dtype == np.float32
    assert np.all(
        add_init_fp16_array
        == np.asarray([np.finfo(np.float16).min, np.finfo(np.float16).max], dtype=np.float16)
    )
    assert np.all(add_init_fp32_array == add_init_out_of_range)


def test_bf16_no_clamping_initializers_out_of_range(model_with_multiple_consumers):
    model, value_info_map, initializer_map, node_to_init_map = model_with_multiple_consumers

    # Initializer is out of FP16 range, but that does not affect BF16 conversion
    add_init_out_of_range = np.array([[2 << 16, 2 << 17]], dtype=np.float32)
    add_init = numpy_helper.from_array(add_init_out_of_range, name="add_init")
    model.graph.initializer[1].CopyFrom(add_init)

    converter = PrecisionConverter(
        model,
        value_info_map,
        initializer_map,
        node_to_init_map,
        low_precision_type="bf16",
    )
    converter._convert_initializers(low_precision_nodes=["add1", "add2"], high_precision_nodes=[])

    # Verify initializer is not clamped
    add_init_converted = [
        init for init in converter.model.graph.initializer if init.name == "add_init"
    ]
    assert len(add_init_converted) == 1
    add_init_converted_array = onnx_utils.read_f16_tensor_as_fp32(add_init_converted[0])
    print(add_init_out_of_range)
    print(add_init_converted_array)
    assert add_init_converted_array.shape == (1, 2)
    assert np.all(add_init_converted_array == add_init_out_of_range)


####################################################################################################
# Testing with dynamic shapes, since shape_inference invoked in PrecisionConverter
####################################################################################################
@pytest.fixture
def model_with_dynamic_shapes():
    # Create inputs with different dynamic shapes
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, 4])  # Dynamic batch size
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [8, None])  # Dynamic second dimension
    z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [8, 4])  # Output shape

    # Create initializers
    weight = np.random.randn(4, 8).astype(np.float32)
    bias = np.random.randn(8).astype(np.float32)
    weight_init = numpy_helper.from_array(weight, name="weight")
    bias_init = numpy_helper.from_array(bias, name="bias")

    # Create nodes with shape-changing operations
    matmul_node = helper.make_node("MatMul", ["X", "weight"], ["matmul_out"], name="matmul")
    transpose_node = helper.make_node("Transpose", ["Y"], ["transpose_out"], name="transpose")
    concat_node = helper.make_node(
        "Concat", ["matmul_out", "transpose_out"], ["concat_out"], name="concat", axis=0
    )
    size_y = helper.make_node("Size", ["concat_out"], ["total_size"], name="size")
    const_4 = numpy_helper.from_array(np.array([4], dtype=np.int64), name="const_4")
    first_dim = helper.make_node("Div", ["total_size", "const_4"], ["first_dim"], name="div")
    concat_dims_node = helper.make_node(
        "Concat", ["first_dim", "const_4"], ["final_shape"], name="concat", axis=0
    )
    reshape_node = helper.make_node("Reshape", ["concat_out", "final_shape"], ["Z"], name="reshape")

    # Create graph and model
    graph = helper.make_graph(
        [
            matmul_node,
            transpose_node,
            concat_node,
            size_y,
            first_dim,
            concat_dims_node,
            reshape_node,
        ],
        "model_dynamic",
        [x, y],
        [z],
        [weight_init, bias_init, const_4],
    )
    model = helper.make_model(graph, producer_name="model_dynamic")
    model.opset_import[0].version = 20
    model.ir_version = 10
    model = onnx_utils.infer_shapes(model)

    value_info_map, initializer_map, node_to_init_map = utils.setup_mappings(model)

    return model, value_info_map, initializer_map, node_to_init_map


def test_dynamic_model_conversion(model_with_dynamic_shapes):
    model, value_info_map, initializer_map, node_to_init_map = model_with_dynamic_shapes

    # Test mixed precision conversion
    converter2 = PrecisionConverter(model, value_info_map, initializer_map, node_to_init_map)
    high_precision_nodes = ["matmul"]
    low_precision_nodes = ["transpose", "concat", "size", "div", "concat_dims", "reshape"]

    converted_model = converter2.convert(high_precision_nodes, low_precision_nodes)
    # Verify model is valid
    onnx.checker.check_model(converted_model)


####################################################################################################
# Cast cleanup logic
####################################################################################################
def test_cast_output_pattern():
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 4])
    y1 = helper.make_tensor_value_info("Y1", TensorProto.FLOAT, [3, 4])
    y2 = helper.make_tensor_value_info("Y2", TensorProto.FLOAT, [3, 4])

    init_weight = numpy_helper.from_array(np.random.randn(3, 4).astype(np.float32), name="weight")

    node1 = helper.make_node("Add", ["X", "weight"], ["Y1"], name="node1")
    node2 = helper.make_node("Reciprocal", ["Y1"], ["Y2"], name="node2")

    graph = helper.make_graph(
        [node1, node2],
        "model_output_cast",
        [x],
        [y1, y2],
        [init_weight],
    )
    model = helper.make_model(graph, producer_name="model_double_cast")
    model.opset_import[0].version = 20
    model.ir_version = 10
    model = onnx_utils.infer_shapes(model)

    value_info_map, initializer_map, node_to_init_map = utils.setup_mappings(model)
    converter = PrecisionConverter(model, value_info_map, initializer_map, node_to_init_map)

    # Setting all nodes to FP16 means that the final graph should have no cast nodes
    converted_model = converter.convert(
        high_precision_nodes=[], low_precision_nodes=["node1", "node2"]
    )

    # Verify all cast nodes were removed
    cast_nodes = [n for n in converted_model.graph.node if n.op_type == "Cast"]
    assert len(cast_nodes) == 0
    assert len(converted_model.graph.node) == 2
    # Verify that the output names are the same as in the original model
    for i in range(len(model.graph.output)):
        assert converted_model.graph.output[i].name == model.graph.output[i].name


def test_cast_output_pattern_mixed_precision():
    x1 = helper.make_tensor_value_info("X1", TensorProto.FLOAT, [3, 4])
    x2 = helper.make_tensor_value_info("X2", TensorProto.FLOAT, [3, 4])
    y0 = helper.make_tensor_value_info("Y0", TensorProto.FLOAT, [3, 4])
    y1 = helper.make_tensor_value_info("Y1", TensorProto.FLOAT, [3, 4])
    y2 = helper.make_tensor_value_info("Y2", TensorProto.FLOAT, [3, 4])

    init_weight = numpy_helper.from_array(np.random.randn(3, 4).astype(np.float32), name="weight")

    node0 = helper.make_node("Add", ["X1", "weight"], ["Y0"], name="node0")
    node1 = helper.make_node("Mul", ["X2", "Y0"], ["Y1"], name="node1")
    node2 = helper.make_node("Div", ["X2", "Y0"], ["Y2"], name="node2")

    graph = helper.make_graph(
        [node0, node1, node2],
        "model_output_cast",
        [x1, x2],
        [y0, y1, y2],
        [init_weight],
    )
    model = helper.make_model(graph, producer_name="model_double_cast")
    model.opset_import[0].version = 20
    model.ir_version = 10
    model = onnx_utils.infer_shapes(model)

    value_info_map, initializer_map, node_to_init_map = utils.setup_mappings(model)
    converter = PrecisionConverter(model, value_info_map, initializer_map, node_to_init_map)

    # Network output Y0 has two consumers, one is FP16 and the other is FP32
    converted_model = converter.convert(
        high_precision_nodes=["node0", "node2"], low_precision_nodes=["node1"]
    )

    # Verify that the output names are the same as in the original model
    for i in range(len(model.graph.output)):
        assert converted_model.graph.output[i].name == model.graph.output[i].name


@pytest.mark.parametrize("keep_io_types", [True, False])
def test_chain_of_casts_pattern(keep_io_types):
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 4])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 4])

    init_weight = numpy_helper.from_array(np.random.randn(3, 4).astype(np.float32), name="weight")
    cast1_node = helper.make_node("Cast", ["X"], ["cast1_out"], name="cast1", to=TensorProto.FLOAT)
    cast2_node = helper.make_node(
        "Cast", ["cast1_out"], ["cast2_out"], name="cast2", to=TensorProto.FLOAT
    )
    cast3_node = helper.make_node(
        "Cast", ["cast2_out"], ["cast3_out"], name="cast3", to=TensorProto.FLOAT
    )
    cast4_node = helper.make_node(
        "Cast", ["cast3_out"], ["cast4_out"], name="cast4", to=TensorProto.FLOAT
    )
    add_node = helper.make_node("Add", ["cast4_out", "weight"], ["add_out"], name="add")
    cast5_node = helper.make_node(
        "Cast", ["add_out"], ["cast5_out"], name="cast5", to=TensorProto.FLOAT16
    )
    cast6_node = helper.make_node(
        "Cast", ["cast5_out"], ["cast6_out"], name="cast6", to=TensorProto.FLOAT16
    )
    cast7_node = helper.make_node(
        "Cast", ["cast6_out"], ["cast7_out"], name="cast7", to=TensorProto.FLOAT16
    )
    cast8_node = helper.make_node(
        "Cast", ["cast7_out"], ["Y"], name="cast8", to=TensorProto.FLOAT16
    )

    graph = helper.make_graph(
        [
            cast1_node,
            cast2_node,
            cast3_node,
            cast4_node,
            add_node,
            cast5_node,
            cast6_node,
            cast7_node,
            cast8_node,
        ],
        "model_cast_chain",
        [x],
        [y],
        [init_weight],
    )
    model = helper.make_model(graph, producer_name="model_cast_chain")
    model.opset_import[0].version = 20
    model.ir_version = 10
    model = onnx_utils.infer_shapes(model)

    value_info_map, initializer_map, node_to_init_map = utils.setup_mappings(model)
    converter = PrecisionConverter(
        model, value_info_map, initializer_map, node_to_init_map, keep_io_types=keep_io_types
    )
    converter.convert(high_precision_nodes=["add"], low_precision_nodes=[])

    # Verify cast chain was removed
    cast_nodes = [n for n in converter.model.graph.node if n.op_type == "Cast"]
    expected_cast_nodes = 0 if keep_io_types else 2
    assert len(cast_nodes) == expected_cast_nodes


@pytest.mark.parametrize("low_precision_type", ["fp16", "bf16"])
def test_existing_low_precision_output(low_precision_type):
    # Create a simple model with FP16 output
    x = helper.make_tensor_value_info("X", low_precision_onnx_type(low_precision_type), [3, 4])
    y = helper.make_tensor_value_info("Y", low_precision_onnx_type(low_precision_type), [3, 4])
    add_node = helper.make_node("Add", ["X", "X"], ["Y"], name="add")
    graph = helper.make_graph([add_node], "model_add", [x], [y], [])
    model = helper.make_model(graph, producer_name="model_add")
    model.opset_import[0].version = 20
    model.ir_version = 10

    model = onnx_utils.infer_shapes(model)
    value_info_map, initializer_map, node_to_init_map = utils.setup_mappings(model)

    converter = PrecisionConverter(
        model,
        value_info_map,
        initializer_map,
        node_to_init_map,
        keep_io_types=True,
        low_precision_type=low_precision_type,
    )
    converter.convert(high_precision_nodes=["add"], low_precision_nodes=[])

    assert len(converter.model.graph.node) == 3
    assert converter.model.graph.node[0].op_type == "Cast"
    assert converter.model.graph.node[1].op_type == "Add"
    assert converter.model.graph.node[2].op_type == "Cast"

    # check that the I/O remains in low precision
    assert converter.model.graph.input[0].type.tensor_type.elem_type == low_precision_onnx_type(
        low_precision_type
    )
    assert converter.model.graph.output[0].type.tensor_type.elem_type == low_precision_onnx_type(
        low_precision_type
    )


@pytest.mark.parametrize("low_precision_type", ["fp16", "bf16"])
def test_output_cast_output_pattern(low_precision_type):
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 4])
    y1 = helper.make_tensor_value_info("Y1", TensorProto.FLOAT, [3, 4])
    y2 = helper.make_tensor_value_info("Y2", low_precision_onnx_type(low_precision_type), [3, 4])

    init_weight = numpy_helper.from_array(np.random.randn(3, 4).astype(np.float32), name="weight")

    cast_to = low_precision_onnx_type(low_precision_type)
    add_node = helper.make_node("Add", ["X", "weight"], ["Y1"], name="add")
    cast_node = helper.make_node("Cast", ["Y1"], ["Y2"], name="cast", to=cast_to)

    graph = helper.make_graph(
        [add_node, cast_node],
        "model_output_cast_output",
        [x],
        [y1, y2],
        [init_weight],
    )
    model = helper.make_model(graph, producer_name="model_output_cast_output")
    model.opset_import[0].version = 20
    model.ir_version = 10
    model = onnx_utils.infer_shapes(model)

    value_info_map, initializer_map, node_to_init_map = utils.setup_mappings(model)
    converter = PrecisionConverter(
        model,
        value_info_map,
        initializer_map,
        node_to_init_map,
        keep_io_types=True,
        low_precision_type=low_precision_type,
    )

    # Setting nodes precision to match I/O type means that the final graph should have no cast nodes
    converted_model = converter.convert(high_precision_nodes=["add"], low_precision_nodes=[])

    assert len(converted_model.graph.node) == 2
    assert converted_model.graph.node[0].op_type == "Add"
    assert converted_model.graph.node[1].op_type == "Cast"

    y1_out = next(y for y in converted_model.graph.output if y.name == "Y1")
    y2_out = next(y for y in converted_model.graph.output if y.name == "Y2")
    assert y1_out.type.tensor_type.elem_type == TensorProto.FLOAT
    assert y2_out.type.tensor_type.elem_type == low_precision_onnx_type(low_precision_type)


@pytest.mark.parametrize("low_precision_type", ["fp16", "bf16"])
def test_cast_output_keep_io_types_pattern(low_precision_type):
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 4])
    y1 = helper.make_tensor_value_info("Y1", TensorProto.FLOAT, [3, 4])
    y2 = helper.make_tensor_value_info("Y2", TensorProto.FLOAT, [3, 4])

    init_weight = numpy_helper.from_array(np.random.randn(3, 4).astype(np.float32), name="weight")

    add1_node = helper.make_node("Add", ["X", "weight"], ["Y1"], name="add1")
    add2_node = helper.make_node("Add", ["Y1", "weight"], ["Y2"], name="add2")
    graph = helper.make_graph(
        [add1_node, add2_node],
        "model_cast_output_keep_io_types",
        [x],
        [y1, y2],
        [init_weight],
    )
    model = helper.make_model(graph, producer_name="model_cast_output_keep_io_types")
    model.opset_import[0].version = 20
    model.ir_version = 10
    model = onnx_utils.infer_shapes(model)

    value_info_map, initializer_map, node_to_init_map = utils.setup_mappings(model)
    converter = PrecisionConverter(
        model,
        value_info_map,
        initializer_map,
        node_to_init_map,
        keep_io_types=True,
        low_precision_type=low_precision_type,
    )
    converter.convert(high_precision_nodes=[], low_precision_nodes=["add1", "add2"])

    # Outputs should be FP32
    assert converter.model.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT
    assert converter.model.graph.output[1].type.tensor_type.elem_type == TensorProto.FLOAT


def test_unsupported_op_types_model():
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 4])
    roi = helper.make_tensor_value_info("roi", TensorProto.FLOAT, [3, 4])
    scales = helper.make_tensor_value_info("scales", TensorProto.FLOAT, [4])
    boxes = helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [3, 4, 4])
    scores = helper.make_tensor_value_info("scores", TensorProto.FLOAT, [3, 4])
    celu_out = helper.make_tensor_value_info("celu_out", TensorProto.FLOAT, [3, 4])
    resize_out = helper.make_tensor_value_info("resize_out", TensorProto.FLOAT, [3, 4])
    nms_out = helper.make_tensor_value_info("nms_out", TensorProto.INT64, [-1, 3])

    node1 = helper.make_node("Celu", ["X"], ["celu_out"], name="celu")
    node2 = helper.make_node("Resize", ["X", "roi", "scales"], ["resize_out"], name="resize")
    node3 = helper.make_node("NonMaxSuppression", ["boxes", "scores"], ["nms_out"], name="nms")
    graph = helper.make_graph(
        [node1, node2, node3],
        "model_celu",
        [x, roi, scales, boxes, scores],
        [celu_out, resize_out, nms_out],
        [],
    )
    model = helper.make_model(graph, producer_name="model_celu")
    model = onnx.shape_inference.infer_shapes(model)

    value_info_map, initializer_map, node_to_init_map = utils.setup_mappings(model)
    converter = PrecisionConverter(model, value_info_map, initializer_map, node_to_init_map)
    converter.convert(high_precision_nodes=[], low_precision_nodes=["celu", "resize", "nms"])
    onnx.checker.check_model(converter.model)


@pytest.mark.parametrize("low_precision_type", ["fp16", "bf16"])
@pytest.mark.parametrize("empty_tensor_target", ["low_precision", "high_precision"])
def test_empty_tensor_handling(low_precision_type, empty_tensor_target):
    """Test empty tensor handling for both low and high precision node targets."""
    # Create model with empty float tensor from Constant layer
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2])

    # Create Constant node that outputs empty float tensor
    empty_tensor = np.array([], dtype=np.float32)  # Empty float array
    constant_node = helper.make_node(
        "Constant",
        [],
        ["empty_tensor"],
        name="constant_empty",
        value=numpy_helper.from_array(empty_tensor, name="empty_value"),
    )

    # Use empty tensor in Concat operation (concatenating with itself effectively)
    concat_node = helper.make_node(
        "Concat", ["X", "empty_tensor"], ["concat_output"], name="concat", axis=0
    )
    relu_node = helper.make_node("Relu", ["concat_output"], ["Y"], name="relu")

    graph = helper.make_graph(
        [constant_node, concat_node, relu_node], "model_empty_tensor", [x], [y], []
    )
    model = helper.make_model(graph, producer_name="model_empty_tensor")
    model.opset_import[0].version = 20
    model.ir_version = 10
    onnx.checker.check_model(model)

    model = onnx_utils.infer_shapes(model)
    value_info_map, initializer_map, node_to_init_map = utils.setup_mappings(model)
    converter = PrecisionConverter(
        model,
        value_info_map,
        initializer_map,
        node_to_init_map,
        keep_io_types=True,
        low_precision_type=low_precision_type,
    )

    # Test empty tensor detection
    assert converter._is_empty_tensor("empty_tensor")
    assert not converter._is_empty_tensor("X")

    # Convert based on target
    if empty_tensor_target == "low_precision":
        converted_model = converter.convert(
            high_precision_nodes=["relu"], low_precision_nodes=["concat"]
        )
    else:
        converted_model = converter.convert(
            high_precision_nodes=["concat"], low_precision_nodes=["relu"]
        )

    # Verify model is valid and empty tensor type is updated in value_info
    onnx.checker.check_model(converted_model)
    empty_tensor_info = next(
        vi for vi in converted_model.graph.value_info if vi.name == "empty_tensor"
    )
    expected_type = (
        low_precision_onnx_type(low_precision_type)
        if empty_tensor_target == "low_precision"
        else TensorProto.FLOAT
    )
    assert empty_tensor_info.type.tensor_type.elem_type == expected_type


@pytest.fixture
def model_with_constant_cast_patterns():
    """Create a model with constant->cast patterns for testing folding logic."""
    # Create inputs and outputs
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])

    # Create constant values with different data types
    fp32_scalar = np.array(2.5, dtype=np.float32)  # 0-dimensional tensor
    fp32_array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)

    # Create constant nodes
    const_scalar = helper.make_node(
        "Constant",
        [],
        ["const_scalar"],
        name="const_scalar",
        value=numpy_helper.from_array(fp32_scalar, name="scalar_value"),
    )

    const_array = helper.make_node(
        "Constant",
        [],
        ["const_array"],
        name="const_array",
        value=numpy_helper.from_array(fp32_array, name="array_value"),
    )
    # Create consumer nodes
    add1 = helper.make_node("Add", ["X", "const_scalar"], ["add1_out"], name="add1")
    add2 = helper.make_node("Add", ["X", "const_array"], ["add2_out"], name="add2")

    # Create output node
    mean = helper.make_node("Mean", ["add1_out", "add2_out"], ["Y"], name="mean")

    graph = helper.make_graph(
        [
            const_scalar,
            const_array,
            add1,
            add2,
            mean,
        ],
        "model_constant_cast_patterns",
        [x],
        [y],
        [],
    )

    model = helper.make_model(graph, producer_name="model_constant_cast_patterns")
    model.opset_import[0].version = 20
    model.ir_version = 10
    onnx.checker.check_model(model)

    model = onnx_utils.infer_shapes(model)
    value_info_map, initializer_map, node_to_init_map = utils.setup_mappings(model)

    return model, value_info_map, initializer_map, node_to_init_map


@pytest.mark.parametrize("low_precision_type", ["fp16", "bf16"])
def test_constant_cast_folding(model_with_constant_cast_patterns, low_precision_type):
    """Test constant->cast folding as part of the full conversion process."""
    model, value_info_map, initializer_map, node_to_init_map = model_with_constant_cast_patterns

    converter = PrecisionConverter(
        model,
        value_info_map,
        initializer_map,
        node_to_init_map,
        keep_io_types=True,
        low_precision_type=low_precision_type,
    )

    # Convert with some nodes in low precision to trigger cast insertion
    converted_model = converter.convert(
        high_precision_nodes=["add3"],
        low_precision_nodes=["add1", "add2", "mean"],
    )

    # Verify the model is valid
    onnx.checker.check_model(converted_model)

    # Check Constant nodes are converted to low precision
    const_scalar = next(
        n
        for n in converted_model.graph.node
        if n.op_type == "Constant" and n.name == "const_scalar"
    )
    const_array = next(
        n for n in converted_model.graph.node if n.op_type == "Constant" and n.name == "const_array"
    )
    assert const_scalar.attribute[0].t.data_type == low_precision_onnx_type(low_precision_type)
    assert const_array.attribute[0].t.data_type == low_precision_onnx_type(low_precision_type)

    # Check that the constant nodes are consumed directly by the Add nodes
    assert len(utils.get_consumer_nodes(converted_model, "const_scalar")) == 1
    assert utils.get_consumer_nodes(converted_model, "const_scalar")[0].op_type == "Add"
    assert len(utils.get_consumer_nodes(converted_model, "const_array")) == 1
    assert utils.get_consumer_nodes(converted_model, "const_array")[0].op_type == "Add"


@pytest.fixture
def model_with_multiple_output_node_casted_to_output():
    """Create a model with a Cast node connecting a consumer with multiple outputs to a graph output."""
    # Create inputs and outputs
    x1 = helper.make_tensor_value_info("X1", TensorProto.FLOAT, [1, 2, 16, 16])
    x2 = helper.make_tensor_value_info("X2", TensorProto.FLOAT, [1, 3, 16, 16])
    x3 = helper.make_tensor_value_info("X3", TensorProto.FLOAT, [1, 4, 16, 16])
    y1 = helper.make_tensor_value_info("Y1", TensorProto.FLOAT, [1, 5, 16, 16])
    y2 = helper.make_tensor_value_info("Y2", TensorProto.FLOAT, [1, 9, 16, 16])

    # Create computation nodes
    concat_1_node = helper.make_node(
        "Concat",
        ["X1", "X2"],
        ["concat_1_out"],
        name="concat_1",
        axis=1,
    )
    concat_2_node = helper.make_node(
        "Concat",
        ["concat_1_out", "X3"],
        ["Y2"],
        name="concat_2",
        axis=1,
    )

    # Create a Cast node between 'concat_1' and the graph output
    cast_node = helper.make_node(
        "Cast",
        ["concat_1_out"],
        ["Y1"],
        name="cast_0",
        to=TensorProto.FLOAT,
    )

    graph = helper.make_graph(
        [concat_1_node, concat_2_node, cast_node],
        "model_with_multiple_output_node_casted_to_output",
        [x1, x2, x3],
        [y1, y2],
        [],
    )

    model = helper.make_model(
        graph, producer_name="model_with_multiple_output_node_casted_to_output"
    )
    model.opset_import[0].version = 20
    model.ir_version = 10
    onnx.checker.check_model(model)

    model = onnx_utils.infer_shapes(model)
    value_info_map, initializer_map, node_to_init_map = utils.setup_mappings(model)

    return model, value_info_map, initializer_map, node_to_init_map


@pytest.mark.parametrize("low_precision_type", ["fp16", "bf16"])
def test_multiple_output_node_casted_to_output(
    model_with_multiple_output_node_casted_to_output, low_precision_type
):
    model, value_info_map, initializer_map, node_to_init_map = (
        model_with_multiple_output_node_casted_to_output
    )

    converter = PrecisionConverter(
        model,
        value_info_map,
        initializer_map,
        node_to_init_map,
        keep_io_types=True,
        low_precision_type=low_precision_type,
    )
    converted_model = converter.convert(
        high_precision_nodes=[], low_precision_nodes=["concat_1", "concat_2"]
    )
    onnx.checker.check_model(converted_model)


@pytest.fixture
def model_with_casted_input_to_output():
    """Create a model with an output produced by a Cast node."""
    # Create input and outputs
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
    y1 = helper.make_tensor_value_info("Y1", TensorProto.FLOAT, [2, 3])  # Intermediate output
    y2 = helper.make_tensor_value_info("Y2", TensorProto.FLOAT, [2, 3])  # Final output

    # Create constant value
    const = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)

    # Create constant node
    const_node = helper.make_node(
        "Constant",
        [],
        ["const"],
        name="const",
        value=numpy_helper.from_array(const, name="const_value"),
    )

    # Create computation nodes
    add1 = helper.make_node("Add", ["X", "const"], ["add1_out"], name="add1")
    add2 = helper.make_node("Add", ["add1_out", "const"], ["Y2"], name="add2")

    # Create cast node that feeds directly from input to output
    cast_input = helper.make_node("Cast", ["X"], ["Y1"], name="cast_input", to=TensorProto.FLOAT)

    graph = helper.make_graph(
        [const_node, add1, add2, cast_input],
        "model_with_casted_output",
        [x],
        [y1, y2],
        [],
    )

    model = helper.make_model(graph, producer_name="model_with_casted_output")
    model.opset_import[0].version = 20
    model.ir_version = 10
    onnx.checker.check_model(model)

    model = onnx_utils.infer_shapes(model)
    value_info_map, initializer_map, node_to_init_map = utils.setup_mappings(model)

    return model, value_info_map, initializer_map, node_to_init_map


@pytest.mark.parametrize("low_precision_type", ["fp16", "bf16"])
@pytest.mark.parametrize("keep_io_types", [True, False])
def test_casted_input_to_output_model(
    model_with_casted_input_to_output, low_precision_type, keep_io_types
):
    model, value_info_map, initializer_map, node_to_init_map = model_with_casted_input_to_output

    converter = PrecisionConverter(
        model,
        value_info_map,
        initializer_map,
        node_to_init_map,
        keep_io_types=keep_io_types,
        low_precision_type=low_precision_type,
        min_opset=22 if low_precision_type == "bf16" else 13,
        max_ir_version=LATEST_IR_VERSION_SUPPORTED_BY_ORT,
        trt_plugins=[],
    )
    converted_model = converter.convert(
        high_precision_nodes=["cast_input"], low_precision_nodes=["add1", "add2"]
    )
    onnx.checker.check_model(converted_model)


@pytest.fixture
def create_model_with_resize_op():
    """
    Creates an ONNX model that contains a resize operation in the middle of the computation flow.

    The model structure:
    X -> Add -> Resize -> Relu -> Y
    """
    # Create inputs and outputs
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 32, 32])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 64, 64])

    # Create initializer for add operation
    add_const = np.ones((1, 3, 32, 32), dtype=np.float32)
    add_init = numpy_helper.from_array(add_const, name="add_const")

    # Create resize parameters
    roi_empty = numpy_helper.from_array(np.array([], dtype=np.float32), name="roi")
    scales = numpy_helper.from_array(
        np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32), name="scales"
    )

    # Create nodes: Add -> Resize -> Relu
    add_node = helper.make_node("Add", ["X", "add_const"], ["add_out"], name="add")
    resize_node = helper.make_node(
        "Resize", ["add_out", "roi", "scales"], ["resize_out"], name="resize", mode="nearest"
    )
    relu_node = helper.make_node("Relu", ["resize_out"], ["Y"], name="relu")

    # Build the graph
    graph = helper.make_graph(
        [add_node, resize_node, relu_node],
        "model_with_resize",
        [x],
        [y],
        [add_init, roi_empty, scales],
    )

    model = helper.make_model(graph, producer_name="model_with_resize")
    model.opset_import[0].version = 20
    model.ir_version = 10
    onnx.checker.check_model(model)

    model = onnx_utils.infer_shapes(model)
    value_info_map, initializer_map, node_to_init_map = utils.setup_mappings(model)

    return model, value_info_map, initializer_map, node_to_init_map


@pytest.fixture
def create_model_with_resize_op_tensor_scales():
    """
    Creates an ONNX model that contains a resize operation where the scales
    are computed from a second network input through an Add operation.

    The model structure:
    X -> Add -> Resize -> Relu -> Y
    scales_input -> Add -> scales_tensor /
    """
    # Create inputs and outputs
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 32, 32])
    scales_input = helper.make_tensor_value_info("scales_input", TensorProto.FLOAT, [4])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 64, 64])

    # Create initializers
    add_const = np.ones((1, 3, 32, 32), dtype=np.float32)
    add_init = numpy_helper.from_array(add_const, name="add_const")

    # Create scales computation initializer (add small offset to input scales)
    scales_offset = np.array(
        [0.0, 0.0, 1.0, 1.0], dtype=np.float32
    )  # Will result in [1,1,2,2] when added to [1,1,1,1] input
    scales_offset_init = numpy_helper.from_array(scales_offset, name="scales_offset")

    # Create resize parameters
    roi_empty = numpy_helper.from_array(np.array([], dtype=np.float32), name="roi")

    # Create nodes
    add_node = helper.make_node("Add", ["X", "add_const"], ["add_out"], name="add")
    scales_add_node = helper.make_node(
        "Add", ["scales_input", "scales_offset"], ["scales_tensor"], name="scales_add"
    )
    resize_node = helper.make_node(
        "Resize", ["add_out", "roi", "scales_tensor"], ["resize_out"], name="resize", mode="nearest"
    )
    relu_node = helper.make_node("Relu", ["resize_out"], ["Y"], name="relu")

    # Build the graph
    graph = helper.make_graph(
        [add_node, scales_add_node, resize_node, relu_node],
        "model_with_resize_tensor_scales",
        [x, scales_input],  # Two network inputs
        [y],
        [add_init, scales_offset_init, roi_empty],
    )

    model = helper.make_model(graph, producer_name="model_with_resize_tensor_scales")
    model.opset_import[0].version = 20
    model.ir_version = 10
    onnx.checker.check_model(model)

    model = onnx_utils.infer_shapes(model)
    value_info_map, initializer_map, node_to_init_map = utils.setup_mappings(model)

    return model, value_info_map, initializer_map, node_to_init_map


@pytest.mark.parametrize("keep_io_types", [True, False])
@pytest.mark.parametrize("low_precision_type", ["fp16", "bf16"])
def test_resize_op_initializer_conversion(
    create_model_with_resize_op, keep_io_types, low_precision_type
):
    model, value_info_map, initializer_map, node_to_init_map = create_model_with_resize_op

    converter = PrecisionConverter(
        model,
        value_info_map,
        initializer_map,
        node_to_init_map,
        keep_io_types=keep_io_types,
        low_precision_type=low_precision_type,
    )
    converted_model = converter.convert(
        high_precision_nodes=[], low_precision_nodes=[node.name for node in model.graph.node]
    )
    onnx.checker.check_model(converted_model)


@pytest.mark.parametrize("keep_io_types", [True, False])
@pytest.mark.parametrize("low_precision_type", ["fp16", "bf16"])
def test_resize_op_tensor_scales_conversion(
    create_model_with_resize_op_tensor_scales, keep_io_types, low_precision_type
):
    model, value_info_map, initializer_map, node_to_init_map = (
        create_model_with_resize_op_tensor_scales
    )

    converter = PrecisionConverter(
        model,
        value_info_map,
        initializer_map,
        node_to_init_map,
        keep_io_types=keep_io_types,
        low_precision_type=low_precision_type,
    )
    converted_model = converter.convert(
        high_precision_nodes=[], low_precision_nodes=[node.name for node in model.graph.node]
    )
    onnx.checker.check_model(converted_model)


####################################################################################################
# Testing subgraph support, using If - Then Else subgraphs with initializers
####################################################################################################
@pytest.fixture
def model_with_if_subgraph():
    """Create a model with an If operation containing subgraphs with initializers.

    The model has a preprocessing Add on X, then If branches use initializers.
    This tests both external inputs (X flows through Add) and subgraph initializers.
    """
    # Main graph inputs/outputs
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
    condition = helper.make_tensor_value_info("condition", TensorProto.BOOL, [])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])

    # Add a preprocessing node in main graph to use external input X
    preprocess_weight = numpy_helper.from_array(
        np.ones((2, 3), dtype=np.float32), name="preprocess_weight"
    )
    preprocess_node = helper.make_node(
        "Add", ["X", "preprocess_weight"], ["X_processed"], name="preprocess"
    )

    # Create "then" branch subgraph with initializers
    then_y = helper.make_tensor_value_info("then_y", TensorProto.FLOAT, [2, 4])

    w_true = np.random.randn(2, 4).astype(np.float32)
    b_true = np.random.randn(2, 4).astype(np.float32)
    w_true_init = numpy_helper.from_array(w_true, name="W_true")
    b_true_init = numpy_helper.from_array(b_true, name="b_true")

    then_add = helper.make_node("Add", ["W_true", "b_true"], ["then_y"], name="then_add")

    then_graph = helper.make_graph(
        [then_add],
        "then_branch",
        [],
        [then_y],
        [w_true_init, b_true_init],
    )

    # Create "else" branch subgraph with different initializers
    else_y = helper.make_tensor_value_info("else_y", TensorProto.FLOAT, [2, 4])

    w_false = np.random.randn(2, 4).astype(np.float32) * 2  # Different values
    b_false = np.random.randn(2, 4).astype(np.float32) * 2
    w_false_init = numpy_helper.from_array(w_false, name="W_false")
    b_false_init = numpy_helper.from_array(b_false, name="b_false")

    else_add = helper.make_node("Add", ["W_false", "b_false"], ["else_y"], name="else_add")

    else_graph = helper.make_graph(
        [else_add],
        "else_branch",
        [],
        [else_y],
        [w_false_init, b_false_init],
    )

    # Create If node
    if_node = helper.make_node(
        "If",
        inputs=["condition"],
        outputs=["Y"],
        name="if_node",
        then_branch=then_graph,
        else_branch=else_graph,
    )

    # Create main graph with preprocessing using external input
    main_graph = helper.make_graph(
        [preprocess_node, if_node],
        "model_with_if",
        [x, condition],
        [y],
        [preprocess_weight],
    )

    model = helper.make_model(main_graph, producer_name="model_with_if")
    model.opset_import[0].version = 20
    model.ir_version = 10
    onnx.checker.check_model(model)

    model = onnx_utils.infer_shapes(model)
    value_info_map, initializer_map, node_to_init_map = utils.setup_mappings(model)
    return model, value_info_map, initializer_map, node_to_init_map


@pytest.mark.parametrize("low_precision_type", ["fp16", "bf16"])
@pytest.mark.parametrize("if_precision", ["low", "high"])
def test_if_subgraph_initializer_conversion(
    model_with_if_subgraph, low_precision_type, if_precision
):
    """Test that initializers in If subgraphs are converted based on parent node precision."""
    model, value_info_map, initializer_map, node_to_init_map = model_with_if_subgraph

    converter = PrecisionConverter(
        model,
        value_info_map,
        initializer_map,
        node_to_init_map,
        keep_io_types=True,
        low_precision_type=low_precision_type,
    )

    # Classify the If node based on test parameter
    if if_precision == "low":
        high_precision_nodes = []
        low_precision_nodes = ["if_node"]
        expected_init_type = low_precision_onnx_type(low_precision_type)
    else:
        high_precision_nodes = ["if_node"]
        low_precision_nodes = []
        expected_init_type = TensorProto.FLOAT

    converted_model = converter.convert(high_precision_nodes, low_precision_nodes)

    # Verify the model is valid
    onnx.checker.check_model(converted_model)

    # Find the If node and check its subgraph initializers
    if_node = next(n for n in converted_model.graph.node if n.op_type == "If")

    then_branch = None
    else_branch = None
    for attr in if_node.attribute:
        if attr.name == "then_branch":
            then_branch = attr.g
        elif attr.name == "else_branch":
            else_branch = attr.g

    assert then_branch is not None, "If node should have a then_branch attribute"
    assert else_branch is not None, "If node should have an else_branch attribute"

    # Check that subgraph initializers in both branches were converted
    assert len(then_branch.initializer) == 2, (
        "Then branch should have 2 initializers (W_true, b_true)"
    )
    assert len(else_branch.initializer) == 2, (
        "Else branch should have 2 initializers (W_false, b_false)"
    )

    for init in then_branch.initializer:
        assert init.data_type == expected_init_type, (
            f"Then branch initializer '{init.name}' should be {expected_init_type}, "
            f"but is {init.data_type}"
        )

    for init in else_branch.initializer:
        assert init.data_type == expected_init_type, (
            f"Else branch initializer '{init.name}' should be {expected_init_type}, "
            f"but is {init.data_type}"
        )


@pytest.mark.parametrize("low_precision_type", ["fp16", "bf16"])
def test_if_subgraph_mixed_precision_boundary(model_with_if_subgraph, low_precision_type):
    """Test that types are correctly handled at If subgraph boundaries in mixed precision."""
    model, value_info_map, initializer_map, node_to_init_map = model_with_if_subgraph

    # Add another node after the If to create a mixed precision scenario
    add_weight = numpy_helper.from_array(np.ones((2, 4), dtype=np.float32), name="add_weight")
    model.graph.initializer.append(add_weight)

    add_node = helper.make_node("Add", ["Y", "add_weight"], ["output"], name="add_after_if")
    model.graph.node.append(add_node)

    # Update output
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 4])
    model.graph.output.append(output_tensor)

    # Refresh mappings
    value_info_map, initializer_map, node_to_init_map = utils.setup_mappings(model)

    converter = PrecisionConverter(
        model,
        value_info_map,
        initializer_map,
        node_to_init_map,
        keep_io_types=True,
        low_precision_type=low_precision_type,
    )

    # If in low precision, Add in high precision
    converted_model = converter.convert(
        high_precision_nodes=["add_after_if"], low_precision_nodes=["if_node"]
    )

    # Verify the model is valid (this tests type inference through subgraph boundaries)
    onnx.checker.check_model(converted_model)

    # Verify a cast was inserted between If output and Add input
    cast_nodes = [n for n in converted_model.graph.node if n.op_type == "Cast"]
    assert len(cast_nodes) > 0, "Should have cast nodes for mixed precision"


@pytest.fixture
def model_with_if_outer_scope_reference():
    """Create a minimal model where If subgraphs reference outer scope variables.

    This tests that subgraph value_info for outer scope variables is preserved during type clearing.
    """
    # Main graph inputs/outputs
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4])
    condition = helper.make_tensor_value_info("condition", TensorProto.BOOL, [])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])

    # Create "then" branch: Identity on X from outer scope
    then_y = helper.make_tensor_value_info("then_y", TensorProto.FLOAT, [2, 4])
    then_identity = helper.make_node("Identity", ["X"], ["then_y"], name="then_identity")
    then_graph = helper.make_graph([then_identity], "then_branch", [], [then_y])
    # Add X to value_info - this is what needs to be preserved
    then_graph.value_info.extend([helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4])])

    # Create "else" branch: Identity on X from outer scope
    else_y = helper.make_tensor_value_info("else_y", TensorProto.FLOAT, [2, 4])
    else_identity = helper.make_node("Identity", ["X"], ["else_y"], name="else_identity")
    else_graph = helper.make_graph([else_identity], "else_branch", [], [else_y])
    else_graph.value_info.extend([helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4])])

    # Create If node and main graph
    if_node = helper.make_node(
        "If", ["condition"], ["Y"], name="if_node", then_branch=then_graph, else_branch=else_graph
    )
    main_graph = helper.make_graph([if_node], "model_with_outer_scope", [x, condition], [y])

    model = helper.make_model(main_graph, producer_name="model_with_outer_scope")
    model.opset_import[0].version = 20
    onnx.checker.check_model(model)

    model = onnx_utils.infer_shapes(model)
    value_info_map, initializer_map, node_to_init_map = utils.setup_mappings(model)
    return model, value_info_map, initializer_map, node_to_init_map


@pytest.mark.parametrize("low_precision_type", ["fp16", "bf16"])
def test_if_subgraph_outer_scope_type_preservation(
    model_with_if_outer_scope_reference, low_precision_type
):
    """Test that outer scope variable types are preserved in If subgraphs during conversion.

    Without preserving X's value_info in subgraphs, shape inference fails with
    "Element type of input 0 unknown".
    """
    model, value_info_map, initializer_map, node_to_init_map = model_with_if_outer_scope_reference

    converter = PrecisionConverter(
        model,
        value_info_map,
        initializer_map,
        node_to_init_map,
        keep_io_types=True,
        low_precision_type=low_precision_type,
    )

    converted_model = converter.convert(high_precision_nodes=["if_node"], low_precision_nodes=[])
    onnx.checker.check_model(converted_model)

    # Verify X's value_info is preserved in both subgraphs
    if_node = next(n for n in converted_model.graph.node if n.op_type == "If")
    then_branch = next(attr.g for attr in if_node.attribute if attr.name == "then_branch")
    else_branch = next(attr.g for attr in if_node.attribute if attr.name == "else_branch")

    then_x_info = [vi for vi in then_branch.value_info if vi.name == "X"]
    else_x_info = [vi for vi in else_branch.value_info if vi.name == "X"]

    assert len(then_x_info) > 0, "X value_info should be preserved in then branch"
    assert len(else_x_info) > 0, "X value_info should be preserved in else branch"
    assert then_x_info[0].type.tensor_type.elem_type != onnx.TensorProto.UNDEFINED
    assert else_x_info[0].type.tensor_type.elem_type != onnx.TensorProto.UNDEFINED
