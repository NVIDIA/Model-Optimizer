# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import pytest
from onnx import TensorProto, helper, numpy_helper

from modelopt.onnx.export import INT4QuantExporter, MXFP8QuantExporter, NVFP4QuantExporter
from modelopt.onnx.export.nvfp4_exporter import _cast_fp4, _cast_fp8


def create_test_model_with_int4_dq_reshape_transpose_matmul(constant_scale: bool = False):
    """Create a test ONNX model with DequantizeLinear -> Reshape -> Transpose -> MatMul pattern for INT4.
    If constant_scale is True, the scale is a Constant node instead of an initializer."""
    # Create weight tensor (4x8 matrix scaled by 32 blocks)
    weight_data = np.random.randint(-8, 8, size=(32, 8), dtype=np.int8)
    weight_tensor = numpy_helper.from_array(weight_data, "weight")

    # Create scale tensor for block quantization (block_size = 32)
    scale_data = np.random.uniform(0.1, 1.0, size=(32, 1)).astype(np.float32)
    scale_tensor = numpy_helper.from_array(scale_data, "scale")

    # Create reshape shape tensor
    reshape_shape = np.array([16, 16], dtype=np.int64)

    # Create input tensor for MatMul
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 2])

    if constant_scale:
        scale_constant = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["Constant_output_0"],
            value=numpy_helper.from_array(scale_data),
            name="scale_constant",
        )

    # Create nodes
    dq_inputs = ["weight", "Constant_output_0"] if constant_scale else ["weight", "scale"]
    dq_node = helper.make_node(
        "DequantizeLinear", inputs=dq_inputs, outputs=["dq_output"], name="weight_dq"
    )

    reshape_constant = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["reshape_shape_Constant"],
        value=numpy_helper.from_array(reshape_shape),
        name="reshape_constant",
    )

    reshape_node = helper.make_node(
        "Reshape",
        inputs=["dq_output", "reshape_shape_Constant"],
        outputs=["reshape_output"],
        name="weight_reshape",
    )

    cast_node = helper.make_node(
        "Cast",
        inputs=["reshape_output"],
        outputs=["cast_output"],
        to=TensorProto.FLOAT,
        name="weight_cast",
    )

    transpose_node = helper.make_node(
        "Transpose",
        inputs=["cast_output"],
        outputs=["transpose_output"],
        perm=[1, 0],
        name="weight_transpose",
    )

    matmul_node = helper.make_node(
        "MatMul", inputs=["input", "transpose_output"], outputs=["output"], name="matmul"
    )

    # Create value info for intermediate tensors
    reshape_output_info = helper.make_tensor_value_info(
        "reshape_output", TensorProto.FLOAT, [16, 16]
    )

    # Create graph
    nodes = [dq_node, reshape_constant, reshape_node, cast_node, transpose_node, matmul_node]
    if constant_scale:
        nodes.append(scale_constant)
    graph = helper.make_graph(
        nodes=nodes,
        name="test_graph",
        inputs=[input_tensor],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 16])],
        initializer=[weight_tensor, scale_tensor],
        value_info=[reshape_output_info],
    )

    model = helper.make_model(graph)
    return model


def create_test_model_with_cast_nodes():
    """Create a test model with various Cast nodes to test float32->float16 conversion."""
    # Create a simple model with Cast nodes
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [4])

    # Cast to float32 (should be converted to float16)
    cast_fp32 = helper.make_node(
        "Cast",
        inputs=["input"],
        outputs=["cast_fp32_output"],
        to=TensorProto.FLOAT,
        name="regular_cast",
    )

    # Cast in normalization (should be preserved)
    cast_norm = helper.make_node(
        "Cast",
        inputs=["input"],
        outputs=["cast_norm_output"],
        to=TensorProto.FLOAT,
        name="layer_norm/Cast",
    )

    # Output cast (should be preserved)
    cast_output = helper.make_node(
        "Cast", inputs=["input"], outputs=["final_output"], to=TensorProto.FLOAT, name="/Cast"
    )

    graph = helper.make_graph(
        nodes=[cast_fp32, cast_norm, cast_output],
        name="test_graph",
        inputs=[input_tensor],
        outputs=[helper.make_tensor_value_info("final_output", TensorProto.FLOAT, [4])],
        initializer=[],
    )

    model = helper.make_model(graph)
    return model


def create_test_model_with_proj_nodes():
    """Create a test model with projection nodes to test bias and scale casting."""
    # Create bias tensor
    bias_data = np.random.uniform(-1.0, 1.0, size=(16,)).astype(np.float32)
    bias_tensor = numpy_helper.from_array(bias_data, "proj_bias")

    # Create scale tensor for quantization
    scale_data = np.random.uniform(0.1, 1.0, size=(1,)).astype(np.float32)
    scale_tensor = numpy_helper.from_array(scale_data, "quant_scale")

    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [4, 16])

    # Add node (projection bias)
    add_node = helper.make_node(
        "Add", inputs=["input", "proj_bias"], outputs=["add_output"], name="o_proj/Add"
    )

    # Mul node (quantization scale)
    mul_node = helper.make_node(
        "Mul",
        inputs=["add_output", "quant_scale"],
        outputs=["output"],
        name="o_proj/input_quantizer/Mul",
    )

    graph = helper.make_graph(
        nodes=[add_node, mul_node],
        name="test_graph",
        inputs=[input_tensor],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [4, 16])],
        initializer=[bias_tensor, scale_tensor],
    )

    model = helper.make_model(graph)
    return model


def create_test_model_with_mxfp8_dq():
    """Create a test ONNX model with TRT_MXFP8DequantizeLinear nodes for testing MXFP8."""
    # Create weight tensor
    weight_data = np.random.uniform(-1.0, 1.0, size=(64, 32)).astype(np.float32)
    weight_tensor = numpy_helper.from_array(weight_data, "linear.weight")

    # Create scale tensor (constant node) - MXFP8 uses block_size=32
    scale_data = np.random.uniform(0.1, 1.0, size=(2, 1)).astype(np.float32)

    # Create input tensor
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [4, 32])

    # Create scale constant node
    scale_constant = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["Constant_output_0"],
        value=numpy_helper.from_array(scale_data),
        name="scale_constant",
    )

    # Create TRT_MXFP8DequantizeLinear node
    dq_node = helper.make_node(
        "TRT_MXFP8DequantizeLinear",
        inputs=["linear.weight", "Constant_output_0"],
        outputs=["dq_output"],
        name="weight_dq",
        axis=-1,
        block_size=32,
        output_dtype=TensorProto.FLOAT,
    )

    # Create MatMul node
    matmul_node = helper.make_node(
        "MatMul", inputs=["input", "dq_output"], outputs=["output"], name="matmul"
    )

    # Create optional Gelu node to test Gelu approximation update
    gelu_node = helper.make_node(
        "Gelu", inputs=["output"], outputs=["gelu_output"], name="gelu", approximate="none"
    )

    graph = helper.make_graph(
        nodes=[scale_constant, dq_node, matmul_node, gelu_node],
        name="test_graph",
        inputs=[input_tensor],
        outputs=[helper.make_tensor_value_info("gelu_output", TensorProto.FLOAT, [4, 64])],
        initializer=[weight_tensor],
    )

    model = helper.make_model(graph)
    return model


def create_test_model_with_nvfp4_qdq(with_transpose: bool = False):
    """Create a test ONNX model with TRT_FP4QDQ nodes for testing NVFP4.

    Args:
        with_transpose: If True, adds a Transpose node between TRT_FP4QDQ and MatMul.
    """
    if with_transpose:
        # For transpose case, weight shape is (32, 64) to match transpose output
        weight_data = np.random.uniform(-1.0, 1.0, size=(32, 64)).astype(np.float32)
        fp4qdq_output_shape = [32, 64]
        transpose_output_shape = [64, 32]
    else:
        # For non-transpose case, weight shape is (64, 32) (FP16 for testing BFloat16 detection)
        weight_data = np.random.uniform(-1.0, 1.0, size=(64, 32)).astype(np.float16)
        fp4qdq_output_shape = [64, 32]
        transpose_output_shape = None

    weight_tensor = numpy_helper.from_array(weight_data, "linear.weight")

    # Create input tensor
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [4, 32])

    # Create TRT_FP4QDQ node with correct block_size=16 for NVFP4
    fp4qdq_node = helper.make_node(
        "TRT_FP4QDQ",
        inputs=["linear.weight"],
        outputs=["fp4qdq_output"],
        name="weight_fp4qdq",
        block_size=16,
    )

    nodes = [fp4qdq_node]
    value_info = []

    # Create value info for fp4qdq output
    fp4qdq_output_dtype = TensorProto.FLOAT16 if not with_transpose else TensorProto.FLOAT
    fp4qdq_output_info = helper.make_tensor_value_info(
        "fp4qdq_output", fp4qdq_output_dtype, fp4qdq_output_shape
    )
    value_info.append(fp4qdq_output_info)

    if with_transpose:
        # Create Transpose node
        transpose_node = helper.make_node(
            "Transpose",
            inputs=["fp4qdq_output"],
            outputs=["transpose_output"],
            name="transpose",
            perm=[1, 0],
        )
        nodes.append(transpose_node)

        # Create value info for transpose output
        transpose_output_info = helper.make_tensor_value_info(
            "transpose_output", TensorProto.FLOAT, transpose_output_shape
        )
        value_info.append(transpose_output_info)

        # MatMul uses transpose output
        matmul_input = "transpose_output"
    else:
        # MatMul uses fp4qdq output directly
        matmul_input = "fp4qdq_output"

    # Create MatMul node
    matmul_node = helper.make_node(
        "MatMul", inputs=["input", matmul_input], outputs=["output"], name="matmul"
    )
    nodes.append(matmul_node)

    graph = helper.make_graph(
        nodes=nodes,
        name="test_graph",
        inputs=[input_tensor],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [4, 64])],
        initializer=[weight_tensor],
        value_info=value_info,
    )

    model = helper.make_model(graph)
    return model


class TestQuantizeWeightsToInt4:
    """Test suite for quantize_weights_to_int4 function."""

    def test_basic_quantization_with_reshape_transpose(self):
        """Test basic INT4 quantization with Reshape and Transpose removal."""
        model = create_test_model_with_int4_dq_reshape_transpose_matmul()

        # Run quantization
        quantized_model = INT4QuantExporter.process_model(model)

        # Verify weight is converted to INT4
        weight_tensor = next(
            init for init in quantized_model.graph.initializer if init.name == "weight"
        )
        assert weight_tensor.data_type == TensorProto.INT4

        # Verify Reshape and Transpose nodes are removed
        node_types = [node.op_type for node in quantized_model.graph.node]
        assert "Reshape" not in node_types
        assert "Transpose" not in node_types

        # Verify MatMul input is connected directly to DequantizeLinear output
        matmul_node = next(node for node in quantized_model.graph.node if node.op_type == "MatMul")
        dq_node = next(
            node for node in quantized_model.graph.node if node.op_type == "DequantizeLinear"
        )
        assert matmul_node.input[1] == dq_node.output[0]

    def test_quantization_with_constant_scale(self):
        """Test quantization when scale comes from a Constant node."""
        model = create_test_model_with_int4_dq_reshape_transpose_matmul(constant_scale=True)

        # Run quantization
        quantized_model = INT4QuantExporter.process_model(model)

        # Verify Constant node is removed
        constant_nodes = [node for node in quantized_model.graph.node if node.op_type == "Constant"]
        assert len(constant_nodes) == 0

        # Verify new scale initializer is created
        scale_initializers = [
            init for init in quantized_model.graph.initializer if "scale" in init.name
        ]
        assert len(scale_initializers) > 0

        # Verify DequantizeLinear references the new scale
        dq_node = next(
            node for node in quantized_model.graph.node if node.op_type == "DequantizeLinear"
        )
        assert any("scale" in input_name for input_name in dq_node.input)

    def test_projection_bias_and_scale_casting(self):
        """Test that projection biases and quantization scales are cast to float16."""
        model = create_test_model_with_proj_nodes()

        # Run quantization
        quantized_model = INT4QuantExporter.process_model(model)

        # Verify bias tensor is cast to float16
        bias_tensor = next(
            init for init in quantized_model.graph.initializer if "proj_bias" in init.name
        )
        assert bias_tensor.data_type == TensorProto.FLOAT16

        # Verify quantization scale is cast to float16
        scale_tensor = next(
            init for init in quantized_model.graph.initializer if "quant_scale" in init.name
        )
        assert scale_tensor.data_type == TensorProto.FLOAT16


class TestCastFunctions:
    """Test suite for _cast_fp8 and _cast_fp4 functions."""

    @pytest.mark.parametrize(
        ("input_array", "expected_array"),
        [
            (
                np.array([1.0, 0.5, 2.0], dtype=np.float32),
                np.array([56, 48, 64], dtype=(np.uint8, [("e4m3fn", "u1")])),
            ),
            (
                np.array([-1.0, -0.5, -2.0], dtype=np.float32),
                np.array([184, 176, 192], dtype=(np.uint8, [("e4m3fn", "u1")])),
            ),
            (
                np.array([0.0, -0.0], dtype=np.float32),
                np.array([0, 128], dtype=(np.uint8, [("e4m3fn", "u1")])),
            ),
            (
                np.array([1e10, -1e10, 1e-10, -1e-10], dtype=np.float32),
                np.array([126, 254, 0, 128], dtype=(np.uint8, [("e4m3fn", "u1")])),
            ),
        ],
    )
    def test_cast_fp8(self, input_array, expected_array):
        """Test FP8 casting functionality."""
        result = _cast_fp8(input_array)
        assert result.dtype == np.dtype((np.uint8, [("e4m3fn", "u1")]))
        assert result.shape == expected_array.shape
        assert np.all(result == expected_array)

    @pytest.mark.parametrize(
        ("input_array", "expected_array"),
        [
            # Basic positive values
            (
                np.array([[0.0, 0.5], [1.0, 1.5]], dtype=np.float32),
                np.array([[16, 50]], dtype=np.uint8),
            ),
            # Basic negative values
            (
                np.array([[-0.5, -1.0], [-1.5, 1.75]], dtype=np.float32),
                np.array([[169, 75]], dtype=np.uint8),
            ),
            # Boundary values with rounding
            (
                np.array([[0.0, 0.75], [1.75, 3.5]], dtype=np.float32),
                np.array([[32, 100]], dtype=np.uint8),
            ),
            # Large values (saturate to max)
            (
                np.array([[10.0], [-10.0]], dtype=np.float32),
                np.array([[247]], dtype=np.uint8),
            ),
            # Very small values (map to zero)
            (
                np.array([[0.1], [-0.1]], dtype=np.float32),
                np.array([[128]], dtype=np.uint8),
            ),
            # Zero and negative zero
            (
                np.array([[0.0], [-0.0]], dtype=np.float32),
                np.array([[0]], dtype=np.uint8),
            ),
        ],
    )
    def test_cast_fp4(self, input_array, expected_array):
        """Test FP4 casting functionality."""
        result = _cast_fp4(input_array)
        assert result.dtype == np.dtype(np.uint8)
        assert result.shape == expected_array.shape
        assert np.all(result == expected_array)


class TestMXFP8QuantExporter:
    """Test suite for MXFP8QuantExporter."""

    def test_basic_mxfp8_quantization(self):
        """Test basic MXFP8 quantization with TRT_MXFP8DequantizeLinear nodes."""
        model = create_test_model_with_mxfp8_dq()

        # Run MXFP8 quantization
        quantized_model = MXFP8QuantExporter.process_model(model)

        # Verify weight is converted to FP8
        weight_tensor = next(
            init for init in quantized_model.graph.initializer if init.name == "linear.weight"
        )
        assert weight_tensor.data_type == TensorProto.FLOAT8E4M3FN

        # Verify scale tensor is created and is uint8
        scale_tensors = [init for init in quantized_model.graph.initializer if "scale" in init.name]
        assert len(scale_tensors) > 0
        scale_tensor = scale_tensors[0]
        assert scale_tensor.data_type == TensorProto.UINT8

        # Verify Constant node is removed
        constant_nodes = [node for node in quantized_model.graph.node if node.op_type == "Constant"]
        assert len(constant_nodes) == 0

        # Verify DQ node references the new scale
        dq_node = next(
            node
            for node in quantized_model.graph.node
            if node.op_type == "TRT_MXFP8DequantizeLinear"
        )
        assert any("scale" in input_name for input_name in dq_node.input)

    def test_mxfp8_output_dtype_update(self):
        """Test that output_dtype attribute is updated to FP16."""
        model = create_test_model_with_mxfp8_dq()

        # Run MXFP8 quantization
        quantized_model = MXFP8QuantExporter.process_model(model)

        # Verify output_dtype is set to FP16
        dq_node = next(
            node
            for node in quantized_model.graph.node
            if node.op_type == "TRT_MXFP8DequantizeLinear"
        )
        output_dtype_attr = next(attr for attr in dq_node.attribute if attr.name == "output_dtype")
        assert output_dtype_attr.i == TensorProto.FLOAT16

    def test_mxfp8_gelu_approximation_update(self):
        """Test that Gelu nodes are updated to use tanh approximation."""
        model = create_test_model_with_mxfp8_dq()

        # Run MXFP8 quantization
        quantized_model = MXFP8QuantExporter.process_model(model)

        # Verify Gelu approximation is set to tanh
        gelu_node = next(node for node in quantized_model.graph.node if node.op_type == "Gelu")
        approximate_attr = next(attr for attr in gelu_node.attribute if attr.name == "approximate")
        assert approximate_attr.s == b"tanh"

    def test_mxfp8_with_missing_attributes(self):
        """Test MXFP8 quantization with missing axis and block_size attributes."""
        # Create a model without axis and block_size attributes
        weight_data = np.random.uniform(-1.0, 1.0, size=(64, 32)).astype(np.float32)
        weight_tensor = numpy_helper.from_array(weight_data, "linear.weight")

        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [4, 32])

        scale_data = np.random.uniform(0.1, 1.0, size=(2, 1)).astype(np.float32)
        scale_constant = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["Constant_output_0"],
            value=numpy_helper.from_array(scale_data),
            name="scale_constant",
        )

        # Create TRT_MXFP8DequantizeLinear node without axis and block_size
        dq_node = helper.make_node(
            "TRT_MXFP8DequantizeLinear",
            inputs=["linear.weight", "Constant_output_0"],
            outputs=["dq_output"],
            name="weight_dq",
            output_dtype=TensorProto.FLOAT,
        )

        matmul_node = helper.make_node(
            "MatMul", inputs=["input", "dq_output"], outputs=["output"], name="matmul"
        )

        graph = helper.make_graph(
            nodes=[scale_constant, dq_node, matmul_node],
            name="test_graph",
            inputs=[input_tensor],
            outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [4, 64])],
            initializer=[weight_tensor],
        )

        model = helper.make_model(graph)

        # Run MXFP8 quantization (should use default values)
        quantized_model = MXFP8QuantExporter.process_model(model)

        # Verify the model is still processed correctly
        weight_tensor = next(
            init for init in quantized_model.graph.initializer if init.name == "linear.weight"
        )
        assert weight_tensor.data_type == TensorProto.FLOAT8E4M3FN


class TestFP4QDQTo2DQ:
    """Test suite for NVFP4QuantExporter."""

    @pytest.mark.parametrize("with_transpose", [False, True])
    def test_fp4qdq_conversion(self, with_transpose):
        """Test FP4QDQ to 2DQ conversion with and without Transpose node."""
        model = create_test_model_with_nvfp4_qdq(with_transpose=with_transpose)

        # Run FP4QDQ to 2DQ conversion
        converted_model = NVFP4QuantExporter.process_model(model)

        # Verify TRT_FP4QDQ node is removed
        fp4qdq_nodes = [node for node in converted_model.graph.node if node.op_type == "TRT_FP4QDQ"]
        assert len(fp4qdq_nodes) == 0

        # Verify two DequantizeLinear nodes are created
        dq_nodes = [
            node for node in converted_model.graph.node if node.op_type == "DequantizeLinear"
        ]
        assert len(dq_nodes) == 2

        # Verify new initializers are created
        initializer_names = {init.name for init in converted_model.graph.initializer}
        assert "linear.weight_f4" in initializer_names
        assert "linear.weight_f8_scale" in initializer_names
        assert "linear.weight_f8_scale_f32_scale" in initializer_names

        # Verify original weight initializer is removed
        assert "linear.weight" not in initializer_names

        # Verify FP4 weight tensor has correct data type
        fp4_weight = next(
            init for init in converted_model.graph.initializer if init.name == "linear.weight_f4"
        )
        assert fp4_weight.data_type == TensorProto.FLOAT4E2M1

        # Verify FP8 scale tensor has correct data type
        fp8_scale = next(
            init
            for init in converted_model.graph.initializer
            if init.name == "linear.weight_f8_scale"
        )
        assert fp8_scale.data_type == TensorProto.FLOAT8E4M3FN

        # Additional verification for transpose case
        if with_transpose:
            # Verify Cast nodes are added for input type conversion
            cast_nodes = [node for node in converted_model.graph.node if node.op_type == "Cast"]
            assert len(cast_nodes) >= 1  # At least one cast node should be added


def create_test_model_with_int4_dq_matmul():
    """Create a simple test model with INT4 DequantizeLinear -> MatMul pattern.

    Returns the model and original weight/scale arrays for verification.
    """
    from modelopt.onnx.quantization.quant_utils import pack_float32_to_4bit_cpp_based

    # Create INT4 quantized weight tensor (K=32, N=16)
    # Using int8 storage for INT4 values in range [-8, 7]
    weight_data = np.random.randint(-8, 8, size=(32, 16), dtype=np.int8)

    # Pack INT4 data (2 values per byte) for ORT compatibility
    packed_weight = pack_float32_to_4bit_cpp_based(weight_data, signed=True).astype(np.int8)
    weight_tensor = helper.make_tensor(
        "weight",
        TensorProto.INT4,
        dims=weight_data.shape,
        vals=packed_weight.tobytes(),
        raw=True,
    )

    # Create scale tensor for block quantization (block_size=32, so 1 scale per column)
    scale_data = np.random.uniform(0.1, 1.0, size=(1, 16)).astype(np.float16)
    scale_tensor = numpy_helper.from_array(scale_data, "scale")

    # Create input tensor for MatMul (batch=4, K=32)
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT16, [4, 32])

    # Create DequantizeLinear node with INT4 blocked quantization
    dq_node = helper.make_node(
        "DequantizeLinear",
        inputs=["weight", "scale"],
        outputs=["dq_output"],
        name="weight_dq",
        axis=0,
        block_size=32,
    )

    # Create MatMul node: input (4, 32) @ weight (32, 16) -> output (4, 16)
    matmul_node = helper.make_node(
        "MatMul",
        inputs=["input", "dq_output"],
        outputs=["output"],
        name="matmul",
    )

    graph = helper.make_graph(
        nodes=[dq_node, matmul_node],
        name="test_graph",
        inputs=[input_tensor],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT16, [4, 16])],
        initializer=[weight_tensor, scale_tensor],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    model.ir_version = 10  # ORT only supports IR version up to 10
    return model, weight_data, scale_data


class TestColumnMajorTransformation:
    """Test suite for column-major storage transformation functions."""

    def test_column_major_transformation_graph_structure(self):
        """Test that column-major transformation produces correct graph structure.

        Verifies: DQ(W) -> MatMul becomes DQ(W^T) -> Transpose -> MatMul
        """
        import onnx_graphsurgeon as gs

        from modelopt.onnx.quantization.qdq_utils import (
            apply_column_major_transformation,
            insert_transpose_nodes_for_column_major,
        )

        model, original_weight, original_scale = create_test_model_with_int4_dq_matmul()

        # Get weights and scales as dicts (simulating what int4.py does)
        weights_dict = {"weight": original_weight.copy()}
        scales_dict = {"scale": original_scale.copy()}

        # Apply column-major transformation (transposes in-place)
        apply_column_major_transformation(weights_dict, scales_dict)

        # Verify weights and scales are transposed
        assert weights_dict["weight"].shape == (16, 32), (
            f"Expected transposed weight shape (16, 32), got {weights_dict['weight'].shape}"
        )
        assert scales_dict["scale"].shape == (16, 1), (
            f"Expected transposed scale shape (16, 1), got {scales_dict['scale'].shape}"
        )

        # Verify the transposed values match
        assert np.array_equal(weights_dict["weight"], original_weight.T)
        assert np.array_equal(scales_dict["scale"], original_scale.T)

        # Now test insert_transpose_nodes_for_column_major on a graph
        # Create a fresh model and apply the full transformation
        model2, _, _ = create_test_model_with_int4_dq_matmul()
        graph2 = gs.import_onnx(model2)

        # Add transpose nodes for column-major
        insert_transpose_nodes_for_column_major(graph2)

        # Export and verify structure
        transformed_model = gs.export_onnx(graph2)

        # Check that Transpose node was added
        node_types = [node.op_type for node in transformed_model.graph.node]
        assert "Transpose" in node_types, "Transpose node should be added after DQ"
        assert "DequantizeLinear" in node_types
        assert "MatMul" in node_types

        # Verify the order: DQ -> Transpose -> MatMul
        dq_node = next(n for n in transformed_model.graph.node if n.op_type == "DequantizeLinear")
        transpose_node = next(n for n in transformed_model.graph.node if n.op_type == "Transpose")
        matmul_node = next(n for n in transformed_model.graph.node if n.op_type == "MatMul")

        # DQ output should be Transpose input
        assert dq_node.output[0] == transpose_node.input[0], "DQ output should feed into Transpose"
        # Transpose output should be MatMul weight input
        assert transpose_node.output[0] == matmul_node.input[1], (
            "Transpose output should feed into MatMul"
        )

        # Verify transpose permutation is [1, 0]
        perm_attr = next((a for a in transpose_node.attribute if a.name == "perm"), None)
        assert perm_attr is not None, "Transpose should have perm attribute"
        assert list(perm_attr.ints) == [1, 0], "Transpose perm should be [1, 0]"

    def test_column_major_transformation_output_equivalence(self):
        """Test that column-major transformed graph produces equivalent output.

        Creates two graphs:
        1. Original: DQ(W) -> MatMul
        2. Transformed: DQ(W^T) -> Transpose -> MatMul

        Verifies both produce the same output for the same input.
        """
        import onnxruntime as ort

        from modelopt.onnx.quantization.quant_utils import pack_float32_to_4bit_cpp_based

        # Create original model
        original_model, original_weight, original_scale = create_test_model_with_int4_dq_matmul()

        # Create input data
        input_data = np.random.randn(4, 32).astype(np.float16)

        # Run original model
        original_session = ort.InferenceSession(original_model.SerializeToString())
        original_output = original_session.run(None, {"input": input_data})[0]

        # Create transformed model
        # We need to manually create a model with transposed weights
        transposed_weight = original_weight.T.copy()  # Shape: (16, 32)
        transposed_scale = original_scale.T.copy()  # Shape: (16, 1)

        # Pack INT4 data (2 values per byte) for ORT compatibility
        packed_transposed_weight = pack_float32_to_4bit_cpp_based(
            transposed_weight, signed=True
        ).astype(np.int8)
        weight_tensor = helper.make_tensor(
            "weight",
            TensorProto.INT4,
            dims=transposed_weight.shape,
            vals=packed_transposed_weight.tobytes(),
            raw=True,
        )
        scale_tensor = numpy_helper.from_array(transposed_scale, "scale")

        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT16, [4, 32])

        # DQ node with axis=1 for column-major (transposed weight)
        dq_node = helper.make_node(
            "DequantizeLinear",
            inputs=["weight", "scale"],
            outputs=["dq_output"],
            name="weight_dq",
            axis=1,
            block_size=32,
        )

        # Transpose node to convert back: (16, 32) -> (32, 16)
        transpose_node = helper.make_node(
            "Transpose",
            inputs=["dq_output"],
            outputs=["transpose_output"],
            name="transpose_back",
            perm=[1, 0],
        )

        # MatMul: input (4, 32) @ transposed_back (32, 16) -> output (4, 16)
        matmul_node = helper.make_node(
            "MatMul",
            inputs=["input", "transpose_output"],
            outputs=["output"],
            name="matmul",
        )

        transformed_graph = helper.make_graph(
            nodes=[dq_node, transpose_node, matmul_node],
            name="test_graph",
            inputs=[input_tensor],
            outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT16, [4, 16])],
            initializer=[weight_tensor, scale_tensor],
        )

        transformed_model = helper.make_model(
            transformed_graph, opset_imports=[helper.make_opsetid("", 21)]
        )
        transformed_model.ir_version = 10  # ORT only supports IR version up to 10

        # Run transformed model
        transformed_session = ort.InferenceSession(transformed_model.SerializeToString())
        transformed_output = transformed_session.run(None, {"input": input_data})[0]

        # Print output values for visibility
        print(f"Original model output shape: {original_output.shape}")
        print(f"Transformed model output shape: {transformed_output.shape}")
        print(f"Original output (first 5): {original_output.flatten()[:5]}")
        print(f"Transformed output (first 5): {transformed_output.flatten()[:5]}")

        # Verify outputs are equivalent (allowing small numerical tolerance)
        assert original_output.shape == transformed_output.shape, (
            f"Output shapes should match: {original_output.shape} vs {transformed_output.shape}"
        )
        np.testing.assert_allclose(
            original_output,
            transformed_output,
            rtol=1e-3,
            atol=1e-3,
            err_msg="Column-major transformed model should produce equivalent output",
        )

    def test_column_major_gemm_trans_b_flip(self):
        """Test that Gemm with transB=1 gets flipped to transB=0 for column-major.

        When weights are already transposed (column-major), Gemm nodes with transB=1
        should have transB flipped to 0 instead of inserting a Transpose node.
        Also verifies output equivalence between original and transformed models.
        """
        import onnx_graphsurgeon as gs
        import onnxruntime as ort

        from modelopt.onnx.quantization.qdq_utils import (
            apply_column_major_transformation,
            insert_transpose_nodes_for_column_major,
        )
        from modelopt.onnx.quantization.quant_utils import pack_float32_to_4bit_cpp_based

        # Original model: weight (N=16, K=32) with Gemm transB=1
        # Gemm computes: A @ B^T = (4, 32) @ (16, 32)^T = (4, 16)
        weight_data = np.random.randint(-8, 8, size=(16, 32), dtype=np.int8)  # Shape (N, K)
        scale_data = np.random.uniform(0.1, 1.0, size=(16, 1)).astype(np.float16)

        # Pack INT4 data for original model
        packed_weight = pack_float32_to_4bit_cpp_based(weight_data, signed=True).astype(np.int8)
        weight_tensor = helper.make_tensor(
            "weight",
            TensorProto.INT4,
            dims=weight_data.shape,
            vals=packed_weight.tobytes(),
            raw=True,
        )
        scale_tensor = numpy_helper.from_array(scale_data, "scale")

        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT16, [4, 32])

        dq_node = helper.make_node(
            "DequantizeLinear",
            inputs=["weight", "scale"],
            outputs=["dq_output"],
            name="weight_dq",
            axis=1,
            block_size=32,
        )

        gemm_node = helper.make_node(
            "Gemm",
            inputs=["input", "dq_output"],
            outputs=["output"],
            name="gemm",
            transB=1,
        )

        graph = helper.make_graph(
            nodes=[dq_node, gemm_node],
            name="test_graph",
            inputs=[input_tensor],
            outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT16, [4, 16])],
            initializer=[weight_tensor, scale_tensor],
        )

        original_model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
        original_model.ir_version = 10

        # Run original model
        input_data = np.random.randn(4, 32).astype(np.float16)
        original_session = ort.InferenceSession(original_model.SerializeToString())
        original_output = original_session.run(None, {"input": input_data})[0]

        # Apply column-major transformation using the actual functions
        weights_dict = {"weight": weight_data.copy()}
        scales_dict = {"scale": scale_data.copy()}
        apply_column_major_transformation(weights_dict, scales_dict)

        # Build transformed model with transposed weights/scales
        transposed_weight = weights_dict["weight"]  # Now (32, 16)
        transposed_scale = scales_dict["scale"]  # Now (1, 16)

        packed_transposed = pack_float32_to_4bit_cpp_based(transposed_weight, signed=True).astype(
            np.int8
        )
        transposed_weight_tensor = helper.make_tensor(
            "weight",
            TensorProto.INT4,
            dims=transposed_weight.shape,
            vals=packed_transposed.tobytes(),
            raw=True,
        )
        transposed_scale_tensor = numpy_helper.from_array(transposed_scale, "scale")

        # Build model with transposed weights but keep transB=1 initially
        dq_node_col = helper.make_node(
            "DequantizeLinear",
            inputs=["weight", "scale"],
            outputs=["dq_output"],
            name="weight_dq",
            axis=0,
            block_size=32,
        )

        gemm_node_col = helper.make_node(
            "Gemm",
            inputs=["input", "dq_output"],
            outputs=["output"],
            name="gemm",
            transB=1,  # Still transB=1, will be flipped by insert_transpose_nodes_for_column_major
        )

        col_graph = helper.make_graph(
            nodes=[dq_node_col, gemm_node_col],
            name="test_graph",
            inputs=[input_tensor],
            outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT16, [4, 16])],
            initializer=[transposed_weight_tensor, transposed_scale_tensor],
        )

        col_model = helper.make_model(col_graph, opset_imports=[helper.make_opsetid("", 21)])
        col_model.ir_version = 10

        # Apply insert_transpose_nodes_for_column_major to flip transB
        gs_graph = gs.import_onnx(col_model)
        insert_transpose_nodes_for_column_major(gs_graph)
        transformed_model = gs.export_onnx(gs_graph)
        transformed_model.ir_version = 10  # ORT only supports IR version up to 10

        # Verify transB was flipped to 0
        gemm_nodes = [n for n in transformed_model.graph.node if n.op_type == "Gemm"]
        assert len(gemm_nodes) == 1
        trans_b_attr = next((a for a in gemm_nodes[0].attribute if a.name == "transB"), None)
        trans_b_value = trans_b_attr.i if trans_b_attr else 0
        assert trans_b_value == 0, f"transB should be 0, got {trans_b_value}"

        # Verify no Transpose node was added
        transpose_nodes = [n for n in transformed_model.graph.node if n.op_type == "Transpose"]
        assert len(transpose_nodes) == 0, (
            f"No Transpose should be added, found {len(transpose_nodes)}"
        )

        # Run transformed model and verify output equivalence
        transformed_session = ort.InferenceSession(transformed_model.SerializeToString())
        transformed_output = transformed_session.run(None, {"input": input_data})[0]

        print(f"Original model output shape: {original_output.shape}")
        print(f"Transformed model output shape: {transformed_output.shape}")
        print(f"Original output (first 5): {original_output.flatten()[:5]}")
        print(f"Transformed output (first 5): {transformed_output.flatten()[:5]}")

        np.testing.assert_allclose(
            original_output,
            transformed_output,
            rtol=1e-3,
            atol=1e-3,
            err_msg="Gemm transB flip should produce equivalent output",
        )

        print(f"transB flipped: 1 -> {trans_b_value}")
        print(f"Transpose nodes: {len(transpose_nodes)}")
