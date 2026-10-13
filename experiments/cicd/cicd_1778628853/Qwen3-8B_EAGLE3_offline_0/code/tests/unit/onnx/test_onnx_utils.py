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

import os

import numpy as np
import onnx
import pytest
from _test_utils.torch.vision_models import get_tiny_resnet_and_input
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_opsetid,
    make_tensor,
    make_tensor_value_info,
)

from modelopt.onnx.trt_utils import load_onnx_model
from modelopt.onnx.utils import (
    get_input_names_from_bytes,
    get_output_names_from_bytes,
    randomize_weights_onnx_bytes,
    remove_node_training_mode,
    remove_weights_data,
    save_onnx_bytes_to_dir,
    validate_onnx,
)
from modelopt.torch._deploy.utils import OnnxBytes, get_onnx_bytes_and_metadata


@pytest.mark.parametrize(
    "onnx_bytes",
    [None, b"", b"invalid onnx"],
)
def test_validate_onnx(onnx_bytes):
    assert not validate_onnx(onnx_bytes)


def test_save_onnx(tmp_path):
    save_onnx_bytes_to_dir(b"test_onnx_bytes", tmp_path, "test")
    assert os.path.exists(os.path.join(tmp_path, "test.onnx"))


def make_onnx_model_for_matmul_op():
    input_left = np.array([1, 2])
    input_right = np.array([1, 3])
    output_shape = np.matmul(input_left, input_right).shape
    node = make_node("MatMul", ["X", "Y"], ["Z"], name="matmul")
    graph = make_graph(
        [node],
        "test_graph",
        [
            make_tensor_value_info("X", onnx.TensorProto.FLOAT, input_left.shape),
            make_tensor_value_info("Y", onnx.TensorProto.FLOAT, input_right.shape),
        ],
        [make_tensor_value_info("Z", onnx.TensorProto.FLOAT, output_shape)],
    )
    model = make_model(graph, producer_name="Omniengine Tester")
    return model.SerializeToString()


def test_input_names():
    model_bytes = make_onnx_model_for_matmul_op()
    input_names = get_input_names_from_bytes(model_bytes)
    assert input_names == ["X", "Y"]


def test_output_names():
    model_bytes = make_onnx_model_for_matmul_op()
    output_names = get_output_names_from_bytes(model_bytes)
    assert output_names == ["Z"]


def _get_avg_var_of_weights(model):
    inits = model.graph.initializer
    avg_var_dict = {}

    for init in inits:
        if len(init.dims) > 1:
            dtype = onnx.helper.tensor_dtype_to_np_dtype(init.data_type)
            if dtype in ["float16", "float32", "float64"]:
                np_tensor = np.frombuffer(init.raw_data, dtype=dtype)
                avg_var_dict[init.name + "_avg"] = np.average(np_tensor)
                avg_var_dict[init.name + "_var"] = np.var(np_tensor)

    return avg_var_dict


def test_random_onnx_weights():
    model, args, kwargs = get_tiny_resnet_and_input()
    assert not kwargs

    onnx_bytes, _ = get_onnx_bytes_and_metadata(model, args)
    onnx_bytes_obj = OnnxBytes.from_bytes(onnx_bytes)
    model_bytes = onnx_bytes_obj.get_onnx_model_file_bytes()
    model = onnx.load_from_string(model_bytes)

    original_avg_var_dict = _get_avg_var_of_weights(model)
    original_model_size = len(model_bytes)

    onnx_model_wo_weights = remove_weights_data(model_bytes)
    # Removed model weights should be greater than 18 MB
    assert original_model_size - len(onnx_model_wo_weights) > 18e6

    # After assigning random weights, model size should be slightly greater than the the original
    # size due to some extra metadata
    onnx_model_randomized = randomize_weights_onnx_bytes(onnx_model_wo_weights)
    assert len(onnx_model_randomized) > original_model_size

    randomized_avg_var_dict = _get_avg_var_of_weights(onnx.load_from_string(onnx_model_randomized))
    for key, value in original_avg_var_dict.items():
        assert abs(value - randomized_avg_var_dict[key]) < 0.1


def test_reproducible_random_weights():
    model, args, kwargs = get_tiny_resnet_and_input()
    assert not kwargs

    onnx_bytes, _ = get_onnx_bytes_and_metadata(model, args)
    onnx_bytes_obj = OnnxBytes.from_bytes(onnx_bytes)
    model_bytes = onnx_bytes_obj.get_onnx_model_file_bytes()
    model = onnx.load_from_string(model_bytes)

    # Check if the randomization produces the same weights
    onnx_bytes_1 = randomize_weights_onnx_bytes(model_bytes)
    onnx_bytes_2 = randomize_weights_onnx_bytes(model_bytes)
    assert onnx_bytes_1 == onnx_bytes_2


def _make_bn_initializer(name: str, shape, value=1.0):
    """Helper to create an initializer tensor for BatchNorm."""
    data = np.full(shape, value, dtype=np.float32)
    return make_tensor(name, onnx.TensorProto.FLOAT, shape, data.flatten())


def _make_batchnorm_model(bn_node, extra_value_infos=None):
    """Helper to create an ONNX model with a BatchNormalization node.

    The created model has the following schematic structure:

        graph name: "test_graph"
          inputs:
            - input: FLOAT [1, 3, 224, 224]
          initializers:
            - scale: FLOAT [3]
            - bias:  FLOAT [3]
            - mean:  FLOAT [3]
            - var:   FLOAT [3]
          nodes:
            - BatchNormalization (name comes from `bn_node`), with:
                inputs  = ["input", "scale", "bias", "mean", "var"]
                outputs = as provided by `bn_node` (e.g., ["output"], or
                          ["output", "running_mean", "running_var", "saved_mean"])
          outputs:
            - output: FLOAT [1, 3, 224, 224]

    If `extra_value_infos` is provided (e.g., value_info for non-training outputs
    like "running_mean"/"running_var" and/or training-only outputs like
    "saved_mean"/"saved_inv_std"), they are attached to the graph's value_info.
    Some tests subsequently invoke utilities (e.g., remove_node_training_mode)
    that prune training-only outputs and their value_info entries, while keeping
    regular outputs such as "running_mean" and "running_var" intact.
    """
    initializers = [
        _make_bn_initializer("scale", [3], 1.0),
        _make_bn_initializer("bias", [3], 0.0),
        _make_bn_initializer("mean", [3], 0.0),
        _make_bn_initializer("var", [3], 1.0),
    ]

    graph_outputs = []
    for output_name, shape in [
        ("output", [1, 3, 224, 224]),
        ("running_mean", [3]),
        ("running_var", [3]),
    ]:
        if output_name in bn_node.output:
            graph_outputs.append(make_tensor_value_info(output_name, onnx.TensorProto.FLOAT, shape))

    graph_def = make_graph(
        [bn_node],
        "test_graph",
        [make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 3, 224, 224])],
        graph_outputs,
        initializer=initializers,
        value_info=extra_value_infos or [],
    )

    return make_model(graph_def, opset_imports=[make_opsetid("", 14)])


def test_remove_node_training_mode_attribute():
    """Test removal of training_mode attribute from BatchNormalization nodes."""
    bn_node = make_node(
        "BatchNormalization",
        inputs=["input", "scale", "bias", "mean", "var"],
        outputs=["output"],
        name="bn1",
        training_mode=1,  # This attribute should be removed
    )

    model = _make_batchnorm_model(bn_node)
    result_model = remove_node_training_mode(model, "BatchNormalization")

    bn_node_result = result_model.graph.node[0]
    assert bn_node_result.op_type == "BatchNormalization"

    # Check that training_mode attribute is not present
    attr_names = [attr.name for attr in bn_node_result.attribute]
    assert "training_mode" not in attr_names


def test_remove_node_extra_training_outputs():
    """Test removal of extra training outputs from BatchNormalization nodes."""
    bn_node = make_node(
        "BatchNormalization",
        inputs=["input", "scale", "bias", "mean", "var"],
        outputs=[
            "output",
            "running_mean",
            "running_var",
            "saved_mean",
            "saved_inv_std",
        ],
        name="bn1",
        training_mode=1,
    )

    # Extra training outputs are attached to the graph's value_info
    value_infos = [
        make_tensor_value_info("saved_mean", onnx.TensorProto.FLOAT, [3]),
        make_tensor_value_info("saved_inv_std", onnx.TensorProto.FLOAT, [3]),
    ]

    model = _make_batchnorm_model(bn_node, extra_value_infos=value_infos)
    result_model = remove_node_training_mode(model, "BatchNormalization")

    # Verify only the non-training outputs remain
    bn_node_result = result_model.graph.node[0]
    print(bn_node_result.output)
    assert len(bn_node_result.output) == 3
    assert bn_node_result.output[0] == "output"
    assert bn_node_result.output[1] == "running_mean"
    assert bn_node_result.output[2] == "running_var"

    # Verify value_info entries for removed outputs are cleaned up
    value_info_names = [vi.name for vi in result_model.graph.value_info]
    assert "saved_mean" not in value_info_names
    assert "saved_inv_std" not in value_info_names


def _make_matmul_relu_model(ir_version=12):
    # Define your model inputs and outputs
    input_names = ["input_0"]
    output_names = ["output_0"]
    input_shapes = [(1, 1024, 1024)]
    output_shapes = [(1, 1024, 16)]

    inputs = [
        make_tensor_value_info(input_name, onnx.TensorProto.FLOAT, input_shape)
        for input_name, input_shape in zip(input_names, input_shapes)
    ]
    outputs = [
        make_tensor_value_info(output_name, onnx.TensorProto.FLOAT, output_shape)
        for output_name, output_shape in zip(output_names, output_shapes)
    ]

    # Create the ONNX graph with the nodes
    nodes = [
        make_node(
            op_type="MatMul",
            inputs=["input_0", "weights_1"],
            outputs=["matmul1_matmul/MatMul:0"],
            name="matmul1_matmul/MatMul",
        ),
        make_node(
            op_type="Relu",
            inputs=["matmul1_matmul/MatMul:0"],
            outputs=["output_0"],
            name="relu1_relu/Relu",
        ),
    ]

    # Create the ONNX initializers
    initializers = [
        make_tensor(
            name="weights_1",
            data_type=onnx.TensorProto.FLOAT,
            dims=(1024, 16),
            vals=np.random.uniform(low=0.5, high=1.0, size=1024 * 16),
        ),
    ]

    # Create the ONNX graph with the nodes and initializers
    graph = make_graph(
        nodes, f"matmul_relu_ir_{ir_version}", inputs, outputs, initializer=initializers
    )

    # Create the ONNX model
    model = make_model(graph)
    model.opset_import[0].version = 13
    model.ir_version = ir_version

    # Check the ONNX model
    model_inferred = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model_inferred)

    return model_inferred


def test_ir_version_support(tmp_path):
    model = _make_matmul_relu_model(ir_version=12)
    model_path = os.path.join(tmp_path, "test_matmul_relu.onnx")
    onnx.save(model, model_path)
    model_reload, _, _, _, _ = load_onnx_model(model_path, intermediate_generated_files=[])
    assert model_reload.ir_version == 10, (
        f"The maximum supported IR version is 10, but version {model_reload.ir_version} was detected."
    )
