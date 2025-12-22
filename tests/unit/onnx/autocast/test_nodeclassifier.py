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
import pytest
from onnx import TensorProto, helper, numpy_helper
from onnxruntime import __version__ as ort_version
from packaging.version import Version

from modelopt.onnx.autocast.logging_config import configure_logging
from modelopt.onnx.autocast.nodeclassifier import (
    DepthOfReductionRule,
    DisabledNodeNameRegexRule,
    InitializerRangeRule,
    IORangeRule,
    NodeClassifier,
)
from modelopt.onnx.autocast.referencerunner import ReferenceRunner

configure_logging("DEBUG")


@pytest.fixture
def test_model():
    # Create a simple model with Add and Mul operations
    node1 = helper.make_node("Add", ["X", "Y"], ["add_out"], name="add_node")
    node2 = helper.make_node("Mul", ["add_out", "Z"], ["output"], name="mul_node")

    # Create graph inputs
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 2])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 2])

    # Create initializers
    y_init = numpy_helper.from_array(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), name="Y")
    z_init = numpy_helper.from_array(np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32), name="Z")

    graph = helper.make_graph(
        [node1, node2],
        "test_model",
        [x],
        [output],
        initializer=[y_init, z_init],
    )

    model = helper.make_model(graph)
    model.opset_import[0].version = 20
    model.ir_version = 10
    return model


def test_disabled_node_name_regex_rule():
    rule = DisabledNodeNameRegexRule([r".*add.*", r".*layer\.1.*"])
    node1 = helper.make_node("Add", ["X", "Y"], ["Z"], name="/test/layer.0/add_1")
    node2 = helper.make_node("Mul", ["A", "B"], ["C"], name="/test/layer.1/mul_2")
    node3 = helper.make_node("Sub", ["D", "E"], ["F"], name="/test/layer.2/sub_1")
    node4 = helper.make_node("Mul", ["G", "H"], ["I"], name="/test/layer.2/mul_2")

    assert rule.check(node1) is True
    assert rule.check(node2) is True
    assert rule.check(node3) is False
    assert rule.check(node4) is False


def test_initializer_range_rule():
    # Create test initializers
    init1 = numpy_helper.from_array(np.array([900], dtype=np.float32))
    init2 = numpy_helper.from_array(np.array([1100], dtype=np.float32))
    init3 = numpy_helper.from_array(np.array([-900], dtype=np.float32))
    init4 = numpy_helper.from_array(np.array([-1100], dtype=np.float32))

    node_to_init_map = {"node1": [init1], "node2": [init2], "node3": [init3], "node4": [init4]}

    rule = InitializerRangeRule(1000, node_to_init_map)

    node1 = helper.make_node("TestOp", [], [], name="node1")
    node2 = helper.make_node("TestOp", [], [], name="node2")
    node3 = helper.make_node("TestOp", [], [], name="node3")
    node4 = helper.make_node("TestOp", [], [], name="node4")

    assert rule.check(node1) is False
    assert rule.check(node2) is True
    assert rule.check(node3) is False
    assert rule.check(node4) is True


def test_output_range_rule():
    reference_outputs = {
        "out1": np.array([9000], dtype=np.float32),
        "out2": np.array([11000], dtype=np.float32),
        "out3": np.array([-9000], dtype=np.float32),
        "out4": np.array([-11000], dtype=np.float32),
    }

    rule = IORangeRule(10000, reference_outputs, node_to_init_map={})

    node1 = helper.make_node("TestOp", [], ["out1"], name="node1")
    node2 = helper.make_node("TestOp", [], ["out2"], name="node2")
    node3 = helper.make_node("TestOp", [], ["out3"], name="node3")
    node4 = helper.make_node("TestOp", [], ["out4"], name="node4")
    node5 = helper.make_node("TestOp", ["out1"], [], name="node5")
    node6 = helper.make_node("TestOp", ["out2"], [], name="node6")
    node7 = helper.make_node("TestOp", ["out3"], [], name="node7")
    node8 = helper.make_node("TestOp", ["out4"], [], name="node8")

    assert rule.check(node1) is False
    assert rule.check(node2) is True
    assert rule.check(node3) is False
    assert rule.check(node4) is True
    assert rule.check(node5) is False
    assert rule.check(node6) is True
    assert rule.check(node7) is False
    assert rule.check(node8) is True


def test_depth_of_reduction_rule():
    # Create axes initializer for reduce operations
    axes_init_large = numpy_helper.from_array(np.array([1], dtype=np.int64), name="axes_large")
    axes_init_small = numpy_helper.from_array(np.array([2], dtype=np.int64), name="axes_small")
    reduce_init = numpy_helper.from_array(np.ones([10, 50], dtype=np.float32), name="reduce_init")

    reference_data = {
        "matmul_output": np.ones([10, 30], dtype=np.float32),
        "small_matmul_output": np.ones([5, 8], dtype=np.float32),
        "conv_output": np.ones([1, 64, 62, 62], dtype=np.float32),
        "small_conv_output": np.ones([1, 16, 15, 15], dtype=np.float32),
        "reduce_output": np.ones([10, 1, 20], dtype=np.float32),
        "small_reduce_output": np.ones([10, 100, 1], dtype=np.float32),
        "matmul_input_a": np.ones([10, 50], dtype=np.float32),
        "matmul_input_b": np.ones([50, 30], dtype=np.float32),
        "small_matmul_a": np.ones([5, 10], dtype=np.float32),
        "small_matmul_b": np.ones([10, 8], dtype=np.float32),
        "conv_input": np.ones([1, 32, 64, 64], dtype=np.float32),
        "conv_weight": np.ones([64, 32, 3, 3], dtype=np.float32),
        "small_conv_input": np.ones([1, 8, 16, 16], dtype=np.float32),
        "reduce_input": np.ones([10, 100, 20], dtype=np.float32),
    }

    node_to_init_map = {
        "matmul_node": [],
        "small_matmul_node": [],
        "conv_node": [],
        "small_conv_node": [],
        "reduce_node": [axes_init_large],
        "small_reduce_node": [axes_init_small],
        "reduce_init_node": [reduce_init, axes_init_large],
    }
    initializer_map = {
        "reduce_init": reduce_init,
        "axes_large": axes_init_large,
        "axes_small": axes_init_small,
    }
    # Test with threshold of 40
    rule = DepthOfReductionRule(
        max_depth_of_reduction=40,
        reference_data=reference_data,
        node_to_init_map=node_to_init_map,
        initializer_map=initializer_map,
    )

    # MatMul nodes
    matmul_node = helper.make_node(
        "MatMul", ["matmul_input_a", "matmul_input_b"], ["matmul_output"], name="matmul_node"
    )
    small_matmul_node = helper.make_node(
        "MatMul",
        ["small_matmul_a", "small_matmul_b"],
        ["small_matmul_output"],
        name="small_matmul_node",
    )

    # Conv nodes
    conv_node = helper.make_node(
        "Conv", ["conv_input", "conv_weight"], ["conv_output"], name="conv_node"
    )
    small_conv_node = helper.make_node(
        "Conv",
        ["small_conv_input", "small_conv_weight"],
        ["small_conv_output"],
        name="small_conv_node",
    )

    # Reduce nodes
    reduce_node = helper.make_node(
        "ReduceSum", ["reduce_input", "axes_large"], ["reduce_output"], name="reduce_node"
    )
    small_reduce_node = helper.make_node(
        "ReduceSum",
        ["reduce_input", "axes_small"],
        ["small_reduce_output"],
        name="small_reduce_node",
    )

    reduce_init_node = helper.make_node(
        "ReduceSum", ["reduce_init", "axes_large"], ["reduce_init_output"], name="reduce_init_node"
    )

    # Test MatMul: reduction depth 50 > 40, should be blocked
    assert rule.check(matmul_node) is True

    # Test small MatMul: reduction depth 10 < 40, should not be blocked
    assert rule.check(small_matmul_node) is False

    # Test Conv: reduction depth 288 > 40, should be blocked
    assert rule.check(conv_node) is True

    # Test small Conv: reduction depth 32 < 40, should not be blocked
    assert rule.check(small_conv_node) is False

    # Test Reduce: reduction depth 100 > 40, should be blocked
    assert rule.check(reduce_node) is True

    # Test small Reduce: reduction depth 20 < 40, should not be blocked
    assert rule.check(small_reduce_node) is False

    # Test ReduceInit: reduction depth 50 > 40, should be blocked
    assert rule.check(reduce_init_node) is True


@pytest.mark.skipif(
    Version(ort_version) < Version("1.21.0"), reason="WAR: Requires onnxruntime>=1.21.0"
)
def test_node_classifier(test_model):
    node_to_init_map = {key: [] for key in ["add_node", "mul_node"]}

    # Test with data range constraints
    ref_runner = ReferenceRunner(test_model)
    classifier = NodeClassifier(
        model=test_model,
        node_to_init_map=node_to_init_map,
        data_max=4.1,
    )

    # First set of inputs
    # add_out = X + Y : requires fp32
    calibration_data = {"X": np.array([[0.2, 0.3], [0.3, 0.4]], dtype=np.float32)}
    # Obtain reference data
    ref_outputs_dict = ref_runner.run(calibration_data)
    fp16_nodes, fp32_nodes = classifier.run(ref_outputs_dict)
    # Compute expected outputs for first set of inputs
    assert "add_node" in fp32_nodes
    assert "mul_node" in fp32_nodes  # add_out is input of mul_node
    assert len(fp16_nodes) + len(fp32_nodes) == 2
    # Test that no node is in both fp16 and fp32 lists
    assert not set(fp16_nodes).intersection(set(fp32_nodes))

    # Second set of inputs
    # all within fp16 range
    calibration_data = {"X": np.array([[-0.5, -1.5], [-3, -4]], dtype=np.float32)}
    # Obtain reference data
    ref_outputs_dict = ref_runner.run(calibration_data)
    fp16_nodes, fp32_nodes = classifier.run(ref_outputs_dict)
    assert "add_node" in fp16_nodes
    assert "mul_node" in fp16_nodes
    assert len(fp16_nodes) + len(fp32_nodes) == 2
    # Test that no node is in both fp16 and fp32 lists
    assert not set(fp16_nodes).intersection(set(fp32_nodes))


@pytest.mark.skipif(
    Version(ort_version) < Version("1.21.0"), reason="WAR: Requires onnxruntime>=1.21.0"
)
def test_node_classifier_custom_rule(test_model):
    node_to_init_map = {
        "add_node": [numpy_helper.from_array(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))],
        "mul_node": [numpy_helper.from_array(np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32))],
    }

    from modelopt.onnx.autocast.nodeclassifier import NodeRuleBase

    class CustomRule(NodeRuleBase):
        def _check_inner(self, node):
            # Return True if any initializer contains zeros
            print(node.input)
            return any("X" in input for input in node.input)

    classifier = NodeClassifier(
        model=test_model,
        node_to_init_map=node_to_init_map,
        custom_rule=CustomRule(),
    )

    fp16_nodes, fp32_nodes = classifier.run()
    assert "add_node" in fp32_nodes
    assert "mul_node" in fp16_nodes
    assert len(fp16_nodes) + len(fp32_nodes) == 2
    # Test that no node is in both fp16 and fp32 lists
    assert not set(fp16_nodes).intersection(set(fp32_nodes))


@pytest.mark.skipif(
    Version(ort_version) < Version("1.21.0"), reason="WAR: Requires onnxruntime>=1.21.0"
)
def test_node_classifier_op_types_to_exclude(test_model):
    node_to_init_map = {
        "add_node": [numpy_helper.from_array(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))],
        "mul_node": [numpy_helper.from_array(np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32))],
    }
    classifier = NodeClassifier(
        model=test_model,
        node_to_init_map=node_to_init_map,
        op_types_to_exclude=["Add"],
    )
    fp16_nodes, fp32_nodes = classifier.run()
    assert "add_node" in fp32_nodes
    assert "mul_node" in fp16_nodes
    assert len(fp16_nodes) + len(fp32_nodes) == 2
    # Test that no node is in both fp16 and fp32 lists
    assert not set(fp16_nodes).intersection(set(fp32_nodes))

    classifier2 = NodeClassifier(
        model=test_model,
        node_to_init_map=node_to_init_map,
        op_types_to_exclude=["Mul"],
    )
    fp16_nodes, fp32_nodes = classifier2.run()
    assert "add_node" in fp16_nodes
    assert "mul_node" in fp32_nodes
    assert len(fp16_nodes) + len(fp32_nodes) == 2
    # Test that no node is in both fp16 and fp32 lists
    assert not set(fp16_nodes).intersection(set(fp32_nodes))


# Test that nodes_to_include and op_types_to_include force nodes into low precision,
# even if they would otherwise be excluded by other rules.
def test_node_classifier_force_include(test_model):
    node_to_init_map = {
        "add_node": [
            numpy_helper.from_array(np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32))
        ],
        "mul_node": [numpy_helper.from_array(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))],
    }

    # Set init_max low so both nodes would normally be excluded (kept in FP32)
    # Force add_node to low precision, despite exceeding init_max
    classifier = NodeClassifier(
        model=test_model,
        node_to_init_map=node_to_init_map,
        init_max=1.0,
        nodes_to_include=["add_node"],
    )
    fp16_nodes, fp32_nodes = classifier.run()
    # add_node should be in fp16_nodes due to nodes_to_include, despite exceeding data_max
    assert "add_node" in fp16_nodes
    assert "mul_node" in fp32_nodes
    assert "add_node" not in fp32_nodes
    assert len(fp16_nodes) + len(fp32_nodes) == 2
    assert not set(fp16_nodes).intersection(set(fp32_nodes))

    # Test that include op rule override exclude op rule
    classifier2 = NodeClassifier(
        model=test_model,
        node_to_init_map=node_to_init_map,
        op_types_to_exclude=["Add"],
        nodes_to_include=["add_node"],  # Should override op_types_to_exclude
    )
    fp16_nodes, fp32_nodes = classifier2.run()
    assert "add_node" in fp16_nodes
    assert "add_node" not in fp32_nodes
    assert not set(fp16_nodes).intersection(set(fp32_nodes))

    # Set init_max low so both nodes would normally be excluded (kept in FP32)
    # Force op type Mul to low precision, despite exceeding init_max
    classifier3 = NodeClassifier(
        model=test_model,
        node_to_init_map=node_to_init_map,
        init_max=1.0,
        op_types_to_include=["Mul"],
    )
    fp16_nodes, fp32_nodes = classifier3.run()
    assert "mul_node" in fp16_nodes
    assert "add_node" in fp32_nodes
    assert not set(fp16_nodes).intersection(set(fp32_nodes))
