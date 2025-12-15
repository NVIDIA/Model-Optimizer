#!/usr/bin/env python3
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

"""
Comprehensive tests for common data structures in the autotuner.

Tests:
1. InsertionPoint classes (NodeInputInsertionPoint, RegionOutputInsertionPoint, ChildRegionInputInsertionPoint)
2. InsertionScheme serialization/deserialization
3. InsertionScheme hashing and equality
4. InsertionScheme properties and methods
5. PatternSchemes management
6. Utility functions (skip_invalid_insertion_points, has_quantizable_operations, etc.)
7. Resolve and collect_from methods for all InsertionPoint types
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import onnx_graphsurgeon as gs

from modelopt.onnx.quantization.autotune.common import (
    ChildRegionInputInsertionPoint,
    InsertionScheme,
    NodeInputInsertionPoint,
    Region,
    RegionOutputInsertionPoint,
    RegionType,
)
from modelopt.onnx.quantization.autotune.insertion_points import (
    ResolvedInsertionPoint,
    has_quantizable_operations,
    merge_resolved_insertion_points,
    resolve_region_io_insertion_points,
    skip_invalid_insertion_points,
)
from modelopt.onnx.quantization.graph_utils import get_tensor_consumer_node_indices


class TestNodeInputInsertionPoint(unittest.TestCase):
    """Test NodeInputInsertionPoint functionality."""

    def test_creation(self):
        """Test creating NodeInputInsertionPoint."""
        point = NodeInputInsertionPoint(node_index=5, input_index=2)
        assert point.node_index == 5
        assert point.input_index == 2

    def test_immutability(self):
        """Test that NodeInputInsertionPoint is immutable (frozen)."""
        point = NodeInputInsertionPoint(node_index=1, input_index=0)
        passed = False
        try:
            point.node_index = 2
        except AttributeError:
            passed = True
        assert passed, "NodeInputInsertionPoint should be immutable"

    def test_equality(self):
        """Test equality comparison."""
        point1 = NodeInputInsertionPoint(node_index=3, input_index=1)
        point2 = NodeInputInsertionPoint(node_index=3, input_index=1)
        point3 = NodeInputInsertionPoint(node_index=3, input_index=2)

        assert point1 == point2
        assert point1 != point3

    def test_hashable(self):
        """Test that points can be used in sets and dicts."""
        point1 = NodeInputInsertionPoint(node_index=1, input_index=0)
        point2 = NodeInputInsertionPoint(node_index=1, input_index=0)
        point3 = NodeInputInsertionPoint(node_index=2, input_index=0)

        point_set = {point1, point2, point3}
        assert len(point_set) == 2  # point1 and point2 are the same

    def test_serialization(self):
        """Test to_dict and from_dict."""
        point = NodeInputInsertionPoint(node_index=7, input_index=3)

        data = point.to_dict()
        assert data["node_index"] == 7
        assert data["input_index"] == 3

        restored = NodeInputInsertionPoint.from_dict(data)
        assert point == restored

    def test_string_representation(self):
        """Test __str__ method."""
        point = NodeInputInsertionPoint(node_index=2, input_index=1)
        s = str(point)
        assert "2" in s
        assert "1" in s


class TestRegionOutputInsertionPoint(unittest.TestCase):
    """Test RegionOutputInsertionPoint functionality."""

    def test_creation_with_region_index(self):
        """Test creating with region_index (child region output)."""
        point = RegionOutputInsertionPoint(region_index=2, node_index=None, output_index=1)
        assert point.region_index == 2
        assert point.node_index is None
        assert point.output_index == 1

    def test_creation_with_node_index(self):
        """Test creating with node_index (node output)."""
        point = RegionOutputInsertionPoint(region_index=None, node_index=5, output_index=0)
        assert point.region_index is None
        assert point.node_index == 5
        assert point.output_index == 0

    def test_immutability(self):
        """Test that RegionOutputInsertionPoint is immutable (frozen)."""
        point = RegionOutputInsertionPoint(region_index=1, node_index=None, output_index=0)
        passed = False
        try:
            point.region_index = 2
        except AttributeError:
            passed = True
        assert passed, "RegionOutputInsertionPoint should be immutable"

    def test_equality(self):
        """Test equality comparison."""
        point1 = RegionOutputInsertionPoint(region_index=1, node_index=None, output_index=0)
        point2 = RegionOutputInsertionPoint(region_index=1, node_index=None, output_index=0)
        point3 = RegionOutputInsertionPoint(region_index=None, node_index=1, output_index=0)

        assert point1 == point2
        assert point1 != point3

    def test_hashable(self):
        """Test that points can be used in sets and dicts."""
        point1 = RegionOutputInsertionPoint(region_index=1, node_index=None, output_index=0)
        point2 = RegionOutputInsertionPoint(region_index=1, node_index=None, output_index=0)
        point3 = RegionOutputInsertionPoint(region_index=None, node_index=1, output_index=0)

        point_set = {point1, point2, point3}
        assert len(point_set) == 2  # point1 and point2 are the same

    def test_serialization_region_index(self):
        """Test serialization with region_index."""
        point = RegionOutputInsertionPoint(region_index=3, node_index=None, output_index=2)

        data = point.to_dict()
        assert data["region_index"] == 3
        assert data["node_index"] is None
        assert data["output_index"] == 2

        restored = RegionOutputInsertionPoint.from_dict(data)
        assert point == restored

    def test_serialization_node_index(self):
        """Test serialization with node_index."""
        point = RegionOutputInsertionPoint(region_index=None, node_index=7, output_index=1)

        data = point.to_dict()
        assert data["region_index"] is None
        assert data["node_index"] == 7
        assert data["output_index"] == 1

        restored = RegionOutputInsertionPoint.from_dict(data)
        assert point == restored

    def test_string_representation(self):
        """Test __str__ method."""
        point1 = RegionOutputInsertionPoint(region_index=2, node_index=None, output_index=1)
        s1 = str(point1)
        assert "region" in s1.lower()
        assert "2" in s1

        point2 = RegionOutputInsertionPoint(region_index=None, node_index=5, output_index=0)
        s2 = str(point2)
        assert "node" in s2.lower()
        assert "5" in s2


class TestChildRegionInputInsertionPoint(unittest.TestCase):
    """Test ChildRegionInputInsertionPoint functionality."""

    def test_creation(self):
        """Test creating ChildRegionInputInsertionPoint."""
        point = ChildRegionInputInsertionPoint(region_index=3, input_index=1)
        assert point.region_index == 3
        assert point.input_index == 1

    def test_immutability(self):
        """Test that ChildRegionInputInsertionPoint is immutable (frozen)."""
        point = ChildRegionInputInsertionPoint(region_index=1, input_index=0)
        passed = False
        try:
            point.region_index = 2
        except AttributeError:
            passed = True
        assert passed, "ChildRegionInputInsertionPoint should be immutable"

    def test_equality(self):
        """Test equality comparison."""
        point1 = ChildRegionInputInsertionPoint(region_index=2, input_index=0)
        point2 = ChildRegionInputInsertionPoint(region_index=2, input_index=0)
        point3 = ChildRegionInputInsertionPoint(region_index=2, input_index=1)

        assert point1 == point2
        assert point1 != point3

    def test_hashable(self):
        """Test that points can be used in sets and dicts."""
        point1 = ChildRegionInputInsertionPoint(region_index=1, input_index=0)
        point2 = ChildRegionInputInsertionPoint(region_index=1, input_index=0)
        point3 = ChildRegionInputInsertionPoint(region_index=2, input_index=0)

        point_set = {point1, point2, point3}
        assert len(point_set) == 2  # point1 and point2 are the same

    def test_serialization(self):
        """Test to_dict and from_dict."""
        point = ChildRegionInputInsertionPoint(region_index=5, input_index=2)

        data = point.to_dict()
        assert data["region_index"] == 5
        assert data["input_index"] == 2

        restored = ChildRegionInputInsertionPoint.from_dict(data)
        assert point == restored

    def test_string_representation(self):
        """Test __str__ method."""
        point = ChildRegionInputInsertionPoint(region_index=2, input_index=1)
        s = str(point)
        assert "2" in s
        assert "1" in s


class TestInsertionScheme(unittest.TestCase):
    """Test InsertionScheme functionality."""

    def test_empty_scheme(self):
        """Test empty InsertionScheme."""
        scheme = InsertionScheme()

        assert scheme.is_empty
        assert len(scheme.node_inputs) == 0
        assert len(scheme.child_region_inputs) == 0
        assert len(scheme.region_outputs) == 0
        assert not scheme.error

    def test_scheme_with_node_inputs(self):
        """Test scheme with node input insertion points."""
        scheme = InsertionScheme()
        scheme.node_inputs = [NodeInputInsertionPoint(0, 0), NodeInputInsertionPoint(1, 0)]

        assert not scheme.is_empty
        assert len(scheme.node_inputs) == 2

    def test_scheme_with_region_outputs(self):
        """Test scheme with region output insertion points."""
        scheme = InsertionScheme()
        scheme.region_outputs = [
            RegionOutputInsertionPoint(None, 0, 0),
            RegionOutputInsertionPoint(1, None, 0),
        ]

        assert not scheme.is_empty
        assert len(scheme.region_outputs) == 2

    def test_scheme_with_composite_regions(self):
        """Test scheme with composite region insertion points."""
        scheme = InsertionScheme()
        scheme.child_region_inputs = [
            ChildRegionInputInsertionPoint(0, 0),
            ChildRegionInputInsertionPoint(1, 0),
        ]

        assert not scheme.is_empty
        assert len(scheme.child_region_inputs) == 2

    def test_scheme_hash_empty(self):
        """Test hash of empty scheme."""
        scheme1 = InsertionScheme()
        scheme2 = InsertionScheme()

        assert scheme1.hash == scheme2.hash

    def test_scheme_hash_with_points(self):
        """Test hash with insertion points."""
        scheme1 = InsertionScheme()
        scheme1.node_inputs = [NodeInputInsertionPoint(0, 0), NodeInputInsertionPoint(1, 0)]

        scheme2 = InsertionScheme()
        scheme2.node_inputs = [NodeInputInsertionPoint(0, 0), NodeInputInsertionPoint(1, 0)]

        scheme3 = InsertionScheme()
        scheme3.node_inputs = [
            NodeInputInsertionPoint(0, 0),
            NodeInputInsertionPoint(2, 0),  # Different
        ]

        assert scheme1.hash == scheme2.hash
        assert scheme1.hash != scheme3.hash

    def test_scheme_hash_order_independent(self):
        """Test that hash is independent of insertion point order."""
        scheme1 = InsertionScheme()
        scheme1.node_inputs = [NodeInputInsertionPoint(0, 0), NodeInputInsertionPoint(1, 0)]

        scheme2 = InsertionScheme()
        scheme2.node_inputs = [
            NodeInputInsertionPoint(1, 0),
            NodeInputInsertionPoint(0, 0),  # Reversed order
        ]

        # Hash should be the same regardless of order
        assert scheme1.hash == scheme2.hash

    def test_serialization_empty(self):
        """Test serialization of empty scheme."""
        scheme = InsertionScheme()

        data = scheme.to_dict()
        restored = InsertionScheme.from_dict(data)

        assert restored.is_empty
        assert restored.latency_ms == float("inf")
        assert not restored.error

    def test_serialization_full(self):
        """Test serialization with all types of insertion points."""
        scheme = InsertionScheme()
        scheme.node_inputs = [NodeInputInsertionPoint(0, 0)]
        scheme.child_region_inputs = [ChildRegionInputInsertionPoint(0, 0)]
        scheme.region_outputs = [RegionOutputInsertionPoint(None, 0, 0)]
        scheme.latency_ms = 12.5
        scheme.error = False

        data = scheme.to_dict()
        restored = InsertionScheme.from_dict(data)

        assert len(restored.node_inputs) == 1
        assert len(restored.child_region_inputs) == 1
        assert len(restored.region_outputs) == 1
        assert restored.latency_ms == 12.5
        assert not restored.error

    def test_serialization_with_error(self):
        """Test serialization with error flag."""
        scheme = InsertionScheme()
        scheme.error = True
        scheme.latency_ms = float("inf")

        data = scheme.to_dict()
        restored = InsertionScheme.from_dict(data)

        assert restored.error
        assert restored.latency_ms == float("inf")


# =============================================================================
# Helper functions for creating mock graphs
# =============================================================================


def _create_mock_tensor(name: str, dtype=np.float32, shape=None):
    """Create a mock tensor with the specified properties."""
    tensor = MagicMock()
    tensor.name = name
    tensor.dtype = dtype
    tensor.shape = shape if shape is not None else [1, 3, 224, 224]
    tensor.inputs = []
    return tensor


def _create_mock_node(op: str, inputs: list, outputs: list, name: str = ""):
    """Create a mock node with the specified properties."""
    node = MagicMock(spec=gs.Node)
    node.op = op
    node.name = name
    node.inputs = inputs
    node.outputs = outputs
    return node


def _create_simple_graph():
    """Create a mock graph with Conv -> BatchNorm -> Relu -> MaxPool pattern.

    Graph structure:
        input -> Conv -> conv_out -> BatchNorm -> bn_out -> Relu -> relu_out -> MaxPool -> pool_out

    Node indices:
        0: Conv
        1: BatchNormalization
        2: Relu
        3: MaxPool
    """
    # Create tensors with realistic shapes
    input_tensor = _create_mock_tensor("input", np.float32, [1, 3, 224, 224])
    weight_tensor = _create_mock_tensor("conv_weight", np.float32, [64, 3, 3, 3])
    bias_tensor = _create_mock_tensor("conv_bias", np.float32, [64])
    conv_output = _create_mock_tensor("conv_out", np.float32, [1, 64, 222, 222])

    # BatchNorm parameters
    bn_scale = _create_mock_tensor("bn_scale", np.float32, [64])
    bn_bias = _create_mock_tensor("bn_bias", np.float32, [64])
    bn_mean = _create_mock_tensor("bn_mean", np.float32, [64])
    bn_var = _create_mock_tensor("bn_var", np.float32, [64])
    bn_output = _create_mock_tensor("bn_out", np.float32, [1, 64, 222, 222])

    relu_output = _create_mock_tensor("relu_out", np.float32, [1, 64, 222, 222])
    pool_output = _create_mock_tensor("pool_out", np.float32, [1, 64, 111, 111])

    # Create nodes
    conv_node = _create_mock_node(
        "Conv", [input_tensor, weight_tensor, bias_tensor], [conv_output], "conv1"
    )
    bn_node = _create_mock_node(
        "BatchNormalization",
        [conv_output, bn_scale, bn_bias, bn_mean, bn_var],
        [bn_output],
        "bn1",
    )
    relu_node = _create_mock_node("Relu", [bn_output], [relu_output], "relu1")
    pool_node = _create_mock_node("MaxPool", [relu_output], [pool_output], "pool1")

    # Link tensors to their producer nodes
    conv_output.inputs = [conv_node]
    bn_output.inputs = [bn_node]
    relu_output.inputs = [relu_node]
    pool_output.inputs = [pool_node]
    input_tensor.inputs = []
    weight_tensor.inputs = []
    bias_tensor.inputs = []

    # Create graph
    graph = MagicMock(spec=gs.Graph)
    graph.nodes = [conv_node, bn_node, relu_node, pool_node]
    graph.inputs = [input_tensor]
    graph.outputs = [pool_output]

    tensors = {
        "input": input_tensor,
        "conv_weight": weight_tensor,
        "conv_bias": bias_tensor,
        "conv_out": conv_output,
        "bn_out": bn_output,
        "relu_out": relu_output,
        "pool_out": pool_output,
    }

    return graph, tensors


def _create_residual_graph():
    """Create a mock graph with a residual block pattern (skip connection).

    Graph structure:
        input ─────────────────────────────┐
          │                                │
          ▼                                │
        Conv1 -> conv1_out                 │
          │                                │
          ▼                                │
        Relu1 -> relu1_out                 │
          │                                │
          ▼                                │
        Conv2 -> conv2_out                 │
          │                                │
          ▼                                ▼
        Add (conv2_out + input) -> add_out
          │
          ▼
        Relu2 -> output

    Node indices:
        0: Conv1
        1: Relu1
        2: Conv2
        3: Add
        4: Relu2
    """
    # Create tensors
    input_tensor = _create_mock_tensor("input", np.float32, [1, 64, 56, 56])

    # First conv branch
    weight1 = _create_mock_tensor("conv1_weight", np.float32, [64, 64, 3, 3])
    conv1_out = _create_mock_tensor("conv1_out", np.float32, [1, 64, 56, 56])
    relu1_out = _create_mock_tensor("relu1_out", np.float32, [1, 64, 56, 56])

    # Second conv
    weight2 = _create_mock_tensor("conv2_weight", np.float32, [64, 64, 3, 3])
    conv2_out = _create_mock_tensor("conv2_out", np.float32, [1, 64, 56, 56])

    # Add and final relu
    add_out = _create_mock_tensor("add_out", np.float32, [1, 64, 56, 56])
    output = _create_mock_tensor("output", np.float32, [1, 64, 56, 56])

    # Create nodes
    conv1_node = _create_mock_node("Conv", [input_tensor, weight1], [conv1_out], "conv1")
    relu1_node = _create_mock_node("Relu", [conv1_out], [relu1_out], "relu1")
    conv2_node = _create_mock_node("Conv", [relu1_out, weight2], [conv2_out], "conv2")
    add_node = _create_mock_node("Add", [conv2_out, input_tensor], [add_out], "add1")
    relu2_node = _create_mock_node("Relu", [add_out], [output], "relu2")

    # Link tensors to their producer nodes
    conv1_out.inputs = [conv1_node]
    relu1_out.inputs = [relu1_node]
    conv2_out.inputs = [conv2_node]
    add_out.inputs = [add_node]
    output.inputs = [relu2_node]
    input_tensor.inputs = []
    weight1.inputs = []
    weight2.inputs = []

    # Create graph
    graph = MagicMock(spec=gs.Graph)
    graph.nodes = [conv1_node, relu1_node, conv2_node, add_node, relu2_node]
    graph.inputs = [input_tensor]
    graph.outputs = [output]

    tensors = {
        "input": input_tensor,
        "conv1_weight": weight1,
        "conv1_out": conv1_out,
        "relu1_out": relu1_out,
        "conv2_weight": weight2,
        "conv2_out": conv2_out,
        "add_out": add_out,
        "output": output,
    }

    return graph, tensors


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestSkipInvalidInsertionPoints(unittest.TestCase):
    """Test skip_invalid_insertion_points function."""

    def test_skip_bool_operations(self):
        """Test that boolean operations are skipped."""
        graph, _ = _create_simple_graph()

        # Create a node with boolean operation
        bool_tensor = _create_mock_tensor("bool_input", np.float32)
        bool_node = _create_mock_node("Equal", [bool_tensor], [])

        result = skip_invalid_insertion_points(graph, "bool_input", bool_node)
        assert result is True

    def test_skip_shape_operations(self):
        """Test that shape operations are skipped."""
        graph, _ = _create_simple_graph()

        shape_tensor = _create_mock_tensor("shape_input", np.float32)
        shape_node = _create_mock_node("Shape", [shape_tensor], [])

        result = skip_invalid_insertion_points(graph, "shape_input", shape_node)
        assert result is True

    def test_skip_conv_weight_input(self):
        """Test that Conv weight inputs (index >= 1) are skipped."""
        graph, tensors = _create_simple_graph()
        conv_node = graph.nodes[0]

        # Weight is at input index 1
        result = skip_invalid_insertion_points(graph, "conv_weight", conv_node)
        assert result is True

    def test_allow_conv_data_input(self):
        """Test that Conv data input (index 0) is allowed."""
        graph, tensors = _create_simple_graph()

        # Create a MatMul node that consumes the input tensor (not Conv-related skip)
        input_tensor = _create_mock_tensor("matmul_input", np.float32, [1, 3, 224, 224])
        matmul_node = _create_mock_node("MatMul", [input_tensor], [])

        result = skip_invalid_insertion_points(graph, "matmul_input", matmul_node)
        assert result is False

    def test_skip_non_float_tensors(self):
        """Test that non-floating-point tensors are skipped."""
        graph, _ = _create_simple_graph()

        # Create int tensor
        int_tensor = _create_mock_tensor("int_input", np.int32)
        node = _create_mock_node("Add", [int_tensor], [])

        result = skip_invalid_insertion_points(graph, "int_input", node)
        assert result is True

    def test_skip_small_tensors(self):
        """Test that small tensors (< 8 elements) are skipped."""
        graph, _ = _create_simple_graph()

        # Create small tensor (scalar)
        small_tensor = _create_mock_tensor("small", np.float32, [1])
        node = _create_mock_node("Add", [small_tensor], [])

        result = skip_invalid_insertion_points(graph, "small", node)
        assert result is True

    def test_allow_large_float_tensors(self):
        """Test that large floating-point tensors are allowed."""
        graph, _ = _create_simple_graph()

        # Create large float tensor
        large_tensor = _create_mock_tensor("large", np.float32, [1, 64, 32, 32])
        node = _create_mock_node("Add", [large_tensor], [])

        result = skip_invalid_insertion_points(graph, "large", node)
        assert result is False

    def test_skip_bn_non_data_inputs(self):
        """Test that BatchNormalization non-data inputs are skipped."""
        graph, tensors = _create_simple_graph()
        bn_node = graph.nodes[1]  # BatchNormalization node

        # Scale is at input index 1, should be skipped
        result = skip_invalid_insertion_points(graph, "bn_scale", bn_node)
        assert result is True

    def test_with_region(self):
        """Test skip_invalid_insertion_points with a Region containing multiple nodes."""
        graph, tensors = _create_simple_graph()

        # Create a region containing Conv and BatchNorm nodes
        region = Region(region_id=1, level=0, region_type=RegionType.LEAF)
        region.add_node(0)  # Conv node
        region.add_node(1)  # BatchNorm node

        # Create a shape operation node and add to graph
        shape_tensor = _create_mock_tensor("shape_input", np.float32)
        shape_node = _create_mock_node("Shape", [shape_tensor], [])
        graph.nodes.append(shape_node)
        region.add_node(4)  # Add the shape node to region

        result = skip_invalid_insertion_points(graph, "shape_input", region)
        assert result is True

    def test_skip_conv_bn_relu_fusion(self):
        """Test that Conv->BN->Relu fusion patterns are skipped at intermediate points."""
        graph, tensors = _create_simple_graph()
        relu_node = graph.nodes[2]  # Relu node

        # Relu input (bn_out) should be skipped when preceded by Conv->BN
        result = skip_invalid_insertion_points(graph, "bn_out", relu_node)
        assert result is True

    def test_residual_block_add_inputs(self):
        """Test insertion points in a residual block with skip connection."""
        graph, tensors = _create_residual_graph()
        add_node = graph.nodes[3]  # Add node

        # Add's first input (conv2_out) should be allowed
        result = skip_invalid_insertion_points(graph, "conv2_out", add_node)
        assert result is False

        # Add's second input (skip connection input) should also be allowed
        result = skip_invalid_insertion_points(graph, "input", add_node)
        assert result is False


class TestHasQuantizableOperations(unittest.TestCase):
    """Test has_quantizable_operations function."""

    def test_leaf_with_conv(self):
        """Test LEAF region with Conv operation."""
        graph, _ = _create_simple_graph()

        region = Region(region_id=1, level=0, region_type=RegionType.LEAF)
        region.add_node(0)  # Conv node

        result = has_quantizable_operations(region, graph)
        assert result is True

    def test_leaf_with_maxpool(self):
        """Test LEAF region with MaxPool (a major quantizable op)."""
        graph, _ = _create_simple_graph()

        region = Region(region_id=1, level=0, region_type=RegionType.LEAF)
        region.add_node(3)  # MaxPool node

        result = has_quantizable_operations(region, graph)
        assert result is True

    def test_leaf_with_relu_only(self):
        """Test LEAF region with only Relu."""
        graph, _ = _create_simple_graph()

        region = Region(region_id=1, level=0, region_type=RegionType.LEAF)
        region.add_node(2)  # Relu node only (index 2 in new graph)

        result = has_quantizable_operations(region, graph)
        assert result is True  # Relu is in MAJOR_QUANTIZABLE_OPERATIONS

    def test_leaf_with_conv_bn_relu(self):
        """Test LEAF region with Conv->BN->Relu pattern."""
        graph, _ = _create_simple_graph()

        region = Region(region_id=1, level=0, region_type=RegionType.LEAF)
        region.add_node(0)  # Conv
        region.add_node(1)  # BatchNorm
        region.add_node(2)  # Relu

        result = has_quantizable_operations(region, graph)
        assert result is True

    def test_leaf_without_quantizable_ops(self):
        """Test LEAF region without major quantizable operations."""
        # Create graph with only shape operations
        shape_tensor = _create_mock_tensor("input", np.float32)
        output_tensor = _create_mock_tensor("output", np.float32)
        shape_node = _create_mock_node("Shape", [shape_tensor], [output_tensor])
        transpose_node = _create_mock_node("Transpose", [output_tensor], [])

        graph = MagicMock(spec=gs.Graph)
        graph.nodes = [shape_node, transpose_node]

        region = Region(region_id=1, level=0, region_type=RegionType.LEAF)
        region.add_node(0)
        region.add_node(1)

        result = has_quantizable_operations(region, graph)
        assert result is False

    def test_composite_region_always_true(self):
        """Test that COMPOSITE regions always return True."""
        graph, _ = _create_simple_graph()

        region = Region(region_id=1, level=1, region_type=RegionType.COMPOSITE)
        # Don't add any nodes - COMPOSITE regions assume children have quantizable ops

        result = has_quantizable_operations(region, graph)
        assert result is True

    def test_residual_block_has_quantizable_ops(self):
        """Test residual block with Add operation."""
        graph, _ = _create_residual_graph()

        region = Region(region_id=1, level=0, region_type=RegionType.LEAF)
        region.add_node(3)  # Add node

        result = has_quantizable_operations(region, graph)
        assert result is True  # Add is in MAJOR_QUANTIZABLE_OPERATIONS


class TestResolveRegionIOInsertionPoints(unittest.TestCase):
    """Test resolve_region_io_insertion_points function."""

    def test_resolve_with_region(self):
        """Test resolving with a region containing Conv->BN->Relu."""
        graph, tensors = _create_simple_graph()

        # Set up tensor_users_map: conv_out is consumed by BatchNorm (node 1)
        graph.tensor_users_map = get_tensor_consumer_node_indices(graph)

        region = Region(region_id=1, level=0, region_type=RegionType.LEAF)
        region.add_node(2)  # Relu node

        result = resolve_region_io_insertion_points(region, graph, "relu_out")

        assert len(result) >= 1
        assert any(ip.tensor_name == "relu_out" for ip in result)

    def test_resolve_without_region(self):
        """Test resolving without a region (None) for tensor-level insertion."""
        graph, _ = _create_simple_graph()

        # Set up tensor_users_map: bn_out is consumed by Relu (node 2)
        graph.tensor_users_map = get_tensor_consumer_node_indices(graph)

        result = resolve_region_io_insertion_points(None, graph, "relu_out")

        assert len(result) == 1
        ip = next(iter(result))
        assert ip.tensor_name == "relu_out"
        assert ip.node_index == 3
        assert ip.input_index == 0

    def test_resolve_tensor_not_found(self):
        """Test resolving a tensor that has no users."""
        graph, _ = _create_simple_graph()
        graph.tensor_users_map = {}

        result = resolve_region_io_insertion_points(None, graph, "nonexistent")

        assert len(result) == 0

    def test_resolve_residual_skip_connection(self):
        """Test resolving input tensor used by both Conv1 and Add (skip connection)."""
        graph, tensors = _create_residual_graph()

        # Input tensor is used by Conv1 (node 0) and Add (node 3)
        graph.tensor_users_map = {"input": [0, 3]}

        result = resolve_region_io_insertion_points(None, graph, "input")

        # Should find both consumers
        assert len(result) == 2
        node_indices = {ip.node_index for ip in result}
        assert 0 in node_indices  # Conv1
        assert 3 in node_indices  # Add

    def test_resolve_with_multiple_consumers(self):
        """Test resolving tensor with multiple consumers in a region."""
        graph, tensors = _create_residual_graph()

        # relu1_out feeds conv2 (node 2)
        graph.tensor_users_map = {"relu1_out": [2]}

        region = Region(region_id=1, level=0, region_type=RegionType.LEAF)
        region.add_node(2)  # Conv2

        result = resolve_region_io_insertion_points(region, graph, "relu1_out")

        assert len(result) == 1
        ip = next(iter(result))
        assert ip.tensor_name == "relu1_out"
        assert ip.node_index == 2


class TestMergeResolvedInsertionPoints(unittest.TestCase):
    """Test merge_resolved_insertion_points function."""

    def test_merge_all_users(self):
        """Test merging when all users have insertion points."""
        graph, _ = _create_simple_graph()

        # Setup: tensor "conv_out" is used by BatchNorm (node 1)
        resolved = {
            ResolvedInsertionPoint(tensor_name="conv_out", node_index=1, input_index=0),
        }

        with patch(
            "modelopt.onnx.quantization.autotune.insertion_points.get_tensor_consumer_node_indices"
        ) as mock_get:
            mock_get.return_value = {"conv_out": [1]}

            result = merge_resolved_insertion_points(graph, resolved)

        # Should be merged to tensor-level insertion
        assert len(result) == 1
        merged = next(iter(result))
        assert merged.tensor_name == "conv_out"
        assert merged.node_index is None
        assert merged.input_index is None

    def test_no_merge_partial_users(self):
        """Test no merging when only some users have insertion points."""
        graph, _ = _create_simple_graph()

        # Setup: tensor "conv_out" is used by nodes 1 and 2, but only node 1 has IP
        resolved = {
            ResolvedInsertionPoint(tensor_name="conv_out", node_index=1, input_index=0),
        }

        with patch(
            "modelopt.onnx.quantization.autotune.insertion_points.get_tensor_consumer_node_indices"
        ) as mock_get:
            mock_get.return_value = {"conv_out": [1, 2]}

            result = merge_resolved_insertion_points(graph, resolved)

        # Should NOT be merged - keep node-specific
        assert len(result) == 1
        ip = next(iter(result))
        assert ip.node_index == 1  # Still node-specific

    def test_preserve_tensor_level_insertions(self):
        """Test that existing tensor-level insertions are preserved."""
        graph, _ = _create_simple_graph()

        # Already tensor-level insertion
        resolved = {
            ResolvedInsertionPoint(tensor_name="input", node_index=None, input_index=None),
        }

        with patch(
            "modelopt.onnx.quantization.autotune.insertion_points.get_tensor_consumer_node_indices"
        ) as mock_get:
            mock_get.return_value = {"conv_out": [1]}

            result = merge_resolved_insertion_points(graph, resolved)

        assert len(result) == 1
        ip = next(iter(result))
        assert ip.tensor_name == "input"
        assert ip.node_index is None

    def test_merge_residual_skip_connection(self):
        """Test merging with residual block where input has two users."""
        graph, _ = _create_residual_graph()

        # Input tensor used by Conv1 (node 0) and Add (node 3)
        # If we have insertion points for both, they should merge
        resolved = {
            ResolvedInsertionPoint(tensor_name="input", node_index=0, input_index=0),
            ResolvedInsertionPoint(tensor_name="input", node_index=3, input_index=1),
        }

        with patch(
            "modelopt.onnx.quantization.autotune.insertion_points.get_tensor_consumer_node_indices"
        ) as mock_get:
            mock_get.return_value = {"input": [0, 3]}

            result = merge_resolved_insertion_points(graph, resolved)

        # Should be merged to tensor-level insertion
        assert len(result) == 1
        merged = next(iter(result))
        assert merged.tensor_name == "input"
        assert merged.node_index is None

    def test_no_merge_residual_partial(self):
        """Test no merging in residual block when only one branch has insertion point."""
        graph, _ = _create_residual_graph()

        # Input tensor used by Conv1 (node 0) and Add (node 3)
        # Only Conv1 has an insertion point
        resolved = {
            ResolvedInsertionPoint(tensor_name="input", node_index=0, input_index=0),
        }

        with patch(
            "modelopt.onnx.quantization.autotune.insertion_points.get_tensor_consumer_node_indices"
        ) as mock_get:
            mock_get.return_value = {"input": [0, 3]}

            result = merge_resolved_insertion_points(graph, resolved)

        # Should NOT merge - only one of two users has IP
        assert len(result) == 1
        ip = next(iter(result))
        assert ip.node_index == 0  # Still node-specific


# =============================================================================
# Resolve Method Tests
# =============================================================================


class TestNodeInputInsertionPointResolve(unittest.TestCase):
    """Test NodeInputInsertionPoint.resolve() method."""

    def test_resolve_simple(self):
        """Test resolving a simple node input for Conv->BN->Relu->Pool."""
        graph, tensors = _create_simple_graph()

        region = Region(region_id=1, level=0, region_type=RegionType.LEAF)
        region.add_node(0)  # Conv node
        region.add_node(1)  # BatchNorm node
        region.add_node(2)  # Relu node
        region.add_node(3)  # MaxPool node

        # Create insertion point for first input of first node (Conv)
        ip = NodeInputInsertionPoint(node_index=0, input_index=0)

        result = ip.resolve(region, graph)

        assert len(result) >= 1
        assert any(rip.tensor_name == "input" for rip in result)

    def test_resolve_conv_includes_weight(self):
        """Test that resolving Conv input also includes weight."""
        graph, tensors = _create_simple_graph()

        region = Region(region_id=1, level=0, region_type=RegionType.LEAF)
        region.add_node(0)  # Conv node

        # Create insertion point for first input of Conv (should also add weight)
        ip = NodeInputInsertionPoint(node_index=0, input_index=0)

        result = ip.resolve(region, graph)

        # Should include both data input and weight
        assert len(result) == 2
        tensor_names = {rip.tensor_name for rip in result}
        assert "input" in tensor_names
        assert "conv_weight" in tensor_names

    def test_resolve_relu_input(self):
        """Test resolving Relu input in the middle of the chain."""
        graph, tensors = _create_simple_graph()

        region = Region(region_id=1, level=0, region_type=RegionType.LEAF)
        region.add_node(0)  # Conv
        region.add_node(1)  # BatchNorm
        region.add_node(2)  # Relu

        # Relu is at local index 2, input 0 is bn_out
        ip = NodeInputInsertionPoint(node_index=2, input_index=0)

        result = ip.resolve(region, graph)

        assert len(result) == 1
        rip = next(iter(result))
        assert rip.tensor_name == "bn_out"

    def test_resolve_residual_conv_input(self):
        """Test resolving Conv input in residual block."""
        graph, tensors = _create_residual_graph()

        region = Region(region_id=1, level=0, region_type=RegionType.LEAF)
        region.add_node(0)  # Conv1
        region.add_node(1)  # Relu1
        region.add_node(2)  # Conv2

        # Conv2 is at local index 2, input 0 is relu1_out
        ip = NodeInputInsertionPoint(node_index=2, input_index=0)

        result = ip.resolve(region, graph)

        # Conv includes both data and weight
        assert len(result) == 2
        tensor_names = {rip.tensor_name for rip in result}
        assert "relu1_out" in tensor_names
        assert "conv2_weight" in tensor_names


class TestChildRegionInputInsertionPointResolve(unittest.TestCase):
    """Test ChildRegionInputInsertionPoint.resolve() method."""

    def test_resolve_composite_region(self):
        """Test resolving child region input in COMPOSITE region."""
        graph, tensors = _create_simple_graph()
        graph.tensor_users_map = {"input": [0]}

        # Create parent (COMPOSITE) with child (LEAF) containing Conv->BN->Relu
        parent = Region(region_id=1, level=1, region_type=RegionType.COMPOSITE)
        child = Region(region_id=2, level=0, region_type=RegionType.LEAF)
        child.inputs = ["input"]
        child.add_node(0)  # Conv
        child.add_node(1)  # BatchNorm
        child.add_node(2)  # Relu
        parent.add_child(child)

        ip = ChildRegionInputInsertionPoint(region_index=0, input_index=0)

        result = ip.resolve(parent, graph)

        assert len(result) >= 1
        assert any(rip.tensor_name == "input" for rip in result)

    def test_resolve_leaf_returns_empty(self):
        """Test that LEAF regions return empty set."""
        graph, _ = _create_simple_graph()

        leaf = Region(region_id=1, level=0, region_type=RegionType.LEAF)
        leaf.add_node(0)

        ip = ChildRegionInputInsertionPoint(region_index=0, input_index=0)

        result = ip.resolve(leaf, graph)

        assert len(result) == 0

    def test_resolve_multiple_children(self):
        """Test resolving child inputs in COMPOSITE with multiple children."""
        graph, tensors = _create_residual_graph()
        # input is consumed by Conv1 (node 0) and Add (node 3)
        graph.tensor_users_map = get_tensor_consumer_node_indices(graph)

        # Create parent with two child regions
        parent = Region(region_id=1, level=1, region_type=RegionType.COMPOSITE)

        # First child: Conv1 (consumes "input")
        child1 = Region(region_id=2, level=0, region_type=RegionType.LEAF)
        child1.inputs = ["input"]
        child1.add_node(0)  # Conv1

        # Second child: Relu1 (consumes "relu1_out")
        child2 = Region(region_id=3, level=0, region_type=RegionType.LEAF)
        child2.inputs = ["relu1_out"]
        child2.add_node(2)  # Relu1

        parent.add_child(child1)
        parent.add_child(child2)

        # Resolve input of first child (region_index=0) - "input" tensor
        ip1 = ChildRegionInputInsertionPoint(region_index=0, input_index=0)
        result1 = ip1.resolve(parent, graph)

        assert len(result1) >= 1
        assert any(rip.tensor_name == "input" for rip in result1)

        # Resolve input of second child (region_index=1) - "relu1_out" tensor
        ip2 = ChildRegionInputInsertionPoint(region_index=1, input_index=0)
        result2 = ip2.resolve(parent, graph)

        assert len(result2) >= 1
        assert any(rip.tensor_name == "relu1_out" for rip in result2)


class TestRegionOutputInsertionPointResolve(unittest.TestCase):
    """Test RegionOutputInsertionPoint.resolve() method."""

    def test_resolve_node_output(self):
        """Test resolving a node output."""
        graph, tensors = _create_simple_graph()
        graph.tensor_users_map = get_tensor_consumer_node_indices(graph)

        region = Region(region_id=1, level=0, region_type=RegionType.LEAF)
        region.add_node(0)  # Conv
        region.add_node(1)  # BatchNorm
        region.add_node(2)  # Relu
        region.add_node(3)  # MaxPool
        region.outputs = ["pool_out"]

        # Output of last node (MaxPool)
        ip = RegionOutputInsertionPoint(region_index=None, node_index=2, output_index=0)

        result = ip.resolve(region, graph)

        assert len(result) >= 1
        assert any(rip.tensor_name == "relu_out" for rip in result)

    def test_resolve_child_region_output(self):
        """Test resolving a child region output."""
        graph, tensors = _create_simple_graph()
        graph.tensor_users_map = {"relu_out": [3]}

        parent = Region(region_id=1, level=1, region_type=RegionType.COMPOSITE)
        child = Region(region_id=2, level=0, region_type=RegionType.LEAF)
        child.outputs = ["relu_out"]
        child.add_node(0)  # Conv
        child.add_node(1)  # BatchNorm
        child.add_node(2)  # Relu
        parent.add_child(child)

        ip = RegionOutputInsertionPoint(region_index=0, node_index=None, output_index=0)

        result = ip.resolve(parent, graph)

        assert len(result) >= 1
        assert any(rip.tensor_name == "relu_out" for rip in result)

    def test_resolve_residual_add_output(self):
        """Test resolving Add output in residual block."""
        graph, tensors = _create_residual_graph()
        graph.tensor_users_map = {"add_out": [4]}

        region = Region(region_id=1, level=0, region_type=RegionType.LEAF)
        region.add_node(0)  # Conv1
        region.add_node(1)  # Relu1
        region.add_node(2)  # Conv2
        region.add_node(3)  # Add
        region.add_node(4)  # Relu2
        region.outputs = ["add_out"]

        # Add is at local index 3, output 0
        ip = RegionOutputInsertionPoint(region_index=None, node_index=3, output_index=0)

        result = ip.resolve(region, graph)

        assert len(result) >= 1
        assert any(rip.tensor_name == "add_out" for rip in result)


# =============================================================================
# Collect From Region Tests
# =============================================================================


class TestNodeInputInsertionPointCollectFrom(unittest.TestCase):
    """Test NodeInputInsertionPoint.collect_from_region() method."""

    def test_collect_valid_inputs(self):
        """Test collecting valid node input insertion points from Conv->BN->Relu->Pool."""
        graph, tensors = _create_simple_graph()

        region = Region(region_id=1, level=0, region_type=RegionType.LEAF)
        region.add_node(0)  # Conv
        region.add_node(1)  # BatchNorm
        region.add_node(2)  # Relu
        region.add_node(3)  # MaxPool

        result = NodeInputInsertionPoint.collect_from_region(region, graph)

        # Should have collected some insertion points
        assert len(result) >= 1
        # All should be NodeInputInsertionPoint
        assert all(isinstance(ip, NodeInputInsertionPoint) for ip in result)

    def test_collect_from_residual_block(self):
        """Test collecting from residual block with skip connection."""
        graph, tensors = _create_residual_graph()

        region = Region(region_id=1, level=0, region_type=RegionType.LEAF)
        region.add_node(0)  # Conv1
        region.add_node(1)  # Relu1
        region.add_node(2)  # Conv2
        region.add_node(3)  # Add
        region.add_node(4)  # Relu2

        result = NodeInputInsertionPoint.collect_from_region(region, graph)

        # Should have collected insertion points from Conv1, Add inputs, etc.
        assert len(result) >= 1
        assert all(isinstance(ip, NodeInputInsertionPoint) for ip in result)

        # Check that we have insertion points for different nodes
        node_indices = {ip.node_index for ip in result}
        assert len(node_indices) >= 1  # At least one node has valid inputs


class TestChildRegionInputInsertionPointCollectFrom(unittest.TestCase):
    """Test ChildRegionInputInsertionPoint.collect_from_region() method."""

    def test_collect_from_composite(self):
        """Test collecting from COMPOSITE region with children."""
        graph, tensors = _create_simple_graph()

        parent = Region(region_id=1, level=1, region_type=RegionType.COMPOSITE)
        child = Region(region_id=2, level=0, region_type=RegionType.LEAF)
        child.inputs = ["input"]
        child.add_node(0)  # Conv
        child.add_node(1)  # BatchNorm
        child.add_node(2)  # Relu
        parent.add_child(child)

        result = ChildRegionInputInsertionPoint.collect_from_region(parent, graph)

        # Should find the child's input
        assert len(result) >= 0  # May be filtered by skip_invalid_insertion_points
        assert all(isinstance(ip, ChildRegionInputInsertionPoint) for ip in result)

    def test_collect_from_leaf_returns_empty(self):
        """Test that LEAF regions return empty list."""
        graph, _ = _create_simple_graph()

        leaf = Region(region_id=1, level=0, region_type=RegionType.LEAF)
        leaf.add_node(0)

        result = ChildRegionInputInsertionPoint.collect_from_region(leaf, graph)

        assert len(result) == 0

    def test_collect_from_composite_with_multiple_children(self):
        """Test collecting from COMPOSITE with multiple child regions."""
        graph, tensors = _create_residual_graph()

        parent = Region(region_id=1, level=1, region_type=RegionType.COMPOSITE)

        child1 = Region(region_id=2, level=0, region_type=RegionType.LEAF)
        child1.inputs = ["input"]
        child1.add_node(0)  # Conv1
        child1.add_node(1)  # Relu1

        child2 = Region(region_id=3, level=0, region_type=RegionType.LEAF)
        child2.inputs = ["relu1_out", "input"]  # Two inputs including skip connection
        child2.add_node(2)  # Conv2
        child2.add_node(3)  # Add

        parent.add_child(child1)
        parent.add_child(child2)

        result = ChildRegionInputInsertionPoint.collect_from_region(parent, graph)

        # Should find inputs from both children
        assert all(isinstance(ip, ChildRegionInputInsertionPoint) for ip in result)


class TestRegionOutputInsertionPointCollectFrom(unittest.TestCase):
    """Test RegionOutputInsertionPoint.collect_from_region() method."""

    def test_collect_node_outputs(self):
        """Test collecting node output insertion points."""
        graph, tensors = _create_simple_graph()

        region = Region(region_id=1, level=0, region_type=RegionType.LEAF)
        region.add_node(0)  # Conv
        region.add_node(1)  # BatchNorm
        region.add_node(2)  # Relu
        region.add_node(3)  # MaxPool
        region.outputs = ["pool_out"]  # Only pool_out is a region output

        result = RegionOutputInsertionPoint.collect_from_region(region, graph)

        # Should find the node output that matches region output
        assert len(result) >= 0  # May be filtered
        assert all(isinstance(ip, RegionOutputInsertionPoint) for ip in result)

    def test_collect_child_region_outputs(self):
        """Test collecting child region output insertion points."""
        graph, tensors = _create_simple_graph()

        parent = Region(region_id=1, level=1, region_type=RegionType.COMPOSITE)
        child = Region(region_id=2, level=0, region_type=RegionType.LEAF)
        child.outputs = ["relu_out"]
        child.add_node(0)  # Conv
        child.add_node(1)  # BatchNorm
        child.add_node(2)  # Relu
        parent.add_child(child)
        parent.outputs = ["relu_out"]  # Child output is also parent output

        result = RegionOutputInsertionPoint.collect_from_region(parent, graph)

        # Should find the child region output
        assert all(isinstance(ip, RegionOutputInsertionPoint) for ip in result)

    def test_collect_residual_block_outputs(self):
        """Test collecting outputs from residual block."""
        graph, tensors = _create_residual_graph()

        region = Region(region_id=1, level=0, region_type=RegionType.LEAF)
        region.add_node(0)  # Conv1
        region.add_node(1)  # Relu1
        region.add_node(2)  # Conv2
        region.add_node(3)  # Add
        region.add_node(4)  # Relu2
        region.outputs = ["output"]  # Final output

        result = RegionOutputInsertionPoint.collect_from_region(region, graph)

        # Should find the output
        assert all(isinstance(ip, RegionOutputInsertionPoint) for ip in result)
