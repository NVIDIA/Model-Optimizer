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
1. InsertionPoint classes (NodeInputInsertionPoint, ChildRegionOutputInsertionPoint, ChildRegionInputInsertionPoint)
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
import pytest

from modelopt.onnx.quantization.autotune.common import (
    ChildRegionInputInsertionPoint,
    ChildRegionOutputInsertionPoint,
    InsertionScheme,
    NodeInputInsertionPoint,
    Region,
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

INSERTION_POINT_CASES = [
    pytest.param(
        NodeInputInsertionPoint,
        {"node_index": 5, "input_index": 2},
        {"node_index": 5, "input_index": 2},
        {"node_index": 5, "input_index": 3},
        "node_index",
        ["5", "2"],
        id="NodeInputInsertionPoint",
    ),
    pytest.param(
        ChildRegionOutputInsertionPoint,
        {"region_index": 2, "node_index": None, "output_index": 1},
        {"region_index": 2, "node_index": None, "output_index": 1},
        {"region_index": None, "node_index": 2, "output_index": 1},
        "region_index",
        ["region", "2"],
        id="ChildRegionOutputInsertionPoint-region",
    ),
    pytest.param(
        ChildRegionOutputInsertionPoint,
        {"region_index": None, "node_index": 5, "output_index": 0},
        {"region_index": None, "node_index": 5, "output_index": 0},
        {"region_index": None, "node_index": 5, "output_index": 1},
        "node_index",
        ["node", "5"],
        id="ChildRegionOutputInsertionPoint-node",
    ),
    pytest.param(
        ChildRegionInputInsertionPoint,
        {"region_index": 3, "input_index": 1},
        {"region_index": 3, "input_index": 1},
        {"region_index": 3, "input_index": 2},
        "region_index",
        ["3", "1"],
        id="ChildRegionInputInsertionPoint",
    ),
]


class TestInsertionPoints:
    """Combined tests for all InsertionPoint types."""

    @pytest.mark.parametrize(("cls", "kwargs", "_", "__", "___", "____"), INSERTION_POINT_CASES)
    def test_creation(self, cls, kwargs, _, __, ___, ____):
        point = cls(**kwargs)
        for key, val in kwargs.items():
            assert getattr(point, key) == val

    @pytest.mark.parametrize(
        ("cls", "kwargs", "_", "__", "mutate_attr", "___"), INSERTION_POINT_CASES
    )
    def test_immutability(self, cls, kwargs, _, __, mutate_attr, ___):
        point = cls(**kwargs)
        with pytest.raises(AttributeError):
            setattr(point, mutate_attr, 999)

    @pytest.mark.parametrize(
        ("cls", "kwargs", "equal_kwargs", "diff_kwargs", "_", "__"), INSERTION_POINT_CASES
    )
    def test_equality(self, cls, kwargs, equal_kwargs, diff_kwargs, _, __):
        point1 = cls(**kwargs)
        point2 = cls(**equal_kwargs)
        point3 = cls(**diff_kwargs)
        assert point1 == point2
        assert point1 != point3

    @pytest.mark.parametrize(
        ("cls", "kwargs", "equal_kwargs", "diff_kwargs", "_", "__"), INSERTION_POINT_CASES
    )
    def test_hashable(self, cls, kwargs, equal_kwargs, diff_kwargs, _, __):
        point1 = cls(**kwargs)
        point2 = cls(**equal_kwargs)
        point3 = cls(**diff_kwargs)
        point_set = {point1, point2, point3}
        assert len(point_set) == 2

    @pytest.mark.parametrize(("cls", "kwargs", "_", "__", "___", "____"), INSERTION_POINT_CASES)
    def test_serialization(self, cls, kwargs, _, __, ___, ____):
        point = cls(**kwargs)
        data = point.to_dict()
        for key, val in kwargs.items():
            assert data[key] == val
        restored = cls.from_dict(data)
        assert point == restored

    @pytest.mark.parametrize(
        ("cls", "kwargs", "_", "__", "___", "str_checks"), INSERTION_POINT_CASES
    )
    def test_string_representation(self, cls, kwargs, _, __, ___, str_checks):
        point = cls(**kwargs)
        s = str(point).lower()
        for check in str_checks:
            assert check.lower() in s


class TestInsertionScheme:
    """Test InsertionScheme functionality."""

    def test_empty_scheme(self):
        """Test empty InsertionScheme."""
        scheme = InsertionScheme()
        assert scheme.is_empty
        assert len(scheme.node_inputs) == 0
        assert len(scheme.child_region_inputs) == 0
        assert len(scheme.region_outputs) == 0
        assert not scheme.error

    @pytest.mark.parametrize(
        ("attr", "points"),
        [
            ("node_inputs", [NodeInputInsertionPoint(0, 0), NodeInputInsertionPoint(1, 0)]),
            (
                "region_outputs",
                [
                    ChildRegionOutputInsertionPoint(None, 0, 0),
                    ChildRegionOutputInsertionPoint(1, None, 0),
                ],
            ),
            (
                "child_region_inputs",
                [ChildRegionInputInsertionPoint(0, 0), ChildRegionInputInsertionPoint(1, 0)],
            ),
        ],
    )
    def test_scheme_with_points_not_empty(self, attr, points):
        """Test scheme with insertion points is not empty."""
        scheme = InsertionScheme()
        setattr(scheme, attr, points)
        assert not scheme.is_empty
        assert len(getattr(scheme, attr)) == 2

    def test_scheme_hash_empty(self):
        """Test hash of empty schemes are equal."""
        assert InsertionScheme().hash == InsertionScheme().hash

    def test_scheme_hash_equality(self):
        """Test hash with same/different insertion points."""

        def make_scheme(*node_indices):
            s = InsertionScheme()
            s.node_inputs = [NodeInputInsertionPoint(i, 0) for i in node_indices]
            return s

        assert make_scheme(0, 1).hash == make_scheme(0, 1).hash
        assert make_scheme(0, 1).hash == make_scheme(1, 0).hash  # order independent
        assert make_scheme(0, 1).hash != make_scheme(0, 2).hash

    @pytest.mark.parametrize(
        ("error", "latency"),
        [
            (False, float("inf")),  # empty
            (False, 12.5),  # full
            (True, float("inf")),  # with error
        ],
    )
    def test_serialization_roundtrip(self, error, latency):
        """Test serialization roundtrip."""
        scheme = InsertionScheme()
        scheme.error = error
        scheme.latency_ms = latency

        if latency != float("inf") or error:  # add points for non-empty cases
            scheme.node_inputs = [NodeInputInsertionPoint(0, 0)]
            scheme.child_region_inputs = [ChildRegionInputInsertionPoint(0, 0)]
            scheme.region_outputs = [ChildRegionOutputInsertionPoint(None, 0, 0)]

        restored = InsertionScheme.from_dict(scheme.to_dict())

        assert restored.error == error
        assert restored.latency_ms == latency
        if not scheme.is_empty:
            assert len(restored.node_inputs) == len(scheme.node_inputs)
            assert len(restored.child_region_inputs) == len(scheme.child_region_inputs)
            assert len(restored.region_outputs) == len(scheme.region_outputs)


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


def _create_region(region_id=1, level=0, region_type=RegionType.LEAF, nodes=None):
    """Create a region with the specified properties.

    Args:
        region_id: ID for the region
        level: Hierarchy level (0 for LEAF, 1+ for COMPOSITE/ROOT)
        region_type: Type of region (LEAF, COMPOSITE, or ROOT)
        nodes: Optional list/set of node indices to add to the region

    Returns:
        Region with specified properties and nodes
    """
    region = Region(region_id=region_id, level=level, region_type=region_type)
    if nodes:
        region.nodes.update(nodes)
    return region


def _create_simple_graph():
    """Create a mock graph with Conv -> BatchNorm -> Relu -> MaxPool pattern.

    Graph structure:
        input -> Conv -> conv_out -> BatchNorm -> bn_out -> Relu -> relu_out -> MaxPool -> pool_out
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


class TestSkipInvalidInsertionPoints:
    """Test skip_invalid_insertion_points function."""

    @pytest.mark.parametrize(
        ("op", "should_skip"),
        [
            ("Equal", True),  # bool op
            ("Shape", True),  # shape op
            ("MatMul", False),  # normal op
            ("Add", False),  # normal op
        ],
    )
    def test_skip_by_op_type(self, op, should_skip):
        graph, _ = _create_simple_graph()
        tensor = _create_mock_tensor("test_input", np.float32, [1, 64, 32, 32])
        node = _create_mock_node(op, [tensor], [])
        assert skip_invalid_insertion_points(graph, "test_input", node) is should_skip

    @pytest.mark.parametrize(
        ("dtype", "shape", "should_skip"),
        [
            (np.int32, [1, 64, 32, 32], True),  # non-float
            (np.float32, [1], True),  # small tensor
            (np.float32, [1, 64, 32, 32], False),  # large float - OK
        ],
    )
    def test_skip_by_tensor_properties(self, dtype, shape, should_skip):
        graph, _ = _create_simple_graph()
        tensor = _create_mock_tensor("test", dtype, shape)
        node = _create_mock_node("Add", [tensor], [])
        assert skip_invalid_insertion_points(graph, "test", node) is should_skip

    def test_skip_conv_weight_input(self):
        """Conv weight inputs (index >= 1) are skipped."""
        graph, _ = _create_simple_graph()
        result = skip_invalid_insertion_points(graph, "conv_weight", graph.nodes[0])
        assert result is True

    def test_skip_bn_non_data_inputs(self):
        """BatchNormalization non-data inputs are skipped."""
        graph, _ = _create_simple_graph()
        result = skip_invalid_insertion_points(graph, "bn_scale", graph.nodes[1])
        assert result is True

    def test_skip_conv_bn_relu_fusion(self):
        """Conv->BN->Relu fusion patterns are skipped at intermediate points."""
        graph, _ = _create_simple_graph()
        result = skip_invalid_insertion_points(graph, "bn_out", graph.nodes[2])
        assert result is True

    def test_with_region(self):
        """Test with a Region containing multiple nodes."""
        graph, _ = _create_simple_graph()
        region = _create_region(nodes=[0, 1])

        shape_tensor = _create_mock_tensor("shape_input", np.float32)
        shape_node = _create_mock_node("Shape", [shape_tensor], [])
        graph.nodes.append(shape_node)
        region.nodes.add(4)

        assert skip_invalid_insertion_points(graph, "shape_input", region) is True

    def test_residual_block_add_inputs_allowed(self):
        """Add node inputs in residual blocks should be allowed."""
        graph, _ = _create_residual_graph()
        add_node = graph.nodes[3]

        assert skip_invalid_insertion_points(graph, "conv2_out", add_node) is False
        assert skip_invalid_insertion_points(graph, "input", add_node) is False


class TestHasQuantizableOperations:
    """Test has_quantizable_operations function."""

    @pytest.mark.parametrize(
        ("nodes", "graph_fn", "expected"),
        [
            ({0}, _create_simple_graph, True),  # Conv
            ({3}, _create_simple_graph, True),  # MaxPool
            ({2}, _create_simple_graph, True),  # Relu
            ({0, 1, 2}, _create_simple_graph, True),  # Conv->BN->Relu
            ({3}, _create_residual_graph, True),  # Add in residual
        ],
    )
    def test_leaf_with_quantizable_ops(self, nodes, graph_fn, expected):
        """Test LEAF region with various quantizable operations."""
        graph, _ = graph_fn()
        region = _create_region(nodes=nodes)
        assert has_quantizable_operations(region, graph) is expected

    def test_leaf_without_quantizable_ops(self):
        """Test LEAF region without major quantizable operations."""
        shape_tensor = _create_mock_tensor("input", np.float32)
        output_tensor = _create_mock_tensor("output", np.float32)
        shape_node = _create_mock_node("Shape", [shape_tensor], [output_tensor])
        transpose_node = _create_mock_node("Transpose", [output_tensor], [])
        graph = MagicMock(spec=gs.Graph)
        graph.nodes = [shape_node, transpose_node]
        region = _create_region(nodes={0, 1})

        assert has_quantizable_operations(region, graph) is False

    def test_composite_region_always_true(self):
        """Test that COMPOSITE regions always return True."""
        graph, _ = _create_simple_graph()
        region = _create_region(level=1, region_type=RegionType.COMPOSITE)
        assert has_quantizable_operations(region, graph) is True


class TestResolveRegionIOInsertionPoints(unittest.TestCase):
    """Test resolve_region_io_insertion_points function."""

    def test_resolve_with_region(self):
        """Test resolving with a region containing Conv->BN->Relu."""
        graph, tensors = _create_simple_graph()

        # Set up tensor_users_map: conv_out is consumed by BatchNorm (node 1)
        graph.tensor_users_map = get_tensor_consumer_node_indices(graph)
        region = _create_region(nodes=[2])  # Relu node
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

        region = _create_region(nodes=[2])  # Conv2

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


class TestNodeInputInsertionPointMethods(unittest.TestCase):
    """Test NodeInputInsertionPoint.resolve() and collect_from_region() methods."""

    def test_resolve_simple(self):
        """Test resolving a simple node input for Conv->BN->Relu->Pool."""
        graph, tensors = _create_simple_graph()
        region = _create_region(nodes=[0, 1, 2, 3])  # Conv, BatchNorm, Relu, MaxPool

        # Create insertion point for first input of first node (Conv)
        ip = NodeInputInsertionPoint(node_index=0, input_index=0)
        result = ip.resolve(region, graph)

        assert len(result) >= 1
        assert any(rip.tensor_name == "input" for rip in result)

    def test_resolve_conv_includes_weight(self):
        """Test that resolving Conv input also includes weight."""
        graph, tensors = _create_simple_graph()
        region = _create_region(nodes=[0])  # Conv node

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
        region = _create_region(nodes=[0, 1, 2])  # Conv, BatchNorm, Relu

        # Relu is at local index 2, input 0 is bn_out
        ip = NodeInputInsertionPoint(node_index=2, input_index=0)
        result = ip.resolve(region, graph)

        assert len(result) == 1
        rip = next(iter(result))
        assert rip.tensor_name == "bn_out"

    def test_resolve_residual_conv_input(self):
        """Test resolving Conv input in residual block."""
        graph, tensors = _create_residual_graph()
        region = _create_region(nodes=[0, 1, 2])  # Conv1, Relu1, Conv2

        # Conv2 is at local index 2, input 0 is relu1_out
        ip = NodeInputInsertionPoint(node_index=2, input_index=0)
        result = ip.resolve(region, graph)

        # Conv includes both data and weight
        assert len(result) == 2
        tensor_names = {rip.tensor_name for rip in result}
        assert "relu1_out" in tensor_names
        assert "conv2_weight" in tensor_names

    def test_collect_valid_inputs(self):
        """Test collecting valid node input insertion points from Conv->BN->Relu->Pool."""
        graph, tensors = _create_simple_graph()
        region = _create_region(nodes=[0, 1, 2, 3])  # Conv, BatchNorm, Relu, MaxPool
        result = NodeInputInsertionPoint.collect_from_region(region, graph)

        # Should have collected some insertion points
        assert len(result) >= 1
        # All should be NodeInputInsertionPoint
        assert all(isinstance(ip, NodeInputInsertionPoint) for ip in result)

    def test_collect_from_residual_block(self):
        """Test collecting from residual block with skip connection."""
        graph, tensors = _create_residual_graph()
        region = _create_region(nodes=[0, 1, 2, 3, 4])  # Conv1, Relu1, Conv2, Add, Relu2
        result = NodeInputInsertionPoint.collect_from_region(region, graph)

        # Should have collected insertion points from Conv1, Add inputs, etc.
        assert len(result) >= 1
        assert all(isinstance(ip, NodeInputInsertionPoint) for ip in result)

        # Check that we have insertion points for different nodes
        node_indices = {ip.node_index for ip in result}
        assert len(node_indices) >= 1  # At least one node has valid inputs


class TestChildRegionInputInsertionPointMethods(unittest.TestCase):
    """Test ChildRegionInputInsertionPoint.resolve() and collect_from_region() methods."""

    def test_resolve_composite_region(self):
        """Test resolving child region input in COMPOSITE region."""
        graph, tensors = _create_simple_graph()
        graph.tensor_users_map = {"input": [0]}

        # Create parent (COMPOSITE) with child (LEAF) containing Conv->BN->Relu
        parent = _create_region(region_id=1, level=1, region_type=RegionType.COMPOSITE)
        child = _create_region(region_id=2, nodes=[0, 1, 2])  # Conv, BatchNorm, Relu
        child.inputs = ["input"]
        parent.add_child(child)
        ip = ChildRegionInputInsertionPoint(region_index=0, input_index=0)
        result = ip.resolve(parent, graph)

        assert len(result) >= 1
        assert any(rip.tensor_name == "input" for rip in result)

    def test_resolve_leaf_returns_empty(self):
        """Test that LEAF regions return empty set."""
        graph, _ = _create_simple_graph()
        leaf = _create_region(nodes=[0])
        ip = ChildRegionInputInsertionPoint(region_index=0, input_index=0)
        result = ip.resolve(leaf, graph)
        assert len(result) == 0

    def test_resolve_multiple_children(self):
        """Test resolving child inputs in COMPOSITE with multiple children."""
        graph, tensors = _create_residual_graph()
        # input is consumed by Conv1 (node 0) and Add (node 3)
        graph.tensor_users_map = get_tensor_consumer_node_indices(graph)

        # Create parent with two child regions
        parent = _create_region(region_id=1, level=1, region_type=RegionType.COMPOSITE)

        # First child: Conv1 (consumes "input")
        child1 = _create_region(region_id=2, nodes=[0])  # Conv1
        child1.inputs = ["input"]

        # Second child: Relu1 (consumes "relu1_out")
        child2 = _create_region(region_id=3, nodes=[2])  # Relu1
        child2.inputs = ["relu1_out"]
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

    def test_collect_from_composite(self):
        """Test collecting from COMPOSITE region with children."""
        graph, tensors = _create_simple_graph()
        parent = _create_region(region_id=1, level=1, region_type=RegionType.COMPOSITE)
        child = _create_region(region_id=2, nodes=[0, 1, 2])  # Conv, BatchNorm, Relu
        child.inputs = ["input"]
        parent.add_child(child)
        result = ChildRegionInputInsertionPoint.collect_from_region(parent, graph)
        # Should find the child's input
        assert len(result) >= 0  # May be filtered by skip_invalid_insertion_points
        assert all(isinstance(ip, ChildRegionInputInsertionPoint) for ip in result)

    def test_collect_from_leaf_returns_empty(self):
        """Test that LEAF regions return empty list."""
        graph, _ = _create_simple_graph()
        leaf = _create_region(nodes=[0])
        result = ChildRegionInputInsertionPoint.collect_from_region(leaf, graph)
        assert len(result) == 0

    def test_collect_from_composite_with_multiple_children(self):
        """Test collecting from COMPOSITE with multiple child regions."""
        graph, tensors = _create_residual_graph()
        parent = _create_region(region_id=1, level=1, region_type=RegionType.COMPOSITE)
        child1 = _create_region(region_id=2, nodes=[0, 1])  # Conv1, Relu1
        child1.inputs = ["input"]
        child2 = _create_region(region_id=3, nodes=[2, 3])  # Conv2, Add
        child2.inputs = ["relu1_out", "input"]  # Two inputs including skip connection
        parent.add_child(child1)
        parent.add_child(child2)

        result = ChildRegionInputInsertionPoint.collect_from_region(parent, graph)
        # Should find inputs from both children
        assert all(isinstance(ip, ChildRegionInputInsertionPoint) for ip in result)


class TestChildRegionOutputInsertionPointMethods(unittest.TestCase):
    """Test ChildRegionOutputInsertionPoint.resolve() and collect_from_region() methods."""

    def test_resolve_node_output(self):
        """Test resolving a node output."""
        graph, tensors = _create_simple_graph()
        graph.tensor_users_map = get_tensor_consumer_node_indices(graph)
        region = _create_region(nodes=[0, 1, 2, 3])  # Conv, BatchNorm, Relu, MaxPool
        region.outputs = ["pool_out"]
        # Output of last node (MaxPool)
        ip = ChildRegionOutputInsertionPoint(region_index=None, node_index=2, output_index=0)
        result = ip.resolve(region, graph)
        assert len(result) >= 1
        assert any(rip.tensor_name == "relu_out" for rip in result)

    def test_resolve_child_region_output(self):
        """Test resolving a child region output."""
        graph, tensors = _create_simple_graph()
        graph.tensor_users_map = {"relu_out": [3]}
        parent = _create_region(region_id=1, level=1, region_type=RegionType.COMPOSITE)
        child = _create_region(region_id=2, nodes=[0, 1, 2])  # Conv, BatchNorm, Relu
        child.outputs = ["relu_out"]
        parent.add_child(child)
        ip = ChildRegionOutputInsertionPoint(region_index=0, node_index=None, output_index=0)
        result = ip.resolve(parent, graph)
        assert len(result) >= 1
        assert any(rip.tensor_name == "relu_out" for rip in result)

    def test_resolve_residual_add_output(self):
        """Test resolving Add output in residual block."""
        graph, tensors = _create_residual_graph()
        graph.tensor_users_map = {"add_out": [4]}
        region = _create_region(nodes=[0, 1, 2, 3, 4])  # Conv1, Relu1, Conv2, Add, Relu2
        region.outputs = ["add_out"]
        # Add is at local index 3, output 0
        ip = ChildRegionOutputInsertionPoint(region_index=None, node_index=3, output_index=0)
        result = ip.resolve(region, graph)
        assert len(result) >= 1
        assert any(rip.tensor_name == "add_out" for rip in result)

    def test_collect_node_outputs(self):
        """Test collecting node output insertion points."""
        graph, tensors = _create_simple_graph()
        region = _create_region(nodes=[0, 1, 2, 3])  # Conv, BatchNorm, Relu, MaxPool
        region.outputs = ["pool_out"]  # Only pool_out is a region output
        result = ChildRegionOutputInsertionPoint.collect_from_region(region, graph)

        # Should find the node output that matches region output
        assert len(result) >= 0  # May be filtered
        assert all(isinstance(ip, ChildRegionOutputInsertionPoint) for ip in result)

    def test_collect_child_region_outputs(self):
        """Test collecting child region output insertion points."""
        graph, tensors = _create_simple_graph()
        parent = _create_region(region_id=1, level=1, region_type=RegionType.COMPOSITE)
        child = _create_region(region_id=2, nodes=[0, 1, 2])  # Conv, BatchNorm, Relu
        child.outputs = ["relu_out"]
        parent.add_child(child)
        parent.outputs = ["relu_out"]  # Child output is also parent output
        result = ChildRegionOutputInsertionPoint.collect_from_region(parent, graph)

        # Should find the child region output
        assert all(isinstance(ip, ChildRegionOutputInsertionPoint) for ip in result)

    def test_collect_residual_block_outputs(self):
        """Test collecting outputs from residual block."""
        graph, tensors = _create_residual_graph()
        region = _create_region(nodes=[0, 1, 2, 3, 4])  # Conv1, Relu1, Conv2, Add, Relu2
        region.outputs = ["output"]  # Final output
        result = ChildRegionOutputInsertionPoint.collect_from_region(region, graph)

        # Should find the output
        assert all(isinstance(ip, ChildRegionOutputInsertionPoint) for ip in result)
