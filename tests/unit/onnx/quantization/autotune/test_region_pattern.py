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
Tests for RegionPattern functionality in the autotuner.

Tests pattern generation, matching, and tree visualization.
"""

import os
import sys
import unittest

import numpy as np
import onnx_graphsurgeon as gs

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modelopt.onnx.quantization.autotune.common import Region, RegionType
from modelopt.onnx.quantization.autotune.region_pattern import RegionPattern


class TestRegionPattern(unittest.TestCase):
    """Test RegionPattern functionality."""

    # =========================================================================
    # Helper Methods
    # =========================================================================

    @staticmethod
    def _create_simple_graph():
        """Create a simple Conv->Relu graph for testing.

        Graph structure:
            input -> Conv -> Relu -> output
        """
        # Create inputs and outputs
        inp = gs.Variable(name="input", dtype=np.float32, shape=[1, 3, 224, 224])
        conv_out = gs.Variable(name="conv_out", dtype=np.float32)
        relu_out = gs.Variable(name="output", dtype=np.float32)

        # Create weights
        conv_weight = gs.Constant(
            name="conv_weight", values=np.ones((64, 3, 3, 3), dtype=np.float32)
        )

        # Create nodes
        conv = gs.Node(
            name="Conv_0",
            op="Conv",
            inputs=[inp, conv_weight],
            outputs=[conv_out],
            attrs={"kernel_shape": [3, 3], "strides": [1, 1], "pads": [1, 1, 1, 1]},
        )
        relu = gs.Node(
            name="Relu_0",
            op="Relu",
            inputs=[conv_out],
            outputs=[relu_out],
        )

        # Create graph
        graph = gs.Graph(
            nodes=[conv, relu],
            inputs=[inp],
            outputs=[relu_out],
            opset=13,
        )
        return graph

    @staticmethod
    def _create_hierarchical_graph():
        """Create a hierarchical graph with composite regions.

        Graph structure:
            input -> Conv -> Relu -> Add -> MatMul -> Relu -> output
                               ^
                               |
                          other_input

        Region structure:
            ROOT
            ├── COMPOSITE (Conv->Relu->Add)
            │   ├── LEAF (Conv->Relu)
            │   └── LEAF (Add)
            └── COMPOSITE (MatMul->Relu)
                └── LEAF (MatMul->Relu)
        """
        # Create inputs and intermediate tensors
        inp = gs.Variable(name="input", dtype=np.float32, shape=[1, 64, 64, 64])
        other_inp = gs.Variable(name="other_input", dtype=np.float32, shape=[1, 64, 64, 64])
        conv_out = gs.Variable(name="conv_out", dtype=np.float32)
        relu1_out = gs.Variable(name="relu1_out", dtype=np.float32)
        add_out = gs.Variable(name="add_out", dtype=np.float32)
        matmul_out = gs.Variable(name="matmul_out", dtype=np.float32)
        output = gs.Variable(name="output", dtype=np.float32)

        # Create constants
        conv_weight = gs.Constant(
            name="conv_weight", values=np.ones((64, 64, 1, 1), dtype=np.float32)
        )
        matmul_weight = gs.Constant(
            name="matmul_weight", values=np.ones((64, 64), dtype=np.float32)
        )

        # Create nodes (order matters for node indices)
        conv = gs.Node(
            name="Conv_0",
            op="Conv",
            inputs=[inp, conv_weight],
            outputs=[conv_out],
            attrs={"kernel_shape": [1, 1]},
        )  # Node 0
        relu1 = gs.Node(name="Relu_0", op="Relu", inputs=[conv_out], outputs=[relu1_out])  # Node 1
        add = gs.Node(
            name="Add_0", op="Add", inputs=[relu1_out, other_inp], outputs=[add_out]
        )  # Node 2
        matmul = gs.Node(
            name="MatMul_0", op="MatMul", inputs=[add_out, matmul_weight], outputs=[matmul_out]
        )  # Node 3
        relu2 = gs.Node(name="Relu_1", op="Relu", inputs=[matmul_out], outputs=[output])  # Node 4

        # Create graph
        graph = gs.Graph(
            nodes=[conv, relu1, add, matmul, relu2],
            inputs=[inp, other_inp],
            outputs=[output],
            opset=13,
        )
        return graph

    @staticmethod
    def _create_test_region(
        region_id: int, level: int, region_type: RegionType, node_indices: list[int] | None = None
    ) -> Region:
        """Create a test region."""
        region = Region(region_id, level, region_type)
        if node_indices:
            region.add_nodes(node_indices)
        return region

    # =========================================================================
    # Test Cases
    # =========================================================================

    def test_pattern_creation(self):
        """Test basic RegionPattern creation."""
        pattern = RegionPattern(signature="Conv->Relu", size=2)

        assert pattern.signature == "Conv->Relu"
        assert pattern.size == 2
        assert not pattern.is_empty
        assert pattern.is_leaf
        assert not pattern.is_composite

    def test_pattern_equality(self):
        """Test RegionPattern equality based on signature."""
        pattern1 = RegionPattern(signature="Conv->Relu", size=2)
        pattern2 = RegionPattern(signature="Conv->Relu", size=5)  # Different size
        pattern3 = RegionPattern(signature="Gemm->Relu", size=2)

        # Same signature = equal (size doesn't affect equality)
        assert pattern1 == pattern2
        # Different signature = not equal
        assert pattern1 != pattern3

    def test_pattern_hash(self):
        """Test RegionPattern hashing for use in dicts/sets."""
        pattern1 = RegionPattern(signature="Conv->Relu", size=2)
        pattern2 = RegionPattern(signature="Conv->Relu", size=5)

        # Same signature = same hash
        assert hash(pattern1) == hash(pattern2)

        # Can be used as dict keys
        pattern_dict = {pattern1: "scheme1"}
        assert pattern_dict[pattern2] == "scheme1"  # pattern2 finds pattern1's entry

    def test_pattern_from_simple_region(self):
        """Test pattern computation from a simple region."""
        graph = self._create_simple_graph()

        # Create a leaf region with Conv and Relu nodes
        region = self._create_test_region(
            region_id=1, level=0, region_type=RegionType.LEAF, node_indices=[0, 1]
        )

        pattern = RegionPattern.from_region(region, graph)

        # Should capture both operations
        assert "Conv" in pattern.signature
        assert "Relu" in pattern.signature
        assert pattern.size == 2
        assert pattern.is_leaf

    def test_pattern_from_composite_region(self):
        """Test pattern computation from a composite region with children."""
        graph = self._create_hierarchical_graph()

        # Create leaf regions
        leaf1 = self._create_test_region(
            region_id=1,
            level=0,
            region_type=RegionType.LEAF,
            node_indices=[0, 1],  # Conv, Relu
        )
        leaf2 = self._create_test_region(
            region_id=2,
            level=0,
            region_type=RegionType.LEAF,
            node_indices=[2],  # Add
        )

        # Create composite region
        composite = self._create_test_region(
            region_id=3, level=1, region_type=RegionType.COMPOSITE, node_indices=[]
        )
        composite.add_child(leaf1)
        composite.add_child(leaf2)

        pattern = RegionPattern.from_region(composite, graph)

        assert pattern.is_composite
        assert "COMPOSITE" in pattern.signature
        assert pattern.size == 3  # Total nodes in region hierarchy

    def test_pattern_get_hash(self):
        """Test cryptographic hash generation."""
        pattern = RegionPattern(signature="Conv->Relu", size=2)
        hash_val = pattern.get_hash()

        # Hash should be 32 hex characters (128-bit truncated SHA-256)
        assert len(hash_val) == 32
        assert all(c in "0123456789abcdef" for c in hash_val)

        # Same signature = same hash
        pattern2 = RegionPattern(signature="Conv->Relu", size=5)
        assert pattern.get_hash() == pattern2.get_hash()

    def test_pattern_get_short_signature(self):
        """Test signature truncation."""
        long_sig = "COMPOSITE(" + "Conv->Relu->" * 20 + "Output)"
        pattern = RegionPattern(signature=long_sig, size=20)

        short_sig = pattern.get_short_signature(max_length=50)
        assert len(short_sig) == 50
        assert short_sig.endswith("...")

        # Short signature stays unchanged
        short_pattern = RegionPattern(signature="Conv", size=1)
        assert short_pattern.get_short_signature(max_length=50) == "Conv"

    def test_print_tree(self):
        """Test format_tree to visualize region structure.

        This test demonstrates how to use format_tree to display
        the hierarchical structure of regions and their patterns.
        """
        graph = self._create_hierarchical_graph()

        # Build a hierarchical region structure:
        #   ROOT (level=2)
        #   ├── COMPOSITE (level=1) [Conv->Relu + Add]
        #   │   ├── LEAF (level=0) [Conv, Relu - nodes 0,1]
        #   │   └── LEAF (level=0) [Add - node 2]
        #   └── LEAF (level=0) [MatMul, Relu - nodes 3,4]

        # Create leaf regions
        leaf_conv_relu = self._create_test_region(
            region_id=1, level=0, region_type=RegionType.LEAF, node_indices=[0, 1]
        )
        leaf_add = self._create_test_region(
            region_id=2, level=0, region_type=RegionType.LEAF, node_indices=[2]
        )
        leaf_matmul_relu = self._create_test_region(
            region_id=3, level=0, region_type=RegionType.LEAF, node_indices=[3, 4]
        )

        # Create composite region containing conv_relu and add
        composite = self._create_test_region(
            region_id=4, level=1, region_type=RegionType.COMPOSITE, node_indices=[]
        )
        composite.add_child(leaf_conv_relu)
        composite.add_child(leaf_add)

        # Create root region containing everything
        root = self._create_test_region(
            region_id=5, level=2, region_type=RegionType.ROOT, node_indices=[]
        )
        root.add_child(composite)
        root.add_child(leaf_matmul_relu)

        # Generate pattern for root and print tree
        root_pattern = RegionPattern.from_region(root, graph)
        tree_output = root_pattern.format_tree(root, graph)

        print("\n" + "=" * 60)
        print("Region Tree Structure:")
        print("=" * 60)
        print(tree_output)
        print("=" * 60)

        # Verify tree output contains expected elements
        assert "Region 5" in tree_output  # Root
        assert "Region 4" in tree_output  # Composite
        assert "Region 1" in tree_output  # Leaf conv_relu
        assert "Region 2" in tree_output  # Leaf add
        assert "Region 3" in tree_output  # Leaf matmul_relu

        # Verify indentation shows hierarchy
        lines = tree_output.strip().split("\n")
        assert len(lines) >= 3  # At least root + children

        # Root should have no indentation
        assert lines[0].startswith("Region 5")

        # Children should be indented
        indented_lines = [line for line in lines if line.startswith("  ")]
        assert len(indented_lines) > 0

    def test_pattern_matches_pattern(self):
        """Test pattern-to-pattern matching."""
        pattern1 = RegionPattern(signature="Conv->Relu", size=2)
        pattern2 = RegionPattern(signature="Conv->Relu", size=5)
        pattern3 = RegionPattern(signature="Gemm->Relu", size=2)

        assert pattern1.matches(pattern2)  # Same signature
        assert not pattern1.matches(pattern3)  # Different signature

    def test_pattern_matches_region(self):
        """Test pattern-to-region matching."""
        graph = self._create_simple_graph()

        # Create region
        region = self._create_test_region(
            region_id=1, level=0, region_type=RegionType.LEAF, node_indices=[0, 1]
        )

        # Create pattern from region
        pattern = RegionPattern.from_region(region, graph)

        # Match should return node IDs
        node_ids = pattern.matches(region, graph)
        assert node_ids is not None
        assert set(node_ids) == {0, 1}

    def test_empty_region_pattern(self):
        """Test pattern for empty region."""
        graph = self._create_simple_graph()

        # Create empty region
        empty_region = self._create_test_region(
            region_id=1, level=0, region_type=RegionType.LEAF, node_indices=[]
        )

        pattern = RegionPattern.from_region(empty_region, graph)

        assert pattern.is_empty
        assert pattern.signature == "EMPTY"
        assert pattern.size == 0

    def test_symmetric_operation_signature(self):
        """Test that symmetric operations (Add, Mul) have consistent signatures."""
        # Create two graphs with Add inputs in different order
        inp1 = gs.Variable(name="input1", dtype=np.float32, shape=[1, 64])
        inp2 = gs.Variable(name="input2", dtype=np.float32, shape=[1, 64])
        out = gs.Variable(name="output", dtype=np.float32)

        # Graph 1: Add(inp1, inp2)
        add1 = gs.Node(name="Add_0", op="Add", inputs=[inp1, inp2], outputs=[out])
        graph1 = gs.Graph(nodes=[add1], inputs=[inp1, inp2], outputs=[out], opset=13)

        # Graph 2: Add(inp2, inp1) - reversed inputs
        add2 = gs.Node(name="Add_0", op="Add", inputs=[inp2, inp1], outputs=[out])
        graph2 = gs.Graph(nodes=[add2], inputs=[inp1, inp2], outputs=[out], opset=13)

        # Create regions
        region1 = self._create_test_region(1, 0, RegionType.LEAF, [0])
        region2 = self._create_test_region(1, 0, RegionType.LEAF, [0])

        pattern1 = RegionPattern.from_region(region1, graph1)
        pattern2 = RegionPattern.from_region(region2, graph2)

        # Patterns should be equal regardless of input order
        assert pattern1 == pattern2

    def test_pattern_repr_and_str(self):
        """Test string representations."""
        pattern = RegionPattern(signature="Conv->Relu", size=2)

        # str() shows just signature
        assert str(pattern) == "Conv->Relu"

        # repr() shows full info
        assert "RegionPattern" in repr(pattern)
        assert "Conv->Relu" in repr(pattern)
        assert "size=2" in repr(pattern)
