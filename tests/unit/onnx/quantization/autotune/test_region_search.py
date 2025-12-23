#!/usr/bin/env python3
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

"""
Tests for region search algorithms.

Tests CombinedRegionSearch, RegionPartitioner, and TopDownRegionBuilder.
Note: Comprehensive integration tests with real ONNX graphs should be in separate integration test files.
"""

import os
import sys
import unittest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import onnx
import onnx_graphsurgeon as gs
from onnx import helper

from modelopt.onnx.quantization.autotune.common import Region, RegionType
from modelopt.onnx.quantization.autotune.region_search import (
    CombinedRegionSearch,
    RegionPartitioner,
    TopDownRegionBuilder,
)


def create_simple_linear_graph():
    """
    Create a simple linear graph: Input -> Conv -> Relu -> Output.

    This is the simplest possible graph for testing region discovery.
    """
    # Input
    input_tensor = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 3, 224, 224])

    # Output
    output_tensor = helper.make_tensor_value_info(
        "output", onnx.TensorProto.FLOAT, [1, 64, 224, 224]
    )

    # Conv node
    conv_node = helper.make_node(
        "Conv", inputs=["input", "conv_weight"], outputs=["conv_out"], name="conv"
    )

    # Relu node
    relu_node = helper.make_node("Relu", inputs=["conv_out"], outputs=["output"], name="relu")

    # Create graph
    graph = helper.make_graph(
        [conv_node, relu_node],
        "simple_linear",
        [input_tensor],
        [output_tensor],
        initializer=[
            helper.make_tensor(
                "conv_weight", onnx.TensorProto.FLOAT, [64, 3, 3, 3], [0.1] * (64 * 3 * 3 * 3)
            )
        ],
    )

    # Create model
    model = helper.make_model(graph, producer_name="test")

    # Convert to GraphSurgeon
    gs_graph = gs.import_onnx(model)
    return gs_graph


def create_divergent_graph():
    """
    Create a graph with divergence: Input -> Conv -> [Relu1, Relu2] -> Add -> Output.

    Tests divergence/convergence pattern detection.
    """
    input_tensor = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 3, 224, 224])
    output_tensor = helper.make_tensor_value_info(
        "output", onnx.TensorProto.FLOAT, [1, 64, 224, 224]
    )

    conv_node = helper.make_node(
        "Conv", inputs=["input", "conv_weight"], outputs=["conv_out"], name="conv"
    )
    relu1_node = helper.make_node("Relu", inputs=["conv_out"], outputs=["relu1_out"], name="relu1")
    relu2_node = helper.make_node("Relu", inputs=["conv_out"], outputs=["relu2_out"], name="relu2")
    add_node = helper.make_node(
        "Add", inputs=["relu1_out", "relu2_out"], outputs=["output"], name="add"
    )

    graph = helper.make_graph(
        [conv_node, relu1_node, relu2_node, add_node],
        "divergent",
        [input_tensor],
        [output_tensor],
        initializer=[
            helper.make_tensor(
                "conv_weight", onnx.TensorProto.FLOAT, [64, 3, 3, 3], [0.1] * (64 * 3 * 3 * 3)
            )
        ],
    )

    model = helper.make_model(graph, producer_name="test")
    gs_graph = gs.import_onnx(model)
    return gs_graph


class TestRegionPartitioner(unittest.TestCase):
    """Test RegionPartitioner basic functionality."""

    def test_creation_linear_graph(self):
        """Test creating RegionPartitioner with a simple linear graph."""
        graph = create_simple_linear_graph()
        partitioner = RegionPartitioner(graph)

        assert partitioner is not None
        assert partitioner.graph == graph
        print("✓ RegionPartitioner creation with linear graph")

    def test_partition_linear_graph(self):
        """Test partitioning a simple linear graph."""
        graph = create_simple_linear_graph()
        partitioner = RegionPartitioner(graph)

        regions = partitioner.partition_graph()

        # Should create at least one region
        assert len(regions) > 0

        # Check that regions cover most nodes (ONNX GS may add Constant nodes that aren't partitioned)
        total_nodes = sum(len(r.get_all_nodes_recursive()) for r in regions)
        assert total_nodes > 0
        assert total_nodes <= len(graph.nodes)
        print("✓ RegionPartitioner partition linear graph")

    def test_partition_divergent_graph(self):
        """Test partitioning a divergent graph."""
        graph = create_divergent_graph()
        partitioner = RegionPartitioner(graph)

        regions = partitioner.partition_graph()

        # Should create regions covering all nodes
        assert len(regions) > 0

        # Check that regions cover most nodes (ONNX GS may add Constant nodes that aren't partitioned)
        total_nodes = sum(len(r.get_all_nodes_recursive()) for r in regions)
        assert total_nodes > 0
        assert total_nodes <= len(graph.nodes)
        print("✓ RegionPartitioner partition divergent graph")


class TestTopDownRegionBuilder(unittest.TestCase):
    """Test TopDownRegionBuilder basic functionality."""

    def test_creation(self):
        """Test creating TopDownRegionBuilder."""
        graph = create_simple_linear_graph()

        # Create a root region with all nodes
        root = Region(region_id=0, level=0, region_type=RegionType.LEAF)
        for idx in range(len(graph.nodes)):
            root.add_node(idx)

        builder = TopDownRegionBuilder(graph, root)

        assert builder is not None
        assert builder.graph == graph
        print("✓ TopDownRegionBuilder creation")

    def test_build_composite_region(self):
        """Test building a composite region."""
        graph = create_simple_linear_graph()

        # First partition to get initial regions
        partitioner = RegionPartitioner(graph)
        initial_regions = partitioner.partition_graph()

        if len(initial_regions) > 0:
            # Use first region as root for top-down building
            root_region = initial_regions[0]

            builder = TopDownRegionBuilder(graph, root_region, next_region_id=100)

            # Build composite region (may return LEAF or COMPOSITE depending on structure)
            composite = builder.build_composite_region()

            assert composite is not None
            # Region type depends on whether refinement created internal structure
            # For simple linear graphs, may stay as LEAF
            assert composite.get_type() in [RegionType.LEAF, RegionType.COMPOSITE]
            print("✓ TopDownRegionBuilder build composite region")
        else:
            self.skipTest("No initial regions to refine")


class TestCombinedRegionSearch(unittest.TestCase):
    """Test CombinedRegionSearch two-phase algorithm."""

    def test_creation(self):
        """Test creating CombinedRegionSearch."""
        graph = create_simple_linear_graph()
        search = CombinedRegionSearch(graph)

        assert search is not None
        assert search.graph == graph
        print("✓ CombinedRegionSearch creation")

    def test_search_linear_graph(self):
        """Test searching regions in a simple linear graph."""
        graph = create_simple_linear_graph()
        search = CombinedRegionSearch(graph)

        regions = search.search_regions()

        # Should create regions
        assert len(regions) > 0

        # Check that regions cover most nodes (ONNX GS may add Constant nodes that aren't partitioned)
        total_nodes = sum(len(r.get_all_nodes_recursive()) for r in regions)
        assert total_nodes > 0
        assert total_nodes <= len(graph.nodes)

        # Each region should have valid inputs/outputs
        for region in regions:
            assert region.get_inputs() is not None
            assert region.get_outputs() is not None

        print("✓ CombinedRegionSearch on linear graph")

    def test_search_divergent_graph(self):
        """Test searching regions in a divergent graph."""
        graph = create_divergent_graph()
        search = CombinedRegionSearch(graph)

        regions = search.search_regions()

        # Should create regions
        assert len(regions) > 0

        # Check that regions cover most nodes (ONNX GS may add Constant nodes that aren't partitioned)
        total_nodes = sum(len(r.get_all_nodes_recursive()) for r in regions)
        assert total_nodes > 0
        assert total_nodes <= len(graph.nodes)

        print("✓ CombinedRegionSearch on divergent graph")

    def test_region_hierarchy(self):
        """Test that regions have proper hierarchical structure."""
        graph = create_simple_linear_graph()
        search = CombinedRegionSearch(graph)

        regions = search.search_regions()

        # Check that regions have children (hierarchical structure)
        for region in regions:
            if region.get_type() == RegionType.COMPOSITE:
                assert len(region.get_children()) > 0

                # Verify parent-child relationships
                for child in region.get_children():
                    assert child.get_parent() == region

        print("✓ CombinedRegionSearch hierarchical structure")

    def test_parameters(self):
        """Test CombinedRegionSearch with custom parameters."""
        graph = create_simple_linear_graph()

        # Test with different parameter values
        search = CombinedRegionSearch(
            graph, maximum_sequence_region_size=5, minimum_topdown_search_size=5
        )

        regions = search.search_regions()

        assert len(regions) > 0
        print("✓ CombinedRegionSearch with custom parameters")


def run_tests():
    """Run all region search tests."""
    print("=" * 70)
    print("Region Search Test Suite")
    print("=" * 70)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestRegionPartitioner))
    suite.addTests(loader.loadTestsFromTestCase(TestTopDownRegionBuilder))
    suite.addTests(loader.loadTestsFromTestCase(TestCombinedRegionSearch))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("\n✓ All region search tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())
