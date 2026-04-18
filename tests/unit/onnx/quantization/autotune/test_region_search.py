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
Tests for region search algorithms.

Tests CombinedRegionSearch, RegionPartitioner, and TopDownRegionBuilder.
Note: Comprehensive integration tests with real ONNX graphs should be in separate integration test files.
"""

import io

import onnx
import onnx_graphsurgeon as gs
import pytest
from onnx import helper

from modelopt.onnx.quantization.autotune.common import Region, RegionType
from modelopt.onnx.quantization.autotune.region_search import (
    CombinedRegionSearch,
    RegionPartitioner,
    TopDownRegionBuilder,
)


@pytest.fixture
def simple_linear_graph():
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
    return gs.import_onnx(model)


@pytest.fixture
def divergent_graph():
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
    return gs.import_onnx(model)


class TestRegionPartitioner:
    """Test RegionPartitioner basic functionality."""

    def test_partition_linear_graph(self, simple_linear_graph):
        """Test partitioning a simple linear graph."""
        partitioner = RegionPartitioner(simple_linear_graph)

        regions = partitioner.partition_graph()

        # Should create at least one region
        assert len(regions) > 0

        # Check that regions cover most nodes (ONNX GS may add Constant nodes that aren't partitioned)
        total_nodes = sum(len(r.get_region_nodes_and_descendants()) for r in regions)
        assert total_nodes > 0
        assert total_nodes <= len(simple_linear_graph.nodes)

    def test_partition_divergent_graph(self, divergent_graph):
        """Test partitioning a divergent graph."""
        partitioner = RegionPartitioner(divergent_graph)

        regions = partitioner.partition_graph()

        # Should create regions covering all nodes
        assert len(regions) > 0

        # Check that regions cover most nodes (ONNX GS may add Constant nodes that aren't partitioned)
        total_nodes = sum(len(r.get_region_nodes_and_descendants()) for r in regions)
        assert total_nodes > 0
        assert total_nodes <= len(divergent_graph.nodes)


class TestTopDownRegionBuilder:
    """Test TopDownRegionBuilder basic functionality."""

    def test_build_composite_region(self, simple_linear_graph):
        """Test building a composite region."""
        # First partition to get initial regions
        partitioner = RegionPartitioner(simple_linear_graph)
        initial_regions = partitioner.partition_graph()

        if len(initial_regions) > 0:
            # Use first region as root for top-down building
            root_region = initial_regions[0]

            builder = TopDownRegionBuilder(simple_linear_graph, root_region, next_region_id=100)

            # Build composite region (may return LEAF or COMPOSITE depending on structure)
            composite = builder.build_composite_region()

            assert composite is not None
            # Region type depends on whether refinement created internal structure
            # For simple linear graphs, may stay as LEAF
            assert composite.type in [RegionType.LEAF, RegionType.COMPOSITE]
        else:
            pytest.skip("No initial regions to refine")


class TestCombinedRegionSearch:
    """Test CombinedRegionSearch two-phase algorithm."""

    def test_search_linear_graph(self, simple_linear_graph):
        """Test searching regions in a simple linear graph."""
        search = CombinedRegionSearch(simple_linear_graph)

        regions = search.search_regions()

        # Should create regions
        assert len(regions) > 0

        # Check that regions cover most nodes (ONNX GS may add Constant nodes that aren't partitioned)
        total_nodes = sum(len(r.get_region_nodes_and_descendants()) for r in regions)
        assert total_nodes > 0
        assert total_nodes <= len(simple_linear_graph.nodes)

        # Each region should have valid inputs/outputs
        for region in regions:
            assert region.inputs is not None
            assert region.outputs is not None

    def test_search_divergent_graph(self, divergent_graph):
        """Test searching regions in a divergent graph."""
        search = CombinedRegionSearch(divergent_graph)

        regions = search.search_regions()

        # Should create regions
        assert len(regions) > 0

        # Check that regions cover most nodes (ONNX GS may add Constant nodes that aren't partitioned)
        total_nodes = sum(len(r.get_region_nodes_and_descendants()) for r in regions)
        assert total_nodes > 0
        assert total_nodes <= len(divergent_graph.nodes)

    def test_region_hierarchy(self, simple_linear_graph):
        """Test that regions have proper hierarchical structure."""
        search = CombinedRegionSearch(simple_linear_graph)

        regions = search.search_regions()

        # Check that regions have children (hierarchical structure)
        for region in regions:
            if region.type == RegionType.COMPOSITE:
                assert len(region.get_children()) > 0

                # Verify parent-child relationships
                for child in region.get_children():
                    assert child.parent == region

    def test_parameters(self, simple_linear_graph):
        """Test CombinedRegionSearch with custom parameters."""
        # Test with different parameter values
        search = CombinedRegionSearch(
            simple_linear_graph,
            maximum_sequence_region_size=5,
            minimum_topdown_search_size=5,
        )

        regions = search.search_regions()

        assert len(regions) > 0


class TestPrintTree:
    """Test print_tree functionality."""

    def test_print_tree_output_content(self, simple_linear_graph):
        """Test that print_tree output contains region, node, and I/O information."""
        search = CombinedRegionSearch(simple_linear_graph)
        search.search_regions()

        output = io.StringIO()
        search.print_tree(file=output)
        result = output.getvalue()

        # Region information
        assert "Region" in result
        assert "Level" in result
        assert "Type:" in result

        # Node counts
        assert "Direct nodes:" in result
        assert "Total nodes:" in result
        assert "Children:" in result

        # I/O information
        assert "Inputs:" in result
        assert "Outputs:" in result

    def test_print_tree_divergent_graph(self, divergent_graph):
        """Test print_tree on a divergent graph with more complex structure."""
        search = CombinedRegionSearch(divergent_graph)
        search.search_regions()

        output = io.StringIO()
        search.print_tree(file=output)

        result = output.getvalue()

        # Should produce valid output
        assert "Region" in result
        assert len(result) > 0

    def test_print_tree_max_nodes_to_show(self, simple_linear_graph):
        """Test print_tree with custom max_nodes_to_show parameter."""
        search = CombinedRegionSearch(simple_linear_graph)
        search.search_regions()

        # Test with different max_nodes_to_show values
        output1 = io.StringIO()
        search.print_tree(max_items=1, file=output1)

        output2 = io.StringIO()
        search.print_tree(max_items=10, file=output2)

        # Both should produce output
        assert len(output1.getvalue()) > 0
        assert len(output2.getvalue()) > 0

    def test_print_tree_specific_region(self, simple_linear_graph):
        """Test print_tree with a specific region instead of root."""
        search = CombinedRegionSearch(simple_linear_graph)
        regions = search.search_regions()

        if len(regions) > 0:
            # Print a specific region
            output = io.StringIO()
            search.print_tree(region=regions[0], file=output)

            result = output.getvalue()
            assert "Region" in result
            assert f"Region {regions[0].id}" in result

    def test_print_tree_partitioner(self, simple_linear_graph):
        """Test print_tree on RegionPartitioner."""
        partitioner = RegionPartitioner(simple_linear_graph)
        partitioner.partition_graph()

        output = io.StringIO()
        partitioner.print_tree(file=output)

        result = output.getvalue()
        assert "Region" in result
        assert len(result) > 0

    def test_print_tree_top_down_builder(self, simple_linear_graph):
        """Test print_tree on TopDownRegionBuilder."""
        # Create a root region with all nodes
        root = Region(region_id=0, level=0, region_type=RegionType.LEAF)
        root.nodes.update(range(len(simple_linear_graph.nodes)))

        builder = TopDownRegionBuilder(simple_linear_graph, root)
        # Compute region I/O boundaries before building
        builder.compute_region_boundaries(root)
        builder.build_composite_region()

        output = io.StringIO()
        builder.print_tree(file=output)

        result = output.getvalue()
        print("\n" + "=" * 60)
        print("Region Tree Structure:")
        print("=" * 60)
        print(result)
        print("=" * 60)

        assert "Region" in result
        assert len(result) > 0
